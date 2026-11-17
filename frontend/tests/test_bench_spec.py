"""The greedy diagnostic must exercise the requested runtime policy, without a GPU or model downloads."""

import importlib.util
import os
import pathlib
import sys
from types import SimpleNamespace

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("bench_spec_under_test", ROOT / "scripts/bench_spec.py")
bench = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bench)


def run_check(monkeypatch, args=(), plain=(1, 2), drafted=(1, 2)):
    seen = []

    class Engine:
        def __init__(self, *a, **kw):
            seen.append({k: v for k, v in os.environ.items() if k.startswith("Q27_")})

        def reset(self):
            pass

        def close(self):
            pass

        def spec_stats(self):
            return {"lookup_steps": 0, "steps": 1}

    tok = SimpleNamespace(decode=lambda ids: str(ids))
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(AutoTokenizer=SimpleNamespace(from_pretrained=lambda p: tok)))
    monkeypatch.setitem(sys.modules, "q27lib", SimpleNamespace(Engine=Engine))
    monkeypatch.setitem(sys.modules, "qwen27b.weights", SimpleNamespace(
        DFLASH2=("draft", "rev"), DSPINE=("dspine", "rev"), TARGET=("target", "rev"),
        engine_target=lambda p: p, snapshot=lambda *a: "unused"))
    monkeypatch.setattr(bench, "chat_ids", lambda *a: [1])
    monkeypatch.setattr(bench, "run_plain", lambda *a: list(plain))
    monkeypatch.setattr(bench, "run_spec", lambda *a: (list(drafted), 1, 0.0, 0.0))
    monkeypatch.setattr(sys, "argv", ["bench_spec.py", "--check-greedy", "2", "--prompts", "code", *args])
    with pytest.raises(SystemExit) as result:
        bench.main()
    return result.value.code, seen


@pytest.mark.parametrize("settings", [{}, {"Q27_SKINNY_EXACT": "0", "Q27_SPEC_PATHS": "4", "Q27_ATTN_TC": "1"},
                                      {"Q27_SKINNY_EXACT": "1", "Q27_SPEC_PATHS": "2", "Q27_ATTN_TC": "0"}])
def test_default_check_preserves_runtime_settings(monkeypatch, settings, capsys):
    for key in ("Q27_SKINNY_EXACT", "Q27_SPEC_PATHS", "Q27_ATTN_TC"):
        monkeypatch.delenv(key, raising=False)
    for key, value in settings.items():
        monkeypatch.setenv(key, value)
    before = {k: v for k, v in os.environ.items() if k.startswith("Q27_")}
    status, seen = run_check(monkeypatch)
    assert status == 0
    assert seen == [before]
    output = capsys.readouterr().out
    assert "runtime settings" in output
    assert "Q27_SKINNY_EXACT" in output


def test_exact_check_is_explicit_and_aligns_tree_attention(monkeypatch, capsys):
    monkeypatch.setenv("Q27_SKINNY_EXACT", "0")
    monkeypatch.setenv("Q27_SPEC_PATHS", "4")
    monkeypatch.setenv("Q27_ATTN_TC", "1")
    status, seen = run_check(monkeypatch, ["--check-exact"])
    assert status == 0
    assert seen[0]["Q27_SKINNY_EXACT"] == "1"
    assert seen[0]["Q27_ATTN_TC"] == "0"
    assert "exact policy" in capsys.readouterr().out


@pytest.mark.parametrize("drafted", [(1, 3), (1,)])
def test_mismatch_fails_and_never_reports_identical(monkeypatch, drafted, capsys):
    status, _ = run_check(monkeypatch, drafted=drafted)
    assert status == 1
    output = capsys.readouterr().out
    assert "identical" not in output
    assert "differs at 1" in output


def test_exact_without_a_greedy_check_is_rejected(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["bench_spec.py", "--check-exact"])
    with pytest.raises(SystemExit) as result:
        bench.main()
    assert result.value.code == 2
    assert "requires --check-greedy" in capsys.readouterr().err


@pytest.mark.parametrize("thinking", [False, True])
def test_chat_template_uses_the_requested_reasoning_mode(thinking):
    def template(messages, **kwargs):
        assert kwargs["enable_thinking"] is thinking
        assert kwargs["add_generation_prompt"] is True
        return {"input_ids": [1, 2]}

    assert bench.chat_ids(SimpleNamespace(apply_chat_template=template), "explain", thinking) == [1, 2]
