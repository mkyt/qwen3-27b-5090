"""Text files are UTF-8 whatever the locale says: on Windows, Python's default encoding is the ANSI code page."""

import ast
import json
import os
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
PACKAGED = [*ROOT.glob("frontend/*.py"), *ROOT.glob("engine/python/**/*.py"), ROOT / "scripts/download_weights.py",
            *(ROOT / "reference/qwen27b" / f for f in ("__init__.py", "paths.py", "image_tokens.py"))]  # package.ps1


def test_fingerprint_in_a_locale_that_is_not_utf8(tmp_path):
    (tmp_path / "tokenizer.json").write_text(json.dumps(
        {"model": {"type": "BPE", "vocab": {"é": 0}, "merges": []}, "added_tokens": []}, ensure_ascii=False),
        encoding="utf-8")
    code = ("import locale, tokenizer; assert locale.getencoding().lower() not in ('utf-8', 'utf8'); "
            f"tokenizer.fingerprint({str(tmp_path)!r})")
    env = {**os.environ, "LC_ALL": "C", "PYTHONUTF8": "0", "PYTHONCOERCECLOCALE": "0"}
    r = subprocess.run([sys.executable, "-c", code], cwd=ROOT / "frontend", env=env, capture_output=True)
    assert r.returncode == 0, r.stderr.decode(errors="replace")


def test_packaged_code_names_the_encoding_of_text_files():
    missing = []
    for path in PACKAGED:
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if not isinstance(node, ast.Call):
                continue
            name = getattr(node.func, "attr", None) or getattr(node.func, "id", None)
            binary = any(isinstance(a, ast.Constant) and "b" in str(a.value) for a in node.args[1:])
            builtin_open = name == "open" and isinstance(node.func, ast.Name) and not binary
            text_io = name in ("read_text", "write_text") or builtin_open
            if text_io and not any(k.arg == "encoding" for k in node.keywords):
                missing.append(f"{path.relative_to(ROOT)}:{node.lineno}")
    assert not missing
