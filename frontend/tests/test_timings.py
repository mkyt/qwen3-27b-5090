"""llama.cpp's `timings` object of the API server (no engine needed)."""

import pathlib
import queue
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import worker  # noqa: E402
from server import Generation  # noqa: E402


def generation(prompt, cached, tokens, info):
    gen = Generation.__new__(Generation)
    gen.prompt_tokens, gen.cached, gen.tokens, gen.timings = prompt, cached, tokens, info
    return gen


def done(computed=900, prefill_s=0.1, ttft=0.2, total=1.2):
    return {"computed_tokens": computed, "prefill_s": prefill_s, "ttft": ttft, "total": total}


def test_fields_match_llama_cpp():
    t = generation(1000, 100, 50, done()).timing_fields["timings"]
    assert list(t) == ["cache_n", "prompt_n", "prompt_ms", "prompt_per_token_ms", "prompt_per_second",
                       "predicted_n", "predicted_ms", "predicted_per_token_ms", "predicted_per_second"]
    assert (t["cache_n"], t["prompt_n"], t["predicted_n"]) == (100, 900, 50)
    assert abs(t["prompt_ms"] - 100) < 1e-6 and abs(t["prompt_per_second"] - 9000) < 1e-6
    assert abs(t["prompt_per_token_ms"] - 100 / 900) < 1e-9
    assert abs(t["predicted_ms"] - 1000) < 1e-6 and abs(t["predicted_per_second"] - 50) < 1e-6
    assert abs(t["predicted_per_token_ms"] - 20) < 1e-6


def test_preempted_request_keeps_the_prompt_count_and_the_real_rate():
    t = generation(1000, 100, 50, done(computed=1800, prefill_s=0.2)).timing_fields["timings"]
    assert t["prompt_n"] + t["cache_n"] == 1000
    assert abs(t["prompt_per_second"] - 9000) < 1e-6


def test_edges_do_not_divide_by_zero_or_invent_rates():
    assert generation(10, 0, 0, None).timing_fields == {}
    t = generation(10, 9, 0, done(computed=0, prefill_s=0.0, ttft=None)).timing_fields["timings"]
    assert t["prompt_per_second"] == t["prompt_per_token_ms"] == t["predicted_ms"] == t["predicted_per_second"] == 0
    t = generation(10, 0, 1, done(computed=10, ttft=0.2, total=0.2000001)).timing_fields["timings"]
    assert t["predicted_per_second"] == t["predicted_per_token_ms"] == 0


def test_draft_counts_only_with_drafts():
    t = generation(1000, 100, 50, dict(done(), drafted=70, draft_accepted=39)).timing_fields["timings"]
    assert (t["draft_n"], t["draft_n_accepted"]) == (70, 39)
    assert "draft_n" not in generation(1000, 100, 50, dict(done(), drafted=0, draft_accepted=0)).timing_fields["timings"]


def test_verify_step_counts_drafts_as_llama_cpp():
    sch = object.__new__(worker.Scheduler)
    s = worker.Seq(1, {"ids": [1], "max_new": 8, "temperature": 0.0, "top_k": 20, "top_p": 1.0}, 0.0)
    sch.draft_block = 7
    sch.count_draft(s, [5])          # all rejected: only the bonus token
    sch.count_draft(s, [1, 2, 3, 4])
    assert (s.drafted, s.draft_accepted) == (14, 3)
    sch.draft_block = 0              # several paths: no count of drafts, so the server leaves them out
    sch.count_draft(s, [1, 2])
    assert s.drafted == 14


def stopped(*msgs):
    gen = generation(10, 0, 1, None)
    gen.q = queue.Queue()
    for m in msgs:
        gen.q.put(m)
    return gen


def test_await_done_reads_the_report_after_a_stop_string():
    gen = stopped(("tok", "r", [1, 2]), ("done", "r", "cancelled", done()))
    gen._await_done()
    assert gen.timings == done() and gen.tokens == 1  # (what the cancel overtook is not part of the answer)


def test_await_done_gives_up_without_a_report():
    gen = stopped(("error", "r", "failed", 500))
    gen._await_done()
    assert gen.timing_fields == {}
    gen = stopped()
    gen._await_done(timeout=0.01)
    assert gen.timing_fields == {}
