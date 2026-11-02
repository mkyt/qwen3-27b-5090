"""llama.cpp's `timings` object of the API server (no engine needed)."""

import pathlib
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
    gen = generation(10, 0, 0, None)
    gen.prefilled, gen.toks, gen.first_tok = None, [], None
    for m in msgs:  # (as Generation.__iter__ takes them)
        if m[0] == "prefilled":
            gen.prefilled, gen.toks = m[1], gen.toks[-1:]
        else:
            gen.tokens += m[2]
            gen.toks.append((m[1], m[2]))
            gen.first_tok = gen.first_tok or m[1]
    gen._stopped_timings()
    return gen


def prefilled(drafted=0, accepted=0, block=7):
    return ("prefilled", {"ttft": 0.2, "prefill_s": 0.1, "computed_tokens": 900, "drafted": drafted,
                          "draft_accepted": accepted, "draft_block": block})


def test_stop_string_timings_without_waiting_for_the_engine():
    # the prefill's token at t=10.0, then verify steps of 3 and 1 tokens: 2 steps, 14 drafts, 2 accepted
    gen = stopped(("tok", 10.0, 1), prefilled(), ("tok", 10.5, 3), ("tok", 11.0, 1))
    assert dict(gen.timings, total=round(gen.timings["total"], 9)) == {
        "computed_tokens": 900, "prefill_s": 0.1, "ttft": 0.2, "total": 1.2, "drafted": 14, "draft_accepted": 2}
    t = gen.timing_fields["timings"]
    assert t["predicted_n"] == 5 and abs(t["predicted_ms"] - 1000) < 1e-6


def test_stop_string_timings_after_a_preemption():
    # preempted after 2 steps: the new prefill's info has the drafts so far, the decode time runs from the first token
    gen = stopped(("tok", 10.0, 1), prefilled(), ("tok", 10.5, 3), ("tok", 11.0, 1),
                  ("tok", 13.0, 1), prefilled(drafted=14, accepted=2), ("tok", 13.5, 2))
    assert (gen.timings["drafted"], gen.timings["draft_accepted"]) == (21, 3)
    assert abs(gen.timings["total"] - 3.7) < 1e-9


def test_stop_string_timings_without_drafts_or_reports():
    gen = stopped(("tok", 10.0, 1), prefilled(block=0), ("tok", 10.5, 1))
    assert (gen.timings["drafted"], gen.timings["draft_accepted"]) == (0, 0)
    assert "draft_n" not in gen.timing_fields["timings"]
    assert stopped(("tok", 10.0, 1)).timing_fields == {}  # (no "prefilled" yet)
