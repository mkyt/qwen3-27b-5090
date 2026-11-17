"""Sampling settings of a request (no engine needed)."""

import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from server import ApiError, checked_token_ids, sampling_of, token_controls_of  # noqa: E402


def test_defaults():
    s = sampling_of({})
    assert (s["temperature"], s["top_k"], s["top_p"]) == (1.0, 20, 0.95)
    assert (s["min_p"], s["presence_penalty"], s["frequency_penalty"]) == (0.0, 0.0, 0.0)


def test_min_p_and_penalties():
    s = sampling_of({"min_p": 0.05, "presence_penalty": 1.5, "frequency_penalty": -0.5, "repetition_penalty": 1.1})
    assert (s["min_p"], s["presence_penalty"], s["frequency_penalty"]) == (0.05, 1.5, -0.5)
    assert sampling_of({"presence_penalty": None})["presence_penalty"] == 0.0


@pytest.mark.parametrize("body", [{"min_p": 1.5}, {"min_p": -0.1}, {"presence_penalty": 2.5},
                                  {"frequency_penalty": -3}, {"presence_penalty": "high"}])
def test_out_of_range(body):
    with pytest.raises(ApiError) as e:
        sampling_of(body)
    assert e.value.code == 400


def test_token_controls():
    assert token_controls_of({}, 1000, 64) == {}
    assert token_controls_of({"logit_bias": {"5": -100, "7": 2.5, "9": 0}}, 1000, 64) == {"logit_bias": {5: -100.0, 7: 2.5}}
    assert token_controls_of({"logit_bias": {}, "min_tokens": 0, "ignore_eos": False}, 1000, 64) == {}
    assert token_controls_of({"min_tokens": 64, "ignore_eos": True}, 1000, 64) == {"min_tokens": 64, "ignore_eos": True}
    assert len(token_controls_of({"logit_bias": {str(i): 1 for i in range(300)}}, 1000, 64)["logit_bias"]) == 300


@pytest.mark.parametrize("body", [{"logit_bias": [1, 2]}, {"logit_bias": {"x": 1}}, {"logit_bias": {"1000": 1}},
                                  {"logit_bias": {"-1": 1}}, {"logit_bias": {"5": 101}}, {"logit_bias": {"5": "1"}},
                                  {"logit_bias": {str(i): 1 for i in range(301)}}, {"min_tokens": -1},
                                  {"min_tokens": 65}, {"min_tokens": 1.5}, {"ignore_eos": "yes"}])
def test_bad_token_controls(body):
    with pytest.raises(ApiError) as e:
        token_controls_of(body, 1000, 64)
    assert e.value.code == 400


@pytest.mark.parametrize("body", [{"seed": {}}, {"seed": []}, {"seed": "1"}, {"seed": True}, {"seed": 1.5},
                                  {"temperature": float("nan")}, {"temperature": float("inf")}, {"temperature": -1},
                                  {"top_p": float("nan")}, {"top_p": 2}, {"top_k": 2**32}, {"top_k": 1.5}])
def test_invalid_sampling_is_a_client_error(body):
    with pytest.raises(ApiError) as e:
        sampling_of(body)
    assert e.value.code == 400


def test_integer_seeds_and_unlimited_top_k():
    for seed in (None, 0, -123, 2**63):
        assert sampling_of({"seed": seed})["seed"] == seed
    assert sampling_of({"top_k": -1})["top_k"] == 0


@pytest.mark.parametrize("ids", [[-1], [248320], [2**31], [1.5], [True], [{}], [None], "hello"])
def test_bad_prompt_tokens_are_rejected(ids):
    with pytest.raises(ApiError) as e:
        checked_token_ids(ids, 248320)
    assert e.value.code == 400


def test_prompt_tokens_in_vocabulary():
    assert checked_token_ids([0, 248319], 248320) == [0, 248319]
