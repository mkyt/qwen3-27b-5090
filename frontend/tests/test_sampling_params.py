"""Sampling settings of a request (no engine needed)."""

import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from server import ApiError, sampling_of  # noqa: E402


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
