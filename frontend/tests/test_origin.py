"""Cross-origin POSTs (server.origin_allowed): no engine needed."""

import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from server import origin_allowed  # noqa: E402


@pytest.mark.parametrize("headers", [
    {"Host": "127.0.0.1:8000"},                                            # an API client: no Origin
    {"Host": "127.0.0.1:8000", "Origin": "http://127.0.0.1:8000"},         # the built-in UI
    {"Host": "localhost:8000", "Origin": "http://LOCALHOST:8000/"},
    {"Host": "pc.lan", "Origin": "http://pc.lan:80"},                       # default ports
    {"Host": "[::1]:8000", "Origin": "http://[::1]:8000"},
    {"Host": "127.0.0.1:8000", "X-Forwarded-Host": "llm.example.internal", "Origin": "https://llm.example.internal"},
])
def test_allowed(headers):
    assert origin_allowed(headers)


@pytest.mark.parametrize("headers", [
    {"Host": "127.0.0.1:8000", "Origin": "https://evil.example"},
    {"Host": "localhost:8000", "Origin": "http://localhost:3000"},          # another port of the same machine
    {"Host": "127.0.0.1:8000", "Origin": "null"},
    {"Host": "[::1]:8000", "Origin": "http://[::1]"},
    {"Origin": "http://127.0.0.1:8000"},                                    # no Host
])
def test_rejected(headers):
    assert not origin_allowed(headers)


def test_trusted_origins(monkeypatch):
    h = {"Host": "127.0.0.1:8000", "Origin": "https://chat.example.internal"}
    monkeypatch.setenv("TRUSTED_ORIGINS", "https://chat.example.internal/, http://other")
    assert origin_allowed(h)
    monkeypatch.setenv("TRUSTED_ORIGINS", "*")
    assert origin_allowed({"Host": "x", "Origin": "null"})
    monkeypatch.delenv("TRUSTED_ORIGINS")
    assert not origin_allowed(h)
