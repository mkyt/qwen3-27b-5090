"""HTTP bounds, origin checks and request admission without loading an engine."""

import email.message
import http.client
import io
import pathlib
import sys
import threading
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import server


def handler(headers=None, body=b"{}", path="/admin/concurrency"):
    h = object.__new__(server.Handler)
    h.path, h.close_connection = path, False
    h.headers = email.message.Message()
    for k, v in (headers if headers is not None else [("Content-Type", "application/json"), ("Content-Length", str(len(body)))]):
        h.headers[k] = v
    h.rfile, h.responses = io.BytesIO(body), []
    h._json = lambda status, obj: h.responses.append((status, obj))
    h.connection = SimpleNamespace(settimeout=lambda t: None)
    http = object.__new__(server.HTTPServer)
    http.body_lock, http.body_bytes = threading.Lock(), 0
    h.server = http
    h.applied = []
    h.srv = SimpleNamespace(engine=SimpleNamespace(configure=lambda c: h.applied.append(c) or c))
    return h


def parsed(headers, monkeypatch):
    monkeypatch.delenv("API_ALLOWED_HOSTS", raising=False)
    h = handler()
    h.raw_requestline = b"POST /admin/concurrency HTTP/1.1\r\n"
    h.rfile = io.BytesIO((headers + "\r\n").encode())
    h.request_version, h.command = "HTTP/1.1", None
    return h, h.parse_request()


@pytest.mark.parametrize("headers", ["Host: evil.example\r\n", "Host: localhost:8000\r\nOrigin: https://evil.example\r\n",
                                    "Host: localhost:8000\r\nOrigin: null\r\n",
                                    "Host: 127.0.0.1:8000\r\nSec-Fetch-Site: cross-site\r\n"])
def test_cross_origin_or_rebinding_host_is_rejected(monkeypatch, headers):
    h, ok = parsed(headers, monkeypatch)
    assert not ok
    assert h.responses[0][0] == 403
    assert h.close_connection


@pytest.mark.parametrize("headers", ["Host: 127.0.0.1:8000\r\n", "Host: [::1]:8000\r\n",
                                    "Host: localhost:8000\r\nOrigin: http://localhost:8000\r\n"])
def test_local_clients_and_same_origin_ui(monkeypatch, headers):
    _, ok = parsed(headers, monkeypatch)
    assert ok


def test_plain_text_cannot_change_settings():
    body = b'{"target_seqs":1}'
    h = handler([("Content-Type", "text/plain"), ("Content-Length", str(len(body)))], body)
    h.do_POST()
    assert h.responses[0][0] == 415
    assert not h.applied


@pytest.mark.parametrize("lengths", [[], ["-1"], ["nonsense"], ["1", "1"]])
def test_invalid_content_length(lengths):
    h = handler([("Content-Type", "application/json"), *[("Content-Length", n) for n in lengths]])
    h.do_POST()
    assert h.responses[0][0] == 400
    assert h.rfile.tell() == 0


def test_oversize_rejected_before_reading(monkeypatch):
    monkeypatch.setattr(server, "MAX_REQUEST_BYTES", 4)
    h = handler([("Content-Type", "application/json"), ("Content-Length", "5")])
    h.do_POST()
    assert h.responses[0][0] == 413
    assert h.rfile.tell() == 0


def test_shared_body_budget_and_release(monkeypatch):
    monkeypatch.setattr(server, "HTTP_BODY_BUDGET", 10)
    h = handler()
    assert h.server.reserve_body(9)
    h.do_POST()
    assert h.responses[-1][0] == 503
    h.server.release_body(9)
    h = handler()  # overload closes that connection; a new connection can proceed
    h.do_POST()
    assert h.responses[-1][0] == 200
    assert h.server.body_bytes == 0
    assert not h.close_connection  # complete requests still support connection reuse


def test_bad_json_releases_body_budget():
    h = handler(body=b"invalid")
    h.do_POST()
    assert h.responses[0][0] == 400
    assert h.server.body_bytes == 0


def test_body_deadline_is_absolute(monkeypatch):
    h = handler(body=b'{"target_seqs":1}')
    ticks = iter([0, 0, 31])
    monkeypatch.setattr(server.time, "monotonic", lambda: next(ticks))
    h.rfile = SimpleNamespace(read1=lambda n: b"{")
    h.do_POST()
    assert h.responses[0][0] == 408
    assert h.server.body_bytes == 0


def test_engine_admission_is_bounded(monkeypatch):
    monkeypatch.setattr(server, "MAX_PENDING_REQUESTS", 1)
    e = object.__new__(server.EngineClient)
    e.lock, e.ready = threading.Lock(), threading.Event()
    e.ready.set()
    e.on_demand, e.handles = None, {"running": object()}
    with pytest.raises(server.ApiError) as error:
        e.submit({})
    assert error.value.code == 503
    assert list(e.handles) == ["running"]


def test_buffered_header_reader_keeps_an_absolute_deadline(monkeypatch):
    ticks = iter([0.0, 0.1, 1.1])
    monkeypatch.setattr(server.time, "monotonic", lambda: next(ticks))

    def recv(buf):
        buf[0] = ord("x")
        return 1  # data keeps arriving, but never a complete header line

    sock = SimpleNamespace(settimeout=lambda t: None, recv_into=recv)
    reader = io.BufferedReader(server.DeadlineReader(sock, lambda: 1.0))
    with pytest.raises(TimeoutError, match="deadline"):
        reader.readline()
    reader.close()


def test_http_connection_reuses_after_complete_post(monkeypatch):
    monkeypatch.setattr(server, "HTTP_MAX_CONNECTIONS", 1)

    class Echo(server.Handler):
        def tokenize(self, body):
            self._json(200, body)

    listener = server.HTTPServer(("127.0.0.1", 0), Echo)
    thread = threading.Thread(target=listener.serve_forever, daemon=True)
    thread.start()
    client = http.client.HTTPConnection(*listener.server_address, timeout=3)
    try:
        for _ in range(2):
            client.request("POST", "/tokenize", b'{"prompt":"hello"}', {"Content-Type": "application/json"})
            response = client.getresponse()
            assert response.status == 200
            assert response.read() == b'{"prompt": "hello"}'
            assert not response.will_close
            other = http.client.HTTPConnection(*listener.server_address, timeout=3)
            try:
                other.request("GET", "/health")
                overloaded = other.getresponse()
                assert overloaded.status == 503
                assert overloaded.read() == b""
            finally:
                other.close()
        assert listener.body_bytes == 0
    finally:
        client.close()
        listener.shutdown()
        listener.server_close()
        thread.join(3)
