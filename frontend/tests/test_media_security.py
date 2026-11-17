"""Media destination checks and bounded reads, without accessing the network."""

import io
import pathlib
import socket
import sys
import time
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import media


def addresses(ip):
    family = socket.AF_INET6 if ":" in ip else socket.AF_INET
    return [(family, socket.SOCK_STREAM, 6, "", (ip, 80))]


@pytest.mark.parametrize("ip", ["127.0.0.1", "10.0.0.1", "172.16.1.2", "192.168.1.2", "169.254.169.254",
                              "0.0.0.0", "224.0.0.1", "::1", "fe80::1", "::ffff:127.0.0.1"])
def test_private_destinations_are_refused(monkeypatch, ip):
    monkeypatch.delenv("MEDIA_PRIVATE_HOSTS", raising=False)
    monkeypatch.setattr(socket, "getaddrinfo", lambda *a, **k: addresses(ip))
    with pytest.raises(ValueError, match="private or reserved"):
        media.destination("http://images.example/p.png")


def test_mixed_dns_answer_is_refused(monkeypatch):
    monkeypatch.delenv("MEDIA_PRIVATE_HOSTS", raising=False)
    monkeypatch.setattr(socket, "getaddrinfo", lambda *a, **k: addresses("1.1.1.1") + addresses("127.0.0.1"))
    with pytest.raises(ValueError):
        media.destination("http://images.example/p.png")


def test_private_host_is_an_explicit_exact_exception(monkeypatch):
    monkeypatch.setenv("MEDIA_PRIVATE_HOSTS", "images.local")
    monkeypatch.setattr(socket, "getaddrinfo", lambda *a, **k: addresses("127.0.0.1"))
    assert media.destination("http://images.local/p.png")[1] == "images.local"
    with pytest.raises(ValueError):
        media.destination("http://images.local.attacker.example/p.png")


@pytest.mark.parametrize("url", ["file:///etc/passwd", "http://user:pass@images.example/p", "http://images.example:bad/p",
                               "http://images.example/p\r\nInjected:1"])
def test_bad_urls(url):
    with pytest.raises(ValueError):
        media.destination(url)


def test_connection_uses_checked_ip_and_https_hostname(monkeypatch):
    calls = []
    monkeypatch.setattr(socket, "getaddrinfo", lambda *a, **k: calls.append(("dns", a[0])) or addresses("1.1.1.1"))
    sock = SimpleNamespace(settimeout=lambda t: None, connect=lambda a: calls.append(("connect", a)), close=lambda: None)
    monkeypatch.setattr(socket, "socket", lambda *a: sock)
    context = SimpleNamespace(wrap_socket=lambda s, server_hostname: calls.append(("tls", server_hostname)) or s)
    monkeypatch.setattr(media.ssl, "create_default_context", lambda: context)
    connection = SimpleNamespace(request=lambda *a, **k: calls.append(("request", a)), getresponse=lambda: object())
    monkeypatch.setattr(media.http.client, "HTTPConnection", lambda *a, **k: connection)
    media.open_response("https://images.example/p.png?q=1", time.monotonic() + 5)
    assert calls == [("dns", "images.example"), ("connect", ("1.1.1.1", 80)), ("tls", "images.example"),
                     ("request", ("GET", "/p.png?q=1"))]
    assert connection.sock is sock


class Response(io.BytesIO):
    def __init__(self, body, status=200, **headers):
        super().__init__(body)
        self.status, self.headers = status, headers

    def getheader(self, key):
        return self.headers.get(key)


def opened(response):
    return SimpleNamespace(close=lambda: None), response, SimpleNamespace(settimeout=lambda t: None, close=lambda: None)


def test_redirect_is_rechecked(monkeypatch):
    monkeypatch.delenv("MEDIA_PRIVATE_HOSTS", raising=False)
    monkeypatch.setattr(socket, "getaddrinfo", lambda host, *a, **k: addresses("1.1.1.1" if host == "public.example" else "127.0.0.1"))
    visited = []

    def redirect(url, deadline):
        media.destination(url)
        visited.append(url)
        return opened(Response(b"", 302, Location="http://private.example/secret"))

    monkeypatch.setattr(media, "open_response", redirect)
    with pytest.raises(ValueError, match="private or reserved"):
        media.fetch("http://public.example/p.png", 100, 5)
    assert visited == ["http://public.example/p.png"]


@pytest.mark.parametrize("headers", [{}, {"Content-Length": "6"}])
def test_download_limit_with_or_without_length(monkeypatch, headers):
    monkeypatch.setattr(media, "open_response", lambda *a: opened(Response(b"123456", **headers)))
    with pytest.raises(ValueError, match="byte limit"):
        media.fetch("http://public.example/", 5, 5)


def test_exact_limit(monkeypatch):
    monkeypatch.setattr(media, "open_response", lambda *a: opened(Response(b"12345")))
    assert media.fetch("http://public.example/", 5, 5) == b"12345"
