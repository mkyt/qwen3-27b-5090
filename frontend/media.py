"""Bounded media downloads. Resolve and check every redirect, then connect to that exact address.

No environment proxy is used: it would resolve the destination outside these checks. An operator can allow a
specific private media server with MEDIA_PRIVATE_HOSTS (comma-separated hostnames or IPs, no wildcards).
"""

import http.client
import ipaddress
import os
import socket
import ssl
import time
import urllib.parse


def destination(url):
    u = urllib.parse.urlsplit(url)
    if u.scheme not in ("http", "https") or not u.hostname or u.username is not None or u.password is not None:
        raise ValueError("media URL must be http(s), with a host and without credentials")
    host = u.hostname.encode("idna").decode("ascii").lower()
    if any(c.isspace() or ord(c) < 32 for c in url):
        raise ValueError("invalid media URL")
    port = u.port or (443 if u.scheme == "https" else 80)
    private = {h.strip().lower().rstrip(".") for h in os.environ.get("MEDIA_PRIVATE_HOSTS", "").split(",") if h.strip()}
    addresses = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
    if not addresses:
        raise ValueError("media host has no addresses")
    if host.rstrip(".") not in private:
        for _, _, _, _, address in addresses:
            ip = ipaddress.ip_address(address[0])
            if not ip.is_global or ip.is_multicast or ip.is_unspecified:
                raise ValueError("private or reserved media addresses are not allowed (MEDIA_PRIVATE_HOSTS)")
    path = urllib.parse.urlunsplit(("", "", u.path or "/", u.query, ""))
    return u.scheme, host, port, path, addresses


def open_response(url, deadline):
    scheme, host, port, path, addresses = destination(url)
    error = None
    for family, kind, proto, _, address in addresses:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("media download timed out")
        sock = socket.socket(family, kind, proto)
        conn = http.client.HTTPConnection(host, port, timeout=remaining)
        try:
            sock.settimeout(remaining)
            sock.connect(address)  # the checked numeric address; no second DNS lookup
            if scheme == "https":
                sock = ssl.create_default_context().wrap_socket(sock, server_hostname=host)
            conn.sock = sock  # HTTPConnection supplies Host; TLS above verifies the original hostname
            conn.request("GET", path, headers={"User-Agent": "qwen3-27b-5090-media", "Accept-Encoding": "identity"})
            return conn, conn.getresponse(), sock
        except (OSError, http.client.HTTPException) as e:
            error = e
            conn.close()
            sock.close()
    raise error


def fetch(url, limit, timeout):
    deadline = time.monotonic() + timeout
    for hop in range(6):
        conn, response, sock = open_response(url, deadline)
        try:
            if response.status in (301, 302, 303, 307, 308):
                location = response.getheader("Location")
                if not location or hop == 5:
                    raise ValueError("invalid or too many media redirects")
                url = urllib.parse.urljoin(url, location)
                continue
            if response.status != 200:
                raise ValueError(f"media server returned HTTP {response.status}")
            length = response.getheader("Content-Length")
            if length is not None and (not length.isdecimal() or int(length) > limit):
                raise ValueError("media is larger than its byte limit or has an invalid length")
            data = bytearray()
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("media download timed out")
                sock.settimeout(remaining)
                chunk = response.read1(min(65536, limit + 1 - len(data)))
                if not chunk:
                    return bytes(data)
                data.extend(chunk)
                if len(data) > limit:
                    raise ValueError("media is larger than its byte limit")
        finally:
            response.close()
            conn.close()
            sock.close()
