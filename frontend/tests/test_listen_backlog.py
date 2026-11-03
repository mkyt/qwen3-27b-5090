"""The listen backlog holds a burst of connections while the accept loop is busy (no engine needed)."""

import pathlib
import socket
import sys
from http.server import BaseHTTPRequestHandler

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from server import HTTPServer  # noqa: E402


def test_a_burst_of_20_connects_without_accept():
    # Python's default backlog of 5 refuses the rest of the burst at once on Windows (and leaves it hanging on Linux).
    server = HTTPServer(("127.0.0.1", 0), BaseHTTPRequestHandler)
    clients = []
    try:
        for _ in range(20):
            c = socket.socket()
            c.settimeout(2)
            clients.append(c)
            c.connect(server.server_address)
    finally:
        for c in clients:
            c.close()
        server.server_close()
