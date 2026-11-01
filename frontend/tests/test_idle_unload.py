"""Unloading the engine when idle (ENGINE_IDLE_UNLOAD_S, server.EngineClient) against a fake engine process: no GPU."""

import email.message
import multiprocessing as mp
import pathlib
import sys
import threading
import time
import types

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import server  # noqa: E402

IDLE = 0.3
GiB = 1 << 30


class Hub:
    engine_state = "loading"

    def engine_state_set(self, state, _text):
        self.engine_state = state

    def engine_starting(self):
        self.engine_state = "loading"

    def engine_ready(self, _info, _cfg):
        self.engine_state = "ready"

    def __getattr__(self, _name):
        return lambda *a, **k: None


class FakeProcess:
    """The engine process: loads for `load` seconds, finishes requests when told to (or at once)."""

    exitcode = 0

    def __init__(self, conn, load, hold, refuse=None):
        self.conn, self.load, self.hold, self.refuse = conn, load, hold, refuse
        self.closing = threading.Lock()  # (run and terminate may close the pipe at once: EBADF)
        self.submitted, self.done = [], threading.Event()
        self.thread = threading.Thread(target=self.run, daemon=True)
        self.thread.start()

    def run(self):
        time.sleep(self.load)
        if self.refuse:
            self.conn.send(("refused", self.refuse))
            self.terminate()
            return
        self.conn.send(("ready", {}))
        try:
            while True:
                msg = self.conn.recv()
                if msg[0] == "stop":
                    break
                if msg[0] == "submit":
                    self.submitted.append(msg[1])
                    if self.hold:
                        self.done.wait(10)
                    self.conn.send(("done", msg[1], "stop", {}))
        except (EOFError, OSError):
            pass
        self.terminate()

    def is_alive(self):
        return self.thread.is_alive()

    def join(self, timeout=None):
        self.thread.join(timeout)

    def terminate(self):
        with self.closing:
            self.conn.close()

    kill = terminate


class Client(server.EngineClient):
    load, hold, refuse = 0.0, False, None

    def start(self):
        self.starts = getattr(self, "starts", 0) + 1
        server.HUB.engine_starting()
        parent, child = mp.Pipe()
        self.proc = FakeProcess(child, self.load, self.hold, self.refuse)
        self.conn = parent
        threading.Thread(target=self.reader, args=(self.gen,), daemon=True).start()


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(server, "HUB", Hub())
    made = []

    def make(idle=IDLE, autostart=True, load=0.0, hold=False):
        Client.load, Client.hold, Client.refuse = load, hold, None
        c = Client({}, autostart=autostart, idle_unload=idle)
        made.append(c)
        return c

    yield make
    for c in made:
        c.stopping = True
        c.stop()


def until(cond, timeout=5.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if cond():
            return True
        time.sleep(0.01)
    return False


def finish(q):
    return q.get(timeout=5)[0]


def test_off_by_default_the_engine_stays_ready(client):
    c = client(idle=0)
    assert c.ready.wait(5)
    time.sleep(1.2)
    assert c.ready.is_set() and c.on_demand is None and c.starts == 1


def test_unloads_when_idle_and_a_request_loads_it_again(client):
    c = client()
    assert c.ready.wait(5)
    assert until(lambda: c.on_demand == "unloaded" and c.state() == "stopped" and not c.proc.is_alive())
    Client.load = 0.2
    _, q = c.submit({})
    assert finish(q) == "done" and c.starts == 2 and c.on_demand is None and c.state() == "ready"


def test_requests_while_loading_wait_for_the_same_engine(client):
    c = client()
    assert c.ready.wait(5)
    assert until(lambda: c.state() == "stopped")
    Client.load = 0.3
    results = []
    threads = [threading.Thread(target=lambda: results.append(finish(c.submit({})[1]))) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(5)
    assert results == ["done"] * 4 and c.starts == 2


def test_a_running_request_keeps_the_engine_loaded(client):
    c = client(hold=True)
    assert c.ready.wait(5)
    _, q = c.submit({})
    time.sleep(IDLE * 4)
    assert c.ready.is_set() and c.on_demand is None
    c.proc.done.set()
    assert finish(q) == "done"
    assert until(lambda: c.on_demand == "unloaded")  # (idle counts from the end of the request)


def test_a_stop_from_the_admin_api_is_not_undone_by_a_request(client):
    c = client()
    assert c.ready.wait(5)
    c.control("stop")
    with pytest.raises(server.ApiError) as e:
        c.submit({})
    assert e.value.code == 503 and c.starts == 1


def test_a_stop_after_an_unload_is_not_undone_by_a_request(client):
    c = client()
    assert c.ready.wait(5)
    assert until(lambda: c.state() == "stopped")
    c.control("stop")
    with pytest.raises(server.ApiError):
        c.submit({})
    assert c.starts == 1


def test_a_start_from_the_admin_api_after_an_unload(client):
    c = client()
    assert c.ready.wait(5)
    assert until(lambda: c.state() == "stopped")
    c.idle_unload = 60
    c.control("start")
    assert c.on_demand is None and c.ready.wait(5) and c.starts == 2


def test_autostart_0_waits_for_the_first_request(client):
    c = client(autostart=False)
    assert c.on_demand == "unloaded" and c.proc is None and not c.ready.is_set()
    _, q = c.submit({})
    assert q.get(timeout=5)[0] == "done"
    assert c.ready.is_set() and c.on_demand is None and c.starts == 1


def test_autostart_0_without_idle_unload_stays_stopped_until_a_start(client):
    c = client(idle=0, autostart=False)
    with pytest.raises(server.ApiError):
        c.submit({})
    c.control("start")
    assert c.ready.wait(2)


def test_the_idle_state_counts_down_and_says_why_not(client):
    c = client(idle=5)
    assert c.ready.wait(5)
    s = c.idle_state()
    assert s["idle_unload_s"] == 5 and not s["unloaded"] and 0 < s["unload_in"] <= 5 and s["refused"] is None
    assert client(autostart=False).idle_state()["unloaded"]
    c = client(idle=5, hold=True)
    assert c.ready.wait(5)
    _, q = c.submit({})
    assert c.idle_state()["unload_in"] is None  # (in use)
    c.proc.done.set()
    assert finish(q) == "done"


def test_the_idle_time_changes_at_once(client):
    c = client(idle=0)
    assert c.ready.wait(5)
    assert c.idle_state()["unload_in"] is None
    time.sleep(1.2)
    assert c.state() == "ready"  # (off)
    c.set_idle_unload(0.3)
    assert until(lambda: c.on_demand == "unloaded" and c.state() == "stopped")
    _, q = c.submit({})  # (the first request loads it)
    assert finish(q) == "done"
    c.set_idle_unload(0)
    time.sleep(1.2)
    assert c.state() == "ready" and c.idle_state()["unload_in"] is None


def test_save_keeps_settings_for_the_next_load_without_loading(client, monkeypatch):
    class Launch:
        settings = {"max_seqs": 8}
        checked = []

        def validate(self, changes, device=None, check_budget=True):
            self.checked.append(check_budget)
            return dict(self.settings, **changes)

        def plan_many(self, _settings, _device):
            return [{"error": "not enough VRAM"}]  # (other programs hold it now)

        def config(self, s):
            return {"max_seqs": s["max_seqs"], "settings": s}

    monkeypatch.setattr(server.HUB, "plan_device", lambda: ((GiB, 32 * GiB), "nvml"), raising=False)
    c = client(autostart=False)
    c.cfg = {"max_seqs": 8}
    answers = []
    h = types.SimpleNamespace(headers=email.message.Message(), srv=types.SimpleNamespace(engine=c, launch=Launch(), use_template_of=lambda _w: None),
                              _json=lambda code, obj: answers.append((code, obj)))
    h.headers["Content-Type"] = "application/json"
    control = lambda body: server.Handler.engine_control(h, body)
    s = {"weights": "w", "max_seqs": 4}
    h.srv.launch.settings = {"weights": "w", "max_seqs": 8}
    control({"action": "save", "settings": s})
    assert answers[-1][0] == 200 and answers[-1][1]["settings"]["max_seqs"] == 4
    assert c.on_demand == "unloaded" and c.proc is None  # (still waiting, no fit asked)
    assert h.srv.launch.checked == [False] and c.cfg["max_seqs"] == 4
    with pytest.raises(server.ApiError):
        control({"action": "start", "settings": s})  # (a start does ask)
    with pytest.raises(server.ApiError):
        control({"action": "save"})
    _, q = c.submit({})  # (the first request loads with them)
    assert finish(q) == "done" and c.cfg["max_seqs"] == 4
    with pytest.raises(server.ApiError) as e:
        control({"action": "save", "settings": s})
    assert e.value.code == 409


def test_no_request_is_lost_while_requests_race_the_unload(client):
    c = client(idle=0.05)
    assert c.ready.wait(5)
    for _ in range(40):
        _, q = c.submit({})
        assert finish(q) == "done"
        time.sleep(0.06)


def test_a_load_that_does_not_finish_gets_503(client, monkeypatch):
    c = client()
    assert c.ready.wait(5)
    assert until(lambda: c.state() == "stopped")
    Client.load = 2.0
    monkeypatch.setattr(Client, "LOAD_WAIT_S", 0.3)
    with pytest.raises(server.ApiError) as e:
        c.submit({})
    assert e.value.code == 503


def unloaded(client, refuse="not enough VRAM: the KV pool would hold 1000 tokens"):
    c = client()
    assert c.ready.wait(5)
    assert until(lambda: c.state() == "stopped")
    Client.load, Client.refuse = 0.1, refuse
    return c


def test_too_little_vram_gets_503_with_the_reason_and_no_retries(client):
    c = unloaded(client)
    with pytest.raises(server.ApiError) as e:
        c.submit({})
    assert e.value.code == 503 and "not enough VRAM" in str(e.value)
    time.sleep(2.5)  # (the first retry after a crash would have started by now)
    assert c.starts == 2 and c.state() == "stopped" and c.on_demand == "unloaded" and c.refused


def test_requests_waiting_for_a_refused_load_do_not_each_try_again(client):
    c = unloaded(client)
    errors = []

    def one():
        try:
            c.submit({})
        except server.ApiError as e:
            errors.append(e.code)

    threads = [threading.Thread(target=one) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(5)
    assert errors == [503] * 4 and c.starts == 2


def test_the_next_request_tries_again_when_the_vram_is_back(client):
    c = unloaded(client)
    with pytest.raises(server.ApiError):
        c.submit({})
    Client.refuse = None
    _, q = c.submit({})
    assert finish(q) == "done" and c.starts == 3 and c.refused is None and c.state() == "ready"


def test_without_idle_unload_too_little_vram_retries_as_after_a_crash(client):
    c = client(idle=0, autostart=False)
    Client.refuse = "not enough VRAM"
    c.control("start")
    assert until(lambda: c.starts == 2, 5)  # (2 s later)
    assert c.refused is None and c.on_demand is None
    c.control("stop")
