"""API server of the custom engine (Phase 6): OpenAI- and Anthropic-compatible, a drop-in for the vLLM /
SGLang environments (same port, model name, parsers and request options).

Endpoints: POST /v1/chat/completions, /v1/completions, /v1/responses (Codex), /v1/messages,
/v1/messages/count_tokens, /tokenize, /detokenize; GET /v1/models, /health, /metrics; GET / POST
/admin/concurrency (the admission settings, P7-14); GET / POST /admin/engine (stop, start or restart the engine,
with other start settings: launch.py), POST /admin/engine/plan (what start settings leave for the context); GET /ui (live stats in the browser: GPU, VRAM, requests, log;
/ui/stream is its event stream, /ui/state one snapshot as JSON; stats.py). Streaming everywhere (SSE), reasoning (`<think>`, returned as
`reasoning_content` / `reasoning`, or Anthropic thinking blocks), tool calls (qwen3_coder XML, streamed call by
call), images (data URLs or http(s) URLs; VISION=host | fp8 | bf16).

The engine runs in its own process (worker.py, which schedules up to MAX_SEQS requests at once; how many run
is decided while it runs: a target number, room in the KV pool, MIN_SEQ_TPS). If it
dies (a CUDA error), requests in flight fail with 503, /health answers 503 and a new engine is started.
Requests are cancelled when the client disconnects or a stop string matches.

    python frontend/server.py [--port 8000] [--max-tokens 262144] [--kv-dtype nvfp4|fp8|bf16] [--spec dflash2|none]
"""

import argparse
import base64
import json
import multiprocessing as mp
import os
import pathlib
import queue
import select
import signal
import socket
import sys
import threading
import time
import traceback
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "frontend"))
sys.path.insert(0, str(ROOT / "engine/python"))
sys.path.insert(0, str(ROOT / "reference"))

import worker  # noqa: E402
from launch import Launch, model_dir  # noqa: E402
from stats import HUB, request_meta  # noqa: E402
from chat import (STOP_REASON, AnthropicError, OutputParser, _partial_suffix, anthropic_messages,  # noqa: E402,F401
                  anthropic_template_kwargs, new_id, normalize_messages, openai_call, parse_output,
                  parse_tool_calls, responses_messages, responses_template_kwargs, split_reasoning,
                  template_kwargs, thinking_signature)

MODEL_NAME = os.environ.get("SERVED_MODEL_NAME", "qwen3.8-27b")
DEFAULTS = {"temperature": 1.0, "top_k": 20, "top_p": 0.95}  # generation_config.json of the checkpoint
SPEC_MARGIN = 8  # a verify block writes up to 8 positions past the committed length
KEEPALIVE_S = 10.0
DUMP_DIR = os.environ.get("API_DUMP_DIR")  # debug: write every POST body there
UI_PAGE = ROOT / "frontend/ui/index.html"
GENERATE = ("/v1/chat/completions", "/v1/completions", "/v1/responses", "/v1/messages")


def log(msg):
    print(f"[api] {msg}", file=sys.stderr, flush=True)


class ApiError(Exception):
    def __init__(self, code, msg, kind="invalid_request_error"):
        super().__init__(msg)
        self.code, self.kind = code, kind


# ---------------------------------------------------------------- images


class Images:
    """Image inputs (P4-12): data URLs (or http(s) URLs) -> pixel_values of the checkpoint's image processor."""

    MAX_BYTES = 32 << 20

    def __init__(self, path, max_pixels):
        from imageproc import ImageProcessor

        self.proc = ImageProcessor(path)
        self.max_pixels = max_pixels

    @staticmethod
    def collect(messages):
        """The image URLs of the messages, in the order the chat template renders their pads."""
        urls = []
        for m in messages:
            content = m.get("content")
            if not isinstance(content, list):
                continue
            for part in content:
                if not isinstance(part, dict):
                    continue
                if part.get("type") in ("video", "video_url") or "video" in part or "video_url" in part:
                    raise ApiError(400, "video input is not supported")
                if "image_url" in part or "image" in part or part.get("type") == "image":
                    src = part.get("image_url", part.get("image"))
                    urls.append(src.get("url") if isinstance(src, dict) else src)
        return urls

    def load(self, url):
        import io

        from PIL import Image

        if isinstance(url, str) and url.startswith("data:") and ";base64," in url:
            raw = base64.b64decode(url.split(";base64,", 1)[1])
        elif isinstance(url, str) and url.startswith(("http://", "https://")):
            try:
                with urllib.request.urlopen(url, timeout=30) as r:
                    raw = r.read(self.MAX_BYTES + 1)
            except Exception as e:  # noqa: BLE001 - any fetch error is the client's
                raise ApiError(400, f"cannot fetch image {url}: {e}") from e
            if len(raw) > self.MAX_BYTES:
                raise ApiError(400, "image larger than 32 MiB")
        else:
            raise ApiError(400, "images must be data URLs (data:image/...;base64,...) or http(s) URLs")
        try:
            img = Image.open(io.BytesIO(raw)).convert("RGB")
        except Exception as e:  # noqa: BLE001 - any decoder error is the client's
            raise ApiError(400, f"cannot decode image: {e}") from e
        return raw, img

    def prepare(self, ids, urls, pad_id):
        """(ids with each image's pad expanded to its token ids, [(start, gh, gw)], [pixel_values per image])."""
        from qwen27b.image_tokens import expand_image_pads, image_token_ids

        raws, imgs = zip(*(self.load(u) for u in urls))
        pv, grids = self.proc(imgs, self.max_pixels)
        grids = [list(g) for g in grids]
        keys = [raw + repr((g, self.max_pixels)).encode() for raw, g in zip(raws, grids)]
        ids, spans = expand_image_pads(ids, grids, pad_id, fill=lambda i, n: image_token_ids(keys[i], n))
        out, r = [], 0
        for t, h, w in grids:
            out.append(pv[r:r + t * h * w])
            r += t * h * w
        return ids, [(s.start, s.gh, s.gw) for s in spans], out


# ---------------------------------------------------------------- the engine process


class EngineClient:
    """Starts the engine process (worker.py) and routes its messages to the requests' queues; starts a new
    one when it dies. The engine holds all the VRAM: control() stops it (the server stays up and answers 503),
    starts it again or restarts it (POST /admin/engine, the stats page)."""

    def __init__(self, cfg, autostart=True):
        self.cfg = cfg
        self.lock = threading.Lock()       # handles
        self.send_lock = threading.Lock()  # (a Connection is not thread-safe)
        self.handles = {}
        self.ready = threading.Event()
        self.info = {}
        self.metrics = {}
        self.config = {}                   # the worker's admission settings (worker.Scheduler.settings)
        self.config_seen = threading.Condition()
        self.config_answers = {}           # change number -> the worker's settings after it
        self.config_n = 0
        self.restarts = 0
        self.stopping = False              # the server is shutting down
        self.wanted = autostart            # the engine should run (stopped from the admin API: False)
        self.life = threading.Lock()       # start / stop / restart, one at a time
        self.gen = 0                       # engine generation (a stop or start makes a pending auto-restart stale)
        self.proc = None
        if autostart:
            self.start()
        else:
            HUB.engine_state_set("stopped", "engine not started (ENGINE_AUTOSTART=0)")

    def start(self):
        HUB.engine_starting()
        ctx = mp.get_context("spawn")
        parent, child = ctx.Pipe()
        self.proc = ctx.Process(target=worker.run, args=(child, self.cfg), daemon=True, name="q27-engine")
        self.proc.start()
        child.close()
        self.conn = parent
        threading.Thread(target=self.reader, args=(self.gen,), daemon=True, name="engine-reader").start()

    def state(self):
        """stopped | loading | ready | restarting (after a crash) | stopping."""
        return HUB.engine_state

    def control(self, action):
        """start | stop | restart the engine process; returns the state afterwards (start: "loading")."""
        with self.life:
            if action in ("stop", "restart"):
                self._stop()
            if action in ("start", "restart") and not (self.proc is not None and self.proc.is_alive()):
                self.wanted, self.restarts = True, 0
                self.gen += 1
                self.start()
        return self.state()

    def _stop(self):
        self.wanted = False
        self.gen += 1
        proc = self.proc
        if proc is None or not proc.is_alive():
            HUB.engine_state_set("stopped", "engine stopped")
            return
        HUB.engine_state_set("stopping", "stopping the engine")
        if self.ready.is_set():  # (while loading it reads no messages)
            try:
                with self.send_lock:
                    self.conn.send(("stop",))
            except OSError:
                pass
            proc.join(timeout=15)
        for kill in (proc.terminate, proc.kill):
            if proc.is_alive():
                kill()
                proc.join(timeout=10)
        self.ready.clear()
        log(f"engine stopped (exit code {proc.exitcode})")
        HUB.engine_state_set("stopped", "engine stopped: its VRAM is free")

    def reader(self, gen):
        conn = self.conn
        try:
            while True:
                msg = conn.recv()
                kind = msg[0]
                if kind == "ready":
                    self.info = msg[1]
                    self.ready.set()
                    log(f"engine ready: {self.info}")
                    HUB.engine_ready(self.info, self.cfg)
                elif kind == "memory":
                    HUB.set_memory(msg[1])
                elif kind == "stats":
                    HUB.set_snapshot(msg[1])
                elif kind == "event":
                    HUB.event(msg[1], msg[2], msg[3])
                elif kind == "metrics":
                    self.metrics.update(msg[1])
                elif kind == "config":
                    with self.config_seen:
                        self.config = msg[1]
                        if msg[2] is not None:
                            self.config_answers[msg[2]] = msg[1]
                        self.config_seen.notify_all()
                elif kind == "fatal":
                    log(f"engine failed: {msg[1]}")
                else:
                    rid = msg[1]
                    if kind == "done":
                        self.metrics.update(msg[3].pop("metrics", {}))
                        HUB.finished(rid, msg[2], msg[3])
                    elif kind == "error":
                        HUB.failed(rid, msg[2])
                    with self.lock:
                        q = self.handles.get(rid)
                        if kind in ("done", "error"):
                            self.handles.pop(rid, None)
                    if q is not None:
                        q.put(msg)
        except (EOFError, OSError):
            pass
        self.ready.clear()
        planned = self.stopping or not self.wanted or gen != self.gen
        with self.lock:
            handles, self.handles = self.handles, {}
        for q in handles.values():
            q.put(("error", None, "the engine was stopped" if planned else "the engine stopped; it is restarting", 503))
        if planned:
            return
        self.proc.join(timeout=10)
        code = self.proc.exitcode
        self.restarts += 1
        delay = min(60, 2 ** min(self.restarts, 6))
        log(f"engine exited with code {code}; starting a new one in {delay} s")
        HUB.engine_stopped(f"engine exited with code {code}; starting a new one in {delay} s")
        time.sleep(delay)
        with self.life:
            if self.wanted and gen == self.gen and not self.stopping:
                self.start()

    def not_ready(self):
        if not self.wanted:
            return ApiError(503, "the engine is stopped (start it at /ui or with POST /admin/engine)", "overloaded_error")
        return ApiError(503, "the engine is loading", "overloaded_error")

    def submit(self, req):
        if not self.ready.is_set():
            raise self.not_ready()
        rid = new_id("req-", 16)
        q = queue.Queue()
        with self.lock:
            self.handles[rid] = q
        try:
            with self.send_lock:
                self.conn.send(("submit", rid, req))
        except OSError as e:
            with self.lock:
                self.handles.pop(rid, None)
            raise ApiError(503, "the engine is restarting", "overloaded_error") from e
        HUB.submitted(rid)
        return rid, q

    def configure(self, changes):
        """Changes the admission settings (kept for a restarted engine); returns the worker's settings."""
        if not self.ready.is_set():
            raise self.not_ready()
        self.cfg.update(changes)
        with self.config_seen:
            self.config_n += 1
            n = self.config_n
            try:
                with self.send_lock:
                    self.conn.send(("config", changes, n))
            except OSError as e:
                raise ApiError(503, "the engine is restarting", "overloaded_error") from e
            if not self.config_seen.wait_for(lambda: n in self.config_answers, timeout=10):
                raise ApiError(503, "the engine did not answer", "overloaded_error")
            return self.config_answers.pop(n)

    def cancel(self, rid):
        with self.lock:
            live = rid in self.handles
        if live:
            try:
                with self.send_lock:
                    self.conn.send(("cancel", rid))
            except OSError:
                pass

    def stop(self):
        self.stopping = True
        if self.proc is None:
            return
        try:
            with self.send_lock:
                self.conn.send(("stop",))
        except OSError:
            pass
        self.proc.join(timeout=10)
        if self.proc.is_alive():
            self.proc.terminate()


# ---------------------------------------------------------------- one generation


class ClientGone(Exception):
    pass


def client_gone(sock):
    """The client closed the connection (readable with nothing to read). Data waiting (a pipelined request)
    does not count."""
    try:
        r, _, _ = select.select([sock], [], [], 0)
        return bool(r) and sock.recv(1, socket.MSG_PEEK) == b""
    except (OSError, ValueError):
        return True


class Detokenizer:
    """Incremental detokenization: text is released only at complete UTF-8 characters."""

    def __init__(self, tok, lock):
        self.tok, self.lock = tok, lock
        self.ids, self.prefix, self.read = [], 0, 0

    def _decode(self, ids):
        with self.lock:
            return self.tok.decode(ids)

    def add(self, ids, final=False):
        self.ids += ids
        prefix_text = self._decode(self.ids[self.prefix:self.read])
        new_text = self._decode(self.ids[self.prefix:])
        if len(new_text) > len(prefix_text) and (final or not new_text.endswith("�")):
            self.prefix, self.read = self.read, len(self.ids)
            return new_text[len(prefix_text):]
        return ""


class Generation:
    """One request on the engine. Iterating yields text pieces; afterwards finish ("stop" | "length"),
    stop_sequence, usage and timings are set. Stop strings are matched here (text that may still start one
    is held back) and cancel the request; so does a closed connection (ClientGone)."""

    def __init__(self, server, ids, max_new, sampling, stop=(), images=None, sock=None, keepalive=None):
        self.server, self.sock, self.keepalive = server, sock, keepalive
        self.stops = [s for s in stop if s]
        self.prompt_tokens = len(ids)
        self.finish, self.stop_sequence, self.cached = None, None, 0
        self.tokens = 0
        req = dict(ids=ids, max_new=max_new, images=images, **sampling)
        self.rid, self.q = server.engine.submit(req)
        self.detok = Detokenizer(server.tok, server.tok_lock)
        self.text, self.sent = "", 0

    def cancel(self):
        self.server.engine.cancel(self.rid)

    def _release(self, final=False):
        """The text that is safe to send (not possibly the start of a stop string)."""
        if self.stops:
            start = max(0, self.sent - max(map(len, self.stops)))
            hits = [(self.text.find(s, start), s) for s in self.stops]
            hits = [(i, s) for i, s in hits if i >= 0]
            if hits:
                i, s = min(hits)
                self.text = self.text[:i]
                self.finish, self.stop_sequence = "stop", s
                self.cancel()
                final = True
            safe = len(self.text) if final else len(self.text) - (max(map(len, self.stops)) - 1)
        else:
            safe = len(self.text)
        if safe > self.sent:
            piece, self.sent = self.text[self.sent:safe], safe
            return piece
        return ""

    def __iter__(self):
        last = checked = time.time()
        try:
            while self.finish is None:
                if self.sock is not None and time.time() - checked > 1.0:  # (also while tokens keep coming)
                    checked = time.time()
                    if client_gone(self.sock):
                        raise ClientGone()
                try:
                    msg = self.q.get(timeout=0.5)
                except queue.Empty:
                    if self.keepalive and time.time() - last > KEEPALIVE_S:
                        self.keepalive()
                        last = time.time()
                    continue
                kind = msg[0]
                if kind == "start":
                    self.cached = msg[2]["cached"]
                elif kind == "tok":
                    self.tokens += len(msg[2])
                    self.text += self.detok.add(msg[2])
                    piece = self._release()
                    if piece:
                        last = time.time()
                        yield piece
                elif kind == "done":
                    info = msg[3]
                    self.cached = info.get("cached_tokens", self.cached)
                    self.timings = info
                    self.text += self.detok.add([], final=True)
                    self.finish = self.finish or ("stop" if msg[2] in ("stop", "cancelled") else msg[2])
                    piece = self._release(final=True)
                    if piece:
                        yield piece
                elif kind == "error":
                    raise ApiError(msg[3] if len(msg) > 3 else 500, msg[2],
                                   "overloaded_error" if (len(msg) > 3 and msg[3] == 503) else "api_error")
        except BaseException:
            if self.finish is None:
                self.cancel()
            raise

    @property
    def usage(self):
        return {"prompt_tokens": self.prompt_tokens, "completion_tokens": self.tokens,
                "total_tokens": self.prompt_tokens + self.tokens,
                "prompt_tokens_details": {"cached_tokens": self.cached}}


# ---------------------------------------------------------------- HTTP


class Server:
    """What the handlers share: the engine client, tokenizer, images, limits."""

    def __init__(self, engine, tok, max_tokens, launch, thinking_default=True):
        self.engine, self.tok, self.launch = engine, tok, launch
        self.tok_lock = threading.Lock()
        self.images_lock = threading.Lock()
        self._images = None
        self.max_tokens_conf = max_tokens
        self.thinking_default = thinking_default
        self.pad_id = tok.token_id("<|image_pad|>")

    @property
    def max_tokens(self):
        """The engine's context capacity (MAX_TOKENS, lowered to what the VRAM fits; MAX_TOKENS_FIT)."""
        return self.engine.info.get("max_tokens", self.max_tokens_conf)

    @property
    def images(self):
        """The image processor while the engine is set up with the vision tower (made when first needed)."""
        cfg = self.engine.cfg
        if cfg["env"]["VISION"] == "off":
            return None
        with self.images_lock:
            if self._images is None:
                self._images = Images(cfg["model"], int(os.environ.get("IMAGE_MAX_PIXELS", 4194304)))
            return self._images

    def encode(self, text):
        with self.tok_lock:
            return self.tok.encode(text)

    def use_template_of(self, weights):
        """Renders prompts with the chat template of `weights` from here on (an engine start on them; their tokenizer
        has the server's ids, Launch.weights_choices). The tokenizer itself stays the server's."""
        from tokenizer import Tokenizer

        path = model_dir(weights)
        if path == self.tok.template_path:
            return
        tok = Tokenizer(self.tok.path, template=path)
        with self.tok_lock:
            self.tok = tok

    def render(self, messages, tools, kwargs):
        """Prompt ids (images expanded) and the images, of chat messages."""
        try:
            with self.tok_lock:
                prompt = self.tok.render(messages, tools=tools or None, **kwargs)
        except Exception as e:  # noqa: BLE001 - template errors (no user message, bad roles) are the client's
            raise ApiError(400, f"chat template: {e}") from e
        ids = self.encode(prompt)
        urls, images = Images.collect(messages), None
        if urls:
            proc = self.images
            if proc is None:
                raise ApiError(400, "this server does not take images (start the engine with VISION=host, fp8 or bf16)")
            ids, spans, pixels = proc.prepare(ids, urls, self.pad_id)
            images = (spans, pixels)
        return ids, images

    def limit(self, n_prompt, anthropic=False):
        left = self.max_tokens - n_prompt - SPEC_MARGIN
        if left <= 0:
            if anthropic:
                raise ApiError(400, f"prompt is too long: {n_prompt} tokens > {self.max_tokens - SPEC_MARGIN - 1} maximum")
            raise ApiError(400, f"This model's maximum context length is {self.max_tokens} tokens. However, your "
                                f"request has {n_prompt} input tokens. Please reduce the length of the input.")
        return left


def sampling_of(body, top_k_key="top_k"):
    temp = body.get("temperature")
    top_p = body.get("top_p")
    top_k = body.get(top_k_key)
    return {"temperature": float(DEFAULTS["temperature"] if temp is None else temp),
            "top_k": int(DEFAULTS["top_k"] if top_k is None else top_k) if (top_k is None or top_k >= 0) else 0,
            "top_p": float(DEFAULTS["top_p"] if top_p is None else top_p), "seed": body.get("seed")}


def stops_of(value):
    if value is None:
        return []
    return [value] if isinstance(value, str) else [s for s in value if isinstance(s, str)]


class Handler(BaseHTTPRequestHandler):
    srv: Server = None
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):
        if self.command == "GET" and self.path.split("?")[0] in ("/health", "/metrics", "/ui/state"):
            return  # (polled by the healthcheck and the benchmarks)
        sys.stderr.write("[api] " + fmt % args + "\n")

    # ------------------------------------------------------------ plumbing

    def _json(self, code, obj):
        body = json.dumps(obj, ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _error(self, code, msg, kind="invalid_request_error"):
        if self.path.split("?")[0].startswith("/v1/messages"):
            kind = {"invalid_request_error": "invalid_request_error", "not_found": "not_found_error"}.get(kind, kind)
            if code == 503:
                kind = "overloaded_error"
            self._json(code, {"type": "error", "error": {"type": kind, "message": msg}})
        else:
            self._json(code, {"error": {"message": msg, "type": kind, "code": code}, "object": "error",
                              "message": msg})

    def _sse_start(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()
        self.close_connection = True

    def _sse(self, data, event=None):
        head = f"event: {event}\n" if event else ""
        self.wfile.write(f"{head}data: {json.dumps(data, ensure_ascii=False)}\n\n".encode())
        self.wfile.flush()

    def _model(self, body):
        m = body.get("model")
        return m if isinstance(m, str) and m else MODEL_NAME

    def do_GET(self):
        path = self.path.split("?")[0].rstrip("/")
        if path == "/health":
            if self.srv.engine.ready.is_set():
                self._json(200, {"status": "ok"})
            else:
                self._json(503, {"status": self.srv.engine.state()})
        elif path == "/metrics":
            body = "".join(f"{k} {v}\n" for k, v in sorted(self.srv.engine.metrics.items())).encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/plain; version=0.0.4")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif path in ("/v1/models", f"/v1/models/{MODEL_NAME}"):
            card = {"id": MODEL_NAME, "object": "model", "type": "model", "display_name": MODEL_NAME,
                    "created": int(START_TIME), "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(START_TIME)),
                    "owned_by": "local", "max_model_len": self.srv.max_tokens}
            if path == "/v1/models":
                self._json(200, {"object": "list", "data": [card], "has_more": False, "first_id": MODEL_NAME,
                                 "last_id": MODEL_NAME})
            else:
                self._json(200, card)
        elif path == "/admin/concurrency":
            if not self.srv.engine.ready.is_set():
                e = self.srv.engine.not_ready()
                return self._error(e.code, str(e), e.kind)
            self._json(200, self.srv.engine.config)
        elif path == "/admin/engine":
            self._json(200, self.engine_settings())
        elif path == "/version":
            self._json(200, {"version": "q27-phase6"})
        elif path == "/ui":
            body = UI_PAGE.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            self.wfile.write(body)
        elif path == "/ui/state":
            self._json(200, dict(HUB.init(), tick=HUB.tick()))
        elif path == "/ui/stream":
            self.ui_stream()
        else:
            self._error(404, "not found", "not_found")

    def ui_stream(self):
        """Server-sent events of the stats UI: "init" (configuration, sample history, log), then "tick" (the
        requests, VRAM, last GPU sample) every second and "log" (new entries) as they come."""
        self._sse_start()
        try:
            init = dict(HUB.init(), tick=HUB.tick())
            self._sse(init, "init")
            seen = init["log"][-1]["n"] if init["log"] else 0
            last = time.time()
            while True:
                HUB.wait(1.0)
                new = HUB.log_since(seen)
                if new:
                    seen = new[-1]["n"]
                    self._sse(new, "log")
                if time.time() - last >= 1.0:
                    last = time.time()
                    self._sse(HUB.tick(), "tick")
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass

    def do_HEAD(self):
        self.send_response(200 if self.srv.engine.ready.is_set() else 503)
        self.send_header("Content-Length", "0")
        self.end_headers()

    def do_POST(self):
        try:
            n = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(n) or b"{}")
            if not isinstance(body, dict):
                raise ValueError("the request body must be a JSON object")
        except (ValueError, json.JSONDecodeError) as e:
            return self._error(400, f"bad JSON: {e}")
        path = self.path.split("?")[0].rstrip("/")
        if DUMP_DIR:  # (debug: the requests as clients send them)
            name = f"{time.strftime('%H%M%S')}-{time.time_ns() % 10**9:09d}{path.replace('/', '_')}.json"
            (pathlib.Path(DUMP_DIR) / name).write_text(json.dumps(body, ensure_ascii=False, indent=1))
        routes = {"/v1/chat/completions": self.chat, "/v1/completions": self.completions, "/v1/responses": self.responses_api,
                  "/v1/messages": self.messages, "/v1/messages/count_tokens": self.count_tokens,
                  "/tokenize": self.tokenize, "/detokenize": self.detokenize, "/admin/concurrency": self.concurrency,
                  "/admin/engine": self.engine_control, "/admin/engine/plan": self.engine_plan}
        fn = routes.get(path)
        if fn is None:
            return self._error(404, "not found", "not_found")
        HUB.local.meta = request_meta(path, self.headers, body) if path in GENERATE else None
        try:
            fn(body)
        except ApiError as e:
            HUB.http_error(path, e.code, str(e))
            self._error(e.code, str(e), e.kind)
        except (AnthropicError, ValueError, KeyError, TypeError) as e:
            self._error(400, f"{type(e).__name__}: {e}" if not isinstance(e, (AnthropicError, ValueError)) else str(e))
        except (BrokenPipeError, ConnectionResetError, ClientGone):
            log(f"{path}: client disconnected")
            self.close_connection = True
        except Exception as e:  # noqa: BLE001 - report instead of dropping the connection
            traceback.print_exc()
            try:
                self._error(500, f"{type(e).__name__}: {e}", "api_error")
            except OSError:
                pass

    # ------------------------------------------------------------ OpenAI chat completions

    def chat(self, body):
        if (body.get("n") or 1) != 1:
            raise ApiError(400, "n > 1 is not supported")
        if body.get("logprobs") or body.get("top_logprobs"):
            raise ApiError(400, "logprobs are not supported")
        kwargs = template_kwargs(body, self.srv.thinking_default)
        thinking = kwargs.get("enable_thinking", True) is not False
        tools = body.get("tools") or None
        parse_tools = tools is not None and body.get("tool_choice") != "none"
        messages = normalize_messages(body.get("messages"))
        ids, images = self.srv.render(messages, tools, kwargs)
        left = self.srv.limit(len(ids))
        max_new = body.get("max_completion_tokens") or body.get("max_tokens") or left
        if max_new <= 0:
            raise ApiError(400, "max_tokens must be positive")
        stream = bool(body.get("stream"))
        keep = (lambda: self._keepalive_comment()) if stream else None
        gen = Generation(self.srv, ids, min(max_new, left), sampling_of(body), stops_of(body.get("stop")), images,
                         self.connection, keep)
        parser = OutputParser(thinking, tools if parse_tools else None)
        rid, created, model = f"chatcmpl-{gen.rid[4:]}", int(time.time()), self._model(body)
        single_call = body.get("parallel_tool_calls") is False
        if stream:
            return self._chat_stream(gen, parser, rid, created, model, body, single_call)
        reasoning, content, calls = [], [], []
        for piece in gen:
            for kind, v in parser.feed(piece):
                (reasoning if kind == "reasoning" else content if kind == "content" else calls).append(v)
        for kind, v in parser.finish():
            (reasoning if kind == "reasoning" else content if kind == "content" else calls).append(v)
        if single_call:
            calls = calls[:1]
        msg = {"role": "assistant", "content": "".join(content) or (None if calls else "")}
        if thinking:
            msg["reasoning_content"] = msg["reasoning"] = "".join(reasoning)
        finish = gen.finish
        if calls:
            msg["tool_calls"] = [openai_call(c) for c in calls]
            finish = "tool_calls"
        self._json(200, {"id": rid, "object": "chat.completion", "created": created, "model": model,
                         "choices": [{"index": 0, "message": msg, "finish_reason": finish, "logprobs": None,
                                      "stop_reason": gen.stop_sequence}],
                         "usage": gen.usage})

    def _keepalive_comment(self):
        self.wfile.write(b": keep-alive\n\n")
        self.wfile.flush()

    def _chat_stream(self, gen, parser, rid, created, model, body, single_call):
        self._sse_start()

        def chunk(delta, finish=None):
            self._sse({"id": rid, "object": "chat.completion.chunk", "created": created, "model": model,
                       "choices": [{"index": 0, "delta": delta, "finish_reason": finish, "logprobs": None}]})

        chunk({"role": "assistant", "content": ""})
        ncalls = 0

        def send(events):
            nonlocal ncalls
            for kind, v in events:
                if kind == "reasoning":
                    chunk({"reasoning_content": v, "reasoning": v})
                elif kind == "content":
                    chunk({"content": v})
                elif not (single_call and ncalls):
                    chunk({"tool_calls": [openai_call(v, ncalls)]})
                    ncalls += 1

        try:
            for piece in gen:
                send(parser.feed(piece))
            send(parser.finish())
        except ApiError as e:
            self._sse({"error": {"message": str(e), "type": e.kind, "code": e.code}})
            self.wfile.write(b"data: [DONE]\n\n")
            return
        chunk({}, "tool_calls" if ncalls else gen.finish)
        if (body.get("stream_options") or {}).get("include_usage"):
            self._sse({"id": rid, "object": "chat.completion.chunk", "created": created, "model": model,
                       "choices": [], "usage": gen.usage})
        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()

    # ------------------------------------------------------------ OpenAI completions

    def completions(self, body):
        prompt = body.get("prompt", "")
        if isinstance(prompt, list) and prompt and isinstance(prompt[0], (str, list)):
            if len(prompt) != 1:
                raise ApiError(400, "only one prompt per request is supported")
            prompt = prompt[0]
        ids = prompt if isinstance(prompt, list) else self.srv.encode(prompt)
        if not ids:
            raise ApiError(400, "empty prompt")
        left = self.srv.limit(len(ids))
        max_new = min(body.get("max_tokens") or 16, left)
        stream = bool(body.get("stream"))
        gen = Generation(self.srv, ids, max_new, sampling_of(body), stops_of(body.get("stop")), None,
                         self.connection, (lambda: self._keepalive_comment()) if stream else None)
        rid, created, model = f"cmpl-{gen.rid[4:]}", int(time.time()), self._model(body)
        if stream:
            self._sse_start()
            for piece in gen:
                self._sse({"id": rid, "object": "text_completion", "created": created, "model": model,
                           "choices": [{"index": 0, "text": piece, "finish_reason": None, "logprobs": None}]})
            self._sse({"id": rid, "object": "text_completion", "created": created, "model": model,
                       "choices": [{"index": 0, "text": "", "finish_reason": gen.finish, "logprobs": None}]})
            if (body.get("stream_options") or {}).get("include_usage"):
                self._sse({"id": rid, "object": "text_completion", "created": created, "model": model,
                           "choices": [], "usage": gen.usage})
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
            return
        text = "".join(gen)
        self._json(200, {"id": rid, "object": "text_completion", "created": created, "model": model,
                         "choices": [{"index": 0, "text": text, "finish_reason": gen.finish, "logprobs": None}],
                         "usage": gen.usage})

    # ------------------------------------------------------------ Anthropic messages

    def _anthropic_prompt(self, body):
        messages, tools = anthropic_messages(body)
        kwargs = anthropic_template_kwargs(body, self.srv.thinking_default)
        ids, images = self.srv.render(normalize_messages(messages), tools, kwargs)
        return ids, images, tools, kwargs.get("enable_thinking", True) is not False

    def count_tokens(self, body):
        ids, _, _, _ = self._anthropic_prompt(body)
        self._json(200, {"input_tokens": len(ids)})

    def messages(self, body):
        if not isinstance(body.get("max_tokens"), int) or body["max_tokens"] <= 0:
            raise ApiError(400, "max_tokens: a positive integer is required")
        ids, images, tools, thinking = self._anthropic_prompt(body)
        left = self.srv.limit(len(ids), anthropic=True)
        tc = body.get("tool_choice") or {}
        parse_tools = bool(tools) and tc.get("type") != "none"
        single_call = bool(tc.get("disable_parallel_tool_use"))
        stream = bool(body.get("stream"))
        keep = (lambda: self._sse({"type": "ping"}, "ping")) if stream else None
        gen = Generation(self.srv, ids, min(body["max_tokens"], left), sampling_of(body),
                         stops_of(body.get("stop_sequences")), images, self.connection, keep)
        parser = OutputParser(thinking, tools if parse_tools else None)
        mid, model = f"msg_{gen.rid[4:]}", self._model(body)
        omit = (body.get("thinking") or {}).get("display") == "omitted" if isinstance(body.get("thinking"), dict) else False

        def usage(final):
            u = {"input_tokens": len(ids) - (gen.cached if final else 0), "output_tokens": gen.tokens if final else 0,
                 "cache_creation_input_tokens": 0, "cache_read_input_tokens": gen.cached if final else 0}
            return u

        def stop_reason(ncalls):
            if ncalls:
                return "tool_use"
            if gen.stop_sequence is not None:
                return "stop_sequence"
            return STOP_REASON.get(gen.finish, "end_turn")

        if not stream:
            blocks, ncalls = [], 0
            for kind, v in [e for piece in gen for e in parser.feed(piece)] + parser.finish():
                if kind == "tool_call":
                    if single_call and ncalls:
                        continue
                    ncalls += 1
                    blocks.append({"type": "tool_use", "id": new_id("toolu_"), "name": v["name"], "input": v["arguments"]})
                    continue
                btype = "thinking" if kind == "reasoning" else "text"
                if blocks and blocks[-1]["type"] == btype:
                    blocks[-1][btype] += v
                else:
                    blocks.append({"type": btype, btype: v, **({"signature": ""} if btype == "thinking" else {})})
            if omit:
                for b in blocks:
                    if b["type"] == "thinking":
                        b["signature"], b["thinking"] = thinking_signature(b["thinking"]), ""
            self._json(200, {"id": mid, "type": "message", "role": "assistant", "model": model, "content": blocks,
                             "stop_reason": stop_reason(ncalls), "stop_sequence": gen.stop_sequence,
                             "usage": usage(True)})
            return

        self._sse_start()
        self._sse({"type": "message_start", "message": {
            "id": mid, "type": "message", "role": "assistant", "model": model, "content": [], "stop_reason": None,
            "stop_sequence": None, "usage": usage(False)}}, "message_start")
        state = {"index": -1, "type": None, "calls": 0, "thinking": ""}

        def close_block():
            if state["type"] == "thinking" and omit:
                self._sse({"type": "content_block_delta", "index": state["index"],
                           "delta": {"type": "signature_delta", "signature": thinking_signature(state["thinking"])}},
                          "content_block_delta")
            if state["type"] is not None:
                self._sse({"type": "content_block_stop", "index": state["index"]}, "content_block_stop")
                state["type"] = None

        def open_block(block):
            close_block()
            state["index"] += 1
            state["type"] = block["type"]
            self._sse({"type": "content_block_start", "index": state["index"], "content_block": block},
                      "content_block_start")

        def send(events):
            for kind, v in events:
                if kind == "tool_call":
                    if single_call and state["calls"]:
                        continue
                    state["calls"] += 1
                    open_block({"type": "tool_use", "id": new_id("toolu_"), "name": v["name"], "input": {}})
                    self._sse({"type": "content_block_delta", "index": state["index"],
                               "delta": {"type": "input_json_delta",
                                         "partial_json": json.dumps(v["arguments"], ensure_ascii=False)}},
                              "content_block_delta")
                    close_block()
                elif kind == "reasoning":
                    if state["type"] != "thinking":
                        open_block({"type": "thinking", "thinking": "", "signature": ""})
                        state["thinking"] = ""
                    state["thinking"] += v
                    if not omit:
                        self._sse({"type": "content_block_delta", "index": state["index"],
                                   "delta": {"type": "thinking_delta", "thinking": v}}, "content_block_delta")
                else:
                    if state["type"] != "text":
                        open_block({"type": "text", "text": ""})
                    self._sse({"type": "content_block_delta", "index": state["index"],
                               "delta": {"type": "text_delta", "text": v}}, "content_block_delta")

        try:
            for piece in gen:
                send(parser.feed(piece))
            send(parser.finish())
        except ApiError as e:
            self._sse({"type": "error", "error": {"type": "overloaded_error" if e.code == 503 else "api_error",
                                                  "message": str(e)}}, "error")
            return
        close_block()
        self._sse({"type": "message_delta", "delta": {"stop_reason": stop_reason(state["calls"]),
                                                      "stop_sequence": gen.stop_sequence},
                   "usage": usage(True)}, "message_delta")
        self._sse({"type": "message_stop"}, "message_stop")

    # ------------------------------------------------------------ OpenAI Responses API (Codex)

    def responses_api(self, body):  # (not "responses": that is the base class status table)
        if body.get("previous_response_id"):
            raise ApiError(400, "previous_response_id is not supported (the server stores no responses; send the input)")
        messages, tools, namespaces, custom = responses_messages(body)
        kwargs = responses_template_kwargs(body, self.srv.thinking_default)
        thinking = kwargs.get("enable_thinking", True) is not False
        ids, images = self.srv.render(normalize_messages(messages), tools, kwargs)
        left = self.srv.limit(len(ids))
        max_new = min(body.get("max_output_tokens") or left, left)
        parse_tools = bool(tools) and body.get("tool_choice") != "none"
        stream = bool(body.get("stream"))
        gen = Generation(self.srv, ids, max_new, sampling_of(body), (), images, self.connection,
                         (lambda: self._keepalive_comment()) if stream else None)
        parser = OutputParser(thinking, tools if parse_tools else None)
        resp = {"id": f"resp_{gen.rid[4:]}", "object": "response", "created_at": int(time.time()), "status": "in_progress",
                "model": self._model(body), "output": [], "instructions": body.get("instructions"),
                "parallel_tool_calls": body.get("parallel_tool_calls", True), "tool_choice": body.get("tool_choice", "auto"),
                "tools": body.get("tools") or [], "temperature": body.get("temperature"), "top_p": body.get("top_p"),
                "reasoning": body.get("reasoning"), "store": False, "text": body.get("text"),
                "max_output_tokens": body.get("max_output_tokens"), "usage": None, "error": None,
                "incomplete_details": None, "metadata": body.get("metadata") or {}}
        out = ResponsesOut(self._sse if stream else None, "reasoning.encrypted_content" in (body.get("include") or []),
                           namespaces, custom, body.get("parallel_tool_calls") is False)
        if stream:
            self._sse_start()
            out.ev("response.created", response=resp)
            out.ev("response.in_progress", response=resp)
        try:
            for piece in gen:
                out.feed(parser.feed(piece))
            out.feed(parser.finish())
        except ApiError as e:
            if not stream:
                raise
            out.ev("response.failed", response=dict(resp, status="failed", error={"code": "server_error", "message": str(e)}))
            return
        out.close()
        done = gen.finish == "length" and not out.ncalls
        resp.update(output=out.items, status="incomplete" if done else "completed",
                    incomplete_details={"reason": "max_output_tokens"} if done else None,
                    usage={"input_tokens": len(ids), "input_tokens_details": {"cached_tokens": gen.cached},
                           "output_tokens": gen.tokens, "output_tokens_details": {"reasoning_tokens": out.reasoning_tokens(self.srv)},
                           "total_tokens": len(ids) + gen.tokens})
        if stream:
            out.ev("response.incomplete" if done else "response.completed", response=resp)
            self.wfile.flush()
        else:
            text = "".join(c.get("text", "") for it in out.items if it["type"] == "message" for c in it["content"])
            self._json(200, dict(resp, output_text=text))

    # ------------------------------------------------------------ tokenizer endpoints (vLLM)

    def concurrency(self, body):
        """Admission settings (P7-14): target_seqs (1 .. MAX_SEQS), min_seq_tps (0: off), admit_kv, kv_reserve."""
        least = {"target_seqs": 1, "min_seq_tps": 0, "kv_reserve": 0}  # (admit_kv: a boolean)
        for k, v in body.items():
            if k == "admit_kv":
                ok = isinstance(v, bool)
            elif k in least:
                ok = isinstance(v, (int, float)) and not isinstance(v, bool) and v >= least[k]
                ok = ok and (k == "min_seq_tps" or float(v).is_integer())
            else:
                raise ApiError(400, f"unknown setting {k!r} (settings: admit_kv, {', '.join(least)})")
            if not ok:
                raise ApiError(400, f"{k}: bad value {v!r}")
        changes = {k: int(v) if k in ("target_seqs", "kv_reserve") else v for k, v in body.items()}
        self._json(200, self.srv.engine.configure(changes))

    def engine_settings(self):
        """GET /admin/engine: the state, the settings of the running engine and of the next start, the weights."""
        eng, launch = self.srv.engine, self.srv.launch
        (device, source) = HUB.plan_device()
        running = eng.cfg.get("settings") if eng.state() != "stopped" else None
        return {"state": eng.state(), "running": running, "settings": launch.settings,
                "weights": launch.weights_choices(), "device": {"free": device[0], "total": device[1], "source": source}
                if device else None}

    def engine_plan(self, body):
        """{"settings": {...}} (changes to the next start's settings): their memory plan and, for each setting, the
        plans of its other values."""
        settings = body.get("settings") or {}
        if not isinstance(settings, dict):
            raise ApiError(400, "settings: an object")
        device, source = HUB.plan_device()
        if device is None:
            raise ApiError(503, "the device's memory is unknown (no NVML and no engine started yet)", "overloaded_error")
        s = self.srv.launch.validate(settings, device=device, check_budget=False)
        self._json(200, dict(self.srv.launch.plans(s, device), device={"free": device[0], "total": device[1],
                                                                        "source": source}))

    def engine_control(self, body):
        """{"action": "start" | "stop" | "restart", "settings": {...}}: the engine process (stop frees its VRAM;
        requests in flight fail with 503). Start and restart take changes to the start settings (launch.py), kept for
        later starts. Needs Content-Type: application/json, so that a web page cannot send it with a plain form."""
        if self.headers.get_content_type() != "application/json":
            raise ApiError(415, "send the action as JSON (Content-Type: application/json)")
        action, settings = body.get("action"), body.get("settings")
        if action not in ("start", "stop", "restart"):
            raise ApiError(400, "action: start, stop or restart")
        eng, launch = self.srv.engine, self.srv.launch
        note = ""
        if settings is not None:
            if action == "stop" or not isinstance(settings, dict):
                raise ApiError(400, "settings: an object, with start or restart")
            if action == "start" and eng.state() != "stopped":
                raise ApiError(409, "the engine is running: restart it to change its settings")
            device, _ = HUB.plan_device()
            s = launch.validate(settings, device=device)
            if device is not None:
                plan = launch.plan_many([s], device)[0]
                if "error" in plan or not plan["fits"]:
                    raise ApiError(400, f"these settings do not fit: {plan.get('error') or 'the device pools exceed the budget'}")
            try:
                cfg = launch.config(s)
            except FileNotFoundError as e:
                raise ApiError(400, str(e)) from e
            # (the admission settings of /admin/concurrency stay; the target number of requests with the same MAX_SEQS)
            keep = ("admit_kv", "kv_reserve", "min_seq_tps") + (("target_seqs",) if s["max_seqs"] == eng.cfg["max_seqs"] else ())
            self.srv.use_template_of(s["weights"])
            launch.settings, eng.cfg = s, dict(cfg, **{k: eng.cfg[k] for k in keep if k in eng.cfg})
            note = " with " + ", ".join(f"{k}={v}" for k, v in s.items())
        HUB.add_log("admin", None, f"engine {action} requested{note}", level="warn" if action != "start" else "info")
        self._json(200, {"state": eng.control(action), "settings": launch.settings})

    def tokenize(self, body):
        if "messages" in body:
            kwargs = template_kwargs(body, self.srv.thinking_default)
            ids, _ = self.srv.render(normalize_messages(body["messages"]), body.get("tools"), kwargs)
        else:
            ids = self.srv.encode(body.get("prompt", ""))
        self._json(200, {"tokens": ids, "count": len(ids), "max_model_len": self.srv.max_tokens})

    def detokenize(self, body):
        with self.srv.tok_lock:
            text = self.srv.tok.decode(body.get("tokens") or [])
        self._json(200, {"prompt": text})


class ResponsesOut:
    """Output items of a Responses API answer from the parser's events, with the streaming events (sse: None
    for a non-streaming answer). Reasoning goes out as reasoning_text (and, when the client asks for
    `reasoning.encrypted_content`, also in it, so that it comes back with the next request)."""

    def __init__(self, sse, encrypted, namespaces, custom, single_call):
        self.sse, self.encrypted, self.namespaces, self.custom, self.single = sse, encrypted, namespaces, custom, single_call
        self.items, self.cur, self.seq, self.ncalls = [], None, 0, 0

    def ev(self, kind, **kw):
        if self.sse:
            self.sse({"type": kind, "sequence_number": self.seq, **kw}, kind)
            self.seq += 1

    def _public(self, item):
        return {k: v for k, v in item.items() if not k.startswith("_")}

    def _open(self, item):
        self.close()
        self.items.append(item)
        self.cur = item
        self.ev("response.output_item.added", output_index=len(self.items) - 1, item=self._public(item))

    def close(self):
        it, idx = self.cur, len(self.items) - 1
        if it is None:
            return
        text = it.pop("_text")
        if it["type"] == "reasoning":
            it["content"] = [{"type": "reasoning_text", "text": text}]
            if self.encrypted:
                it["encrypted_content"] = thinking_signature(text)
            self.ev("response.reasoning_text.done", item_id=it["id"], output_index=idx, content_index=0, text=text)
        else:
            part = {"type": "output_text", "text": text, "annotations": [], "logprobs": []}
            self.ev("response.output_text.done", item_id=it["id"], output_index=idx, content_index=0, text=text, logprobs=[])
            self.ev("response.content_part.done", item_id=it["id"], output_index=idx, content_index=0, part=part)
            it["content"], it["status"] = [part], "completed"
        self.ev("response.output_item.done", output_index=idx, item=it)
        self.cur = None

    def feed(self, events):
        for kind, v in events:
            if kind == "reasoning":
                if self.cur is None or self.cur["type"] != "reasoning":
                    self._open({"type": "reasoning", "id": new_id("rs_"), "summary": [], "_text": ""})
                self.cur["_text"] += v
                self.ev("response.reasoning_text.delta", item_id=self.cur["id"], output_index=len(self.items) - 1,
                        content_index=0, delta=v)
            elif kind == "content":
                if self.cur is None or self.cur["type"] != "message":
                    self._open({"type": "message", "id": new_id("msg_"), "role": "assistant", "status": "in_progress",
                                "content": [], "_text": ""})
                    self.ev("response.content_part.added", item_id=self.cur["id"], output_index=len(self.items) - 1,
                            content_index=0, part={"type": "output_text", "text": "", "annotations": [], "logprobs": []})
                self.cur["_text"] += v
                self.ev("response.output_text.delta", item_id=self.cur["id"], output_index=len(self.items) - 1,
                        content_index=0, delta=v, logprobs=[])
            elif not (self.single and self.ncalls):
                self.close()
                self.ncalls += 1
                name, args = v["name"], v["arguments"]
                idx = len(self.items)
                if name in self.custom:
                    text = args.get("input", "") if isinstance(args, dict) else str(args)
                    item = {"type": "custom_tool_call", "id": new_id("ctc_"), "call_id": new_id("call_"), "name": name,
                            "input": text, "status": "completed"}
                    self.items.append(item)
                    self.ev("response.output_item.added", output_index=idx, item=dict(item, input="", status="in_progress"))
                    self.ev("response.custom_tool_call_input.delta", item_id=item["id"], output_index=idx, delta=text)
                    self.ev("response.custom_tool_call_input.done", item_id=item["id"], output_index=idx, input=text)
                else:
                    arguments = json.dumps(args, ensure_ascii=False)
                    item = {"type": "function_call", "id": new_id("fc_"), "call_id": new_id("call_"), "name": name,
                            "arguments": arguments, "status": "completed"}
                    if name in self.namespaces:
                        item["namespace"] = self.namespaces[name]
                    self.items.append(item)
                    self.ev("response.output_item.added", output_index=idx, item=dict(item, arguments="", status="in_progress"))
                    self.ev("response.function_call_arguments.delta", item_id=item["id"], output_index=idx, delta=arguments)
                    self.ev("response.function_call_arguments.done", item_id=item["id"], output_index=idx, arguments=arguments)
                self.ev("response.output_item.done", output_index=idx, item=item)

    def reasoning_tokens(self, srv):
        text = "".join(c["text"] for it in self.items if it["type"] == "reasoning" for c in it.get("content", []))
        return len(srv.encode(text)) if text else 0


START_TIME = time.time()
# Engine settings whose library default (q27lib.Engine, for the tests and scripts) is not the server's, set when
# unset: the server starts the same with or without compose.yaml, which lists them too (tests/test_parsing.py).
# MAX_TOKENS is an upper bound the engine lowers to what the VRAM budget fits (e.g. with MAX_SEQS=8); the prefix
# cache keeps 4 GiB of pinned state checkpoints.
ENGINE_ENV_DEFAULTS = {"MAX_TOKENS_FIT": "1", "STATE_HOST_CACHE": "4"}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    env = os.environ.get
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=int(env("PORT", 8000)))
    ap.add_argument("--max-tokens", type=int, default=int(env("MAX_TOKENS", 262144)))
    ap.add_argument("--prefill-chunk", type=int, default=int(env("PREFILL_CHUNK", 2048)))
    ap.add_argument("--kv-dtype", default=env("KV_DTYPE", "nvfp4"), choices=["bf16", "fp8", "nvfp4"])
    ap.add_argument("--spec", default=env("SPEC", "dflash2"), choices=["dflash2", "none"])
    ap.add_argument("--weights", default=env("WEIGHTS", "gdnfp4"),
                    help="gdnfp4: GDN projections in NVFP4, attention FP8 (Phase Q, scripts/build-gdnfp4.sh); nvidia: "
                         "FP8 GDN / attention projections; quasar: every linear in NVFP4; or a model directory")
    ap.add_argument("--max-seqs", type=int, default=int(env("MAX_SEQS", 8)),
                    help="requests the engine can run at once (1..8); fewer run while the KV pool is short")
    ap.add_argument("--min-seq-tps", type=float, default=float(env("MIN_SEQ_TPS", 0)),
                    help="start another request only if each one keeps this many tok/s (0: off)")
    ap.add_argument("--admit-kv", type=int, default=int(env("ADMIT_KV", 1)),
                    help="1: a request waits until the KV pool has room for it (instead of suspending one later)")
    ap.add_argument("--kv-reserve", type=int, default=int(env("KV_RESERVE_TOKENS", 4096)),
                    help="tokens of output per request the KV admission keeps room for")
    ap.add_argument("--decode-share", type=float, default=float(env("DECODE_SHARE", 0.5)),
                    help="decode time given to running requests per second of prefill")
    ap.add_argument("--quantum-ms", type=float, default=float(env("SCHED_QUANTUM_MS", 50)),
                    help="decode turn of a request when several decode")
    ap.add_argument("--batch-decode", type=int, default=int(env("BATCH_DECODE", 1)),
                    help="1: requests in decode step together (batched speculative steps); 0: they take turns")
    args = ap.parse_args()

    from tokenizer import Tokenizer, fingerprint

    path = model_dir(args.weights)  # (the tokenizer: other weights must have the same, Launch.weights_choices)
    tok = Tokenizer(path)
    for k, v in ENGINE_ENV_DEFAULTS.items():
        os.environ.setdefault(k, v)
    vocab = ROOT / "bench/draft-vocab/vocab-128k.txt"
    if "Q27_DRAFT_VOCAB" not in os.environ and vocab.exists():
        os.environ["Q27_DRAFT_VOCAB"] = str(vocab)
    base = {"prefill_chunk": args.prefill_chunk, "min_seq_tps": args.min_seq_tps, "admit_kv": bool(args.admit_kv),
            "kv_reserve": args.kv_reserve, "decode_share": args.decode_share, "quantum_ms": args.quantum_ms,
            "batch_decode": bool(args.batch_decode), "im_start": tok.token_id("<|im_start|>"),
            "engine_python": str(ROOT / "engine/python")}
    # Images by default with the vision tower in pinned host memory (no KV pool cost); the library default is off.
    budget = env("VRAM_BUDGET_GIB")
    launch = Launch(base, {"weights": args.weights, "kv_dtype": args.kv_dtype, "spec": args.spec,
                           "vision": env("VISION") or "host", "max_seqs": max(1, min(args.max_seqs, 8)),
                           "max_tokens": args.max_tokens, "vram_budget_gib": float(budget) if budget else None},
                    tokenizer=fingerprint(path))
    HUB.start()
    engine = EngineClient(launch.config(launch.settings), autostart=env("ENGINE_AUTOSTART", "1") != "0")
    Handler.srv = Server(engine, tok, args.max_tokens, launch)
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    server.daemon_threads = True

    def stop(*_):
        log("stopping")
        engine.stopping = True
        threading.Thread(target=server.shutdown, daemon=True).start()

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    log(f"serving {MODEL_NAME} on {args.host}:{args.port} (the engine is loading)")
    server.serve_forever()
    engine.stop()


if __name__ == "__main__":
    main()
