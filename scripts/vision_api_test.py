"""Image requests against the running server (P4-12d): answers, the prefix cache across turns, errors.

1. the chart image with a question (greedy, thinking off) -> the answer names March / 190;
2. a second turn on the same conversation -> the image lies in the cached prefix and is not encoded again
   (the server log says "encoded 0"; here: cached_tokens covers the image);
3. the same question with another image -> no reuse of the first image's tokens beyond the system prefix;
4. two images in one message; 5. a non-data URL -> HTTP 400.

    python3 scripts/vision_api_test.py [--base-url http://127.0.0.1:8000]   # (standard library only)
"""

import argparse
import base64
import json
import pathlib
import random
import sys
import urllib.error
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parents[1]


def data_url(name):
    raw = (ROOT / "bench/vision" / name).read_bytes()
    mime = "image/png" if name.endswith(".png") else "image/jpeg"
    return f"data:{mime};base64,{base64.b64encode(raw).decode()}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:8000")
    args = ap.parse_args()
    off = {"enable_thinking": False}

    class Resp:
        def __init__(self, code, body):
            self.status_code, self.body = code, body

        def json(self):
            return json.loads(self.body)

        def raise_for_status(self):
            if self.status_code != 200:
                raise RuntimeError(f"HTTP {self.status_code}: {self.body[:500]}")

    def chat(messages, max_tokens=64):
        body = json.dumps({"model": "qwen3.8-27b", "messages": messages, "temperature": 0, "max_tokens": max_tokens,
                           "chat_template_kwargs": off}).encode()
        req = urllib.request.Request(args.base_url + "/v1/chat/completions", body, {"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=600) as r:
                return Resp(r.status, r.read())
        except urllib.error.HTTPError as e:
            return Resp(e.code, e.read())

    ok = True

    def check(cond, what):
        nonlocal ok
        ok &= bool(cond)
        print(("PASS " if cond else "FAIL ") + what, flush=True)

    q = "Which month has the highest sales, and what is its value? Answer briefly."
    # (a fresh system text per run, so that earlier runs leave nothing in the prefix cache to match)
    sysm = {"role": "system", "content": f"You are a helpful assistant. Session {random.getrandbits(48):012x}."}
    m1 = [sysm, {"role": "user", "content": [{"type": "image_url", "image_url": {"url": data_url("chart.png")}},
                                       {"type": "text", "text": q}]}]
    r = chat(m1)
    r.raise_for_status()
    a1 = r.json()["choices"][0]["message"]["content"]
    u1 = r.json()["usage"]
    print(f"turn 1: {a1!r} {u1}")
    check(("Mar" in a1) and "190" in a1, "chart answer names March and 190")

    m2 = m1 + [{"role": "assistant", "content": a1},
               {"role": "user", "content": "And which month is the lowest? Answer briefly."}]
    r = chat(m2)
    r.raise_for_status()
    a2, u2 = r.json()["choices"][0]["message"]["content"], r.json()["usage"]
    print(f"turn 2: {a2!r} {u2}")
    check("Feb" in a2, "second turn names February")
    # (checkpoints sit at multiples of 64: the one before the first turn's generation prompt covers the image)
    check(u2["prompt_tokens_details"]["cached_tokens"] >= u1["prompt_tokens"] - 64,
          "second turn resumes from the first turn's prefix (image cached, not encoded again)")

    m3 = [sysm, {"role": "user", "content": [{"type": "image_url", "image_url": {"url": data_url("astronaut.jpg")}},
                                       {"type": "text", "text": q}]}]
    r = chat(m3)
    r.raise_for_status()
    u3 = r.json()["usage"]
    print(f"other image: {r.json()['choices'][0]['message']['content']!r} {u3}")
    # (the same system prompt before the image may be reused; the chart's image tokens must not be)
    check(u3["prompt_tokens_details"]["cached_tokens"] < 64, "another image does not reuse the first one's tokens")

    m4 = [{"role": "user", "content": [
        {"type": "text", "text": "Picture A:"}, {"type": "image_url", "image_url": {"url": data_url("astronaut.jpg")}},
        {"type": "text", "text": "Picture B:"}, {"type": "image_url", "image_url": {"url": data_url("chart.png")}},
        {"type": "text", "text": "In one sentence each, what do pictures A and B show?"}]}]
    r = chat(m4, 96)
    r.raise_for_status()
    a4 = r.json()["choices"][0]["message"]["content"]
    print(f"two images: {a4!r}")
    check("astronaut" in a4.lower() and ("chart" in a4.lower() or "graph" in a4.lower()), "two images both described")

    r = chat([{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "http://example.com/x.png"}},
                                           {"type": "text", "text": "hi"}]}])
    check(r.status_code == 400, f"a non-data URL is rejected with 400 ({r.status_code})")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
