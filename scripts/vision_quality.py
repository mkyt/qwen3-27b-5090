"""Image question answering with exact answers (P4-12e): the vision modes of this engine vs other servers.

`make` draws a fixed set of synthetic images (seeded; Pillow, so in the dev image) with one question and one
exact answer each: labeled bar values (English and Japanese questions), the tallest unlabeled bar, counting
shapes, reading a code, a table lookup and the color at a grid position, at sizes from 384 px to beyond
2048 x 2048 (the server shrinks those). `run` sends them to an OpenAI-compatible server (greedy, thinking off,
standard library only, so from the host) and scores the replies.

    docker compose --profile dev run --rm dev-cpu python scripts/vision_quality.py make
    python3 scripts/vision_quality.py run --label custom-bf16 --out bench/results/<stamp>-vision-quality-custom-bf16.json
"""

import argparse
import base64
import json
import pathlib
import re
import time
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parents[1]
DATA = ROOT / "cache/vision-quality"
PER_KIND = 10

COLORS = {"red": (220, 30, 30), "blue": (30, 70, 220), "green": (30, 160, 50), "yellow": (240, 210, 20),
          "purple": (130, 40, 170), "orange": (245, 130, 20), "black": (0, 0, 0), "pink": (245, 140, 190),
          "brown": (120, 70, 30)}
CATS = ["apple", "banana", "cherry", "grape", "lemon", "mango", "peach", "plum"]
NAMES = ["Alice", "Bruno", "Chen", "Dana", "Emil", "Fatima", "Goro", "Hana"]
CODE_CHARS = "ACDEFGHJKLMNPQRTUVWXY34679"  # no look-alikes (0/O, 1/I, 2/Z, 5/S, 8/B)
POSITIONS = {(0, 0): "top-left", (0, 1): "top-center", (0, 2): "top-right", (1, 0): "middle-left",
             (1, 1): "center", (1, 2): "middle-right", (2, 0): "bottom-left", (2, 1): "bottom-center",
             (2, 2): "bottom-right"}


def make(args):
    import random

    from PIL import Image, ImageDraw, ImageFont

    rng = random.Random(20261006)
    DATA.mkdir(parents=True, exist_ok=True)
    sizes = [(384, 288), (640, 480), (800, 600), (1024, 768), (1280, 960), (1600, 1200), (2048, 1536),
             (2048, 2048), (2560, 1920), (3000, 3000)]
    font = lambda px: ImageFont.load_default(size=px)  # noqa: E731
    cases = []

    def bars(img, d, cats, vals, labels):
        W, H = img.size
        s = W / 800
        x0, y0, y1 = int(80 * s), int(60 * s), H - int(70 * s)
        bw = (W - x0 - int(40 * s)) / len(cats)
        top = max(vals) * 1.15
        d.line([(x0, y0), (x0, y1), (W - int(30 * s), y1)], fill=(0, 0, 0), width=max(1, int(2 * s)))
        for i, (c, v) in enumerate(zip(cats, vals)):
            bx = x0 + i * bw + bw * 0.15
            by = y1 - (y1 - y0) * v / top
            d.rectangle([bx, by, bx + bw * 0.7, y1], fill=(70, 120, 200))
            f = font(max(10, int(18 * s)))
            d.text((bx + bw * 0.35, y1 + 8 * s), c, fill=(0, 0, 0), font=f, anchor="mt")
            if labels:
                d.text((bx + bw * 0.35, by - 6 * s), str(v), fill=(0, 0, 0), font=f, anchor="mb")

    for kind in ("bar_value", "bar_value_ja", "bar_max", "count", "code", "table", "color"):
        for k in range(PER_KIND):
            W, H = sizes[k]
            img = Image.new("RGB", (W, H), (255, 255, 255))
            d = ImageDraw.Draw(img)
            s = W / 800
            if kind.startswith("bar"):
                cats = rng.sample(CATS, rng.randint(4, 7))
                if kind == "bar_max":
                    vals = rng.sample(range(10, 100, 7), len(cats))
                else:
                    vals = [rng.randint(5, 250) for _ in cats]
                bars(img, d, cats, vals, labels=kind != "bar_max")
                if kind == "bar_max":
                    q, a = "Which fruit has the tallest bar? Answer with the fruit name only.", cats[vals.index(max(vals))]
                else:
                    i = rng.randrange(len(cats))
                    q = (f"What is the value of {cats[i]}? Answer with the number only." if kind == "bar_value" else
                         f"このグラフで {cats[i]} の値はいくつですか。数字だけで答えてください。")
                    a = str(vals[i])
            elif kind == "count":
                n = rng.randint(2, 9)
                shapes = ["circle"] * n + rng.choices(["square", "triangle", "bluecircle"], k=rng.randint(3, 8))
                rng.shuffle(shapes)
                r = int(min(W, H) / 14)
                placed = []
                for sh in shapes:
                    for _ in range(1000):
                        cx, cy = rng.randint(r + 4, W - r - 4), rng.randint(r + 4, H - r - 4)
                        if all((cx - px) ** 2 + (cy - py) ** 2 > (2.4 * r) ** 2 for px, py in placed):
                            break
                    placed.append((cx, cy))
                    box = [cx - r, cy - r, cx + r, cy + r]
                    if sh == "circle":
                        d.ellipse(box, fill=COLORS["red"])
                    elif sh == "bluecircle":
                        d.ellipse(box, fill=COLORS["blue"])
                    elif sh == "square":
                        d.rectangle(box, fill=COLORS["green"])
                    else:
                        d.polygon([(cx, cy - r), (cx - r, cy + r), (cx + r, cy + r)], fill=COLORS["orange"])
                q, a = "How many red circles are in the image? Answer with the number only.", str(n)
            elif kind == "code":
                for _ in range(int(60 * s * s)):  # light background noise
                    x, y = rng.randrange(W), rng.randrange(H)
                    d.line([(x, y), (x + rng.randint(-40, 40) * s, y + rng.randint(-40, 40) * s)],
                           fill=tuple(rng.randint(170, 235) for _ in range(3)), width=max(1, int(s)))
                a = "-".join("".join(rng.choice(CODE_CHARS) for _ in range(m)) for m in (3, 4, 2))
                d.text((W / 2 + rng.randint(-50, 50) * s, H / 2 + rng.randint(-60, 60) * s), a, fill=(20, 20, 20),
                       font=font(int(rng.randint(36, 60) * s)), anchor="mm")
                q = "What is the code written in the image? Answer with the code only."
            elif kind == "table":
                rows, cols = rng.sample(NAMES, 4), ["Q1", "Q2", "Q3", "Q4"]
                vals = [[rng.randint(10, 99) for _ in cols] for _ in rows]
                cw, ch = W / 5.5, H / 6
                f = font(max(10, int(min(cw, ch) / 3.2)))
                for r_ in range(5):
                    for c_ in range(5):
                        x, y = cw * 0.25 + c_ * cw, ch * 0.5 + r_ * ch
                        d.rectangle([x, y, x + cw, y + ch], outline=(0, 0, 0), width=max(1, int(s)))
                        t = ("" if c_ == 0 else cols[c_ - 1]) if r_ == 0 else (rows[r_ - 1] if c_ == 0 else str(vals[r_ - 1][c_ - 1]))
                        d.text((x + cw / 2, y + ch / 2), t, fill=(0, 0, 0), font=f, anchor="mm")
                i, j = rng.randrange(4), rng.randrange(4)
                q, a = f"In the table, what is the value for {rows[i]} in {cols[j]}? Answer with the number only.", str(vals[i][j])
            else:  # color
                names = rng.sample(list(COLORS), 9)
                cw, ch = W / 3, H / 3
                for (r_, c_), _ in POSITIONS.items():
                    d.rectangle([c_ * cw + cw * 0.1, r_ * ch + ch * 0.1, c_ * cw + cw * 0.9, r_ * ch + ch * 0.9],
                                fill=COLORS[names[r_ * 3 + c_]])
                pos = rng.choice(list(POSITIONS))
                q = f"What color is the square at the {POSITIONS[pos]} of the 3x3 grid? Answer with one color word."
                a = names[pos[0] * 3 + pos[1]]
            name = f"{kind}-{k:02d}"
            img.save(DATA / f"{name}.png")
            cases.append({"name": name, "kind": kind, "size": [W, H], "question": q, "answer": a})
    (DATA / "cases.jsonl").write_text("".join(json.dumps(c, ensure_ascii=False) + "\n" for c in cases))
    print(f"{len(cases)} cases in {DATA}")


def correct(kind, answer, reply):
    reply = reply.strip()
    if answer.isdigit():
        m = re.search(r"\d+", reply.replace(",", ""))
        return bool(m) and m.group(0) == answer
    if kind == "code":
        return answer in re.sub(r"\s", "", reply).upper()
    return re.search(rf"\b{re.escape(answer.lower())}\b", reply.lower()) is not None


def run(args):
    cases = [json.loads(line) for line in (DATA / "cases.jsonl").read_text().splitlines()]
    url = args.base_url.rstrip("/")
    model = json.loads(urllib.request.urlopen(url + "/v1/models", timeout=30).read())["data"][0]["id"]
    rows, by_kind = [], {}
    for c in cases:
        raw = (DATA / f"{c['name']}.png").read_bytes()
        msgs = [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": "data:image/png;base64," + base64.b64encode(raw).decode()}},
            {"type": "text", "text": c["question"]}]}]
        body = json.dumps({"model": model, "messages": msgs, "temperature": 0, "max_tokens": args.max_tokens,
                           "chat_template_kwargs": {"enable_thinking": False}}).encode()
        req = urllib.request.Request(url + "/v1/chat/completions", body, {"Content-Type": "application/json"})
        t0 = time.time()
        out = json.loads(urllib.request.urlopen(req, timeout=600).read())
        dt = time.time() - t0
        reply = out["choices"][0]["message"]["content"] or ""
        ok = correct(c["kind"], c["answer"], reply)
        rows.append({**c, "reply": reply, "ok": ok, "seconds": round(dt, 3),
                     "prompt_tokens": out.get("usage", {}).get("prompt_tokens")})
        by_kind.setdefault(c["kind"], []).append(ok)
        print(f"{c['name']:16s} {'ok ' if ok else 'NG '} {dt:6.2f} s  {c['answer']!r:14s} {reply[:60]!r}", flush=True)
    total = sum(r["ok"] for r in rows)
    summary = {k: f"{sum(v)}/{len(v)}" for k, v in by_kind.items()}
    print(f"{args.label}: {total}/{len(rows)}  " + "  ".join(f"{k} {v}" for k, v in summary.items()))
    if args.out:
        pathlib.Path(args.out).write_text(json.dumps({"label": args.label, "model": model, "correct": total,
                                                      "cases": len(rows), "by_kind": summary, "rows": rows},
                                                     ensure_ascii=False, indent=1) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["make", "run"])
    ap.add_argument("--base-url", default="http://127.0.0.1:8000")
    ap.add_argument("--label", default="server")
    ap.add_argument("--out")
    ap.add_argument("--max-tokens", type=int, default=32)
    args = ap.parse_args()
    {"make": make, "run": run}[args.mode](args)


if __name__ == "__main__":
    main()
