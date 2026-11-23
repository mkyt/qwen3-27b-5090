#!/usr/bin/env python3
"""Plot decode speed of several engines (docs/images/decode-*.svg): one request (code / JSON / prose, the median of
`bench_agentic.py --scenarios decode`) and concurrent reasoning requests (the sum of the streams' rates from
`bench_concurrent.py`, mean of the runs). Writes one SVG for light and one for dark themes.

Each --engine names the result files of each measurement, since some engines serve the scenarios from different
configurations:

    python3 scripts/plot_decode.py \
        --engine 'This engine v0.7.0|decode=docs/results/*-dspine-v070-decode.json|concurrent=docs/results/*-dspine-v070-concurrent.json' \
        --engine 'SGLang v0.5.23|decode=...-fast-decode.json|c1=...-fast-concurrent1.json|c4=...-agent-concurrent14.json|c8=...-parallel-concurrent8.json'

`concurrent=` supplies C=1, 4 and 8 from one file; `c1=` / `c4=` / `c8=` override single levels.
"""
import argparse
import glob
import json
from pathlib import Path
from statistics import mean

PALETTES = {
    "light": dict(bg="#ffffff", fg="#1f1f1e", muted="#6b6a64", grid="#e4e3df", base="#b5b4ad",
                  series=["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]),
    "dark": dict(bg="#1a1a19", fg="#f0efec", muted="#a8a79d", grid="#3a3936", base="#5a5954",
                 series=["#3987e5", "#d95926", "#199e70", "#c98500"]),
}
PROMPTS = [("code", "code"), ("json", "JSON"), ("prose", "prose")]
LEVELS = [1, 4, 8]


def one(pattern):
    files = sorted(glob.glob(pattern))
    if len(files) != 1:
        raise SystemExit(f"{pattern}: expected one file, found {len(files)}")
    return json.load(open(files[0]))["results"]


def load(spec):
    name, *parts = spec.split("|")
    src = dict(p.split("=", 1) for p in parts)
    dec = one(src["decode"])["decode"]["per_prompt_tps"]
    conc = {}
    for c in LEVELS:
        pattern = src.get(f"c{c}", src.get("concurrent"))
        runs = [r["sum_stream_tps"] for r in one(pattern)["reason"] if r["concurrency"] == c]
        if not runs:
            raise SystemExit(f"{name}: no C={c} in {pattern}")
        conc[c] = mean(runs)
    return name, dict(decode=[dec[k]["p50"] for k, _ in PROMPTS], concurrent=[conc[c] for c in LEVELS])


def bar(x, y0, w, h, r=4):
    """A bar anchored to the baseline y0, rounded at the data end only."""
    r = min(r, w / 2, h)
    top = y0 - h
    return (f"M{x:.1f},{y0:.1f} V{top + r:.1f} Q{x:.1f},{top:.1f} {x + r:.1f},{top:.1f} "
            f"H{x + w - r:.1f} Q{x + w:.1f},{top:.1f} {x + w:.1f},{top + r:.1f} V{y0:.1f} Z")


def panel(o, c, engines, x0, x1, T, y0, title, groups, values, ymax, step):
    Y = lambda v: y0 - (y0 - T) * v / ymax
    o.append(f'<text x="{x0}" y="{T - 14}" font-size="13" font-weight="600" fill="{c["fg"]}">{title}</text>')
    v = 0
    while v <= ymax:
        o.append(f'<line x1="{x0}" x2="{x1}" y1="{Y(v):.1f}" y2="{Y(v):.1f}" stroke="{c["grid"]}"/>')
        o.append(f'<text x="{x0 - 8}" y="{Y(v) + 4:.1f}" font-size="11" text-anchor="end" fill="{c["muted"]}">{v:,}</text>')
        v += step
    gw = (x1 - x0) / len(groups)
    n = len(engines)
    bw = min(26.0, (gw - 24) / n - 2)
    for gi, g in enumerate(groups):
        gx = x0 + gi * gw + (gw - n * (bw + 2) + 2) / 2
        for ei in range(n):
            val = values[ei][gi]
            o.append(f'<path d="{bar(gx + ei * (bw + 2), y0, bw, y0 - Y(val))}" fill="{c["series"][ei]}"/>')
        o.append(f'<text x="{x0 + gi * gw + gw / 2:.1f}" y="{y0 + 18}" font-size="12" text-anchor="middle" fill="{c["muted"]}">{g}</text>')
    o.append(f'<line x1="{x0}" x2="{x1}" y1="{y0}" y2="{y0}" stroke="{c["base"]}"/>')


def svg(engines, theme, title, note):
    c = PALETTES[theme]
    W, H = 760, 400
    T, y0 = 108, 350
    o = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" '
         f'font-family="-apple-system,Segoe UI,Helvetica,Arial,sans-serif">',
         f'<rect width="{W}" height="{H}" fill="{c["bg"]}"/>',
         f'<text x="56" y="22" font-size="15" font-weight="600" fill="{c["fg"]}">{title}</text>',
         f'<text x="56" y="40" font-size="12" fill="{c["muted"]}">{note}</text>']
    lx = 56
    for (name, _), col in zip(engines, c["series"]):
        o.append(f'<rect x="{lx}" y="54" width="12" height="12" rx="3" fill="{col}"/>')
        o.append(f'<text x="{lx + 18}" y="64" font-size="12" fill="{c["fg"]}">{name}</text>')
        lx += 18 + 7.2 * len(name) + 22
    panel(o, c, engines, 56, 360, T, y0, "One request (decode tok/s)", [p for _, p in PROMPTS],
          [e["decode"] for _, e in engines], 800, 200)
    panel(o, c, engines, 440, 744, T, y0, "Concurrent requests (total tok/s)", [f"C={n}" for n in LEVELS],
          [e["concurrent"] for _, e in engines], 1600, 400)
    o.append(f'<text x="{W / 2}" y="{H - 12}" font-size="11" text-anchor="middle" fill="{c["muted"]}">'
             f'1,024 output tokens per prompt (median of 3) | reasoning prompt, 4,096 tokens each (mean of 2)</text>')
    o.append("</svg>")
    return "\n".join(o) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", action="append", required=True, help="NAME|decode=GLOB|concurrent=GLOB[|cN=GLOB]")
    ap.add_argument("--title", default="Decode speed (RTX 5090, 400 W)")
    ap.add_argument("--note", default="")
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent.parent / "docs" / "images" / "decode"))
    args = ap.parse_args()
    engines = [load(s) for s in args.engine]
    for name, e in engines:
        print(f'{name}: decode {", ".join(f"{v:.0f}" for v in e["decode"])}; '
              f'C=1/4/8 {", ".join(f"{v:,.0f}" for v in e["concurrent"])}')
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    for theme in PALETTES:
        Path(f"{args.out}-{theme}.svg").write_text(svg(engines, theme, args.title, args.note))
        print(f"wrote {args.out}-{theme}.svg")


if __name__ == "__main__":
    main()
