#!/usr/bin/env python3
"""Plot prefill speed against the depth of the prompt for several engines (docs/images/prefill-depth-*.svg).

Reads the longctx results of `bench_agentic.py --scenarios longctx --longctx-tokens 32000,120000,250000` (one file per
seed and engine), takes the speed between consecutive prompt lengths, (difference in tokens) / (difference in time to
the first token), and fits the time per token as a + b * position. Writes one SVG for light and one for dark themes.

    python3 scripts/plot_prefill_depth.py \
        --engine 'This engine v0.7.0=bench/results/20261123-*-dspine-v070-prefill-s*.json' \
        --engine 'NInfer ebdc67e=bench/results/20261123-*-ninfer-ebdc67e-c1-prefill-s*.json' ...
"""
import argparse
import glob
import json
from pathlib import Path
from statistics import mean

# Categorical slots 1-4, light and dark steps (validated for adjacent-pair CVD separation).
PALETTES = {
    "light": dict(bg="#ffffff", fg="#1f1f1e", muted="#6b6a64", grid="#e4e3df",
                  series=["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]),
    "dark": dict(bg="#1a1a19", fg="#f0efec", muted="#a8a79d", grid="#3a3936",
                 series=["#3987e5", "#d95926", "#199e70", "#c98500"]),
}


def load(pattern):
    runs = [json.load(open(f))["results"]["longctx"] for f in sorted(glob.glob(pattern))]
    if not runs:
        raise SystemExit(f"no files match {pattern}")
    pts = [[(x["cold"]["prompt_tokens"], x["cold"]["ttft"]) for x in r] for r in runs]
    speeds, mids = [], []
    for i in range(len(pts[0])):
        s, m = [], []
        for p in pts:
            n0, t0 = (0, 0.0) if i == 0 else p[i - 1]
            n, t = p[i]
            s.append((n - n0) / (t - t0))
            m.append((n + n0) / 2)
        speeds.append(mean(s))
        mids.append(mean(m))
    # Least squares of microseconds per token against the position in thousands of tokens.
    xs, ys = [m / 1000 for m in mids], [1e6 / s for s in speeds]
    mx, my = mean(xs), mean(ys)
    b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sum((x - mx) ** 2 for x in xs)
    return dict(speeds=speeds, mids=mids, a=my - b * mx, b=b, ends=[p[-1][0] for p in pts])


def svg(engines, theme, title):
    c = PALETTES[theme]
    W, H, L, R, T, B = 760, 380, 64, 170, 52, 46
    xmax = 152.0
    ymax = 14000
    X = lambda k: L + (W - L - R) * k / xmax
    Y = lambda v: T + (H - T - B) * (1 - v / ymax)
    o = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" '
         f'font-family="-apple-system,Segoe UI,Helvetica,Arial,sans-serif">',
         f'<rect width="{W}" height="{H}" fill="{c["bg"]}"/>',
         f'<text x="{L}" y="22" font-size="15" font-weight="600" fill="{c["fg"]}">{title}</text>',
         f'<text x="{L}" y="40" font-size="12" fill="{c["muted"]}">Markers: speed between two prompt lengths '
         f'(mean of 2 documents). Lines: time per token = a + b × position.</text>']
    for v in range(0, ymax + 1, 2000):
        o.append(f'<line x1="{L}" x2="{W - R}" y1="{Y(v):.1f}" y2="{Y(v):.1f}" stroke="{c["grid"]}"/>')
        o.append(f'<text x="{L - 8}" y="{Y(v) + 4:.1f}" font-size="11" text-anchor="end" fill="{c["muted"]}">{v:,}</text>')
    for k in range(0, 151, 25):
        o.append(f'<text x="{X(k):.1f}" y="{H - B + 18}" font-size="11" text-anchor="middle" fill="{c["muted"]}">{k}K</text>')
    o.append(f'<text x="{(L + W - R) / 2:.1f}" y="{H - 8}" font-size="12" text-anchor="middle" fill="{c["muted"]}">'
             f'Position in the prompt (tokens)</text>')
    o.append(f'<text x="14" y="{(T + H - B) / 2:.1f}" font-size="12" text-anchor="middle" fill="{c["muted"]}" '
             f'transform="rotate(-90 14 {(T + H - B) / 2:.1f})">Prefill tok/s</text>')
    labels = []
    for (name, e), col in zip(engines, c["series"]):
        d = " ".join(f'{"M" if k == 1 else "L"}{X(k):.1f},{Y(1e6 / (e["a"] + e["b"] * k)):.1f}' for k in range(1, int(xmax) + 1))
        o.append(f'<path d="{d}" fill="none" stroke="{col}" stroke-width="2" stroke-linecap="round"/>')
        for m, s in zip(e["mids"], e["speeds"]):
            o.append(f'<circle cx="{X(m / 1000):.1f}" cy="{Y(s):.1f}" r="4.5" fill="{col}" stroke="{c["bg"]}" stroke-width="2"/>')
        labels.append([Y(1e6 / (e["a"] + e["b"] * xmax)), name, col])
    labels.sort()
    for i in range(1, len(labels)):
        labels[i][0] = max(labels[i][0], labels[i - 1][0] + 15)
    for y, name, col in labels:
        o.append(f'<line x1="{W - R + 6}" x2="{W - R + 18}" y1="{y:.1f}" y2="{y:.1f}" stroke="{col}" stroke-width="2"/>')
        o.append(f'<text x="{W - R + 22}" y="{y + 4:.1f}" font-size="12" fill="{c["fg"]}">{name}</text>')
    o.append("</svg>")
    return "\n".join(o) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", action="append", required=True, help="NAME=GLOB of the longctx result files")
    ap.add_argument("--title", default="Prefill speed at each depth of the prompt (RTX 5090, 400 W)")
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent.parent / "docs" / "images" / "prefill-depth"))
    args = ap.parse_args()
    engines = []
    for spec in args.engine:
        name, pattern = spec.split("=", 1)
        engines.append((name, load(pattern)))
    for name, e in engines:
        print(f'{name}: {", ".join(f"{s:,.0f}" for s in e["speeds"])} tok/s; a {e["a"]:.1f} us, b {e["b"]:.3f} us per 1K')
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    for theme in PALETTES:
        Path(f"{args.out}-{theme}.svg").write_text(svg(engines, theme, args.title))
        print(f"wrote {args.out}-{theme}.svg")


if __name__ == "__main__":
    main()
