#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = ["httpx==0.28.1"]
# ///
"""Tool-calling accuracy against an OpenAI-compatible server (quality gate for lossy options, docs/decisions.md "Quality gates").

BFCL v3 (Berkeley Function-Calling Leaderboard, Apache-2.0) non-live AST categories: simple 400, multiple 200,
parallel 200, parallel_multiple 200. Each question goes out with its functions as OpenAI tools (thinking off,
greedy); the returned tool_calls are checked against the possible answers like BFCL's AST checker, simplified:
the same functions and number of calls (any order for parallel), every required argument present, no unknown
argument, each value among the accepted ones (strings compared lowercase without spaces and ,./-_*^ ; an int
is accepted for a float; lists element by element; dicts key by key). Data is downloaded once into
bench/quality-data/bfcl/ at a pinned revision.

Long context (--context TOKENS, P7-9): each question comes after the history of a coding agent that read the files
of this repository at a commit (--history-rev, default HEAD; read_file calls and their results, real code and
docs) until the prompt is about TOKENS long; the question is the next user turn. The result records the commit:
pass it again to compare runs. Questions are split into groups of --group whose functions are offered together
(plus read_file), so a group shares its prefix and the server prefills it once.

    uv run scripts/quality_tools.py gen --label custom-nvidia
    uv run scripts/quality_tools.py gen --label long32k --categories parallel parallel_multiple --context 32768
    uv run scripts/quality_tools.py compare <a.json> <b.json>     # paired: items only one of them got right
"""

import argparse
import asyncio
import itertools
import json
import re
import subprocess
import time
from datetime import datetime
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent.parent
REV = "61fc0608cfd831fcfbbaa676ebdfef0ed963eeda"
URL = f"https://huggingface.co/datasets/gorilla-llm/Berkeley-Function-Calling-Leaderboard/resolve/{REV}"
DATA = ROOT / "bench/quality-data/bfcl"
CATEGORIES = ("simple", "multiple", "parallel", "parallel_multiple")
HISTORY_REV = "HEAD"  # default commit whose files make up the long-context history (--history-rev)
HISTORY_FILE_CHARS = 12000  # a read_file result is cut here, as agents read long files in parts
READ_FILE = {"type": "function", "function": {
    "name": "read_file", "description": "Read a text file of the repository.",
    "parameters": {"type": "object", "properties": {"path": {"type": "string", "description": "Path from the repository root."}},
                   "required": ["path"]}}}


def load(cat):
    DATA.mkdir(parents=True, exist_ok=True)
    for name, remote in ((f"{cat}.json", f"BFCL_v3_{cat}.json"), (f"{cat}-answer.json", f"possible_answer/BFCL_v3_{cat}.json")):
        if not (DATA / name).exists():
            (DATA / name).write_bytes(httpx.get(f"{URL}/{remote}", follow_redirects=True, timeout=60).raise_for_status().content)
    rows = [json.loads(line) for line in (DATA / f"{cat}.json").read_text().splitlines() if line.strip()]
    answers = {a["id"]: a["ground_truth"] for a in
               (json.loads(line) for line in (DATA / f"{cat}-answer.json").read_text().splitlines() if line.strip())}
    return [(r, answers[r["id"]]) for r in rows]


# BFCL's schema types -> JSON schema (and "." in function names, which OpenAI tool names do not allow).
TYPES = {"dict": "object", "float": "number", "tuple": "array", "any": None}


def schema(s):
    if isinstance(s, dict):
        out = {}
        for k, v in s.items():
            if k == "type" and isinstance(v, str):  # (a property may itself be named "type")
                t = TYPES.get(v, v)
                if t is not None:
                    out[k] = t
            else:
                out[k] = schema(v)
        return out
    if isinstance(s, list):
        return [schema(x) for x in s]
    return s


def tool_name(name):
    return name.replace(".", "_")


def norm_str(s):
    return re.sub(r"[ ,./\-_*^]", "", s.lower())


def value_ok(v, accepted):
    for a in accepted:
        if a == "":
            continue
        if isinstance(a, bool) or isinstance(v, bool):
            if a is v:
                return True
        elif isinstance(a, float) and isinstance(v, (int, float)):
            if float(v) == a:
                return True
        elif isinstance(a, int) and isinstance(v, int):
            if v == a:
                return True
        elif isinstance(a, str) and isinstance(v, str):
            if norm_str(v) == norm_str(a):
                return True
        elif isinstance(a, list) and isinstance(v, list):
            if len(a) == len(v) and all(value_ok(x, [y]) for x, y in zip(v, a)):
                return True
        elif isinstance(a, dict) and isinstance(v, dict):
            if set(v) <= set(a) and all(k in v or "" in a[k] for k in a) and all(value_ok(v[k], a[k]) for k in v):
                return True
    return False


def call_ok(call, truth):
    """call: (name, args); truth: {name: {param: [accepted values]}}."""
    (tname, params), = truth.items()
    name, args = call
    if name != tool_name(tname) or not isinstance(args, dict):
        return False
    if any(k not in params for k in args):
        return False
    for k, acc in params.items():
        if k not in args:
            if "" not in acc:
                return False
        elif not value_ok(args[k], acc):
            return False
    return True


def check(calls, truth):
    if len(calls) != len(truth):
        return False
    if len(calls) > 6:  # (permutations; BFCL's parallel cases have at most a handful of calls)
        return all(call_ok(c, t) for c, t in zip(calls, truth))
    return any(all(call_ok(c, t) for c, t in zip(perm, truth)) for perm in itertools.permutations(calls))


def is_doc(f):
    """Markdown at the top (but not agent instructions) and under docs/."""
    return f.endswith(".md") and ("/" not in f and f not in ("AGENTS.md", "CLAUDE.md") or f.startswith("docs/"))


def history_files(rev):
    """Files of the history in a fixed order: engine sources, scripts and the docs, interleaved."""
    files = subprocess.run(["git", "ls-tree", "-r", "--name-only", rev], cwd=ROOT, capture_output=True,
                           text=True, check=True).stdout.split()
    groups = [[f for f in files if f.startswith("engine/src/")], [f for f in files if f.startswith("scripts/") and f.endswith(".py")],
              [f for f in files if is_doc(f)]]
    out = []
    while any(groups):
        for g in groups:
            if g:
                out.append(g.pop(0))
    return out


async def build_history(client, tokens, rev):
    """System + a coding agent's read_file turns over the repository, about `tokens` long (without the tools)."""
    msgs = [{"role": "system", "content": "You are a helpful assistant with access to tools. You are working in a "
                                          "software repository with the user."},
            {"role": "user", "content": "このリポジトリの主要なファイルを読んで、構成を把握してください。"}]
    for k, path in enumerate(history_files(rev)):
        text = subprocess.run(["git", "show", f"{rev}:./{path}"], cwd=ROOT, capture_output=True, text=True,
                              check=True).stdout
        if len(text) > HISTORY_FILE_CHARS:
            text = text[:HISTORY_FILE_CHARS] + "\n... (truncated)"
        msgs += [{"role": "assistant", "content": "", "tool_calls": [{
                     "id": f"call_{k}", "type": "function",
                     "function": {"name": "read_file", "arguments": json.dumps({"path": path})}}]},
                 {"role": "tool", "tool_call_id": f"call_{k}", "content": text}]
        r = await client.post("/tokenize", json={"messages": msgs, "chat_template_kwargs": {"enable_thinking": False}})
        if r.json()["count"] >= tokens:
            break
    else:
        raise SystemExit(f"history: the files at {rev} do not reach {tokens} tokens")
    msgs.append({"role": "assistant", "content": "主要なファイルを読みました。次の指示をどうぞ。"})
    return msgs


def make_groups(items, size):
    """Split items into groups of `size` whose function names do not clash (first fit, in order)."""
    groups = []
    for i, (cat, row, truth) in enumerate(items):
        names = {tool_name(f["name"]) for f in row["function"]}
        for g in groups:
            if len(g["items"]) < size and not names & g["names"]:
                break
        else:
            g = {"items": [], "names": set()}
            groups.append(g)
        g["items"].append(i)
        g["names"] |= names
    return [g["items"] for g in groups]


async def gen(args):
    items = [(cat, r, a) for cat in args.categories for r, a in load(cat)]
    sem = asyncio.Semaphore(args.concurrency)
    results = [None] * len(items)
    prompt_tokens = []
    rev = subprocess.run(["git", "rev-parse", args.history_rev], cwd=ROOT, capture_output=True, text=True,
                         check=True).stdout.strip() if args.context else None
    async with httpx.AsyncClient(base_url=args.base_url, timeout=1800) as client:
        history = await build_history(client, args.context, rev) if args.context else []
        groups = make_groups(items, args.group) if args.context else [list(range(len(items)))]

        def fn_tools(row):
            return [{"type": "function", "function": {"name": tool_name(f["name"]), "description": f["description"],
                                                      "parameters": schema(f["parameters"])}} for f in row["function"]]

        async def one(i, tools):
            cat, row, truth = items[i]
            async with sem:
                r = await client.post("/v1/chat/completions", json={
                    "model": args.model, "temperature": args.temperature, "max_tokens": args.max_tokens, "tools": tools,
                    **({"top_p": 0.8, "top_k": 20, "seed": args.seed * 100000 + i} if args.temperature > 0 else {}),
                    "messages": history + row["question"][0], "chat_template_kwargs": {"enable_thinking": False},
                })
                body = r.json()
                prompt_tokens.append(body.get("usage", {}).get("prompt_tokens", 0))
                msg = body["choices"][0]["message"]
                calls, bad = [], False
                for tc in msg.get("tool_calls") or []:
                    try:
                        a = tc["function"]["arguments"]
                        calls.append((tc["function"]["name"], json.loads(a) if isinstance(a, str) else a))
                    except (KeyError, json.JSONDecodeError):
                        bad = True
                results[i] = {"id": row["id"], "category": cat, "ok": not bad and check(calls, truth),
                              "calls": calls, "content": msg.get("content")}

        t0 = time.time()
        for g in groups:
            if args.context:  # one tool list per group (read_file first), so the group shares its prefix
                tools = [READ_FILE] + [t for i in g for t in fn_tools(items[i][1])]
                await asyncio.gather(*(one(i, tools) for i in g))
            else:
                await asyncio.gather(*(one(i, fn_tools(items[i][1])) for i in g))
    acc = {c: sum(r["ok"] for r in results if r["category"] == c) / sum(r["category"] == c for r in results)
           for c in args.categories}
    total = sum(r["ok"] for r in results) / len(results)
    print(f"{args.label}: " + ", ".join(f"{c} {v * 100:.1f}%" for c, v in acc.items()) +
          f", all {total * 100:.1f}% ({sum(r['ok'] for r in results)}/{len(results)}) in {time.time() - t0:.0f} s"
          f", prompt {min(prompt_tokens)}-{max(prompt_tokens)} tokens")
    out = ROOT / "bench/results" / f"{datetime.now():%Y%m%d-%H%M%S}-tools-{args.label}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"label": args.label, "rev": REV, "context": args.context, "group": args.group,
                              "history_rev": rev if args.context else None, "prompt_tokens": prompt_tokens, "accuracy": acc, "all": total, "results": results},
                              indent=1, ensure_ascii=False))
    print(out)


def compare(args):
    a, b = (json.loads(Path(p).read_text()) for p in (args.a, args.b))
    rb = {r["id"]: r for r in b["results"]}
    for cat in CATEGORIES:
        pairs = [(r["ok"], rb[r["id"]]["ok"]) for r in a["results"] if r["category"] == cat and r["id"] in rb]
        if pairs:
            print(f"{cat:18s} n {len(pairs):4d}  {a['label']} {sum(x for x, _ in pairs) / len(pairs) * 100:5.1f}%  "
                  f"{b['label']} {sum(y for _, y in pairs) / len(pairs) * 100:5.1f}%  only a {sum(x and not y for x, y in pairs)}"
                  f"  only b {sum(y and not x for x, y in pairs)}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("gen")
    g.add_argument("--label", required=True)
    g.add_argument("--base-url", default="http://127.0.0.1:8000")
    g.add_argument("--model", default="qwen3.8-27b")
    g.add_argument("--categories", nargs="*", default=list(CATEGORIES), choices=CATEGORIES)
    g.add_argument("--concurrency", type=int, default=4)
    g.add_argument("--max-tokens", type=int, default=1024)
    g.add_argument("--temperature", type=float, default=0.0,
                   help="0: greedy (the default gate); > 0: sampled with top_p 0.8 / top_k 20 and per-question seeds")
    g.add_argument("--seed", type=int, default=0, help="sampled runs: seed of the run")
    g.add_argument("--context", type=int, default=0, help="tokens of agent history before each question (0: none)")
    g.add_argument("--history-rev", default=HISTORY_REV, help="with --context: commit whose files make the history")
    g.add_argument("--group", type=int, default=10, help="with --context: questions per shared tool list / prefix")
    c = sub.add_parser("compare")
    c.add_argument("a")
    c.add_argument("b")
    args = ap.parse_args()
    asyncio.run(gen(args)) if args.cmd == "gen" else compare(args)


if __name__ == "__main__":
    main()
