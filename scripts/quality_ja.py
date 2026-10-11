#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = ["httpx==0.28.1", "sacrebleu==2.5.1"]
# ///
"""Japanese quality set (docs/decisions.md "Quality gates"): reading comprehension, translation, summarization.

Scored two ways: against the dataset's references (EM / char F1, chrF, ROUGE-L) and by an LLM judge that sees
the source, the reference and the output. Generation and judging are separate steps, because the GPU holds one
model at a time: generate with the configuration under test, then start the judge model and judge the file.

    uv run scripts/quality_ja.py gen --label sglang-agent             # -> bench/results/<ts>-ja-<label>.json
    uv run scripts/quality_ja.py judge bench/results/<file>.json      # judge server on :8000 (default Gemma 4)
    uv run scripts/quality_ja.py judge <file> --control shuffled      # sanity check: outputs of other items
    uv run scripts/quality_ja.py summary <file> [<file> ...]
    uv run scripts/quality_ja.py compare <baseline.json> <candidate.json>   # paired differences, bootstrap CI

Datasets (downloaded once into bench/quality-data/, pinned by sha256; selection is a seeded shuffle):
  jsquad  JGLUE JSQuAD v1.3 valid (CC BY-SA 4.0)
  mt      FLORES-200 devtest jpn_Jpan <-> eng_Latn (CC BY-SA 4.0)
  sum     XL-Sum Japanese test (CC BY-NC-SA 4.0)
"""

import argparse
import asyncio
import hashlib
import io
import json
import random
import re
import tarfile
import time
from datetime import datetime
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "bench/quality-data"

SOURCES = {
    "jsquad": ("https://raw.githubusercontent.com/yahoojapan/JGLUE/main/datasets/jsquad-v1.3/valid-v1.3.json",
               "fb0a57a35281ff03bbe7f0e878cff5b577985fbe6fcb242e7e8ddb63d1c0ab11"),
    "flores": ("https://dl.fbaipublicfiles.com/nllb/flores200_dataset.tar.gz",
               "b8b0b76783024b85797e5cc75064eb83fc5288b41e9654dabc7be6ae944011f6"),
    "xlsum": ("https://huggingface.co/datasets/csebuetnlp/xlsum/resolve/main/data/japanese_XLSum_v2.0.tar.bz2",
              "d277164a97aaa8a4bab0ea219075255db1074b5aaccedb202156db3bd02529ff"),
}

# items per task in the default set
COUNTS = {"jsquad": 150, "mt-ja-en": 100, "mt-en-ja": 100, "sum": 60}
SUM_MAX_CHARS = 4000  # skip very long articles (keeps the set fast and inside short-context behaviour)

PROMPTS = {
    "jsquad": "次の文章を読んで、質問に答えてください。答えは文章中の語句をそのまま抜き出し、答えだけを 1 行で書いてください。\n\n"
              "文章:\n{context}\n\n質問: {question}",
    "mt-ja-en": "次の日本語の文を英語に翻訳してください。訳文だけを出力してください。\n\n{src}",
    "mt-en-ja": "次の英語の文を日本語に翻訳してください。訳文だけを出力してください。\n\n{src}",
    "sum": "次のニュース記事を、日本語で 2〜3 文に要約してください。要約だけを出力してください。\n\n{src}",
}
MAX_TOKENS = {"jsquad": 128, "mt-ja-en": 512, "mt-en-ja": 512, "sum": 512}


# ---------------------------------------------------------------- data


def fetch(name: str) -> bytes:
    url, sha = SOURCES[name]
    path = DATA / url.rsplit("/", 1)[-1]
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(httpx.get(url, follow_redirects=True, timeout=300).raise_for_status().content)
    blob = path.read_bytes()
    if hashlib.sha256(blob).hexdigest() != sha:
        raise SystemExit(f"{path}: sha256 mismatch (dataset changed upstream?)")
    return blob


def tar_member(blob: bytes, name: str, mode: str) -> str:
    with tarfile.open(fileobj=io.BytesIO(blob), mode=mode) as t:
        return t.extractfile(name).read().decode()


def load_items(seed: int, counts: dict) -> list[dict]:
    items = []
    rng = lambda tag: random.Random(f"{seed}-{tag}")  # noqa: E731  independent order per task

    if counts.get("jsquad"):
        d = json.loads(fetch("jsquad"))
        qas = [(p["context"], q) for a in d["data"] for p in a["paragraphs"] for q in p["qas"] if not q["is_impossible"]]
        rng("jsquad").shuffle(qas)
        for ctx, q in qas[:counts["jsquad"]]:
            items.append({"task": "jsquad", "id": q["id"], "context": ctx.replace(" [SEP] ", "\n"),
                          "question": q["question"], "refs": sorted({a["text"] for a in q["answers"]})})

    if counts.get("mt-ja-en") or counts.get("mt-en-ja"):
        blob = fetch("flores")
        ja = tar_member(blob, "./flores200_dataset/devtest/jpn_Jpan.devtest", "r:gz").splitlines()
        en = tar_member(blob, "./flores200_dataset/devtest/eng_Latn.devtest", "r:gz").splitlines()
        for task, src, ref in (("mt-ja-en", ja, en), ("mt-en-ja", en, ja)):
            idx = list(range(len(src)))
            rng(task).shuffle(idx)
            for i in idx[:counts.get(task, 0)]:
                items.append({"task": task, "id": f"flores-devtest-{i}", "src": src[i], "refs": [ref[i]]})

    if counts.get("sum"):
        rows = [json.loads(line) for line in tar_member(fetch("xlsum"), "./japanese_test.jsonl", "r:bz2").splitlines()]
        rows = [r for r in rows if len(r["text"]) <= SUM_MAX_CHARS]
        rng("sum").shuffle(rows)
        for r in rows[:counts["sum"]]:
            items.append({"task": "sum", "id": r["id"], "src": r["text"], "refs": [r["summary"]]})
    return items


# ---------------------------------------------------------------- reference metrics

_PUNCT = re.compile(r"[\s「」『』（）()、。・,.!?！？:：;；\"'`]")


def norm_ja(s: str) -> str:
    return _PUNCT.sub("", s.strip().splitlines()[0] if s.strip() else "")


def char_f1(pred: str, ref: str) -> float:
    from collections import Counter

    p, r = Counter(pred), Counter(ref)
    common = sum((p & r).values())
    if not common:
        return 0.0
    prec, rec = common / len(pred), common / len(ref)
    return 2 * prec * rec / (prec + rec)


def lcs(a: str, b: str) -> int:
    prev = [0] * (len(b) + 1)
    for ca in a:
        cur = [0]
        for j, cb in enumerate(b):
            cur.append(prev[j] + 1 if ca == cb else max(prev[j + 1], cur[j]))
        prev = cur
    return prev[-1]


def rouge_l_char(pred: str, ref: str) -> float:
    pred, ref = re.sub(r"\s", "", pred), re.sub(r"\s", "", ref)
    if not pred or not ref:
        return 0.0
    n = lcs(pred, ref)
    if not n:
        return 0.0
    prec, rec = n / len(pred), n / len(ref)
    return 2 * prec * rec / (prec + rec)


def ref_metrics(item: dict, out: str) -> dict:
    from sacrebleu.metrics import CHRF

    if item["task"] == "jsquad":
        p = norm_ja(out)
        refs = [norm_ja(r) for r in item["refs"]]
        return {"em": float(p in refs), "f1": max(char_f1(p, r) for r in refs) if p else 0.0}
    chrf = CHRF().sentence_score(out.strip(), item["refs"]).score
    if item["task"] == "sum":
        return {"chrf": chrf, "rougeL": 100 * rouge_l_char(out, item["refs"][0])}
    return {"chrf": chrf}


GATE_DROP = 3.0  # judge points (0-100) a candidate may lose on a task before the quality gate fails

PRIMARY = {"jsquad": "f1", "mt-ja-en": "chrf", "mt-en-ja": "chrf", "sum": "rougeL"}


# ---------------------------------------------------------------- generation


async def run_pool(n, concurrency, fn):
    sem = asyncio.Semaphore(concurrency)

    async def one(i):
        async with sem:
            await fn(i)

    await asyncio.gather(*(one(i) for i in range(n)))


async def chat(client, model, prompt, max_tokens, thinking=False, temperature=0.0):
    r = await client.post("/v1/chat/completions", json={
        "model": model, "temperature": temperature, "max_tokens": max_tokens,
        "messages": [{"role": "user", "content": prompt}],
        "chat_template_kwargs": {"enable_thinking": thinking},
    })
    r.raise_for_status()
    ch = r.json()["choices"][0]
    return ch["message"]["content"] or "", ch.get("finish_reason")


def cmd_gen(args):
    counts = {k: (args.n if args.n is not None else v) for k, v in COUNTS.items() if not args.tasks or k in args.tasks}
    items = load_items(args.seed, counts)

    async def main():
        async with httpx.AsyncClient(base_url=args.base_url, timeout=600) as client:
            async def one(i):
                it = items[i]
                prompt = PROMPTS[it["task"]].format(**it)
                it["output"], it["finish"] = await chat(client, args.model, prompt, MAX_TOKENS[it["task"]],
                                                       temperature=args.temperature)
                it["metrics"] = ref_metrics(it, it["output"])
            await run_pool(len(items), args.concurrency, one)

    t0 = time.time()
    asyncio.run(main())
    out = ROOT / "bench/results" / f"{datetime.now():%Y%m%d-%H%M%S}-ja-{args.label}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"label": args.label, "model": args.model, "seed": args.seed, "temperature": args.temperature,
                               "datasets": {k: v[1] for k, v in SOURCES.items()}, "items": items},
                              ensure_ascii=False, indent=1))
    print(f"{len(items)} items in {time.time() - t0:.0f} s -> {out.relative_to(ROOT)}")
    summarize([out])


# ---------------------------------------------------------------- LLM judge

JUDGE_TAIL = "\n\nまず根拠を 2〜3 文で書き、最後の行に JSON で {\"score\": <整数>} とだけ書いてください。"
JUDGE = {
    "jsquad": ("あなたは読解問題の採点者です。文章・質問・正解例をもとに、解答が正しいかを判定してください。\n"
               "正解例と同じ内容を指していれば、表記の揺れや余分な語があっても正解です。違うものを指している、"
               "または答えになっていない場合は不正解です。score は正解なら 1、不正解なら 0。\n\n"
               "文章:\n{context}\n\n質問: {question}\n正解例: {refs}\n\n解答: {output}"),
    "mt": ("あなたは翻訳の採点者です。原文と参考訳をもとに、訳文の品質を 1〜5 で採点してください。\n"
           "5: 意味が完全に正確で自然。4: 意味は正確だが、わずかに不自然な点がある。3: 小さな誤訳や訳抜けがある。"
           "2: 重要な誤訳や訳抜けがある。1: 意味がほとんど伝わらない、または別の言語や無関係な出力。\n"
           "参考訳と言い回しが違うだけなら減点しないでください。\n\n原文: {src}\n参考訳: {refs}\n\n訳文: {output}"),
    "sum": ("あなたは要約の採点者です。記事と参考要約をもとに、要約の品質を 1〜5 で採点してください。\n"
            "観点: 記事にない内容や誤りがないこと（最重要）、記事の要点を押さえていること、自然な日本語で 2〜3 文に収まっていること。\n"
            "5: 正確で要点を押さえ、簡潔。4: 正確だが要点が少し欠ける、または少し長い。3: 小さな誤りがある、または要点が大きく欠ける。"
            "2: 重要な誤りや記事にない内容がある。1: 要約になっていない。\n"
            "参考要約と取り上げ方が違うだけなら減点しないでください。\n\n記事:\n{src}\n\n参考要約: {refs}\n\n要約: {output}"),
}
JUDGE_MAX = {"jsquad": 1, "mt-ja-en": 5, "mt-en-ja": 5, "sum": 5}


def judge_prompt(it, output):
    tmpl = JUDGE["mt" if it["task"].startswith("mt") else it["task"]]
    return tmpl.format(**{**it, "refs": " / ".join(it["refs"]), "output": output.strip()}) + JUDGE_TAIL


def parse_score(text):
    m = re.findall(r"\{\s*\"score\"\s*:\s*(-?\d+)\s*\}", text)
    return int(m[-1]) if m else None


def cmd_judge(args):
    path = Path(args.result)
    d = json.loads(path.read_text())
    items = d["items"]
    outputs = [it["output"] for it in items]
    if args.control == "shuffled":
        # each item gets the output of another item of the same task: a judge that is paying attention scores these low
        by_task = {}
        for i, it in enumerate(items):
            by_task.setdefault(it["task"], []).append(i)
        for idx in by_task.values():
            rot = idx[1:] + idx[:1]
            for i, j in zip(idx, rot):
                outputs[i] = items[j]["output"]
    key = args.model + ("" if args.control == "none" else f"/{args.control}")

    async def main():
        async with httpx.AsyncClient(base_url=args.base_url, timeout=600) as client:
            async def one(i):
                it = items[i]
                text, fin = await chat(client, args.model, judge_prompt(it, outputs[i]), args.max_tokens, args.thinking)
                it.setdefault("judge", {})[key] = {"score": parse_score(text), "text": text, "finish": fin}
            await run_pool(len(items), args.concurrency, one)

    t0 = time.time()
    asyncio.run(main())
    d.setdefault("judges", [])
    if key not in d["judges"]:
        d["judges"].append(key)
    path.write_text(json.dumps(d, ensure_ascii=False, indent=1))
    bad = sum(it["judge"][key]["score"] is None for it in items)
    print(f"judged {len(items)} items with {key} in {time.time() - t0:.0f} s ({bad} unparsable)")
    summarize([path])


# ---------------------------------------------------------------- reports


def judge_norm(it, key):
    """Judge score scaled to 0-100 (None if missing)."""
    j = it.get("judge", {}).get(key)
    if not j or j["score"] is None:
        return None
    lo = 0 if JUDGE_MAX[it["task"]] == 1 else 1
    return 100 * (j["score"] - lo) / (JUDGE_MAX[it["task"]] - lo)


def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else float("nan")


def summarize(paths):
    for p in paths:
        d = json.loads(Path(p).read_text())
        print(f"\n{d['label']}  ({Path(p).name})")
        judges = d.get("judges", [])
        print(f"  {'task':<9}{'n':>5}{'cut':>5}  {'reference metric':<22}" + "".join(f"{k[:28]:>30}" for k in judges))
        for task in COUNTS:
            its = [it for it in d["items"] if it["task"] == task]
            if not its:
                continue
            m = PRIMARY[task]
            ref = f"{m} {mean(it['metrics'][m] for it in its):6.1f}"
            if task == "jsquad":
                ref = f"em {100 * mean(it['metrics']['em'] for it in its):5.1f} f1 {100 * mean(it['metrics']['f1'] for it in its):5.1f}"
            cut = sum(it.get("finish") == "length" for it in its)
            js = "".join(f"{mean(judge_norm(it, k) for it in its):30.1f}" for k in judges)
            print(f"  {task:<9}{len(its):>5}{cut:>5}  {ref:<22}{js}")
        if judges:
            print("  (judge columns: mean score scaled to 0-100)")


def cmd_compare(args):
    A, B = (json.loads(Path(p).read_text()) for p in (args.a, args.b))
    bi = {(it["task"], it["id"]): it for it in B["items"]}
    judges = [k for k in A.get("judges", []) if k in B.get("judges", []) and "/" not in k]
    rng = random.Random(0)
    print(f"{B['label']} - {A['label']} (paired; 95% bootstrap CI)")
    worse = []
    for task in COUNTS:
        pairs = [(a, bi[(task, a["id"])]) for a in A["items"] if a["task"] == task and (task, a["id"]) in bi]
        if not pairs:
            continue
        rows = [(f"ref {PRIMARY[task]}", lambda it: it["metrics"][PRIMARY[task]] * (100 if task == "jsquad" else 1))]
        rows += [(f"judge {k}", lambda it, k=k: judge_norm(it, k)) for k in judges]
        for name, f in rows:
            diffs = [f(b) - f(a) for a, b in pairs if f(a) is not None and f(b) is not None]
            if not diffs:
                continue
            boots = sorted(mean(rng.choices(diffs, k=len(diffs))) for _ in range(2000))
            # Gate (docs/decisions.md "Quality gates"): a judge drop of 3+ points (0-100) whose CI excludes 0. Equivalent configurations
            # differed by <= 2.3; greedy -> temperature 1.0 dropped 4-9 points on translation / summaries.
            flag = name.startswith("judge") and mean(diffs) <= -GATE_DROP and boots[1949] < 0
            if flag:
                worse.append(f"{task} ({name})")
            print(f"  {task:<9}{name[:34]:<36}n={len(diffs):<4} diff {mean(diffs):+6.2f}  "
                  f"[{boots[50]:+6.2f}, {boots[1949]:+6.2f}]{'  WORSE' if flag else ''}")
    print(f"gate: {'FAIL: ' + ', '.join(worse) if worse else 'pass'} (judge drop >= {GATE_DROP} points with CI < 0)")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("gen")
    g.add_argument("--label", required=True)
    g.add_argument("--base-url", default="http://127.0.0.1:8000")
    g.add_argument("--model", default="qwen3.8-27b")
    g.add_argument("--seed", type=int, default=0)
    g.add_argument("--tasks", nargs="*", choices=list(COUNTS), help="default: all")
    g.add_argument("--n", type=int, help="items per task (probe); default: the fixed set sizes")
    g.add_argument("--concurrency", type=int, default=4)
    g.add_argument("--temperature", type=float, default=0.0, help="0 = greedy (the gate); >0 only for controls")
    j = sub.add_parser("judge")
    j.add_argument("result")
    j.add_argument("--base-url", default="http://127.0.0.1:8000")
    j.add_argument("--model", default="gemma-4-26b-a4b")
    j.add_argument("--control", choices=["none", "shuffled"], default="none")
    j.add_argument("--thinking", action="store_true")
    j.add_argument("--max-tokens", type=int, default=1024)
    j.add_argument("--concurrency", type=int, default=8)
    s = sub.add_parser("summary")
    s.add_argument("results", nargs="+")
    c = sub.add_parser("compare")
    c.add_argument("a")
    c.add_argument("b")
    args = ap.parse_args()
    {"gen": cmd_gen, "judge": cmd_judge, "summary": lambda a: summarize(a.results), "compare": cmd_compare}[args.cmd](args)


if __name__ == "__main__":
    main()
