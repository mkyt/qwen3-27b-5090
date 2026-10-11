"""Draft vocabulary (docs/architecture.md §7): the token ids the DFlash2 selector may propose, counted from the
model's own outputs in English, Japanese and code (plus the prompts it reads), so that the candidate
top-16 runs a fraction of the 248K-row lm_head. Tokens outside the vocabulary are never drafted (the
verify stays lossless; such a position just rejects).

    # 1. generate and count (GPU, engine with speculative decoding; about 10 minutes)
    docker compose --profile dev run --rm dev python tools/draft_vocab.py collect --out bench/draft-vocab/counts.json
    # 2. count a text corpus (CPU): Japanese / English datasets in bench/quality-data and source code
    docker compose --profile dev run --rm dev-cpu python tools/draft_vocab.py corpus --out bench/draft-vocab/corpus.json
    # 3. build a vocabulary of K ids (multiple of 128) and report the coverage on held-out text (this
    #    repository's documents, which the corpus leaves out)
    docker compose --profile dev run --rm dev-cpu python tools/draft_vocab.py build bench/draft-vocab/counts.json \\
        --corpus bench/draft-vocab/corpus.json --k 65536 --out bench/draft-vocab/vocab-64k.txt
"""

import argparse
import collections
import io
import json
import pathlib
import random
import sys
import tarfile
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "engine/python"))
sys.path.insert(0, str(ROOT / "reference"))
DATA = ROOT / "bench/quality-data"

CODE = [
    "Write a Python function that parses an ISO 8601 duration string and returns a timedelta. Include tests.",
    "Implement a thread-safe bounded queue in Rust with blocking push and pop.",
    "Write a TypeScript React component for a paginated table with sorting and filtering.",
    "Implement Dijkstra's algorithm in C++ using a priority queue, with a small example.",
    "Write a Go HTTP server with JSON endpoints for a todo list, stored in memory with a mutex.",
    "Write a bash script that backs up a directory to a timestamped tar.gz and keeps the last 7 backups.",
    "Implement an LRU cache in Java with O(1) get and put.",
    "Write SQL to find the top 3 customers by revenue per month from orders and order_items tables.",
    "Refactor this Python code to use dataclasses and type hints:\nclass P:\n  def __init__(self,x,y):\n    self.x=x\n    self.y=y\n  def d(self,o):\n    return ((self.x-o.x)**2+(self.y-o.y)**2)**.5",
    "Write a CUDA kernel that computes a row-wise softmax for a [N][1024] float matrix, with warp shuffles.",
    "Write a Python asyncio crawler that fetches URLs concurrently with a semaphore and retries.",
    "Write a Dockerfile and docker-compose.yaml for a Flask app with PostgreSQL.",
    "Implement a trie in Python with insert, search and prefix listing.",
    "Write unit tests with pytest for a function that merges overlapping intervals.",
    "Explain and fix the bug: for i in range(len(a)): if a[i] == x: a.remove(x)",
    "Write a C function that reverses a singly linked list in place.",
    "Write a JSON schema for a user profile with nested address and optional phone numbers.",
    "Implement binary search on a rotated sorted array in JavaScript.",
    "Write a Makefile for a C project with src/ and include/ directories and a test target.",
    "Write a Python script using argparse that renames files by a regex pattern, with a dry-run flag.",
    "PythonでCSVファイルを読み込み、列ごとの平均と標準偏差を計算する関数を書いてください。",
    "Rustで簡単なコマンドライン電卓を実装してください。",
    "JavaScriptでデバウンス関数を実装し、使い方の例も示してください。",
    "Pythonのジェネレータとイテレータの違いをコード例付きで説明してください。",
    "SQLでユーザーごとの直近30日のログイン回数を集計するクエリを書いてください。",
]
JSON_PROMPTS = [
    "Generate a JSON array of 10 fictional products with id, name, price, tags and stock.",
    "Return a JSON object describing a weekly workout plan with days, exercises, sets and reps.",
    "Output a JSON list of 8 cities with country, population and a short description.",
    "架空の社員5人分のデータをJSON配列で出力してください。項目はid、名前、部署、入社日、スキル。",
    "Generate an OpenAPI 3 YAML spec for a simple bookstore API with books and authors.",
]
PROSE = [
    "Explain how public-key cryptography works, with an example.",
    "Describe the causes and consequences of the 2008 financial crisis.",
    "Write a short story about a lighthouse keeper who finds a message in a bottle.",
    "Explain the difference between TCP and UDP and when to use each.",
    "Summarize the history of the Roman Empire in a few paragraphs.",
    "How does a transformer language model generate text? Explain for a beginner.",
    "Give practical tips for improving sleep quality.",
    "Explain how garbage collection works in modern programming languages.",
    "量子コンピュータの仕組みを高校生にもわかるように説明してください。",
    "日本の四季と行事について説明してください。",
    "リモートワークのメリットとデメリットを整理してください。",
    "短い物語を書いてください：月に住むうさぎが地球を訪れる話。",
    "機械学習の過学習とは何か、防ぐ方法とあわせて説明してください。",
    "東京から京都への旅行プランを2泊3日で提案してください。",
    "大規模言語モデルの推論を高速化する手法をいくつか説明してください。",
]


def load_tasks(rng):
    tasks = [(p, False) for p in CODE + JSON_PROMPTS + PROSE]
    tasks += [(p, True) for p in CODE[:8] + PROSE[:4] + PROSE[8:12]]  # thinking on
    gsm = [json.loads(line) for line in open(DATA / "gsm8k-test.jsonl")]
    tasks += [(q["question"], False) for q in rng.sample(gsm, 30)]
    tasks += [(q["question"], True) for q in rng.sample(gsm, 10)]
    squad = json.load(open(DATA / "valid-v1.3.json"))
    qas = [(p["context"].replace(" [SEP] ", "\n"), qa["question"]) for a in squad["data"] for p in a["paragraphs"]
           for qa in p["qas"]]
    tasks += [(f"{c}\n\n質問: {q}\n本文に基づいて答えてください。", False) for c, q in rng.sample(qas, 30)]
    with tarfile.open(DATA / "flores200_dataset.tar.gz") as tf:
        def lines(name):
            m = next(m for m in tf.getmembers() if m.name.endswith(name))
            return tf.extractfile(m).read().decode().splitlines()
        jpn, eng = lines("devtest/jpn_Jpan.devtest"), lines("devtest/eng_Latn.devtest")
    idx = rng.sample(range(len(jpn)), 40)
    tasks += [(f"次の英文を日本語に翻訳してください。\n{eng[i]}", False) for i in idx[:20]]
    tasks += [(f"Translate the following Japanese into English.\n{jpn[i]}", False) for i in idx[20:]]
    with tarfile.open(DATA / "japanese_XLSum_v2.0.tar.bz2") as tf:
        m = next(m for m in tf.getmembers() if m.name.endswith("japanese_test.jsonl"))
        arts = [json.loads(line) for line in io.TextIOWrapper(tf.extractfile(m), encoding="utf-8")]
    tasks += [(f"次の記事を3文で要約してください。\n{a['text'][:3000]}", False) for a in rng.sample(arts, 20)]
    return tasks


def collect(args):
    from transformers import AutoTokenizer

    from q27lib import Engine
    from qwen27b.weights import DFLASH2, TARGET, snapshot

    rng = random.Random(0)
    tasks = load_tasks(rng)
    tok = AutoTokenizer.from_pretrained(snapshot(*TARGET))
    eng = Engine(snapshot(*TARGET), max_tokens=8192, draft=snapshot(*DFLASH2))
    gen, prompt = collections.Counter(), collections.Counter()
    eos = {248046, 248044}
    t0 = time.perf_counter()
    for i, (text, thinking) in enumerate(tasks):
        ids = tok.apply_chat_template([{"role": "user", "content": text}], add_generation_prompt=True,
                                      enable_thinking=thinking)
        ids = list(ids["input_ids"]) if not isinstance(ids, list) else ids
        prompt.update(ids)
        eng.reset()
        out = [eng.spec_begin(ids, 1.0, 20, 0.95, seed=i, uniform=rng.random())]
        while len(out) < args.max_new and out[-1] not in eos:
            out += eng.spec_step()
        gen.update(out)
        if (i + 1) % 20 == 0:
            print(f"{i + 1}/{len(tasks)} tasks, {sum(gen.values())} generated tokens, {time.perf_counter() - t0:.0f} s",
                  flush=True)
    eng.close()
    pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    pathlib.Path(args.out).write_text(json.dumps({"tasks": len(tasks), "generated": dict(gen.most_common()),
                                                  "prompt": dict(prompt.most_common())}) + "\n")
    print(f"wrote {args.out}: {sum(gen.values())} generated ({len(gen)} distinct), "
          f"{sum(prompt.values())} prompt tokens ({len(prompt)} distinct)")


CODE_DIRS = [  # (directory, suffixes): source code in the dev image and the extracted SGLang sources
    (ROOT / "cache/sglang-src", (".py",)),
    (pathlib.Path("/opt/venv/lib/python3.12/site-packages/transformers"), (".py",)),
    (pathlib.Path("/opt/venv/lib/python3.12/site-packages/torch"), (".py",)),
    (pathlib.Path("/opt/cutlass/include"), (".h", ".hpp", ".cuh")),
    (ROOT / "engine", (".cu", ".cc", ".h", ".cuh", ".py")),
    (ROOT / "reference", (".py",)),
]


def corpus(args):
    from transformers import AutoTokenizer

    from qwen27b.weights import TARGET, snapshot

    tok = AutoTokenizer.from_pretrained(snapshot(*TARGET))
    counts = collections.Counter()
    sizes = {}

    def add(label, texts, limit_chars):
        n = 0
        batch = []
        for t in texts:
            batch.append(t)
            n += len(t)
            if len(batch) >= 256:
                for ids in tok(batch, add_special_tokens=False).input_ids:
                    counts.update(ids)
                batch = []
            if n >= limit_chars:
                break
        if batch:
            for ids in tok(batch, add_special_tokens=False).input_ids:
                counts.update(ids)
        sizes[label] = n
        print(f"{label}: {n / 1e6:.1f} M chars, {sum(counts.values())} tokens so far", flush=True)

    with tarfile.open(DATA / "japanese_XLSum_v2.0.tar.bz2") as tf:
        arts = []
        for m in tf.getmembers():
            if m.name.endswith(".jsonl"):
                arts += [json.loads(line) for line in io.TextIOWrapper(tf.extractfile(m), encoding="utf-8")]
    add("xlsum-ja", (a["title"] + "\n" + a["summary"] + "\n" + a["text"] for a in arts), 40e6)
    squad = json.load(open(DATA / "valid-v1.3.json"))
    add("jsquad", (p["context"].replace(" [SEP] ", "\n") + "\n" + "\n".join(q["question"] for q in p["qas"])
                   for a in squad["data"] for p in a["paragraphs"]), 10e6)
    with tarfile.open(DATA / "flores200_dataset.tar.gz") as tf:
        def lines(name):
            m = next(m for m in tf.getmembers() if m.name.endswith(name))
            return tf.extractfile(m).read().decode().splitlines()
        add("flores", lines("dev/jpn_Jpan.dev") + lines("devtest/jpn_Jpan.devtest") + lines("dev/eng_Latn.dev") +
            lines("devtest/eng_Latn.devtest"), 10e6)
    gsm = [json.loads(line) for line in open(DATA / "gsm8k-test.jsonl")]
    add("gsm8k", (q["question"] + "\n" + q["answer"] for q in gsm), 10e6)
    for d, suffixes in CODE_DIRS:
        if not d.is_dir():
            print(f"(skip {d})")
            continue
        files = sorted(f for f in d.rglob("*") if f.suffix in suffixes and f.is_file())
        add(f"code:{d.name}", (f.read_text(errors="ignore") for f in files), 30e6)
    pathlib.Path(args.out).write_text(json.dumps({"sizes": sizes, "counts": dict(counts.most_common())}) + "\n")
    print(f"wrote {args.out}: {sum(counts.values())} tokens, {len(counts)} distinct")


def build(args):
    d = json.loads(pathlib.Path(args.counts).read_text())
    gen = collections.Counter({int(k): v for k, v in d["generated"].items()})
    prompt = collections.Counter({int(k): v for k, v in d["prompt"].items()})
    score = collections.Counter()
    for t, c in gen.items():
        score[t] += c / sum(gen.values())
    for t, c in prompt.items():
        score[t] += args.prompt_weight * c / sum(prompt.values())
    if args.corpus:
        cc = {int(k): v for k, v in json.loads(pathlib.Path(args.corpus).read_text())["counts"].items()}
        total = sum(cc.values())
        for t, c in cc.items():
            score[t] += args.corpus_weight * c / total
    chosen = [t for t, _ in score.most_common()][: args.k]
    # Fill up with the lowest ids not seen (BPE merge order: roughly frequency order).
    seen = set(chosen)
    t = 0
    while len(chosen) < args.k:
        if t not in seen:
            chosen.append(t)
        t += 1
    chosen = sorted(chosen)
    assert len(chosen) == args.k and args.k % 128 == 0
    total = sum(gen.values())
    covered = sum(c for t, c in gen.items() if t in set(chosen))
    print(f"k={args.k}: {len(score)} scored ids; generated-token coverage {covered / total:.5f} "
          f"(seen distinct {len(gen)}, missing {sum(1 for t in gen if t not in set(chosen))})")
    pathlib.Path(args.out).write_text("\n".join(map(str, chosen)) + "\n")
    print(f"wrote {args.out}")
    vs = set(chosen)
    for line in open(ROOT / "bench/golden/prompts.jsonl"):
        g = json.loads(line)
        if g["name"].endswith(("-4096", "-16384")):
            ids = g["ids"]
            miss = sum(1 for t in ids if t not in vs)
            print(f"  held-out {g['name']:18s} coverage {1 - miss / len(ids):.4f} ({miss} missing of {len(ids)})")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("collect")
    c.add_argument("--out", default=str(ROOT / "bench/draft-vocab/counts.json"))
    c.add_argument("--max-new", type=int, default=512)
    cp = sub.add_parser("corpus")
    cp.add_argument("--out", default=str(ROOT / "bench/draft-vocab/corpus.json"))
    b = sub.add_parser("build")
    b.add_argument("counts")
    b.add_argument("--corpus", default="")
    b.add_argument("--corpus-weight", type=float, default=1.0)
    b.add_argument("--k", type=int, default=65536)
    b.add_argument("--prompt-weight", type=float, default=0.5)
    b.add_argument("--out", required=True)
    args = ap.parse_args()
    {"collect": collect, "corpus": corpus, "build": build}[args.cmd](args)


if __name__ == "__main__":
    main()
