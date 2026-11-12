# Benchmarks

Speed and quality of the default configuration (`docker compose up -d serve` with no settings changed: weights
`gdnfp4`, KV cache `nvfp4`, DSpine speculative decoding, `MAX_SEQS=8`, `MAX_TOKENS=262144`), and how to reproduce
each number. The JSON files of the speed numbers are in [results/](results/).

The speed through the server was measured again when DSpine became the default draft (2026-11-13), together with
DFlash2, the previous default (`SPEC=dflash2`), on the same build. The in-process table by context length runs
without speculative decoding and is from 2026-10-11; the comparison with other engines and the power limit section
keep their own dates.

## Conditions

| | |
| --- | --- |
| GPU | RTX 5090 (32 GB), power limit **400 W** (below the card's default 575 W; every number here is at 400 W) |
| Host | WSL2 (kernel 6.18), Docker with the NVIDIA Container Toolkit; NVIDIA driver 621.14 (2026-11-13), 617.42 (2026-10-11) |
| Build | the serving image (CUDA 13.3), code `9e74583` (2026-11-13) |
| Date | 2026-11-13 through the server; 2026-10-11 in-process and against other engines |

Prefill is power-bound at 400 W: the FP4 / FP8 GEMMs and attention all run at the power limit. The card's default
575 W limit and the energy per token at both limits are in [Power limit and energy](#power-limit-and-energy).

## Speed

### One request

Through the server (`scripts/bench_agentic.py --scenarios decode`): three short prompts, 1,024 output tokens, the
checkpoint's default sampling (temperature 1.0, top-k 20, top-p 0.95), thinking off, three runs each. Decode tok/s is
(tokens - 1) / (time of the last token - time of the first).

| Prompt | DSpine (default), decode tok/s | DFlash2 | DSpine / DFlash2 (median) |
| --- | ---: | ---: | ---: |
| code | 389 / 457 / 410 | 325 / 392 / 382 | 1.07 |
| JSON | 734 / 663 / 674 | 496 / 493 / 501 | 1.36 |
| prose | 267 / 263 / 258 | 265 / 257 / 256 | 1.02 |

The speed depends on how many draft tokens are accepted, which depends on the text: structured output goes faster
than prose. DSpine drafts blocks of 16 tokens and verifies 16 or 8 rows depending on how many it has been getting
accepted, so it gains most where long runs of tokens are predictable (JSON, code). Without speculative decoding (`SPEC=none`) one request decodes at about 93 tok/s at a 19K-token context.

### Prefill and decode by context length

In-process, one request, no speculative decoding (`scripts/bench_decode.py`). Prefill is the best of two cold runs
after a warm-up; decode is the median of 64 greedy steps after the prefill. The context is the longest golden prompt
(Japanese prose) repeated.

| Context | Prefill tok/s | Prefill time | Decode tok/s |
| ---: | ---: | ---: | ---: |
| 19,000 | 11,295 | 1.68 s | 92.9 |
| 73,000 | 8,773 | 8.32 s | 87.2 |
| 194,000 | 5,875 | 33.0 s | 76.7 |

### Concurrent requests

Through the server (`scripts/bench_concurrent.py`): C requests start together on the same reasoning prompt with
distinct seeds, 4,096 output tokens each, the default sampling. "Sum" adds each stream's own decode rate; "wall"
divides all output tokens by the time from the start to the last response. Two runs each.

| C | Sum tok/s | Per request (p50) | Wall tok/s | DFlash2, sum tok/s |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 355–356 | 355–356 | 354 | 310 |
| 4 | 892–893 | 223 | 843–844 | 928–930 |
| 8 | 1,304–1,306 | 162 | 1,244–1,246 | 1,391–1,393 |

All C requests run as one batch, and each request's tokens are bit-identical to running it alone. In a batch, every
request drafts 8 tokens per step with either draft; DSpine accepts fewer of them here, so with four or eight requests
at once DFlash2 is 4% and 7% faster. `SPEC=dflash2` is the better choice for a server that mostly runs many
requests at once.

### Agent sessions and long documents

Through the server (`scripts/bench_agentic.py --scenarios agent,longctx`).

- **agent**: sessions that share a 4,000-token system prompt and 20 tools, each running 8 turns of tool calls with
  800–3,000-token tool results (the history grows to about 20K tokens). One session: time to the first token of a
  later turn 0.22 s (p50), decode 283 tok/s (DFlash2 256), prefix cache hit rate 84% of the prompt tokens, 99% of the
  ones that can be reused. Four at once: 0.23 s, 170 tok/s each (DFlash2 158), 98% of the reusable tokens. The turns
  differ between runs (the model decides when to call tools), so the decode rates are rough.
- **longctx**: a long unique document, then a follow-up question on it. 19.5K tokens: first token after 1.75 s cold,
  0.05 s for the follow-up. 73K tokens: 8.27 s cold, 0.12 s for the follow-up.

### KV cache capacity

The KV pool holds 601,856 tokens with the defaults (605,504 with `SPEC=dflash2`: DSpine's tables take 0.25 GiB more) (the server log prints the memory plan at startup). One request can
use the full 262,144-token context; the scheduler admits a new request only when the free KV cache can hold its prompt
plus 4,096 output tokens, so eight long requests wait in line instead of evicting each other.

### Reproduce

```bash
docker compose up -d serve
uv run scripts/bench_agentic.py --label decode --scenarios decode --decode-reps 3
uv run scripts/bench_concurrent.py --label concurrent --prompts reason --concurrency 1,4,8 --max-tokens 4096 --reps 2 \
  --extra-body '{"temperature":1.0,"top_p":0.95,"top_k":20,"presence_penalty":0}'
uv run scripts/bench_agentic.py --label agent --scenarios agent,longctx
uv run scripts/bench_agentic.py --label prefill-s0 --seed 0 --scenarios longctx --longctx-tokens 32000,120000,250000
SPEC=dflash2 docker compose up -d serve   # the DFlash2 columns: the first three commands again
docker compose stop serve
docker compose --profile dev run --rm dev python scripts/bench_decode.py --weights gdnfp4 --kv-dtype nvfp4 \
  --ctx 19000,73000,194000 --tokens 64 --prefill-reps 2
```

The results go to `bench/results/`. The runs above are [results/](results/) `20261113-*-dspine-release-*.json`
(DSpine), `20261113-*-dflash2-release-*.json` (DFlash2) and
[results/20261011-pr4-prefill-decode.json](results/20261011-pr4-prefill-decode.json) (by context length). The
measurements with DFlash2 as the default on 2026-10-11 are `20261011-*-pr4-release-*.json`.

### Other engines

The same three scripts against each engine's own serving setup, on the same day (2026-10-11), GPU, driver and 400 W
limit, with
the sampling pinned for all of them (`--extra-body '{"temperature":1.0,"top_p":0.95,"top_k":20,"presence_penalty":0}'`).
Each engine runs its own weights and KV cache type, and some need different settings for different scenarios
because one configuration cannot do everything; the settings are listed below the table. The first column is this
engine with today's default (DSpine, 2026-11-13, the same scripts and settings); the second is the same-day run
with DFlash2, the default then.

| | This engine, DSpine (2026-11-13) | This engine, DFlash2 | [NInfer](https://github.com/Neroued/ninfer) | [SGLang](https://github.com/sgl-project/sglang) | [vLLM](https://github.com/vllm-project/vllm) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Context limit / KV cache, tokens | 262,144 / 601,856 | 262,144 / 605,504 | 16,384 / 43,904 (x8), 94,208 / 96,128 (x4) | 90,000 / 98,134 (fast), 196,608 / 214,219 (agent) | 160,000 / 173,913 |
| Decode, one request, code / JSON / prose (tok/s) | **410 / 674 / 263** | 386 / 500 / 258 | 285 / 362 / 197 | 246 / 320 / 167 (fast) | 154 / 180 / 128 |
| Concurrent, sum at C=1 (tok/s) | **355–356** | 312–313 | 234 | 197–205 (fast) | 124 |
| Concurrent, sum at C=4 | 892–893 | **935–936** | 761–796 | 457–466 (agent) | 498–500 |
| Concurrent, sum at C=8 | 1,304–1,306 | **1,395–1,397** | 1,260–1,277 | 837–851 (parallel) | 912–930 |
| Agent turn, time to first token, 1 / 4 sessions (p50) | 0.22 / **0.23 s** | **0.19** / 0.24 s | 0.39 / 0.75 s (x4) | 0.31 / 0.46 s (agent) | 0.57 / 1.59 s |
| A second question on a 19.5K / 73K-token document, time to first token | **0.05 / 0.12 s** | 0.06 / 0.14 s | 2.20 / 10.94 s (x4) | 0.22 / 0.49 s (agent) | 0.39 / 1.01 s |

Decode is the median of three runs; the concurrent sums are two runs; prefill is below. With DFlash2 on 2026-11-13
the agent turn took 0.22 / 0.38 s, so the time to the first token moved with the day, not with the draft. The agent scenario's turns
differ in length between engines (the model decides when to call tools), so only the time to the first token of a
later turn is compared.

#### Prefill

A long document that is not in any cache (`scripts/bench_agentic.py --scenarios longctx --longctx-tokens
32000,120000,250000`, seeds 0 and 1, two different documents), through the server. The first three rows are the usual
measure: prompt tokens divided by the time to the first token, an average over the whole prompt. Because attention
grows with the position, the speed at a given depth is lower; the second set of rows gives the speed between two
lengths, (difference in tokens) / (difference in time), the mean of the two seeds.

| Prompt tokens | This engine, DSpine (2026-11-13) | This engine, DFlash2 | NInfer | SGLang | vLLM |
| --- | ---: | ---: | ---: | ---: | ---: |
| 19.5K, average tok/s (time to first token) | **11,438–11,594** (1.68–1.71 s) | 10,705–10,929 (1.78–1.82 s) | 9,271–9,735 (2.00–2.11 s) | 7,231–7,804 (2.50–2.69 s) | 7,530–7,789 (2.50–2.59 s) |
| 73K | **9,029–9,193** (7.94–8.08 s) | 8,519–8,675 (8.41–8.56 s) | 6,861–7,048 (10.35–10.63 s) | 4,771–4,854 (15.03–15.28 s) | 4,704–4,814 (15.16–15.50 s) |
| 152K | **6,796–6,864** (22.1–22.4 s) | 6,508–6,575 (23.1–23.4 s) | 4,889–4,941 (30.8–31.1 s) | 2,993–3,022 (50.3–50.8 s) | 2,707–2,741 (55.5–56.1 s) |
| 0 → 19.5K, tok/s in the interval | **11,516** | 10,817 | 9,503 | 7,517 | 7,659 |
| 19.5K → 73K | **8,466** | 7,998 | 6,335 | 4,256 | 4,182 |
| 73K → 152K | **5,548** | 5,359 | 3,868 | 2,234 | 1,953 |

The settings are those in the table below, except NInfer, which ran without speculative decoding (`--max-context
262144 --max-concurrency 1`): 152K does not fit with DFlash2 under WSL2. Without speculative decoding it also skips
the pass of the draft model (or MTP head) over the prompt that the other engines run. The time to the first token includes HTTP,
tokenization and the first decode step, so these are slightly below the in-process prefill numbers above
(11,295 / 8,773 tok/s at 19K / 73K).

| Engine | Version | Weights | KV cache | Speculative decoding | Settings |
| --- | --- | --- | --- | --- | --- |
| This engine | this repository (`9e74583` / 2026-10-11) | `gdnfp4` | NVFP4 | DSpine, 16-token blocks (adaptive 16 / 8 rows) / DFlash2, 8-token blocks | defaults (`MAX_SEQS=8`) |
| [NInfer](https://github.com/Neroued/ninfer) | [`81c8ce0`](https://github.com/Neroued/ninfer/commit/81c8ce093b2c1646a87566a8e59d807fcf0ec95c), built on this repository's `dev` image | `neroued/Qwen3.8-27B-nvfp4-NInfer` | FP8 | DFlash2, 7 draft tokens, `--lm-head-draft` | x8: `--max-context 16384 --max-concurrency 8 --device-state-slots 0` (decode, concurrency); x4: `--max-context 94208 --max-concurrency 4` (agent, documents); larger settings do not fit under WSL2 |
| [SGLang](https://github.com/sgl-project/sglang) | [v0.5.21](https://github.com/sgl-project/sglang/releases/tag/v0.5.21) with two unmerged patches: embedding in host memory ([#37826](https://github.com/sgl-project/sglang/pull/37826), backported) and MTP weight sharing ([#37155](https://github.com/sgl-project/sglang/pull/37155)) | `nvidia/Qwen3.8-27B-NVFP4` | FP8 | fast: DFlash2 (one request at a time); agent / parallel: the checkpoint's MTP head, 5 steps | fast: 90K, 1 request; agent: 196K, 4 requests; parallel: 160K, 8 requests |
| [vLLM](https://github.com/vllm-project/vllm) | [v0.29.0](https://github.com/vllm-project/vllm/releases/tag/v0.29.0) | `nvidia/Qwen3.8-27B-NVFP4` | FP8 | MTP, 3 tokens | 160K, 8 requests, a fixed 6.5 GiB KV cache |

The JSON files are in [results/](results/) (`20261011-1*-pr6-*.json`, this engine on 2026-11-13
`20261113-*-dspine-release-*.json`). Notes:

- NInfer sizes its memory from what `cudaMemGetInfo` reports free minus a 1 GiB margin. Under WSL2 that leaves
  9.5 GB after its weights, which limits the context with several requests. It reuses a prefix only at the end of a
  conversation turn, so a new question about the same document is prefilled again.
- SGLang's DFlash2 mode spends about 5 GB on the draft model and per-request verification states, so it serves one
  request at a time with a 90K context.
- `gdnfp4` also runs on vLLM (`scripts/export_gdnfp4.py` writes a checkpoint it reads); there it decoded 7 to 15%
  faster than NVIDIA's checkpoint with the same GSM8K and BFCL results (measured 2026-10-10).

## Power limit and energy

The same scripts at 400 W and at the card's default 575 W, on the same day (2026-10-27) and the same code (`1b1b430`),
serving with the defaults. The energy is the whole board's, read from NVML's cumulative energy counter before and after
each request (`scripts/bench_energy.py`), so it includes the idle power (about 30 W) during the request.

| | 575 W | 400 W | 400 W vs 575 W |
| --- | ---: | ---: | --- |
| Decode, one request, code / JSON / prose (tok/s, median) | 391 / 507 / 263 | 386 / 500 / 259 | 1–2% slower |
| Concurrent, sum at C=1 / 4 / 8 (tok/s) | 319 / 1,020 / 1,546 | 313 / 939 / 1,403 | 2 / 8 / 9% slower |
| Time to first token, 19.5K / 73K-token document | 1.48 / 7.04 s | 1.86 / 8.71 s | 20% slower |
| Board power: decode / C=8 / prefill (W) | 420–456 / 564 / 570 | 374–408 / 402 / 390–396 | |
| Energy, decode code / JSON / prose (tok/J) | 0.72–0.88 / 1.09–1.14 / 0.58–0.59 | 0.80–0.95 / 1.22 / 0.63–0.66 | 8–13% more tokens per joule |
| Energy, concurrent C=1 / 4 / 8 (tok/J) | 0.62 / 1.68 / 2.45 | 0.70 / 2.14 / 3.15 | 13 / 27 / 29% more |
| Energy, prefill of 11.9K / 44.4K tokens (prompt tok/J) | 23.3 / 20.5 | 27.1 / 23.6 | 15–16% more |

Decode of one request draws 420–456 W even when it may draw 575 W, so the 400 W limit barely slows it. Prefill and
many concurrent requests use whatever the limit allows: 575 W buys 9–26% more speed for 40% more power. At 575 W the
board peaked at 580 W and 81 °C. The energy rows of the concurrent requests are one run of 2,048 tokens per stream,
timed until the last stream ends, so their tok/s is below the rows above (4,096 tokens, decoding streams only).

```bash
uv run scripts/bench_energy.py --label energy
```

The runs are [results/](results/) `20261027-*-pl575-*.json` and `20261027-*-pl400-*.json`.

## Quality

### Next-token distributions (golden)

Teacher forcing over 13 prompts, 35,328 positions (Japanese and English prose, Python and C++; `bench/golden/`).
Each recording keeps the top-20 next-token distribution at every position; two recordings are compared by top-1
agreement and KL divergence ([results/20261011-pr2-golden-v2.json](results/20261011-pr2-golden-v2.json)).

| Compared | Top-1 agreement | KL |
| --- | ---: | ---: |
| Reference model, KV bf16 vs KV fp8 (the noise floor of W4A4 rounding) | 89.1% | 0.0359 |
| Engine vs reference model (NVIDIA's checkpoint, KV bf16) | 89.3% | 0.0363 |
| NVIDIA's checkpoint vs the BF16 original (reference model) | 88.7% | 0.0361 |
| Engine, default (`gdnfp4`, KV nvfp4) vs the BF16 original | 87.4% | 0.0470 |
| Engine, `gdnfp4`, KV fp8 vs the BF16 original | 87.6% | 0.0445 |
| Engine, `nvidia`, KV fp8 vs the BF16 original | 88.7% | 0.0362 |
| Engine, `quasar`, KV fp8 vs the BF16 original | 87.2% | 0.0475 |

The engine is as far from the reference model as the reference model is from itself with another KV cache type, so
the kernels compute the model correctly; what remains against the BF16 original is the quantization of the weights
and the KV cache.

### Tasks

The default configuration through the server (thinking off, greedy unless noted):

| Task | Result | Script |
| --- | --- | --- |
| GSM8K (300 fixed questions) | 95.3% | `quality_gsm8k.py` |
| BFCL v3 non-live AST (simple, multiple, parallel, parallel_multiple; 1,000 questions) | 88.9% | `quality_tools.py` |
| Needle retrieval, 16 facts at 190K tokens | 16 / 16 | `needle_test.py --tokens 190000 --needles 16` |
| Needle retrieval, 16 facts at 256K tokens, seeds 0–3 | 56 / 64 | `needle_test.py --tokens 256000 --needles 16 --seed N` |
| Japanese: reading (JSQuAD), translation both ways (FLORES-200), summarization (XL-Sum), LLM-judged | same as KV fp8 within the confidence intervals | `quality_ja.py` |
| Synthetic image questions (70, exact answers) | 67 / 70 (vLLM 70) | `vision_quality.py` |

Notes:

- The NVFP4 KV cache costs parallel tool calls on short prompts: over BFCL's parallel categories (400 questions,
  temperature 0.7, three seeds) it gets 11 fewer right than the FP8 KV cache, nearly all by stopping a parallel
  call list early. After an agent's 32K- or 96K-token history (`quality_tools.py --context`) the difference is gone
  (-4 and +2 questions), which is how agents use the server. `KV_DTYPE=fp8` trades 42% of the KV capacity for it.
- The image questions were measured with NVIDIA's weights; the three misses are close counting questions that the
  reference model gets wrong the same way.
- The evaluation data is downloaded on first use into `bench/quality-data/` (pinned revisions). XL-Sum is
  CC BY-NC-SA 4.0 and is not redistributed here.

## Known weak spots

- **Several long requests decoding at once.** Decode attention is tuned for one long request or several short ones;
  four requests with long contexts each decode slower per request than the bandwidth would allow.
- **Prefill beyond 100K tokens** slows down as attention takes over (5,875 tok/s at 194K), and is power-bound.
- **More than eight concurrent requests** need kernel changes (the batched GEMMs handle up to 64 rows: 8 requests of
  up to 8 tokens).
- Only the RTX 5090 is supported.
