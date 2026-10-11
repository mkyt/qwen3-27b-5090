# Configuration

The `serve` service (`compose.yaml`) passes these environment variables to `frontend/server.py`; running the server
directly uses the same defaults. The startup log prints the memory plan. Task IDs such as `P7-13` refer to
[decisions.md](decisions.md).

## Serving

| Variable | Default | Meaning |
| --- | --- | --- |
| `SERVED_MODEL_NAME` | `qwen3.8-27b` | Name in `/v1/models` (requests may use any model name) |
| `MAX_TOKENS` | 262144 | Context limit of one request. If the KV pool cannot hold it, `MAX_TOKENS_FIT` lowers it; the value in effect is in the startup log (`max_tokens ... lowered to`) and in `/v1/models` (`max_model_len`) |
| `MAX_TOKENS_FIT` | 1 | 1: start with `MAX_TOKENS` lowered to what the KV pool holds (a multiple of 1,024). 0: fail to start instead (the service then restarts every 60 s). The engine library (`q27lib.Engine`) defaults to 0 |
| `MAX_SEQS` | 8 | Requests the engine can run at once (1–8); more wait in a queue. How many actually run is decided while running (`ADMIT_KV`, `MIN_SEQ_TPS`, `target_seqs` of the admin API). Each request beyond the first has a 189.6 MiB parking area at the top of the KV pool, lent to the KV cache while unused (P7-13), so `MAX_SEQS=8` costs about 0.25 GiB of fixed per-request buffers. KV pool with gdnfp4 and NVFP4 KV: 620,096 tokens with 4, 605,504 with 8 |
| `ADMIT_KV` | 1 | 1: a waiting request starts only when the KV pool can take it: free pages plus pages held only by prefix-cache checkpoints must cover the remaining prompts of the running requests, `KV_RESERVE_TOKENS` of output per request, and one more parking area. With nothing running a request always starts. 0: start by count only (a request that later runs out of KV is suspended) |
| `KV_RESERVE_TOKENS` | 4096 | Output tokens per request that `ADMIT_KV` keeps room for (or the request's remaining `max_tokens`, if smaller) |
| `MIN_SEQ_TPS` | 0 | Lower bound of tokens per second per request; 0 disables it. Another request does not start if it would push each below this. The prediction uses the measured speed at each number of running requests (decode turns, about the last 10 s); unmeasured counts are scaled from smaller ones by the step time |
| `DECODE_SHARE` | 0.5 | Seconds of decoding given to running requests per second of prefill, so output does not stall during a long prefill |
| `SCHED_QUANTUM_MS` | 50 | Length of a decode turn (with `BATCH_DECODE=0`, the interval between requests' turns; a switch costs about 1 ms) |
| `BATCH_DECODE` | 1 | 1: requests in decode advance together in one batched speculative step (P6-10; about 2.3x aggregate at 4 requests; each request's tokens are the same as alone). 0: requests take turns |
| `PREFILL_CHUNK` | 2048 | Prefill chunk in tokens (8,192 is only 2.5% faster at 32K and needs 4x the activations, P5-4) |
| `KV_DTYPE` | `nvfp4` | `nvfp4`, `fp8` or `bf16`. NVFP4 holds about 1.7x the tokens of FP8 (589,888 against 339,968 at 262K with gdnfp4), prefills as fast or faster (+1 / +6 / +9% at 19K / 73K / 194K, P7-8), and costs a little KL against the BF16 original (0.0470 against 0.0445 with FP8 KV); GSM8K, the Japanese set and needle retrieval are the same. BFCL `parallel` is about 3 points lower than FP8 on short prompts and the same after 32K / 96K of agent context (P4-11, P7-8, P7-9) |
| `WEIGHTS` | `gdnfp4` | `gdnfp4`: GDN projections in NVFP4, attention projections FP8 ([juzbox/Qwen3.8-27B-NVFP4-GDN](https://huggingface.co/juzbox/Qwen3.8-27B-NVFP4-GDN), pinned revision, `scripts/download-weights.sh`; PQ-14). `nvidia`: NVIDIA's checkpoint as is (GDN / attention projections FP8). `quasar`: the QUASAR-QAT checkpoint, every projection NVFP4, weaker at parallel tool calls (`scripts/download-weights.sh --quasar`; its BF16 `lm_head` is replaced by NVIDIA's NVFP4 one through `Q27_LM_HEAD_CKPT`). Or a model directory inside the container. From Python: `qwen27b.weights.engine_target(...)`; scripts take `--weights` |
| `SPEC` | `dflash2` | `dflash2` (speculative decoding with the DFlash2 draft) or `none` |
| `Q27_DRAFT_QUANT` | `fp4` | Draft linear layers: `fp4`, `fp8` or `bf16` (same acceptance, FP4 is fastest) |
| `Q27_DRAFT_VOCAB` | `bench/draft-vocab/vocab-128k.txt` | Candidate vocabulary of the draft (`tools/draft_vocab.py`); `none` for the full vocabulary |
| `STATE_HOST_CACHE` | 4 | GiB of pinned host memory for prefix-cache checkpoints (4 = 21 checkpoints; 0 disables the cache; the engine library defaults to 0) |
| `KV_POOL_TOKENS` | 0 | KV pool size in tokens (0 = all the VRAM budget leaves). When set, it is the KV cache with every parking area in use; the pool is that plus the parking areas (P7-13) |
| `CHECKPOINT_INTERVAL` | 8192 | Distance between checkpoints inside a long prompt (a multiple of 64) |
| `VISION` | `host` | Image input. `host`: the vision tower's BF16 weights (0.86 GiB) stay in pinned host memory and are streamed to the GPU one layer ahead of the computation; the KV pool loses 192 tokens, small images take +13 ms (the copy is about 20 ms), 1024x1024 and larger +1–4 ms; output identical to `bf16`. `fp8`: E4M3 weights with per-row scales on the GPU, widened per layer (0.43 GiB; embedding error about 3x that of `bf16`, answers can change). `bf16`: BF16 weights on the GPU (fastest, 0.86 GiB). `off`: requests with images get HTTP 400 (the engine library's default). Activations and the weight staging (90 MiB) share the prefill's buffers |
| `IMAGE_MAX_PIXELS` | 4194304 | Images are resized to at most this many pixels (2048x2048 = 4,096 tokens, 370 ms in the vision tower) |
| `IMAGE_TOKENS` | 16384 | Image tokens per request (embeddings in pinned host memory, 10 KiB per token: 160 MiB) |
| `VRAM_BUDGET_GIB` / `WDDM_MARGIN_GIB` | free − 0.6 / 0.6 | VRAM budget (at most 30.3 GiB) and the margin kept free for the display driver |
| `API_DUMP_DIR` | unset | Debugging: write the body of every POST to this directory (to see what a client sends; may contain personal data) |

### Attention numerics

| Variable | Default | Meaning |
| --- | --- | --- |
| `Q27_ATTN_FP8` | empty | Prefill attention MMAs, read per call. Empty: `2` with FP8 KV, `3` with NVFP4 KV (`2` without the staging buffer). FP8 KV: `2` QK^T and P·V in FP8, `1` P·V only, `0` BF16. NVFP4 KV: `3` QK^T on native FP4 MMAs (Q quantized to NVFP4, K as cached, V from the E4M3 staging buffer; P7-8), `2` K and V rounded to E4M3 for FP8 MMAs, otherwise exact in f16 |
| `Q27_ATTN_FP4_STAGE` | 1 | NVFP4 KV prefill: `1` converts the context to an E4M3 staging buffer once per layer and chunk (mode 3: V only, 1 KiB per token of `max_tokens`, 256 MiB at 262K; with `Q27_ATTN_FP8=2` at startup K and V, 2 KiB per token, bit-identical to per-CTA conversion). `0`: convert in every CTA (mode 2). Sized at startup, read per call |
| `Q27_ATTN_TC_F8` | empty | Decode / verify attention MMAs, read per call. Empty: FP8 with NVFP4 KV (P7-7), `1`: FP8 also with FP8 KV, `0`: BF16 / f16. FP8 means Q in E4M3 with per-row scales, NVFP4 KV re-rounded to E4M3, P in E4M3: 8% faster speculative steps at 131K and at 8 x 32K, no change on golden or needle retrieval |

## Concurrency

- Up to `MAX_SEQS` requests run at once. Prefill goes to the request with the least prompt left; requests in decode
  advance together in batched steps (with `BATCH_DECODE=0` they take turns every `SCHED_QUANTUM_MS`).
- New requests wait until the KV pool can take them (`ADMIT_KV`). If the KV cache still runs out, the latest request
  (including the one that ran out) is suspended to a checkpoint and goes back to the head of the queue. When requests
  that arrive together share a prefix, the later one waits until the earlier one has passed it.
- Admin API (no authentication; the service listens on 127.0.0.1 only): `GET /admin/concurrency` returns the settings
  and the state (running, waiting, why requests wait, measured tok/s per number of running requests);
  `POST /admin/concurrency` changes `target_seqs` (1 to `MAX_SEQS`; lowering it does not stop running requests),
  `min_seq_tps`, `admit_kv` and `kv_reserve`. Changes survive an engine restart, not a service restart.
  Example: `curl -s -X POST localhost:8000/admin/concurrency -d '{"target_seqs": 4}'`.
- `/metrics` (Prometheus): among others `q27:target_seqs`, `q27:admission_blocked{reason=...}`,
  `q27:admission_waits_total{reason=...}`, `q27:cached_tokens_total`, `q27:reusable_tokens_total`.

## Prefix cache

The server takes checkpoints at message boundaries and at the end of each generation; the next request resumes from
the deepest match. The log line `[api] prompt N tokens, cached C of R reusable` and the `q27:cached_tokens_total` /
`q27:reusable_tokens_total` metrics show how well it hits. Clients must send the previous turns' reasoning back, since
the chat template renders it.

## Development switches

All default to the faster setting. Except those marked static, they can be switched within one process with
`Engine.configure()` (`bench_spec.py --ab`).

| Variable | Meaning |
| --- | --- |
| `Q27_PROFILE=1` | GPU time per operation, also inside CUDA graphs (`Engine.profile_dump()`) |
| `Q27_NO_GRAPH=1` / `Q27_NO_PDL=1` / `Q27_NO_SKINNY=1` | Disable CUDA graphs / programmatic dependent launch / the skinny GEMMs |
| `Q27_SKINNY_CHECK=1` | Compare the skinny GEMMs with CUTLASS and log the difference |
| `Q27_SKINNY_EXACT=1` | Make a row's skinny-GEMM result independent of M (tests of tree verification and preemption) |
| `Q27_SKINNY_EXT_REDUCE_M` | Rows from which the skinny GEMM's split-K reduction runs as a separate kernel (default 1: M >= 2; at M = 1 the last CTA reduces; same result; P7-4) |
| `Q27_DRAFT_ATTN_TC=0` | The draft's attention on CUDA cores (default: tensor cores; P7-4) |
| `Q27_SKIP=norm,silu,gdn,attn,gemm,gemm4,gemm8` | Skip a kind of kernel to measure its share of the time (`gemm4` / `gemm8`: FP4 / FP8 GEMMs only; the output is garbage) |
| `Q27_DECODE_MEMCPY=1` | Decode inputs and outputs by copies instead of mapped pinned memory |
| `Q27_ATTN_TC=0` | Decode / verify attention on CUDA cores (tree verification always uses them) |
| `Q27_ATTN_TC_DEC=0` | Decode (T = 1) attention with one warp per token (same result) |
| `Q27_ATTN_TC8_GRP=0` | With `Q27_ATTN_TC_F8=1`, compute T <= 2 in one warp instead of four (same result; static) |
| `Q27_KV_BLOCK=N` | Pages per block of the KV pool (P7-13; default the smallest count whose slice is >= 2 MiB: 57 for NVFP4, 32 for FP8, 16 for BF16). 1 lays pages out one by one (long-context decode 1.5% slower). Read at startup |
| `Q27_PAGES_HIGH=1` | Hand out KV pages from the top (tests, P7-13): the KV cache then occupies the parking areas, so every park moves pages. Read at startup |
| `Q27_ATTN_PREFILL_TC=0` | Prefill attention on CUDA cores (static) |
| `Q27_GDN_CHUNK=0` | Sequential GDN in prefill (static) |
| `Q27_MLP_BF16=0` | fp32 outputs for the prefill MLP's gate / up |
| `Q27_GEMM_MXFP8=0` | Prefill FP8 GEMMs with the plain FP8 CUTLASS kernel (default: block-scaled MXFP8 with unit scales, bit-identical, P5-11; read per call) |
| `Q27_SPEC_PATHS=K` | Paths of tree verification (default 1 = a chain; 2 / 4 accept more but take longer and break even) |
| `Q27_LOOKUP=1` | Lookup drafting (off: no gain with the 8-row chain) |
| `Q27_SPEC_PIPELINE=0` | Enqueue a speculative step after reading the previous result (default 1: before) |
| `Q27_SKINNY_WAVES` / `Q27_SKINNY_FIT` / `Q27_SKINNY_L2_CHUNKS` / `Q27_ATTN_WAVES` / `Q27_ATTN_MIN_KEYS` / `Q27_ATTN_PREFETCH` / `Q27_ATTN_TC_SPLITS` | Tuning |
| `Q27_LIB` / `Q27_TRACE` | Another build of `libq27.so` (e.g. for tracing) / where a `-DQ27_TRACE` build writes its timeline |
