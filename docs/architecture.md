# Architecture

How the engine is built and why. The numbers are measured on an RTX 5090 at a 400 W power limit
([benchmarks.md](benchmarks.md)); the options that were tried and dropped, with their measurements, are in
[decisions.md](decisions.md), whose task IDs (such as `P6-10`) also appear in code comments.

Contents:

1. [Goals](#1-goals)
2. [The model and the GPU](#2-the-model-and-the-gpu)
3. [Processes](#3-processes)
4. [Weights](#4-weights)
5. [Memory plan](#5-memory-plan)
6. [A decode step](#6-a-decode-step)
7. [Speculative decoding](#7-speculative-decoding)
8. [Prefill](#8-prefill)
9. [Prefix cache](#9-prefix-cache)
10. [Scheduler and concurrency](#10-scheduler-and-concurrency)
11. [Vision tower](#11-vision-tower)
12. [API server](#12-api-server)
13. [Correctness](#13-correctness)

## 1. Goals

- Run Qwen3.8-27B on one RTX 5090 with prefill and decode close to the hardware's limits (memory bandwidth for
  decode, tensor-core throughput at the power limit for prefill).
- Waste no VRAM, so that one request can use the model's full 262,144-token context and several requests can run
  at once.
- Serve one person's coding agents: a main agent and a few sub-agents, long contexts that grow turn by turn,
  thinking, tool calls. Batch work (up to eight independent requests) is second.
- Make every setting that trades quality for speed or capacity an option, and make it the default only after it
  passes the same quality tests as the checkpoint ([§13](#13-correctness), [benchmarks.md](benchmarks.md)).

Out of scope: other models and GPUs, more than one GPU, KV cache on disk, more than eight concurrent requests.

## 2. The model and the GPU

**Qwen3.8-27B** has 64 layers: 48 Gated DeltaNet (GDN, linear attention with a recurrent state) and 16 gated full
attention layers (every fourth layer). Hidden size 5120, MLP 17408, vocabulary 248,320.

- Attention: 24 query heads, 4 KV heads, head dimension 256, an output gate, partial RoPE (64 of 256 dimensions),
  q / k norms. KV cache per token: 64 KiB in BF16, 32 KiB in FP8, 18 KiB in NVFP4. One 262K-token request needs
  16 / 8 / 4.5 GiB.
- GDN: 16 key heads and 48 value heads of 128, a short convolution (kernel 4), a gated RMSNorm. Its state is
  48 x 128 x 128 per layer and sequence, fixed in size: 144 MiB per sequence in FP32 for all 48 layers.
- Because only 16 layers keep a KV cache, the full context fits in 32 GB with an 8-bit or 4-bit KV cache.

**The RTX 5090** (SM120): 170 SMs, 32 GB of GDDR7. Measured: 1.70 TB/s for streaming reads, and register-only MMA
loops at 2,014 TFLOPS (NVFP4), 1,007 TFLOPS (FP8 as block-scaled `mxf8f6f4`; the plain FP8 MMA with FP32
accumulation runs at half that rate) and 253 TFLOPS (BF16) (`micro/`). SM120 has the warp-level `mma.sync`
instructions with block scales, but not the `tcgen05` / TMEM / `wgmma` of data-center Blackwell and Hopper, so
kernels written for those GPUs do not run; the engine's kernels, prefill GEMMs included, are its own (no CUTLASS
since P5-12).

At 400 W, compute-bound work runs at the power limit: an FP4 GEMM holds the SM clock at about 1.8 GHz. Prefill
speed is therefore set by energy per FLOP, which favors the FP4 and FP8 tensor cores over BF16. Decode is
memory-bound and is not affected by the limit.

Under WSL2, the GPU driver does not fail an allocation when VRAM runs out: it silently moves memory to system RAM
and everything becomes several times slower. Measured: process allocations up to 30.75 GiB stay in VRAM, at
31.0 GiB they spill, and `cudaMemGetInfo` already reports 0 free before that. The engine therefore plans its memory
against an explicit budget instead of relying on allocation failures ([§5](#5-memory-plan)). Pinned host memory
counts toward Windows' "shared GPU memory" without slowing anything down.

## 3. Processes

```
 clients (Claude Code, Codex, OpenAI / Anthropic SDKs)
    |  HTTP 127.0.0.1:8000
    v
 frontend/server.py   API server (asyncio): chat template, tokenizer, reasoning and tool-call parsing,
    |                 image preprocessing, streaming, metrics, admin API
    |  multiprocessing pipe
    v
 frontend/worker.py   engine process: the scheduler (Python) and the engine library
    |  ctypes (engine/python/q27lib.py, C API in engine/src/capi.cc)
    v
 libq27.so            C++20 / CUDA, no PyTorch: weights, memory plan, kernels, speculative decoding,
                      prefix cache, vision tower
```

- **Two processes.** A CUDA error kills only the engine process: the server answers 503, starts a new engine and
  keeps its settings. The server never touches CUDA, so it takes nothing from the VRAM budget, and tokenization
  does not compete with the scheduler for the GIL.
- **No PyTorch in the engine.** The CUDA context and libraries stay small, nothing allocates while serving, and the
  memory plan does not depend on a profiling run. (In the vLLM setup this replaced, the KV cache size varied
  between 4.3 and 6.5 GiB from one start to the next.) PyTorch is used only by the reference model and the tests.
  The server does not use it either, nor transformers: the tokenizer is `tokenizers` on the checkpoint's
  `tokenizer.json`, the chat template renders with jinja2 as transformers sets it up, and images are
  preprocessed with PIL and numpy (`frontend/requirements.txt`; `frontend/tests/test_tokenizer.py` compares
  prompts, ids and image grids with transformers). The serving image (`Dockerfile`) is therefore 350 MB: Ubuntu, the
  CUDA runtime library, `libq27.so`, Python and those packages (1.21 GB on the nvidia/cuda base image, 456 MB of it
  the forward-compatibility driver package, which GeForce GPUs do not use).
- **The scheduler is Python.** A speculative step takes 12 to 25 ms; the scheduler's decisions take microseconds by
  comparison, and the engine queues the next step before the host reads the previous one ([§6](#6-a-decode-step)).

## 4. Weights

| Part | Format in `gdnfp4` (default) | GiB on the GPU | Read every decode step |
| --- | --- | ---: | :---: |
| MLP (gate, up, down) | NVFP4, W4A4 (16-element blocks, FP8 block scales) | 8.96 | yes |
| GDN projections (qkv, z, out; a and b) | NVFP4, W4A4 (a and b: BF16) | 2.94 | yes |
| Attention projections (q, k, v, o) | FP8, W8A8 (per-tensor scales) | 1.56 | yes |
| lm_head | NVFP4 | 0.67 | yes |
| Embedding | BF16, in pinned host memory | 0 | one row per token |
| Vision tower | BF16, in pinned host memory (`VISION=host`) | 0 | only for images |
| DFlash2 draft | NVFP4, quantized at load time | about 1.2 | yes |

- **The checkpoint is read as it is**: NVIDIA's NVFP4 checkpoint, or `gdnfp4`, which replaces its GDN projections
  (FP8 in NVIDIA's) with NVFP4 ones quantized by GPTQ from the BF16 original. The loader rearranges the tensors on
  the GPU into the kernels' layouts; there is no converted copy on disk. The attention projections stay FP8
  because quantizing them costs parallel tool calls; [decisions.md](decisions.md#weights) has the measurements.
  `scripts/build-gdnfp4.sh` rebuilds it (it needs the BF16 original and a few hours).
- **Fused projections.** GDN's `qkv | z`, attention's `q (+gate) | k | v` and the MLP's `gate | up` each run as one
  GEMM. Their activation scales are equal in every layer of the checkpoint, so fusing does not change the
  activation quantization.
- **Embedding in host memory.** A decode step reads one 10 KB row per token over PCIe by UVA (a kernel reads pinned
  host memory directly); a prefill chunk reads 2,048 rows. This frees 2.37 GiB of VRAM, about 138K tokens of NVFP4
  KV cache.
- **The draft model** (DFlash2, BF16 on disk) is quantized to NVFP4 while loading, with per-row scales and
  per-token dynamic activation scales ([§7](#7-speculative-decoding)).

## 5. Memory plan

At startup the engine computes a plan, allocates each pool once and never calls `cudaMalloc` while serving.

- **Budget**: the free VRAM at startup minus `WDDM_MARGIN_GIB` (0.6 GiB), at most 30.3 GiB. If a configuration
  does not fit, the engine stops with the breakdown in the log; the server's `MAX_TOKENS_FIT=1` instead lowers
  `MAX_TOKENS` to what fits.
- **Pools** (separate, no padding to a common page size):
  1. Weights.
  2. Activations: one arena laid out from the largest prefill chunk (2,048 tokens) and the largest batched step
     (64 rows), shared in time by the prefill, the vision tower and its weight staging.
  3. The running sequence's state: GDN states (FP32), convolution states, the draft's context ring.
  4. Per-lane buffers of batched steps ([§10](#10-scheduler-and-concurrency)), about 0.25 GiB for eight lanes.
  5. **The KV pool**: everything left, in pages of 64 tokens. With the defaults (`gdnfp4`, NVFP4 KV,
     `MAX_SEQS=8`, `VISION=host`) it holds 605,504 tokens.
- **Parking areas live inside the KV pool.** Each request that is not running keeps its GDN, convolution and
  draft state (189.6 MiB) in a parking area; there are `MAX_SEQS - 1` of them. While an area is unused, its pages
  hold KV cache like any other page. The KV pool is laid out page-major in blocks (a block holds all 32 slices,
  16 layers x K / V, of G consecutive pages, with G = 57 for NVFP4 so that a slice is at least 2 MiB), so a run of
  consecutive pages is also contiguous memory and a parking area is simply the top N pages of the pool
  (171 pages for NVFP4). Parking a request moves the KV pages that occupy the area elsewhere (one 2D copy per
  page, then the page references are rewritten); if there is no room, it evicts prefix-cache checkpoints, and
  failing that it raises `KvFull` without changing anything. The kernels see only page numbers: the page table
  holds `(p / G) * 32G + p % G`. As a result, starting with `MAX_SEQS=8` costs only the lanes' buffers, and one
  request still gets the full 262K context.
- **Host memory** (pinned): the embedding table (2.4 GiB), the prefix-cache checkpoints (`STATE_HOST_CACHE`,
  4 GiB), the vision tower (0.9 GiB) and the image embeddings.

`tools/model_budget.py` computes the plan for other settings without a GPU (and the decode roofline).

## 6. A decode step

Decode reads all the weights once per step, so the step time is set by bandwidth: 14.1 GiB of weights in `gdnfp4`
(15.2 GB, 8.9 ms at the measured 1.70 TB/s), plus the GDN state and the KV cache. Without speculative decoding one
request decodes at 96.6 tok/s at a short context.

- **Skinny GEMMs** (`skinny.cu`): the decode and verify GEMMs have 1 to 64 rows. They use `mma.sync` with the
  weights as the A operand (16 output channels per fragment) and the activations as B (8 tokens per fragment),
  a cp.async ring of 3 stages per warp with `ldmatrix`, and split-K so that 170 SMs stay busy. At 1 to 8 rows the
  loads of the activations are 4-byte, at 16 to 32 rows 16-byte with a transpose in registers (P6-11), and at
  64 rows the activations are staged once per CTA in shared memory (P7-4). When the outputs are split along K,
  a separate kernel sums the partials (P7-4) from two rows up. CUTLASS's 128 x 128 tiles reached 0.08 to
  0.9 TB/s on these shapes; the skinny GEMMs reach 1.3 to 1.67 TB/s.
- **One CUDA graph per step, chained with PDL.** Every kernel triggers its dependents first and then waits for its
  predecessor (`griddepcontrol`), so the next GEMM is resident and prefetches its weights into L2 while the
  previous kernel finishes. Loads that do not depend on the previous kernel (weights, the GDN state, positions,
  RoPE tables) are issued before the wait. Positions and the sampling inputs live on the device, so the graph does
  not change between steps.
- **Fused small kernels**: RMSNorm with the activation quantization, residual adds in GEMM epilogues, q / k norm
  with RoPE and the KV quantization and page write, the attention gate with quantization, GDN's convolution, gates
  and recurrent update in one kernel.
- **Decode / verify attention** (`attention.cu`): split-KV over fixed 32-key chunks (42 splits, one wave), so
  every query row (all tokens of a step x 6 query heads per KV head) reads the KV once. Producer warps copy chunks
  with cp.async and convert FP8 / NVFP4 to the MMA type in shared memory; compute warps pack the (token, head)
  columns into n8 tiles (P4-13, P4-14). With the NVFP4 KV cache the MMAs are block-scaled FP8 (P7-7): the verify
  step is no longer MMA-bound, and a 131K-token verify layer takes 130 instead of 200 us. Because the split of the
  keys does not depend on the number of rows, a token's row is bit-identical whether it is decoded alone or
  verified with seven others.
- **GDN decode and verify** share one kernel (a cluster of 4 CTAs per head). The verify step does not write the
  state; see [§7](#7-speculative-decoding).
- **Host interaction**: the inputs and outputs of a step (token ids, random numbers, accepted tokens) are in mapped
  pinned memory. The engine queues the next speculative step before the host has read the current one
  (`Q27_SPEC_PIPELINE`); if the host decides not to continue, the queued step is discarded at the cost of at most
  one step.

## 7. Speculative decoding

The draft is [DFlash2](https://huggingface.co/incoai/Qwen3.8-27B-DFlash2): five non-causal "block diffusion"
layers that read the target's hidden states from five layers, attend to the last 2,048 tokens and propose 7 tokens
in one pass; a low-rank selector chooses among the top-16 candidates at each position. One step drafts 7 tokens
from an anchor and verifies all 8 rows in one forward of the target.

- **Draft in NVFP4.** Its 36 linear layers are quantized at load time (1.2 GiB instead of 3.6). Acceptance is the
  same as with BF16 or FP8 weights and the step is 0.35 ms faster.
- **Draft vocabulary.** The selector's codebook and the draft's lm_head are restricted to 128K token ids
  (`bench/draft-vocab/vocab-128k.txt`), counted from the model's own outputs in Japanese, English and code
  (`tools/draft_vocab.py`). A token outside it can never be proposed, which only lowers acceptance; the output
  distribution is unchanged.
- **Chain, not tree.** Trees of 2 or 4 paths accept 4 to 23% more tokens per step but make the step that much
  longer; lookup drafting (copying from the prompt) adds nothing to an 8-row chain. Both exist behind switches
  (`Q27_SPEC_PATHS`, `Q27_LOOKUP`); see [decisions.md](decisions.md#speculative-decoding).
- **GDN without state copies.** Verifying 8 rows of a recurrent layer usually needs the state after each row
  (8 state copies per request, 0.9 GiB with DFlash2 in other engines). Here the verify step reads the committed
  state and writes nothing. It saves the small per-token inputs of the recurrence (k, v, beta, g; a few KB per
  token and layer), and the next step's GDN kernel first replays the accepted tokens onto the state, in the same
  kernel and before its PDL wait. One state slot per request is enough.
- **Lossless sampling.** Acceptance is rejection sampling against the target's distribution with the request's
  own temperature, top-k and top-p, so the output has the target's distribution exactly (checked with chi-square
  tests in `scripts/spec_sampling_test.py`). With greedy decoding, speculative and plain decoding give identical
  tokens (`scripts/bench_spec.py --check-greedy`).
- **Stops at EOS and `max_tokens`.** A step never commits tokens past them, so the end of a generation is a clean
  prefix of the next turn's prompt ([§9](#9-prefix-cache)).

Acceptance depends on the text: about 6 tokens per step on JSON, 4.5 to 5 on code, 3.5 on English prose and less
on Japanese prose (a property of the draft model, also seen in other engines).

## 8. Prefill

Prefill runs in chunks of 2,048 tokens through the same weights.

- **GEMMs** (`gemm_mma.cu`, P5-12): the engine's own kernels on `mma.sync` with 128 x 128 tiles, TMA loads into a
  ring of shared-memory stages filled by a producer warp, 8 MMA warps and TMA stores (or TMA reduce-adds for the
  residual) in the epilogue. NVFP4 reads the swizzled block scales in place; E4M3 runs the block-scaled MXFP8 MMA
  with unit scales (P5-11: bit-identical to the plain FP8 MMA and twice as fast on this GPU); BF16 (draft context,
  vision) any N and K. The MLP's gate | up GEMM writes silu(gate) * up NVFP4-quantized from its epilogue. Results
  are bit-identical to those of CUTLASS's SM120 kernels, which they replaced: at 400 W they were 2.5-7% faster on
  most prefill shapes (gdn z 1.8-2.7% and BF16 0.5-1% slower) and the prefill 0.6% / 0.0% faster at 19K / 73K. The K
  loop keeps its stage arithmetic on the uniform datapath, the producer warp issues its loads from one elected lane
  without a branch, and the tiles go N fastest unless the weights exceed L2. The GEMMs run at the power limit (P5-3), so instructions and bytes, not stalls, set their speed.
- **Attention** (`attention.cu`, flash attention on tensor cores, `Q27_ATTN_FP8`):
  - FP8 KV cache (mode 2): Q K^T and P V on FP8 MMAs; K and V as stored, Q with a per-row scale, P scaled by 256
    into E4M3. Exponentials in f16x2 with base 2, masking only on the diagonal chunk (P5-10).
  - NVFP4 KV cache (mode 3, default): Q K^T on the native FP4 MMA with K as stored and Q quantized to NVFP4, and
    P V in FP8. V is converted from NVFP4 to E4M3 once per layer and call into a staging buffer (1 KiB per token
    of context), instead of once per CTA (P7-8).
  - Tiles follow absolute positions (64-token blocks, blocks ending past 1,024 split their keys in two), so a row's
    result does not depend on where a chunk starts.
- **GDN**: a chunked delta rule with chunks of 64 (WY representation: the per-chunk matrices in FP32 by forward
  substitution, the inter-chunk state update on BF16 MMAs; P5-2). Chunk boundaries are fixed at multiples of 64
  absolute positions, and so are the prefill chunk boundaries and the checkpoints. A prefill resumed from a
  checkpoint is therefore bit-identical to one that ran through.
- **Draft context**: only the last 2,048 tokens of a prompt go through the draft.
- **Logits** only for the last token.

At 19K tokens prefill runs at 11,295 tok/s; at 194K, 5,875 tok/s, where attention takes most of the time.

## 9. Prefix cache

In a hybrid model, attention's KV cache can be reused from any position, but the GDN state exists only where it
was saved. The cache is therefore a list of **checkpoints**: a token sequence `[0, pos)`, the KV pages that cover
it (reference-counted, shared between checkpoints of the same conversation) and a copy of the state (GDN in FP32,
convolution, the draft's context ring: 189.6 MiB).

- **Where checkpoints are taken**: at message boundaries the server marks (the end of the first message, which
  holds the system prompt and tools; the starts of the last message and of the generation prompt), where the prompt diverges from the previous
  one, every `CHECKPOINT_INTERVAL` (8,192) tokens inside a long prompt (the last four), and at the end of each
  generation. Positions are rounded down to multiples of 64 ([§8](#8-prefill)).
- **Where they are kept**: in pinned host memory (`STATE_HOST_CACHE`, 4 GiB = 21 checkpoints). Saving copies the
  state to one staging buffer on the GPU and then to the host on a copy stream; restoring copies from the host
  (4 to 6 ms). Eviction is LRU, and the KV pool also reclaims pages from the least recently used checkpoints.
- **Lookup** is a linear scan with `memcmp` over the token sequences; there are at most a few dozen checkpoints.
- **Exactness**: the state is kept in FP32 and the boundaries follow the GDN chunks, so a request resumed from a
  checkpoint produces exactly the tokens it would produce without the cache (`scripts/prefix_cache_test.py`).

An agent's turn after a 20K-token conversation reaches its first token in 0.19 s: everything but the new tool
output and user message comes from the cache.

## 10. Scheduler and concurrency

Up to `MAX_SEQS` (8) requests are in the engine at once. One of them owns the "running" state buffers; the others
are parked ([§5](#5-memory-plan)). KV pages stay where they are; only the state moves, by an in-place swap with a
parking area (about 1 ms).

- **Batched decode** (P6-10): all requests in decode advance in one speculative step. Request z is lane z: its
  8 verify rows are rows 8z to 8z + 7 of every row buffer, so the GEMMs have up to 64 rows and read the weights
  once. Attention, GDN, the draft and the acceptance run as kernels with the lane as a grid dimension; each lane's
  state is passed by pointer (the running buffers or a parking area), so nothing is copied. Each request's tokens
  are bit-identical to running it alone (`scripts/batch_test.py check`), and the kernels of a single request are
  unchanged down to the SASS.
- **Prefill and decode interleave by time**: prefills go first, shortest remaining first (so an agent's short warm
  turn is not stuck behind a long cold prompt), one chunk at a time, and every second of prefill gives the decoding
  requests `DECODE_SHARE` (0.5) seconds.
- **Admission** (P7-14): a waiting request starts only if fewer than `target_seqs` requests run (settable through
  `/admin/concurrency`), the KV pool has room for every running prompt plus `KV_RESERVE_TOKENS` (4,096) of output
  each and one more parking area (`ADMIT_KV`), and, if `MIN_SEQ_TPS` is set, one more request would not slow each
  below it (measured per number of requests). Requests start in arrival order.
- **When the KV pool runs out anyway**, the most recent request is suspended: its state becomes a checkpoint, its
  pages go back, and it returns to the head of the queue to resume from the checkpoint once there is room.
- **Requests sharing a prefix** that arrive together: the later one waits until the first has passed the shared
  part, then hits the cache.

A step takes 12.4 / 13.8 / 16.4 / 22.0 ms for 1 / 2 / 4 / 8 requests at short contexts, which is why eight
requests decode about 4.5 times as many tokens per second as one.

## 11. Vision tower

`VISION=host` (the default) keeps the vision tower's BF16 weights (0.86 GB) in pinned host memory. The ViT runs
stage by stage (patch embedding, 27 blocks, the merger), with each stage's weight matrices copied into one of two
staging buffers on a second stream while the previous stage computes. The staging buffers and the ViT's
activations reuse the prefill arena, so images cost almost no VRAM (192 tokens of KV pool). A 1024 x 1024 image
takes 47 ms (43 ms with the weights in VRAM, `VISION=bf16`); outputs are bit-identical between the two.
`VISION=fp8` keeps E4M3 weights (per-row scales) in VRAM: 0.43 GiB, slightly less accurate, faster only for small
images.

- The image processor (resizing, patching) is the checkpoint's own in the server, so the inputs match the
  reference.
- Image tokens enter the prompt as out-of-vocabulary ids derived from the image's content. The prefix cache, which
  compares token ids, therefore distinguishes images without any change and skips encoding images that are already
  in a cached prefix. The embeddings are written to a pinned host table that the embedding gather reads.
- M-RoPE: rows of an image get (t, h, w) positions; after the last image, positions continue as position + delta
  (one device integer), so decode and the speculative graphs are unchanged, and text-only requests are
  bit-identical to the text path.

## 12. API server

`frontend/server.py` (asyncio, standard library HTTP) implements OpenAI chat completions, completions and the
Responses API (for Codex), and Anthropic messages and count_tokens (for Claude Code), all with streaming.

- The chat template is the checkpoint's Jinja template; reasoning (`<think>`) and tool calls (Qwen's XML format)
  are parsed by one state machine for streaming and non-streaming responses, and tool calls stream one by one.
- Anthropic `thinking` is honored; with `display: omitted`, the reasoning travels in the signature so that the next
  turn's prompt (and the prefix cache) still has it.
- The server marks message boundaries for the prefix cache and reports, per request, how many tokens came from it.
- `/metrics` exposes Prometheus metrics; `/admin/concurrency` reads and sets the admission settings. There is no
  authentication: the server is meant to listen on 127.0.0.1 for one user.
- `/ui` is a stats page (`frontend/stats.py`, `frontend/ui/index.html`): the engine process sends a snapshot of its
  requests, KV pool and device memory every 0.5 s while it works and its memory plan once at startup
  (`q27_memory_report`); the server adds NVML samples and a log of requests and engine events, and streams them to
  the page as server-sent events. The page (and `POST /admin/engine`) also stops, starts and restarts the engine
  process: it holds all the VRAM, the server none.
- Start settings (`frontend/launch.py`): an engine can start with other weights, KV type, draft, vision tower,
  `MAX_SEQS`, `MAX_TOKENS` or VRAM budget. What each leaves for the context comes from the engine's own memory
  planner: `q27_plan` runs the constructor's planning (the pools, `MAX_TOKENS_FIT`, the KV pool) for a given free
  and total device memory and stops before the first CUDA call, so the prediction is the startup log's numbers
  exactly. A helper process runs the plans, since each sets the environment the engine reads.
- Loading (`engine/src/prefetch.cc`): the loaders plan the byte ranges they read, `Q27_LOAD_THREADS` threads read
  them ahead into pinned slots (the embedding straight into its pinned table), and the loaders upload them in plan
  order as before. The prefix cache's pinned state slots (4 GiB) are allocated after startup, one by one while the
  engine is idle (`q27_cache_grow`) or when a checkpoint needs one.

## 13. Correctness

The engine is checked against `reference/`, a PyTorch implementation of the model that dequantizes the checkpoint
and computes plainly.

- **Kernels.** Unit tests (`engine/tests/`) compare every kernel with PyTorch, including boundary cases of pool
  sizes and offsets. The reference model's numerics follow transformers' implementation of the architecture, and
  its activation quantization can be switched off (`golden.py --no-act-quant`) to separate logic errors from
  rounding.
- **W4A4 logits are not reproducible bit for bit across implementations.** A different summation order flips the
  rounding of an FP4 or FP8 activation, and 64 layers amplify it. The reference model itself, run with an FP8
  instead of a BF16 KV cache, differs from itself by a mean KL of 0.036 and 89% top-1 agreement over the golden
  positions. The engine is accepted when its distance to the reference is of the same size: mean KL <= 0.05, top-1
  >= 88%, positions with KL > 0.5 <= 1% (`reference/golden.py`, [benchmarks.md](benchmarks.md)).
- **Invariants that are tested bit for bit** (they keep the tests simple and catch logic errors that the noise
  above would hide):
  - a verify row equals the decode of the same token (speculative and plain greedy decoding agree),
  - a request in a batch equals the request alone,
  - a prefill resumed from a checkpoint equals the prefill that ran through, at any 64-aligned boundary,
  - parking, swapping and moving KV pages change nothing,
  - `VISION=host` equals `VISION=bf16`.

  One caveat: the skinny GEMM splits K differently for one row than for several (it is faster), so a decode row
  and a verify row of the same token can differ in the last bit. Tests that compare across row counts set
  `Q27_SKINNY_EXACT=1`; the verify rows of a step and the rows of a batch always use the same split.
- **Speculative sampling** is tested statistically (`scripts/spec_sampling_test.py`): the token frequencies with
  and without speculation agree (chi-square tests over 2,000 samples each).
- **Quality options** (weights, KV cache type, attention numerics) are judged on the BF16 original's distributions
  (golden), GSM8K, BFCL, needle retrieval, a Japanese set and images ([benchmarks.md](benchmarks.md#quality)).
