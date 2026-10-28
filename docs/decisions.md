# Decisions

Why the engine is built the way it is: each choice with the measurement behind it, and the options that were tried
and dropped. [architecture.md](architecture.md) describes the result.

Task IDs such as `P6-10` appear in code comments; each one has an entry below. The first number is the phase of
development:

| Phase | Work |
| --- | --- |
| 0 | Measurements of the GPU (bandwidth, MMA peaks, PCIe, the WSL2 memory limit), the reference model, golden recordings |
| 1 | A working engine: loader, memory plan, straightforward kernels, decode and prefill without speculation |
| 2 | Decode speed: skinny GEMMs, fused kernels, CUDA graphs and PDL |
| 3 | Speculative decoding with DFlash2 |
| 4 | Memory: prefix cache, NVFP4 KV cache, vision tower |
| 5 | Prefill speed |
| 6 | Serving: API server, scheduler, several requests at once |
| 7 | Eight concurrent requests |
| Q | Weights with more NVFP4 layers (`gdnfp4`) |

Unless noted, numbers are at a 400 W power limit with the configuration of the time; [benchmarks.md](benchmarks.md)
has the current ones. "Bit-identical" means the outputs were compared bit for bit, not within a tolerance.

Contents: [Foundations](#foundations) · [Quality gates](#quality-gates) · [Decode](#decode) · [Speculative decoding](#speculative-decoding) ·
[Prefix cache and KV pages](#prefix-cache-and-kv-pages) · [KV cache precision](#kv-cache-precision) ·
[Prefill](#prefill) · [Vision](#vision) · [Weights](#weights) · [Serving and concurrency](#serving-and-concurrency) ·
[Not done](#not-done)

## Foundations

- **A new engine rather than patches to vLLM or SGLang.** The gains come from things a general engine cannot do
  for one model: one memory plan with separate pools, GDN state handling specific to speculative decoding,
  kernels shaped for this model's matrices on this GPU. Patching would have meant following those engines'
  abstractions (paged state, a common page size, profiling-based memory sizing) that cost the VRAM this project
  wanted back.
- **No PyTorch in the engine** ([architecture §3](architecture.md#3-processes)): a small CUDA context, no caching
  allocator, a memory plan that is the same at every start.
- **Straightforward kernels first** (Phase 1). Attention and GDN were written from scratch instead of ported from
  FlashInfer / FLA, which assume Python and Triton. Prefill GEMMs use CUTLASS's SM120 kernels from the start.
- **Correctness by noise floor, logic by bit identity.** W4A4 logits differ between any two implementations
  ([architecture §13](architecture.md#13-correctness)), so the engine is judged by being as close to the reference
  as the reference is to itself with another KV cache type (mean KL <= 0.05, top-1 >= 88%, KL > 0.5 at <= 1% of
  positions). Everything that should not change the numbers (speculation, batching, the prefix cache, parking) is
  built to be bit-identical, so that it can be tested exactly.
- **400 W.** All targets and measurements are at a 400 W power limit, below the card's 575 W default; the engine
  and the scripts never change power settings.

## Quality gates

Options that change the numerics (weights, KV cache type, attention precision) become defaults only after these
tests, run through the server against the configuration they would replace. Speed is never traded for a failed
gate without an explicit decision, recorded below with its numbers.

- **Golden** (`reference/golden.py`): KL to the BF16 original's next-token distributions over 35,328 positions must
  not get much worse.
- **Needle retrieval** (`scripts/needle_test.py`): 16 facts at 190K tokens all found; at 256K (four seeds, 64 facts) within a fact or two of the configuration it replaces.
- **GSM8K** (`scripts/quality_gsm8k.py`, 300 fixed questions, greedy, thinking off): within 1 point.
- **Tool calls** (`scripts/quality_tools.py`, BFCL v3 non-live AST, 1,000 questions): no fewer right, with the
  parallel categories looked at separately, since that is where quantization showed first. Differences of a few
  questions are within the run-to-run spread at temperature 0.7, so close calls get several seeds and a rank test.
- **Japanese** (`scripts/quality_ja.py`): reading comprehension (JSQuAD, 150), translation both ways (FLORES-200,
  100 each) and summarization (XL-Sum, 60), greedy, scored against references (exact match / character F1, chrF,
  character ROUGE-L) and by an LLM judge from another model family (Gemma 4 26B-A4B: correct / wrong for reading,
  1 to 5 for the others). A configuration fails if any task's judge score drops by 3 or more points (on 0 to 100)
  with a 95% bootstrap interval that excludes 0. The threshold comes from measurement: two configurations that
  should be equivalent differed by at most 2.3 points, while sampling at temperature 1.0 instead of greedy dropped
  translation and summaries by 4 to 9 points. The reference metrics move less than the judge and are shown
  alongside.
- **Reasoning loops** (for weights): thinking responses at low effort and temperature 0.6 must not loop more often.

## Decode

- **Skinny GEMMs for decode, CUTLASS for prefill** (Phase 2). CUTLASS's 128 x 128 tiles reached 0.08 to 0.9 TB/s
  at one row; the skinny GEMM (`mma.sync`, weights as A, a 16-byte cp.async ring per warp with `ldmatrix`) reaches
  1.3 to 1.67 TB/s. Plain 4-byte loads reached only 0.4 to 0.74 TB/s. 8 warps x 3 stages (2 CTAs per SM) was the
  best configuration; split-K is the smallest count above 2 x 170 CTAs.
- **CUDA graphs with PDL, trigger first.** One decode step went from 33.5 ms (eager) to 11.9 ms: the graph removed
  1.5 ms of launch overhead, PDL with an L2 prefetch of the next weights about 0.7 ms. Each kernel triggers its
  dependents at its start and then waits; triggering after the wait kept a GEMM from starting until the small
  kernel between two GEMMs ran. Loads that do not depend on the previous kernel (weights, states, positions) are
  issued before the wait: the memory queues are so full during decode that a dependent round trip takes several
  microseconds (-0.1 ms for GDN and norms, -0.2 ms for attention).
- **Tried and dropped** (same-process A/B, differences within +-0.1 ms count as none): split-K reduction through
  thread-block clusters (+0.13 to 0.5 ms), delaying the GDN state write to the next step (the next GEMM slows by
  the same amount), computing RMSNorm / SiLU inside the next GEMM's loop (+0.13 to 0.25 ms), L2 prefetch of the
  NVFP4 scales, continuous L2 prefetch, other prefetch depths and wave counts, spinning on the sample result
  instead of synchronizing.
- **Fused projections only where the row count allows.** Fusing `gate | up`, `qkv | z` and `q | k | v` saves
  launches, but changes the GEMM shape and therefore the split-K partition. Decode and verify (up to 8 rows per
  request) use the fused shapes so that a verify row equals a decode row.
- **P4-13: decode attention with producer warps.** Before, one CTA ran wait, convert, compute per chunk in series
  (about 1.8 us per chunk), and one warp used only one SM sub-partition's tensor core. Producer warps now copy and
  convert (FP8 / NVFP4 to BF16 / F16) while compute warps run the MMAs, and one token is computed by 4 warps.
  One 131K layer (with the combine): NVFP4 KV 169 -> 107 us (the read alone takes 97), FP8 178 -> 168 us. Decode
  at 131K: 69.1 -> 74.5 tok/s with NVFP4 KV. Bit-identical to before.
- **P4-14: fewer MMAs in the verify attention.** The 48 (token, head) columns of an 8-token verify are packed into
  6 n8 tiles (tiles 0 to 3 one warp each, the rest in groups of 4 warps), instead of one tile per token with 6 of
  8 columns used: 512 -> 384 MMAs per chunk. One verify layer at 32K: FP8 KV 57.8 -> 50.3 us, NVFP4 67.2 ->
  55.0 us. Bit-identical. The group's tile count must be a compile-time constant (a runtime value cost 5 us at
  32K for one token).
- **P7-7: FP8 MMAs in the verify attention with the NVFP4 KV cache.** The BF16 / F16 MMA (253 TFLOPS with FP32
  accumulation) made the 8-token verify MMA-bound. Block-scaled FP8 (`mxf8f6f4`, 1,007 TFLOPS) with Q per-row
  scaled, P in E4M3 and NVFP4 keys rounded to E4M3 (the prefill's mode-2 numerics): one 131K verify layer
  200 -> 130 us, the speculative step at 131K 17.34 -> 15.97 ms (-7.9%), eight requests of 32K each -8%. Plain
  decode +0.2 to 1.4%. Golden on the decode path against the BF16 original: KL 0.0344 -> 0.0359; needle retrieval
  at 256K 56 -> 57 of 64. With the FP8 KV cache the verify is already near its read time and gains 0 to 2%, so it
  is not the default there (`Q27_ATTN_TC_F8=1`). Rows of one token stay bit-identical across 1 to 8 tokens.
- **P5-11 for decode: not adopted.** The block-scaled FP8 MMA is twice as fast for prefill, but decode is
  bandwidth-bound and gained nothing, so the skinny GEMMs kept the plain FP8 MMA.

## Speculative decoding

- **DFlash2 with an 8-row chain** (Phase 3). The chain verify was built first, with every verify row
  bit-identical to the decode of the same token, so greedy speculative decoding must reproduce plain decoding
  exactly (a strict test). Short-context decode came out 1.36 to 1.62 times SGLang's DFlash2 on the same GPU.
- **GDN verify writes no state; the next step replays the accepted tokens** (Phase 3). Engines that keep one state
  per verified position need 8 state copies per request with DFlash2 (0.9 GiB per request; at most 5 concurrent
  requests in one measurement). Replay needs one state per request and no separate commit pass. A race was found
  on the way: the replay's convolution-state write-back must wait until all blocks have read the old state.
- **Draft in NVFP4.** Greedy acceptance was the same for BF16, FP8 and NVFP4 draft weights (code 4.72, prose 3.81;
  JSON 7.08 for NVFP4 against 6.89), and the step took 16.2 / 15.1 / 14.7 ms.
- **Draft vocabulary of 128K tokens.** 64K lowered acceptance on JSON; 96K was 0.05 ms faster than 128K but missed
  more tokens. The vocabulary is counted from the model's own outputs in Japanese, English and code: a
  vocabulary built from English, Danish and code covered 6% of Chinese output in another project and collapsed
  acceptance.
- **Tree verification: not the default.** Two or four paths (merged into a trie, recursive rejection sampling)
  accept 4 to 23% more tokens per step but the step grows from 14.1 to 15.7 (2 paths) and about 20 ms (4 paths):
  with sampling, 2 paths were +4% on code, +6% on prose and -4% on JSON, about break-even. Kept as
  `Q27_SPEC_PATHS`.
- **Lookup drafting: not the default.** On a prompt that rewrites a file, DFlash2 alone already accepted 7.91 of 8
  (it sees the context through the target's hidden states); on JSON, lookup followed repeated keys and lowered
  acceptance from 6.24 to about 6.0. Kept as `Q27_LOOKUP`.
- **No MTP head.** In SGLang on the same GPU, MTP beat DFlash2 only on Japanese prose (+17%).
- **The draft's sampling temperature stays 1.0.** Drawing the draft's walk at 0.7, 0.4 or argmax did not raise
  acceptance beyond noise (argmax lowered it).
- **P3-25: the next step is queued before the host reads the current one.** -0.07 ms per step; a discarded step
  costs at most one step.
- **Steps stop at EOS and `max_tokens`**, so the end of a generation is exactly the start of the next turn's
  prompt, which the prefix cache needs. The output distribution is unchanged.
- **Acceptance is measured with the default sampling.** Greedy acceptance is higher than what users get, and
  random-token benchmarks say nothing about speculative decoding. Sampled acceptance varies by +-5% between runs,
  so configurations are compared by step time and greedy acceptance on the same outputs.

## Prefix cache and KV pages

- **P4-3: one KV page pool with a free list and reference counts.** Pages of 64 tokens; page 0 is a scratch page
  for unallocated entries; the device page table is updated from a host copy only as far as needed (8 pages
  ahead). `MAX_TOKENS` limits one sequence; the pool is everything the budget leaves.
- **Checkpoints instead of a radix tree** (Phase 4). GDN can resume only from a saved state, and KV pages after
  the last checkpoint are recomputed anyway on resume, so pages need only be owned (reference-counted) by
  checkpoints. A linear scan over a few dozen checkpoints replaces the tree.
- **Checkpoints in host memory only.** Restoring from pinned host memory takes 4 to 6 ms, small next to any
  prefill, so there is no GPU tier: the VRAM goes to the KV pool.
- **States kept in FP32.** A BF16 state halves the checkpoint (and SGLang measured no quality loss), but FP32 keeps
  a resumed request bit-identical to an uncached one, which makes the cache testable exactly.
- **P4-6: where checkpoints are taken.** Prefill chunks are cut at the checkpoint positions the caller gives
  (message boundaries, the end of the last generation) and every 8K tokens, keeping the last four interval
  checkpoints and skipping one within 64 tokens of another. The draft's context ring is filled with the 2,048
  tokens before each checkpoint. `scripts/prefix_cache_test.py` checks 22 cases (plain and speculative, resuming
  from the end of a generation, a shared prefix or an interval checkpoint, EOS and `max_tokens`) against the same
  computation without the cache: greedy outputs identical.
- **The hit rate is measured on reusable tokens.** An agent benchmark adds new tool output every turn, so even a
  perfect cache hits only about 85% of all prompt tokens; the target (95% of reusable tokens) is met with 98 to
  100%.
- **P7-13: parking areas are KV pages while unused.** Before, the `MAX_SEQS - 1` parking areas (189.6 MiB each)
  were a separate pool: with `MAX_SEQS=8` they took 1.30 GiB (75,712 NVFP4 tokens) even with one request running,
  and `MAX_TOKENS` had to be lowered on NVIDIA's weights with the FP8 KV cache. Three layouts were compared:
  - spreading each area over the 32 per-layer KV arrays: the kernels need contiguous runs of 3 to 4 MiB, so the
    area grows to 224 MiB (+18%), or the GDN kernels need split pointers;
  - **making the KV pool page-major** (chosen): a run of pages is contiguous, so an area is just N pages at the
    top of the pool with no waste and no kernel change beyond the page table's values;
  - shrinking the state (BF16): independent of the above and a quality question.

  One page per block (pure page-major) made decode at 131K 1.5% slower (one layer's K scattered at 1.125 MiB
  strides touches about 2,000 large pages per layer); blocks of G pages (G = 57 for NVFP4, a slice of at least
  2 MiB) are as fast as the old layout. Parking evacuates the area's pages (one 2D copy per page, references
  rewritten in running and parked sequences and checkpoints), evicts checkpoints if needed, and otherwise raises
  `KvFull` with nothing changed. KV pool with the defaults: 533,376 -> 605,504 tokens, and `MAX_SEQS=8` keeps the
  full 262K context. Bit-identical to before, also with `Q27_PAGES_HIGH=1`, which forces a move on every park.
  A bug found on the way: `seq_park` did not reset the running buffers, so a sequence started right after a park
  (without `prefix_begin`) inherited the old state.

## KV cache precision

- **P4-11: an NVFP4 KV cache** (`KV_DTYPE=nvfp4`; K and V both NVFP4 with 16-element blocks, 144-byte rows, global
  scale 1). Decode reads it exactly in F16; prefill originally rounded it to E4M3 for the FP8 attention.
  Capacity: 1.78 times FP8 for the same memory. At first it stayed an option: KL to the BF16 original 0.0401
  against 0.0369 for FP8, needle, GSM8K and BFCL equal, decode not faster at the time. A mixed FP8-K / NVFP4-V
  type was not built: NVFP4 keys were good enough.
- **P7-9: the NVFP4 KV cache became the default** (2026-10-11), after the prefill attention work below made it as
  fast as or faster than FP8. Quality with `gdnfp4` through the server:
  - Japanese set: passes (LLM-judged differences +0.0 / -1.5 / +1.5 / -0.8 points, every interval crossing 0).
  - BFCL parallel and parallel_multiple (400 questions, temperature 0.7, three seeds): FP8 317.3 right on average,
    NVFP4 306.0 (-11 questions); the six FP8 runs and three NVFP4 runs do not overlap (rank test p ~ 0.012).
    Nearly all of the loss is stopping a list of parallel calls early (missing calls 196 -> 225 over 1,200
    questions); wrong calls do not increase. About half comes from the NVFP4 cache itself (mode 2: -6.3) and half
    from mode 3 below.
  - The same questions after an agent's history of reading this repository's files (`quality_tools.py --context`,
    32K and 96K tokens): FP8 343.3 / 340.3, NVFP4 339.3 / 342.7, no difference; stopping early drops to a third
    in both.

  Since agents send long contexts, capacity (1.73 times FP8: 589,888 against 339,968 tokens at the time) and
  prefill speed won. `KV_DTYPE=fp8` remains.

## Prefill

- **P5-2: a chunked GDN kernel.** Delta rule in chunks of 64 (WY representation): one layer of 2,048 tokens
  2.73 -> 0.34 ms, a 32K prefill 6.09 -> 5.00 s. Chunk boundaries change the rounding, so they are fixed at
  multiples of 64 absolute positions, and prefill chunks and checkpoints follow; a resumed prefill recomputes at
  most 63 extra tokens and stays bit-identical.
- **P5-3: CUTLASS tiles unchanged.** FP4 GEMMs hold the SM clock at about 1.79 GHz at 400 W (power throttling);
  corrected for the clock they run at about 87% of the MMA peak. Of nine tile, schedule and split configurations
  (`micro/gemm_tune.cu`, removed with CUTLASS), only BF16 output was faster (2 to 9%), used where the next step needs no more (the MLP's
  gate / up, which only feed SiLU and FP4 quantization).
- **P5-4: chunks of 2,048 tokens.** At 32K, chunks of 2,048 / 4,096 / 8,192 took 4.66 / 4.60 / 4.54 s: 2.5% for 4
  times the activation memory.
- **FP8 prefill attention** (`Q27_ATTN_FP8=2`, the FP8 KV default). Q K^T and P V on FP8 MMAs: one 32K layer
  81.4 -> 47.1 ms, a 194K prefill 65 -> 45 s. Golden and needle retrieval (256K: 57 of 64 against 58 for BF16)
  indistinguishable from the BF16 attention. Subtracting K's mean (SageAttention) is unnecessary: K is already
  stored in FP8 with scale 1.
- **Prefill is energy-bound.** At 32K, of 1,529 J: FP8 GEMMs 505 J, FP4 GEMMs 465 J, attention 420 J, the rest
  231 J. This is why the remaining gains are in FP4 / FP8 paths and in less work, not in tiling.
- **P5-10: less energy in attention without changing results.** Exponentials in f16x2 with base 2, masking only
  the diagonal chunk: one 32K layer 19.6 -> 17.9 J (-9%). Rescaling the output only when the running max moves
  saved no energy.
- **P5-11: block-scaled FP8 MMAs with unit scales.** On the RTX 5090 the plain FP8 MMA with FP32 accumulation runs
  at half rate (507 against 1,007 TFLOPS); the block-scaled `mxf8f6f4` with scales of 1.0 gives bit-identical
  results at full rate (a technique from [NInfer](https://github.com/Neroued/ninfer)). Prefill FP8 GEMMs use
  CUTLASS's MXFP8 kernel, and the FP8 attention the same instruction: 19K / 73K / 194K 9,319 -> 9,726,
  7,080 -> 7,600, 4,584 -> 5,113 tok/s.
- **Attention tiles on absolute positions.** A 2,048-token chunk made 2.26 waves; blocks that end past 1,024 split
  their keys in two, which depends only on the block, so a row's result is the same wherever a chunk starts.
- **P7-8: prefill attention with the NVFP4 KV cache.**
  - (A) Each CTA converted the keys it read from NVFP4 to E4M3, so a key was converted (position / 64) x 3 times
    and a layer was 15 to 16% slower than with FP8 KV. Converting the context once per layer and call into a
    staging buffer and running the FP8 KV kernel made it as fast as FP8 KV, bit-identically (32K / 131K layer
    39.4 / 642 -> 34.5 / 566 ms).
  - (B, default) Q K^T on the native FP4 MMA (`mxf4nvf4`, K as stored, Q quantized to NVFP4), P V in FP8 with only
    V staged (1 KiB per token): one 131K layer 542 -> 487 ms (-11% energy); prefill 19K / 73K / 194K 10,992 /
    8,402 / 5,436 -> 11,040 / 8,734 / 5,857 tok/s, faster than FP8 KV by 1 / 6 / 9%. Quality: KL to the BF16
    original 0.0401 -> 0.0416, needle at 256K 57 -> 56 of 64, GSM8K 95.7 -> 95.3%, BFCL 89.3 -> 88.9% (sign test on
    the questions only one got right: p ~ 0.17). P V cannot use FP4: V's scales run along the head dimension, not
    along the keys the MMA reduces over.
  - (C, dropped) A softmax per 64 keys (half the output rescales): -0.5 to 1.5%, within noise. Three heads per CTA
    ran out of registers.
- **P5-12: the MLP's gate | up stored interleaved (blocks of 64 rows, gate and up in turn).** The fused prefill
  GEMM (`gemm_fp4_glu_mma`, `Q27_MLP_FUSED=1`, the default since 2026-10-28: gate | up, then silu(gate) * up NVFP4-quantized in the epilogue)
  needs the gate and the up rows of the same outputs in one tile. With the two stored apart, a 128-row tile took
  half of a gate and half of an up scale atom (the NVFP4 scale layout groups 128 rows), 50% more B-scale bytes per
  K step: the interleaved layout makes it one atom, -3.8% time and about -2% energy per call. The loader places
  the rows (no extra memory), the GEMMs on gate | up run with alpha 1 and the SiLU step applies the gate's and the
  up's. Decode and the fused prefill are bit-identical to before; the BF16 prefill (`Q27_MLP_FUSED=0`) now writes
  one BF16 gate | up (KL to the previous engine 0.048, to the BF16 original unchanged). With the fused path, prefill
  at 19K / 73K is 2.2 / 1.7% faster and uses 2.1% less energy per token than the BF16 path. Quality against it:
  GSM8K 95.3 / 95.3%, BFCL 88.9 / 89.1%, Japanese reference scores within their CIs (JSQuAD F1 +1.6), needles at
  256K 56 / 58 of 64, KL to the BF16 original 0.0470 / 0.0454.
- **P5-12: own GEMMs for every many-row GEMM; CUTLASS optional.** `gemm_mma.cu` now also runs E4M3 (the block-scaled
  MMA with unit scales) and BF16 (the draft's context, vision; any N and K through TMA's edge handling) with the same
  tiles and epilogue: the m16n8k32 E4M3 and m16n8k16 BF16 fragments hold the same bytes per K step as the NVFP4 k64
  ones. Results are bit-identical to CUTLASS. CUTLASS (SM120 3.x and the 2.x BF16 kernel) is built with
  `-DQ27_CUTLASS_GEMM=ON` (default; off for MSVC, which cannot compile CuTe) and chosen per call with
  `Q27_GEMM_CUTLASS` (default 1 in such builds). At 400 W the own E4M3 GEMM is 1.5-5.6% faster than CUTLASS's MXFP8
  one; the own NVFP4 and BF16 GEMMs are 1-3% slower: same L2 traffic and active cycles at a fixed clock, but more
  power per cycle, so a lower clock at the power limit. Two changes closed part of the gap: a suspend-time hint on
  the mbarrier waits (as CUTLASS's; spinning waits were ~20% of the instructions) and a 2 KiB epilogue staging per
  warp (one more ring stage). The fused MLP GEMM's epilogue dropped denormal handling it never needs (ftz exp2 and
  reciprocal, a branch-free division by 6; bit-identical): -28% of its instructions, -0.9% time.
  Later the rest of the gap was found in the K loop's integer instructions: the stage addresses and barrier phases ran
  on the vector datapath (~30 instructions per 32 MMAs), CUTLASS's on the uniform one. With the warp index made
  warp-uniform (`__shfl_sync`, as CUTLASS's `canonical_warp_idx_sync`), the empty-barrier arrive predicated instead of
  branched and only the per-lane offsets kept opaque to the compiler, the loop has no vector integer arithmetic left
  and the NVFP4 GEMMs lost 1.5-2% of their energy. The tile order then mattered by 2-5%: N fastest (the CTAs running
  at the same time share activation rows) beats M fastest except where the weights do not stay in L2 between the
  passes (the fused gate | up, 89 MB: M fastest there). At 400 W, against CUTLASS: NVFP4 gdn qkv / out / down -2.5 to
  -3.6%, gdn z +1.8%; E4M3 -4.3 to -7%; BF16 +0.5-1.1%; the fused MLP GEMM -3% against its previous version.
  Prefill with all own kernels: +0.6% / +0.0% at 19K / 73K.
- **P5-12: CUTLASS removed.** With the own GEMMs as fast as CUTLASS's in the prefill and bit-identical, `gemm.cu`,
  `gemm_bf16.cu`, the CMake option, the GEMM workspace, `Q27_GEMM_CUTLASS` / `Q27_GEMM_MXFP8` and CUTLASS in both
  images went; the engine builds with the CUDA toolkit alone (a precondition for MSVC, which cannot compile CuTe).
  Last tries on gdn z (N 6,144, the one shape still 1.8-2.7% slower; identical at a fixed clock, time, L2 bytes and
  instructions alike): one GEMM for qkv | z (slower than the two), as many tiles on every CTA (154 x 5), a dynamic
  tile counter, the producer's loads from an elected lane without the per-lane loop the lane-0 branch compiled to
  (14% of the instructions: kept, but no faster). As a pair qkv and z are 1.7% faster than CUTLASS's.
- **Not done: FP4 attention for P V (SageAttention3 style), a megakernel.** See [Not done](#not-done).

## Vision

- **P4-12: the vision tower** (`VISION=off / host / fp8 / bf16`): the ViT on BF16 GEMMs (CUTLASS's until P5-12) with FP32
  residuals and LayerNorm, a non-causal flash attention per image (head dimension 72 padded to 80 in shared
  memory), M-RoPE. One image per call, so no variable-length attention is needed. Its workspace (305 MiB for a
  4,096-token image) reuses the prefill activations.
- **Image tokens as content-derived out-of-vocabulary ids.** The prefix cache compares token ids, so it tells
  images apart without changes, and a cached prefix never re-encodes its images (`scripts/vision_api_test.py`:
  the second turn does not encode the image again; a different image is not reused).
- **The checkpoint's image processor, in the server.** A hand-written resize would differ in bicubic details and
  shift the inputs.
- **M-RoPE through a position delta.** Only the chunks up to the last image get explicit 3-axis positions; after
  that, decode and the speculative graphs add one device integer, so they are unchanged and text-only requests are
  bit-identical to before. The draft ignores the shift (it affects only acceptance).
- **P4-12d: images in the API.** OpenAI `image_url` parts as data URLs (and http(s) URLs), HTTP 400 with
  `VISION=off`, `IMAGE_MAX_PIXELS` (4,194,304: without a limit, a 16.7-megapixel image overflowed the activations in
  vLLM).
- **P4-12e: `host` as the default.** Weights stream from pinned host memory stage by stage into two staging
  buffers on a second stream: ViT time at 256^2 / 1024^2 / 2048^2 pixels 19.8 / 46.5 / 371 ms against 7.1 / 43.0 /
  367 ms for `bf16`, with bit-identical outputs and 192 tokens of KV pool instead of 23K. `fp8` (per-row E4M3
  scales) is 12 ms faster than `host` only for small images, costs 0.43 GiB and embeds with 4 to 4.8% rms error
  (finer scales still left 3.6%: the 3-bit mantissa dominates). On 70 synthetic questions with exact answers:
  `bf16` and `host` 67 (identical replies), `fp8` 68, vLLM 70; the differences are three close counting questions
  that the FP32 reference model also misses the same way.

## Weights

- **Quality gate for weight options** (before any default change), against NVIDIA's checkpoint: needle retrieval
  all correct, GSM8K and the Japanese set within 1 point, no fewer tool-call and agent successes, KL to the BF16
  original not much worse.
- **QUASAR (a public QAT checkpoint with every projection in NVFP4): an option only.** Prefill +18 to 26%, decode
  +15%, but BFCL parallel 79.0 -> 65.5% (it stops after the first call), needle at 256K 59 -> 53 of 64, KL 0.037 ->
  0.049. Available as `WEIGHTS=quasar`.
- **Own quantization of the projections (Phase Q).** Round-to-nearest with a searched block scale reached KL
  0.055 to 0.057; GPTQ with a Hessian from 397K calibration tokens run through NVIDIA's model reached 0.047. But
  with every projection in NVFP4, parallel tool calls dropped however they were made (GPTQ 73.5% against 79.0%).
  Keeping parts in FP8 located the cause in the **attention projections**, spread across all 16 layers rather
  than in any quarter of the model; the GDN projections in NVFP4 do not hurt tool calls.
- **`gdnfp4` (default since 2026-10-10): only the GDN projections in NVFP4.** It passes the gate: GSM8K 94.7%
  (NVIDIA's 94.3), Japanese passes, BFCL 90.5% with parallel 80.0% (90.1 / 79.0), needle at 190K 16 of 16 and at
  256K 58 of 64 (59), KL 0.041 (0.037), no reasoning loops in 200 thinking responses (none for NVIDIA's either).
  Speed: prefill +5 to 11%, decode without speculation +11%, the speculative step 13.79 -> 12.57 ms, KV pool
  +55K tokens: 60 to 80% of what every projection in NVFP4 would give. The attention projections are 23% of the
  projection weights.
- **Shared global scale for `qkv | z`.** vLLM runs the two as one GEMM with the larger global scale, so `gdnfp4`
  quantizes them with a common one; in vLLM it decodes 7 to 15% faster than NVIDIA's checkpoint with the same
  GSM8K and BFCL.
- **Other public NVFP4 checkpoints** keep their projections in FP8 and are no faster than NVIDIA's.

## Serving and concurrency

- **P6-1: several sequences by time-sharing first.** The running sequence's state swaps in place with a parking
  area (about 1 ms, bit-identical), prompts advance a chunk at a time, `KvFull` is a recoverable error, and graphs
  are cached per sampling setting. Existing kernels and graphs work unchanged. Time-sharing alone capped the sum
  of all requests at one request's speed (four agents: 197 tok/s of output, SGLang's batching 248), with warm
  first-token times on par (0.36 s). Piggybacking decode rows on prefill chunks (Sarathi) was not built.
- **The scheduler in Python, in its own process** (Phase 6) rather than C++ in the engine: its decisions are
  negligible next to a 14 ms step, and a crashed engine can be restarted behind a server that stays up. Prefills
  go shortest-remaining first; requests that share a prefix and arrive together wait for the first one (four
  agents' hit rate 94.1 -> 98.5%).
- **P6-7: soak test.** `scripts/soak_test.py`: concurrent agents, cancellations and long documents for 30
  minutes; 1,003 requests, no failures, constant VRAM, no leaked pages (later 1,662 requests with eight running).
- **P6-10: batched decode on top of time-sharing.** Requests are lanes; their states are passed by pointer, the
  GDN replay works on each lane's single state, and only chain verification is batched. A second state copy per
  lane would have cost about 14K tokens of KV pool and the single 262K request. The single-request kernels gained
  only trailing arguments and `if constexpr` branches, and their SASS was compared to make sure they did not
  change. Four requests: 19.7 ms per step and 880 tok/s together (2.2 times one); four agents' output 197 ->
  309 to 372 tok/s.
- **P6-11: wider loads in batched GEMMs.** At 32 rows most of the extra time was 4-byte activation loads; 16-byte
  loads with a quad transpose: four requests 19.75 -> 18.85 ms, bit-identical. For the 1,024-tile lm_head the old
  loads stay (250 -> 261 us otherwise). Wider split-K reductions per thread, a higher GDN occupancy and 2 CTAs per
  SM for its prep did not help.
- **Eight requests, not more** (Phase 7). Batch jobs (OCR, evaluations, independent tasks) benefit from eight; at
  16 the GDN states (189.6 MiB per request) would shrink the KV pool to about 190K tokens and the verify rows
  would become compute-bound. P7-2 confirmed eight lanes bit-identical to single requests.
- **P7-4: the eight-request step.**
  - 64-row GEMMs need 170 to 225 registers, one CTA per SM: activations are staged per CTA in shared memory
    (swizzled, `ldmatrix`): 29.15 -> 25.95 ms. Limiting registers to fit 2 CTAs spilled and was slower (30.7 ms
    for five requests, 38.5 for eight).
  - The split-K reduction moved to its own kernel from two rows up: the last CTA of each tile had kept 5120-wide
    GEMMs running 17 us longer. 1 / 2 / 4 / 8 requests 13.97 / 16.0 / 18.65 / 25.9 -> 13.84 / 15.48 / 18.14 /
    24.4 ms, bit-identical. At one row (plain decode) the last CTA stays faster (11.47 against 11.74 ms).
  - The draft's attention on tensor cores: on CUDA cores it was compute-bound (shuffles and exponentials per key,
    214 us per layer with eight requests); the step with eight requests 26.8 -> 26.0 ms. The draft's proposals change slightly
    (outputs remain lossless).
  - Through the server (reasoning prompt, 4,096 tokens): C=8 1,262 tok/s against NInfer's 1,302 at the time; with
    `gdnfp4` and the NVFP4 KV cache, 1,404 to 1,411.
  - Halving split-K made eight requests faster but four slower; the split cannot depend on the request count
    without breaking bit identity with single requests.
- **P7-11: step times by request count and context.** 1 / 2 / 4 / 8 requests: 12.40 / 13.76 / 16.39 / 21.99 ms at
  short contexts, 13.17 / 15.05 / 19.21 / 27.79 ms at 32K each (NVFP4 KV; FP8 KV 30.55 ms for eight, so NVFP4 is
  9% faster there), 13.73 / 16.16 / 21.45 / 32.36 ms at 64K each. The admission's step-time model
  (11.1 + 1.36 n ms) is fitted to these.
- **`MAX_TOKENS_FIT`.** Before P7-13, `MAX_SEQS=8` with 262K did not fit and the server kept failing to start; the
  server now lowers `MAX_TOKENS` to what fits (in multiples of 1,024) and reports it in `/v1/models`. The engine
  library still fails instead, so measurements never change conditions silently.
- **P7-14: deciding the number of running requests while running.** `MAX_SEQS=8` is the default since P7-13 made
  it nearly free (the lanes' fixed buffers, about 2.4% of the KV pool). A request waits instead of starting when
  the KV pool cannot hold the running prompts plus 4,096 tokens of output each: waiting wastes nothing, while
  suspending a half-computed request costs its recomputation. On a small KV pool (142K tokens, eight 12K prompts)
  this ran at most six at once with no suspensions, in the same total time. `MIN_SEQ_TPS` (off by default) holds
  back requests that would push per-request speed below a floor (with 200 tok/s: five requests at 212 tok/s
  each). The admin API has no authentication, like the rest of the server, which is for one local user.
  Found on the way: when the pool ran out, the scheduler suspended the latest request other than the one that ran
  out, and two requests could suspend each other thousands of times. It now suspends the latest request including
  itself, and a suspended request resumes only when the KV pool has room.

## Not done

- **A megakernel** (the whole step as one persistent kernel). After graphs and PDL, the non-GEMM kernels add
  about 1 ms to an 11 to 12 ms step, which bounds the gain.
- **Adaptive verify width, MTP, BF16 GDN states** (`GDN_STATE_DTYPE`): designed, not built (see above for MTP and
  the state precision).
- **FP4 for P V in prefill attention** (SageAttention3 style): changes quality; not tried.
- **Decode attention for several long requests at once.** Four requests with long contexts each are slower per
  request than the bandwidth allows ([benchmarks.md](benchmarks.md#known-weak-spots)).
- **More than eight requests**: needs the batched GEMMs beyond 64 rows.
