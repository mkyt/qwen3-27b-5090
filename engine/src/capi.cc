// C ABI of libq27.so: the engine (used by the Python frontend and golden.py) and thin synchronous
// wrappers of single kernels for the unit tests (q27t_*; pointers are device pointers from torch).
#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "common.h"
#include "kernels.h"
#include "draft.h"
#include "model.h"

using namespace q27;

namespace {

thread_local std::string g_error;

template <class F>
int guard(F&& f) {
  try {
    f();
    return 0;
  } catch (const KvFull& e) {  // (recoverable: the caller frees pages and retries)
    g_error = e.what();
    return -2;
  } catch (const std::exception& e) {
    g_error = e.what();
    return -1;
  }
}

float g_last_us = 0.f;

// The kernel wrappers launch on this stream (a real stream: PDL applies and launches can be captured).
cudaStream_t ts() {
  static cudaStream_t s = [] {
    cudaStream_t x;
    Q27_CUDA(cudaStreamCreateWithFlags(&x, cudaStreamNonBlocking));
    return x;
  }();
  return s;
}

// Q27T_REPEAT=n (read per call): after the normal launch, also capture n launches into a CUDA graph
// and time its replay (no host launch overhead; consecutive launches overlap through PDL as in the
// engine); the average per launch is kept for q27t_last_us.
template <class F>
int kernel(F&& f) {
  return guard([&] {
    // ts() is a non-blocking stream: without this, the kernel may run while the caller's torch
    // kernels (on the legacy default stream) are still writing its inputs.
    Q27_CUDA(cudaDeviceSynchronize());
    f();
    Q27_CUDA(cudaStreamSynchronize(ts()));
    const char* e = getenv("Q27T_REPEAT");
    const int rep = e ? atoi(e) : 0;
    if (rep > 0) {
      cudaGraph_t g;
      cudaGraphExec_t ge;
      Q27_CUDA(cudaStreamBeginCapture(ts(), cudaStreamCaptureModeThreadLocal));
      for (int i = 0; i < rep; i++) f();
      Q27_CUDA(cudaStreamEndCapture(ts(), &g));
      Q27_CUDA(cudaGraphInstantiate(&ge, g, 0));
      cudaEvent_t e0, e1;
      Q27_CUDA(cudaEventCreate(&e0));
      Q27_CUDA(cudaEventCreate(&e1));
      Q27_CUDA(cudaGraphLaunch(ge, ts()));  // warm-up (clocks)
      Q27_CUDA(cudaEventRecord(e0, ts()));
      Q27_CUDA(cudaGraphLaunch(ge, ts()));
      Q27_CUDA(cudaEventRecord(e1, ts()));
      Q27_CUDA(cudaEventSynchronize(e1));
      float ms = 0.f;
      Q27_CUDA(cudaEventElapsedTime(&ms, e0, e1));
      g_last_us = 1000.f * ms / rep;
      cudaEventDestroy(e0);
      cudaEventDestroy(e1);
      cudaGraphExecDestroy(ge);
      cudaGraphDestroy(g);
    }
    Q27_CUDA(cudaDeviceSynchronize());
  });
}

// A device copy of a position for the kernels that read it from device memory (one persistent slot,
// so that repeated launches can be captured).
struct DevInt {
  int32_t* p = nullptr;
  explicit DevInt(int v) {
    static int32_t* slot = [] {
      int32_t* x;
      Q27_CUDA(cudaMalloc(&x, 4));
      return x;
    }();
    p = slot;
    // On ts(): a cudaMemcpy from pageable memory may return before its DMA lands and is not ordered
    // with the non-blocking ts(), so the kernel could read the previous call's value. While capturing,
    // the slot already holds v (kernel() runs the same call uncaptured first).
    cudaStreamCaptureStatus cs;
    Q27_CUDA(cudaStreamIsCapturing(ts(), &cs));
    if (cs == cudaStreamCaptureStatusNone) Q27_CUDA(cudaMemcpyAsync(p, &v, 4, cudaMemcpyHostToDevice, ts()));
  }
};

// The engine's options: the arguments of q27_create and the environment (docs/configuration.md).
Options engine_options(const char* model_dir, int max_tokens, int prefill_chunk, int kv_fp8, const char* draft_dir) {
  Options o;
  o.model_dir = model_dir;
  if (draft_dir && *draft_dir) o.draft_dir = draft_dir;
  if (max_tokens > 0) o.max_tokens = max_tokens;
  if (prefill_chunk > 0) o.prefill_chunk = prefill_chunk;
  o.logits_rows = std::min(o.logits_rows, o.prefill_chunk);
  Q27_CHECK(kv_fp8 >= 0 && kv_fp8 <= 2, "kv type: 0 BF16, 1 FP8, 2 NVFP4");
  o.kv_type = KvType(kv_fp8);  // (kv_fp8: the KV type code; 1 / true stays FP8)
  // Prefix cache (docs/configuration.md): STATE_HOST_CACHE (GiB of pinned checkpoint slots; 0 / unset: off),
  // KV_POOL_TOKENS (0: what the VRAM budget leaves), CHECKPOINT_INTERVAL (tokens).
  if (const char* e = getenv("STATE_HOST_CACHE")) o.state_cache_gib = atof(e);
  if (const char* e = getenv("KV_POOL_TOKENS")) o.kv_pool_tokens = atoi(e);
  if (const char* e = getenv("CHECKPOINT_INTERVAL")) o.checkpoint_interval = atoi(e);
  // Vision tower (P4-12): VISION (off | bf16 | host | fp8), IMAGE_MAX_PIXELS, IMAGE_TOKENS (image tokens
  // per request).
  if (const char* e = getenv("VISION"); e && *e) {
    const std::string v = e;
    Q27_CHECK(v == "off" || v == "bf16" || v == "host" || v == "fp8", "VISION: off, bf16, host or fp8");
    o.vision = v == "bf16" ? 1 : v == "host" ? 2 : v == "fp8" ? 3 : 0;
  }
  if (const char* e = getenv("IMAGE_MAX_PIXELS")) o.image_max_pixels = atoi(e);
  if (const char* e = getenv("IMAGE_TOKENS")) o.image_tokens = atoi(e);
  // Several sequences (Phase 6): MAX_SEQS - 1 parking areas (unset / 1: one sequence at a time).
  if (const char* e = getenv("MAX_SEQS"); e && *e) o.park_slots = std::max(0, atoi(e) - 1);
  // MAX_TOKENS_FIT=1: lower max_tokens to what the VRAM budget fits (q27_max_tokens) instead of failing.
  if (const char* e = getenv("MAX_TOKENS_FIT"); e && *e) o.fit_max_tokens = atoi(e) != 0;
  return o;
}

}  // namespace

namespace q27 {
void gdn_probe_read(long long* out);  // gdn.cu (debug build with -DQ27_GDN_PROBE)
}

extern "C" {

const char* q27_last_error() { return g_error.c_str(); }
float q27t_last_us() { return g_last_us; }
void q27t_gdn_probe(long long* out) { q27::gdn_probe_read(out); }

// ---------------------------------------------------------------- engine

void* q27_create(const char* model_dir, int max_tokens, int prefill_chunk, int kv_fp8, const char* draft_dir) {
  Model* m = nullptr;
  guard([&] { m = new Model(engine_options(model_dir, max_tokens, prefill_chunk, kv_fp8, draft_dir)); });
  return m;
}

// The memory plan of q27_create(...) on a device with `free` of `total` bytes free after the CUDA context, without
// CUDA: nothing is allocated or loaded (the checkpoints' headers are read). Ask it with q27_memory_report,
// q27_max_tokens, q27_cache_stats and q27_device_memory, then q27_destroy it (the stats page's start settings).
void* q27_plan(const char* model_dir, int max_tokens, int prefill_chunk, int kv_fp8, const char* draft_dir, int64_t free,
               int64_t total) {
  Model* m = nullptr;
  guard([&] {
    Options o = engine_options(model_dir, max_tokens, prefill_chunk, kv_fp8, draft_dir);
    o.plan_only = true, o.plan_free = size_t(free), o.plan_total = size_t(total);
    m = new Model(o);
  });
  return m;
}

void q27_destroy(void* m) { delete static_cast<Model*>(m); }

int q27_reset(void* m) {
  return guard([&] { static_cast<Model*>(m)->reset(); });
}

int q27_pos(void* m) { return static_cast<Model*>(m)->pos(); }

// The context capacity (max_tokens, lowered by MAX_TOKENS_FIT when the VRAM budget does not fit it).
int q27_max_tokens(void* m) { return static_cast<Model*>(m)->max_tokens(); }

// Top-k next-token log-probs for every one of the n positions: out_ids / out_logp [n][k].
int q27_forward_topk(void* m, const int32_t* ids, int n, int k, int32_t* out_ids, float* out_logp) {
  return guard([&] {
    auto* model = static_cast<Model*>(m);
    int p0 = model->pos();
    model->forward_topk(ids, n, k, [&](int pos, const int32_t* ti, const float* tl) {
      memcpy(out_ids + size_t(pos - p0) * k, ti, size_t(k) * 4);
      memcpy(out_logp + size_t(pos - p0) * k, tl, size_t(k) * 4);
    });
  });
}

// Runs n tokens and samples the next one (temperature <= 0: greedy).
int q27_forward_sample(void* m, const int32_t* ids, int n, float temperature, int top_k, float top_p, float uniform,
                       int32_t* out) {
  return guard([&] { *out = static_cast<Model*>(m)->forward_sample(ids, n, temperature, top_k, top_p, uniform); });
}

// Speculative decoding: prompt + first token, then steps of 1..8 new tokens (out[8]; returns the count).
int q27_spec_begin(void* m, const int32_t* ids, int n, float temperature, int top_k, float top_p, uint64_t seed,
                   float uniform, int max_new, int32_t* out) {
  return guard([&] {
    SpecSampling smp{temperature, top_k, top_p, 0};  // paths 0: Q27_SPEC_PATHS
    *out = static_cast<Model*>(m)->spec_begin(ids, n, smp, seed, uniform, max_new);
  });
}

int q27_spec_step(void* m, int32_t* out) {
  int n = -1;
  if (int rc = guard([&] { n = static_cast<Model*>(m)->spec_step(out); }); rc != 0) return rc;
  return n;
}

// ahead 0: do not launch the next step before returning (another sequence runs next).
int q27_spec_step2(void* m, int32_t* out, int ahead) {
  int n = -1;
  if (int rc = guard([&] { n = static_cast<Model*>(m)->spec_step(out, ahead != 0); }); rc != 0) return rc;
  return n;
}

// Prompt chunks before q27_forward_sample / q27_spec_begin (*done: tokens run, see Model::prefill).
int q27_prefill(void* m, const int32_t* ids, int n, int budget, int spec, int* done) {
  return guard([&] { *done = static_cast<Model*>(m)->prefill(ids, n, budget, spec != 0); });
}

// Several sequences (Phase 6, Model::seq_*). Return 0, -1 on an error, -2 when the KV pool is full.
int q27_park_slots(void* m) { return static_cast<Model*>(m)->park_slots(); }
int q27_seq_park(void* m, int* area) {
  return guard([&] { *area = static_cast<Model*>(m)->seq_park(); });
}
int q27_seq_unpark(void* m, int area) {
  return guard([&] { static_cast<Model*>(m)->seq_unpark(area); });
}
int q27_seq_swap(void* m, int area) {
  return guard([&] { static_cast<Model*>(m)->seq_swap(area); });
}
int q27_seq_drop(void* m, int area) {
  return guard([&] { static_cast<Model*>(m)->seq_drop(area); });
}
int q27_seq_suspend(void* m, int area) {
  return guard([&] { static_cast<Model*>(m)->seq_suspend(area); });
}

// Batched speculative steps (P6-10, Model::spec_step_lanes): lanes[n] (-1: the running sequence, else a
// parking area); out [n][8] new tokens, counts [n]. 0, -1 on an error, -2 when the KV pool is full.
int q27_max_lanes(void* m) { return static_cast<Model*>(m)->max_lanes(); }
int q27_spec_step_lanes(void* m, const int32_t* lanes, int n, int32_t* out, int32_t* counts) {
  return guard([&] { static_cast<Model*>(m)->spec_step_lanes(lanes, n, out, counts); });
}

int q27_has_draft(void* m) { return static_cast<Model*>(m)->has_draft() ? 1 : 0; }

// Speculative counters since the last call: steps, lookup steps, tokens.
int q27_spec_stats(void* m, int64_t* out) {
  return guard([&] { static_cast<Model*>(m)->spec_stats(out); });
}

// Prefix cache: resets the sequence, restores the deepest cached prefix of ids (*cached tokens) and plans
// checkpoints at at[0..nat) for the prefill of the rest; spec: the rest goes through q27_spec_begin.
int q27_prefix_begin(void* m, const int32_t* ids, int n, const int32_t* at, int nat, int spec, int* cached) {
  return guard([&] { *cached = static_cast<Model*>(m)->prefix_begin(ids, n, at, nat, spec != 0); });
}

int q27_checkpoint(void* m) {
  return guard([&] { static_cast<Model*>(m)->checkpoint(); });
}

// out[12]: lookups, hits, prompt tokens, cached tokens, saved, evicted, live checkpoints, free KV pages, pages
// only checkpoints hold, pool pages, KV pages of a parking area, state slots allocated.
int q27_cache_stats(void* m, int64_t* out) {
  return guard([&] { static_cast<Model*>(m)->cache_stats(out); });
}

// Allocates up to n more pinned state slots of the prefix cache (*added: how many; 0 once all are).
int q27_cache_grow(void* m, int n, int* added) {
  return guard([&] { *added = static_cast<Model*>(m)->grow_state_cache(n); });
}

// Memory (stats UI): the pool report (Model::memory_report) copied into buf[0..cap) with a terminating 0;
// returns its length (call again with a larger buffer when it is >= cap), or -1 on an error.
int q27_memory_report(void* m, char* buf, int cap) {
  std::string r;
  if (guard([&] { r = static_cast<Model*>(m)->memory_report(); }) != 0) return -1;
  if (cap > 0) {
    const size_t n = std::min(r.size(), size_t(cap - 1));
    memcpy(buf, r.data(), n);
    buf[n] = 0;
  }
  return int(r.size());
}

// out[4]: device bytes free now, total, free when the engine started, VRAM budget of the pools.
int q27_device_memory(void* m, int64_t* out) {
  return guard([&] { static_cast<Model*>(m)->device_memory(out); });
}

// Images (P4-12): spans [n][3] = (start, gh, gw) of the request (after q27_reset / q27_prefix_begin), the
// vision tower on image i (pixels [4 * gh * gw][1536] fp32), its embeddings [gh * gw][5120] BF16 (tests).
int q27_has_vision(void* m) { return static_cast<Model*>(m)->has_vision() ? 1 : 0; }
int q27_mm_set(void* m, const int32_t* spans, int n) {
  return guard([&] { static_cast<Model*>(m)->mm_set(spans, n); });
}
int q27_mm_encode(void* m, int i, const float* pixels) {
  return guard([&] { static_cast<Model*>(m)->mm_encode(i, pixels); });
}
int q27_mm_embeds(void* m, int i, bf16* out) {
  return guard([&] { static_cast<Model*>(m)->mm_embeds(i, out); });
}

int q27_mm_put(void* m, int i, const bf16* in) {
  return guard([&] { static_cast<Model*>(m)->mm_put(i, in); });
}

int q27_debug_kv(void* m, int layer, int pos, int n, void* out) {
  return guard([&] { static_cast<Model*>(m)->debug_kv(layer, pos, n, out); });
}
int q27_debug_ssm(void* m, int gdn_layer, float* out) {
  return guard([&] { static_cast<Model*>(m)->debug_ssm(gdn_layer, out); });
}

int q27_configure(void* m) {
  return guard([&] { static_cast<Model*>(m)->configure(); });
}

int q27_profile_dump(void* m) {
  return guard([&] { static_cast<Model*>(m)->profile_dump(); });
}

// ---------------------------------------------------------------- kernels (tests)

int q27t_rmsnorm(const float* x, const bf16* w, int T, int D, float eps, int zc, float* out) {
  return kernel([&] { rmsnorm(x, w, T, D, eps, zc != 0, out, ts()); });
}
int q27t_embed_gather(const bf16* table, const int32_t* ids, int T, int D, float* out) {
  return kernel([&] { embed_gather(table, ids, T, D, out, ts()); });
}
int q27t_quant_fp4(const float* x, int M, int K, float input_scale, uint8_t* q, uint8_t* sf) {
  return kernel([&] { quant_fp4(x, M, K, input_scale, q, sf, ts()); });
}
int q27t_quant_fp8(const float* x, int64_t n, float input_scale, uint8_t* q) {
  return kernel([&] { quant_fp8(x, n, input_scale, q, ts()); });
}
int q27t_silu_mul(const float* g, const float* u, int64_t n, float* out) {
  return kernel([&] { silu_mul(g, u, n, out, ts()); });
}
int q27t_linear_bf16(const float* x, const bf16* w, int T, int N, int K, float* out) {
  return kernel([&] { linear_bf16(x, w, T, N, K, out, ts()); });
}
int q27t_gdn_decode_prep(const float* qkv, float* conv_state, const bf16* conv_w, const float* h, const bf16* wa,
                         const bf16* wb, float* q, float* k, float* v, float* a, float* b) {
  return kernel([&] { gdn_decode_prep(qkv, conv_state, conv_w, h, wa, wb, q, k, v, a, b, ts()); });
}
int q27t_gdn_decode(const float* q, const float* k, const float* v, const float* a, const float* b, const bf16* A_log,
                    const bf16* dt_bias, float* state, const float* z, const bf16* norm_w, float eps, float* out,
                    uint8_t* q8, float in_scale) {
  return kernel([&] { gdn_decode(q, k, v, a, b, A_log, dt_bias, state, z, norm_w, eps, out, q8, in_scale, ts()); });
}
int q27t_rmsnorm_quant(const float* x, const bf16* w, int T, float eps, float* out, int kind, float in_scale,
                       uint8_t* q, uint8_t* sf) {
  return kernel([&] { rmsnorm_quant(x, w, T, eps, out, {QuantOut::Kind(kind), in_scale, q, sf}, ts()); });
}
int q27t_silu_mul_quant_fp4(const float* g, const float* u, int64_t ld, int T, int I, float in_scale, uint8_t* q,
                            uint8_t* sf) {
  return kernel([&] { silu_mul_quant_fp4(g, u, ld, T, I, in_scale, q, sf, ts()); });
}
int q27t_silu_mul_quant_fp4_bf16(const bf16* g, const bf16* u, int64_t ld, int T, int I, float in_scale, uint8_t* q,
                                 uint8_t* sf) {
  return kernel([&] { silu_mul_quant_fp4(g, u, ld, T, I, in_scale, q, sf, ts()); });
}
int q27t_silu_mul_quant_fp4_glu(const float* gu, int T, int I, float ag, float au, float in_scale, uint8_t* q,
                                uint8_t* sf) {
  return kernel([&] { silu_mul_quant_fp4_glu(gu, T, I, ag, au, in_scale, q, sf, ts()); });
}
int q27t_silu_mul_quant_fp4_glu_bf16(const bf16* gu, int T, int I, float ag, float au, float in_scale, uint8_t* q,
                                     uint8_t* sf) {
  return kernel([&] { silu_mul_quant_fp4_glu(gu, T, I, ag, au, in_scale, q, sf, ts()); });
}
int q27t_gemm_fp4_bf16(int M, int N, int K, const uint8_t* a, const uint8_t* a_sf, const uint8_t* b, const uint8_t* b_sf,
                       float alpha, bf16* d) {
  return kernel([&] { gemm_fp4_bf16(M, N, K, a, a_sf, b, b_sf, alpha, d, ts()); });
}
int q27t_linear_bf16_pair(const float* x, const bf16* wa, const bf16* wb, int T, int N, int K, float* oa, float* ob) {
  return kernel([&] { linear_bf16_pair(x, wa, wb, T, N, K, oa, ob, ts()); });
}
int q27t_swizzle_sf(const uint8_t* src, int rows, int kblocks, uint8_t* dst) {
  return kernel([&] { swizzle_sf(src, rows, kblocks, dst, ts()); });
}
int q27t_swizzle_sf_glu(const uint8_t* src, int rows, int kblocks, uint8_t* dst, int half) {
  return kernel([&] { swizzle_sf(src, rows, kblocks, dst, ts(), half); });
}
int64_t q27t_sf_bytes(int64_t rows, int64_t kblocks) { return sf_bytes(rows, kblocks); }
int q27t_gemm_fp4(int M, int N, int K, const uint8_t* a, const uint8_t* a_sf, const uint8_t* b, const uint8_t* b_sf,
                  float alpha, const float* c, float beta, float* d) {
  return kernel([&] { gemm_fp4(M, N, K, a, a_sf, b, b_sf, alpha, c, beta, d, ts()); });
}
int q27t_gemm_fp4_glu(int M, int I, int K, const uint8_t* a, const uint8_t* a_sf, const uint8_t* gate_up,
                      const uint8_t* gate_up_sf, float alpha_gate, float alpha_up, float out_scale, uint8_t* q,
                      uint8_t* q_sf) {
  return kernel([&] { gemm_fp4_glu(M, I, K, a, a_sf, gate_up, gate_up_sf, alpha_gate, alpha_up, out_scale, q, q_sf, ts()); });
}
int q27t_gemm_fp8(int M, int N, int K, const uint8_t* a, const uint8_t* b, float alpha, const float* c, float beta,
                  float* d) {
  return kernel([&] { gemm_fp8(M, N, K, a, b, alpha, c, beta, d, ts()); });
}
int q27t_gemm_fp8_scaled(int M, int N, int K, const uint8_t* a, const float* ascale, const uint8_t* b,
                         const float* wscale, const float* c, float beta, float* d) {
  return kernel([&] { gemm_fp8_scaled(M, N, K, a, ascale, b, wscale, c, beta, d, ts()); });
}
int q27t_skinny_fp8_scaled(int M, int N, int K, const uint8_t* a, const float* ascale, const uint8_t* w,
                           const float* wscale, const float* c, float beta, float* d, void* ws, size_t ws_bytes) {
  return kernel([&] { skinny_fp8_scaled(M, N, K, a, ascale, w, wscale, c, beta, d, ws, ws_bytes, ts()); });
}
int q27t_quant_fp8_rows(const float* x, int64_t ld, int T, int K, uint8_t* q, float* scale) {
  return kernel([&] { quant_fp8_rows(x, ld, T, K, q, scale, ts()); });
}
int q27t_silu_mul_quant_fp8_rows(const float* g, const float* u, int64_t ld, int T, int I, uint8_t* q, float* scale) {
  return kernel([&] { silu_mul_quant_fp8_rows(g, u, ld, T, I, q, scale, ts()); });
}
int q27t_rmsnorm_quant_rows(const float* x, const bf16* w, int T, float eps, float* out, uint8_t* q, float* scale) {
  return kernel([&] { rmsnorm_quant(x, w, T, eps, out, {QuantOut::kFp8Row, 1.f, q, nullptr, scale}, ts()); });
}
int q27t_dequant_fp8_rows(const uint8_t* w, const float* scale, int rows, int K, bf16* out) {
  return kernel([&] { dequant_fp8_rows(w, scale, rows, K, out, ts()); });
}
int q27t_skinny_fp4(int M, int N, int K, const uint8_t* a, const uint8_t* a_sf, const uint8_t* w, const uint8_t* w_sf,
                    float alpha, const float* c, float beta, float* d, void* ws, size_t ws_bytes) {
  return kernel([&] { skinny_fp4(M, N, K, a, a_sf, w, w_sf, AlphaSegments::one(alpha), c, beta, d, ws, ws_bytes, ts()); });
}
int q27t_skinny_fp8(int M, int N, int K, const uint8_t* a, const uint8_t* w, float alpha, const float* c, float beta,
                    float* d, void* ws, size_t ws_bytes) {
  return kernel([&] { skinny_fp8(M, N, K, a, w, AlphaSegments::one(alpha), c, beta, d, ws, ws_bytes, ts()); });
}
// Microbenchmark: average time (us) of `iters` calls, cycling through `nbuf` weight copies spaced
// w_stride / sf_stride bytes apart (to defeat L2). fp4 != 0: skinny_fp4, else skinny_fp8; many != 0: the many-row
// GEMM (gemm_fp4 / gemm_fp8) instead.
int q27t_bench_gemm(int fp4, int many, int M, int N, int K, const uint8_t* a, const uint8_t* a_sf,
                    const uint8_t* w, const uint8_t* w_sf, int64_t w_stride, int64_t sf_stride, int nbuf, int iters,
                    float* d, void* ws, size_t ws_bytes, float* us) {
  return kernel([&] {
    cudaEvent_t e0, e1;
    Q27_CUDA(cudaEventCreate(&e0));
    Q27_CUDA(cudaEventCreate(&e1));
    auto run = [&](int i) {
      const uint8_t* wi = w + (i % nbuf) * w_stride;
      const uint8_t* si = w_sf ? w_sf + (i % nbuf) * sf_stride : nullptr;
      if (fp4 && many) gemm_fp4(M, N, K, a, a_sf, wi, si, 1.f, nullptr, 0.f, d, ts());
      else if (many) gemm_fp8(M, N, K, a, wi, 1.f, nullptr, 0.f, d, ts());
      else if (fp4) skinny_fp4(M, N, K, a, a_sf, wi, si, AlphaSegments::one(1.f), nullptr, 0.f, d, ws, ws_bytes, ts());
      else skinny_fp8(M, N, K, a, wi, AlphaSegments::one(1.f), nullptr, 0.f, d, ws, ws_bytes, ts());
    };
    for (int i = 0; i < std::max(nbuf, iters); i++) run(i);  // warm-up (clocks, L2 state)
    Q27_CUDA(cudaEventRecord(e0, ts()));
    for (int i = 0; i < iters; i++) run(i);
    Q27_CUDA(cudaEventRecord(e1, ts()));
    Q27_CUDA(cudaEventSynchronize(e1));
    float ms = 0;
    Q27_CUDA(cudaEventElapsedTime(&ms, e0, e1));
    *us = 1000.f * ms / iters;
    cudaEventDestroy(e0);
    cudaEventDestroy(e1);
  });
}
int q27t_gdn_prep(const float* qkv, float* conv_state, const bf16* conv_w, const float* a, const float* b,
                  const bf16* A_log, const bf16* dt_bias, int T, float* q, float* k, float* v, float* g, float* beta) {
  return kernel([&] { gdn_prep(qkv, dims::kConvDim, conv_state, conv_w, a, b, A_log, dt_bias, T, q, k, v, g, beta, ts()); });
}
int q27t_gdn_recurrent(const float* q, const float* k, const float* v, const float* g, const float* beta, int T,
                       float* state, float* o) {
  return kernel([&] { gdn_recurrent(q, k, v, g, beta, T, state, o, ts()); });
}
int q27t_gdn_chunked(const float* q, const float* k, float* v, const float* g, const float* beta, int T, int pos0,
                     float* state, float* o, void* ws) {
  return kernel([&] { gdn_chunked(q, k, v, g, beta, T, pos0, state, o, ws, ts()); });
}
size_t q27t_gdn_chunk_ws_bytes(int T) { return gdn_chunk_ws_bytes(T); }
int q27t_gdn_gated_norm(const float* o, const float* z, const bf16* w, int T, float eps, float* out, uint8_t* q8,
                        float in_scale) {
  return kernel([&] { gdn_gated_norm(o, z, dims::kVHeads * dims::kVDim, w, T, eps, out, ts(), q8, in_scale); });
}
int q27t_attn_prep(const float* qg, const float* k, const float* v, const bf16* q_norm, const bf16* k_norm, int T,
                   int pos0, void* kc, void* vc, int kv_fp8, const int32_t* page_table, float* q_out) {
  return kernel([&] {
    DevInt pos(pos0);
    attn_prep(qg, dims::kHeads * 2 * dims::kHeadDim, k, v, dims::kKvHeads * dims::kHeadDim, q_norm, k_norm, T, pos.p,
              {kc, vc}, KvType(kv_fp8), page_table, q_out, ts());
  });
}
int q27t_attention(const float* q, const float* qg, int T, int pos0, void* kc, void* vc, int kv_fp8,
                   const int32_t* page_table, float* out, void* ws, size_t ws_bytes) {
  return kernel([&] {
    // NVFP4 KV: a stage of its own, kept and grown (P7-8; Q27_ATTN_FP4_STAGE=0 converts per CTA).
    static void* stage = nullptr;
    static size_t cap = 0;
    const size_t sb = attention_prefill_stage_bytes(pos0 + T, false);  // (K and V: any mode)
    if (KvType(kv_fp8) == KvType::NVFP4 && sb > cap) {
      Q27_CUDA(cudaStreamSynchronize(ts()));
      if (stage) Q27_CUDA(cudaFree(stage));
      Q27_CUDA(cudaMalloc(&stage, sb));
      cap = sb;
    }
    attention(q, qg, T, pos0, {kc, vc}, KvType(kv_fp8), page_table, out, ws, ws_bytes, ts(), stage, cap);
  });
}
int q27t_attention_decode(const float* q, const float* qg, int T, int pos0, void* kc, void* vc, int kv_fp8,
                          const int32_t* page_table, float* out, void* ws, size_t ws_bytes) {
  return kernel([&] {
    DevInt pos(pos0);
    attention_decode(q, qg, dims::kHeads * 2 * dims::kHeadDim, T, pos.p, {kc, vc}, KvType(kv_fp8),
                     page_table, out, ws, ws_bytes, ts());
  });
}
int q27t_skinny_bf16(int M, int N, int K, const bf16* a, const bf16* w, float* d, void* ws, size_t ws_bytes) {
  return kernel([&] { skinny_bf16(M, N, K, a, w, 1.f, nullptr, 0.f, d, ws, ws_bytes, ts()); });
}
int q27t_gemm_bf16(int M, int N, int K, const bf16* a, const bf16* w, float* d) {
  return kernel([&] { gemm_bf16(M, N, K, a, w, 1.f, nullptr, 0.f, d, ts()); });
}
// The general form (C, beta).
int q27t_gemm_bf16_ex(int M, int N, int K, const bf16* a, const bf16* w, float alpha, const float* c, float beta,
                      float* d) {
  return kernel([&] { gemm_bf16(M, N, K, a, w, alpha, c, beta, d, ts()); });
}
// GDN verify of T tokens on one layer: ptrs of the two input buffers (raw, k, v, ab each), the parity and the
// commit count of the previous verify (device ints).
int q27t_gdn_verify(const float* qkv, int ld, int T, float* conv_state, const bf16* conv_w, const float* h,
                    const bf16* wa, const bf16* wb, float* q, float* raw0, float* raw1, float* k0, float* k1, float* v0,
                    float* v1, float* ab0, float* ab1, const int32_t* par, const int32_t* ncommit, const bf16* A_log,
                    const bf16* dt_bias, float* state, const float* z, int ldz, const bf16* norm_w, float eps,
                    float* out, uint8_t* q8, float in_scale, int which) {
  return kernel([&] {
    // A chain of T rows as the tree; the committed state stays in `state` (both buffers).
    static SpecTree* tree = nullptr;
    static int32_t* zeros = nullptr;
    static int tree_T = -1;
    if (!tree) {
      Q27_CUDA(cudaMalloc(&tree, sizeof(SpecTree)));
      Q27_CUDA(cudaMalloc(&zeros, 64 * 4));
      std::vector<int32_t> idx(64);
      for (int i = 0; i < 64; i++) idx[i] = i % kSpecBlock;
      Q27_CUDA(cudaMemcpy(zeros, idx.data(), 64 * 4, cudaMemcpyHostToDevice));
      Q27_CUDA(cudaDeviceSynchronize());  // (the DMA of a pageable copy may land after the return)
    }
    if (tree_T != T) {
      SpecTree h{};
      h.n = std::max(T, 1), h.npaths = 1;
      for (int i = 0; i < kTreeMax; i++) {
        const int d = i < std::min(T, kSpecBlock) ? i : 0;
        h.depth[i] = int8_t(d);
        for (int j = 0; j <= d; j++) h.path[i][j] = int8_t(d ? j : i);
        h.owner[i] = int8_t(i < kSpecBlock ? 0 : -1);
      }
      h.leaf[0] = int8_t(std::max(0, std::min(T, kSpecBlock) - 1));
      Q27_CUDA(cudaMemcpy(tree, &h, sizeof h, cudaMemcpyHostToDevice));
      Q27_CUDA(cudaDeviceSynchronize());
      tree_T = T;
    }
    GdnSpec sp{};
    sp.raw[0] = raw0, sp.raw[1] = raw1, sp.k[0] = k0, sp.k[1] = k1, sp.v[0] = v0, sp.v[1] = v1, sp.ab[0] = ab0,
    sp.ab[1] = ab1;
    sp.par = par, sp.ncommit = ncommit, sp.cpath = zeros, sp.conv_state = conv_state;
    sp.st[0] = sp.st[1] = state, sp.sbuf = par, sp.tree = tree, sp.paths = 1;
    if (which & 1) gdn_verify_prep(qkv, ld, T, conv_state, conv_w, h, wa, wb, q, sp, ts());
    if (which & 2) gdn_verify(q, T, A_log, dt_bias, state, z, ldz, norm_w, eps, out, q8, in_scale, sp, ts());
  });
}
int q27t_topk_rows(const float* logits, int T, int V, int k, float* vals, int32_t* ids) {
  return kernel([&] { topk_rows(logits, T, V, k, vals, ids, ts()); });
}
int q27t_logsoftmax_topk(const float* logits, int T, int V, int k, int32_t* ids, float* logp) {
  return kernel([&] { logsoftmax_topk(logits, T, V, k, ids, logp, ts()); });
}
int q27t_sample(const float* logits, int T, int V, float temperature, int top_k, float top_p, const float* uniform,
                int32_t* out) {
  return kernel([&] { sample(logits, T, V, temperature, top_k, top_p, uniform, out, ts()); });
}

}  // extern "C"
