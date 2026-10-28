// Launchers of all device kernels. Pointers are device pointers (or UVA pointers to mapped host
// memory where noted); all launches go to `stream` and are asynchronous.
#pragma once

#include <cstdint>

#include <cuda_bf16.h>
#include <cuda_runtime.h>

namespace q27 {

using bf16 = __nv_bfloat16;

enum class KvType : int { BF16 = 0, FP8 = 1, NVFP4 = 2 };
// Bytes of the cache per (token, KV head): BF16 / FP8 values, or NVFP4 (128 bytes of e2m1 + 16 e4m3 scales).
constexpr size_t kv_row_bytes(KvType t) {
  return t == KvType::BF16 ? 512 : t == KvType::FP8 ? 256 : 128 + 16;  // (head_dim 256)
}

// Batched speculative steps (P6-10): up to kMaxLanes sequences ("lanes") run one step together, lane z
// with the kLaneRows verify rows 8 z .. 8 z + 7 of every row buffer. The batched kernels take the lane
// from a grid dimension (or the row) and its scalars at strided offsets (pos + z, ...); what differs per
// lane otherwise (states, page tables) comes as pointer arrays.
constexpr int kMaxLanes = 8;  // MAX_SEQS 1..8 (Phase 7)
constexpr int kLaneRows = 8;  // a chain verify block (kSpecBlock)
constexpr int kMaxBatchRows = kMaxLanes * kLaneRows;  // rows of a batched step
struct SeqLanes {
  int n = 0;                                   // lanes (0: an unbatched call)
  const int32_t* pos = nullptr;                // [n] committed lengths
  const int32_t* rope_delta = nullptr;         // [n] rotary delta of the images
  const int32_t* page_table[kMaxLanes] = {};   // logical -> physical page of each lane
};

// Quantized copy of an activation for the next GEMM (written by the op that produces it).
struct QuantOut {
  enum Kind : int { kNone = 0, kFp8 = 1, kFp4 = 2 };
  Kind kind = kNone;
  float in_scale = 1.f;
  uint8_t* q = nullptr;   // fp8 [T][D] / packed e2m1 [T][D/2]
  uint8_t* sf = nullptr;  // NVFP4 only: swizzled scale factors
};

// ---------------------------------------------------------------- elementwise (kernels.cu)

// out[t] = table[ids[t]] as fp32. `table` may be a UVA pointer to pinned host memory.
// mm (optional, images): ids >= vocab read row ids[t] - vocab of mm instead (also UVA).
void embed_gather(const bf16* table, const int32_t* ids, int T, int D, float* out, cudaStream_t s,
                  const bf16* mm = nullptr, int vocab = 0);
// RMSNorm over the last dim D: x / rms(x) * (1 + w) if zero_centered, else * w. In-place allowed.
void rmsnorm(const float* x, const bf16* w, int T, int D, float eps, bool zero_centered, float* out, cudaStream_t s);
// Dynamic NVFP4 quantization (modelopt / FlashInfer numerics, see reference/qwen27b/quant.py):
// q [M][K/2] packed e2m1 (low nibble = even column), sf in the CUTLASS SM1xx swizzled layout.
void quant_fp4(const float* x, int M, int K, float input_scale, uint8_t* q, uint8_t* sf, cudaStream_t s);
// Zero-centered RMSNorm over the hidden size (5120) per token, writing the fp32 result to `out` (may be
// null) and the quantized activation (same numerics as rmsnorm + quant_fp4 / quant_fp8).
// cap (optional): a BF16 copy of the input rows x (the residual stream), rows cap_ld elements apart.
void rmsnorm_quant(const float* x, const bf16* w, int T, float eps, float* out, QuantOut q, cudaStream_t s,
                   bf16* cap = nullptr, int cap_ld = 0);
// NVFP4 quantization of silu(gate) * up ([T][I], gate / up rows ld floats apart).
void silu_mul_quant_fp4(const float* gate, const float* up, int64_t ld, int T, int I, float input_scale, uint8_t* q,
                        uint8_t* sf, cudaStream_t s);
void silu_mul_quant_fp4(const bf16* gate, const bf16* up, int64_t ld, int T, int I, float input_scale, uint8_t* q,
                        uint8_t* sf, cudaStream_t s);
// The same from the [T][2 I] output of the interleaved gate | up weights (glu_row: gate and up in alternate blocks
// of 64 rows), with the gate's and the up's alphas applied here (the GEMM ran with alpha 1).
void silu_mul_quant_fp4_glu(const float* gu, int T, int I, float alpha_gate, float alpha_up, float input_scale,
                            uint8_t* q, uint8_t* sf, cudaStream_t s);
void silu_mul_quant_fp4_glu(const bf16* gu, int T, int I, float alpha_gate, float alpha_up, float input_scale,
                            uint8_t* q, uint8_t* sf, cudaStream_t s);
// x' = e4m3(x / input_scale), [n] elements.
void quant_fp8(const float* x, int64_t n, float input_scale, uint8_t* q, cudaStream_t s);
// out = silu(gate) * up, [n] elements.
void silu_mul(const float* gate, const float* up, int64_t n, float* out, cudaStream_t s);
// out[t][n] = x[t] . w[n] for a small BF16 weight [N][K] (GDN in_proj_a / in_proj_b).
void linear_bf16(const float* x, const bf16* w, int T, int N, int K, float* out, cudaStream_t s);
// Two BF16 linear layers [N][K] on the same input: outa = x . wa^T, outb = x . wb^T ([T][N] each).
// T > 8 (N = 48): tensor cores, x as bf16 hi + lo (about 1e-5 relative to the fp32 dots).
void linear_bf16_pair(const float* x, const bf16* wa, const bf16* wb, int T, int N, int K, float* outa, float* outb,
                      cudaStream_t s);
// Debug timeline of the decode step (no-ops unless built with -DQ27_TRACE; see launch.cuh).
void trace_clear_launches();
void trace_step_begin(cudaStream_t s);
void trace_step_end(const char* path);
// *pos += T (the sequence position kept on the device, so that a captured decode step can advance it).
void add_pos(int32_t* pos, int T, cudaStream_t s);
// *pos += *n (speculative decoding: the number of tokens the accept step committed).
void add_pos_dev(int32_t* pos, const int32_t* n, cudaStream_t s);
// Exchanges two device buffers in place (16-byte aligned, bytes a multiple of 16): parking a sequence's
// state while resuming another one (Model::seq_swap).
void swap_bytes(void* a, void* b, size_t bytes, cudaStream_t s);
// The MLP's gate | up weights are stored interleaved (Weights::fp4_glu): blocks of 64 rows, gate and up in turn,
// so that a 128-row tile (and its 128-row scale atom) holds the gate and the up rows of the same 64 outputs
// (the fused prefill GEMM gemm_fp4_glu_mma reads one scale atom per tile instead of two half-used ones).
// Row r of the gate (half 0) or the up (half 1) is row glu_row(r, half); column c of the gate in the GEMM's
// output is gate_glu_offset(c) (the up 64 further).
__host__ __device__ inline int64_t glu_row(int64_t r, int half) { return (r / 64) * 128 + half * 64 + r % 64; }
__host__ __device__ inline int64_t gate_glu_offset(int64_t c) { return glu_row(c, 0); }
// Reorders row-major e4m3 scale factors [rows][kblocks] into the CUTLASS swizzled layout (glu_half 0 / 1: the
// rows of the gate / the up into the interleaved gate | up, dst its start).
void swizzle_sf(const uint8_t* src, int rows, int kblocks, uint8_t* dst, cudaStream_t s, int glu_half = -1);
// Byte offset of scale factor (row, kblock) in the swizzled layout (host and device).
__host__ __device__ inline int64_t sf_offset(int64_t row, int64_t kb, int64_t kblocks) {
  int64_t katoms = (kblocks + 3) / 4;
  return ((row / 128) * katoms + kb / 4) * 512 + (row % 32) * 16 + ((row % 128) / 32) * 4 + (kb % 4);
}
// Bytes of a swizzled scale-factor buffer for [rows][kblocks] (rows padded to 128, kblocks to 4).
inline int64_t sf_bytes(int64_t rows, int64_t kblocks) {
  return ((rows + 127) / 128) * 128 * ((kblocks + 3) / 4) * 4;
}

// ---------------------------------------------------------------- GEMM (gemm.cu, CUTLASS SM120)

// D[M][N] = alpha * A[M][K] . B[N][K]^T + beta * C[M][N] (fp32, row-major; C may alias D or be null).
void gemm_fp4(int M, int N, int K, const uint8_t* a, const uint8_t* a_sf, const uint8_t* b, const uint8_t* b_sf,
              float alpha, const float* c, float beta, float* d, void* workspace, size_t ws_bytes, cudaStream_t s);
// The same with BF16 output (no C).
void gemm_fp4_bf16(int M, int N, int K, const uint8_t* a, const uint8_t* a_sf, const uint8_t* b, const uint8_t* b_sf,
                   float alpha, bf16* d, void* workspace, size_t ws_bytes, cudaStream_t s);
void gemm_fp8(int M, int N, int K, const uint8_t* a, const uint8_t* b, float alpha, const float* c, float beta,
              float* d, void* workspace, size_t ws_bytes, cudaStream_t s);
// The NVFP4 GEMMs on mma.sync (gemm_mma.cu): no workspace; N and K multiples of 128.
void gemm_fp4_mma(int M, int N, int K, const uint8_t* a, const uint8_t* a_sf, const uint8_t* b, const uint8_t* b_sf,
                  float alpha, const float* c, float beta, float* d, cudaStream_t s);
void gemm_fp4_bf16_mma(int M, int N, int K, const uint8_t* a, const uint8_t* a_sf, const uint8_t* b,
                       const uint8_t* b_sf, float alpha, bf16* d, cudaStream_t s);
// The MLP's gate | up GEMM with silu(gate) * up NVFP4-quantized in the epilogue: gate_up is the [2 I][K]
// weight (gate rows, then up rows; scales likewise), q [M][I / 2] and q_sf (swizzled) the down GEMM's
// activations, quantized with the global scale out_scale (I a multiple of 128).
void gemm_fp4_glu_mma(int M, int I, int K, const uint8_t* a, const uint8_t* a_sf, const uint8_t* gate_up,
                      const uint8_t* gate_up_sf, float alpha_gate, float alpha_up, float out_scale, uint8_t* q,
                      uint8_t* q_sf, cudaStream_t s);
size_t gemm_workspace_bytes(int max_m);
// Fills the MXFP8 unit scales of a workspace of gemm_workspace_bytes (gemm_fp8 then runs block-scaled).
void gemm_workspace_init(void* workspace, size_t ws_bytes, cudaStream_t s);
// BF16 x BF16 -> fp32 for many rows (gemm_bf16.cu): D[M][N] = alpha * A[M][K] . W[N][K]^T + beta * C.
void gemm_bf16(int M, int N, int K, const bf16* a, const bf16* w, float alpha, const float* c, float beta, float* d,
               cudaStream_t s);

// ---------------------------------------------------------------- skinny GEMM (skinny.cu, decode)

// Per-output-channel alpha for fused projections: alpha[i] applies from column begin[i] on.
struct AlphaSegments {
  static constexpr int kMax = 4;
  int count = 1;
  int begin[kMax] = {0, 0, 0, 0};
  float alpha[kMax] = {1.f, 1.f, 1.f, 1.f};
  static AlphaSegments one(float a) {
    AlphaSegments s;
    s.alpha[0] = a;
    return s;
  }
};
// Same contract as gemm_fp4 / gemm_fp8 for M = 1..64 (N a multiple of 128). The workspace holds the
// split-K partials and per-tile counters; it must be zeroed once before the first call (the kernels
// leave the counters at zero).
void skinny_fp4(int M, int N, int K, const uint8_t* a, const uint8_t* a_sf, const uint8_t* w, const uint8_t* w_sf,
                const AlphaSegments& alpha, const float* c, float beta, float* d, void* ws, size_t ws_bytes,
                cudaStream_t s);
void skinny_fp8(int M, int N, int K, const uint8_t* a, const uint8_t* w, const AlphaSegments& alpha, const float* c,
                float beta, float* d, void* ws, size_t ws_bytes, cudaStream_t s);
// Draft model weights quantized at load: d[t][n] = acc * wscale[n] * ascale[t] (per-output-channel weight
// scales, per-token activation scales of the dynamic quantization).
void skinny_fp8_scaled(int M, int N, int K, const uint8_t* a, const float* ascale, const uint8_t* w, const float* wscale,
                       float* d, void* ws, size_t ws_bytes, cudaStream_t s);
void skinny_fp4_scaled(int M, int N, int K, const uint8_t* a, const uint8_t* a_sf, const float* ascale, const uint8_t* w,
                       const uint8_t* w_sf, const float* wscale, float* d, void* ws, size_t ws_bytes, cudaStream_t s);
// BF16 activations [M][K] x BF16 weights [N][K] (the draft model).
void skinny_bf16(int M, int N, int K, const bf16* a, const bf16* w, float alpha, const float* c, float beta, float* d,
                 void* ws, size_t ws_bytes, cudaStream_t s);

// ---------------------------------------------------------------- GDN (gdn.cu)

// Causal depthwise conv (kernel 4) over [conv_state | qkv] + SiLU, l2norm of q and k per key head,
// q scaled by dk^-0.5, beta = sigmoid(b), g = -exp(A_log) * softplus(a + dt_bias).
// qkv [T][10240] raw projection output (rows ld floats apart); conv_state [10240][3] fp32 (oldest first),
// updated in place. Outputs: q, k [T][16][128], v [T][48][128], g, beta [T][48].
void gdn_prep(const float* qkv, int ld, float* conv_state, const bf16* conv_w, const float* a, const float* b,
              const bf16* A_log, const bf16* dt_bias, int T, float* q, float* k, float* v, float* g, float* beta,
              cudaStream_t s);
// Gated delta rule, token by token. state [48][128 (dk)][128 (dv)] fp32, updated in place; o [T][48][128].
void gdn_recurrent(const float* q, const float* k, const float* v, const float* g, const float* beta, int T,
                   float* state, float* o, cudaStream_t s);
// The same in chunks of 64 positions with tensor cores (gdn.cu; bf16 operands, fp32 state). pos0: the
// position of token 0 (chunks sit at multiples of 64, so prefills cut at multiples of 64 match uncut
// ones bit for bit). v is overwritten. ws: gdn_chunk_ws_bytes(T) bytes.
constexpr int kGdnChunk = 64;
size_t gdn_chunk_ws_bytes(int T);
void gdn_chunked(const float* q, const float* k, float* v, const float* g, const float* beta, int T, int pos0,
                 float* state, float* o, void* ws, cudaStream_t s);
// Decode (one token), same results as linear_bf16_pair + gdn_prep + gdn_recurrent + gdn_gated_norm:
// gdn_decode_prep: conv (+ state shift), q / k l2norm, a / b from the normed hidden state h [5120];
// gdn_decode: g, beta, the delta rule on `state`, gated RMSNorm with z into out (+ FP8 copy q8).
void gdn_decode_prep(const float* qkv, float* conv_state, const bf16* conv_w, const float* h, const bf16* wa,
                     const bf16* wb, float* q, float* k, float* v, float* a, float* b, cudaStream_t s);
void gdn_decode(const float* q, const float* k, const float* v, const float* a, const float* b, const bf16* A_log,
                const bf16* dt_bias, float* state, const float* z, const bf16* norm_w, float eps, float* out,
                uint8_t* q8, float in_scale, cudaStream_t s);
// out[t][h*128 + j] = o / rms(o) * w * silu(z) per value head (o, out: [T][48][128]; z rows ldz apart).
// q8 (optional): also e4m3(out / in_scale) for the out_proj GEMM.
void gdn_gated_norm(const float* o, const float* z, int ldz, const bf16* w, int T, float eps, float* out,
                    cudaStream_t s, uint8_t* q8 = nullptr, float in_scale = 1.f);
// Speculative verify of T <= 32 tree nodes (see gdn.cu): the decode kernels' arithmetic per token along
// each node's path, without advancing the states; the tokens the previous verify committed are
// replayed first. Per-row inputs of one layer live in two buffers selected by the step parity *par.
constexpr int kVerifyMaxT = 32;  // verify rows (nodes of the draft tree, see spec.h)
struct SpecTree;
struct GdnSpec {
  float* raw[2];  // [32][10240] raw conv inputs (by verify row)
  float* k[2];    // [32][16][128] normalized keys
  float* v[2];    // [32][48][128]
  float* ab[2];   // [32][96]: a (48) then b (48)
  const int32_t* par;      // current step parity
  const int32_t* ncommit;  // tokens committed by the previous verify (replayed now)
  const int32_t* cpath;    // their rows in the previous verify, in path order
  float* conv_state;       // [10240][3] of the layer (committed by gdn_verify, after its PDL wait)
  float* st[2];            // the layer's recurrent state in the two buffers: the verify reads st[*sbuf]
  const int32_t* sbuf;     // and writes the replayed state to the other (path CTAs may still be reading)
  const SpecTree* tree;    // the current draft tree (rows: nodes)
  int paths;               // tree paths launched (the tree's npaths is at most this)
  // Batched chain verify (lanes > 0, P6-10): T rows per lane, lane z's rows from kLaneRows * z in every row
  // buffer (raw / k / v / ab included), its scalars at par + z, ncommit + z, cpath + kLaneRows * z, tree + z,
  // and its states lconv[z] / lst[z]. A lane's commit is replayed in place (one state buffer: every CTA
  // reads and writes only its own columns, so nothing else reads it meanwhile); sbuf / st are not used.
  int lanes = 0;
  float* lconv[kMaxLanes] = {};
  float* lst[kMaxLanes] = {};
};
void gdn_verify_prep(const float* qkv, int ld, int T, float* conv_state, const bf16* conv_w, const float* h,
                     const bf16* wa, const bf16* wb, float* q, const GdnSpec& sp, cudaStream_t s);
void gdn_verify(const float* q, int T, const bf16* A_log, const bf16* dt_bias, float* state, const float* z, int ldz,
                const bf16* norm_w, float eps, float* out, uint8_t* q8, float in_scale, const GdnSpec& sp,
                cudaStream_t s);

// ---------------------------------------------------------------- attention (attention.cu)

struct KvLayer {
  void* k;  // [pages][64][4][256] of bf16 or e4m3
  void* v;
};
constexpr int kPageTokens = 64;

// For T new tokens at positions pos0..pos0+T-1 (pos0 = *pos_dev): q_norm + RoPE on q (from qg [T][24][512], first half
// of each head, rows ldq floats apart), k_norm + RoPE on k [T][4][256], then K/V (rows of k / v ldkv apart) are
// written to the paged cache. q_out [T][24][256] fp32.
// tree (optional, speculative verify): row t is the tree node t, rotated at position pos0 + its depth.
// Images (M-RoPE, P4-12): rope.delta (optional, device int) is added to the rotary position of every row
// (tokens after images sit at index + delta); rope.mpos (optional, [T][3] device ints) gives the (t, h, w)
// positions of each row instead (interleaved sections: frequency i takes axis i % 3).
struct RopePos {
  const int32_t* delta = nullptr;
  const int32_t* mpos = nullptr;
};
void attn_prep(const float* qg, int ldq, const float* k, const float* v, int ldkv, const bf16* q_norm,
               const bf16* k_norm, int T, const int32_t* pos_dev, KvLayer kv, KvType kv_type,
               const int32_t* page_table, float* q_out, cudaStream_t s, const SpecTree* tree = nullptr,
               RopePos rope = {});
// Causal attention of the T queries over the cache positions 0..pos0+T-1, multiplied by
// sigmoid(gate) (the second half of each head in qg). out [T][24*256] fp32. The workspace holds the
// partials of the key splits (attention_prefill_ws_bytes(T)). NVFP4 KV (P7-8): with a stage of
// attention_prefill_stage_bytes(pos0 + T, v_only) bytes, the context is rounded to E4M3 there once per call:
// mode 3 (the default with a stage) stages V and runs Q K^T on the FP4 MMA; mode 2 stages K and V for the
// FP8 KV kernel (the rows of the per-CTA conversion, which it falls back to with a V-only stage).
// Q27_ATTN_FP4_STAGE=0 (read per call) converts per CTA (mode 2).
void attention(const float* q, const float* qg, int T, int pos0, KvLayer kv, KvType kv_type,
               const int32_t* page_table, float* out, void* ws, size_t ws_bytes, cudaStream_t s,
               void* stage = nullptr, size_t stage_bytes = 0);
size_t attention_prefill_ws_bytes(int T);
size_t attention_prefill_stage_bytes(int keys, bool v_only);
bool attention_prefill_stage_on();
// Q27_ATTN_FP8 (read per call); unset or empty: 3 for NVFP4 KV with a stage (staged), else 2.
int attention_prefill_mode(KvType kv_type, bool staged);
// Same result for T <= kDecodeMaxT new tokens at *pos_dev, split over the keys (flash-decoding). The
// launch depends only on T, so it can be captured in a CUDA graph. The split count does not depend on
// T and each token splits its own keys 0..pos0+t, so a token's result is the same as when it is
// decoded alone (speculative verify). The workspace holds T * 24 * splits * 258 floats
// (splits = min(128, 2 * SMs / 4)); kAttnDecodeWsBytes is enough. qg rows are ldq floats apart.
// q8 (optional): also e4m3(out / in_scale) for the o_proj GEMM.
constexpr int kDecodeMaxT = 32;
constexpr int kDecodeMaxSplits = 128;
// tree (optional, speculative verify): row t is node t of the draft tree; it sees the committed keys
// and its ancestors (stored at slots pos0 + node) as if its path were decoded alone.
constexpr size_t kAttnDecodeWsBytes = size_t(kDecodeMaxT) * 24 * kDecodeMaxSplits * 258 * 4;
void attention_decode(const float* q, const float* qg, int ldq, int T, const int32_t* pos_dev, KvLayer kv,
                      KvType kv_type, const int32_t* page_table, float* out, void* ws, size_t ws_bytes,
                      cudaStream_t s, uint8_t* q8 = nullptr, float in_scale = 1.f, const SpecTree* tree = nullptr);
// Batched chain verify (P6-10): lanes.n blocks of kLaneRows rows (lane z: rows 8 z .., its position, page
// table and rotary delta), each computed exactly as attn_prep / attention_decode of that lane alone.
void attn_prep_lanes(const float* qg, int ldq, const float* k, const float* v, int ldkv, const bf16* q_norm,
                     const bf16* k_norm, const SeqLanes& lanes, KvLayer kv, KvType kv_type, float* q_out,
                     cudaStream_t s);
void attention_decode_lanes(const float* q, const float* qg, int ldq, const SeqLanes& lanes, KvLayer kv,
                            KvType kv_type, float* out, void* ws, size_t ws_bytes, cudaStream_t s,
                            uint8_t* q8 = nullptr, float in_scale = 1.f);
// After a speculative step: the KV of the accepted path (rows cpath[0..*ncommit) of the tree, stored at
// slots pos + row) moves to pos + i in all nlayers attention layers (layers_dev: device array).
void kv_commit(KvLayer* layers_dev, int nlayers, KvType kv_type, const int32_t* page_table, const int32_t* pos_dev,
               const int32_t* ncommit, const int32_t* cpath, cudaStream_t s);

// ---------------------------------------------------------------- sampling (sampling.cu)

// Per row: log-softmax, then the k most likely tokens (descending). k <= 64.
void logsoftmax_topk(const float* logits, int T, int V, int k, int32_t* ids, float* logp, cudaStream_t s);
// Samples one token per row from softmax(logits / temperature) restricted to top_k (1..64) and top_p.
// temperature <= 0 means greedy. `uniform` holds one U(0,1) number per row.
void sample(const float* logits, int T, int V, float temperature, int top_k, float top_p, const float* uniform,
            int32_t* out, cudaStream_t s);

}  // namespace q27
