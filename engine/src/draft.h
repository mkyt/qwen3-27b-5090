// Block draft models for speculative decoding (Draft::kind()):
// - DFlash2 (incoai/Qwen3.8-27B-DFlash2): a 5-layer non-causal block drafter that reads target hidden
//   states (layers 5 / 19 / 33 / 47 / 61) as context, predicts 7 tokens after the anchor in one pass,
//   and a candidate selector that walks one path through the top-16 candidates of each position.
// - DSpine (tencent/Qwen3.8-27B-DSpine-b16): the same backbone without the convolutions, block-causal,
//   layers 1 / 16 / 31 / 46 / 61, an 8,192-token window and blocks of up to 16 rows. After every layer a
//   code bridge writes each row's predecessor into it (a gated message from the predecessor's code: its
//   hidden state read into a 1,024-dim code, the anchor's from a codebook). The proposals: the top-16
//   candidates of each row, then the last layer's write is redone for every candidate of the predecessor
//   (its codebook row as the code), giving a 16 x 16 table of logits per position that the selector walks.
#pragma once

#include <memory>
#include <string>
#include <vector>

#include "kernels.h"
#include "memory.h"
#include "safetensors.h"
#include "spec.h"

namespace q27 {

namespace ddims {
constexpr int kHidden = 5120;
constexpr int kHeads = 32;
constexpr int kKvHeads = 8;
constexpr int kHeadDim = 128;
constexpr int kInter = 17408;
constexpr int kLayers = 5;
constexpr int kBlock = 8;          // anchor + 7 drafted tokens
constexpr int kDraft = kBlock - 1;
constexpr int kWindow = 2048;      // sliding window (window_left = 2047) and the context ring
constexpr int kGroups = 320;       // conv groups of 16 channels
constexpr int kTaps = 2;
constexpr int kCoef = 2 * kTaps * kGroups;  // kernel_projection outputs: [side][tap][group]
constexpr int kSelK = 16;          // candidates per position
constexpr int kSelRank = 256;
constexpr int kMaskId = 248070;
constexpr int kCtxFeatures = 5;
constexpr int kCtxLayers[kCtxFeatures] = {5, 19, 33, 47, 61};  // target layers whose output is the context
constexpr int kQkv = (kHeads + 2 * kKvHeads) * kHeadDim;      // 6144
constexpr int kMaxK = kCtxFeatures * kHidden;                 // widest GEMM input (fc)
constexpr float kRopeTheta = 1e7f;
constexpr float kEps = 1e-6f;
// DSpine
constexpr int kMaxBlock = 16;  // trained block (anchor + 15); served blocks 2..16 (Q27_DRAFT_BLOCK)
constexpr int kMaxDraft = kMaxBlock - 1;
constexpr int kCode = 1024;    // code bridge: code size
constexpr int kBridgeRank = 512;
constexpr int kDSpineWindow = 8192;
constexpr int kDSpineCtxLayers[kCtxFeatures] = {1, 16, 31, 46, 61};
}  // namespace ddims

// Precision of the draft's linear layers (Q27_DRAFT_QUANT): BF16 as in the checkpoint, or quantized
// on the GPU at load with per-output-channel scales (NVFP4 also has 16-element block scales), with the
// activations quantized dynamically per token.
enum class DraftQuant : int { kBf16 = 0, kFp8 = 1, kFp4 = 2 };
enum class DraftKind : int { kDFlash2 = 0, kDSpine = 1 };

struct DLinear {
  const void* w = nullptr;      // bf16 [N][K] / e4m3 [N][K] / packed e2m1 [N][K/2]
  const uint8_t* sf = nullptr;  // NVFP4: block scales (CUTLASS swizzled layout)
  const float* scale = nullptr; // FP8 / NVFP4: per-output-channel scale [N]
  int N = 0, K = 0;
};

// Activation rows for the next GEMM, in the draft's format.
struct ActOut {
  int fmt = 0;             // DraftQuant
  void* q = nullptr;       // bf16 [T][K] / e4m3 [T][K] / packed e2m1 [T][K/2]
  uint8_t* sf = nullptr;   // NVFP4 block scales
  float* scale = nullptr;  // per-row scale (FP8 / NVFP4)
};

// Batched steps (P6-10): the lanes' positions [n] and context rings (K / V bases; one layer's, or all
// layers' in Draft::propose_lanes / append_ctx_lanes).
struct DraftLanes {
  int n = 0;
  const int32_t* pos = nullptr;
  const bf16* ring_k[kMaxLanes] = {};
  const bf16* ring_v[kMaxLanes] = {};
};
// The sampling of each lane (the selector's walk).
struct SelectLanes {
  SpecSampling smp[kMaxLanes] = {};
};

struct DraftLayer {
  const bf16 *in_norm = nullptr, *post_norm = nullptr;
  const bf16 *attn_base = nullptr, *mlp_base = nullptr;  // conv base kernels [2][2][5120]
  const bf16 *q_norm = nullptr, *k_norm = nullptr;
  DLinear attn_kproj, mlp_kproj;  // [1280][5120]
  DLinear qkv;                    // q | k | v [6144][5120]
  DLinear o;                      // [5120][4096]
  DLinear gate_up;                // gate | up [34816][5120]
  DLinear down;                   // [5120][17408]
};

class Draft {
 public:
  // Plans the weights and buffers in the given pools (committed by the caller); bind() and load() after.
  // lanes: blocks of a batched step (P6-10) the buffers are sized for.
  // host: pinned host memory (DSpine's codebook, read through UVA).
  Draft(const std::string& dir, Pool& weights, Pool& act, Pool& host, int max_ctx_rows, int lanes = 1);
  void bind();
  void load(cudaStream_t s);
  // With a draft vocabulary (Q27_DRAFT_VOCAB): the compact candidate head from the target's lm_head
  // (after the target weights are loaded).
  void build_vocab_head(const uint8_t* lm_w, const uint8_t* lm_sf, cudaStream_t s);
  bool has_vocab() const { return !vocab_.empty(); }  // (then the target's lm_head is read only by build_vocab_head)

  // Draft block at positions *pos .. *pos + 7: block_ids [8] = anchor, mask x 7 (device). Writes the 7
  // drafted tokens to ids_out[1..7] and the selector's choice. embed: target embedding (UVA), lm_head:
  // the target head (NVFP4) with its activation scale; aq / asf / logits: the target's scratch.
  void propose(const int32_t* block_ids, const int32_t* pos, const bf16* embed, const uint8_t* lm_w,
               const uint8_t* lm_sf, float lm_in_scale, float lm_alpha, const SpecSampling& smp, const uint64_t* rng,
               int32_t* ids_out, uint8_t* aq, uint8_t* asf, float* logits, void* ws, size_t ws_bytes,
               const Lookup& lk, cudaStream_t s);
  // Context K/V for `rows` target rows whose BF16 features (layers 5..61) are in cap [rows][25600], at
  // positions *pos + pos_add + r; only rows < *n_dev (if n_dev) are written. More than 8 rows go through
  // the BF16 GEMM with the weights dequantized into `scratch` (BF16 slices).
  // rowmap (optional, <= 8 rows): row r of the context is cap row rowmap[r] (the accepted tree path).
  void append_ctx(const bf16* cap, int rows, const int32_t* pos, int pos_add, const int32_t* n_dev, void* ws,
                  size_t ws_bytes, void* scratch, size_t scratch_bytes, cudaStream_t s,
                  const int32_t* rowmap = nullptr);
  // Batched steps (P6-10): the draft blocks of lanes.n sequences at once (block_ids / ids_out [n][8], rows 8 z ..
  // of every buffer), each computed exactly as propose / append_ctx of that sequence alone (chains only).
  // trees / src: [n]; rng: [n][2]; lattice / counters: the selector's scratch of each lane (see lane_*).
  void propose_lanes(const DraftLanes& lanes, const int32_t* block_ids, const bf16* embed, const uint8_t* lm_w,
                     const uint8_t* lm_sf, float lm_in_scale, float lm_alpha, const SelectLanes& smp,
                     const uint64_t* rng, int32_t* ids_out, SpecTree* trees, uint8_t* aq, uint8_t* asf, float* logits,
                     void* ws, size_t ws_bytes, cudaStream_t s);
  // Context K/V of the lanes' accepted rows: cap rows [n][8] (row 8 z + rowmap[8 z + i] of lane z), only
  // rows i < n_dev[z], at positions pos[z] + i.
  void append_ctx_lanes(const DraftLanes& lanes, const bf16* cap, const int32_t* n_dev, const int32_t* rowmap,
                        void* ws, size_t ws_bytes, cudaStream_t s);
  // The candidate ids of a batched step ([8 n][16]: lane z's draft positions are rows 8 z + 1 .. 8 z + 7).
  const int32_t* lane_cand() const { return cand_i_; }
  int32_t* lane_src() const { return src_lanes_; }
  int lanes() const { return lanes_; }
  void reset(cudaStream_t s);
  DraftQuant quant() const { return quant_; }
  DraftKind kind() const { return kind_; }
  bool dspine() const { return kind_ == DraftKind::kDSpine; }
  // Rows of a draft block (anchor + drafted tokens) of a single sequence; a batched step's blocks have
  // kBlock (8) rows (kLaneRows).
  int block() const { return block_; }
  int window() const { return window_; }  // the context ring: positions >= pos - (window - 1) are seen
  int ctx_layer(int f) const { return ctx_layers_[f]; }  // target layers whose output is the context
  // The context ring of all layers (K, then V; part of a state checkpoint).
  void* ring_k() const { return ring_k_; }
  void* ring_v() const { return ring_v_; }
  size_t ring_layers_bytes() const { return ring_bytes_ * ddims::kLayers; }

  DraftChoice choice;

 private:
  std::unique_ptr<SafeTensors> ckpt_;
  Pool& wpool_;
  Pool& apool_;
  Pool& hpool_;
  DraftQuant quant_ = DraftQuant::kFp4;
  DraftKind kind_ = DraftKind::kDFlash2;
  int block_ = ddims::kBlock, window_ = ddims::kWindow;
  int ctx_layers_[ddims::kCtxFeatures] = {};
  // DSpine: the code bridge (BF16 as in the checkpoint) and the codebook (pinned host).
  const bf16 *early_read_ = nullptr, *last_read_ = nullptr;  // [4][1024][5120], [1024][5120]
  const bf16* write_w_ = nullptr;                           // [5][5120][512]
  const bf16* gate_b_ = nullptr;                            // [512]
  DLinear packed_;                                          // gate | predecessor_gate | prev [1536][1024]
  size_t codebook_off_ = 0;
  const bf16* codebook_ = nullptr;                          // [vocab][1024]
  void propose_dspine(int nb, int T, const DraftLanes& lanes, const int32_t* block_ids, const bf16* embed,
                      const uint8_t* lm_w, const uint8_t* lm_sf, float lm_in_scale, float lm_alpha,
                      const SpecSampling* smp, const SelectLanes* sl, const uint64_t* rng, int32_t* ids_out,
                      SpecTree* trees, uint8_t* aq, uint8_t* asf, float* logits, void* ws, size_t ws_bytes,
                      cudaStream_t s);
  std::vector<DraftLayer> layers_;
  const bf16 *hidden_norm_ = nullptr, *norm_ = nullptr;
  const bf16 *pred_cb_ = nullptr, *succ_cb_ = nullptr;
  DLinear fc_;     // [5120][25600]
  DLinear hproj_;  // selector hidden projection [256][5120], always BF16
  // Draft vocabulary: the candidate top-16 only over these ids (sorted); head rows gathered from lm_head.
  std::vector<int32_t> vocab_;
  size_t vocab_ids_off_ = 0, vocab_w_off_ = 0, vocab_sf_off_ = 0;
  int32_t* vocab_ids_ = nullptr;
  uint8_t *vocab_w_ = nullptr, *vocab_sf_ = nullptr;

  // Loading: plain BF16 tensors are copied; linear layers are (quantized and) written at a row offset.
  struct Plain {
    std::string name;
    size_t offset;
    const void** dst;
  };
  struct Lin {
    std::string name;
    DLinear* dst;
    int row0;
    size_t w_off, sf_off, scale_off;  // pool offsets of the whole (fused) layer
  };
  std::vector<Plain> plain_;
  std::vector<Lin> lin_;
  void tensor(const std::string& name, const bf16** dst, int64_t rows = -1, int64_t cols = -1);
  // Plans a linear layer made of the given checkpoint tensors stacked along the output dimension.
  void linear(DLinear& l, const std::vector<std::string>& parts, int K, bool keep_bf16 = false);
  // Forward helpers.
  void gemm(int M, const DLinear& l, float* d, void* ws, size_t ws_bytes, cudaStream_t s) const;
  ActOut act() const;

  // Buffers (activation pool offsets until bind()).
  struct Off {
    size_t x, h, hb, coef, act_q, act_sf, act_scale, qkv, q, kb, vb, part, o32, y, gu, ring_k, ring_v, ctx, ctxb, kv,
        hp, cand_v, cand_i, choice_q, choice_path, choice_src, lattice, sel_cnt, src_lanes;
    size_t code, codeb, pk, msg, delta, raw, rms, cand_c, sid, ecode, egv, emsg, refined, rnorm;  // DSpine
  } off_{};
  int max_ctx_rows_;
  int lanes_ = 1;
  int32_t* src_lanes_ = nullptr;
  float *x_ = nullptr, *h_ = nullptr, *coef_ = nullptr, *qkv_ = nullptr, *q_ = nullptr, *kb_ = nullptr, *vb_ = nullptr;
  float *part_ = nullptr, *o32_ = nullptr, *y_ = nullptr, *gu_ = nullptr, *ctx_ = nullptr, *kv_ = nullptr;
  float *hp_ = nullptr, *cand_v_ = nullptr, *lattice_ = nullptr;
  int32_t *cand_i_ = nullptr, *sel_cnt_ = nullptr;
  bf16 *hb_ = nullptr, *ring_k_ = nullptr, *ring_v_ = nullptr, *ctxb_ = nullptr;
  // DSpine: the bridge (code: read-out fp32, codeb: BF16 codes, pk: packed gate / predecessor gate / value,
  // msg: messages, delta: their write), the last write's pre-write rows (raw, rms; q in pk), the refinement
  // (cand_c: candidates as head rows, sid: senders, ecode: their codebook rows, egv: gate / value, emsg:
  // messages, refined: the rewritten rows).
  float *code_ = nullptr, *pk_ = nullptr, *delta_ = nullptr, *raw_ = nullptr, *rms_ = nullptr;
  float *egv_ = nullptr, *refined_ = nullptr;
  bf16 *codeb_ = nullptr, *msg_ = nullptr, *ecode_ = nullptr, *emsg_ = nullptr;
  int32_t *cand_c_ = nullptr, *sid_ = nullptr;
  void* act_q_ = nullptr;
  uint8_t* act_sf_ = nullptr;
  float* act_scale_ = nullptr;
  size_t ring_bytes_ = 0;
};

// ---------------------------------------------------------------- kernels (draft.cu)

// Standard RMSNorm (x / rms * w) of T rows of 5120: fp32 (out, optional), the next GEMM's input (ao)
// and a BF16 copy (hb, optional).
void draft_norm(const float* x, const bf16* w, int T, float* out, ActOut ao, bf16* hb, cudaStream_t s);
// h' = c0 * h[n] + c1 * h[n - 1] (row n of the 8-row block; no second term for row 0), with
// c = base[side][tap][ch] + coef[n][side][tap][ch / 16], into the next GEMM's input.
void draft_conv_in(const float* h, const float* coef, const bf16* base, int T, ActOut ao, cudaStream_t s);
// x += conv_side1(y) (residual), then h = RMSNorm(x) * w: fp32 h, GEMM input ao, BF16 copy hb (optional).
void draft_conv_out_norm(const float* y, const float* coef, const bf16* base, int T, float* x, const bf16* w,
                         float* h, ActOut ao, bf16* hb, cudaStream_t s);
// q / k norm + RoPE for the block at positions *pos + t: q [8][32][128] fp32, block K / V [8][8][128] fp32.
void draft_qk_prep(const float* qkv, const bf16* q_norm, const bf16* k_norm, const int32_t* pos, float* q, float* kb,
                   float* vb, cudaStream_t s, int T = ddims::kBlock);
// Sliding-window attention of the T (8 or 16) block rows over the context ring (positions
// max(0, pos - window + 1) .. pos - 1) and the block keys (all of them, or 0..t for row t when causal);
// out fp32 [T][4096].
void draft_attention(const float* q, const float* kb, const float* vb, const bf16* ring_k, const bf16* ring_v,
                     const int32_t* pos, float* part, float* out, cudaStream_t s, int T = ddims::kBlock,
                     int window = ddims::kWindow, bool causal = false);
size_t draft_attention_part_floats(int rows);
// Rows [T][K] (fp32 or BF16) into the next GEMM's input.
void draft_quant_rows(const float* x, int T, int K, ActOut ao, cudaStream_t s);
void draft_quant_rows(const bf16* x, int T, int K, ActOut ao, cudaStream_t s, const int32_t* rowmap = nullptr);
// The same over the blocks of a batched step (P6-10): row r of lane r / 8 is its block's row rowmap[r].
void draft_quant_rows_lanes(const bf16* x, int T, int K, ActOut ao, cudaStream_t s, const int32_t* rowmap);
// silu(gate) * up ([T][I], gate / up rows ld apart) into the next GEMM's input.
void draft_silu_mul(const float* gu, int64_t ld, int T, int I, ActOut ao, cudaStream_t s);
// Context K/V: k_norm + RoPE of kv [rows][2048] (k | v of 8 heads) at positions *pos + pos_add + r into
// the ring (slot = position % 2048); rows >= *n_dev are skipped when n_dev is given.
void draft_ctx_write(const float* kv, int rows, const bf16* k_norm, const int32_t* pos, int pos_add,
                     const int32_t* n_dev, bf16* ring_k, bf16* ring_v, cudaStream_t s, int window = ddims::kWindow);
// Lattice scores and the walk (one kernel): writes the drafted tokens to ids_out[1..7] and the choice.
// lattice: [7][16][16] scratch, counter: one int, zero between calls.
void draft_select(const float* cand_v, const int32_t* cand_i, const float* hp, const bf16* pred_cb,
                  const bf16* succ_cb, const int32_t* anchor, SpecSampling smp, const uint64_t* rng,
                  int32_t* ids_out, DraftChoice choice, float* lattice, int32_t* counter, const Lookup& lk,
                  cudaStream_t s);
// Batched steps (P6-10): the kernels above for the blocks of `lanes` sequences (rows 8 z .. of lane z).
void draft_qk_prep_lanes(int lanes, const float* qkv, const bf16* q_norm, const bf16* k_norm, const int32_t* pos,
                         float* q, float* kb, float* vb, cudaStream_t s);
void draft_attention_lanes(const DraftLanes& lanes, const float* q, const float* kb, const float* vb, float* part,
                           float* out, cudaStream_t s, int window = ddims::kWindow, bool causal = false);
void draft_ctx_write_lanes(const DraftLanes& lanes, const float* kv, const bf16* k_norm, const int32_t* n_dev,
                           cudaStream_t s, int window = ddims::kWindow);
void draft_select_lanes(int lanes, const SelectLanes& sl, const float* cand_v, const int32_t* cand_i, const float* hp,
                        const bf16* pred_cb, const bf16* succ_cb, const int32_t* anchor, const uint64_t* rng,
                        int32_t* ids_out, DraftChoice choice, float* lattice, int32_t* counter, cudaStream_t s);
// DSpine (Draft::propose_dspine): rows of nb blocks of T rows (batched steps: T = 8, lanes true).
// x += y, then RMSNorm(x) * w into the next GEMM's input.
void dspine_add_norm(const float* y, float* x, const bf16* w, int T, ActOut ao, cudaStream_t s);
// The bridge's read: f = x + y into x (and raw), rms, f / (rms + eps) in BF16 (nb).
void dspine_read(const float* y, float* x, float* raw, float* rms, bf16* nb, int T, cudaStream_t s);
// Codes (BF16): sqrt(C) * codebook[anchor] on a block's first row, sqrt(C) * proj / |proj| on the others.
void dspine_codes(const float* proj, const int32_t* block_ids, const bf16* codebook, int rows, int T, bf16* codes,
                  cudaStream_t s);
// Messages (BF16 [rows][512]) from the packed projections pk [rows][1536].
void dspine_msg(const float* pk, const bf16* gb, int rows, int T, bf16* msg, cudaStream_t s);
// x += rms * delta (not on a block's first row), h = RMSNorm(x) * w (fp32 h optional, the next GEMM's input ao).
void dspine_write_norm(const float* delta, const float* rms, int rows, int T, float* x, const bf16* w, float* h,
                       ActOut ao, cudaStream_t s);
// The refinement: candidate token ids and senders, the senders' codes, messages, and the 16 x 16 table.
void dspine_cand(int nb, int T, bool lanes, const int32_t* cand_c, const int32_t* vocab_ids, const int32_t* block_ids,
                 int32_t* cand_i, int32_t* sid, cudaStream_t s);
void dspine_gather_codes(const bf16* codebook, const int32_t* ids, int rows, bf16* out, cudaStream_t s);
void dspine_emsg(int nb, int T, bool lanes, const float* egv, const float* pk, const bf16* gb, bf16* emsg,
                 cudaStream_t s);
void dspine_table(int nb, int T, bool lanes, const float* raw, const float* rms, const float* delta,
                  const bf16* norm_w, const int32_t* cand_c, const uint8_t* head_w, const uint8_t* head_sf,
                  float wscale, float* lattice, cudaStream_t s);
// The walk over the table (the selector's, see draft_select): one block (sl null) or nb lanes.
void dspine_select(int nb, int nd, const SelectLanes* sl, const SpecSampling& smp, const int32_t* cand_i,
                   const int32_t* anchor, const uint64_t* rng, int32_t* ids_out, DraftChoice choice, float* lattice,
                   cudaStream_t s);
// Weight quantization at load: rows [n][K] of BF16 w into rows row0.. of l (format fmt).
void draft_quant_weight(const bf16* w, int n, int K, DraftQuant fmt, const DLinear& l, int row0, cudaStream_t s);
// Rows ids[0..n) of an NVFP4 head [V][K] (packed values + swizzled block scales) into a compact head [n][K].
void draft_gather_head(const uint8_t* w, const uint8_t* sf, const int32_t* ids, int n, int K, uint8_t* dw, uint8_t* dsf,
                       cudaStream_t s);
// Rows row0 .. row0 + n of a quantized layer back to BF16 [n][K] (prompt context through the BF16 GEMM).
void draft_dequant_weight(const DLinear& l, DraftQuant fmt, int row0, int n, bf16* out, cudaStream_t s);

}  // namespace q27
