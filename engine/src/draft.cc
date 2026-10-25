#include "draft.h"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <fstream>

#include "common.h"
#include "prefetch.h"

namespace q27 {

using namespace ddims;

void Draft::tensor(const std::string& name, const bf16** dst, int64_t rows, int64_t cols) {
  const auto& info = ckpt_->info(name);
  Q27_CHECK(info.dtype == "BF16", name + ": expected BF16");
  if (rows >= 0) Q27_CHECK(info.shape.size() >= 1 && info.shape[0] == rows, name + ": unexpected shape");
  if (cols >= 0) Q27_CHECK(info.shape.size() == 2 && info.shape[1] == cols, name + ": unexpected shape");
  plain_.push_back({name, wpool_.add("draft " + name, size_t(info.nbytes)), reinterpret_cast<const void**>(dst)});
}

void Draft::linear(DLinear& l, const std::vector<std::string>& parts, int K, bool keep_bf16) {
  const DraftQuant q = keep_bf16 ? DraftQuant::kBf16 : quant_;
  int N = 0;
  for (auto& p : parts) {
    const auto& info = ckpt_->info(p);
    Q27_CHECK(info.dtype == "BF16" && info.shape.size() == 2 && info.shape[1] == K, p + ": unexpected tensor");
    N += int(info.shape[0]);
  }
  Q27_CHECK(N % 128 == 0 && K % 64 == 0, parts[0] + ": shape not supported by the skinny GEMMs");
  l.N = N, l.K = K;
  const std::string what = "draft " + parts[0];
  size_t w_off = 0, sf_off = 0, scale_off = 0;
  if (q == DraftQuant::kBf16) {
    w_off = wpool_.add(what, size_t(N) * K * 2);
  } else if (q == DraftQuant::kFp8) {
    w_off = wpool_.add(what, size_t(N) * K);
    scale_off = wpool_.add(what + " scales", size_t(N) * 4);
  } else {
    w_off = wpool_.add(what, size_t(N) * K / 2);
    sf_off = wpool_.add(what + " block scales", size_t(sf_bytes(N, K / 16)));
    scale_off = wpool_.add(what + " scales", size_t(N) * 4);
  }
  int row0 = 0;
  for (auto& p : parts) {
    lin_.push_back({p, &l, row0, w_off, sf_off, scale_off});
    row0 += int(ckpt_->info(p).shape[0]);
  }
  // The format of a kept-BF16 layer is recorded through its (absent) scales.
  if (keep_bf16) lin_.back().scale_off = size_t(-1);
}

Draft::Draft(const std::string& dir, Pool& weights, Pool& act, int max_ctx_rows, int lanes)
    : wpool_(weights), apool_(act), max_ctx_rows_(std::max(max_ctx_rows, kBlock * lanes)), lanes_(lanes) {
  Q27_CHECK(lanes >= 1 && lanes <= kMaxLanes, "draft: 1..8 lanes");
  ckpt_ = std::make_unique<SafeTensors>(dir);
  const Json& cfg = ckpt_->config();
  const Json& dc = cfg["dflash_config"];
  Q27_CHECK(cfg["hidden_size"].as_int() == kHidden && cfg["num_hidden_layers"].as_int() == kLayers &&
                cfg["num_attention_heads"].as_int() == kHeads && cfg["num_key_value_heads"].as_int() == kKvHeads &&
                cfg["head_dim"].as_int() == kHeadDim && cfg["intermediate_size"].as_int() == kInter &&
                cfg["sliding_window"].as_int() == kWindow && dc["block_size"].as_int() == kBlock &&
                dc["mask_token_id"].as_int() == kMaskId && dc["selector_top_k"].as_int() == kSelK &&
                dc["selector_rank"].as_int() == kSelRank && dc["conv_kernel_size"].as_int() == kTaps &&
                dc["conv_group_size"].as_int() == kHidden / kGroups,
            "draft config.json does not match the DFlash2 dimensions this runtime is built for");
  for (int i = 0; i < kCtxFeatures; i++)
    Q27_CHECK(dc["target_layer_ids"][i].as_int() == kCtxLayers[i], "unexpected draft target_layer_ids");
  if (const char* e = getenv("Q27_DRAFT_QUANT")) {
    if (!strcmp(e, "bf16"))
      quant_ = DraftQuant::kBf16;
    else if (!strcmp(e, "fp8"))
      quant_ = DraftQuant::kFp8;
    else if (!strcmp(e, "fp4"))
      quant_ = DraftQuant::kFp4;
    else
      fail(std::string("Q27_DRAFT_QUANT must be bf16, fp8 or fp4 (got ") + e + ")");
  }
  log("draft linear layers: %s", quant_ == DraftQuant::kBf16 ? "bf16" : quant_ == DraftQuant::kFp8 ? "fp8" : "nvfp4");

  layers_.resize(kLayers);
  for (int i = 0; i < kLayers; i++) {
    auto& L = layers_[i];
    const std::string p = "layers." + std::to_string(i);
    tensor(p + ".input_layernorm.weight", &L.in_norm, kHidden);
    tensor(p + ".post_attention_layernorm.weight", &L.post_norm, kHidden);
    tensor(p + ".attention_conv.base_kernel", &L.attn_base, 2);
    tensor(p + ".mlp_conv.base_kernel", &L.mlp_base, 2);
    tensor(p + ".self_attn.q_norm.weight", &L.q_norm, kHeadDim);
    tensor(p + ".self_attn.k_norm.weight", &L.k_norm, kHeadDim);
    linear(L.attn_kproj, {p + ".attention_conv.kernel_projection.weight"}, kHidden);
    linear(L.mlp_kproj, {p + ".mlp_conv.kernel_projection.weight"}, kHidden);
    linear(L.qkv, {p + ".self_attn.q_proj.weight", p + ".self_attn.k_proj.weight", p + ".self_attn.v_proj.weight"},
           kHidden);
    linear(L.o, {p + ".self_attn.o_proj.weight"}, kHeads * kHeadDim);
    linear(L.gate_up, {p + ".mlp.gate_proj.weight", p + ".mlp.up_proj.weight"}, kHidden);
    linear(L.down, {p + ".mlp.down_proj.weight"}, kInter);
  }
  linear(fc_, {"fc.weight"}, kCtxFeatures * kHidden);
  linear(hproj_, {"candidate_selector.hidden_projection.weight"}, kHidden, true);
  tensor("hidden_norm.weight", &hidden_norm_, kHidden);
  tensor("norm.weight", &norm_, kHidden);
  tensor("candidate_selector.predecessor_codebook", &pred_cb_, -1, kSelRank);
  tensor("candidate_selector.successor_codebook", &succ_cb_, -1, kSelRank);

  if (const char* vf = getenv("Q27_DRAFT_VOCAB"); vf && *vf) {
    std::ifstream in(vf);
    Q27_CHECK(in.good(), std::string("cannot read the draft vocabulary ") + vf);
    for (int64_t id; in >> id;) {
      Q27_CHECK(id >= 0 && id < dims::kVocab, "draft vocabulary: id out of range");
      vocab_.push_back(int32_t(id));
    }
    std::sort(vocab_.begin(), vocab_.end());
    vocab_.erase(std::unique(vocab_.begin(), vocab_.end()), vocab_.end());
    Q27_CHECK(!vocab_.empty() && vocab_.size() % 128 == 0, "draft vocabulary: the id count must be a multiple of 128");
    const int n = int(vocab_.size());
    vocab_ids_off_ = wpool_.add("draft vocabulary ids", size_t(n) * 4);
    vocab_w_off_ = wpool_.add("draft vocabulary head", size_t(n) * kHidden / 2);
    vocab_sf_off_ = wpool_.add("draft vocabulary head scales", size_t(sf_bytes(n, kHidden / 16)));
    log("draft vocabulary: %d ids (%s)", n, vf);
  }

  const int R = max_ctx_rows_;
  const int B = kBlock * lanes;  // block rows of a (batched) step
  auto f32 = [&](const char* what, int64_t n) { return act.add(std::string("draft ") + what, size_t(n) * 4); };
  auto b16 = [&](const char* what, int64_t n) { return act.add(std::string("draft ") + what, size_t(n) * 2); };
  off_.x = f32("x", B * kHidden);
  off_.h = f32("h", B * kHidden);
  off_.hb = b16("hb", B * kHidden);
  off_.coef = f32("coef", B * kCoef);
  off_.act_q = act.add("draft GEMM input", size_t(B) * kMaxK * 2);
  off_.act_sf = act.add("draft GEMM input scales", size_t(sf_bytes(B, kMaxK / 16)));
  off_.act_scale = f32("GEMM input row scales", B);
  off_.qkv = f32("qkv", B * kQkv);
  off_.q = f32("q", B * kHeads * kHeadDim);
  off_.kb = f32("kb", B * kKvHeads * kHeadDim);
  off_.vb = f32("vb", B * kKvHeads * kHeadDim);
  off_.part = f32("attention partials", int64_t(draft_attention_part_floats()) * lanes);
  off_.o32 = f32("attention out", B * kHeads * kHeadDim);
  off_.y = f32("y", B * kHidden);
  off_.gu = f32("gate_up", B * 2 * kInter);
  ring_bytes_ = size_t(kWindow) * kKvHeads * kHeadDim * 2;
  off_.ring_k = act.add("draft ring K", ring_bytes_ * kLayers);
  off_.ring_v = act.add("draft ring V", ring_bytes_ * kLayers);
  off_.ctx = f32("ctx", int64_t(R) * kHidden);
  off_.ctxb = b16("ctx bf16", int64_t(R) * kHidden);
  off_.kv = f32("ctx kv", int64_t(R) * 2 * kKvHeads * kHeadDim);
  // Selector: a batched step computes the candidates of every block row (lane z: rows 8 z + 1 ..).
  const int SR = lanes > 1 ? B : kDraft;
  off_.hp = f32("selector hidden", SR * kSelRank);
  off_.cand_v = f32("candidates", SR * kSelK);
  off_.cand_i = act.add("draft candidate ids", SR * kSelK * 4);
  off_.choice_q = f32("choice q", kDraft * kSelK);
  off_.choice_path = act.add("draft choice path", kDraft * 4);
  off_.choice_src = act.add("draft choice source", 4);
  off_.lattice = f32("lattice", lanes * kDraft * kSelK * kSelK);
  off_.sel_cnt = act.add("draft selector counter", lanes * 4);
  off_.src_lanes = act.add("draft choice source of the lanes", kMaxLanes * 4);
}

void Draft::bind() {
  for (auto& j : plain_) *j.dst = wpool_.at<void>(j.offset);
  for (auto& j : lin_) {
    j.dst->w = wpool_.at<void>(j.w_off);
    const bool bf = j.scale_off == size_t(-1) || quant_ == DraftQuant::kBf16;
    j.dst->scale = bf ? nullptr : wpool_.at<float>(j.scale_off);
    j.dst->sf = !bf && quant_ == DraftQuant::kFp4 ? wpool_.at<uint8_t>(j.sf_off) : nullptr;
  }
  x_ = apool_.at<float>(off_.x), h_ = apool_.at<float>(off_.h), hb_ = apool_.at<bf16>(off_.hb);
  coef_ = apool_.at<float>(off_.coef);
  act_q_ = apool_.at<void>(off_.act_q), act_sf_ = apool_.at<uint8_t>(off_.act_sf);
  act_scale_ = apool_.at<float>(off_.act_scale);
  qkv_ = apool_.at<float>(off_.qkv), q_ = apool_.at<float>(off_.q), kb_ = apool_.at<float>(off_.kb);
  vb_ = apool_.at<float>(off_.vb), part_ = apool_.at<float>(off_.part), o32_ = apool_.at<float>(off_.o32);
  y_ = apool_.at<float>(off_.y), gu_ = apool_.at<float>(off_.gu);
  ring_k_ = apool_.at<bf16>(off_.ring_k), ring_v_ = apool_.at<bf16>(off_.ring_v);
  ctx_ = apool_.at<float>(off_.ctx), ctxb_ = apool_.at<bf16>(off_.ctxb), kv_ = apool_.at<float>(off_.kv);
  hp_ = apool_.at<float>(off_.hp), cand_v_ = apool_.at<float>(off_.cand_v), cand_i_ = apool_.at<int32_t>(off_.cand_i);
  lattice_ = apool_.at<float>(off_.lattice);
  choice.cand = cand_i_;
  choice.q = apool_.at<float>(off_.choice_q);
  choice.path = apool_.at<int32_t>(off_.choice_path);
  choice.src = apool_.at<int32_t>(off_.choice_src);
  sel_cnt_ = apool_.at<int32_t>(off_.sel_cnt);
  Q27_CUDA(cudaMemset(sel_cnt_, 0, size_t(lanes_) * 4));
  src_lanes_ = apool_.at<int32_t>(off_.src_lanes);
  if (!vocab_.empty()) {
    vocab_ids_ = wpool_.at<int32_t>(vocab_ids_off_);
    vocab_w_ = wpool_.at<uint8_t>(vocab_w_off_);
    vocab_sf_ = wpool_.at<uint8_t>(vocab_sf_off_);
  }
  Q27_CUDA(cudaMemset(act_sf_, 0, size_t(sf_bytes(kBlock * lanes_, kMaxK / 16))));
}

void Draft::load(cudaStream_t s) {
  auto t0 = std::chrono::steady_clock::now();
  constexpr int64_t kChunk = Prefetcher::kSlotBytes;
  // Every tensor in slot-sized chunks, read ahead by several threads (prefetch.h), in the order uploaded below.
  std::vector<Prefetcher::Read> plan;
  auto plan_tensor = [&](const std::string& name) {
    const int64_t n = ckpt_->info(name).nbytes;
    for (int64_t off = 0; off < n; off += kChunk) plan.push_back({ckpt_.get(), name, off, std::min(kChunk, n - off), nullptr});
  };
  for (auto& j : plain_) plan_tensor(j.name);
  for (auto& j : lin_) plan_tensor(j.name);
  Prefetcher pf(std::move(plan));
  size_t ri = 0, total = 0;
  // Streams one tensor into device memory at dst through the prefetcher's pinned slots.
  auto upload = [&](const std::string& name, char* dst) {
    const auto& info = ckpt_->info(name);
    total += size_t(info.nbytes);
    for (int64_t off = 0; off < info.nbytes; off += kChunk) {
      void* buf = pf.get(ri, name, off);
      Q27_CUDA(cudaMemcpyAsync(dst + off, buf, size_t(std::min(kChunk, info.nbytes - off)), cudaMemcpyHostToDevice, s));
      pf.release(ri++, s);
    }
  };
  for (auto& j : plain_) upload(j.name, wpool_.at<char>(j.offset));
  // Linear layers: BF16 straight into place; quantized ones through a temporary BF16 copy.
  size_t tmp_bytes = 0;
  for (auto& j : lin_) tmp_bytes = std::max(tmp_bytes, size_t(ckpt_->info(j.name).nbytes));
  bf16* tmp = nullptr;
  if (quant_ != DraftQuant::kBf16) Q27_CUDA(cudaMalloc(&tmp, tmp_bytes));
  for (auto& j : lin_) {
    const auto& info = ckpt_->info(j.name);
    const int n = int(info.shape[0]);
    if (!j.dst->scale) {  // BF16
      upload(j.name, static_cast<char*>(const_cast<void*>(j.dst->w)) + size_t(j.row0) * j.dst->K * 2);
      continue;
    }
    upload(j.name, reinterpret_cast<char*>(tmp));
    draft_quant_weight(tmp, n, j.dst->K, quant_, *j.dst, j.row0, s);
    Q27_CUDA(cudaStreamSynchronize(s));  // tmp is reused by the next tensor
  }
  Q27_CUDA(cudaStreamSynchronize(s));
  if (tmp) Q27_CUDA(cudaFree(tmp));
  double sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  log("loaded the draft (%.2f GB) in %.1f s", total / 1e9, sec);
}

void Draft::build_vocab_head(const uint8_t* lm_w, const uint8_t* lm_sf, cudaStream_t s) {
  if (vocab_.empty()) return;
  const int n = int(vocab_.size());
  Q27_CUDA(cudaMemcpyAsync(vocab_ids_, vocab_.data(), size_t(n) * 4, cudaMemcpyHostToDevice, s));
  Q27_CUDA(cudaMemsetAsync(vocab_sf_, 0, size_t(sf_bytes(n, kHidden / 16)), s));
  draft_gather_head(lm_w, lm_sf, vocab_ids_, n, kHidden, vocab_w_, vocab_sf_, s);
  Q27_CUDA(cudaStreamSynchronize(s));
}

void Draft::reset(cudaStream_t s) {
  // The ring only ever serves positions that were written; nothing to clear but the selector counters.
  Q27_CUDA(cudaMemsetAsync(sel_cnt_, 0, size_t(lanes_) * 4, s));
}

ActOut Draft::act() const { return ActOut{int(quant_), act_q_, act_sf_, act_scale_}; }

// GEMM on the activation in act() (M <= 8 rows; a batched step 8 per lane).
void Draft::gemm(int M, const DLinear& l, float* d, void* ws, size_t ws_bytes, cudaStream_t s) const {
  if (!l.scale) {
    skinny_bf16(M, l.N, l.K, static_cast<const bf16*>(act_q_), static_cast<const bf16*>(l.w), 1.f, nullptr, 0.f, d,
                ws, ws_bytes, s);
  } else if (quant_ == DraftQuant::kFp8) {
    skinny_fp8_scaled(M, l.N, l.K, static_cast<const uint8_t*>(act_q_), act_scale_, static_cast<const uint8_t*>(l.w),
                      l.scale, d, ws, ws_bytes, s);
  } else {
    skinny_fp4_scaled(M, l.N, l.K, static_cast<const uint8_t*>(act_q_), act_sf_, act_scale_,
                      static_cast<const uint8_t*>(l.w), l.sf, l.scale, d, ws, ws_bytes, s);
  }
}

namespace {
// Rows r0 .. r0 + n of a layer as a layer of its own (r0 a multiple of 128).
DLinear rows_of(const DLinear& l, DraftQuant q, int r0, int n) {
  DLinear r = l;
  r.N = n;
  if (!l.scale) {
    r.w = static_cast<const bf16*>(l.w) + size_t(r0) * l.K;
    return r;
  }
  r.scale = l.scale + r0;
  if (q == DraftQuant::kFp8) {
    r.w = static_cast<const uint8_t*>(l.w) + size_t(r0) * l.K;
  } else {
    r.w = static_cast<const uint8_t*>(l.w) + size_t(r0) * l.K / 2;
    r.sf = l.sf + sf_bytes(r0, l.K / 16);
  }
  return r;
}
}  // namespace

void Draft::propose(const int32_t* block_ids, const int32_t* pos, const bf16* embed, const uint8_t* lm_w,
                    const uint8_t* lm_sf, float lm_in_scale, float lm_alpha, const SpecSampling& smp,
                    const uint64_t* rng, int32_t* ids_out, uint8_t* aq, uint8_t* asf, float* logits, void* ws,
                    size_t ws_bytes, const Lookup& lk, cudaStream_t s) {
  constexpr int T = kBlock;
  const ActOut ao = act();
  embed_gather(embed, block_ids, T, kHidden, x_, s);
  draft_norm(x_, layers_[0].in_norm, T, h_, ao, nullptr, s);
  for (int l = 0; l < kLayers; l++) {
    const auto& L = layers_[l];
    bf16* rk = ring_k_ + ring_bytes_ / 2 * l;
    bf16* rv = ring_v_ + ring_bytes_ / 2 * l;
    gemm(T, L.attn_kproj, coef_, ws, ws_bytes, s);
    draft_conv_in(h_, coef_, L.attn_base, T, ao, s);
    gemm(T, L.qkv, qkv_, ws, ws_bytes, s);
    draft_qk_prep(qkv_, L.q_norm, L.k_norm, pos, q_, kb_, vb_, s);
    draft_attention(q_, kb_, vb_, rk, rv, pos, part_, o32_, s);
    draft_quant_rows(o32_, T, kHeads * kHeadDim, ao, s);
    gemm(T, L.o, y_, ws, ws_bytes, s);
    draft_conv_out_norm(y_, coef_, L.attn_base, T, x_, L.post_norm, h_, ao, nullptr, s);
    gemm(T, L.mlp_kproj, coef_, ws, ws_bytes, s);
    draft_conv_in(h_, coef_, L.mlp_base, T, ao, s);
    gemm(T, L.gate_up, gu_, ws, ws_bytes, s);
    draft_silu_mul(gu_, 2 * kInter, T, kInter, ao, s);
    gemm(T, L.down, y_, ws, ws_bytes, s);
    const bool last = l + 1 == kLayers;
    draft_conv_out_norm(y_, coef_, L.mlp_base, T, x_, last ? norm_ : layers_[l + 1].in_norm, h_,
                        last ? ActOut{int(DraftQuant::kBf16), act_q_, nullptr, nullptr} : ao, last ? hb_ : nullptr, s);
  }
  // Candidates from the target head on rows 1..7 (row 0 is the anchor).
  // With a draft vocabulary, only its rows (compact head; ids mapped back in the top-k).
  const int V = vocab_.empty() ? dims::kVocab : int(vocab_.size());
  quant_fp4(h_ + kHidden, kDraft, kHidden, lm_in_scale, aq, asf, s);
  skinny_fp4(kDraft, V, kHidden, aq, asf, vocab_.empty() ? lm_w : vocab_w_, vocab_.empty() ? lm_sf : vocab_sf_,
             AlphaSegments::one(lm_alpha), nullptr, 0.f, logits, ws, ws_bytes, s);
  topk_rows(logits, kDraft, V, kSelK, cand_v_, cand_i_, s, vocab_.empty() ? nullptr : vocab_ids_);
  skinny_bf16(kDraft, kSelRank, kHidden, hb_ + kHidden, static_cast<const bf16*>(hproj_.w), 1.f, nullptr, 0.f, hp_, ws,
              ws_bytes, s);
  draft_select(cand_v_, cand_i_, hp_, pred_cb_, succ_cb_, block_ids, smp, rng, ids_out, choice, lattice_, sel_cnt_, lk,
               s);
}

// The layer-l slice of the lanes' context rings.
static DraftLanes ring_layer(const DraftLanes& lanes, size_t elems, int l) {
  DraftLanes r = lanes;
  for (int z = 0; z < lanes.n; z++) r.ring_k[z] += elems * l, r.ring_v[z] += elems * l;
  return r;
}

void Draft::propose_lanes(const DraftLanes& lanes, const int32_t* block_ids, const bf16* embed, const uint8_t* lm_w,
                          const uint8_t* lm_sf, float lm_in_scale, float lm_alpha, const SelectLanes& smp,
                          const uint64_t* rng, int32_t* ids_out, SpecTree* trees, uint8_t* aq, uint8_t* asf,
                          float* logits, void* ws, size_t ws_bytes, cudaStream_t s) {
  Q27_CHECK(lanes.n >= 1 && lanes.n <= lanes_, "draft propose_lanes: more lanes than planned");
  const int T = kBlock * lanes.n;
  const ActOut ao = act();
  embed_gather(embed, block_ids, T, kHidden, x_, s);
  draft_norm(x_, layers_[0].in_norm, T, h_, ao, nullptr, s);
  for (int l = 0; l < kLayers; l++) {
    const auto& L = layers_[l];
    gemm(T, L.attn_kproj, coef_, ws, ws_bytes, s);
    draft_conv_in(h_, coef_, L.attn_base, T, ao, s);
    gemm(T, L.qkv, qkv_, ws, ws_bytes, s);
    draft_qk_prep_lanes(lanes.n, qkv_, L.q_norm, L.k_norm, lanes.pos, q_, kb_, vb_, s);
    draft_attention_lanes(ring_layer(lanes, ring_bytes_ / 2, l), q_, kb_, vb_, part_, o32_, s);
    draft_quant_rows(o32_, T, kHeads * kHeadDim, ao, s);
    gemm(T, L.o, y_, ws, ws_bytes, s);
    draft_conv_out_norm(y_, coef_, L.attn_base, T, x_, L.post_norm, h_, ao, nullptr, s);
    gemm(T, L.mlp_kproj, coef_, ws, ws_bytes, s);
    draft_conv_in(h_, coef_, L.mlp_base, T, ao, s);
    gemm(T, L.gate_up, gu_, ws, ws_bytes, s);
    draft_silu_mul(gu_, 2 * kInter, T, kInter, ao, s);
    gemm(T, L.down, y_, ws, ws_bytes, s);
    const bool last = l + 1 == kLayers;
    draft_conv_out_norm(y_, coef_, L.mlp_base, T, x_, last ? norm_ : layers_[l + 1].in_norm, h_,
                        last ? ActOut{int(DraftQuant::kBf16), act_q_, nullptr, nullptr} : ao, last ? hb_ : nullptr, s);
  }
  // Candidates of every block row (the anchor rows' are not used; the rows do not depend on each other).
  const int V = vocab_.empty() ? dims::kVocab : int(vocab_.size());
  quant_fp4(h_, T, kHidden, lm_in_scale, aq, asf, s);
  skinny_fp4(T, V, kHidden, aq, asf, vocab_.empty() ? lm_w : vocab_w_, vocab_.empty() ? lm_sf : vocab_sf_,
             AlphaSegments::one(lm_alpha), nullptr, 0.f, logits, ws, ws_bytes, s);
  topk_rows(logits, T, V, kSelK, cand_v_, cand_i_, s, vocab_.empty() ? nullptr : vocab_ids_);
  skinny_bf16(T, kSelRank, kHidden, hb_, static_cast<const bf16*>(hproj_.w), 1.f, nullptr, 0.f, hp_, ws, ws_bytes, s);
  DraftChoice ch = choice;
  ch.tree = trees, ch.src = src_lanes_;
  draft_select_lanes(lanes.n, smp, cand_v_, cand_i_, hp_, pred_cb_, succ_cb_, block_ids, rng, ids_out, ch, lattice_,
                     sel_cnt_, s);
}

void Draft::append_ctx_lanes(const DraftLanes& lanes, const bf16* cap, const int32_t* n_dev, const int32_t* rowmap,
                             void* ws, size_t ws_bytes, cudaStream_t s) {
  const int rows = kBlock * lanes.n;
  const ActOut ao = act();
  draft_quant_rows_lanes(cap, rows, kCtxFeatures * kHidden, ao, s, rowmap);
  gemm(rows, fc_, ctx_, ws, ws_bytes, s);
  draft_norm(ctx_, hidden_norm_, rows, nullptr, ao, nullptr, s);
  for (int l = 0; l < kLayers; l++) {
    const auto& L = layers_[l];
    gemm(rows, rows_of(L.qkv, quant_, kHeads * kHeadDim, 2 * kKvHeads * kHeadDim), kv_, ws, ws_bytes, s);
    draft_ctx_write_lanes(ring_layer(lanes, ring_bytes_ / 2, l), kv_, L.k_norm, n_dev, s);
  }
}

void Draft::append_ctx(const bf16* cap, int rows, const int32_t* pos, int pos_add, const int32_t* n_dev, void* ws,
                       size_t ws_bytes, void* scratch, size_t scratch_bytes, cudaStream_t s, const int32_t* rowmap) {
  if (rows == 0) return;
  Q27_CHECK(rows <= max_ctx_rows_, "draft append_ctx: too many rows");
  const ActOut ao = act();
  if (rows <= kBlock) {  // decode step: skinny GEMMs on the draft's format
    draft_quant_rows(cap, rows, kCtxFeatures * kHidden, ao, s, rowmap);
    gemm(rows, fc_, ctx_, ws, ws_bytes, s);
    draft_norm(ctx_, hidden_norm_, rows, nullptr, ao, nullptr, s);
    for (int l = 0; l < kLayers; l++) {
      const auto& L = layers_[l];
      gemm(rows, rows_of(L.qkv, quant_, kHeads * kHeadDim, 2 * kKvHeads * kHeadDim), kv_, ws, ws_bytes, s);
      draft_ctx_write(kv_, rows, L.k_norm, pos, pos_add, n_dev, ring_k_ + ring_bytes_ / 2 * l,
                      ring_v_ + ring_bytes_ / 2 * l, s);
    }
    return;
  }
  Q27_CHECK(!rowmap, "draft append_ctx: rowmap needs <= 8 rows");
  // Prompt: BF16 GEMMs through CUTLASS, quantized weights dequantized slice by slice into scratch.
  auto big = [&](const DLinear& l, int r0, int n, const bf16* a, float* d, int ldd) {
    if (!l.scale) {
      Q27_CHECK(ldd == n, "draft: strided output");
      gemm_bf16(rows, n, l.K, a, static_cast<const bf16*>(l.w) + size_t(r0) * l.K, 1.f, nullptr, 0.f, d, s);
      return;
    }
    const int slice = std::max(128, int(scratch_bytes / (size_t(l.K) * 2)) / 128 * 128);
    Q27_CHECK(slice >= 128, "draft: dequantization scratch too small");
    Q27_CHECK(ldd == n || slice >= n, "draft: strided output");
    for (int c = 0; c < n; c += slice) {
      const int m = std::min(slice, n - c);
      draft_dequant_weight(l, quant_, r0 + c, m, static_cast<bf16*>(scratch), s);
      gemm_bf16(rows, m, l.K, a, static_cast<const bf16*>(scratch), 1.f, nullptr, 0.f, d + c, s);
    }
  };
  // fc: column slices of the output need a strided store; use whole-output slices when possible.
  if (!fc_.scale || size_t(fc_.N) * fc_.K * 2 <= scratch_bytes) {
    big(fc_, 0, fc_.N, cap, ctx_, fc_.N);
  } else {
    // Slice by output rows of fc: D[:, c..c+m] written with a row stride -> compute into kv_ then copy.
    const int slice = std::min(2 * kKvHeads * kHeadDim, int(scratch_bytes / (size_t(fc_.K) * 2)) / 128 * 128);
    for (int c = 0; c < fc_.N; c += slice) {
      const int m = std::min(slice, fc_.N - c);
      draft_dequant_weight(fc_, quant_, c, m, static_cast<bf16*>(scratch), s);
      gemm_bf16(rows, m, fc_.K, cap, static_cast<const bf16*>(scratch), 1.f, nullptr, 0.f, kv_, s);
      Q27_CUDA(cudaMemcpy2DAsync(ctx_ + c, size_t(fc_.N) * 4, kv_, size_t(m) * 4, size_t(m) * 4, rows,
                                 cudaMemcpyDeviceToDevice, s));
    }
  }
  draft_norm(ctx_, hidden_norm_, rows, nullptr, ActOut{int(DraftQuant::kBf16), ctxb_, nullptr, nullptr}, nullptr, s);
  for (int l = 0; l < kLayers; l++) {
    const auto& L = layers_[l];
    big(L.qkv, kHeads * kHeadDim, 2 * kKvHeads * kHeadDim, ctxb_, kv_, 2 * kKvHeads * kHeadDim);
    draft_ctx_write(kv_, rows, L.k_norm, pos, pos_add, n_dev, ring_k_ + ring_bytes_ / 2 * l,
                    ring_v_ + ring_bytes_ / 2 * l, s);
  }
}

}  // namespace q27
