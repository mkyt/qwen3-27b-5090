#include "model.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdarg>
#include <cstring>
#include <fstream>
#include <iterator>
#include <tuple>
#include <vector>

#include "common.h"
#include "json.h"

namespace q27 {

using namespace dims;
using namespace std::string_literals;

void fail(const std::string& msg) { throw std::runtime_error(msg); }

void log(const char* fmt, ...) {
  va_list ap;
  va_start(ap, fmt);
  fprintf(stderr, "[q27] ");
  vfprintf(stderr, fmt, ap);
  fprintf(stderr, "\n");
  va_end(ap);
}

Model::Model(const Options& opt) : opt_(opt) {
  Q27_CHECK(opt.prefill_chunk > 0 && opt.prefill_chunk % 128 == 0, "prefill_chunk must be a multiple of 128");
  Q27_CHECK(opt.logits_rows > 0 && opt.logits_rows % 128 == 0 && opt.logits_rows <= opt.prefill_chunk,
            "logits_rows must be a multiple of 128 and <= prefill_chunk");
  Budget budget = vram_ = make_budget();
  Q27_CUDA(cudaStreamCreateWithFlags(&s_, cudaStreamNonBlocking));
  ckpt_ = std::make_unique<SafeTensors>(opt.model_dir);
  // End-of-sequence ids (speculative steps end at an accepted one).
  if (std::ifstream f(opt.model_dir + "/generation_config.json"); f) {
    const Json gc = Json::parse(std::string(std::istreambuf_iterator<char>(f), {}));
    if (gc.has("eos_token_id")) {
      const Json& e = gc["eos_token_id"];
      if (e.kind == Json::Array)
        for (size_t i = 0; i < e.arr.size() && i < 2; i++) stop_[i] = int32_t(e[i].as_int());
      else
        stop_[0] = int32_t(e.as_int());
    }
  }
  if (const char* lm = getenv("Q27_LM_HEAD_CKPT"); lm && *lm && ckpt_->info("lm_head.weight").dtype == "BF16")
    lm_ckpt_ = std::make_unique<SafeTensors>(lm);
  w_.plan(*ckpt_, lm_ckpt_.get(), dev_weights_, host_weights_);
  if (w_.fp4_proj) log("GDN / attention projections: NVFP4");
  if (opt.vision) {
    Q27_CHECK(ckpt_->config()["image_token_id"].as_int() == kImagePad, "unexpected image_token_id");
    Q27_CHECK(opt.image_tokens > 0 && opt.image_max_pixels >= 4 * 256, "image_tokens / image_max_pixels");
    // (A checkpoint without the vision tower takes it from the lm_head one, the nvidia checkpoint.)
    const bool own = ckpt_->has("model.visual.patch_embed.proj.weight");
    Q27_CHECK(own || lm_ckpt_, "the checkpoint has no vision tower");
    w_.plan_vision(own ? *ckpt_ : *lm_ckpt_, dev_weights_, host_weights_, vision_mode());
    host_mm_.add("image embeddings", size_t(opt.image_tokens) * kHidden * sizeof(bf16));
  }

  // Sequence state (a verify block writes up to 8 positions past the committed length).
  max_pages_ = int(ceil_div(opt.max_tokens + kSpecBlock, kPageTokens));
  kv_page_bytes_ = size_t(kPageTokens) * kKvHeads * kv_row_bytes(opt.kv_type);
  // Pages per KV block: a block's slice spans 2 MiB at least (Q27_KV_BLOCK: another size, for A/B).
  kv_block_ = int(ceil_div(size_t(2) << 20, kv_page_bytes_));
  if (const char* e = getenv("Q27_KV_BLOCK"); e && atoi(e) > 0) kv_block_ = atoi(e);
  cache_on_ = opt.state_cache_gib > 0;
  conv_bytes_ = size_t(kGdnLayers) * kConvDim * (kConvK - 1) * sizeof(float);
  ssm_bytes_ = size_t(kGdnLayers) * kVHeads * kKDim * kVDim * sizeof(float);
  const bool spec = !opt.draft_dir.empty();
  size_t o_conv = state_.add("conv", conv_bytes_), o_ssm = state_.add("ssm", ssm_bytes_);
  size_t o_ssm2 = spec ? state_.add("ssm (second buffer)", ssm_bytes_) : 0;
  size_t o_pt = state_.add("page table", max_pages_ * sizeof(int32_t));
  size_t o_pos = state_.add("position", sizeof(int32_t));
  size_t o_delta = state_.add("rope delta", sizeof(int32_t));
  size_t o_rep = 0, o_par = 0, o_nc = 0, o_rng = 0, o_idv = 0, o_blk = 0, o_tv = 0, o_ti = 0, o_cap = 0, o_hist = 0,
         o_lk = 0, o_tree = 0, o_cpath = 0, o_sbuf = 0, o_kvl = 0, o_bud = 0;
  // Lanes of a batched step (P6-10; MAX_SEQS with a draft model).
  const int lanes = spec && opt.park_slots > 0 ? std::min(kMaxLanes, opt.park_slots + 1) : 1;
  if (spec) {
    rep_rows_ = std::max(kVerifyMaxT, kLaneRows * lanes);
    o_rep = state_.add("gdn verify inputs", 2 * kGdnLayers * rep_rows_ * kRepTokenFloats * sizeof(float));
    o_par = state_.add("spec parity", 4);
    o_nc = state_.add("spec commit count", 4);
    o_bud = state_.add("spec token budget", 4);
    o_rng = state_.add("spec rng", 16);
    o_idv = state_.add("spec verify ids", kTreeMax * 4);
    o_tree = state_.add("spec tree", sizeof(SpecTree));
    o_cpath = state_.add("spec commit path", kSpecBlock * 4);
    o_sbuf = state_.add("gdn state buffer", 4);
    o_kvl = state_.add("kv layer table", kAttnLayers * sizeof(KvLayer));
    o_blk = state_.add("spec block ids", kSpecBlock * 4);
    o_hist = state_.add("token history", size_t(max_pages_) * kPageTokens * 4);
    o_lk = state_.add("lookup match", 4);
  }
  // Batched steps (P6-10): the step scalars and page tables of up to MAX_SEQS lanes.
  size_t o_bsc = 0, o_bpt = 0, o_brng = 0, o_btree = 0;
  constexpr int kLaneInts = 5 + 3 * kSpecBlock;  // pos, delta, par, ncommit, budget, ids, block ids, commit path
  if (lanes > 1) {
    o_bsc = state_.add("batch lane scalars", size_t(kMaxLanes) * kLaneInts * 4);
    o_bpt = state_.add("batch lane page tables", size_t(kMaxLanes) * max_pages_ * 4);
    o_brng = state_.add("batch lane rng", size_t(kMaxLanes) * 16);
    o_btree = state_.add("batch lane trees", size_t(kMaxLanes) * sizeof(SpecTree));
  }

  // Activations.
  const int64_t Tc = opt.prefill_chunk, Tp = round_up(Tc, 128), L = opt.logits_rows;
  auto f32 = [&](const char* what, int64_t n) { return act_.add(what, size_t(n) * 4); };
  size_t o_x = f32("x", Tc * kHidden), o_h = f32("h", Tc * kHidden);
  size_t o_b1 = f32("big1", Tc * kInter), o_b2 = f32("big2", Tc * kInter);
  size_t o_q = f32("q", Tc * kKHeads * kKDim), o_k = f32("k", Tc * kKHeads * kKDim);
  size_t o_v = f32("v", Tc * kHeads * kHeadDim), o_o = f32("o", Tc * kHeads * kHeadDim);
  size_t o_n = f32("n", Tc * kVHeads * kVDim);
  size_t o_a = f32("a", Tc * kVHeads), o_bb = f32("b", Tc * kVHeads), o_g = f32("g", Tc * kVHeads),
         o_beta = f32("beta", Tc * kVHeads);
  size_t o_aq = act_.add("act quant", size_t(Tp) * std::max(kInter / 2, kHeads * kHeadDim));
  size_t o_asf = act_.add("act sf", size_t(sf_bytes(Tp, kInter / 16)));
  size_t o_ids = act_.add("ids", Tc * 4);
  size_t o_mpos = act_.add("mrope positions", Tc * 3 * 4);
  size_t o_logits = f32("logits", L * kVocab);
  size_t o_tki = act_.add("topk ids", L * 64 * 4), o_tkl = f32("topk logp", L * 64);
  size_t o_u = f32("uniform", 1), o_sm = act_.add("sampled", 4);
  ws_bytes_ = gemm_workspace_bytes(int(Tc));
  size_t o_ws = act_.add("gemm workspace", ws_bytes_);
  size_t o_sws = act_.add("skinny workspace", kSkinnyWsBytes);
  size_t o_aws = act_.add("attention workspace", kAttnDecodeWsBytes);
  // NVFP4 KV (P7-8): the prefill attention reads the context rounded to E4M3 from here: V only (1 KiB / token)
  // for mode 3 (the default), K and V for Q27_ATTN_FP8=2 at start.
  size_t o_fst = 0;
  if (opt.kv_type == KvType::NVFP4 && attention_prefill_stage_on()) {
    const bool v_only = attention_prefill_mode(KvType::NVFP4, true) == 3;
    fa_stage_bytes_ = attention_prefill_stage_bytes(max_pages_ * kPageTokens, v_only);
    o_fst = act_.add("prefill attention E4M3 KV", fa_stage_bytes_);
  }
  // Vision workspace: x_ .. aq_ (adjacent, free between forward calls), or its own region when larger.
  size_t o_vis = 0;
  if (opt.vision) {
    vis_ws_bytes_ = vision_ws_bytes(opt.image_max_pixels / (vdims::kPatchPixels * vdims::kPatchPixels), vision_mode());
    if (vis_ws_bytes_ > o_asf - o_x) o_vis = act_.add("vision workspace", vis_ws_bytes_);
  }
  if (spec) {
    const int vrows = std::max(kTreeMax, kLaneRows * lanes);  // verify rows (a tree, or a batched step)
    o_tv = act_.add("spec top-k values", size_t(vrows) * 64 * 4);
    o_ti = act_.add("spec top-k ids", size_t(vrows) * 64 * 4);
    o_cap = act_.add("draft context features", size_t(Tc) * ddims::kCtxFeatures * kHidden * 2);
    draft_ = std::make_unique<Draft>(opt.draft_dir, dev_weights_, act_, int(Tc), lanes);
  }

  // Prefix cache: a device staging copy of one checkpoint and the pinned host slots.
  ckpt_bytes_ = ssm_bytes_ + conv_bytes_ + (draft_ ? 2 * draft_->ring_layers_bytes() : 0);
  size_t o_stage = 0;
  int slots = 0;
  if (cache_on_) {
    o_stage = state_.add("checkpoint staging", ckpt_bytes_);
    slots = std::max(1, int(opt.state_cache_gib * double(1 << 30) / double(ckpt_bytes_)));
    host_cache_.add("state checkpoints", size_t(slots) * ckpt_bytes_);
  }

  // Parking areas of other sequences (Phase 6; a sequence's state in the checkpoint layout). They are whole
  // KV blocks at the top of the KV pool (P7-13): area k is the pages [kv_total - (k + 1) park_pages_,
  // kv_total - k park_pages_), contiguous memory since the pool is block-major. An area's pages hold KV while
  // no sequence is parked there; seq_park moves them out (reserve_area).
  Q27_CHECK(opt.park_slots >= 0 && opt.park_slots <= 15, "park_slots: 0 .. 15");
  const size_t per_page = 2 * kAttnLayers * kv_page_bytes_;
  park_pages_ = int(round_up(ceil_div(ckpt_bytes_, per_page), kv_block_));
  const int park_all = opt.park_slots * park_pages_;

  // KV pool (planned last): max_tokens, or with the prefix cache what the budget leaves (pages of
  // finished sequences stay cached while their checkpoints live). Page 0 is a scratch page.
  const size_t used = dev_weights_.size() + state_.size() + act_.size() + (size_t(256) << 20);  // graphs, slack
  // (the parking areas are pages of the pool: a sequence of max_tokens fits while none is parked)
  const int left_pages = budget.budget > used ? int((budget.budget - used) / per_page) - 1 : 0;
  // MAX_TOKENS_FIT: a context the budget does not fit (e.g. 262,144 with MAX_SEQS=8) is lowered. The state
  // sized by max_pages_ above stays as planned (a few hundred KiB more than needed).
  if (opt.fit_max_tokens && opt.kv_pool_tokens <= 0 && left_pages < max_pages_) {
    const int fit = (left_pages * kPageTokens - kSpecBlock) / 1024 * 1024;
    Q27_CHECK(fit >= opt.prefill_chunk, "MAX_TOKENS_FIT: the VRAM budget leaves no room for the KV pool");
    log("max_tokens %d does not fit the VRAM budget: lowered to %d (MAX_TOKENS_FIT)", opt.max_tokens, fit);
    opt_.max_tokens = fit;
    max_pages_ = int(ceil_div(fit + kSpecBlock, kPageTokens));
  }
  int pool_pages = max_pages_;
  if (opt.kv_pool_tokens > 0)
    pool_pages = std::max(max_pages_, int(ceil_div(opt.kv_pool_tokens, kPageTokens)));
  else if (cache_on_)
    pool_pages = std::max(max_pages_, left_pages);
  // Whole blocks: with the prefix cache the pool takes what the budget leaves rounded down to blocks.
  int kv_pages = int(round_up(pool_pages + 1, kv_block_));
  if (cache_on_ && opt.kv_pool_tokens <= 0 && kv_pages - 1 > left_pages && kv_pages - kv_block_ - 1 >= max_pages_)
    kv_pages -= kv_block_;
  // Without the prefix cache (tests) the pool holds max_tokens (or KV_POOL_TOKENS) with every area in use.
  const int kv_total = cache_on_ && opt.kv_pool_tokens <= 0 ? kv_pages : kv_pages + park_all;
  Q27_CHECK(kv_total - park_all >= kv_block_, "the KV pool does not hold the parking areas (MAX_SEQS)");
  pool_pages = kv_total - 1;
  const char* high = getenv("Q27_PAGES_HIGH");
  pool_.init(kv_total, high && atoi(high) != 0);
  const size_t o_kv = kv_.add("K | V pages (and parking areas)", size_t(kv_total) * per_page);

  check_and_log(budget, {&dev_weights_, &host_weights_, &kv_, &state_, &act_, &host_cache_, &host_mm_});
  // The pinned state cache (host_cache_, 4 GiB: about 1.8 s to pin on WSL2, and other CUDA calls wait meanwhile) is
  // not committed: its slots are allocated one by one when a checkpoint needs one or while the server is idle
  // (grow_state_cache), so the engine is ready that much sooner.
  auto since = [](auto t) { return std::chrono::duration<double>(std::chrono::steady_clock::now() - t).count(); };
  auto t_commit = std::chrono::steady_clock::now();
  for (Pool* p : {&dev_weights_, &host_weights_, &kv_, &state_, &act_, &host_mm_}) p->commit();
  log("pools allocated in %.2f s", since(t_commit));
  log("kv pool: %d pages (%d tokens), max_tokens %d%s", pool_pages, pool_pages * kPageTokens, opt_.max_tokens,
      cache_on_ ? "" : " (prefix cache off)");
  if (cache_on_) {
    stage_ = state_.at<char>(o_stage);
    slot_used_.assign(slots, false);
    slot_mem_.assign(slots, nullptr);
    Q27_CUDA(cudaStreamCreateWithFlags(&cs_, cudaStreamNonBlocking));
    Q27_CUDA(cudaEventCreateWithFlags(&stage_ev_, cudaEventDisableTiming));
    Q27_CUDA(cudaEventCreateWithFlags(&stage_in_ev_, cudaEventDisableTiming));
    Q27_CUDA(cudaEventRecord(stage_ev_, cs_));
    log("prefix cache: %d state slots of %.1f MiB (%.2f GiB pinned, allocated as needed), checkpoint interval %d", slots,
        ckpt_bytes_ / 1048576.0, slots * double(ckpt_bytes_) / (1 << 30), opt.checkpoint_interval);
  }

  kv_base_ = kv_.at<char>(o_kv);
  for (int l = 0; l < kAttnLayers; l++) kv_layers_.push_back({kv_page(0, 2 * l), kv_page(0, 2 * l + 1)});
  parked_.resize(opt.park_slots);
  for (int i = 0; i < opt.park_slots; i++) {
    const int first = kv_total - (i + 1) * park_pages_;
    Q27_CHECK(first % kv_block_ == 0 && kv_page(first + park_pages_, 0) - kv_page(first, 0) >= ptrdiff_t(ckpt_bytes_),
              "parking area layout");
    parked_[i].dev = kv_page(first, 0);
  }
  if (!parked_.empty())
    log("sequences: %d parking areas of %.1f MiB (%d KV pages each, lent to the KV pool while unused)",
        int(parked_.size()), ckpt_bytes_ / 1048576.0, park_pages_);
  if (spec) {
    Q27_CUDA(cudaMemcpy(state_.at<KvLayer>(o_kvl), kv_layers_.data(), kv_layers_.size() * sizeof(KvLayer),
                        cudaMemcpyHostToDevice));
  }
  conv_ = state_.at<float>(o_conv);
  ssm_ = ssm_buf_[0] = state_.at<float>(o_ssm);
  ssm_buf_[1] = spec ? state_.at<float>(o_ssm2) : ssm_buf_[0];
  page_table_ = state_.at<int32_t>(o_pt);
  pos_dev_ = state_.at<int32_t>(o_pos);
  rope_delta_ = state_.at<int32_t>(o_delta);
  x_ = act_.at<float>(o_x), h_ = act_.at<float>(o_h), big1_ = act_.at<float>(o_b1), big2_ = act_.at<float>(o_b2);
  q_ = act_.at<float>(o_q), k_ = act_.at<float>(o_k), v_ = act_.at<float>(o_v), o_ = act_.at<float>(o_o);
  n_ = act_.at<float>(o_n), a_ = act_.at<float>(o_a), b_ = act_.at<float>(o_bb), g_ = act_.at<float>(o_g);
  beta_ = act_.at<float>(o_beta);
  aq_ = act_.at<uint8_t>(o_aq), asf_ = act_.at<uint8_t>(o_asf);
  ids_ = act_.at<int32_t>(o_ids);
  mpos_ = act_.at<int32_t>(o_mpos);
  if (opt.vision) {
    vis_ws_ = o_vis ? act_.at<void>(o_vis) : act_.at<void>(o_x);
    mm_ = host_mm_.at<bf16>(0);
    static const char* kWhere[] = {"", "BF16 on the device", "BF16 in pinned host memory", "FP8 on the device"};
    log("vision: %s, workspace %.0f MiB%s, images up to %d px, %d image tokens per request", kWhere[opt.vision],
        vis_ws_bytes_ / 1048576.0, o_vis ? "" : " (in the prefill activations)", opt.image_max_pixels,
        opt.image_tokens);
  }
  logits_ = act_.at<float>(o_logits);
  topk_ids_ = act_.at<int32_t>(o_tki), topk_logp_ = act_.at<float>(o_tkl);
  uniform_ = act_.at<float>(o_u), sampled_ = act_.at<int32_t>(o_sm);
  ws_ = act_.at<void>(o_ws);
  gemm_workspace_init(ws_, ws_bytes_, s_);
  skinny_ws_ = act_.at<void>(o_sws);
  attn_ws_ = act_.at<void>(o_aws);
  if (fa_stage_bytes_) fa_stage_ = act_.at<void>(o_fst);
  // The chunked GDN prefill keeps its workspace in big2_ behind z ([Tc][48 x 128], see Model::gdn).
  Q27_CHECK(gdn_chunk_ws_bytes(int(Tc)) <= size_t(Tc) * (kInter - kVHeads * kVDim) * 4, "GDN chunk workspace");
  Q27_CHECK(attention_prefill_ws_bytes(int(Tc)) <= size_t(Tc) * kInter * 4, "attention workspace");
  Q27_CUDA(cudaMemset(skinny_ws_, 0, kSkinnyWsBytes));  // split-K counters start at zero
  if (spec) {
    rep_ = state_.at<float>(o_rep);
    par_ = state_.at<int32_t>(o_par), ncommit_ = state_.at<int32_t>(o_nc), budget_ = state_.at<int32_t>(o_bud);
    rng_ = state_.at<uint64_t>(o_rng);
    ids_ver_ = state_.at<int32_t>(o_idv), block_ids_ = state_.at<int32_t>(o_blk);
    hist_ = state_.at<int32_t>(o_hist);
    tree_ = state_.at<SpecTree>(o_tree);
    cpath_ = state_.at<int32_t>(o_cpath);
    sbuf_ = state_.at<int32_t>(o_sbuf);
    kv_layers_dev_ = state_.at<KvLayer>(o_kvl);
    draft_->choice.tree = tree_;
    Q27_CUDA(cudaMemset(cpath_, 0, kSpecBlock * 4));
    lk_best_ = state_.at<uint32_t>(o_lk);
    Q27_CUDA(cudaMemset(lk_best_, 0, 4));
    tv_ = act_.at<float>(o_tv), ti_ = act_.at<int32_t>(o_ti);
    cap_buf_ = act_.at<bf16>(o_cap);
    draft_->bind();
    std::vector<int32_t> blk(kSpecBlock, ddims::kMaskId);
    Q27_CUDA(cudaMemcpy(block_ids_, blk.data(), blk.size() * 4, cudaMemcpyHostToDevice));
    Q27_CUDA(cudaHostAlloc(reinterpret_cast<void**>(&spec_out_), 2 * sizeof(SpecOut), cudaHostAllocMapped));
    for (auto& e : spec_ev_) Q27_CUDA(cudaEventCreateWithFlags(&e, cudaEventDisableTiming));
    Q27_CUDA(cudaHostGetDevicePointer(reinterpret_cast<void**>(&spec_out_dev_), spec_out_, 0));
  }
  if (lanes > 1) {
    int32_t* sc = state_.at<int32_t>(o_bsc);
    b_pos_ = sc, b_delta_ = sc + kMaxLanes, b_par_ = sc + 2 * kMaxLanes, b_nc_ = sc + 3 * kMaxLanes;
    b_budget_ = sc + 4 * kMaxLanes, b_ids_ = sc + 5 * kMaxLanes, b_blk_ = b_ids_ + kMaxLanes * kSpecBlock;
    b_cpath_ = b_blk_ + kMaxLanes * kSpecBlock;
    b_pt_ = state_.at<int32_t>(o_bpt);
    b_rng_ = state_.at<uint64_t>(o_brng);
    b_tree_ = state_.at<SpecTree>(o_btree);
    Q27_CUDA(cudaMemset(sc, 0, size_t(kMaxLanes) * kLaneInts * 4));
    Q27_CUDA(cudaMemset(b_pt_, 0, size_t(kMaxLanes) * max_pages_ * 4));
    Q27_CUDA(cudaHostAlloc(reinterpret_cast<void**>(&b_out_), size_t(kMaxLanes) * 2 * sizeof(SpecOut),
                           cudaHostAllocMapped));
    Q27_CUDA(cudaHostGetDevicePointer(reinterpret_cast<void**>(&b_out_dev_), b_out_, 0));
    log("batched decode: up to %d sequences per step", lanes);
  }

  Q27_CUDA(cudaMemset(page_table_, 0, size_t(max_pages_) * 4));  // all unmapped (pages come from pool_)
  Q27_CUDA(cudaMemset(asf_, 0, sf_bytes(Tp, kInter / 16)));

  w_.load(s_);
  if (draft_) {
    draft_->load(s_);
    draft_->build_vocab_head(w_.lm_head.w, w_.lm_head.sf, s_);
  }
  reset();

  Q27_CUDA(cudaHostAlloc(reinterpret_cast<void**>(&host_io_), sizeof(HostIo), cudaHostAllocMapped));
  Q27_CUDA(cudaHostGetDevicePointer(reinterpret_cast<void**>(&host_io_dev_), host_io_, 0));
  configure();
}

// (Re)reads the debug / tuning switches from the environment and drops the captured decode graph, so
// that one process can compare settings (scripts/ab_decode.py) without reloading the weights.
void Model::configure() {
  batch_exit();
  Q27_CUDA(cudaStreamSynchronize(s_));
  drop_graphs();
  graph_warm_ = false;
  spec_warm_ = false;
  bwarm_ = false;
  const char* e = getenv("Q27_SKIP");
  skip_ = e ? e : "";
  decode_memcpy_ = getenv("Q27_DECODE_MEMCPY") != nullptr;
  skinny_ = !getenv("Q27_NO_SKINNY");
  skinny_check_ = skinny_ && getenv("Q27_SKINNY_CHECK");
  auto env_int = [](const char* name, int dflt) {
    const char* e = getenv(name);
    return e && *e ? atoi(e) : dflt;
  };
  lookup_.on = env_int("Q27_LOOKUP", 0);  // off: with an 8-row chain the draft copies as well (measured)
  lookup_.min_len = env_int("Q27_LOOKUP_MIN", 8);
  lookup_.agree = env_int("Q27_LOOKUP_AGREE", 2);
  spec_paths_ = std::clamp(env_int("Q27_SPEC_PATHS", 1), 1, kTreePaths);
  spec_pipeline_ = env_int("Q27_SPEC_PIPELINE", 1) != 0;
  mlp_bf16_ = env_int("Q27_MLP_BF16", 1) != 0;
  const char* prof = getenv("Q27_PROFILE");
  prof_.on = prof && *prof && *prof != '0';
  graphs_ = !getenv("Q27_NO_GRAPH") && !skinny_check_;
}

Model::~Model() {
  if (s_) cudaStreamSynchronize(s_);  // (a speculative step may be in flight)
  for (auto e : spec_ev_)
    if (e) cudaEventDestroy(e);
  for (auto* v : {&dgraphs_, &sgraphs_})
    for (auto& g : *v) cudaGraphExecDestroy(g.exec);
  for (auto& g : bgraphs_) cudaGraphExecDestroy(g.exec);
  if (spec_out_) cudaFreeHost(spec_out_);
  if (b_out_) cudaFreeHost(b_out_);
  if (host_io_) cudaFreeHost(host_io_);
  if (cs_) cudaStreamSynchronize(cs_), cudaStreamDestroy(cs_);
  for (char* m : slot_mem_)
    if (m) cudaFreeHost(m);
  for (auto e : {stage_ev_, stage_in_ev_})
    if (e) cudaEventDestroy(e);
  for (auto* m : {&prof_.eager, &prof_.graph})
    for (auto e : m->events) cudaEventDestroy(e);
  if (s_) cudaStreamDestroy(s_);
}

// Marks go to the eager set, or while a decode step is being captured to the graph set (whose event
// records are replayed by every launch of the graph).
void Model::mark(const char* label) {
  if (!prof_.on) return;
  auto& m = prof_.capturing ? prof_.graph : prof_.eager;
  if (m.used == int(m.events.size())) {
    cudaEvent_t e;
    Q27_CUDA(cudaEventCreate(&e));
    m.events.push_back(e);
    m.labels.push_back(nullptr);
  }
  m.labels[m.used] = label;
  // While capturing, a plain record only becomes a dependency edge; External makes it a record node.
  Q27_CUDA(cudaEventRecordWithFlags(m.events[m.used++], s_, prof_.capturing ? cudaEventRecordExternal : 0));
}

void Model::profile_collect(bool graph) {
  if (!prof_.on) return;
  auto& m = graph ? prof_.graph : prof_.eager;
  if (m.used == 0) return;
  if (!graph) mark(nullptr);  // end of the last op (a captured step records its own)
  Q27_CUDA(cudaStreamSynchronize(s_));
  for (int i = 0; i + 1 < m.used; i++) {
    float ms = 0;
    Q27_CUDA(cudaEventElapsedTime(&ms, m.events[i], m.events[i + 1]));
    auto it = std::find_if(prof_.totals.begin(), prof_.totals.end(),
                           [&](const auto& t) { return t.first == m.labels[i]; });
    if (it == prof_.totals.end()) it = prof_.totals.insert(prof_.totals.end(), {m.labels[i], {0.0, 0}});
    it->second.first += ms;
    it->second.second++;
  }
  if (!graph) m.used = 0;
  prof_.steps++;
}

void Model::profile_dump() {
  if (!prof_.on || prof_.steps == 0) return;
  double total = 0;
  for (auto& t : prof_.totals) total += t.second.first;
  log("profile: %d steps, %.3f ms/step", prof_.steps, total / prof_.steps);
  log("  %-16s %10s %8s %10s %8s", "op", "ms/step", "share", "us/call", "calls");
  for (auto& [label, v] : prof_.totals)
    log("  %-16s %10.4f %7.1f%% %10.2f %8.1f", label.c_str(), v.first / prof_.steps, 100 * v.first / total,
        1000 * v.first / v.second, double(v.second) / prof_.steps);
  prof_.totals.clear();
  prof_.steps = 0;
}

void Model::drop_graphs() {
  for (auto* v : {&dgraphs_, &sgraphs_}) {
    for (auto& g : *v) Q27_CUDA(cudaGraphExecDestroy(g.exec));
    v->clear();
  }
  for (auto& g : bgraphs_) Q27_CUDA(cudaGraphExecDestroy(g.exec));
  bgraphs_.clear();
  graph_exec_ = nullptr;
  spec_graph_ = nullptr;
}

void Model::set_sbuf(int b) {
  sbuf_host_ = b;
  ssm_ = ssm_buf_[b];
}

void Model::reset() {
  batch_exit();
  Q27_CUDA(cudaStreamSynchronize(s_));  // (drops a speculative step in flight)
  spec_inflight_ = false;
  par_host_ = 0;
  pos_ = 0;
  set_sbuf(0);
  Q27_CUDA(cudaMemsetAsync(conv_, 0, conv_bytes_, s_));
  Q27_CUDA(cudaMemsetAsync(ssm_, 0, ssm_bytes_, s_));
  Q27_CUDA(cudaMemsetAsync(pos_dev_, 0, sizeof(int32_t), s_));
  Q27_CUDA(cudaMemsetAsync(rope_delta_, 0, sizeof(int32_t), s_));
  mm_spans_.clear();
  mpos_on_ = false;
  if (draft_) {
    Q27_CUDA(cudaMemsetAsync(par_, 0, 4, s_));
    Q27_CUDA(cudaMemsetAsync(ncommit_, 0, 4, s_));
    Q27_CUDA(cudaMemsetAsync(sbuf_, 0, 4, s_));
    draft_->reset(s_);
  }
  spec_pending_ = false;
  release_seq();
  plan_.clear();
  ring_valid_ = false;
  spec_last_ = -1;
  Q27_CUDA(cudaStreamSynchronize(s_));
}

// ---------------------------------------------------------------- KV pages and the prefix cache

void Model::release_seq() {
  for (int32_t p : seq_pages_) pool_.unref(p);
  if (!seq_pages_.empty()) Q27_CUDA(cudaMemsetAsync(page_table_, 0, seq_pages_.size() * 4, s_));
  seq_pages_.clear();
  seq_ids_.clear();
}

void Model::upload_pages(int32_t* table, const int32_t* pages, size_t n) {
  if (n == 0) return;
  pt_up_.resize(n);
  for (size_t i = 0; i < n; i++) pt_up_[i] = page_ref(pages[i]);
  // (pageable source: the call returns once it is staged, so pt_up_ may be reused right away)
  Q27_CUDA(cudaMemcpyAsync(table, pt_up_.data(), n * 4, cudaMemcpyHostToDevice, s_));
}

void Model::ensure_pages(int tokens) { ensure_pages_of(seq_pages_, page_table_, tokens); }

void Model::ensure_pages_of(std::vector<int32_t>& pages, int32_t* table, int tokens) {
  const int need = std::min(max_pages_, int(ceil_div(tokens, kPageTokens)));
  const int have = int(pages.size());
  if (need <= have) return;
  // A few pages ahead, so decode does not stop for a page table update every 64 tokens.
  const int want = std::min(max_pages_, need + 8);
  while (int(pages.size()) < want) {
    int p = pool_.alloc();
    while (p < 0 && int(pages.size()) < need && evict_checkpoint()) p = pool_.alloc();
    if (p < 0) {
      if (int(pages.size()) < need) {  // (the pages taken so far stay with the sequence)
        upload_pages(table + have, pages.data() + have, pages.size() - have);
        throw KvFull("KV pool exhausted");
      }
      break;
    }
    pages.push_back(p);
  }
  upload_pages(table + have, pages.data() + have, pages.size() - have);
}

bool Model::evict_checkpoint() {
  if (ckpts_.empty()) return false;
  auto it = std::min_element(ckpts_.begin(), ckpts_.end(),
                             [](const Checkpoint& a, const Checkpoint& b) { return a.last_use < b.last_use; });
  for (int32_t p : it->pages) pool_.unref(p);
  slot_used_[it->slot] = false;
  ckpts_.erase(it);
  cstats_.evicted++;
  return true;
}

int Model::grow_state_cache(int n) {
  int added = 0;
  for (size_t i = 0; i < slot_mem_.size() && added < n; i++)
    if (!slot_mem_[i]) {
      Q27_CUDA(cudaHostAlloc(reinterpret_cast<void**>(&slot_mem_[i]), ckpt_bytes_,
                             cudaHostAllocMapped | cudaHostAllocPortable));
      added++;
    }
  return added;
}

void Model::save_checkpoint(bool ring) {
  if (!cache_on_ || pos_ == 0) return;
  Q27_CHECK(int(seq_ids_.size()) == pos_, "checkpoint: token history does not match the position");
  save_checkpoint_from(nullptr, seq_ids_, seq_pages_, ring);
}

void Model::save_checkpoint_from(const char* area, const std::vector<int32_t>& ids, const std::vector<int32_t>& pages,
                                 bool ring) {
  const int pos = int(ids.size());
  if (!cache_on_ || pos == 0) return;
  for (auto& c : ckpts_)
    if (c.pos() == pos && c.ring >= ring && std::equal(c.ids.begin(), c.ids.end(), ids.begin())) {
      c.last_use = ++use_clock_;
      return;
    }
  // A free allocated slot; else allocate another one; else evict the least recently used checkpoint.
  int slot = -1;
  for (;;) {
    for (size_t i = 0; i < slot_used_.size() && slot < 0; i++)
      if (slot_mem_[i] && !slot_used_[i]) slot = int(i);
    if (slot >= 0) break;
    if (!grow_state_cache(1) && !evict_checkpoint()) fail("checkpoint: no state slot");
  }
  Checkpoint ck;
  ck.ids = ids;
  ck.pages.assign(pages.begin(), pages.begin() + ceil_div(pos, kPageTokens));
  for (int32_t p : ck.pages) pool_.ref(p);
  ck.slot = slot;
  ck.ring = ring;
  ck.last_use = ++use_clock_;
  slot_used_[slot] = true;
  // State -> staging (device, fast) -> host slot on the copy stream, overlapping what follows.
  Q27_CUDA(cudaStreamWaitEvent(s_, stage_ev_));
  if (area)  // (a parking area has the checkpoint layout)
    Q27_CUDA(cudaMemcpyAsync(stage_, area, ckpt_bytes_, cudaMemcpyDeviceToDevice, s_));
  else
    for (const Region& r : seq_regions(stage_)) Q27_CUDA(cudaMemcpyAsync(r.area, r.dev, r.bytes, cudaMemcpyDeviceToDevice, s_));
  Q27_CUDA(cudaEventRecord(stage_in_ev_, s_));
  Q27_CUDA(cudaStreamWaitEvent(cs_, stage_in_ev_));
  Q27_CUDA(cudaMemcpyAsync(slot_mem_[slot], stage_, ckpt_bytes_, cudaMemcpyDeviceToHost, cs_));
  Q27_CUDA(cudaEventRecord(stage_ev_, cs_));
  ckpts_.push_back(std::move(ck));
  cstats_.saved++;
}

int Model::prefix_begin(const int32_t* ids, int n, const int32_t* at, int nat, bool spec) {
  reset();
  cstats_.lookups++;
  cstats_.prompt_tokens += n;
  if (!cache_on_) return 0;
  const Checkpoint* best = nullptr;
  for (auto& c : ckpts_)
    if (c.pos() < n && (!spec || c.ring) && (!best || c.pos() > best->pos()) &&
        std::equal(c.ids.begin(), c.ids.end(), ids))
      best = &c;
  int cached = 0;
  if (best) {
    cached = best->pos();
    const_cast<Checkpoint*>(best)->last_use = ++use_clock_;
    // The state from its host slot (after its device -> host copy), before anything can evict it.
    Q27_CUDA(cudaStreamSynchronize(cs_));
    const char* src = slot_mem_[best->slot];
    Q27_CUDA(cudaMemcpyAsync(ssm_, src, ssm_bytes_, cudaMemcpyHostToDevice, s_));
    Q27_CUDA(cudaMemcpyAsync(conv_, src + ssm_bytes_, conv_bytes_, cudaMemcpyHostToDevice, s_));
    if (draft_) {
      const size_t rb = draft_->ring_layers_bytes();
      Q27_CUDA(cudaMemcpyAsync(draft_->ring_k(), src + ssm_bytes_ + conv_bytes_, rb, cudaMemcpyHostToDevice, s_));
      Q27_CUDA(cudaMemcpyAsync(draft_->ring_v(), src + ssm_bytes_ + conv_bytes_ + rb, rb, cudaMemcpyHostToDevice, s_));
    }
    Q27_CUDA(cudaStreamSynchronize(s_));
    // Full pages are shared; the partial last page is copied (the sequence writes its later rows).
    const int full = cached / kPageTokens;
    std::vector<int32_t> pages = best->pages;
    for (int32_t p : pages) pool_.ref(p);
    seq_pages_.assign(pages.begin(), pages.begin() + full);
    seq_ids_.assign(ids, ids + cached);
    ring_valid_ = best->ring;
    best = nullptr;  // (may be evicted below)
    if (full < int(pages.size())) {
      int p = pool_.alloc();
      while (p < 0 && evict_checkpoint()) p = pool_.alloc();
      if (p < 0) {  // (the caller retries after another sequence gave pages back; this one is reset then)
        for (size_t i = full; i < pages.size(); i++) pool_.unref(pages[i]);
        throw KvFull("KV pool exhausted");
      }
      for (int sl = 0; sl < kKvSlices; sl++)
        Q27_CUDA(cudaMemcpyAsync(kv_page(p, sl), kv_page(pages[full], sl), kv_page_bytes_, cudaMemcpyDeviceToDevice, s_));
      seq_pages_.push_back(p);
    }
    for (size_t i = full; i < pages.size(); i++) pool_.unref(pages[i]);
    upload_pages(page_table_, seq_pages_.data(), seq_pages_.size());
    pos_ = cached;
    Q27_CUDA(cudaMemcpyAsync(pos_dev_, &pos_, 4, cudaMemcpyHostToDevice, s_));
    if (hist_) copy_hist(hist_, ids, cached);
    Q27_CUDA(cudaStreamSynchronize(s_));
    cstats_.hits++;
    cstats_.cached_tokens += cached;
  }
  // Checkpoints of this prefill: the requested positions, and the last few interval multiples that are
  // more than 64 tokens away from the others and from both ends (a chunk of <= 64 tokens takes the
  // decode GEMMs, so a resumed prefill would no longer match an uncut one bit for bit).
  // Positions are rounded down to multiples of 64 (the GDN chunks): a prefill resumed there matches
  // one that ran through.
  for (int i = 0; i < nat; i++)
    if (const int32_t p = at[i] / kGdnChunk * kGdnChunk; p > cached && p < n) plan_.push_back(p);
  std::sort(plan_.begin(), plan_.end());
  plan_.erase(std::unique(plan_.begin(), plan_.end()), plan_.end());
  if (const int iv = opt_.checkpoint_interval; iv > 0) {
    std::vector<int32_t> fixed = plan_;
    fixed.push_back(cached);
    fixed.push_back(n);
    Q27_CHECK(iv % kGdnChunk == 0, "checkpoint_interval must be a multiple of 64");
    for (int p = (n - 1) / iv * iv, k = 0; p > cached && k < 4; p -= iv, k++)
      if (std::all_of(fixed.begin(), fixed.end(), [&](int32_t q) { return std::abs(q - p) > kSkinnyMaxM; }))
        plan_.push_back(p);
    std::sort(plan_.begin(), plan_.end());
  }
  return cached;
}

void Model::checkpoint() {
  spec_flush();
  save_checkpoint(ring_valid_);
}

void Model::cache_stats(int64_t* out) const {
  out[0] = cstats_.lookups, out[1] = cstats_.hits, out[2] = cstats_.prompt_tokens, out[3] = cstats_.cached_tokens;
  out[4] = cstats_.saved, out[5] = cstats_.evicted, out[6] = int64_t(ckpts_.size()), out[7] = pool_.free_pages();
  // A page only checkpoints hold has as many references as checkpoints listing it (the scheduler counts these
  // as room for a new request, P7-14).
  std::vector<int32_t> n(pool_.pages(), 0);
  int64_t ckpt_only = 0;
  for (const auto& c : ckpts_)
    for (int32_t p : c.pages) ckpt_only += ++n[p] == pool_.refs(p);
  out[8] = ckpt_only, out[9] = pool_.pages() - 1, out[10] = park_pages_;
  out[11] = int64_t(std::count_if(slot_mem_.begin(), slot_mem_.end(), [](char* m) { return m != nullptr; }));
}

std::string Model::memory_report() const {
  std::string out;
  for (const Pool* p : {&dev_weights_, &host_weights_, &kv_, &state_, &act_, &host_cache_, &host_mm_}) {
    out += "pool\t" + p->name() + "\t" + (p->host() ? "1" : "0") + "\t" + std::to_string(p->size()) + "\n";
    for (const auto& [what, bytes] : p->items())
      out += "item\t" + p->name() + "\t" + what + "\t" + std::to_string(bytes) + "\n";
  }
  out += "slots\t" + std::to_string(slot_used_.size()) + "\t" + std::to_string(ckpt_bytes_) + "\n";
  return out;
}

void Model::device_memory(int64_t* out) const {
  size_t free = 0, total = 0;
  Q27_CUDA(cudaMemGetInfo(&free, &total));
  out[0] = int64_t(free), out[1] = int64_t(total), out[2] = int64_t(vram_.free_at_start);
  out[3] = int64_t(vram_.budget);
}

int Model::prefill_cut(int T, bool last) const {
  for (int32_t p : plan_)
    if (p > pos_) {
      if (p - pos_ <= T) return p - pos_;
      break;
    }
  // Otherwise end at a multiple of 64 positions (the GDN chunks, gdn.cu): where a prefill started
  // then does not change its cuts, and a prefill resumed at a multiple of 64 matches an uncut one.
  return last ? T : (pos_ + T) / kGdnChunk * kGdnChunk - pos_;
}


namespace {
// Debug (Q27_SKINNY_CHECK=1): runs `cutlass` into a scratch buffer next to `skinny` and logs the difference.
template <class FC, class FS>
void check_skinny(const char* what, int T, int N, float* out, bool residual, cudaStream_t s, FC cutlass, FS skinny) {
  static float *c0 = nullptr, *c1 = nullptr;
  if (!c0) {
    Q27_CUDA(cudaMalloc(&c0, size_t(8) * 248320 * 4));
    Q27_CUDA(cudaMalloc(&c1, size_t(8) * 248320 * 4));
  }
  size_t n = size_t(T) * N;
  Q27_CUDA(cudaMemcpyAsync(c0, out, n * 4, cudaMemcpyDeviceToDevice, s));
  cutlass(residual ? c0 : nullptr, c1);
  skinny();
  std::vector<float> a(n), b(n);
  Q27_CUDA(cudaMemcpyAsync(a.data(), c1, n * 4, cudaMemcpyDeviceToHost, s));
  Q27_CUDA(cudaMemcpyAsync(b.data(), out, n * 4, cudaMemcpyDeviceToHost, s));
  Q27_CUDA(cudaStreamSynchronize(s));
  double num = 0, den = 0;
  int worst_t = -1;
  double worst = 0;
  for (int t = 0; t < T; t++) {
    double nt = 0, dt = 0;
    for (int i = 0; i < N; i++) {
      double d = double(a[size_t(t) * N + i]) - b[size_t(t) * N + i];
      nt += d * d, dt += double(a[size_t(t) * N + i]) * a[size_t(t) * N + i];
    }
    num += nt, den += dt;
    if (nt / (dt + 1e-30) > worst) worst = nt / (dt + 1e-30), worst_t = t;
  }
  double rel = sqrt(num / (den + 1e-30));
  if (rel > 1e-4) log("skinny check %s T=%d N=%d: rel %.3g (worst token %d: %.3g)", what, T, N, rel, worst_t, sqrt(worst));
}
}  // namespace

// Debug (Q27_SKIP=norm,gemm,gemm4,gemm8,gdn,attn,silu): leaves out kernels of a kind to measure what they
// cost inside the decode step. The results are wrong; only for timing.
bool Model::skip(const char* kind) const {
  if (skip_.empty()) return false;
  return (","s + skip_ + ",").find(","s + kind + ",") != std::string::npos;
}

QuantOut Model::q8(float in_scale) const { return {QuantOut::kFp8, in_scale, aq_, nullptr}; }
QuantOut Model::q4(float in_scale) const { return {QuantOut::kFp4, in_scale, aq_, asf_}; }

void Model::quant_in(const LinearQ& l, const float* x, int T, int K) {
  if (!l.fp4) return;
  mark("quant fp4");
  quant_fp4(x, T, K, l.in_scale(), aq_, asf_, s_);
}

// GEMMs on the activation already quantized into aq_ (and asf_). Fused projections (several alpha
// segments) only run on the skinny path.
void Model::gemm_q(const char* what, const LinearFp8& l, int T, float* out, bool residual) {
  mark(what);
  if (skip("gemm") || skip("gemm8")) return;
  const float* c = residual ? out : nullptr;
  const float beta = residual ? 1.f : 0.f;
  if (T <= kSkinnyMaxM && skinny_) {
    auto run = [&] { skinny_fp8(T, l.out, l.in, aq_, l.w, l.alpha, c, beta, out, skinny_ws_, kSkinnyWsBytes, s_); };
    if (skinny_check_ && l.alpha.count == 1)
      check_skinny(what, T, l.out, out, residual, s_, [&](const float* cc, float* d) {
        gemm_fp8(T, l.out, l.in, aq_, l.w, l.alpha.alpha[0], cc, cc ? 1.f : 0.f, d, ws_, ws_bytes_, s_);
      }, run);
    else
      run();
    return;
  }
  Q27_CHECK(l.alpha.count == 1, std::string(what) + ": fused projections need the skinny path");
  gemm_fp8(T, l.out, l.in, aq_, l.w, l.alpha.alpha[0], c, beta, out, ws_, ws_bytes_, s_);
}

void Model::gemm_q(const char* what, const LinearFp4& l, int T, float* out, bool residual) {
  mark(what);
  if (skip("gemm") || skip("gemm4")) return;
  const float* c = residual ? out : nullptr;
  const float beta = residual ? 1.f : 0.f;
  if (T <= kSkinnyMaxM && skinny_) {
    auto run = [&] {
      skinny_fp4(T, l.out, l.in, aq_, asf_, l.w, l.sf, l.alpha, c, beta, out, skinny_ws_, kSkinnyWsBytes, s_);
    };
    if (skinny_check_ && l.alpha.count == 1)
      check_skinny(what, T, l.out, out, residual, s_, [&](const float* cc, float* d) {
        gemm_fp4(T, l.out, l.in, aq_, asf_, l.w, l.sf, l.alpha.alpha[0], cc, cc ? 1.f : 0.f, d, ws_, ws_bytes_, s_);
      }, run);
    else
      run();
    return;
  }
  Q27_CHECK(l.alpha.count == 1, std::string(what) + ": fused projections need the skinny path");
  gemm_fp4(T, l.out, l.in, aq_, asf_, l.w, l.sf, l.alpha.alpha[0], c, beta, out, ws_, ws_bytes_, s_);
}

// Up to 8 tokens (decode, speculative verify) run the fused projections (qkv|z, q|k|v, gate|up) on the
// skinny path; their outputs are rows of the concatenation, so the parts are offsets with the fused
// row stride. Larger chunks run the projections separately (CUTLASS).
// Fused projections (strided outputs: decode-style attention). A batched verify (up to 64 rows, Phase 7)
// takes them too: its rows must run the GEMMs of a single sequence's verify (another shape splits K
// differently).
bool Model::fused(int T) const { return (batch_n_ ? kLaneRows : T) <= kDecodeMaxT && skinny_; }

// Captured target features for the draft: the input of layer i is the output of layer i - 1.
bf16* Model::cap_for(int i) const {
  if (!cap_) return nullptr;
  for (int f = 0; f < ddims::kCtxFeatures; f++)
    if (ddims::kCtxLayers[f] + 1 == i) return cap_ + f * kHidden;
  return nullptr;
}

GdnSpec Model::gdn_spec(int gi) const {
  GdnSpec sp;
  for (int p = 0; p < 2; p++) {
    float* base = rep_ + (size_t(p) * kGdnLayers + gi) * rep_rows_ * kRepTokenFloats;
    sp.raw[p] = base;
    sp.k[p] = sp.raw[p] + size_t(rep_rows_) * kConvDim;
    sp.v[p] = sp.k[p] + size_t(rep_rows_) * kKHeads * kKDim;
    sp.ab[p] = sp.v[p] + size_t(rep_rows_) * kVHeads * kVDim;
  }
  sp.par = par_;
  sp.ncommit = ncommit_;
  sp.cpath = cpath_;
  sp.conv_state = conv_ + size_t(gi) * kConvDim * (kConvK - 1);
  for (int b = 0; b < 2; b++) sp.st[b] = ssm_buf_[b] + size_t(gi) * kVHeads * kKDim * kVDim;
  sp.sbuf = sbuf_;
  sp.tree = tree_;
  sp.paths = verify_paths_;
  return sp;
}

void Model::gdn(int i, int T) {
  const auto& L = w_.layers[i];
  const auto& G = L.gdn;
  int gi = i - i / kAttnEvery;  // index among the GDN layers
  float* conv = conv_ + size_t(gi) * kConvDim * (kConvK - 1);
  float* ssm = ssm_ + size_t(gi) * kVHeads * kKDim * kVDim;
  mark("norm+quant");
  if (!skip("norm"))  // h_: input of in_proj_a / b
    rmsnorm_quant(x_, L.in_norm, T, kEps, h_, qin(G.qkv), s_, cap_for(i), ddims::kCtxFeatures * kHidden);
  float *qkv = big1_, *z = big2_;
  int ld = kConvDim, ldz = kVHeads * kVDim;
  if (fused(T)) {
    gemm_q("gdn qkvz", G.qkvz, T, big1_, false);
    z = big1_ + kConvDim;
    ld = ldz = G.qkvz.out();
  } else {
    gemm_q("gdn qkv", G.qkv, T, big1_, false);
    gemm_q("gdn z", G.z, T, big2_, false);
  }
  if (verify_) {
    // (a batched verify: the lanes' blocks of kLaneRows rows, each on its own state)
    const GdnSpec sp = batch_n_ ? gdn_spec_lanes(gi) : gdn_spec(gi);
    const int Tv = batch_n_ ? kLaneRows : T;
    mark("gdn prep");
    if (!skip("gdn")) gdn_verify_prep(qkv, ld, Tv, conv, G.conv, h_, G.a, G.b, q_, sp, s_);
    mark("gdn recurrent+norm");
    if (!skip("gdn"))
      gdn_verify(q_, Tv, G.A_log, G.dt_bias, ssm, z, ldz, G.norm, kEps, n_, G.out.fp4 ? nullptr : aq_,
                 G.out.in_scale(), sp, s_);
    quant_in(G.out, n_, T, kVHeads * kVDim);
    gemm_q("gdn out", G.out, T, x_, true);
    return;
  }
  if (T == 1) {
    mark("gdn prep");
    if (!skip("gdn")) gdn_decode_prep(qkv, conv, G.conv, h_, G.a, G.b, q_, k_, v_, a_, b_, s_);
    mark("gdn recurrent+norm");
    if (!skip("gdn"))
      gdn_decode(q_, k_, v_, a_, b_, G.A_log, G.dt_bias, ssm, z, G.norm, kEps, n_, G.out.fp4 ? nullptr : aq_,
                 G.out.in_scale(), s_);
    quant_in(G.out, n_, T, kVHeads * kVDim);
    gemm_q("gdn out", G.out, T, x_, true);
    return;
  }
  mark("gdn a,b");
  linear_bf16_pair(h_, G.a, G.b, T, kVHeads, kHidden, a_, b_, s_);
  mark("gdn prep");
  gdn_prep(qkv, ld, conv, G.conv, a_, b_, G.A_log, G.dt_bias, T, q_, k_, v_, g_, beta_, s_);
  mark("gdn recurrent");
  // Chunks of 64 positions on tensor cores (Q27_GDN_CHUNK=0: token by token); workspace behind z.
  static const bool chunked = !getenv("Q27_GDN_CHUNK") || atoi(getenv("Q27_GDN_CHUNK")) != 0;
  if (chunked)
    gdn_chunked(q_, k_, v_, g_, beta_, T, pos_, ssm, o_, big2_ + size_t(opt_.prefill_chunk) * kVHeads * kVDim, s_);
  else
    gdn_recurrent(q_, k_, v_, g_, beta_, T, ssm, o_, s_);
  mark("gdn gated norm");
  if (G.out.fp4) {  // fp32 in place, then NVFP4
    gdn_gated_norm(o_, z, ldz, G.norm, T, kEps, o_, s_);
    quant_in(G.out, o_, T, kVHeads * kVDim);
  } else {
    gdn_gated_norm(o_, z, ldz, G.norm, T, kEps, nullptr, s_, aq_, G.out.in_scale());  // (only the FP8 copy is used)
  }
  gemm_q("gdn out", G.out, T, x_, true);
}

void Model::attn(int i, int T) {
  const auto& L = w_.layers[i];
  const auto& A = L.attn;
  KvLayer kv = kv_layers_[i / kAttnEvery];
  mark("norm+quant");
  if (!skip("norm"))
    rmsnorm_quant(x_, L.in_norm, T, kEps, nullptr, qin(A.q), s_, cap_for(i), ddims::kCtxFeatures * kHidden);
  // qg [T][24][512]: query | gate per head; k, v [T][4][256].
  float *qg = big1_, *k = k_, *v = q_;  // (q_ is free in attention layers)
  int ldq = kHeads * 2 * kHeadDim, ldkv = kKvHeads * kHeadDim;
  if (fused(T)) {
    gemm_q("attn qkv", A.qkv, T, big1_, false);
    k = big1_ + kHeads * 2 * kHeadDim;
    v = k + kKvHeads * kHeadDim;
    ldq = ldkv = A.qkv.out();
  } else {
    gemm_q("attn q", A.q, T, big1_, false);
    gemm_q("attn k", A.k, T, k_, false);
    gemm_q("attn v", A.v, T, q_, false);
  }
  if (batch_n_) {  // batched verify: each lane's rows on its own positions and pages
    mark("attn prep");
    if (!skip("attn")) attn_prep_lanes(qg, ldq, k, v, ldkv, A.q_norm, A.k_norm, bsl_, kv, opt_.kv_type, v_, s_);
    mark("attn core");
    if (!skip("attn"))
      attention_decode_lanes(v_, qg, ldq, bsl_, kv, opt_.kv_type, o_, attn_ws_, kAttnDecodeWsBytes, s_,
                             A.o.fp4 ? nullptr : aq_, A.o.in_scale());
    quant_in(A.o, o_, T, kHeads * kHeadDim);
    gemm_q("attn o", A.o, T, x_, true);
    return;
  }
  mark("attn prep");
  const SpecTree* tree = verify_ && verify_paths_ > 1 ? tree_ : nullptr;  // (a chain: row t at depth t)
  if (!skip("attn"))
    attn_prep(qg, ldq, k, v, ldkv, A.q_norm, A.k_norm, T, pos_dev_, kv, opt_.kv_type, page_table_, v_, s_, tree,
              {rope_delta_, mpos_on_ && !verify_ ? mpos_ : nullptr});
  mark("attn core");
  if (skip("attn")) {
  } else if (T <= kDecodeMaxT) {
    attention_decode(v_, qg, ldq, T, pos_dev_, kv, opt_.kv_type, page_table_, o_, attn_ws_, kAttnDecodeWsBytes, s_,
                     A.o.fp4 ? nullptr : aq_, A.o.in_scale(), tree);
    quant_in(A.o, o_, T, kHeads * kHeadDim);
  } else {
    Q27_CHECK(ldq == kHeads * 2 * kHeadDim, "attention: unexpected q stride");
    // (big2_ is free in attention layers.)
    attention(v_, qg, T, pos_, kv, opt_.kv_type, page_table_, o_, big2_, size_t(opt_.prefill_chunk) * kInter * 4, s_,
              fa_stage_, fa_stage_bytes_);
    mark("quant fp8");
    if (A.o.fp4)
      quant_in(A.o, o_, T, kHeads * kHeadDim);
    else
      quant_fp8(o_, int64_t(T) * kHeads * kHeadDim, A.o.in_scale(), aq_, s_);
  }
  gemm_q("attn o", A.o, T, x_, true);
}

void Model::mlp(int i, int T) {
  const auto& L = w_.layers[i];
  mark("norm+quant");
  if (!skip("norm")) rmsnorm_quant(x_, L.post_norm, T, kEps, nullptr, q4(L.gate.in_scale), s_);
  float *gate = big1_, *up = big2_;
  int64_t ld = kInter;
  if (fused(T)) {
    gemm_q("mlp gate_up", L.gate_up, T, big1_, false);
    up = big1_ + kInter;
    ld = 2 * kInter;
  } else if (mlp_bf16_ && T > kSkinnyMaxM) {
    // Prefill: gate / up in BF16 (they only feed silu * up and its FP4 quantization).
    bf16 *gb = reinterpret_cast<bf16*>(big1_), *ub = reinterpret_cast<bf16*>(big2_);
    for (auto [what, l, d] : {std::tuple{"mlp gate", &L.gate, gb}, std::tuple{"mlp up", &L.up, ub}}) {
      mark(what);
      if (!skip("gemm") && !skip("gemm4"))
        gemm_fp4_bf16(T, l->out, l->in, aq_, asf_, l->w, l->sf, l->alpha.alpha[0], d, ws_, ws_bytes_, s_);
    }
    mark("mlp silu+quant");
    if (!skip("silu")) silu_mul_quant_fp4(gb, ub, kInter, T, kInter, L.down.in_scale, aq_, asf_, s_);
    gemm_q("mlp down", L.down, T, x_, true);
    return;
  } else {
    gemm_q("mlp gate", L.gate, T, big1_, false);
    gemm_q("mlp up", L.up, T, big2_, false);
  }
  mark("mlp silu+quant");
  if (!skip("silu")) silu_mul_quant_fp4(gate, up, ld, T, kInter, L.down.in_scale, aq_, asf_, s_);
  gemm_q("mlp down", L.down, T, x_, true);
}

void Model::chunk(const int32_t* ids_dev, int T, bool advance) {
  Q27_CHECK(batch_n_ || pos_ + T <= max_pages_ * kPageTokens, "sequence longer than max_tokens");
  mark("embed");
  embed_gather(w_.embed, ids_dev, T, kHidden, x_, s_, mm_, kVocab);
  for (int i = 0; i < kLayers; i++) {
    if (is_attn(i))
      attn(i, T);
    else
      gdn(i, T);
    mlp(i, T);
  }
  if (advance) add_pos(pos_dev_, T, s_);  // the host pos_ is advanced by the caller (chunk may be captured in a graph)
}

// ---------------------------------------------------------------- images (P4-12)

void Model::mm_set(const int32_t* spans, int n) {
  Q27_CHECK(n == 0 || opt_.vision, "images need the vision tower (VISION)");
  batch_exit();
  mm_spans_.clear();
  int rows = 0, delta = 0, end = 0;
  for (int i = 0; i < n; i++) {
    MmSpan sp{spans[3 * i], spans[3 * i + 1], spans[3 * i + 2], rows};
    Q27_CHECK(sp.gh > 0 && sp.gw > 0 && sp.start >= end, "images: bad or overlapping spans");
    rows += sp.len();
    end = sp.start + sp.len();
    delta += std::max(sp.gh, sp.gw) - sp.len();  // the rotary position advances by max(gh, gw)
    mm_spans_.push_back(sp);
  }
  Q27_CHECK(rows <= opt_.image_tokens, "images: more image tokens than IMAGE_TOKENS");
  Q27_CUDA(cudaMemcpyAsync(rope_delta_, &delta, 4, cudaMemcpyHostToDevice, s_));
  Q27_CUDA(cudaStreamSynchronize(s_));
}

void Model::mm_encode(int i, const float* pixels) {
  Q27_CHECK(opt_.vision && i >= 0 && i < int(mm_spans_.size()), "mm_encode: no such image");
  const MmSpan& sp = mm_spans_[i];
  const int gh = sp.gh * vdims::kMerge, gw = sp.gw * vdims::kMerge;
  Q27_CHECK(int64_t(gh) * gw * vdims::kPatchPixels * vdims::kPatchPixels <= opt_.image_max_pixels,
            "image larger than IMAGE_MAX_PIXELS");
  spec_flush();
  Q27_CUDA(cudaStreamSynchronize(s_));  // (the workspace overlays the prefill activations)
  mark("vision");
  vision_encode(w_.vision, pixels, gh, gw, mm_ + size_t(sp.row) * kHidden, vis_ws_, vis_ws_bytes_, s_);
  Q27_CUDA(cudaStreamSynchronize(s_));
  profile_collect();
}

void Model::mm_embeds(int i, bf16* out) {
  Q27_CHECK(opt_.vision && i >= 0 && i < int(mm_spans_.size()), "mm_embeds: no such image");
  const MmSpan& sp = mm_spans_[i];
  memcpy(out, mm_ + size_t(sp.row) * kHidden, size_t(sp.len()) * kHidden * sizeof(bf16));
}

void Model::mm_put(int i, const bf16* in) {
  Q27_CHECK(opt_.vision && i >= 0 && i < int(mm_spans_.size()), "mm_put: no such image");
  const MmSpan& sp = mm_spans_[i];
  memcpy(mm_ + size_t(sp.row) * kHidden, in, size_t(sp.len()) * kHidden * sizeof(bf16));
}

void Model::upload_ids(const int32_t* ids, int T) {
  mpos_on_ = false;
  if (mm_spans_.empty()) {
    Q27_CUDA(cudaMemcpyAsync(ids_, ids, size_t(T) * 4, cudaMemcpyHostToDevice, s_));
    return;
  }
  // Image rows gather row kVocab + r of the embedding table (embed_gather). Chunks before the end of the
  // last image take explicit M-RoPE positions; later rows are index + the final delta (rope_delta_).
  std::vector<int32_t> tr(ids, ids + T);
  mpos_on_ = pos_ < mm_spans_.back().start + mm_spans_.back().len();
  std::vector<int32_t> mp(mpos_on_ ? size_t(T) * 3 : 0);
  int delta = 0;
  size_t k = 0;
  for (int t = 0; t < T; t++) {
    const int p = pos_ + t;
    while (k < mm_spans_.size() && mm_spans_[k].start + mm_spans_[k].len() <= p) {
      delta += std::max(mm_spans_[k].gh, mm_spans_[k].gw) - mm_spans_[k].len();
      k++;
    }
    const bool img = k < mm_spans_.size() && p >= mm_spans_[k].start;
    if (img) {
      const MmSpan& sp = mm_spans_[k];
      const int r = p - sp.start, base = sp.start + delta;
      tr[t] = kVocab + sp.row + r;
      if (mpos_on_) mp[3 * t] = base, mp[3 * t + 1] = base + r / sp.gw, mp[3 * t + 2] = base + r % sp.gw;
    } else if (mpos_on_) {
      mp[3 * t] = mp[3 * t + 1] = mp[3 * t + 2] = p + delta;
    }
  }
  Q27_CUDA(cudaMemcpyAsync(ids_, tr.data(), size_t(T) * 4, cudaMemcpyHostToDevice, s_));
  if (mpos_on_) Q27_CUDA(cudaMemcpyAsync(mpos_, mp.data(), mp.size() * 4, cudaMemcpyHostToDevice, s_));
}

void Model::copy_hist(int32_t* dst, const int32_t* ids, int n) {
  std::vector<int32_t> h(ids, ids + n);
  for (int32_t& t : h)
    if (t >= kVocab || t < 0) t = kImagePad;
  Q27_CUDA(cudaMemcpyAsync(dst, h.data(), size_t(n) * 4, cudaMemcpyHostToDevice, s_));
}

void Model::logits(const float* x_rows, int rows) {
  mark("norm+quant");
  rmsnorm_quant(x_rows, w_.final_norm, rows, kEps, nullptr, q4(w_.lm_head.in_scale), s_);
  gemm_q("lm_head", w_.lm_head, rows, logits_, false);
}

void Model::forward_topk(const int32_t* ids, int n, int k,
                         const std::function<void(int, const int32_t*, const float*)>& cb) {
  spec_flush();
  std::vector<int32_t> hid(size_t(opt_.logits_rows) * k);
  std::vector<float> hlp(hid.size());
  for (int c0 = 0, T = 0; c0 < n; c0 += T) {
    T = n - c0 <= opt_.prefill_chunk ? n - c0 : (pos_ + opt_.prefill_chunk) / kGdnChunk * kGdnChunk - pos_;
    int p0 = pos_;
    ensure_pages(pos_ + T);
    upload_ids(ids + c0, T);
    chunk(ids_, T);
    pos_ += T;
    seq_ids_.insert(seq_ids_.end(), ids + c0, ids + c0 + T);
    ring_valid_ = false;
    for (int r0 = 0; r0 < T; r0 += opt_.logits_rows) {
      int rows = std::min(opt_.logits_rows, T - r0);
      logits(x_ + size_t(r0) * kHidden, rows);
      logsoftmax_topk(logits_, rows, kVocab, k, topk_ids_, topk_logp_, s_);
      Q27_CUDA(cudaMemcpyAsync(hid.data(), topk_ids_, size_t(rows) * k * 4, cudaMemcpyDeviceToHost, s_));
      Q27_CUDA(cudaMemcpyAsync(hlp.data(), topk_logp_, size_t(rows) * k * 4, cudaMemcpyDeviceToHost, s_));
      Q27_CUDA(cudaStreamSynchronize(s_));
      profile_collect();
      for (int r = 0; r < rows; r++) cb(p0 + r0 + r, hid.data() + size_t(r) * k, hlp.data() + size_t(r) * k);
    }
  }
  mpos_on_ = false;
}

int32_t Model::forward_sample(const int32_t* ids, int n, float temperature, int top_k, float top_p, float uniform) {
  Q27_CHECK(n > 0, "forward_sample: no tokens");
  spec_flush();
  if (n == 1 && graphs_ && !cap_) return decode_step(ids[0], temperature, top_k, top_p, uniform);
  int last = 0;
  prompt_chunks(ids, n, n, false, &last);
  logits(x_ + size_t(last) * kHidden, 1);
  Q27_CUDA(cudaMemcpyAsync(uniform_, &uniform, 4, cudaMemcpyHostToDevice, s_));
  mark("sample");
  sample(logits_, 1, kVocab, temperature, top_k, top_p, uniform_, sampled_, s_);
  int32_t out = 0;
  Q27_CUDA(cudaMemcpyAsync(&out, sampled_, 4, cudaMemcpyDeviceToHost, s_));
  Q27_CUDA(cudaStreamSynchronize(s_));
  profile_collect();
  return out;
}

int Model::prefill(const int32_t* ids, int n, int budget, bool spec) {
  Q27_CHECK(!spec || draft_, "prefill: no draft model loaded");
  spec_flush();
  cap_ = spec ? cap_buf_ : nullptr;
  int done = 0;
  try {
    done = prompt_chunks(ids, n, budget, true, nullptr);
  } catch (...) {
    cap_ = nullptr;
    throw;
  }
  cap_ = nullptr;
  return done;
}

// The chunks of a prompt (ids[0..n): the rest of it after pos_), as one forward_sample would cut them.
// keep_tail: stop where at most one prefill chunk is left (prefill()); stop after `budget` tokens.
// *last: the row of the last token in the activations.
int Model::prompt_chunks(const int32_t* ids, int n, int budget, bool keep_tail, int* last) {
  const int end = pos_ + n;
  int c0 = 0;
  while (c0 < n && c0 < budget && !(keep_tail && n - c0 <= opt_.prefill_chunk)) {
    // Chunks stop at the planned checkpoints (prefix_begin).
    const int T = prefill_cut(std::min(opt_.prefill_chunk, n - c0), opt_.prefill_chunk >= n - c0);
    // Decode-path kernels read per-layer state (GDN / conv state, the position) before their PDL wait,
    // which is only safe once the previous chunk has finished (decode steps end with a sync anyway).
    Q27_CUDA(cudaStreamSynchronize(s_));
    ensure_pages(pos_ + T);
    upload_ids(ids + c0, T);
    chunk(ids_, T);
    // spec_begin: the draft's context for the positions in its window before the end of the prompt
    // and before every checkpoint.
    if (cap_) {
      int target = end;
      for (int32_t p : plan_)
        if (p >= pos_ + T) {
          target = std::min(target, int(p));
          break;
        }
      const int skip = std::clamp(target - ddims::kWindow - pos_, 0, T);
      draft_->append_ctx(cap_ + size_t(skip) * ddims::kCtxFeatures * kHidden, T - skip, pos_dev_, skip - T, nullptr,
                         skinny_ws_, kSkinnyWsBytes, big1_, size_t(opt_.prefill_chunk) * kInter * 4, s_);
    }
    pos_ += T;
    seq_ids_.insert(seq_ids_.end(), ids + c0, ids + c0 + T);
    ring_valid_ = cap_ != nullptr;
    if (!plan_.empty() && plan_.front() <= pos_) {
      if (plan_.front() == pos_) save_checkpoint(ring_valid_);
      while (!plan_.empty() && plan_.front() <= pos_) plan_.erase(plan_.begin());
    }
    c0 += T;
    if (last) *last = T - 1;
  }
  mpos_on_ = false;
  return c0;
}

// One decode token through a CUDA graph of the whole step (embedding .. sampling). The first call
// runs eagerly (one-time initialization of the kernels); the second captures the graph, which is
// reused while the sampling parameters stay the same.
int32_t Model::decode_step(int32_t id, float temperature, int top_k, float top_p, float uniform) {
  Q27_CHECK(pos_ + 1 <= max_pages_ * kPageTokens, "sequence longer than max_tokens");
  ensure_pages(pos_ + 1);
  seq_ids_.push_back(id);
  ring_valid_ = false;
  host_io_->id = id;
  host_io_->uniform = uniform;
  // The step's inputs and output live in pinned host memory that the kernels read / write directly
  // (UVA: embed reads the id, the sampler the uniform and writes the token): no copy commands, which
  // left the GPU idle between steps (Q27_DECODE_MEMCPY=1: the copies).
  HostIo* io = host_io_dev_;
  if (decode_memcpy_) {
    Q27_CUDA(cudaMemcpyAsync(ids_, &host_io_->id, 4, cudaMemcpyHostToDevice, s_));
    Q27_CUDA(cudaMemcpyAsync(uniform_, &host_io_->uniform, 4, cudaMemcpyHostToDevice, s_));
  }
  auto body = [&] {
    chunk(decode_memcpy_ ? ids_ : &io->id, 1);
    logits(x_, 1);
    sample(logits_, 1, kVocab, temperature, top_k, top_p, decode_memcpy_ ? uniform_ : &io->uniform,
           decode_memcpy_ ? sampled_ : &io->out, s_);
  };
  static const char* trace = getenv("Q27_TRACE");  // debug timeline (build with -DQ27_TRACE)
  if (trace) trace_step_begin(s_);
  const GraphKey key{temperature, top_k, top_p};
  if (!graph_warm_) {
    body();
    graph_warm_ = true;
  } else {
    if (!graph_exec_ || !(key == graph_key_) || graph_sbuf_ != sbuf_host_) {
      graph_exec_ = nullptr;
      for (auto& g : dgraphs_)
        if (g.key == key && g.sbuf == sbuf_host_) graph_exec_ = g.exec, g.used = ++graph_clock_;
    }
    if (graph_exec_) {
      graph_key_ = key;
      graph_sbuf_ = sbuf_host_;
    } else {
      cudaGraph_t graph;
      trace_clear_launches();
      Q27_CUDA(cudaStreamBeginCapture(s_, cudaStreamCaptureModeThreadLocal));
      prof_.capturing = true;
      prof_.graph.used = 0;
      body();
      mark(nullptr);  // end of the step
      prof_.capturing = false;
      Q27_CUDA(cudaStreamEndCapture(s_, &graph));
      Q27_CUDA(cudaGraphInstantiate(&graph_exec_, graph, 0));
      Q27_CUDA(cudaGraphDestroy(graph));
      graph_key_ = key;
      graph_sbuf_ = sbuf_host_;
      if (dgraphs_.size() >= size_t(kGraphCache)) {
        auto lru = std::min_element(dgraphs_.begin(), dgraphs_.end(), [](auto& a, auto& b) { return a.used < b.used; });
        Q27_CUDA(cudaGraphExecDestroy(lru->exec));
        dgraphs_.erase(lru);
      }
      dgraphs_.push_back({key, {}, sbuf_host_, graph_exec_, ++graph_clock_});
    }
    Q27_CUDA(cudaGraphLaunch(graph_exec_, s_));
  }
  if (decode_memcpy_) Q27_CUDA(cudaMemcpyAsync(&host_io_->out, sampled_, 4, cudaMemcpyDeviceToHost, s_));
  Q27_CUDA(cudaStreamSynchronize(s_));
  profile_collect(graph_exec_ != nullptr);
  static int trace_steps = 0;  // one step after warm-up (writing is slow); Q27_TRACE_STEP picks another
  static const int trace_at = getenv("Q27_TRACE_STEP") ? atoi(getenv("Q27_TRACE_STEP")) : 20;
  if (trace && ++trace_steps == trace_at) trace_step_end(trace);
  pos_ += 1;
  return host_io_->out;
}

// ---------------------------------------------------------------- speculative decoding

void Model::spec_flush() {
  batch_exit();
  if (spec_inflight_) {
    // A step launched ahead (spec_step) is dropped. It began by replaying the last returned step's
    // commit into the other state buffer and the conv state, and flipped sbuf, as a flush would; its
    // own rows are discarded: nothing pending, the position back to the host's. (Its KV, draft context
    // and history writes lie past the position and are overwritten later.)
    Q27_CUDA(cudaStreamSynchronize(s_));
    spec_inflight_ = false;
    set_sbuf(sbuf_host_ ^ 1);
    // Its accept step also made its bonus token the next anchor and took a step of the RNG and its
    // tokens from the budget: undo those, so that spec_step can continue the sequence later (Phase 6:
    // another sequence ran in between).
    int32_t nc = 0, budget = 0;
    uint64_t rng[2];
    Q27_CUDA(cudaMemcpyAsync(&nc, ncommit_, 4, cudaMemcpyDeviceToHost, s_));
    Q27_CUDA(cudaMemcpyAsync(&budget, budget_, 4, cudaMemcpyDeviceToHost, s_));
    Q27_CUDA(cudaMemcpyAsync(rng, rng_, sizeof rng, cudaMemcpyDeviceToHost, s_));
    Q27_CUDA(cudaStreamSynchronize(s_));
    budget += nc;
    rng[1] -= 1;
    Q27_CUDA(cudaMemcpyAsync(budget_, &budget, 4, cudaMemcpyHostToDevice, s_));
    Q27_CUDA(cudaMemcpyAsync(rng_, rng, sizeof rng, cudaMemcpyHostToDevice, s_));
    Q27_CUDA(cudaMemcpyAsync(block_ids_, &spec_last_, 4, cudaMemcpyHostToDevice, s_));
    Q27_CUDA(cudaMemcpyAsync(ids_ver_, &spec_last_, 4, cudaMemcpyHostToDevice, s_));
    const int32_t p = pos_;
    Q27_CUDA(cudaMemsetAsync(ncommit_, 0, 4, s_));
    Q27_CUDA(cudaMemcpyAsync(pos_dev_, &p, 4, cudaMemcpyHostToDevice, s_));
    Q27_CUDA(cudaStreamSynchronize(s_));
    spec_pending_ = false;
    return;
  }
  if (!spec_pending_) return;
  // Verify blocks of zero tokens: only the replay of the last commit (conv and recurrent states).
  for (int i = 0; i < kLayers; i++) {
    if (is_attn(i)) continue;
    const auto& G = w_.layers[i].gdn;
    const int gi = i - i / kAttnEvery;
    const GdnSpec sp = gdn_spec(gi);
    gdn_verify_prep(big1_, kConvDim, 0, conv_ + size_t(gi) * kConvDim * (kConvK - 1), G.conv, h_, G.a, G.b, q_, sp, s_);
    gdn_verify(q_, 0, G.A_log, G.dt_bias, ssm_ + size_t(gi) * kVHeads * kKDim * kVDim, big2_, kVHeads * kVDim, G.norm,
               kEps, n_, nullptr, 1.f, sp, s_);
  }
  // The replayed state went to the other buffer.
  set_sbuf(sbuf_host_ ^ 1);
  Q27_CUDA(cudaMemcpyAsync(sbuf_, &sbuf_host_, 4, cudaMemcpyHostToDevice, s_));
  Q27_CUDA(cudaMemsetAsync(ncommit_, 0, 4, s_));
  Q27_CUDA(cudaStreamSynchronize(s_));
  spec_pending_ = false;
}

int32_t Model::spec_begin(const int32_t* ids, int n, const SpecSampling& smp, uint64_t seed, float uniform,
                          int max_new) {
  Q27_CHECK(draft_, "spec_begin: no draft model loaded");
  spec_flush();
  cap_ = cap_buf_;
  int32_t first = 0;
  try {
    first = forward_sample(ids, n, smp.temperature, smp.top_k, smp.top_p, uniform);
  } catch (...) {
    cap_ = nullptr;
    throw;
  }
  cap_ = nullptr;
  spec_last_ = first;
  const uint64_t rng[2] = {seed, 0};
  Q27_CUDA(cudaMemcpyAsync(rng_, rng, sizeof rng, cudaMemcpyHostToDevice, s_));
  const int32_t budget = max_new > 0 ? max_new - 1 : (1 << 30);  // (the first token is out already)
  Q27_CUDA(cudaMemcpyAsync(budget_, &budget, 4, cudaMemcpyHostToDevice, s_));
  Q27_CUDA(cudaMemcpyAsync(sbuf_, &sbuf_host_, 4, cudaMemcpyHostToDevice, s_));
  // The token history for lookup: the whole sequence so far (this call may follow earlier tokens).
  Q27_CHECK(pos_ >= n, "spec_begin: inconsistent position");
  copy_hist(hist_ + (pos_ - n), ids, n);
  int32_t blk[kSpecBlock];
  for (int i = 0; i < kSpecBlock; i++) blk[i] = i == 0 ? first : ddims::kMaskId;
  Q27_CUDA(cudaMemcpyAsync(block_ids_, blk, sizeof blk, cudaMemcpyHostToDevice, s_));
  Q27_CUDA(cudaMemcpyAsync(ids_ver_, blk, sizeof blk, cudaMemcpyHostToDevice, s_));
  Q27_CUDA(cudaStreamSynchronize(s_));
  spec_smp_ = smp;
  std::copy(stop_, stop_ + 2, spec_smp_.stop);
  if (spec_smp_.paths <= 0) spec_smp_.paths = spec_paths_;
  return first;
}

// One step on the device: draft -> verify (8 rows, features captured) -> accept -> draft context ->
// position / parity / RNG advance.
void Model::spec_body(const SpecSampling& smp) {
  const auto& lm = w_.lm_head;
  Lookup lk = lookup_;
  lk.hist = hist_, lk.pos = pos_dev_, lk.best = lk_best_;
  mark("lookup");
  if (lk.on) spec_lookup(lk, ids_ver_, s_);
  mark("draft");
  draft_->propose(block_ids_, pos_dev_, w_.embed, lm.w, lm.sf, lm.in_scale, lm.alpha.alpha[0], smp, rng_, ids_ver_,
                  aq_, asf_, logits_, skinny_ws_, kSkinnyWsBytes, lk, s_);
  const int W = spec_rows(smp);
  verify_paths_ = smp.paths;
  verify_ = true;
  cap_ = cap_buf_;
  chunk(ids_ver_, W, false);
  verify_ = false;
  cap_ = nullptr;
  logits(x_, W);
  mark("accept");
  const int k = smp.temperature <= 0.f ? 1 : std::clamp(smp.top_k <= 0 ? 64 : smp.top_k, 1, 64);
  topk_rows(logits_, W, kVocab, k, tv_, ti_, s_);
  spec_accept(ids_ver_, block_ids_, tv_, ti_, k, draft_->choice, smp, rng_, ncommit_, cpath_, spec_out_dev_, hist_,
              pos_dev_, s_, par_, budget_);
  if (smp.paths > 1) kv_commit(kv_layers_dev_, kAttnLayers, opt_.kv_type, page_table_, pos_dev_, ncommit_, cpath_, s_);
  mark("draft context");
  draft_->append_ctx(cap_buf_, kSpecBlock, pos_dev_, 0, ncommit_, skinny_ws_, kSkinnyWsBytes, nullptr, 0, s_, cpath_);
  spec_advance(pos_dev_, ncommit_, par_, sbuf_, rng_, s_, budget_);
}

int Model::spec_step(int32_t* out, bool allow_ahead) {
  Q27_CHECK(draft_, "spec_step: no draft model loaded");
  batch_exit();
  const SpecSampling smp = spec_smp_;
  Q27_CHECK(pos_ + spec_rows(smp) <= max_pages_ * kPageTokens, "sequence longer than max_tokens");
  // Pages for this step's rows and for the one launched ahead (the position advances by <= 8).
  ensure_pages(pos_ + kSpecBlock + spec_rows(smp));
  auto same = [](const SpecSampling& a, const SpecSampling& b) {
    return a.temperature == b.temperature && a.top_k == b.top_k && a.top_p == b.top_p && a.paths == b.paths;
  };
  static const char* trace = getenv("Q27_TRACE");  // debug timeline (build with -DQ27_TRACE)
  if (trace) trace_step_begin(s_);
  // Pipelining (Q27_SPEC_PIPELINE, default on): the next step's graph is launched before this step's
  // result is read, so the GPU does not idle between steps. Every input of a step is on the device,
  // so it can start without the host; spec_flush drops it when the caller does not continue.
  int slot = par_host_;  // SpecOut slot of the step returned by this call
  if (spec_inflight_) {
    Q27_CHECK(same(smp, spec_key_), "spec_step: sampling changed with a step in flight");
    slot = inflight_slot_;
  } else if (!graphs_ || !spec_warm_) {
    spec_body(smp);
    spec_warm_ = true;
    par_host_ ^= 1;
  } else {
    if (!spec_graph_ || !same(smp, spec_key_)) {
      spec_graph_ = nullptr;
      for (auto& g : sgraphs_)
        if (same(g.smp, smp)) spec_graph_ = g.exec, g.used = ++graph_clock_, spec_key_ = smp;
    }
    if (!spec_graph_) {
      cudaGraph_t graph;
      trace_clear_launches();
      Q27_CUDA(cudaStreamBeginCapture(s_, cudaStreamCaptureModeThreadLocal));
      prof_.capturing = true;
      prof_.graph.used = 0;
      spec_body(smp);
      mark(nullptr);
      prof_.capturing = false;
      Q27_CUDA(cudaStreamEndCapture(s_, &graph));
      Q27_CUDA(cudaGraphInstantiate(&spec_graph_, graph, 0));
      Q27_CUDA(cudaGraphDestroy(graph));
      spec_key_ = smp;
      if (sgraphs_.size() >= size_t(kGraphCache)) {
        auto lru = std::min_element(sgraphs_.begin(), sgraphs_.end(), [](auto& a, auto& b) { return a.used < b.used; });
        Q27_CUDA(cudaGraphExecDestroy(lru->exec));
        sgraphs_.erase(lru);
      }
      sgraphs_.push_back({{}, smp, 0, spec_graph_, ++graph_clock_});
    }
    Q27_CUDA(cudaGraphLaunch(spec_graph_, s_));
    par_host_ ^= 1;
  }
  static const bool dbg_accept = getenv("Q27_DBG_ACCEPT") != nullptr;
  const bool ahead = allow_ahead && spec_pipeline_ && graphs_ && spec_graph_ && spec_warm_ && same(smp, spec_key_) && !trace &&
                     !prof_.on && !dbg_accept &&
                     pos_ + kSpecBlock + spec_rows(smp) <= max_pages_ * kPageTokens;  // (worst-case position)
  if (ahead) {
    Q27_CUDA(cudaEventRecord(spec_ev_[slot], s_));
    Q27_CUDA(cudaGraphLaunch(spec_graph_, s_));
    inflight_slot_ = par_host_;
    par_host_ ^= 1;
    Q27_CUDA(cudaEventSynchronize(spec_ev_[slot]));
  } else {
    Q27_CUDA(cudaStreamSynchronize(s_));
  }
  spec_inflight_ = ahead;
  profile_collect(spec_graph_ != nullptr && graphs_);
  static int trace_steps = 0;  // one step after warm-up; Q27_TRACE_STEP picks another
  static const int trace_at = getenv("Q27_TRACE_STEP") ? atoi(getenv("Q27_TRACE_STEP")) : 20;
  if (trace && ++trace_steps == trace_at) trace_step_end(trace);
  spec_pending_ = true;
  const SpecOut& so = spec_out_[slot];
  const int n = so.n;
  Q27_CHECK(n >= 1 && n <= kSpecBlock, "spec_step: bad token count");
  // Debug (Q27_DBG_ACCEPT=1, greedy): the step's tree, the target's argmax per row and the accepted path.
  if (dbg_accept && spec_smp_.temperature <= 0.f) {
    SpecTree tr;
    int32_t ti[kTreeMax], cp[kSpecBlock];
    Q27_CUDA(cudaMemcpy(&tr, tree_, sizeof tr, cudaMemcpyDeviceToHost));
    Q27_CUDA(cudaMemcpy(ti, ti_, sizeof ti, cudaMemcpyDeviceToHost));
    Q27_CUDA(cudaMemcpy(cp, cpath_, sizeof cp, cudaMemcpyDeviceToHost));
    fprintf(stderr, "[dbg] pos %d n %d nodes %d paths %d | path", pos_, n, tr.n, tr.npaths);
    for (int i = 0; i < n; i++) fprintf(stderr, " %d", cp[i]);
    fprintf(stderr, "\n");
    for (int i = 0; i < tr.n; i++) {
      fprintf(stderr, "[dbg]   node %2d tok %6d depth %d parent %2d owner %d argmax %6d draws", i, tr.tok[i], tr.depth[i],
              tr.depth[i] ? tr.path[i][tr.depth[i] - 1] : -1, tr.owner[i], ti[i]);
      for (int j = 0; j < tr.ndraws[i]; j++) fprintf(stderr, " %d", tr.draws[i][j]);
      fprintf(stderr, "\n");
    }
  }
  stat_steps_++, stat_lookup_ += so.src, stat_tokens_ += n;
  set_sbuf(sbuf_host_ ^ 1);  // (the device flipped it in spec_advance)
  for (int i = 0; i < n; i++) out[i] = so.tok[i];
  pos_ += n;
  // Fed by this step: the previous last token and the accepted drafts (the new last one is not).
  seq_ids_.push_back(spec_last_);
  seq_ids_.insert(seq_ids_.end(), out, out + n - 1);
  spec_last_ = out[n - 1];
  return n;
}

void Model::spec_stats(int64_t* out) {
  out[0] = stat_steps_, out[1] = stat_lookup_, out[2] = stat_tokens_;
  stat_steps_ = stat_lookup_ = stat_tokens_ = 0;
}

void Model::debug_kv(int layer, int pos, int n, void* out) {
  spec_flush();
  Q27_CUDA(cudaStreamSynchronize(s_));
  const size_t row = size_t(kKvHeads) * kv_row_bytes(opt_.kv_type);
  std::vector<int32_t> pt(max_pages_);
  Q27_CUDA(cudaMemcpy(pt.data(), page_table_, pt.size() * 4, cudaMemcpyDeviceToHost));
  char* o = static_cast<char*>(out);
  for (int kv = 0; kv < 2; kv++)
    for (int i = 0; i < n; i++) {
      const int p = pos + i;
      const char* base = static_cast<const char*>(kv ? kv_layers_[layer].v : kv_layers_[layer].k);
      Q27_CUDA(cudaMemcpy(o, base + (size_t(pt[p / kPageTokens]) * kPageTokens + p % kPageTokens) * row, row,
                          cudaMemcpyDeviceToHost));
      o += row;
    }
}

void Model::debug_ssm(int gi, float* out) {
  spec_flush();
  Q27_CUDA(cudaStreamSynchronize(s_));
  Q27_CUDA(cudaMemcpy(out, ssm_ + size_t(gi) * kVHeads * kKDim * kVDim, size_t(kVHeads) * kKDim * kVDim * 4,
                      cudaMemcpyDeviceToHost));
}

// ---------------------------------------------------------------- several sequences (Phase 6)

std::vector<Model::Region> Model::seq_regions(char* area) const {
  std::vector<Region> r{{ssm_, area, ssm_bytes_}, {conv_, area + ssm_bytes_, conv_bytes_}};
  if (draft_) {
    const size_t rb = draft_->ring_layers_bytes();
    r.push_back({draft_->ring_k(), area + ssm_bytes_ + conv_bytes_, rb});
    r.push_back({draft_->ring_v(), area + ssm_bytes_ + conv_bytes_ + rb, rb});
  }
  return r;
}

// The device scalars of a sequence: the rotary delta of its images, and with a draft the token budget,
// the RNG (seed, step) and the anchor rows (block / verify ids). The step parity, the GDN state buffer
// index and the commit count are the engine's (nothing is pending after spec_flush).
void Model::seq_scalars(int32_t* host, bool save) {
  const std::pair<void*, size_t> vars[] = {{rope_delta_, 4},           {budget_, 4},
                                           {rng_, 16},                  {block_ids_, kSpecBlock * 4},
                                           {ids_ver_, kTreeMax * 4}};
  char* h = reinterpret_cast<char*>(host);
  for (auto [dev, bytes] : vars) {
    if (dev)
      Q27_CUDA(cudaMemcpyAsync(save ? static_cast<void*>(h) : dev, save ? dev : static_cast<void*>(h), bytes,
                               save ? cudaMemcpyDeviceToHost : cudaMemcpyHostToDevice, s_));
    h += bytes;
  }
  static_assert(sizeof(SeqState::scalars) == 4 + 4 + 16 + kSpecBlock * 4 + kTreeMax * 4, "scalar layout");
  Q27_CUDA(cudaStreamSynchronize(s_));
}

void Model::seq_take(SeqState& st) {
  if (!seq_pages_.empty()) Q27_CUDA(cudaMemsetAsync(page_table_, 0, seq_pages_.size() * 4, s_));  // unmapped
  st.pos = pos_;
  st.pages = std::move(seq_pages_);
  st.ids = std::move(seq_ids_);
  st.plan = std::move(plan_);
  st.mm = std::move(mm_spans_);
  st.spec_last = spec_last_;
  st.ring_valid = ring_valid_;
  st.smp = spec_smp_;
  seq_pages_.clear(), seq_ids_.clear(), plan_.clear(), mm_spans_.clear();
  pos_ = 0;
  spec_last_ = -1;
  ring_valid_ = false;
  mpos_on_ = false;
}

void Model::seq_give(SeqState& st) {
  pos_ = st.pos;
  seq_pages_ = std::move(st.pages);
  seq_ids_ = std::move(st.ids);
  plan_ = std::move(st.plan);
  mm_spans_ = std::move(st.mm);
  spec_last_ = st.spec_last;
  ring_valid_ = st.ring_valid;
  spec_smp_ = st.smp;
  st.pages.clear(), st.ids.clear(), st.plan.clear(), st.mm.clear();
  if (!seq_pages_.empty())
    upload_pages(page_table_, seq_pages_.data(), seq_pages_.size());
  Q27_CUDA(cudaMemcpyAsync(pos_dev_, &pos_, 4, cudaMemcpyHostToDevice, s_));
  if (hist_ && pos_ > 0) copy_hist(hist_, seq_ids_.data(), pos_);  // (lookup drafting reads it by position)
  Q27_CUDA(cudaStreamSynchronize(s_));
}

int Model::area_first(int a) const { return pool_.pages() - (a + 1) * park_pages_; }

void Model::reserve_area(int a) {
  const int lo = area_first(a), hi = lo + park_pages_;
  auto held = [&] {  // pages of the area the KV pool holds
    std::vector<int32_t> v;
    for (int p = lo; p < hi; p++)
      if (pool_.refs(p) > 0) v.push_back(p);
    return v;
  };
  std::vector<int32_t> moved = held();
  while (pool_.free_outside(lo, hi) < int(moved.size()) && evict_checkpoint()) moved = held();
  if (pool_.free_outside(lo, hi) < int(moved.size())) throw KvFull("KV pool exhausted (parking area)");
  if (!moved.empty()) {
    batch_exit();  // (the lanes' page tables are uploaded when a batch starts)
    std::vector<std::pair<int32_t, int32_t>> to;
    for (int32_t p : moved) {
      const int q = pool_.alloc_outside(lo, hi);
      pool_.move(p, q);
      to.emplace_back(p, q);
      // The page's slices are kv_block_ pages apart: one 2D copy of kKvSlices rows.
      const size_t pitch = size_t(kv_block_) * kv_page_bytes_;
      Q27_CUDA(cudaMemcpy2DAsync(kv_page(q, 0), pitch, kv_page(p, 0), pitch, kv_page_bytes_, kKvSlices,
                                 cudaMemcpyDeviceToDevice, s_));
    }
    std::sort(to.begin(), to.end());
    auto remap = [&](std::vector<int32_t>& pages) {
      for (int32_t& p : pages)
        if (p >= lo && p < hi) p = std::lower_bound(to.begin(), to.end(), std::make_pair(p, INT32_MIN))->second;
    };
    remap(seq_pages_);
    for (SeqState& st : parked_)
      if (st.used) remap(st.pages);
    for (Checkpoint& c : ckpts_) remap(c.pages);
    if (!seq_pages_.empty()) upload_pages(page_table_, seq_pages_.data(), seq_pages_.size());
    log("parking area %d: moved %d KV pages out", a, int(moved.size()));
  }
  for (int p = lo; p < hi; p++) Q27_CHECK(pool_.take(p), "reserve_area: a page of the area is in use");
}

void Model::release_area(int a) {
  for (int p = area_first(a); p < area_first(a) + park_pages_; p++) pool_.unref(p);
}

int Model::seq_park() {
  // The free area with the fewest pages to move.
  int a = -1, best = 0;
  for (int i = 0; i < int(parked_.size()); i++) {
    if (parked_[i].used) continue;
    int n = 0;
    for (int p = area_first(i); p < area_first(i) + park_pages_; p++) n += pool_.refs(p) > 0;
    if (a < 0 || n < best) a = i, best = n;
  }
  Q27_CHECK(a >= 0, "seq_park: no free parking area");
  spec_flush();
  reserve_area(a);  // (KvFull: nothing changed but evicted checkpoints)
  SeqState& st = parked_[a];
  for (const Region& r : seq_regions(st.dev)) Q27_CUDA(cudaMemcpyAsync(r.area, r.dev, r.bytes, cudaMemcpyDeviceToDevice, s_));
  seq_scalars(st.scalars, true);
  seq_take(st);
  st.used = true;
  reset();  // (no sequence runs now: the state buffers start over, as model.h promises; P7-13)
  return a;
}

void Model::seq_unpark(int a) {
  Q27_CHECK(a >= 0 && a < int(parked_.size()) && parked_[a].used, "seq_unpark: no sequence in that area");
  reset();  // (drops the running sequence, if any)
  SeqState& st = parked_[a];
  for (const Region& r : seq_regions(st.dev)) Q27_CUDA(cudaMemcpyAsync(r.dev, r.area, r.bytes, cudaMemcpyDeviceToDevice, s_));
  seq_scalars(st.scalars, false);
  seq_give(st);
  st.used = false;
  release_area(a);  // (the copies above are on s_ before any KV write to these pages)
}

void Model::seq_swap(int a) {
  Q27_CHECK(a >= 0 && a < int(parked_.size()) && parked_[a].used, "seq_swap: no sequence in that area");
  spec_flush();
  SeqState& st = parked_[a];
  for (const Region& r : seq_regions(st.dev)) swap_bytes(r.dev, r.area, r.bytes, s_);
  int32_t cur[kSeqScalars];
  seq_scalars(cur, true);
  seq_scalars(st.scalars, false);
  SeqState run;
  seq_take(run);
  seq_give(st);
  run.dev = st.dev;
  std::copy(cur, cur + kSeqScalars, run.scalars);
  run.used = true;
  st = std::move(run);
}

void Model::seq_drop(int a) {
  Q27_CHECK(a >= 0 && a < int(parked_.size()) && parked_[a].used, "seq_drop: no sequence in that area");
  batch_exit();
  SeqState& st = parked_[a];
  for (int32_t p : st.pages) pool_.unref(p);
  st.pages.clear(), st.ids.clear(), st.plan.clear(), st.mm.clear();
  st.used = false;
  release_area(a);
}

void Model::seq_suspend(int a) {
  Q27_CHECK(a >= 0 && a < int(parked_.size()) && parked_[a].used, "seq_suspend: no sequence in that area");
  batch_exit();
  SeqState& st = parked_[a];
  if (st.pos > 0 && int(st.ids.size()) == st.pos) save_checkpoint_from(st.dev, st.ids, st.pages, st.ring_valid);
  seq_drop(a);
}


// ---------------------------------------------------------------- batched steps (P6-10)
//
// The lanes are sequences in decode: the running one (-1) and parked ones. Their big state stays where it
// is (the running sequence's buffers, the parking areas); the kernels take it through per-lane pointers
// (captured in the batch graph, so the graphs are keyed by the lanes). Their step scalars (position,
// parity, commit count, budget, RNG, anchor / draft ids) move into the b_* arrays when the batch starts
// and back when it ends. A verify's commit is replayed at the start of the lane's next verify, in place
// (the lane's state has one buffer here); ending the batch replays the last one.

Model::LaneRef Model::lane_ref(int z) {
  const int a = blanes_[z];
  if (a < 0) return {&pos_, &seq_pages_, &seq_ids_, &spec_last_};
  SeqState& st = parked_[a];
  return {&st.pos, &st.pages, &st.ids, &st.spec_last};
}

GdnSpec Model::gdn_spec_lanes(int gi) const {
  GdnSpec sp = gdn_spec(gi);  // (the replay buffers; lane z uses rows 8 z ..)
  sp.par = b_par_, sp.ncommit = b_nc_, sp.cpath = b_cpath_, sp.tree = b_tree_, sp.paths = 1;
  sp.lanes = int(blanes_.size());
  for (int z = 0; z < sp.lanes; z++) {
    sp.lconv[z] = bconv_[z] + size_t(gi) * kConvDim * (kConvK - 1);
    sp.lst[z] = bssm_[z] + size_t(gi) * kVHeads * kKDim * kVDim;
  }
  return sp;
}

// Offsets of the step scalars in SeqState::scalars (seq_scalars): rope delta, budget, rng, block ids, verify ids.
namespace {
constexpr int kScDelta = 0, kScBudget = 1, kScRng = 2, kScBlock = 6, kScIds = 6 + kSpecBlock;
}

void Model::batch_enter(const int32_t* lanes, int n) {
  if (batch_on_ && int(blanes_.size()) == n && std::equal(blanes_.begin(), blanes_.end(), lanes)) return;
  batch_exit();
  spec_flush();  // the running sequence: nothing pending, no step in flight
  Q27_CHECK(n >= 1 && n <= max_lanes(), "spec_step_lanes: more lanes than MAX_SEQS");
  Q27_CHECK(spec_paths_ == 1, "spec_step_lanes: chains only (Q27_SPEC_PATHS=1)");
  for (int z = 0; z < n; z++) {
    const int a = lanes[z];
    Q27_CHECK(a == -1 || (a >= 0 && a < int(parked_.size()) && parked_[a].used), "spec_step_lanes: no such sequence");
    for (int y = 0; y < z; y++) Q27_CHECK(lanes[y] != a, "spec_step_lanes: a sequence twice");
    Q27_CHECK(a >= 0 ? parked_[a].spec_last >= 0 : spec_last_ >= 0, "spec_step_lanes: the sequence is not decoding");
  }
  blanes_.assign(lanes, lanes + n);
  bsl_ = SeqLanes{};
  bsl_.n = n, bsl_.pos = b_pos_, bsl_.rope_delta = b_delta_;
  const size_t rb = draft_->ring_layers_bytes();
  for (int z = 0; z < n; z++) {
    const int a = lanes[z];
    int32_t* pt = b_pt_ + size_t(z) * max_pages_;
    bsl_.page_table[z] = pt;
    SpecSampling smp;
    const std::vector<int32_t>* pages;
    if (a < 0) {  // the running sequence: its device scalars
      const std::pair<void*, const void*> cp[] = {{b_pos_ + z, pos_dev_},     {b_delta_ + z, rope_delta_},
                                                   {b_budget_ + z, budget_},  {b_rng_ + 2 * z, rng_},
                                                   {b_blk_ + kSpecBlock * z, block_ids_}, {b_ids_ + kSpecBlock * z, ids_ver_}};
      const size_t bytes[] = {4, 4, 4, 16, kSpecBlock * 4, kSpecBlock * 4};
      for (int i = 0; i < 6; i++)
        Q27_CUDA(cudaMemcpyAsync(cp[i].first, cp[i].second, bytes[i], cudaMemcpyDeviceToDevice, s_));
      bconv_[z] = conv_, bssm_[z] = ssm_buf_[sbuf_host_];
      bring_k_[z] = static_cast<bf16*>(draft_->ring_k()), bring_v_[z] = static_cast<bf16*>(draft_->ring_v());
      smp = spec_smp_;
      pages = &seq_pages_;
    } else {  // a parked one: its saved scalars (seq_scalars), its state in the area
      SeqState& st = parked_[a];
      Q27_CUDA(cudaMemcpyAsync(b_pos_ + z, &st.pos, 4, cudaMemcpyHostToDevice, s_));
      Q27_CUDA(cudaMemcpyAsync(b_delta_ + z, st.scalars + kScDelta, 4, cudaMemcpyHostToDevice, s_));
      Q27_CUDA(cudaMemcpyAsync(b_budget_ + z, st.scalars + kScBudget, 4, cudaMemcpyHostToDevice, s_));
      Q27_CUDA(cudaMemcpyAsync(b_rng_ + 2 * z, st.scalars + kScRng, 16, cudaMemcpyHostToDevice, s_));
      Q27_CUDA(cudaMemcpyAsync(b_blk_ + kSpecBlock * z, st.scalars + kScBlock, kSpecBlock * 4, cudaMemcpyHostToDevice, s_));
      Q27_CUDA(cudaMemcpyAsync(b_ids_ + kSpecBlock * z, st.scalars + kScIds, kSpecBlock * 4, cudaMemcpyHostToDevice, s_));
      bssm_[z] = reinterpret_cast<float*>(st.dev);
      bconv_[z] = reinterpret_cast<float*>(st.dev + ssm_bytes_);
      bring_k_[z] = reinterpret_cast<bf16*>(st.dev + ssm_bytes_ + conv_bytes_);
      bring_v_[z] = reinterpret_cast<bf16*>(st.dev + ssm_bytes_ + conv_bytes_ + rb);
      smp = st.smp;
      pages = &st.pages;
    }
    if (!pages->empty()) upload_pages(pt, pages->data(), pages->size());
    smp.paths = 1;
    std::copy(stop_, stop_ + 2, smp.stop);
    bacc_.smp[z] = smp;
    bacc_.k[z] = smp.temperature <= 0.f ? 1 : std::clamp(smp.top_k <= 0 ? 64 : smp.top_k, 1, 64);
  }
  for (int z = n; z < kMaxLanes; z++) bacc_.smp[z] = SpecSampling{}, bacc_.k[z] = 0;
  Q27_CUDA(cudaMemsetAsync(b_par_, 0, kMaxLanes * 4, s_));
  Q27_CUDA(cudaMemsetAsync(b_nc_, 0, kMaxLanes * 4, s_));
  Q27_CUDA(cudaStreamSynchronize(s_));
  bpar_ = 0;
  batch_pending_ = false;
  batch_on_ = true;
}

void Model::batch_exit() {
  if (!batch_on_) return;
  batch_on_ = false;
  const int n = int(blanes_.size());
  if (batch_pending_) {  // the last verify's commits: replay-only blocks (conv and recurrent states)
    for (int i = 0; i < kLayers; i++) {
      if (is_attn(i)) continue;
      const auto& G = w_.layers[i].gdn;
      const GdnSpec sp = gdn_spec_lanes(i - i / kAttnEvery);
      gdn_verify(q_, 0, G.A_log, G.dt_bias, sp.lst[0], big2_, kVHeads * kVDim, G.norm, kEps, n_, nullptr, 1.f, sp, s_);
    }
    Q27_CUDA(cudaMemsetAsync(b_nc_, 0, kMaxLanes * 4, s_));
    batch_pending_ = false;
  }
  for (int z = 0; z < n; z++) {
    const int a = blanes_[z];
    if (a < 0) {
      const std::pair<void*, const void*> cp[] = {{pos_dev_, b_pos_ + z},      {budget_, b_budget_ + z},
                                                   {rng_, b_rng_ + 2 * z},      {block_ids_, b_blk_ + kSpecBlock * z},
                                                   {ids_ver_, b_ids_ + kSpecBlock * z}};
      const size_t bytes[] = {4, 4, 16, kSpecBlock * 4, kSpecBlock * 4};
      for (int i = 0; i < 5; i++)
        Q27_CUDA(cudaMemcpyAsync(cp[i].first, cp[i].second, bytes[i], cudaMemcpyDeviceToDevice, s_));
      if (!seq_pages_.empty())
        upload_pages(page_table_, seq_pages_.data(), seq_pages_.size());
      if (hist_ && lookup_.on && pos_ > 0) copy_hist(hist_, seq_ids_.data(), pos_);  // (not kept by batched steps)
    } else {
      SeqState& st = parked_[a];
      Q27_CUDA(cudaMemcpyAsync(st.scalars + kScBudget, b_budget_ + z, 4, cudaMemcpyDeviceToHost, s_));
      Q27_CUDA(cudaMemcpyAsync(st.scalars + kScRng, b_rng_ + 2 * z, 16, cudaMemcpyDeviceToHost, s_));
      Q27_CUDA(cudaMemcpyAsync(st.scalars + kScBlock, b_blk_ + kSpecBlock * z, kSpecBlock * 4, cudaMemcpyDeviceToHost, s_));
      Q27_CUDA(cudaMemcpyAsync(st.scalars + kScIds, b_ids_ + kSpecBlock * z, kSpecBlock * 4, cudaMemcpyDeviceToHost, s_));
    }
  }
  Q27_CUDA(cudaStreamSynchronize(s_));
  blanes_.clear();
}

void Model::batch_body(int n) {
  const auto& lm = w_.lm_head;
  DraftLanes dl;
  dl.n = n, dl.pos = b_pos_;
  SelectLanes sel;
  for (int z = 0; z < n; z++) dl.ring_k[z] = bring_k_[z], dl.ring_v[z] = bring_v_[z], sel.smp[z] = bacc_.smp[z];
  mark("draft");
  draft_->propose_lanes(dl, b_blk_, w_.embed, lm.w, lm.sf, lm.in_scale, lm.alpha.alpha[0], sel, b_rng_, b_ids_, b_tree_,
                        aq_, asf_, logits_, skinny_ws_, kSkinnyWsBytes, s_);
  const int W = kLaneRows * n;
  verify_ = true;
  verify_paths_ = 1;
  cap_ = cap_buf_;
  batch_n_ = n;
  chunk(b_ids_, W, false);
  batch_n_ = 0;
  verify_ = false;
  cap_ = nullptr;
  logits(x_, W);
  mark("accept");
  const int K = *std::max_element(bacc_.k, bacc_.k + n);
  topk_rows(logits_, W, kVocab, K, tv_, ti_, s_);
  spec_accept_lanes(n, b_ids_, b_blk_, tv_, ti_, K, bacc_, draft_->lane_cand(), b_tree_, draft_->lane_src(), b_rng_,
                    b_nc_, b_cpath_, b_out_dev_, b_pos_, b_par_, b_budget_, s_);
  mark("draft context");
  draft_->append_ctx_lanes(dl, cap_buf_, b_nc_, b_cpath_, skinny_ws_, kSkinnyWsBytes, s_);
  spec_advance_lanes(n, b_pos_, b_nc_, b_par_, b_rng_, b_budget_, s_);
}

void Model::spec_step_lanes(const int32_t* lanes, int n, int32_t* out, int32_t* counts) {
  Q27_CHECK(draft_ && max_lanes() > 1, "spec_step_lanes: batched steps need a draft model and MAX_SEQS > 1");
  batch_enter(lanes, n);
  // Pages for the step's rows (the KV of the verify block at pos .. pos + 7).
  for (int z = 0; z < n; z++) {
    LaneRef r = lane_ref(z);
    Q27_CHECK(*r.pos + kSpecBlock <= max_pages_ * kPageTokens, "sequence longer than max_tokens");
    ensure_pages_of(*r.pages, b_pt_ + size_t(z) * max_pages_, *r.pos + kSpecBlock);
  }
  static const char* trace = getenv("Q27_TRACE");  // debug timeline (build with -DQ27_TRACE)
  if (trace) trace_step_begin(s_);
  cudaGraphExec_t exec = nullptr;
  if (!graphs_ || !bwarm_) {
    batch_body(n);
    bwarm_ = true;
  } else {
    // The graph holds the lanes' state pointers: keyed by the lanes (and the running one's state buffer).
    const int sb = std::count(blanes_.begin(), blanes_.end(), -1) ? sbuf_host_ : -1;
    auto same_acc = [&](const AcceptLanes& a) {
      for (int z = 0; z < n; z++) {
        const SpecSampling &x = a.smp[z], &y = bacc_.smp[z];
        if (x.temperature != y.temperature || x.top_k != y.top_k || x.top_p != y.top_p) return false;
      }
      return true;
    };
    for (auto& g : bgraphs_)
      if (g.lanes == blanes_ && g.sbuf == sb && same_acc(g.acc)) exec = g.exec, g.used = ++graph_clock_;
    if (!exec) {
      cudaGraph_t graph;
      trace_clear_launches();
      Q27_CUDA(cudaStreamBeginCapture(s_, cudaStreamCaptureModeThreadLocal));
      prof_.capturing = true;
      prof_.graph.used = 0;
      batch_body(n);
      mark(nullptr);
      prof_.capturing = false;
      Q27_CUDA(cudaStreamEndCapture(s_, &graph));
      Q27_CUDA(cudaGraphInstantiate(&exec, graph, 0));
      Q27_CUDA(cudaGraphDestroy(graph));
      if (bgraphs_.size() >= size_t(kGraphCache)) {
        auto lru = std::min_element(bgraphs_.begin(), bgraphs_.end(), [](auto& a, auto& b) { return a.used < b.used; });
        Q27_CUDA(cudaGraphExecDestroy(lru->exec));
        bgraphs_.erase(lru);
      }
      bgraphs_.push_back({blanes_, sb, bacc_, exec, ++graph_clock_});
    }
    Q27_CUDA(cudaGraphLaunch(exec, s_));
  }
  Q27_CUDA(cudaStreamSynchronize(s_));
  profile_collect(exec != nullptr);
  static int trace_steps = 0;
  static const int trace_at = getenv("Q27_TRACE_STEP") ? atoi(getenv("Q27_TRACE_STEP")) : 20;
  if (trace && ++trace_steps == trace_at) trace_step_end(trace);
  batch_pending_ = true;
  for (int z = 0; z < n; z++) {
    const SpecOut& so = b_out_[2 * z + bpar_];
    const int k = so.n;
    Q27_CHECK(k >= 1 && k <= kSpecBlock, "spec_step_lanes: bad token count");
    LaneRef r = lane_ref(z);
    int32_t* o = out + kSpecBlock * z;
    for (int i = 0; i < k; i++) o[i] = so.tok[i];
    counts[z] = k;
    *r.pos += k;
    // Fed by this step: the previous last token and the accepted drafts (the new last one is not).
    r.ids->push_back(*r.spec_last);
    r.ids->insert(r.ids->end(), o, o + k - 1);
    *r.spec_last = o[k - 1];
    stat_steps_++, stat_tokens_ += k;
  }
  bpar_ ^= 1;
}

}  // namespace q27
