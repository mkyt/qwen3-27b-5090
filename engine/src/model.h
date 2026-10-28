// Model runner, Phase 1: one sequence, chunked prefill and decode through the same path.
#pragma once

#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "common.h"
#include "draft.h"
#include "kernels.h"
#include "memory.h"
#include "prefix_cache.h"
#include "safetensors.h"
#include "vision.h"
#include "weights.h"

namespace q27 {

struct Options {
  std::string model_dir;
  int max_tokens = 32768;     // context capacity of the sequence (KV pages)
  int prefill_chunk = 2048;   // tokens per forward chunk
  int logits_rows = 256;      // rows per lm_head slice when all-position logits are requested
  KvType kv_type = KvType::BF16;
  std::string draft_dir;  // DFlash2 checkpoint (speculative decoding); empty: none
  // Prefix cache (off when state_cache_gib is 0): pinned host memory for state checkpoints, the KV
  // pool (0: what the VRAM budget leaves; without the cache, max_tokens) and the interval of the
  // checkpoints inside long prompts.
  double state_cache_gib = 0;
  int kv_pool_tokens = 0;
  int checkpoint_interval = 8192;
  // Vision tower (P4-12): 0 off, 1 BF16 weights on the device, 2 BF16 weights in pinned host memory,
  // 3 FP8 weights on the device (VisionMode). Images of up to image_max_pixels; the image embeddings of a
  // request (up to image_tokens rows) live in pinned host memory.
  int vision = 0;
  int image_max_pixels = 4194304;
  int image_tokens = 16384;
  // Several sequences (Phase 6): device areas a running sequence's state can be parked in while
  // another one runs (MAX_SEQS - 1 in the server; 0: one sequence at a time).
  int park_slots = 0;
  // Lower max_tokens to what the VRAM budget leaves for the KV pool (rounded down to 1,024) instead of
  // failing (MAX_TOKENS_FIT; the server sets it). Not with kv_pool_tokens.
  bool fit_max_tokens = false;
  // Plan only (q27_plan): size the pools for a device with plan_free of plan_total bytes free and stop there, with
  // no CUDA call (nothing allocated or loaded; memory_report, max_tokens, cache_stats and device_memory answer).
  bool plan_only = false;
  size_t plan_free = 0, plan_total = 0;
};

class Model {
 public:
  explicit Model(const Options& opt);
  ~Model();

  void reset();
  void configure();  // re-reads the Q27_* switches from the environment (see model.cc)
  int pos() const { return pos_; }
  int max_tokens() const { return opt_.max_tokens; }  // (after fit_max_tokens)

  // Runs ids[0..n) after the tokens already in the sequence. For each position p the callback gets
  // the top-k (ids, log-probs) of the next token. k <= 64.
  void forward_topk(const int32_t* ids, int n, int k,
                    const std::function<void(int pos, const int32_t* ids, const float* logp)>& cb);
  // Runs ids[0..n) and samples the token after the last one.
  int32_t forward_sample(const int32_t* ids, int n, float temperature, int top_k, float top_p, float uniform);

  // Per-op GPU time (Q27_PROFILE=1): every op records an event when it starts; prints the totals
  // per label since the last dump to stderr and resets them.
  void profile_dump();

  // Speculative decoding (DFlash2; needs Options::draft_dir). spec_begin runs the prompt like
  // forward_sample (also filling the draft's context) and returns the first token; each spec_step
  // drafts 7 tokens after the last one, verifies them and writes the new tokens (accepted drafts and
  // the bonus token, 1..8) to out. Other forward calls may follow (pending GDN commits are flushed).
  // max_new > 0: the steps output at most max_new tokens in all (with the first one), so the sequence
  // never holds tokens past the caller's limit (the prefix cache checkpoints it at the end).
  int32_t spec_begin(const int32_t* ids, int n, const SpecSampling& smp, uint64_t seed, float uniform,
                     int max_new = 0);
  // ahead (default): launch the next step before returning this one (Q27_SPEC_PIPELINE); false when
  // the caller will not continue right away (another sequence runs next), so no step is wasted.
  int spec_step(int32_t* out, bool ahead = true);
  bool has_draft() const { return draft_ != nullptr; }
  // Counters since the last call: verify steps, steps that used the lookup chain, tokens produced.
  void spec_stats(int64_t* out);
  // Debug: KV rows (K then V, raw bytes) of attention layer `layer` at positions pos..pos+n-1, and the
  // committed recurrent state of GDN layer `gdn_layer` (fp32 [48][128][128]); pending commits flushed.
  void debug_kv(int layer, int pos, int n, void* out);
  void debug_ssm(int gdn_layer, float* out);

  // Prefix cache (Options::state_cache_gib > 0; otherwise this is reset() and returns 0).
  // prefix_begin resets the sequence and restores the deepest checkpoint whose tokens are a prefix of
  // ids[0..n-1) (one token at least is left to run) and returns its length `cached`. The caller then
  // runs ids[cached..n) (forward_sample or spec_begin); that prefill saves a checkpoint at each of
  // at[0..nat) in (cached, n) and at the last few multiples of checkpoint_interval. spec: the caller
  // continues with spec_begin, so only checkpoints with the draft's context qualify.
  int prefix_begin(const int32_t* ids, int n, const int32_t* at, int nat, bool spec);
  // Saves a checkpoint of the sequence so far (e.g. at the end of a generation).
  void checkpoint();
  // out[12]: lookups, hits, prompt tokens, cached tokens, checkpoints saved, evicted, live checkpoints, free
  // pages, pages only checkpoints hold (free after evicting them), pool pages, KV pages of a parking area, state
  // slots allocated.
  void cache_stats(int64_t* out) const;
  // Allocates up to n more pinned state slots of the prefix cache (they are not allocated at startup); returns how
  // many it allocated (0: all are). A checkpoint that finds no free slot allocates one itself.
  int grow_state_cache(int n);
  // Memory (stats UI): one line per pool ("pool\t<name>\t<host 0|1>\t<bytes>") followed by its reservations
  // ("item\t<name>\t<what>\t<bytes>"), and "slots\t<state cache slots>\t<bytes of a checkpoint>". Fixed after
  // the constructor. device_memory: out[4] = free now, total (cudaMemGetInfo), free at the start, budget.
  std::string memory_report() const;
  void device_memory(int64_t* out) const;

  // Prompt chunks before forward_sample / spec_begin (Phase 6: other sequences can run between them).
  // ids[0..n) is the rest of the prompt after the sequence's position; runs chunks while more than one
  // prefill chunk is left, stopping after at least `budget` tokens, and returns the tokens run. The rest
  // then goes to forward_sample / spec_begin with the same chunk cuts as one call over all of it.
  // spec: the caller continues with spec_begin (the draft's context is filled the same way).
  int prefill(const int32_t* ids, int n, int budget, bool spec);

  // Several sequences (Phase 6). The running sequence (its GDN / conv state, the draft's context ring,
  // device scalars, KV pages, position and tokens) can be moved to one of Options::park_slots device
  // areas and resumed later, bit for bit. seq_park moves it to a free area (returned; the engine then
  // has no sequence, as after reset). seq_unpark(a) drops the running sequence (reset) and resumes the
  // one in area a. seq_swap(a) exchanges the running sequence with the one in area a. seq_drop(a) gives
  // a parked sequence's pages back; seq_suspend(a) first saves it as a prefix cache checkpoint (a
  // preempted sequence resumes from the cache). A parked sequence keeps its pages (KV is never copied).
  // P7-13: an area's pages hold KV while it is free; seq_park first moves those pages elsewhere (KvFull,
  // with nothing changed but evicted checkpoints, when the pool cannot take them).
  int seq_park();
  void seq_unpark(int a);
  void seq_swap(int a);
  void seq_drop(int a);
  void seq_suspend(int a);
  int park_slots() const { return int(parked_.size()); }

  // Batched speculative steps (P6-10). One step of each of n (2..4) sequences in decode (after spec_begin):
  // lanes[z] is -1 for the running sequence or a parking area. Their draft blocks and verify rows run as one
  // batch (the GEMMs read the weights once for all), each lane exactly as its own spec_step: the tokens are
  // the same, bit for bit. Lane z's new tokens go to out[8 z ..], their count to counts[z]. While the same
  // lanes keep stepping their state stays in the batch; any other call first ends it (the lanes then hold
  // their sequences as before, nothing pending). Chains only (Q27_SPEC_PATHS 1); no lookup drafting.
  void spec_step_lanes(const int32_t* lanes, int n, int32_t* out, int32_t* counts);
  int max_lanes() const { return draft_ ? draft_->lanes() : 0; }

  // Images (P4-12). The prompt holds each image as a run of gh * gw ids >= kVocab (any values: the
  // frontend derives them from the image, so the prefix cache tells images apart) at positions
  // start .. start + gh * gw - 1, gh x gw the merged grid. mm_set (after reset / prefix_begin, before the
  // forward call) gives the request's images: spans [n][3] = (start, gh, gw), ascending. mm_encode runs
  // the vision tower on image i (pixels [4 * gh * gw][1536] fp32, host) into its embedding rows; images
  // whose positions the restored prefix covers need no encoding. mm_embeds copies them out (tests).
  void mm_set(const int32_t* spans, int n);
  void mm_encode(int i, const float* pixels);
  void mm_embeds(int i, bf16* out);
  void mm_put(int i, const bf16* in);  // replaces image i's embeddings (tests: another vision tower)
  bool has_vision() const { return opt_.vision != 0; }
  VisionMode vision_mode() const {
    return opt_.vision == 2 ? VisionMode::kHost : opt_.vision == 3 ? VisionMode::kFp8 : VisionMode::kDevice;
  }

 private:
  struct Act;
  void upload_ids(const int32_t* ids, int T);  // ids_ of the chunk at pos_ (image rows, M-RoPE positions)
  void copy_hist(int32_t* dst, const int32_t* ids, int n);  // lookup history (image rows as the image pad)
  void chunk(const int32_t* ids_dev, int T, bool advance = true);  // advances the device position (not pos_)
  int prefill_cut(int T, bool last) const;  // chunk length from pos_: up to the next planned checkpoint, or a multiple of 64
  void ensure_pages(int tokens);     // maps the logical pages of positions [0, tokens)
  // ... of a sequence with these pages and device page table (the running one's, or a batch lane's)
  void ensure_pages_of(std::vector<int32_t>& pages, int32_t* table, int tokens);
  // Writes physical pages to a device page table (as the kernels read it: page * kKvSlices, P7-13).
  void upload_pages(int32_t* table, const int32_t* pages, size_t n);
  void release_seq();                // returns the running sequence's pages
  void save_checkpoint(bool ring);   // at pos_ (nothing pending)
  // A checkpoint of tokens ids (pages: their KV pages) whose state is in a parking area (null: the running sequence's).
  void save_checkpoint_from(const char* area, const std::vector<int32_t>& ids, const std::vector<int32_t>& pages,
                            bool ring);
  int prompt_chunks(const int32_t* ids, int n, int budget, bool keep_tail, int* last);
  bool evict_checkpoint();           // the least recently used one; false when there is none
  void spec_body(const SpecSampling& smp);
  void spec_flush();  // replays the last verify's commits into the GDN / conv states
  void set_sbuf(int b);  // host side of the GDN state buffer switch
  int spec_rows(const SpecSampling& smp) const { return 1 + (kSpecBlock - 1) * std::max(1, smp.paths); }
  GdnSpec gdn_spec(int gdn_layer) const;
  bool fused(int T) const;
  bf16* cap_for(int layer) const;
  int32_t decode_step(int32_t id, float temperature, int top_k, float top_p, float uniform);
  void logits(const float* x_rows, int rows);  // final norm + lm_head into logits_
  void gdn(int layer, int T);
  void attn(int layer, int T);
  void mlp(int layer, int T);
  // GEMM on the activation already quantized into aq_ / asf_ (out += when residual).
  void gemm_q(const char* what, const LinearFp8& l, int T, float* out, bool residual);
  void gemm_q(const char* what, const LinearFp4& l, int T, float* out, bool residual);
  void gemm_q(const char* what, const LinearQ& l, int T, float* out, bool residual) {
    l.fp4 ? gemm_q(what, l.f4, T, out, residual) : gemm_q(what, l.f8, T, out, residual);
  }
  QuantOut q8(float in_scale) const;  // quantize into aq_
  QuantOut q4(float in_scale) const;  // quantize into aq_ / asf_
  QuantOut qin(const LinearQ& l) const { return l.fp4 ? q4(l.in_scale()) : q8(l.in_scale()); }
  // The input of a projection from its fp32 rows x [T][K] (FP8 projections: the producing kernel already
  // wrote aq_; NVFP4 ones quantize here).
  void quant_in(const LinearQ& l, const float* x, int T, int K);
  void mark(const char* label);  // start of an op (profiling only)
  void profile_collect(bool graph = false);  // after a step (graph: the replayed marks)

  struct Prof {
    struct Marks {
      std::vector<cudaEvent_t> events;
      std::vector<const char*> labels;  // labels[i] is timed from events[i] to events[i + 1]
      int used = 0;
    } eager, graph;
    bool on = false, capturing = false;
    std::vector<std::pair<std::string, std::pair<double, long>>> totals;  // label -> (ms, count)
    int steps = 0;
  } prof_;

  Options opt_;
  std::unique_ptr<SafeTensors> ckpt_;
  std::unique_ptr<SafeTensors> lm_ckpt_;  // Q27_LM_HEAD_CKPT: the NVFP4 lm_head of a BF16-lm_head model
  Budget vram_;                      // the VRAM budget (make_budget), for device_memory
  Pool dev_weights_{"weights"}, host_weights_{"embedding", true}, kv_{"kv"}, state_{"state"}, act_{"activations"},
      host_cache_{"state cache", true}, host_mm_{"image embeddings", true};
  Weights w_;
  cudaStream_t s_ = nullptr;
  int pos_ = 0;
  int max_pages_ = 0;

  // Sequence state.
  std::vector<KvLayer> kv_layers_;   // per attention layer
  int32_t* page_table_ = nullptr;    // logical -> physical page; unmapped entries hold the scratch page 0
  PagePool pool_;                    // physical pages 1 .. pool size
  std::vector<int32_t> seq_pages_;   // the running sequence's pages (host mirror of page_table_)
  std::vector<int32_t> seq_ids_;     // tokens fed so far (== pos_ once nothing is pending)
  int32_t spec_last_ = -1;           // last token returned by spec_begin / spec_step (fed by the next step)
  bool ring_valid_ = false;          // the draft's context ring holds the last 2,048 positions
  size_t kv_page_bytes_ = 0;         // one page of one layer's K (or V)
  // The KV pool is block-major (P7-13): pages come in blocks of kv_block_; a block holds the K and V slices
  // (kKvSlices: layer l's K is slice 2 l, its V 2 l + 1) of its pages, slice by slice, each slice the block's
  // pages in a row. The device page tables hold page_ref(p), so the kernels address a page of a layer's K as
  // before (base + page_ref * kv_page_bytes_, base = that slice of block 0). Whole blocks in a row are
  // contiguous memory (a parking area), and a slice of a block spans >= 2 MiB, so a long context's pages of
  // one layer do not each touch another GPU page (one page per block costs 1.5% at 131K).
  static constexpr int kKvSlices = 2 * dims::kAttnLayers;
  char* kv_base_ = nullptr;
  int kv_block_ = 1;
  int park_pages_ = 0;               // KV pages of a parking area (whole blocks)
  int32_t page_ref(int p) const { return (p / kv_block_) * kKvSlices * kv_block_ + p % kv_block_; }
  char* kv_page(int p, int slice) const {  // slice `slice` of page p
    return kv_base_ + (size_t(page_ref(p)) + size_t(slice) * kv_block_) * kv_page_bytes_;
  }
  std::vector<int32_t> pt_up_;       // upload_pages staging

  // Prefix cache.
  bool cache_on_ = false;
  std::vector<Checkpoint> ckpts_;
  size_t ckpt_bytes_ = 0;            // ssm + conv + draft ring
  std::vector<char*> slot_mem_;      // pinned [ckpt_bytes_] per slot, allocated as needed (grow_state_cache)
  std::vector<bool> slot_used_;
  char* stage_ = nullptr;            // device copy of one checkpoint on its way to the host
  cudaEvent_t stage_ev_ = nullptr, stage_in_ev_ = nullptr;  // staging free again / filled
  cudaStream_t cs_ = nullptr;        // device -> host copies of checkpoints
  std::vector<int32_t> plan_;        // checkpoint positions of the running prefill (ascending)
  uint64_t use_clock_ = 0;
  CacheStats cstats_;
  int32_t* pos_dev_ = nullptr;       // == pos_ between calls; read by the attention kernels
  float* conv_ = nullptr;            // [48][10240][3]
  float* ssm_ = nullptr;             // [48][48][128][128]: the current buffer (ssm_buf_[sbuf_host_])
  float* ssm_buf_[2] = {nullptr, nullptr};  // speculative verify moves the committed state between them
  int sbuf_host_ = 0;                // which buffer holds the committed state (mirrors the device's)
  int32_t* sbuf_ = nullptr;          // device copy (flipped by every speculative step)
  size_t conv_bytes_ = 0, ssm_bytes_ = 0;

  // Images of the running request (P4-12).
  struct MmSpan {
    int start, gh, gw, row;  // positions start .. start + gh * gw - 1, rows of mm_ from row
    int len() const { return gh * gw; }
  };
  std::vector<MmSpan> mm_spans_;
  bf16* mm_ = nullptr;               // pinned [image_tokens][5120] image embeddings (UVA)
  int32_t* rope_delta_ = nullptr;    // device: rotary position - index after the last image
  int32_t* mpos_ = nullptr;          // device [prefill_chunk][3]: M-RoPE positions of the chunk's rows
  bool mpos_on_ = false;             // the current prefill chunk rotates by mpos_
  void* vis_ws_ = nullptr;           // vision workspace (overlays x_ .. big2_ unless larger)
  size_t vis_ws_bytes_ = 0;

  // Parked sequences (Phase 6). An area holds ssm | conv | draft ring K | V (the checkpoint layout).
  static constexpr int kSeqScalars = 1 + 1 + 4 + kSpecBlock + kTreeMax;  // rope delta, budget, rng, block / verify ids
  struct SeqState {
    char* dev = nullptr;  // the device area
    bool used = false;
    int pos = 0;
    std::vector<int32_t> pages, ids, plan;
    std::vector<MmSpan> mm;
    int32_t spec_last = -1;
    bool ring_valid = false;
    SpecSampling smp{};
    int32_t scalars[kSeqScalars] = {};
  };
  std::vector<SeqState> parked_;

  // Batched steps (P6-10). While a batch is on, its lanes' step scalars live in the b_* arrays (lane z at
  // offset z) and their page tables in the rows of b_pt_; the running sequence's own copies are stale.
  bool batch_on_ = false;
  int batch_n_ = 0;                  // set while chunk() runs a batched verify
  std::vector<int32_t> blanes_;      // the lanes of the batch (-1: the running sequence)
  bool batch_pending_ = false;       // a step ran: its commits are replayed when the batch ends
  int bpar_ = 0;                     // the lanes' step parity (they step together)
  bool bwarm_ = false;
  AcceptLanes bacc_;                 // the lanes' sampling (chains) and top-k list lengths
  SeqLanes bsl_;                     // positions, rotary deltas and page tables of the lanes
  float *bconv_[kMaxLanes] = {}, *bssm_[kMaxLanes] = {};  // their GDN states (replayed in place)
  bf16 *bring_k_[kMaxLanes] = {}, *bring_v_[kMaxLanes] = {};  // their draft context rings
  int32_t *b_pos_ = nullptr, *b_delta_ = nullptr, *b_par_ = nullptr, *b_nc_ = nullptr, *b_budget_ = nullptr;
  int32_t *b_ids_ = nullptr, *b_blk_ = nullptr, *b_cpath_ = nullptr, *b_pt_ = nullptr;  // [lanes][8] / [lanes][max_pages_]
  uint64_t* b_rng_ = nullptr;        // [lanes][2]
  SpecTree* b_tree_ = nullptr;       // [lanes]
  SpecOut *b_out_ = nullptr, *b_out_dev_ = nullptr;  // [lanes][2] (by step parity), mapped pinned
  struct BatchGraph {
    std::vector<int32_t> lanes;
    int sbuf;
    AcceptLanes acc;
    cudaGraphExec_t exec;
    uint64_t used;
  };
  std::vector<BatchGraph> bgraphs_;
  struct LaneRef {  // the host state of a lane's sequence
    int* pos;
    std::vector<int32_t>* pages;
    std::vector<int32_t>* ids;
    int32_t* spec_last;
  };
  LaneRef lane_ref(int z);
  void batch_enter(const int32_t* lanes, int n);
  void batch_exit();                 // replays the pending commits, gives the lanes their step scalars back
  void batch_body(int n);            // one step of the lanes (captured)
  GdnSpec gdn_spec_lanes(int gdn_layer) const;
  struct Region {
    void* dev;
    char* area;
    size_t bytes;
  };
  std::vector<Region> seq_regions(char* area) const;  // the running sequence's device state and its place in an area
  void seq_scalars(int32_t* host, bool save);          // device scalars of the running sequence <-> host
  int area_first(int a) const;                         // the first KV page of parking area a
  void reserve_area(int a);                            // moves the KV pages out of area a and takes it
  void release_area(int a);                            // gives area a's pages to the KV pool
  void seq_take(SeqState& st);                         // the running sequence's host state -> st (none left)
  void seq_give(SeqState& st);                         // st -> the running sequence (after reset / seq_take)

  // Activations (sized for prefill_chunk tokens).
  float *x_ = nullptr, *h_ = nullptr, *big1_ = nullptr, *big2_ = nullptr;
  float *q_ = nullptr, *k_ = nullptr, *v_ = nullptr, *o_ = nullptr, *n_ = nullptr;
  float *a_ = nullptr, *b_ = nullptr, *g_ = nullptr, *beta_ = nullptr;
  uint8_t *aq_ = nullptr, *asf_ = nullptr;
  int32_t* ids_ = nullptr;
  float* logits_ = nullptr;
  int32_t* topk_ids_ = nullptr;
  float* topk_logp_ = nullptr;
  float* uniform_ = nullptr;
  int32_t* sampled_ = nullptr;
  bool mlp_bf16_ = true;                              // prefill MLP gate / up in BF16 (Q27_MLP_BF16)
  bool mlp_fused_ = true;   // prefill MLP gate | up + silu * up + NVFP4 quantization in one GEMM (Q27_MLP_FUSED)
  static constexpr int kSkinnyMaxM = 64;              // decode and verify-sized GEMMs use skinny.cu
  static constexpr size_t kSkinnyWsBytes = 32 << 20;  // split-K partials (up to 64 rows) and counters
  void* skinny_ws_ = nullptr;
  void* attn_ws_ = nullptr;  // split-KV partials of attention_decode
  void* fa_stage_ = nullptr;  // NVFP4 KV: the context rounded to E4M3 for the prefill attention (P7-8)
  size_t fa_stage_bytes_ = 0;
  bool skinny_ = true;         // Q27_NO_SKINNY=1: the many-row GEMMs for every M (Phase 1 path)
  bool skinny_check_ = false;
  std::string skip_;  // Q27_SKIP (debug timing)
  bool skip(const char* kind) const;
  // Decode step as a CUDA graph (Q27_NO_GRAPH=1: eager). Not used while profiling.
  bool graphs_ = true, graph_warm_ = false;
  struct GraphKey {
    float temperature;
    int top_k;
    float top_p;
    bool operator==(const GraphKey& o) const {
      return temperature == o.temperature && top_k == o.top_k && top_p == o.top_p;
    }
  } graph_key_{};
  cudaGraphExec_t graph_exec_ = nullptr;
  int graph_sbuf_ = -1;  // the GDN state buffer the decode graph was captured with
  // Captured graphs by sampling settings (sequences with different settings take turns, Phase 6); the
  // least recently used one is destroyed beyond kGraphCache. graph_exec_ / spec_graph_ point into them.
  static constexpr int kGraphCache = 4;
  struct CachedGraph {
    GraphKey key;
    SpecSampling smp;
    int sbuf;
    cudaGraphExec_t exec;
    uint64_t used;
  };
  std::vector<CachedGraph> dgraphs_, sgraphs_;
  uint64_t graph_clock_ = 0;
  void drop_graphs();
  struct HostIo {  // pinned, mapped: the decode step's input id, uniform and sampled token
    int32_t id, out;
    float uniform;
  }* host_io_ = nullptr;
  HostIo* host_io_dev_ = nullptr;  // its device address
  bool decode_memcpy_ = false;  // Q27_DECODE_MEMCPY=1: copy the step's input / output instead

  // Speculative decoding.
  std::unique_ptr<Draft> draft_;
  bool verify_ = false;         // chunk() runs a verify block (GDN without state advance, replay first)
  int verify_paths_ = 1;        // draft tree paths of that block
  bf16* cap_ = nullptr;         // target hidden of layers 5 / 19 / 33 / 47 / 61: [rows][5][5120] (null: off)
  bf16* cap_buf_ = nullptr;     // [prefill_chunk][25600]
  float* rep_ = nullptr;        // GDN verify inputs: [2][48][rep_rows_ tokens]
  static constexpr size_t kRepTokenFloats =  // raw conv input, k, v, a | b per token
      size_t(dims::kConvDim) + dims::kKHeads * dims::kKDim + dims::kVHeads * dims::kVDim + 2 * dims::kVHeads;
  int rep_rows_ = kVerifyMaxT;  // verify rows (tree nodes, or the rows of a batched step: 8 per lane)
  int32_t *par_ = nullptr, *ncommit_ = nullptr;  // step parity, tokens committed by the last verify
  int32_t* budget_ = nullptr;                    // tokens the steps may still output (spec_begin's max_new)
  uint64_t* rng_ = nullptr;                      // seed, step counter
  int32_t *ids_ver_ = nullptr, *block_ids_ = nullptr;  // verify input [32] (tree nodes); draft block [8]
  float* tv_ = nullptr;
  int32_t* ti_ = nullptr;     // target top-k lists of the verify rows [32][64]
  SpecTree* tree_ = nullptr;  // the step's draft tree
  int32_t* cpath_ = nullptr;  // rows of the last accepted path [8]
  KvLayer* kv_layers_dev_ = nullptr;  // device copy of kv_layers_ (KV commit)
  SpecOut *spec_out_ = nullptr, *spec_out_dev_ = nullptr;  // [2] (by step parity), mapped pinned
  int32_t* hist_ = nullptr;   // committed tokens by position (lookup drafting)
  uint32_t* lk_best_ = nullptr;
  Lookup lookup_{};           // Q27_LOOKUP (default off), Q27_LOOKUP_MIN, Q27_LOOKUP_AGREE
  int64_t stat_steps_ = 0, stat_lookup_ = 0, stat_tokens_ = 0;
  int spec_paths_ = 1;        // Q27_SPEC_PATHS: draft walks merged into the verify tree (1: chain)
  bool spec_pending_ = false;  // a verify ran since the last flush / reset
  bool spec_pipeline_ = true;  // Q27_SPEC_PIPELINE: launch the next step before reading this one's result
  bool spec_inflight_ = false; // a step launched ahead whose result has not been returned
  int inflight_slot_ = 0;      // its SpecOut slot
  int par_host_ = 0;           // the device's step parity at the next launch (= its SpecOut slot)
  cudaEvent_t spec_ev_[2] = {};
  bool spec_warm_ = false;
  cudaGraphExec_t spec_graph_ = nullptr;
  SpecSampling spec_key_{};
  SpecSampling spec_smp_{};
  int32_t stop_[2] = {-1, -1};  // generation_config.json eos_token_id
};

}  // namespace q27
