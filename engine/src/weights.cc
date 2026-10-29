#include "weights.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>

#include "common.h"
#include "prefetch.h"

namespace q27 {
namespace {

const std::string kLM = "model.language_model";

// The input (activation) scale of a layer quantized at load, which has no calibrated one. NVFP4 activations keep
// blocks up to 2688 * scale (e4m3 448 * e2m1 6) unclipped; calibrated lm_head scales are 0.021 (NVIDIA's) and 0.027
// (a fine-tune of the same model). 256 / 2688 leaves room for 3.6 times the larger, and the e4m3 block scales stay in
// their normal range (3 mantissa bits) for blocks down to an amax of 6 * scale * 2^-6 = 0.009.
constexpr float kBf16InScale = 256.f / 2688.f;

// The largest |value| of a BF16 tensor, read in chunks (the global scale of fp4_from_bf16).
float bf16_amax(const SafeTensors& ckpt, const std::string& name) {
  const auto& info = ckpt.info(name);
  constexpr int64_t kChunk = 64 << 20;
  std::vector<uint16_t> buf(size_t(std::min<int64_t>(kChunk, info.nbytes)) / 2);
  uint16_t top = 0;
  for (int64_t off = 0; off < info.nbytes; off += kChunk) {
    const int64_t len = std::min<int64_t>(kChunk, info.nbytes - off);
    ckpt.read_range(name, off, len, buf.data());
    for (int64_t i = 0; i < len / 2; i++) top = std::max<uint16_t>(top, buf[size_t(i)] & 0x7fff);
  }
  Q27_CHECK(top < 0x7f80, name + ": not finite");  // (exponent all ones: inf / nan)
  const uint32_t bits = uint32_t(top) << 16;
  float v;
  std::memcpy(&v, &bits, 4);
  return v;
}

float bf16_value(uint16_t v) {
  const uint32_t bits = uint32_t(v) << 16;
  float f;
  std::memcpy(&f, &bits, 4);
  return f;
}

// Per-channel scales (BF16 or F32) as floats.
std::vector<float> read_scales(const SafeTensors& ckpt, const std::string& name) {
  const auto& info = ckpt.info(name);
  std::vector<uint8_t> raw(size_t(info.nbytes));
  ckpt.read(name, raw.data());
  std::vector<float> out(info.dtype == "F32" ? raw.size() / 4 : raw.size() / 2);
  for (size_t i = 0; i < out.size(); i++) {
    if (info.dtype == "F32") {
      std::memcpy(&out[i], raw.data() + 4 * i, 4);
    } else {
      uint16_t v;
      std::memcpy(&v, raw.data() + 2 * i, 2);
      out[i] = bf16_value(v);
    }
  }
  return out;
}

float e4m3_value(uint8_t v) {
  const int e = (v >> 3) & 15, m = v & 7;
  const float f = e == 0 ? std::ldexp(m / 8.f, -6) : std::ldexp(1.f + m / 8.f, e - 7);
  return v & 0x80 ? -f : f;
}

}  // namespace

void Weights::tensor(const SafeTensors& ckpt, const std::string& name, Pool& pool, const void** dst, int sf_rows,
                     int sf_kblocks) {
  const auto& info = ckpt.info(name);
  size_t bytes = sf_rows ? size_t(sf_bytes(sf_rows, sf_kblocks)) : size_t(info.nbytes);
  size_t off = pool.add(name, bytes);
  jobs_.push_back({&ckpt, name, &pool, off, sf_rows, sf_kblocks, {}});
  ptrs_.push_back({dst, {&pool, off}});
}

void Weights::scalar(const SafeTensors& ckpt, const std::string& name, float* dst, bool inv) {
  const auto& info = ckpt.info(name);  // (here, so that a plan finds a checkpoint without it, e.g. weight-only)
  // (BF16 scalars: Red Hat's NVFP4 weight_global_scale)
  Q27_CHECK((info.dtype == "F32" && info.nbytes == 4) || (info.dtype == "BF16" && info.nbytes == 2),
            name + ": expected an f32 scalar");
  scalars_.push_back({&ckpt, name, dst, inv});
}

bool Weights::fp8(const SafeTensors& ckpt, const std::vector<std::pair<std::string, LinearFp8*>>& group, Pool& dev) {
  for (auto& [p, l] : group) {
    const auto& info = ckpt.info(p + ".weight");
    Q27_CHECK(info.dtype == "F8_E4M3", p + ": expected an FP8 weight");
    l->out = int(info.shape[0]);
    l->in = int(info.shape[1]);
    tensor(ckpt, p + ".weight", dev, reinterpret_cast<const void**>(&l->w));
  }
  int channel = 0;
  for (auto& [p, l] : group) {
    const auto& si = ckpt.info(p + ".weight_scale");
    if (si.nbytes <= 4) {  // one scale, with the calibrated input scale (modelopt, the nvidia checkpoint)
      scalar(ckpt, p + ".weight_scale", &l->w_scale);
      scalar(ckpt, p + ".input_scale", &l->in_scale);
      continue;
    }
    // One per output channel ([out] or [out][1], BF16 or F32), the input quantized per token at run time.
    Q27_CHECK((si.dtype == "BF16" || si.dtype == "F32") &&
                  si.nbytes == int64_t(l->out) * (si.dtype == "F32" ? 4 : 2),
              p + ".weight_scale: expected one scale or one per output channel");
    Q27_CHECK(!ckpt.has(p + ".input_scale") && !ckpt.has(p + ".input_global_scale"),
              p + ": FP8 weights with per-channel scales and a static input scale are not supported");
    const size_t off = dev.add(p + ".weight_scale (f32)", size_t(l->out) * 4);
    jobs_.push_back({&ckpt, p + ".weight_scale", &dev, off});
    jobs_.back().f32 = true;
    ptrs_.push_back({reinterpret_cast<const void**>(&l->w_scales), {&dev, off}});
    channel++;
  }
  Q27_CHECK(channel == 0 || channel == int(group.size()), group.front().first + ": mixed FP8 scale layouts");
  if (channel) dyn8 = true;
  return channel > 0;
}

void Weights::fp4(const SafeTensors& ckpt, const std::vector<std::pair<std::string, LinearFp4*>>& group, Pool& dev) {
  for (auto& [p, l] : group) {
    const bool ct = ckpt.has(p + ".weight_packed");  // compressed-tensors: global scales are inverses
    const std::string wn = p + (ct ? ".weight_packed" : ".weight");
    const auto& info = ckpt.info(wn);
    Q27_CHECK(info.dtype == "U8", p + ": expected an NVFP4 weight");
    l->out = int(info.shape[0]);
    l->in = int(info.shape[1]) * 2;
    tensor(ckpt, wn, dev, reinterpret_cast<const void**>(&l->w));
    scalar(ckpt, p + (ct ? ".weight_global_scale" : ".weight_scale_2"), &l->w_scale2, ct);
    scalar(ckpt, p + (ct ? ".input_global_scale" : ".input_scale"), &l->in_scale, ct);
  }
  for (auto& [p, l] : group)
    tensor(ckpt, p + ".weight_scale", dev, reinterpret_cast<const void**>(&l->sf), l->out, l->in / 16);
}

void Weights::fp4_from_bf16(const SafeTensors& ckpt, const std::string& p, Pool& dev, LinearFp4& l) {
  const auto& info = ckpt.info(p + ".weight");
  Q27_CHECK(info.dtype == "BF16" && info.shape.size() == 2, p + ": expected a BF16 matrix");
  l.out = int(info.shape[0]);
  l.in = int(info.shape[1]);
  Q27_CHECK(l.out % 128 == 0 && l.in % 64 == 0, p + ": unexpected shape for NVFP4");
  const size_t off = dev.add(p + ".weight (NVFP4 at load)", size_t(l.out) * l.in / 2);
  const size_t soff = dev.add(p + ".weight_scale (NVFP4 at load)", size_t(sf_bytes(l.out, l.in / 16)));
  jobs_.push_back({&ckpt, p + ".weight", &dev, off});
  jobs_.back().q4 = &l;
  jobs_.back().q8_scale = soff;
  ptrs_.push_back({reinterpret_cast<const void**>(&l.w), {&dev, off}});
  ptrs_.push_back({reinterpret_cast<const void**>(&l.sf), {&dev, soff}});
  l.in_scale = kBf16InScale;  // (w_scale2: load, from the amax)
}

void Weights::fp4_from_fp8(const SafeTensors& ckpt, const std::string& p, const LinearFp8& src, Pool& dev,
                           LinearFp4& l) {
  l.out = src.out;
  l.in = src.in;
  Q27_CHECK(l.out % 128 == 0 && l.in % 64 == 0, p + ": unexpected shape for NVFP4");
  const size_t off = dev.add(p + ".weight (NVFP4 copy)", size_t(l.out) * l.in / 2);
  const size_t soff = dev.add(p + ".weight_scale (NVFP4 copy)", size_t(sf_bytes(l.out, l.in / 16)));
  jobs_.push_back({&ckpt, p + ".weight", &dev, off});
  jobs_.back().q4 = &l;
  jobs_.back().q8_scale = soff;
  jobs_.back().from8 = &src;
  ptrs_.push_back({reinterpret_cast<const void**>(&l.w), {&dev, off}});
  ptrs_.push_back({reinterpret_cast<const void**>(&l.sf), {&dev, soff}});
  l.in_scale = kBf16InScale;  // (w_scale2: load, from the largest channel scale)
}

void Weights::fp4_glu(const SafeTensors& ckpt, const std::string& gate, const std::string& up, LinearFp4& g,
                      LinearFp4& u, Pool& dev) {
  fp4(ckpt, {{gate, &g}, {up, &u}}, dev);
  // fp4 added the jobs gate data, up data, gate scales, up scales: the up's go to the gate's offsets.
  Job* j = &jobs_[jobs_.size() - 4];
  Q27_CHECK(!dev.host() && g.out == u.out && g.in == u.in && g.out % 128 == 0, "unexpected MLP gate / up");
  j[0].glu = j[2].glu = 1, j[1].glu = j[3].glu = 2;
  j[1].offset = j[0].offset, j[3].offset = j[2].offset;
}

void Weights::proj(const SafeTensors& ckpt, const std::vector<std::pair<std::string, LinearQ*>>& group, Pool& dev) {
  const std::string& p0 = group.front().first;
  if (ckpt.has(p0 + ".weight") && ckpt.info(p0 + ".weight").dtype == "F8_E4M3") {
    std::vector<std::pair<std::string, LinearFp8*>> g;
    for (auto& [p, l] : group) g.push_back({p, &l->f8});
    const bool channel = fp8(ckpt, g, dev);
    for (auto& [p, l] : group) l->dyn = channel;
    return;
  }
  std::vector<std::pair<std::string, LinearFp4*>> g;
  for (auto& [p, l] : group) l->fp4 = true, g.push_back({p, &l->f4});
  fp4(ckpt, g, dev);
  fp4_proj = true;
}

void Weights::bf16_weight(const SafeTensors& ckpt, const std::string& p, Pool& dev, const bf16** dst) {
  if (!ckpt.has(p + ".weight_packed")) {
    Q27_CHECK(ckpt.info(p + ".weight").dtype == "BF16", p + ": expected a BF16 weight");
    tensor(ckpt, p + ".weight", dev, reinterpret_cast<const void**>(dst));
    return;
  }
  const auto& info = ckpt.info(p + ".weight_packed");
  Q27_CHECK(ckpt.info(p + ".weight_scale").nbytes == info.nbytes / 8 && ckpt.info(p + ".weight_global_scale").nbytes == 4,
            p + ": unexpected NVFP4 scales");
  const size_t off = dev.add(p + ".weight", size_t(info.shape[0]) * info.shape[1] * 2 * sizeof(bf16));
  jobs_.push_back({&ckpt, p + ".weight_packed", &dev, off, 0, 0, p});
  ptrs_.push_back({reinterpret_cast<const void**>(dst), {&dev, off}});
}

namespace {

// Bytes [off, off + len) of the gate (half 0) or the up (half 1), in buf, to their places in the interleaved
// gate | up at dst: blocks of 64 rows (block bytes each), the gate's block b at 2 b, the up's at 2 b + 1.
void copy_glu(char* dst, size_t off, const void* buf, size_t len, size_t block, int half, cudaStream_t s) {
  const char* src = static_cast<const char*>(buf);
  auto at = [&](size_t o) { return dst + (2 * (o / block) + half) * block + o % block; };
  const size_t end = off + len;
  if (off % block) {  // the end of a block
    const size_t n = std::min(end, (off / block + 1) * block) - off;
    Q27_CUDA(cudaMemcpyAsync(at(off), src, n, cudaMemcpyHostToDevice, s));
    src += n, off += n;
  }
  if (const size_t rows = (end - off) / block) {  // whole blocks: every other one
    Q27_CUDA(cudaMemcpy2DAsync(at(off), 2 * block, src, block, block, rows, cudaMemcpyHostToDevice, s));
    src += rows * block, off += rows * block;
  }
  if (off < end) Q27_CUDA(cudaMemcpyAsync(at(off), src, end - off, cudaMemcpyHostToDevice, s));
}

// Concatenation along the output dimension of layers stored back to back.
LinearFp8 fuse8(std::initializer_list<const LinearFp8*> parts) {
  LinearFp8 f = **parts.begin();
  f.out = 0;
  f.alpha.count = 0;
  for (const LinearFp8* l : parts) {
    Q27_CHECK(l->w == f.w + size_t(f.out) * f.in && l->in == f.in, "fused FP8 layers are not adjacent");
    Q27_CHECK(l->in_scale == f.in_scale, "fused FP8 layers have different input scales");
    Q27_CHECK(!f.w_scales == !l->w_scales && (!f.w_scales || l->w_scales == f.w_scales + f.out),
              "fused FP8 layers' channel scales are not adjacent");
    Q27_CHECK(f.alpha.count < AlphaSegments::kMax, "too many fused layers");
    f.alpha.begin[f.alpha.count] = f.out;
    f.alpha.alpha[f.alpha.count++] = l->w_scales ? 1.f : l->in_scale * l->w_scale;
    f.out += l->out;
  }
  return f;
}

LinearFp4 fuse4(std::initializer_list<const LinearFp4*> parts) {
  LinearFp4 f = **parts.begin();
  f.out = 0;
  f.alpha.count = 0;
  for (const LinearFp4* l : parts) {
    Q27_CHECK(l->w == f.w + size_t(f.out) * (f.in / 2) && l->in == f.in, "fused NVFP4 layers are not adjacent");
    Q27_CHECK(f.out % 128 == 0 && l->sf == f.sf + sf_bytes(f.out, f.in / 16), "fused NVFP4 scales are not adjacent");
    Q27_CHECK(l->in_scale == f.in_scale, "fused NVFP4 layers have different input scales");
    f.alpha.begin[f.alpha.count] = f.out;
    f.alpha.alpha[f.alpha.count++] = l->in_scale * l->w_scale2;
    f.out += l->out;
  }
  return f;
}

LinearQ fuseq(const LinearQ& a, const LinearQ& b) {
  LinearQ f;
  f.fp4 = a.fp4, f.dyn = a.dyn;
  if (f.fp4) f.f4 = fuse4({&a.f4, &b.f4}); else f.f8 = fuse8({&a.f8, &b.f8});
  return f;
}
LinearQ fuseq(const LinearQ& a, const LinearQ& b, const LinearQ& c) {
  LinearQ f;
  f.fp4 = a.fp4, f.dyn = a.dyn;
  if (f.fp4) f.f4 = fuse4({&a.f4, &b.f4, &c.f4}); else f.f8 = fuse8({&a.f8, &b.f8, &c.f8});
  return f;
}

}  // namespace

void Weights::fuse() {
  auto one8 = [](LinearFp8& l) { l.alpha = AlphaSegments::one(l.w_scales ? 1.f : l.in_scale * l.w_scale); };
  auto one4 = [](LinearFp4& l) { l.alpha = AlphaSegments::one(l.in_scale * l.w_scale2); };
  auto oneq = [&](LinearQ& l) { l.fp4 ? one4(l.f4) : one8(l.f8); };
  for (int i = 0; i < dims::kLayers; i++) {
    auto& L = layers[i];
    if (dims::is_attn(i)) {
      for (auto* l : {&L.attn.q, &L.attn.k, &L.attn.v, &L.attn.o}) oneq(*l);
      L.attn.qkv = fuseq(L.attn.q, L.attn.k, L.attn.v);
    } else {
      for (auto* l : {&L.gdn.qkv, &L.gdn.z, &L.gdn.out}) oneq(*l);
      L.gdn.qkvz = fuseq(L.gdn.qkv, L.gdn.z);
    }
    if (L.mlp8) {
      for (auto* l : {&L.gate8, &L.up8, &L.down8}) one8(*l);
      L.gate_up8 = fuse8({&L.gate8, &L.up8});
      continue;
    }
    for (auto* l : {&L.gate, &L.up, &L.down}) one4(*l);
    // (interleaved: fp4_glu. The GEMMs run with alpha 1, the silu step applies the gate's and the up's.)
    Q27_CHECK(L.up.w == L.gate.w + size_t(L.gate.out) * (L.gate.in / 2) &&
                  L.up.sf == L.gate.sf + sf_bytes(L.gate.out, L.gate.in / 16) && L.up.in_scale == L.gate.in_scale,
              "MLP gate / up are not adjacent");
    L.gate_up = L.gate;
    L.gate_up.out = 2 * L.gate.out;
    L.gate_up.alpha = AlphaSegments::one(1.f);
    for (auto* l : {&L.gate, &L.up}) l->w = nullptr, l->sf = nullptr;
  }
  one4(lm_head);
  if (lm_head8.w) one8(lm_head8);
}

void Weights::plan(const SafeTensors& ckpt, const SafeTensors* lm_ckpt, Pool& dev, Pool& host, bool fp4_head,
                   Pool* head_pool) {
  using namespace dims;
  const Json& cfg = ckpt.config().has("text_config") ? ckpt.config()["text_config"] : ckpt.config();
  Q27_CHECK(cfg["hidden_size"].as_int() == kHidden && cfg["num_hidden_layers"].as_int() == kLayers &&
                cfg["vocab_size"].as_int() == kVocab && cfg["intermediate_size"].as_int() == kInter &&
                cfg["num_attention_heads"].as_int() == kHeads && cfg["num_key_value_heads"].as_int() == kKvHeads &&
                cfg["head_dim"].as_int() == kHeadDim && cfg["linear_num_value_heads"].as_int() == kVHeads &&
                cfg["linear_num_key_heads"].as_int() == kKHeads,
            "config.json does not match the dimensions this runtime is built for");
  for (int i = 0; i < kLayers; i++)
    Q27_CHECK((cfg["layer_types"][i].as_str() == "full_attention") == is_attn(i), "unexpected layer_types");

  auto bf = [&](const std::string& name, const bf16** dst) {
    tensor(ckpt, name, dev, reinterpret_cast<const void**>(dst));
  };
  layers.resize(kLayers);
  for (int i = 0; i < kLayers; i++) {
    auto& L = layers[i];
    std::string p = kLM + ".layers." + std::to_string(i);
    bf(p + ".input_layernorm.weight", &L.in_norm);
    bf(p + ".post_attention_layernorm.weight", &L.post_norm);
    if (is_attn(i)) {
      std::string a = p + ".self_attn";
      proj(ckpt, {{a + ".q_proj", &L.attn.q}, {a + ".k_proj", &L.attn.k}, {a + ".v_proj", &L.attn.v}}, dev);
      proj(ckpt, {{a + ".o_proj", &L.attn.o}}, dev);
      bf(a + ".q_norm.weight", &L.attn.q_norm);
      bf(a + ".k_norm.weight", &L.attn.k_norm);
    } else {
      std::string g = p + ".linear_attn";
      proj(ckpt, {{g + ".in_proj_qkv", &L.gdn.qkv}, {g + ".in_proj_z", &L.gdn.z}}, dev);
      proj(ckpt, {{g + ".out_proj", &L.gdn.out}}, dev);
      bf16_weight(ckpt, g + ".in_proj_a", dev, &L.gdn.a);
      bf16_weight(ckpt, g + ".in_proj_b", dev, &L.gdn.b);
      bf(g + ".A_log", &L.gdn.A_log);
      bf(g + ".dt_bias", &L.gdn.dt_bias);
      bf(g + ".conv1d.weight", &L.gdn.conv);
      bf(g + ".norm.weight", &L.gdn.norm);
    }
    const std::string m = p + ".mlp";
    if (ckpt.has(m + ".gate_proj.weight") && ckpt.info(m + ".gate_proj.weight").dtype == "F8_E4M3") {
      // (FP8 MLPs: per-channel scales only, the down GEMM's input is quantized per token)
      L.mlp8 = mlp8 = true;
      Q27_CHECK(fp8(ckpt, {{m + ".gate_proj", &L.gate8}, {m + ".up_proj", &L.up8}}, dev) &&
                    fp8(ckpt, {{m + ".down_proj", &L.down8}}, dev),
                m + ": FP8 MLPs need per-channel scales");
      Q27_CHECK(L.gate8.out == kInter && L.up8.out == kInter && L.down8.in == kInter, m + ": unexpected shape");
      continue;
    }
    fp4_glu(ckpt, m + ".gate_proj", m + ".up_proj", L.gate, L.up, dev);
    fp4(ckpt, {{m + ".down_proj", &L.down}}, dev);
  }
  bf(kLM + ".norm.weight", &final_norm);
  // lm_head: NVFP4. A BF16 one (QUASAR and its fine-tunes keep it) comes from lm_ckpt if given (QUASAR: the nvidia
  // checkpoint quantized the same original lm_head), else it is quantized here.
  const bool lm_bf16 = ckpt.has("lm_head.weight") && ckpt.info("lm_head.weight").dtype == "BF16";
  if (ckpt.has("lm_head.weight") && ckpt.info("lm_head.weight").dtype == "F8_E4M3") {
    // FP8 with per-channel scales (unsloth's checkpoint): the target's logits from it as it is; the draft model
    // reads an NVFP4 lm_head, quantized from it at load.
    Q27_CHECK(fp8(ckpt, {{"lm_head", &lm_head8}}, dev), "lm_head: an FP8 lm_head needs per-channel scales");
    if (fp4_head) fp4_from_fp8(ckpt, "lm_head", lm_head8, head_pool ? *head_pool : dev, lm_head);
    log("lm_head: FP8 with per-channel scales%s",
        !fp4_head ? "" : head_pool ? " (an NVFP4 copy at load for the draft's vocabulary head)"
                                   : " (and an NVFP4 copy for the draft model)");
  } else if (lm_bf16 && !lm_ckpt) {
    fp4_from_bf16(ckpt, "lm_head", dev, lm_head);
    log("lm_head: BF16 in the checkpoint, quantized to NVFP4 at load");
  } else {
    fp4(lm_bf16 ? *lm_ckpt : ckpt, {{"lm_head", &lm_head}}, dev);
  }
  tensor(ckpt, kLM + ".embed_tokens.weight", host, reinterpret_cast<const void**>(&embed));
}

void Weights::plan_vision(const SafeTensors& ckpt, Pool& dev, Pool& host, VisionMode mode) {
  const Json& v = ckpt.config()["vision_config"];
  Q27_CHECK(v["hidden_size"].as_int() == vdims::kHidden && v["num_heads"].as_int() == vdims::kHeads &&
                v["depth"].as_int() == vdims::kDepth && v["intermediate_size"].as_int() == vdims::kInter &&
                v["out_hidden_size"].as_int() == vdims::kOut && v["patch_size"].as_int() == vdims::kPatchPixels &&
                v["spatial_merge_size"].as_int() == vdims::kMerge &&
                v["num_position_embeddings"].as_int() == vdims::kPosSide * vdims::kPosSide,
            "vision_config does not match the vision tower this runtime is built for");
  const std::string p = "model.visual.";
  auto bf = [&](const std::string& name, const bf16** dst) {
    Q27_CHECK(ckpt.info(p + name).dtype == "BF16", p + name + ": expected BF16");
    tensor(ckpt, p + name, dev, reinterpret_cast<const void**>(dst));
  };
  auto mat = [&](const std::string& name, VisionMat& m) {
    const auto& info = ckpt.info(p + name);
    Q27_CHECK(info.dtype == "BF16", p + name + ": expected BF16");
    m.out = int(info.shape[0]);
    m.in = int(info.nbytes / 2 / info.shape[0]);  // (the patch embedding is [1152][3][2][16][16])
    if (mode == VisionMode::kFp8) {
      const size_t off = dev.add(p + name + " (e4m3)", size_t(m.out) * m.in);
      const size_t soff = dev.add(p + name + " (row scales)", size_t(m.out) * 4);
      jobs_.push_back({&ckpt, p + name, &dev, off, 0, 0, {}, m.out, soff});
      ptrs_.push_back({reinterpret_cast<const void**>(&m.q), {&dev, off}});
      ptrs_.push_back({reinterpret_cast<const void**>(&m.scale), {&dev, soff}});
    } else {
      tensor(ckpt, p + name, mode == VisionMode::kHost ? host : dev, reinterpret_cast<const void**>(&m.w));
    }
  };
  VisionWeights& V = vision;
  V.mode = mode;
  mat("patch_embed.proj.weight", V.patch);
  bf("patch_embed.proj.bias", &V.patch_b);
  bf("pos_embed.weight", &V.pos);
  for (int i = 0; i < vdims::kDepth; i++) {
    const std::string b = "blocks." + std::to_string(i) + ".";
    VisionBlock& B = V.blocks[i];
    bf(b + "norm1.weight", &B.ln1_w), bf(b + "norm1.bias", &B.ln1_b);
    mat(b + "attn.qkv.weight", B.qkv), bf(b + "attn.qkv.bias", &B.qkv_b);
    mat(b + "attn.proj.weight", B.proj), bf(b + "attn.proj.bias", &B.proj_b);
    bf(b + "norm2.weight", &B.ln2_w), bf(b + "norm2.bias", &B.ln2_b);
    mat(b + "mlp.linear_fc1.weight", B.fc1), bf(b + "mlp.linear_fc1.bias", &B.fc1_b);
    mat(b + "mlp.linear_fc2.weight", B.fc2), bf(b + "mlp.linear_fc2.bias", &B.fc2_b);
  }
  bf("merger.norm.weight", &V.norm_w), bf("merger.norm.bias", &V.norm_b);
  mat("merger.linear_fc1.weight", V.fc1), bf("merger.linear_fc1.bias", &V.fc1_b);
  mat("merger.linear_fc2.weight", V.fc2), bf("merger.linear_fc2.bias", &V.fc2_b);
}

void Weights::load(cudaStream_t s) {
  auto t0 = std::chrono::steady_clock::now();
  for (auto& [dst, loc] : ptrs_) *dst = loc.first->at<void>(loc.second);
  // The reads in the order the loop below takes them, read ahead by several threads (prefetch.h): the f32
  // scalars, then per job the embedding (straight into its pinned table), the scale factors and the plain
  // tensors in slot-sized chunks. Small NVFP4 layers, VISION=fp8 matrices and scale factors larger than a slot
  // are read here directly.
  constexpr int64_t kChunk = Prefetcher::kSlotBytes;
  std::vector<Prefetcher::Read> plan;
  for (auto& sj : scalars_) {
    const auto& info = sj.src->info(sj.name);  // (F32, or BF16: read into the low bytes, converted below)
    plan.push_back({sj.src, sj.name, 0, info.nbytes, sj.dst});
  }
  for (auto& j : jobs_) {
    const auto& info = j.src->info(j.name);
    if (!j.dq.empty() || j.q8_rows || j.q4 || j.f32 || (j.sf_rows && !Prefetcher::fits(info.nbytes))) continue;
    if (j.sf_rows) {
      plan.push_back({j.src, j.name, 0, info.nbytes, nullptr});
      continue;
    }
    char* host = j.pool->host() ? j.pool->at<char>(j.offset) : nullptr;
    for (int64_t off = 0; off < info.nbytes; off += kChunk)
      plan.push_back({j.src, j.name, off, std::min<int64_t>(kChunk, info.nbytes - off), host ? host + off : nullptr});
  }
  Prefetcher pf(std::move(plan));
  size_t ri = 0;
  for (auto& sj : scalars_) {
    pf.get(ri++, sj.name, 0);
    if (sj.src->info(sj.name).dtype == "BF16") {
      uint16_t v;
      std::memcpy(&v, sj.dst, 2);
      *sj.dst = bf16_value(v);
    }
    if (sj.inv) *sj.dst = 1.f / *sj.dst;
  }
  for (auto& j : jobs_)
    if (j.q4) {
      // (an FP8 source: its rows' largest values are 448 times their scales)
      float amax = 0.f;
      if (j.from8) {
        const std::string sname = j.name.substr(0, j.name.size() - 7) + ".weight_scale";  // (j.name: <prefix>.weight)
        for (float v : read_scales(*j.src, sname)) amax = std::max(amax, 448.f * std::fabs(v));
      } else {
        amax = bf16_amax(*j.src, j.name);
      }
      Q27_CHECK(amax > 0.f, j.name + ": all zero");
      j.q4->w_scale2 = amax / (6.f * 448.f);
    }
  fuse();
  // Two pinned staging buffers for the direct reads: the next chunk is read while the previous one is copied.
  constexpr size_t kStage = 128 << 20;
  void* stage[2] = {nullptr, nullptr};
  cudaEvent_t done[2];
  for (int i = 0; i < 2; i++) {
    Q27_CUDA(cudaEventCreateWithFlags(&done[i], cudaEventDisableTiming));
    Q27_CUDA(cudaEventRecord(done[i], s));
  }
  auto staging = [&] {
    for (int i = 0; i < 2; i++)
      if (!stage[i]) Q27_CUDA(cudaHostAlloc(&stage[i], kStage, cudaHostAllocMapped));
  };
  int cur = 0;
  size_t total = 0;
  void* q8_tmp = nullptr;  // (VISION=fp8: one BF16 matrix at a time)
  size_t q8_tmp_bytes = 0;
  for (auto& j : jobs_) {
    const SafeTensors& ckpt = *j.src;
    const auto& info = ckpt.info(j.name);
    if (!j.from8) total += size_t(info.nbytes);  // (an NVFP4 copy reads nothing)
    char* dst = j.pool->at<char>(j.offset);
    if (j.f32) {  // per-channel scales -> f32
      const std::vector<float> v = read_scales(ckpt, j.name);
      Q27_CHECK(v.size() * 4 <= kStage, j.name + ": larger than the staging buffer");
      staging();
      Q27_CUDA(cudaEventSynchronize(done[cur]));
      std::memcpy(stage[cur], v.data(), v.size() * 4);
      Q27_CUDA(cudaMemcpyAsync(dst, stage[cur], v.size() * 4, cudaMemcpyHostToDevice, s));
      Q27_CUDA(cudaEventRecord(done[cur], s));
      cur ^= 1;
      continue;
    }
    if (!j.dq.empty()) {  // small NVFP4 layer -> BF16 (e2m1 * scale / global, compressed-tensors)
      const int rows = int(info.shape[0]), cols = int(info.shape[1]) * 2;
      std::vector<uint8_t> packed(size_t(info.nbytes)), sf(size_t(rows) * cols / 16);
      float global = 0.f;
      ckpt.read(j.name, packed.data());
      Q27_CHECK(ckpt.info(j.dq + ".weight_scale").nbytes == int64_t(sf.size()), j.dq + ": unexpected scale shape");
      ckpt.read(j.dq + ".weight_scale", sf.data());
      ckpt.read(j.dq + ".weight_global_scale", &global);
      static const float kE2m1[8] = {0.f, 0.5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f};
      Q27_CHECK(size_t(rows) * cols * sizeof(bf16) <= kStage, j.dq + ": larger than the staging buffer");
      staging();
      Q27_CUDA(cudaEventSynchronize(done[cur]));
      bf16* out = static_cast<bf16*>(stage[cur]);
      for (int r = 0; r < rows; r++)
        for (int c = 0; c < cols; c++) {
          const uint8_t code = packed[size_t(r) * cols / 2 + c / 2] >> (4 * (c % 2)) & 15;
          const float v = (code & 8 ? -kE2m1[code & 7] : kE2m1[code & 7]) * e4m3_value(sf[size_t(r) * cols / 16 + c / 16]);
          out[size_t(r) * cols + c] = __float2bfloat16(v / global);
        }
      Q27_CUDA(cudaMemcpyAsync(dst, out, size_t(rows) * cols * sizeof(bf16), cudaMemcpyHostToDevice, s));
      Q27_CUDA(cudaEventRecord(done[cur], s));
      cur ^= 1;
      continue;
    }
    if (j.q4) {  // BF16 rows -> a device temporary (chunks of 128-row multiples) -> NVFP4 + swizzled scales
      const int rows = int(info.shape[0]), cols = int(info.shape[1]);
      const int chunk = std::max(128, int((32 << 20) / (size_t(cols) * 2)) / 128 * 128);
      const size_t chunk_bytes = size_t(chunk) * cols * 2;
      Q27_CHECK(chunk_bytes <= kStage, j.name + ": rows larger than the staging buffer");
      if (chunk_bytes > q8_tmp_bytes) {
        Q27_CUDA(cudaStreamSynchronize(s));
        if (q8_tmp) Q27_CUDA(cudaFree(q8_tmp));
        q8_tmp_bytes = chunk_bytes;
        Q27_CUDA(cudaMalloc(&q8_tmp, q8_tmp_bytes));
      }
      staging();
      for (int r0 = 0; r0 < rows; r0 += chunk) {
        const int n = std::min(chunk, rows - r0);
        const int64_t len = int64_t(n) * cols * 2;
        Q27_CUDA(cudaEventSynchronize(done[cur]));
        if (j.from8) {  // (the FP8 layer is on the device already: its jobs came first, on the same stream)
          dequant_fp8_rows(j.from8->w + size_t(r0) * cols, j.from8->w_scales + r0, n, cols,
                           static_cast<bf16*>(q8_tmp), s);
        } else {
          ckpt.read_range(j.name, int64_t(r0) * cols * 2, len, stage[cur]);
          Q27_CUDA(cudaMemcpyAsync(q8_tmp, stage[cur], size_t(len), cudaMemcpyHostToDevice, s));
        }
        quant_fp4_weight(static_cast<const bf16*>(q8_tmp), n, cols, j.q4->w_scale2,
                  reinterpret_cast<uint8_t*>(dst) + size_t(r0) * cols / 2,
                  j.pool->at<uint8_t>(j.q8_scale) + sf_bytes(r0, cols / 16), s);
        Q27_CUDA(cudaEventRecord(done[cur], s));
        cur ^= 1;
      }
      continue;
    }
    if (j.q8_rows) {  // BF16 -> a device temporary (in chunks) -> e4m3 + row scales
      if (size_t(info.nbytes) > q8_tmp_bytes) {
        Q27_CUDA(cudaStreamSynchronize(s));
        if (q8_tmp) Q27_CUDA(cudaFree(q8_tmp));
        q8_tmp_bytes = size_t(info.nbytes);
        Q27_CUDA(cudaMalloc(&q8_tmp, q8_tmp_bytes));
      }
      staging();
      for (int64_t off = 0; off < info.nbytes; off += kStage) {
        int64_t len = std::min<int64_t>(kStage, info.nbytes - off);
        Q27_CUDA(cudaEventSynchronize(done[cur]));
        ckpt.read_range(j.name, off, len, stage[cur]);
        Q27_CUDA(cudaMemcpyAsync(static_cast<char*>(q8_tmp) + off, stage[cur], size_t(len), cudaMemcpyHostToDevice, s));
        Q27_CUDA(cudaEventRecord(done[cur], s));
        cur ^= 1;
      }
      vision_quant_fp8(static_cast<const bf16*>(q8_tmp), j.q8_rows, int(info.nbytes / 2 / j.q8_rows),
                       reinterpret_cast<uint8_t*>(dst), j.pool->at<float>(j.q8_scale), s);
      continue;
    }
    if (j.sf_rows) {  // whole scale-factor tensor in one staging buffer, swizzled from host memory
      if (Prefetcher::fits(info.nbytes)) {
        swizzle_sf(static_cast<uint8_t*>(pf.get(ri, j.name, 0)), j.sf_rows, j.sf_kblocks, reinterpret_cast<uint8_t*>(dst), s, j.glu - 1);
        pf.release(ri++, s);
        continue;
      }
      Q27_CHECK(size_t(info.nbytes) <= kStage, j.name + ": scale factors larger than the staging buffer");
      staging();
      Q27_CUDA(cudaEventSynchronize(done[cur]));
      ckpt.read(j.name, stage[cur]);
      swizzle_sf(static_cast<uint8_t*>(stage[cur]), j.sf_rows, j.sf_kblocks, reinterpret_cast<uint8_t*>(dst), s, j.glu - 1);
      Q27_CUDA(cudaEventRecord(done[cur], s));
      cur ^= 1;
      continue;
    }
    for (int64_t off = 0; off < info.nbytes; off += kChunk) {
      void* buf = pf.get(ri, j.name, off);
      if (!buf) {  // embedding: read straight into the pinned table
        ri++;
        continue;
      }
      const size_t len = size_t(std::min<int64_t>(kChunk, info.nbytes - off));
      if (j.glu)
        copy_glu(dst, size_t(off), buf, len, size_t(info.shape[1]) * 64, j.glu - 1, s);
      else
        Q27_CUDA(cudaMemcpyAsync(dst + off, buf, len, cudaMemcpyHostToDevice, s));
      pf.release(ri++, s);
    }
  }
  Q27_CUDA(cudaStreamSynchronize(s));
  if (q8_tmp) Q27_CUDA(cudaFree(q8_tmp));
  for (int i = 0; i < 2; i++) {
    cudaEventDestroy(done[i]);
    if (stage[i]) cudaFreeHost(stage[i]);
  }
  double sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  log("loaded %.2f GB of weights in %.1f s (%.2f GB/s)", total / 1e9, sec, total / 1e9 / sec);
}

}  // namespace q27
