#include "weights.h"

#include <chrono>
#include <cmath>
#include <cstring>

#include "common.h"
#include "prefetch.h"

namespace q27 {
namespace {

const std::string kLM = "model.language_model";

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

void Weights::fp8(const SafeTensors& ckpt, const std::string& p, Pool& dev, LinearFp8& l) {
  const auto& info = ckpt.info(p + ".weight");
  Q27_CHECK(info.dtype == "F8_E4M3", p + ": expected an FP8 weight");
  l.out = int(info.shape[0]);
  l.in = int(info.shape[1]);
  tensor(ckpt, p + ".weight", dev, reinterpret_cast<const void**>(&l.w));
  scalars_.push_back({&ckpt, p + ".weight_scale", &l.w_scale});
  scalars_.push_back({&ckpt, p + ".input_scale", &l.in_scale});
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
    scalars_.push_back({&ckpt, p + (ct ? ".weight_global_scale" : ".weight_scale_2"), &l->w_scale2, ct});
    scalars_.push_back({&ckpt, p + (ct ? ".input_global_scale" : ".input_scale"), &l->in_scale, ct});
  }
  for (auto& [p, l] : group)
    tensor(ckpt, p + ".weight_scale", dev, reinterpret_cast<const void**>(&l->sf), l->out, l->in / 16);
}

void Weights::proj(const SafeTensors& ckpt, const std::vector<std::pair<std::string, LinearQ*>>& group, Pool& dev) {
  const std::string& p0 = group.front().first;
  if (ckpt.has(p0 + ".weight") && ckpt.info(p0 + ".weight").dtype == "F8_E4M3") {
    for (auto& [p, l] : group) fp8(ckpt, p, dev, l->f8);
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
  const size_t off = dev.add(p + ".weight", size_t(info.shape[0]) * info.shape[1] * 2 * sizeof(bf16));
  jobs_.push_back({&ckpt, p + ".weight_packed", &dev, off, 0, 0, p});
  ptrs_.push_back({reinterpret_cast<const void**>(dst), {&dev, off}});
}

namespace {

// Concatenation along the output dimension of layers stored back to back.
LinearFp8 fuse8(std::initializer_list<const LinearFp8*> parts) {
  LinearFp8 f = **parts.begin();
  f.out = 0;
  f.alpha.count = 0;
  for (const LinearFp8* l : parts) {
    Q27_CHECK(l->w == f.w + size_t(f.out) * f.in && l->in == f.in, "fused FP8 layers are not adjacent");
    Q27_CHECK(l->in_scale == f.in_scale, "fused FP8 layers have different input scales");
    Q27_CHECK(f.alpha.count < AlphaSegments::kMax, "too many fused layers");
    f.alpha.begin[f.alpha.count] = f.out;
    f.alpha.alpha[f.alpha.count++] = l->in_scale * l->w_scale;
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
  f.fp4 = a.fp4;
  if (f.fp4) f.f4 = fuse4({&a.f4, &b.f4}); else f.f8 = fuse8({&a.f8, &b.f8});
  return f;
}
LinearQ fuseq(const LinearQ& a, const LinearQ& b, const LinearQ& c) {
  LinearQ f;
  f.fp4 = a.fp4;
  if (f.fp4) f.f4 = fuse4({&a.f4, &b.f4, &c.f4}); else f.f8 = fuse8({&a.f8, &b.f8, &c.f8});
  return f;
}

}  // namespace

void Weights::fuse() {
  auto one8 = [](LinearFp8& l) { l.alpha = AlphaSegments::one(l.in_scale * l.w_scale); };
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
    for (auto* l : {&L.gate, &L.up, &L.down}) one4(*l);
    L.gate_up = fuse4({&L.gate, &L.up});
  }
  one4(lm_head);
}

void Weights::plan(const SafeTensors& ckpt, const SafeTensors* lm_ckpt, Pool& dev, Pool& host) {
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
    fp4(ckpt, {{p + ".mlp.gate_proj", &L.gate}, {p + ".mlp.up_proj", &L.up}}, dev);
    fp4(ckpt, {{p + ".mlp.down_proj", &L.down}}, dev);
  }
  bf(kLM + ".norm.weight", &final_norm);
  // lm_head: NVFP4. A BF16 one (QUASAR keeps the original) comes from lm_ckpt (the nvidia checkpoint
  // quantized the same original lm_head).
  const bool lm_bf16 = ckpt.info("lm_head.weight").dtype == "BF16";
  Q27_CHECK(!lm_bf16 || lm_ckpt, "lm_head is BF16: set Q27_LM_HEAD_CKPT to a checkpoint with an NVFP4 lm_head");
  fp4(lm_bf16 ? *lm_ckpt : ckpt, {{"lm_head", &lm_head}}, dev);
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
    const auto& info = sj.src->info(sj.name);
    Q27_CHECK(info.dtype == "F32" && info.nbytes == 4, sj.name + ": expected an f32 scalar");
    plan.push_back({sj.src, sj.name, 0, 4, sj.dst});
  }
  for (auto& j : jobs_) {
    const auto& info = j.src->info(j.name);
    if (!j.dq.empty() || j.q8_rows || (j.sf_rows && !Prefetcher::fits(info.nbytes))) continue;
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
    if (sj.inv) *sj.dst = 1.f / *sj.dst;
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
    total += size_t(info.nbytes);
    char* dst = j.pool->at<char>(j.offset);
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
        swizzle_sf(static_cast<uint8_t*>(pf.get(ri, j.name, 0)), j.sf_rows, j.sf_kblocks, reinterpret_cast<uint8_t*>(dst), s);
        pf.release(ri++, s);
        continue;
      }
      Q27_CHECK(size_t(info.nbytes) <= kStage, j.name + ": scale factors larger than the staging buffer");
      staging();
      Q27_CUDA(cudaEventSynchronize(done[cur]));
      ckpt.read(j.name, stage[cur]);
      swizzle_sf(static_cast<uint8_t*>(stage[cur]), j.sf_rows, j.sf_kblocks, reinterpret_cast<uint8_t*>(dst), s);
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
      Q27_CUDA(cudaMemcpyAsync(dst + off, buf, size_t(std::min<int64_t>(kChunk, info.nbytes - off)), cudaMemcpyHostToDevice, s));
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
