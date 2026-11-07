// Top-k selection over the vocabulary, log-softmax (golden recordings) and sampling. One block per
// row; the top k are found in k rounds of a block-wide argmax over the elements that come after
// the previous pick in (value desc, index asc) order, so no membership test is needed.
#include "common.h"
#include "kernels.h"
#include "launch.cuh"
#include "spec.h"

namespace q27 {
namespace {

constexpr int kThreads = 1024;
constexpr int kMaxK = 64;

struct Pick {
  float v;
  int i;
};

__device__ __forceinline__ bool better(Pick a, Pick b) { return a.v > b.v || (a.v == b.v && a.i < b.i); }

__device__ Pick block_best(Pick p) {
  __shared__ Pick red[kThreads / 32];
  for (int o = 16; o > 0; o >>= 1) {
    Pick q{__shfl_xor_sync(0xffffffffu, p.v, o), __shfl_xor_sync(0xffffffffu, p.i, o)};
    if (better(q, p)) p = q;
  }
  int w = threadIdx.x / 32, lane = threadIdx.x % 32;
  __syncthreads();
  if (lane == 0) red[w] = p;
  __syncthreads();
  p = lane < blockDim.x / 32 ? red[lane] : Pick{-INFINITY, 0x7fffffff};
  for (int o = 16; o > 0; o >>= 1) {
    Pick q{__shfl_xor_sync(0xffffffffu, p.v, o), __shfl_xor_sync(0xffffffffu, p.i, o)};
    if (better(q, p)) p = q;
  }
  return p;
}

__device__ float block_sum(float v) {
  __shared__ float red[kThreads / 32];
  for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
  int w = threadIdx.x / 32, lane = threadIdx.x % 32;
  __syncthreads();
  if (lane == 0) red[w] = v;
  __syncthreads();
  v = lane < blockDim.x / 32 ? red[lane] : 0.f;
  for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
  return v;
}

// Writes the k best (value, index) of row x into sv / si (shared), descending.
__device__ void block_topk(const float* x, int V, int k, float* sv, int* si) {
  Pick last{INFINITY, -1};
  for (int r = 0; r < k; r++) {
    Pick best{-INFINITY, 0x7fffffff};
    for (int i = threadIdx.x; i < V; i += blockDim.x) {
      Pick c{x[i], i};
      if (better(last, c) && better(c, best)) best = c;
    }
    best = block_best(best);
    if (threadIdx.x == 0) sv[r] = best.v, si[r] = best.i;
    last = best;
  }
  __syncthreads();
}

__global__ void __launch_bounds__(kThreads) logsoftmax_topk_kernel(const float* logits, int V, int k, int32_t* ids,
                                                                   float* logp) {
  const float* x = logits + int64_t(blockIdx.x) * V;
  __shared__ float sv[kMaxK];
  __shared__ int si[kMaxK];
  block_topk(x, V, k, sv, si);
  float mx = sv[0], s = 0.f;
  for (int i = threadIdx.x; i < V; i += blockDim.x) s += expf(x[i] - mx);
  float lse = mx + logf(block_sum(s));
  if (threadIdx.x < k) {
    ids[int64_t(blockIdx.x) * k + threadIdx.x] = si[threadIdx.x];
    logp[int64_t(blockIdx.x) * k + threadIdx.x] = sv[threadIdx.x] - lse;
  }
}

// Sampling (decode), two stages:
//   topk_segments_kernel: one warp per 2048-logit segment keeps its 64 values in registers and finds
//     the segment's top k (k rounds of a warp argmax; descending, ties by lower index).
//   merge_sample_kernel: one warp merges the 122 sorted lists (each lane owns a few list heads) into
//     the global top k, then samples. Same picks as k rounds over the whole row in one block, which
//     took ~80 us for greedy and k times that for top-k.
constexpr int kSeg = 2048, kPerLane = kSeg / 32, kMaxLists = 8;  // up to 8 * 32 segments (V <= 512K)

__device__ __forceinline__ Pick warp_best(Pick p) {
  for (int o = 16; o > 0; o >>= 1) {
    Pick q{__shfl_xor_sync(0xffffffffu, p.v, o), __shfl_xor_sync(0xffffffffu, p.i, o)};
    if (better(q, p)) p = q;
  }
  return p;
}

// The lane's best remaining value (ties: the lower j, i.e. the lower index) by a tournament over its
// kPerLane values: depth 6 instead of a 64-long dependent chain.
__device__ __forceinline__ void lane_best(const float (&v)[kPerLane], float& bv, int& bj) {
  float tv[kPerLane / 2];
  int tj[kPerLane / 2];
#pragma unroll
  for (int a = 0; a < kPerLane / 2; a++) {
    const bool hi = v[2 * a + 1] > v[2 * a];
    tv[a] = hi ? v[2 * a + 1] : v[2 * a], tj[a] = hi ? 2 * a + 1 : 2 * a;
  }
#pragma unroll
  for (int w = kPerLane / 4; w >= 1; w /= 2) {
#pragma unroll
    for (int a = 0; a < w; a++) {  // (element a covers lower j than element a + w)
      const bool hi = tv[a + w] > tv[a];
      tv[a] = hi ? tv[a + w] : tv[a], tj[a] = hi ? tj[a + w] : tj[a];
    }
  }
  bv = tv[0], bj = tj[0];
}

__global__ void __launch_bounds__(32) topk_segments_kernel(const float* logits, int V, int k, Pick* cand) {
  pdl_trigger();
  pdl_wait();
  const int seg = blockIdx.x, row = blockIdx.y, lane = threadIdx.x, nseg = gridDim.x;
  const float* x = logits + int64_t(row) * V;
  float v[kPerLane];
#pragma unroll
  for (int j = 0; j < kPerLane; j++) {
    int i = seg * kSeg + j * 32 + lane;
    v[j] = i < V ? x[i] : -INFINITY;
  }
  // Each round: the warp's best of the lanes' best; the winning lane drops that value (-inf) and
  // recomputes its best. Same picks and order as k full scans with ties by the lower index.
  float bv;
  int bj;
  lane_best(v, bv, bj);
  for (int r = 0; r < k; r++) {
    const Pick mine{bv, seg * kSeg + bj * 32 + lane};
    const Pick best = warp_best(mine);
    if (lane == 0) cand[(int64_t(row) * nseg + seg) * k + r] = best;
    if (best.i == mine.i) {
#pragma unroll
      for (int j = 0; j < kPerLane; j++)
        if (j == bj) v[j] = -INFINITY;
      lane_best(v, bv, bj);
    }
  }
}

// Warp merge of the per-segment sorted lists of one row into the global top k (sv / si, shared).
// The first kCopy entries of every list are copied to shared memory up front (independent loads), so
// a round's list advance is a shared-memory read instead of a dependent global load (~1 us each
// while the memory queues are busy; k = 16-20 rounds made the merge 20-40 us).
constexpr int kCopy = 16;
__device__ void merge_lists(const Pick* base, int nseg, int k, float* sv, int* si) {
  const int lane = threadIdx.x % 32;
  const Pick none{-INFINITY, 0x7fffffff};
  __shared__ Pick sl[32 * kMaxLists * kCopy];
  const int P = min(k, kCopy);
#pragma unroll 8
  for (int idx = lane; idx < nseg * P; idx += 32) sl[idx] = base[int64_t(idx / P) * k + idx % P];
  __syncwarp();
  int pos[kMaxLists];
  Pick head[kMaxLists];
#pragma unroll
  for (int m = 0; m < kMaxLists; m++) {
    int l = lane + 32 * m;
    pos[m] = 0;
    head[m] = l < nseg ? sl[l * P] : none;
  }
  for (int r = 0; r < k; r++) {
    Pick mine = none;
    int mi = -1;
#pragma unroll
    for (int m = 0; m < kMaxLists; m++)
      if (better(head[m], mine)) mine = head[m], mi = m;
    Pick best = warp_best(mine);
    if (lane == 0) sv[r] = best.v, si[r] = best.i;
    if (mi >= 0 && mine.i == best.i && mine.v == best.v) {  // indices are unique: exactly one lane advances
#pragma unroll
      for (int m = 0; m < kMaxLists; m++)
        if (m == mi) {
          int l = lane + 32 * m, e = ++pos[m];
          head[m] = e >= k ? none : e < P ? sl[l * P + e] : base[int64_t(l) * k + e];
        }
    }
  }
  __syncwarp();
}

// Penalties (Penalty) on a sorted list v / id of L <= 64 entries in shared memory (warp-wide), sorted
// again (value desc, then the old order). extra[0..nextra): more tokens to count (a verify row's path).
__device__ void penalize_list(float* v, int* id, int L, const Penalty& pen, const int32_t* extra, int nextra) {
  const int lane = threadIdx.x % 32;
  float nv[2];
  int ni[2], rank[2];
  for (int e = 0; e < 2; e++) {
    const int i = lane + 32 * e;
    if (i >= L) continue;
    ni[e] = id[i];
    uint32_t c = pen.counts[ni[e]];
    for (int j = 0; j < nextra; j++) c += extra[j] == ni[e];
    nv[e] = c ? v[i] - pen.presence - pen.frequency * float(c) : v[i];
  }
  __syncwarp();
  for (int e = 0; e < 2; e++)
    if (lane + 32 * e < L) v[lane + 32 * e] = nv[e];
  __syncwarp();
  for (int e = 0; e < 2; e++) {
    const int i = lane + 32 * e;
    if (i >= L) continue;
    int r = 0;
    for (int j = 0; j < L; j++) r += v[j] > nv[e] || (v[j] == nv[e] && j < i);
    rank[e] = r;
  }
  __syncwarp();
  for (int e = 0; e < 2; e++)
    if (lane + 32 * e < L) v[rank[e]] = nv[e], id[rank[e]] = ni[e];
  __syncwarp();
}

// The sampler's distribution over a sorted top-k list: p[i] (unnormalized, i < n after the min_p and
// top-p cuts) and their mass. min_p keeps p >= min_p * p_max, then top-p the smallest prefix whose
// share of what is left reaches top_p.
__device__ int target_dist(const float* v, int k, float temperature, float top_p, float min_p, float* p, float& mass) {
  float total = 0.f;
  for (int i = 0; i < k; i++) total += p[i] = expf((v[i] - v[0]) / temperature);
  if (min_p > 0.f) {  // (p[0] = 1)
    int m = 1;
    while (m < k && p[m] >= min_p) m++;
    if (m < k) {
      k = m, total = 0.f;
      for (int i = 0; i < k; i++) total += p[i];
    }
  }
  int n = k;
  float cum = 0.f;
  for (int i = 0; i < k; i++) {
    cum += p[i] / total;
    if (cum >= top_p) {
      n = i + 1;
      break;
    }
  }
  mass = 0.f;
  for (int i = 0; i < n; i++) mass += p[i];
  return n;
}

// Picks from p[0..n) (unnormalized, total mass) with u in [0, 1).
__device__ int pick(const float* p, int n, float mass, float u, const int32_t* id) {
  float c = 0.f, x = u * mass;
  for (int i = 0; i < n; i++) {
    c += p[i];
    if (x < c) return id[i];
  }
  return id[n - 1];
}

// L: the merged list's length (k, or 64 with penalties); k: the sampler's top-k.
__global__ void __launch_bounds__(32) merge_sample_kernel(const Pick* cand, int nseg, int L, int k, float temperature,
                                                          float top_p, float min_p, Penalty pen, const float* uniform,
                                                          int32_t* out, LogprobSink lp) {
  pdl_trigger();
  pdl_wait();
  const int row = blockIdx.x, lane = threadIdx.x;
  __shared__ float sv[kMaxK];
  __shared__ int si[kMaxK];
  merge_lists(cand + int64_t(row) * nseg * L, nseg, L, sv, si);
  if (lane < lp.n) lp.tv[lane] = sv[lane], lp.ti[lane] = si[lane];  // (the raw list, before the penalties)
  if (pen.counts) penalize_list(sv, si, L, pen, nullptr, 0);
  if (lane != 0) return;
  int tok = si[0];
  if (k > 1) {
    float p[kMaxK], mass;
    const int n = target_dist(sv, k, temperature, top_p, min_p, p, mass);
    tok = pick(p, n, mass, uniform[row], si);
  }
  out[row] = tok;
  if (pen.counts) atomicAdd(pen.counts + tok, 1u);
}

__global__ void __launch_bounds__(32) merge_lists_kernel(const Pick* cand, int nseg, int k, float* vals,
                                                         int32_t* ids, const int32_t* idmap) {
  pdl_trigger();
  pdl_wait();
  const int row = blockIdx.x, lane = threadIdx.x;
  __shared__ float sv[kMaxK];
  __shared__ int si[kMaxK];
  merge_lists(cand + int64_t(row) * nseg * k, nseg, k, sv, si);
  for (int i = lane; i < k; i += 32) vals[row * k + i] = sv[i], ids[row * k + i] = idmap ? idmap[si[i]] : si[i];
}

// Down the draft tree (single thread; k <= 64). Greedy: follow the child whose token is the target's
// argmax. Sampling: at node n the target distribution p (the sampler's, over the row's top-k list) is
// tested against n's draws in draw order; a draw x is accepted with probability min(1, p(x) / q(x)),
// else p <- max(p - q, 0) renormalized; when every draw is rejected (or n is a leaf) the bonus token is
// drawn from p. The draws are i.i.d. from q, so the output follows the target distribution.
// The body of the accept step (thread 0). The top-k lists of the rows are ld entries apart (k <= ld used).
__device__ __forceinline__ void spec_accept_body(int32_t* ids, int32_t* block_ids, const float* tv, const int32_t* ti,
                                                 int k, int ld, const DraftChoice& ch, const SpecSampling& smp,
                                                 const uint64_t* rng, int32_t* ncommit, int32_t* cpath, SpecOut* out,
                                                 int32_t* hist, const int32_t* pos, const int32_t* par,
                                                 const int32_t* budget) {
  if (par) out += *par;  // the step's slot (the host may read the previous step's while this one runs)
  const SpecTree* tr = ch.tree;
  int acc[kSpecBlock], a = 0, n = 0, bonus = 0;
  acc[0] = 0;
  // An accepted end-of-sequence token ends the step like a bonus token: the sequence then holds exactly
  // the tokens before it (a prefix cache checkpoint at the end of the generation matches the next turn).
  // So does reaching the token budget (the caller's max_tokens): the sequence never holds tokens past it.
  auto stop = [&](int t) { return t == smp.stop[0] || t == smp.stop[1] || t == smp.stop[2]; };
  const int limit = budget ? max(1, *budget) : kSpecBlock;  // tokens this step may output
  if (smp.temperature <= 0.f) {
    for (;;) {
      const int t = ti[n * ld];
      int next = -1;
      for (int j = 0; j < tr->ndraws[n]; j++)
        if (tr->tok[tr->draws[n][j]] == t) next = tr->draws[n][j];
      if (next < 0 || stop(t) || a + 2 > limit) {
        bonus = t;
        break;
      }
      acc[++a] = n = next;
    }
  } else {
    float p[kMaxK];
    for (;;) {
      const float* v = tv + n * ld;
      const int32_t* id = ti + n * ld;
      float mass;
      const int m = target_dist(v, k, smp.temperature, smp.top_p, smp.min_p, p, mass);
      for (int i = 0; i < m; i++) p[i] /= mass;
      mass = 1.f;
      const int d = tr->depth[n];
      int next = -1;
      for (int j = 0; j < tr->ndraws[n] && next < 0; j++) {
        const int c = tr->draws[n][j], x = tr->tok[c];
        const float qx = tr->q[n][tr->cidx[c]];
        float px = 0.f;
        for (int i = 0; i < m; i++)
          if (id[i] == x) px = p[i] / mass;
        if (rng_uniform(rng, kRngAccept + 4 * n + j) * qx < px) {
          next = c;
          break;
        }
        float tot = 0.f;  // rejected: the residual max(p - q, 0)
        for (int i = 0; i < m; i++) {
          float qi = 0.f;
          for (int cc = 0; cc < kSpecSelK; cc++)
            if (ch.cand[d * kSpecSelK + cc] == id[i]) qi = tr->q[n][cc];
          p[i] = fmaxf(p[i] / mass - qi, 0.f);
          tot += p[i];
        }
        if (!(tot > 0.f)) {  // numerically empty residual: fall back to p
          target_dist(v, k, smp.temperature, smp.top_p, smp.min_p, p, tot);
        }
        mass = tot;
      }
      if (next < 0) {
        bonus = pick(p, m, mass, rng_uniform(rng, kRngResidual + n), id);
        break;
      }
      if (stop(tr->tok[next]) || a + 2 > limit) {
        bonus = tr->tok[next];
        break;
      }
      acc[++a] = n = next;
    }
  }
  out->n = a + 1;
  for (int i = 1; i <= a; i++) out->tok[i - 1] = tr->tok[acc[i]];
  out->tok[a] = bonus;
  out->src = *ch.src;
  if (smp.pen.counts)
    for (int i = 0; i <= a; i++) atomicAdd(smp.pen.counts + out->tok[i], 1u);
  const int L = *pos;
  for (int i = 0; i <= a; i++) {
    cpath[i] = acc[i];
    if (hist) hist[L + i] = tr->tok[acc[i]];
  }
  *ncommit = a + 1;
  ids[0] = bonus;
  block_ids[0] = bonus;
}

__global__ void spec_accept_kernel(int32_t* ids, int32_t* block_ids, const float* tv, const int32_t* ti, int k,
                                   int ld, DraftChoice ch, SpecSampling smp, const uint64_t* rng, int32_t* ncommit,
                                   int32_t* cpath, SpecOut* out, int32_t* hist, const int32_t* pos,
                                   const int32_t* par, const int32_t* budget) {
  pdl_trigger();
  pdl_wait();
  if (threadIdx.x != 0) return;
  spec_accept_body(ids, block_ids, tv, ti, k, ld, ch, smp, rng, ncommit, cpath, out, hist, pos, par, budget);
}

// Batched steps (P6-10): block z accepts lane z (its rows 8 z .., top-k lists of ld entries, of which the
// lane's sampling uses k[z]; candidates at cand + (8 z + 1) * 16; scalars at z offsets; SpecOut [n][2]).
__global__ void spec_accept_lanes_kernel(int32_t* ids, int32_t* block_ids, const float* tv, const int32_t* ti, int ld,
                                         AcceptLanes al, const int32_t* cand, SpecTree* trees, const int32_t* src,
                                         const uint64_t* rng, int32_t* ncommit, int32_t* cpath, SpecOut* out,
                                         const int32_t* pos, const int32_t* par, const int32_t* budget) {
  pdl_trigger();
  pdl_wait();
  if (threadIdx.x != 0) return;
  const int z = blockIdx.x, r0 = kSpecBlock * z;
  DraftChoice ch;
  ch.cand = const_cast<int32_t*>(cand) + (r0 + 1) * kSpecSelK;
  ch.tree = trees + z;
  ch.src = const_cast<int32_t*>(src) + z;
  spec_accept_body(ids + r0, block_ids + r0, tv + r0 * ld, ti + r0 * ld, lane_sel(al.k, z), ld, ch,
                   lane_sel(al.smp, z), rng + 2 * z, ncommit + z, cpath + r0, out + 2 * z, nullptr, pos + z, par + z,
                   budget + z);
}

__global__ void spec_advance_lanes_kernel(int n, int32_t* pos, const int32_t* ncommit, int32_t* par, uint64_t* rng,
                                          int32_t* budget) {
  pdl_trigger();
  pdl_wait();
  const int z = threadIdx.x;
  if (z >= n) return;
  pos[z] += ncommit[z];
  budget[z] -= ncommit[z];
  par[z] ^= 1;
  rng[2 * z + 1] += 1;
}

// Every position j <= pos - 2 whose token is the anchor: the match length m (tokens ending at j equal
// to those ending at the anchor, up to kLookupMax); the maximum of (m << 20) | j goes to *best.
__global__ void __launch_bounds__(256) lookup_kernel(Lookup lk, const int32_t* anchor) {
  pdl_trigger();
  __shared__ int key[kLookupMax];
  const int tid = threadIdx.x, lane = tid % 32;
  // Everything read here was written by earlier steps (host-synchronized): before the PDL wait.
  const int L = *lk.pos;
  if (tid < kLookupMax) key[tid] = tid == 0 ? *anchor : (L - tid >= 0 ? lk.hist[L - tid] : -1);
  __syncthreads();
  uint32_t mine = 0;
  for (int j = blockIdx.x * blockDim.x + tid; j <= L - 2; j += gridDim.x * blockDim.x) {
    if (lk.hist[j] != key[0]) continue;
    int m = 1;
    while (m < kLookupMax && j - m >= 0 && lk.hist[j - m] == key[m]) m++;
    mine = max(mine, (uint32_t(m) << 20) | uint32_t(j));
  }
  for (int o = 16; o > 0; o >>= 1) mine = max(mine, __shfl_xor_sync(0xffffffffu, mine, o));
  pdl_wait();  // (*best is reset by the previous step's selector)
  if (lane == 0 && mine) atomicMax(lk.best, mine);
}

__global__ void spec_advance_kernel(int32_t* pos, const int32_t* ncommit, int32_t* par, int32_t* sbuf, uint64_t* rng,
                                    int32_t* budget) {
  pdl_trigger();
  pdl_wait();
  *pos += *ncommit;
  if (budget) *budget -= *ncommit;
  *par ^= 1;
  *sbuf ^= 1;
  rng[1] += 1;
}

// Block (r, z): row r of lane z (node r of trees[z]).
__global__ void __launch_bounds__(32) penalize_rows_kernel(float* tv, int32_t* ti, int ld, int rows, PenaltyLanes pl,
                                                           const SpecTree* trees, float* raw_v, int32_t* raw_i) {
  pdl_trigger();
  pdl_wait();
  const int r = blockIdx.x, z = blockIdx.y, lane = threadIdx.x;
  const Penalty& pen = pl.pen[z];
  const SpecTree& tr = trees[z];
  if (r >= tr.n) return;
  float* v = tv + int64_t(z * rows + r) * ld;
  int32_t* id = ti + int64_t(z * rows + r) * ld;
  if (raw_v && lane < kMaxTopLogprobs) {
    const int64_t o = int64_t(z * rows + r) * kMaxTopLogprobs + lane;
    raw_v[o] = v[lane], raw_i[o] = id[lane];
  }
  if (!pen.counts) return;
  __shared__ float sv[kMaxK];
  __shared__ int si[kMaxK];
  __shared__ int32_t path[kSpecBlock];
  for (int i = lane; i < kMaxK; i += 32) sv[i] = v[i], si[i] = id[i];
  const int d = tr.depth[r];
  if (lane < d) path[lane] = tr.tok[tr.path[r][lane + 1]];
  __syncwarp();
  penalize_list(sv, si, kMaxK, pen, path, d);
  for (int i = lane; i < kMaxK; i += 32) v[i] = sv[i], id[i] = si[i];
}

__global__ void count_tokens_kernel(uint32_t* counts, const int32_t* ids, int n, int delta) {
  pdl_trigger();
  pdl_wait();
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x)
    atomicAdd(counts + ids[i], uint32_t(delta));
}

__global__ void count_out_kernel(uint32_t* counts, const SpecOut* so, int delta) {
  pdl_trigger();
  pdl_wait();
  if (int(threadIdx.x) < so->n) atomicAdd(counts + so->tok[threadIdx.x], uint32_t(delta));
}

__global__ void mask_logits_kernel(float* logits, int V, const uint32_t* mask, int words, int rows_per_lane,
                                   uint32_t lanes) {
  pdl_trigger();
  pdl_wait();
  const int row = blockIdx.y;
  if (!(lanes >> (row / rows_per_lane) & 1u)) return;
  const uint32_t* m = mask + int64_t(row) * words;
  float* x = logits + int64_t(row) * V;
  for (int t = blockIdx.x * blockDim.x + threadIdx.x; t < V; t += gridDim.x * blockDim.x)
    if (!(m[t >> 5] >> (t & 31) & 1u)) x[t] = -INFINITY;
}

// Block-wide: the logprobs of token tok in row x (V logits) and the row's top n (tv / ti: its raw sorted list) into
// entry i of out. One pass of an online log-sum-exp.
__device__ void logprob_row(const float* x, int V, int tok, const float* tv, const int32_t* ti, int n, Logprobs* out,
                            int i) {
  float m = -INFINITY, s = 0.f;
  for (int j = threadIdx.x; j < V; j += blockDim.x) {
    const float v = x[j];
    if (v > m) s = s * expf(m - v) + 1.f, m = v;
    else if (v != -INFINITY) s += expf(v - m);  // (masked tokens: -inf - -inf would be NaN)
  }
  const float mx = block_best(Pick{m, int(threadIdx.x)}).v;
  const float lse = mx + logf(block_sum(m == -INFINITY ? 0.f : s * expf(m - mx)));
  if (threadIdx.x == 0) out->lp[i] = x[tok] - lse;
  if (int(threadIdx.x) < n) out->top_id[i][threadIdx.x] = ti[threadIdx.x], out->top_lp[i][threadIdx.x] = tv[threadIdx.x] - lse;
}

// sample() with logprobs: the token of row 0 (*tok) and the raw list merge_sample_kernel kept.
__global__ void __launch_bounds__(kThreads) sample_logprobs_kernel(const float* logits, int V, const int32_t* tok,
                                                                   LogprobSink lp) {
  pdl_trigger();
  pdl_wait();
  if (threadIdx.x == 0) lp.out->n = 1, lp.out->top = lp.n;
  logprob_row(logits, V, *tok, lp.tv, lp.ti, lp.n, lp.out, 0);
}

// Block (i, z): output token i of lane z (see spec_logprobs).
__global__ void __launch_bounds__(kThreads) spec_logprobs_kernel(const float* logits, int V, const float* tv,
                                                                 const int32_t* ti, int ld, int rows, LogprobLanes ln,
                                                                 const SpecTree* trees, const int32_t* cpath,
                                                                 const int32_t* ncommit, const int32_t* ids,
                                                                 Logprobs* out, const int32_t* par) {
  pdl_trigger();
  pdl_wait();
  const int i = blockIdx.x, z = blockIdx.y, n = lane_sel(ln.n, z);
  if (n < 0) return;
  const int nc = ncommit[z];
  Logprobs* o = out + 2 * z + par[z];
  if (i == 0 && threadIdx.x == 0) o->n = nc, o->top = n;
  if (i >= nc) return;
  const int32_t* cp = cpath + kSpecBlock * z;
  const int64_t row = int64_t(z) * rows + cp[i];
  const int tok = i + 1 < nc ? trees[z].tok[cp[i + 1]] : ids[kSpecBlock * z];
  logprob_row(logits + row * V, V, tok, tv + row * ld, ti + row * ld, n, o, i);
}

Pick* g_cand = nullptr;  // candidate lists, kSampleMaxRows rows (allocated on the first, eager call)
constexpr int kSampleMaxRows = std::max(kTreeMax, kMaxBatchRows);  // (a tree, or a batched step)

}  // namespace

void logsoftmax_topk(const float* logits, int T, int V, int k, int32_t* ids, float* logp, cudaStream_t s) {
  Q27_CHECK(k >= 1 && k <= kMaxK, "logsoftmax_topk: k out of range");
  if (T == 0) return;
  logsoftmax_topk_kernel<<<T, kThreads, 0, s>>>(logits, V, k, ids, logp);
  Q27_CUDA(cudaGetLastError());
}

void sample(const float* logits, int T, int V, float temperature, int top_k, float top_p, const float* uniform,
            int32_t* out, cudaStream_t s, float min_p, Penalty pen, LogprobSink lp) {
  if (top_k <= 0 || top_k > kMaxK) top_k = kMaxK;  // candidates are capped at 64
  if (T == 0) return;
  Q27_CHECK(lp.n < 0 || (T == 1 && lp.n <= kMaxTopLogprobs && lp.tv && lp.ti && lp.out), "sample: logprobs out of range");
  const int k = temperature <= 0.f ? 1 : top_k;
  // (penalties pick the top k from the 64 highest logits; the longer list's first k are the same)
  const int L = pen.counts ? kMaxK : std::max(k, lp.n);
  const int nseg = int(ceil_div(V, kSeg));
  Q27_CHECK(nseg <= 32 * kMaxLists && T <= kSampleMaxRows, "sample: vocabulary or rows out of range");
  if (!g_cand) Q27_CUDA(cudaMalloc(&g_cand, sizeof(Pick) * kSampleMaxRows * 32 * kMaxLists * kMaxK));
  launch_pdl(topk_segments_kernel, dim3(nseg, T), dim3(32), 0, s, logits, V, L, g_cand);
  launch_pdl(merge_sample_kernel, dim3(T), dim3(32), 0, s, static_cast<const Pick*>(g_cand), nseg, L, k, temperature,
             top_p, min_p, pen, uniform, out, lp);
  if (lp.n >= 0) launch_pdl(sample_logprobs_kernel, dim3(1), dim3(kThreads), 0, s, logits, V, out, lp);
}

void count_tokens(uint32_t* counts, const int32_t* ids, int n, int delta, cudaStream_t s) {
  if (n <= 0) return;
  launch_pdl(count_tokens_kernel, dim3(std::min(int(ceil_div(n, 256)), 64)), dim3(256), 0, s, counts, ids, n, delta);
  Q27_CUDA(cudaGetLastError());
}

void mask_logits(float* logits, int rows, int V, const uint32_t* mask, int words, int rows_per_lane, uint32_t lanes,
                 cudaStream_t s) {
  Q27_CHECK(words * 32 >= V, "mask_logits: mask rows shorter than the vocabulary");
  launch_pdl(mask_logits_kernel, dim3(64, rows), dim3(256), 0, s, logits, V, mask, words, rows_per_lane, lanes);
  Q27_CUDA(cudaGetLastError());
}

void topk_rows(const float* logits, int T, int V, int k, float* vals, int32_t* ids, cudaStream_t s,
               const int32_t* idmap) {
  const int nseg = int(ceil_div(V, kSeg));
  Q27_CHECK(k >= 1 && k <= kMaxK && nseg <= 32 * kMaxLists && T <= kSampleMaxRows, "topk_rows: out of range");
  if (!g_cand) Q27_CUDA(cudaMalloc(&g_cand, sizeof(Pick) * kSampleMaxRows * 32 * kMaxLists * kMaxK));
  launch_pdl(topk_segments_kernel, dim3(nseg, T), dim3(32), 0, s, logits, V, k, g_cand);
  launch_pdl(merge_lists_kernel, dim3(T), dim3(32), 0, s, static_cast<const Pick*>(g_cand), nseg, k, vals, ids,
             idmap);
  Q27_CUDA(cudaGetLastError());
}

void spec_accept(int32_t* ids, int32_t* block_ids, const float* tv, const int32_t* ti, int k, int ld,
                 DraftChoice choice, SpecSampling smp, const uint64_t* rng, int32_t* ncommit, int32_t* cpath,
                 SpecOut* out, int32_t* hist, const int32_t* pos, cudaStream_t s, const int32_t* par,
                 const int32_t* budget) {
  launch_pdl(spec_accept_kernel, dim3(1), dim3(32), 0, s, ids, block_ids, tv, ti, k, ld, choice, smp, rng, ncommit,
             cpath, out, hist, pos, par, budget);
  Q27_CUDA(cudaGetLastError());
}

void penalize_rows(float* tv, int32_t* ti, int ld, int rows, int n, const PenaltyLanes& pl, const SpecTree* trees,
                   cudaStream_t s, float* raw_v, int32_t* raw_i) {
  Q27_CHECK(ld == kMaxK && rows <= kTreeMax && n >= 1 && n <= kMaxLanes, "penalize_rows: out of range");
  launch_pdl(penalize_rows_kernel, dim3(rows, n), dim3(32), 0, s, tv, ti, ld, rows, pl, trees, raw_v, raw_i);
  Q27_CUDA(cudaGetLastError());
}

void spec_logprobs(int n, const float* logits, int V, const float* tv, const int32_t* ti, int ld, int rows,
                   const LogprobLanes& ln, const SpecTree* trees, const int32_t* cpath, const int32_t* ncommit,
                   const int32_t* ids, Logprobs* out, const int32_t* par, cudaStream_t s) {
  Q27_CHECK(n >= 1 && n <= kMaxLanes, "spec_logprobs: 1..8 lanes");
  for (int z = 0; z < n; z++) Q27_CHECK(ln.n[z] <= std::min(ld, kMaxTopLogprobs), "spec_logprobs: top entries out of range");
  launch_pdl(spec_logprobs_kernel, dim3(kSpecBlock, n), dim3(kThreads), 0, s, logits, V, tv, ti, ld, rows, ln, trees,
             cpath, ncommit, ids, out, par);
  Q27_CUDA(cudaGetLastError());
}

void count_out(uint32_t* counts, const SpecOut* so, int delta, cudaStream_t s) {
  launch_pdl(count_out_kernel, dim3(1), dim3(kSpecBlock), 0, s, counts, so, delta);
  Q27_CUDA(cudaGetLastError());
}

void spec_advance(int32_t* pos, const int32_t* ncommit, int32_t* par, int32_t* sbuf, uint64_t* rng, cudaStream_t s,
                  int32_t* budget) {
  launch_pdl(spec_advance_kernel, dim3(1), dim3(1), 0, s, pos, ncommit, par, sbuf, rng, budget);
  Q27_CUDA(cudaGetLastError());
}

void spec_accept_lanes(int n, int32_t* ids, int32_t* block_ids, const float* tv, const int32_t* ti, int ld,
                       const AcceptLanes& al, const int32_t* cand, SpecTree* trees, const int32_t* src,
                       const uint64_t* rng, int32_t* ncommit, int32_t* cpath, SpecOut* out, const int32_t* pos,
                       const int32_t* par, const int32_t* budget, cudaStream_t s) {
  Q27_CHECK(n >= 1 && n <= kMaxLanes, "spec_accept_lanes: 1..8 lanes");
  launch_pdl(spec_accept_lanes_kernel, dim3(n), dim3(32), 0, s, ids, block_ids, tv, ti, ld, al, cand, trees, src, rng,
             ncommit, cpath, out, pos, par, budget);
  Q27_CUDA(cudaGetLastError());
}

void spec_advance_lanes(int n, int32_t* pos, const int32_t* ncommit, int32_t* par, uint64_t* rng, int32_t* budget,
                        cudaStream_t s) {
  launch_pdl(spec_advance_lanes_kernel, dim3(1), dim3(32), 0, s, n, pos, ncommit, par, rng, budget);
  Q27_CUDA(cudaGetLastError());
}

void spec_lookup(const Lookup& lk, const int32_t* anchor, cudaStream_t s) {
  launch_pdl(lookup_kernel, dim3(64), dim3(256), 0, s, lk, anchor);
  Q27_CUDA(cudaGetLastError());
}

}  // namespace q27
