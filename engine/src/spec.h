// Speculative decoding: what the draft hands to the accept step, the device-side random numbers and
// the accept / advance kernels (spec.cu).
#pragma once

#include <cstdint>

#include <cuda_runtime.h>

#include "kernels.h"  // (kMaxLanes)

namespace q27 {

constexpr int kSpecBlock = 8;  // the draft block: the anchor and 7 drafted tokens (the deepest path)
constexpr int kSpecSelK = 16;  // selector candidates per position
constexpr int kTreeMax = 32;   // verify rows: nodes of the draft tree (node 0 is the anchor)
constexpr int kTreePaths = 4;  // root-to-leaf paths (draft walks)

// The draft tree of a step (written by the selector, read by the verify, accept and commit kernels).
// K walks through the selector's lattice are merged into a trie: node i has token tok[i] at depth
// depth[i] (the anchor: 0), and path[i][0..depth[i]] are the nodes from the root to it. The verify
// computes every node as if its path were decoded alone. A chain is the tree of one walk.
struct SpecTree {
  int32_t n;                              // nodes (rows used of the kTreeMax verify rows)
  int32_t npaths;                         // walks (root-to-leaf paths; duplicates merged)
  int32_t tok[kTreeMax];
  int8_t depth[kTreeMax];
  int8_t path[kTreeMax][kSpecBlock];      // nodes from the root: path[i][depth[i]] == i
  int8_t owner[kTreeMax];                 // the path that writes the node's GDN result (-1: none)
  int8_t leaf[kTreePaths];                // last node of each path
  int8_t cidx[kTreeMax];                  // index of tok[i] among the candidates at its depth (root: 0)
  int8_t ndraws[kTreeMax];                // draws made from the node (one per walk through it)
  int8_t draws[kTreeMax][kTreePaths];     // their child nodes, in draw order (duplicates repeat)
  float q[kTreeMax][kSpecSelK];           // the distribution the node's draws came from
};

// Sampling of the request (the same distribution as the non-speculative sampler).
struct SpecSampling {
  float temperature = 0.f;  // <= 0: greedy
  int top_k = 20;
  float top_p = 1.f;
  int paths = 1;            // draft walks merged into the tree (1: a chain)
  int32_t stop[2] = {-1, -1};  // end-of-sequence ids: an accepted one ends the step as its last token (not fed)
};

// The selector's choice, consumed by the accept kernel.
struct DraftChoice {
  int32_t* cand = nullptr;  // [7][16] candidate ids per position
  float* q = nullptr;       // [7][16] proposal distribution over the candidates (one-hot when greedy)
  int32_t* path = nullptr;  // [7] chosen candidate index per position
  int32_t* src = nullptr;   // 1: the chain came from the lookup (point-mass proposals), 0: the draft
  SpecTree* tree = nullptr;
};

// Lookup drafting (docs/decisions.md "Speculative decoding"): the continuation of the latest earlier occurrence of the context's
// last tokens. spec_lookup finds the longest match (ties: the latest) of the tokens ending at the
// anchor; the selector then takes the looked-up chain instead of the draft's when the match has at
// least min_len tokens, or at least 3 and the draft's greedy path proposes the same first `agree`
// tokens. The rule depends only on the context (not on the sampled draft), so the proposal stays a
// fixed distribution and the verify lossless.
struct Lookup {
  const int32_t* hist = nullptr;  // committed tokens by position
  const int32_t* pos = nullptr;   // committed length
  uint32_t* best = nullptr;       // (match length << 20) | end position; reset by the selector
  int min_len = 8;
  int agree = 2;
  int on = 0;
};
constexpr int kLookupMax = 16;  // longest match measured

// Result of one step in mapped pinned memory: the tokens the host has not seen yet (accepted drafts,
// then the bonus token).
struct SpecOut {
  int32_t n;
  int32_t tok[kSpecBlock];
  int32_t src;  // DraftChoice::src of the step
};

// Counter-based uniform numbers in [0, 1): rng[0] = seed, rng[1] = step counter (advanced on the device).
// Index ranges per step: walk k, position e: 8 k + e; accept test of draw j at node n: 32 + 4 n + j;
// residual / bonus draw at node n: 160 + n.
constexpr uint32_t kRngWalk = 0, kRngAccept = 32, kRngResidual = 160;
__device__ __forceinline__ float rng_uniform(const uint64_t* rng, uint32_t idx) {
  uint64_t z = rng[0] + (rng[1] * 256 + idx + 1) * 0x9E3779B97F4A7C15ull;
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
  z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
  z ^= z >> 31;
  return float(z >> 40) * (1.f / 16777216.f);
}

// Sorted top-k (k <= 64) of each row (T <= 8): values / ids [T][k], descending, ties by lower index.
// idmap (optional): the ids written are idmap[column] (a restricted vocabulary).
void topk_rows(const float* logits, int T, int V, int k, float* vals, int32_t* ids, cudaStream_t s,
               const int32_t* idmap = nullptr);
// Accept step over the verified draft tree (choice.tree; rows = nodes) with the target's sorted top-k
// lists per row (k = 1 when greedy): greedy follows the target's argmax down the tree; sampling runs
// recursive rejection sampling over each node's draws (i.i.d. from the node's q: lossless). The accepted
// tokens and the bonus token go to out, the accepted path's rows (root first) to cpath and their
// count to *ncommit, the bonus becomes the next anchor (ids[0], block_ids[0]), and the committed
// tokens are appended to hist at *pos (lookup drafting). par (optional): the result goes to out[*par].
// budget (optional): at most max(1, *budget) tokens are output (an accepted end-of-sequence id, see
// SpecSampling::stop, also ends the step).
void spec_accept(int32_t* ids, int32_t* block_ids, const float* tv, const int32_t* ti, int k, DraftChoice choice,
                 SpecSampling smp, const uint64_t* rng, int32_t* ncommit, int32_t* cpath, SpecOut* out,
                 int32_t* hist, const int32_t* pos, cudaStream_t s, const int32_t* par = nullptr,
                 const int32_t* budget = nullptr);
// Batched steps (P6-10): the sampling of each lane and the length of its top-k lists (1 when greedy).
struct AcceptLanes {
  SpecSampling smp[kMaxLanes] = {};
  int k[kMaxLanes] = {};
};
// spec_accept for n lanes (block z: lane z). Lane z's verify rows are 8 z .. 8 z + 7 (ids, block_ids,
// cpath, and the top-k lists tv / ti, ld entries per row), its candidates rows 8 z + 1 .. of cand
// ([rows][16]), its tree trees[z], its scalars src / ncommit / pos / par / budget [z], rng [z][2], and
// its results out[2 z + par[z]]. No token history (lookup drafting is off in batched steps).
void spec_accept_lanes(int n, int32_t* ids, int32_t* block_ids, const float* tv, const int32_t* ti, int ld,
                       const AcceptLanes& al, const int32_t* cand, SpecTree* trees, const int32_t* src,
                       const uint64_t* rng, int32_t* ncommit, int32_t* cpath, SpecOut* out, const int32_t* pos,
                       const int32_t* par, const int32_t* budget, cudaStream_t s);
// spec_advance of n lanes (no GDN state buffer: batched verifies replay in place).
void spec_advance_lanes(int n, int32_t* pos, const int32_t* ncommit, int32_t* par, uint64_t* rng, int32_t* budget,
                        cudaStream_t s);
// The lookup match for the next draft block (see Lookup); `anchor` is the uncommitted last token.
void spec_lookup(const Lookup& lk, const int32_t* anchor, cudaStream_t s);
// End of a step: *pos += *ncommit, *par ^= 1, *sbuf ^= 1 (the GDN state buffer), rng[1] += 1, and
// *budget -= *ncommit (if given).
void spec_advance(int32_t* pos, const int32_t* ncommit, int32_t* par, int32_t* sbuf, uint64_t* rng, cudaStream_t s,
                  int32_t* budget = nullptr);

}  // namespace q27
