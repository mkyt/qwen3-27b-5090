"""ctypes binding of libq27.so: the engine (Engine) and the kernel wrappers used by the tests (call)."""

import array
import ctypes
import os

LIB_PATH = os.environ.get("Q27_LIB", "/build/engine/libq27.so")
# Draft vocabulary of the DFlash2 candidates (tools/draft_vocab.py); Q27_DRAFT_VOCAB=none: the full vocabulary.
DRAFT_VOCAB = os.path.join(os.path.dirname(__file__), "../../bench/draft-vocab/vocab-128k.txt")

P, I, I64, F, SZ = ctypes.c_void_p, ctypes.c_int, ctypes.c_int64, ctypes.c_float, ctypes.c_size_t
SIGNATURES = {
    "q27_create": ([ctypes.c_char_p, I, I, I, ctypes.c_char_p], P),
    "q27_spec_begin": ([P, P, I, F, I, F, ctypes.c_uint64, F, I, P], I),
    "q27_spec_step": ([P, P], I),
    "q27_spec_step2": ([P, P, I], I),
    "q27_prefill": ([P, P, I, I, I, P], I),
    "q27_park_slots": ([P], I),
    "q27_seq_park": ([P, P], I),
    "q27_seq_unpark": ([P, I], I),
    "q27_seq_swap": ([P, I], I),
    "q27_seq_drop": ([P, I], I),
    "q27_seq_suspend": ([P, I], I),
    "q27_has_draft": ([P], I),
    "q27_max_lanes": ([P], I),
    "q27_max_tokens": ([P], I),
    "q27_spec_step_lanes": ([P, P, I, P, P], I),
    "q27_spec_stats": ([P, P], I),
    "q27_prefix_begin": ([P, P, I, P, I, I, P], I),
    "q27_checkpoint": ([P], I),
    "q27_cache_stats": ([P, P], I),
    "q27_cache_grow": ([P, I, P], I),
    "q27_memory_report": ([P, ctypes.c_char_p, I], I),
    "q27_device_memory": ([P, P], I),
    "q27_has_vision": ([P], I),
    "q27_mm_set": ([P, P, I], I),
    "q27_mm_encode": ([P, I, P], I),
    "q27_mm_embeds": ([P, I, P], I),
    "q27_mm_put": ([P, I, P], I),
    "q27_debug_kv": ([P, I, I, I, P], I),
    "q27_debug_ssm": ([P, I, P], I),
    "q27t_skinny_bf16": ([I, I, I, P, P, P, P, SZ], I),
    "q27t_gemm_bf16": ([I, I, I, P, P, P], I),
    "q27t_gdn_verify": ([P, I, I, P, P, P, P, P, P] + [P] * 8 + [P, P, P, P, P, P, I, P, F, P, P, F, I], I),
    "q27t_topk_rows": ([P, I, I, I, P, P], I),
    "q27_destroy": ([P], None),
    "q27_reset": ([P], I),
    "q27_pos": ([P], I),
    "q27_forward_topk": ([P, P, I, I, P, P], I),
    "q27_forward_sample": ([P, P, I, F, I, F, F, P], I),
    "q27_profile_dump": ([P], I),
    "q27_configure": ([P], I),
    "q27_last_error": ([], ctypes.c_char_p),
    "q27t_last_us": ([], F),
    "q27t_rmsnorm": ([P, P, I, I, F, I, P], I),
    "q27t_embed_gather": ([P, P, I, I, P], I),
    "q27t_quant_fp4": ([P, I, I, F, P, P], I),
    "q27t_quant_fp8": ([P, I64, F, P], I),
    "q27t_silu_mul": ([P, P, I64, P], I),
    "q27t_linear_bf16": ([P, P, I, I, I, P], I),
    "q27t_swizzle_sf": ([P, I, I, P], I),
    "q27t_sf_bytes": ([I64, I64], I64),
    "q27t_gemm_fp4": ([I, I, I, P, P, P, P, F, P, F, P, P, SZ], I),
    "q27t_gemm_fp8": ([I, I, I, P, P, F, P, F, P, P, SZ], I),
    "q27t_gemm_workspace_bytes": ([], SZ),
    "q27t_gemm_workspace_init": ([P, SZ], I),
    "q27t_skinny_fp4": ([I, I, I, P, P, P, P, F, P, F, P, P, SZ], I),
    "q27t_skinny_fp8": ([I, I, I, P, P, F, P, F, P, P, SZ], I),
    "q27t_bench_gemm": ([I, I, I, I, I, P, P, P, P, I64, I64, I, I, P, P, SZ, P], I),
    "q27t_gdn_prep": ([P, P, P, P, P, P, P, I, P, P, P, P, P], I),
    "q27t_gdn_recurrent": ([P, P, P, P, P, I, P, P], I),
    "q27t_gdn_chunked": ([P, P, P, P, P, I, I, P, P, P], I),
    "q27t_gdn_chunk_ws_bytes": ([I], SZ),
    "q27t_gdn_gated_norm": ([P, P, P, I, F, P, P, F], I),
    "q27t_gdn_decode_prep": ([P, P, P, P, P, P, P, P, P, P, P], I),
    "q27t_gdn_decode": ([P, P, P, P, P, P, P, P, P, P, F, P, P, F], I),
    "q27t_rmsnorm_quant": ([P, P, I, F, P, I, F, P, P], I),
    "q27t_silu_mul_quant_fp4": ([P, P, I64, I, I, F, P, P], I),
    "q27t_silu_mul_quant_fp4_bf16": ([P, P, I64, I, I, F, P, P], I),
    "q27t_gemm_fp4_bf16": ([I, I, I, P, P, P, P, F, P, P, SZ], I),
    "q27t_linear_bf16_pair": ([P, P, P, I, I, I, P, P], I),
    "q27t_attn_prep": ([P, P, P, P, P, I, I, P, P, I, P, P], I),
    "q27t_attention": ([P, P, I, I, P, P, I, P, P, P, SZ], I),
    "q27t_attention_decode": ([P, P, I, I, P, P, I, P, P, P, SZ], I),
    "q27t_logsoftmax_topk": ([P, I, I, I, P, P], I),
    "q27t_sample": ([P, I, I, F, I, F, P, P], I),
}

_lib = None


def lib():
    global _lib
    if _lib is None:
        _lib = ctypes.CDLL(LIB_PATH)
        for name, (args, res) in SIGNATURES.items():
            try:
                fn = getattr(_lib, name)
            except AttributeError:  # an older build (Q27_LIB) without this entry point
                continue
            fn.argtypes, fn.restype = args, res
    return _lib


def _arg(x):
    if hasattr(x, "data_ptr"):
        return x.data_ptr()
    return x


def call(name, *args):
    """Calls a kernel wrapper; torch tensors are passed by data pointer. Raises on a CUDA / check error."""
    rc = getattr(lib(), name)(*map(_arg, args))
    if rc != 0:
        raise RuntimeError(f"{name}: {lib().q27_last_error().decode()}")


class KvFull(RuntimeError):
    """The KV pool cannot hold the running sequence's next positions (C API -2). Nothing ran for the failed
    part: free pages (park and suspend / drop another sequence) and retry. Engine.pos tells how far a
    prompt got."""


class Engine:
    """The engine: one running sequence, others parked (MAX_SEQS, Phase 6). Token ids are plain Python ints."""

    KV_TYPES = {"bf16": 0, "fp8": 1, "nvfp4": 2}

    def __init__(self, model_dir, max_tokens=32768, prefill_chunk=2048, kv_fp8=False, draft=None, kv_dtype=None):
        """kv_dtype: "bf16" | "fp8" | "nvfp4" (default: kv_fp8 picks fp8 or bf16)."""
        self.max_tokens = max_tokens
        if draft and "Q27_DRAFT_VOCAB" not in os.environ and os.path.exists(DRAFT_VOCAB):
            os.environ["Q27_DRAFT_VOCAB"] = os.path.abspath(DRAFT_VOCAB)
        if os.environ.get("Q27_DRAFT_VOCAB") == "none":
            os.environ["Q27_DRAFT_VOCAB"] = ""
        kv = self.KV_TYPES[kv_dtype] if kv_dtype else int(bool(kv_fp8))
        self.h = lib().q27_create(str(model_dir).encode(), max_tokens, prefill_chunk, kv,
                                  str(draft).encode() if draft else None)
        if not self.h:
            raise RuntimeError(f"q27_create: {lib().q27_last_error().decode()}")
        self.max_tokens = lib().q27_max_tokens(self.h)  # (MAX_TOKENS_FIT=1 lowers it to what the VRAM fits)

    def close(self):
        if self.h:
            lib().q27_destroy(self.h)
            self.h = None

    def _check(self, rc, what):
        if rc == -2:
            raise KvFull(f"{what}: {lib().q27_last_error().decode()}")
        if rc != 0:
            raise RuntimeError(f"{what}: {lib().q27_last_error().decode()}")

    def reset(self):
        self._check(lib().q27_reset(self.h), "q27_reset")

    @property
    def pos(self):
        return lib().q27_pos(self.h)

    def forward_topk(self, ids, k=20):
        """Runs ids; returns per position a list of (token, logprob) for the k most likely next tokens."""
        n = len(ids)
        src = array.array("i", ids)
        out_i, out_l = array.array("i", bytes(4 * n * k)), array.array("f", bytes(4 * n * k))
        self._check(lib().q27_forward_topk(self.h, src.buffer_info()[0], n, k, out_i.buffer_info()[0],
                                           out_l.buffer_info()[0]), "q27_forward_topk")
        return [list(zip(out_i[p * k:(p + 1) * k], out_l[p * k:(p + 1) * k])) for p in range(n)]

    def configure(self, **env):
        """Sets / clears (None) Q27_* environment switches and makes the engine re-read them (drops the
        captured decode graph). Compile-time switches still need another build."""
        for k, v in env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = str(v)
        self._check(lib().q27_configure(self.h), "q27_configure")

    def profile_dump(self):
        """Prints the per-op GPU time since the last dump to stderr (needs Q27_PROFILE=1)."""
        self._check(lib().q27_profile_dump(self.h), "q27_profile_dump")

    def forward_sample(self, ids, temperature=0.0, top_k=20, top_p=1.0, uniform=0.5):
        """Runs ids and returns the sampled next token (temperature <= 0: greedy)."""
        src = array.array("i", ids)
        out = ctypes.c_int32()
        self._check(lib().q27_forward_sample(self.h, src.buffer_info()[0], len(ids), temperature, top_k, top_p,
                                             uniform, ctypes.byref(out)), "q27_forward_sample")
        return out.value

    @property
    def has_draft(self):
        return bool(lib().q27_has_draft(self.h))

    def spec_begin(self, ids, temperature=0.0, top_k=20, top_p=1.0, seed=0, uniform=0.5, max_new=0):
        """Speculative decoding: runs the prompt and returns the first token (also fills the draft context)."""
        src = array.array("i", ids)
        out = ctypes.c_int32()
        self._check(lib().q27_spec_begin(self.h, src.buffer_info()[0], len(ids), temperature, top_k, top_p,
                                         seed & (2**64 - 1), uniform, max_new, ctypes.byref(out)), "q27_spec_begin")
        return out.value

    def spec_step(self, ahead=True):
        """One draft + verify step; returns the new tokens (1..8). ahead=False: the next step is not launched
        before returning (the caller switches to another sequence next)."""
        out = (ctypes.c_int32 * 8)()
        n = lib().q27_spec_step2(self.h, out, int(ahead)) if not ahead else lib().q27_spec_step(self.h, out)
        if n < 0:
            self._check(n, "q27_spec_step")
        return list(out[:n])

    @property
    def max_lanes(self):
        """Sequences a batched step can run (MAX_SEQS with a draft model; 0 / 1: no batched steps)."""
        return lib().q27_max_lanes(self.h)

    def spec_step_lanes(self, lanes):
        """One batched speculative step (P6-10) of the decoding sequences `lanes` (None: the running one, else
        a parking area): the same tokens as a spec_step of each. Returns their new tokens (lists of 1..8)."""
        n = len(lanes)
        src = (ctypes.c_int32 * n)(*[-1 if a is None else a for a in lanes])
        out, counts = (ctypes.c_int32 * (8 * n))(), (ctypes.c_int32 * n)()
        self._check(lib().q27_spec_step_lanes(self.h, src, n, out, counts), "q27_spec_step_lanes")
        return [list(out[8 * z:8 * z + counts[z]]) for z in range(n)]

    def prefill(self, ids, budget, spec=False):
        """Prompt chunks of ids (the rest of the prompt) while more than one prefill chunk is left, at least
        `budget` tokens; returns the tokens run. The rest goes to forward_sample / spec_begin (spec: the latter)."""
        src = array.array("i", ids)
        out = ctypes.c_int()
        self._check(lib().q27_prefill(self.h, src.buffer_info()[0], len(ids), budget, int(spec), ctypes.byref(out)),
                    "q27_prefill")
        return out.value

    @property
    def park_slots(self):
        """Parking areas for other sequences (MAX_SEQS - 1)."""
        return lib().q27_park_slots(self.h)

    def seq_park(self):
        """Parks the running sequence; returns its area (the engine then has no sequence)."""
        out = ctypes.c_int()
        self._check(lib().q27_seq_park(self.h, ctypes.byref(out)), "q27_seq_park")
        return out.value

    def seq_unpark(self, area):
        """Drops the running sequence (if any) and resumes the one parked in `area`."""
        self._check(lib().q27_seq_unpark(self.h, area), "q27_seq_unpark")

    def seq_swap(self, area):
        """Parks the running sequence in `area` and resumes the one that was there."""
        self._check(lib().q27_seq_swap(self.h, area), "q27_seq_swap")

    def seq_drop(self, area):
        """Gives the pages of the sequence parked in `area` back."""
        self._check(lib().q27_seq_drop(self.h, area), "q27_seq_drop")

    def seq_suspend(self, area):
        """Saves the sequence parked in `area` as a prefix cache checkpoint, then drops it."""
        self._check(lib().q27_seq_suspend(self.h, area), "q27_seq_suspend")

    def prefix_begin(self, ids, at=(), spec=False):
        """Prefix cache (STATE_HOST_CACHE > 0): resets the sequence and restores the deepest cached prefix of
        ids; returns its length. Run ids[cached:] next (forward_sample, or spec_begin when spec); that prefill
        saves checkpoints at the positions in `at` (and at CHECKPOINT_INTERVAL multiples)."""
        src, pts = array.array("i", ids), array.array("i", sorted(at) or [0])
        out = ctypes.c_int()
        self._check(lib().q27_prefix_begin(self.h, src.buffer_info()[0], len(ids), pts.buffer_info()[0], len(at),
                                           int(spec), ctypes.byref(out)), "q27_prefix_begin")
        return out.value

    def checkpoint(self):
        """Saves a checkpoint of the sequence so far (e.g. after a generation)."""
        self._check(lib().q27_checkpoint(self.h), "q27_checkpoint")

    def cache_grow(self, n=1):
        """Allocates up to n more pinned state slots of the prefix cache (not allocated at startup; about 80 ms each
        on WSL2); returns how many (0: all are)."""
        out = ctypes.c_int()
        self._check(lib().q27_cache_grow(self.h, n, ctypes.byref(out)), "q27_cache_grow")
        return out.value

    def cache_stats(self):
        out = (ctypes.c_int64 * 12)()
        self._check(lib().q27_cache_stats(self.h, out), "q27_cache_stats")
        keys = ("lookups", "hits", "prompt_tokens", "cached_tokens", "saved", "evicted", "checkpoints", "free_pages",
                "checkpoint_only_pages", "pool_pages", "park_pages", "slots_allocated")
        return dict(zip(keys, out))

    def memory_report(self):
        """The memory pools: {"pools": [{"name", "host", "bytes", "items": [(what, bytes)]}], "slots": state cache
        slots, "checkpoint_bytes": bytes of one}. Fixed once the engine is created."""
        cap = 1 << 20
        while True:
            buf = ctypes.create_string_buffer(cap)
            n = lib().q27_memory_report(self.h, buf, cap)
            if n < 0:
                raise RuntimeError(f"q27_memory_report: {lib().q27_last_error().decode()}")
            if n < cap:
                break
            cap = n + 1
        pools, out = {}, {"pools": [], "slots": 0, "checkpoint_bytes": 0}
        for line in buf.value.decode().splitlines():
            f = line.split("\t")
            if f[0] == "pool":
                pools[f[1]] = {"name": f[1], "host": f[2] == "1", "bytes": int(f[3]), "items": []}
                out["pools"].append(pools[f[1]])
            elif f[0] == "item":
                pools[f[1]]["items"].append((f[2], int(f[3])))
            elif f[0] == "slots":
                out["slots"], out["checkpoint_bytes"] = int(f[1]), int(f[2])
        return out

    def device_memory(self):
        """{"free", "total"} device bytes now (cudaMemGetInfo), "free_at_start" (before the pools), "budget"."""
        out = (ctypes.c_int64 * 4)()
        self._check(lib().q27_device_memory(self.h, out), "q27_device_memory")
        return dict(zip(("free", "total", "free_at_start", "budget"), out))

    @property
    def has_vision(self):
        return bool(lib().q27_has_vision(self.h))

    def mm_set(self, spans):
        """Images of this request (after reset / prefix_begin): spans [(start, gh, gw)] with the merged grid."""
        flat = array.array("i", [v for sp in spans for v in sp] or [0])
        self._check(lib().q27_mm_set(self.h, flat.buffer_info()[0], len(spans)), "q27_mm_set")

    def mm_encode(self, i, pixel_values):
        """Runs the vision tower on image i: pixel_values [4 * gh * gw, 1536] (float32, numpy or CPU torch)."""
        import numpy as np

        pv = np.ascontiguousarray(np.asarray(pixel_values, dtype=np.float32))
        self._check(lib().q27_mm_encode(self.h, i, pv.ctypes.data), "q27_mm_encode")

    def mm_embeds(self, i, rows):
        """Embeddings of image i as a uint16 (BF16 bits) numpy array [rows, 5120]."""
        import numpy as np

        out = np.empty((rows, 5120), dtype=np.uint16)
        self._check(lib().q27_mm_embeds(self.h, i, out.ctypes.data), "q27_mm_embeds")
        return out

    def mm_put(self, i, embeds):
        """Replaces the embeddings of image i (uint16 BF16 bits [rows, 5120]; tests with another vision tower)."""
        import numpy as np

        e = np.ascontiguousarray(np.asarray(embeds, dtype=np.uint16))
        self._check(lib().q27_mm_put(self.h, i, e.ctypes.data), "q27_mm_put")

    def spec_stats(self):
        """Counters since the last call: {"steps", "lookup_steps", "tokens"}."""
        out = (ctypes.c_int64 * 3)()
        self._check(lib().q27_spec_stats(self.h, out), "q27_spec_stats")
        return {"steps": out[0], "lookup_steps": out[1], "tokens": out[2]}

