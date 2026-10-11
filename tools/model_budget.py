#!/usr/bin/env python3
"""Memory plan and roofline for the Qwen3.8-27B runtime, computed from the checkpoint headers.

Reads only safetensors headers (no weights), so it runs on the host with the standard library:

    python3 tools/model_budget.py                       # default config (docs/architecture.md §5)
    python3 tools/model_budget.py --kv nvfp4 --weights allfp4 --vision bf16
    python3 tools/model_budget.py --bandwidth 1.62      # after Phase 0 measured the effective GB/s

Numbers that are estimates until Phase 0 measures them (CUDA context, activation arena, graphs)
are flags with the design's estimates as defaults.
"""

import argparse
import collections
import json
import os
import pathlib
import re
import struct

GiB = 2**30
HUB = pathlib.Path(os.environ.get("HF_HOME", pathlib.Path(__file__).resolve().parents[1] / "hf")) / "hub"

TARGET = ("nvidia/Qwen3.8-27B-NVFP4", "dbb8f445b3145f8a4c18ddc769f032d57d32867c")
DFLASH2 = ("incoai/Qwen3.8-27B-DFlash2", "015e795645c74b1a0eeef3b570031fb62e769bc5")

# Bytes per element of a 4-bit block-scaled format (e2m1 + one e4m3 scale per 16 values).
NVFP4 = 0.5 + 1 / 16


def snapshot(repo, rev):
    return HUB / f"models--{repo.replace('/', '--')}" / "snapshots" / rev


def tensors(path):
    """Yield (name, dtype, shape, nbytes) for every tensor of every shard."""
    for f in sorted(path.glob("*.safetensors")):
        with open(f, "rb") as fh:
            n = struct.unpack("<Q", fh.read(8))[0]
            header = json.loads(fh.read(n))
        for name, v in header.items():
            if name != "__metadata__":
                yield name, v["dtype"], v["shape"], v["data_offsets"][1] - v["data_offsets"][0]


def target_category(name):
    if "visual" in name:
        return "vision"
    if name.startswith("mtp."):
        return "mtp"
    if "embed_tokens" in name:
        return "embed"
    if "lm_head" in name:
        return "lm_head"
    if "linear_attn" in name:
        return "gdn"
    if "self_attn" in name:
        return "attn"
    if ".mlp." in name:
        return "mlp"
    return "norms"


def load_target(path):
    """Bytes per category, plus the FP8 GDN/attention parameter count (requantized by allfp4)."""
    by_cat = collections.Counter()
    fp8_params = 0
    for name, dtype, shape, nbytes in tensors(path):
        cat = target_category(name)
        by_cat[cat] += nbytes
        if cat in ("gdn", "attn") and dtype == "F8_E4M3":
            fp8_params += nbytes  # 1 byte per element
    return by_cat, fp8_params


def load_draft(path):
    """(linear params, other bytes): linear layers get quantized, codebooks / norms stay as stored."""
    linear, other = 0, 0
    for name, dtype, shape, nbytes in tensors(path):
        if name.endswith(".weight") and len(shape) == 2 and "codebook" not in name:
            linear += shape[0] * shape[1]
        else:
            other += nbytes
    return linear, other


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--budget", type=float, default=30.3, help="VRAM budget of the engine process, GiB")
    ap.add_argument("--weights", choices=["nvidia", "allfp4"], default="nvidia")
    ap.add_argument("--kv", choices=["bf16", "fp8", "fp8k_nvfp4v", "nvfp4"], default="fp8")
    ap.add_argument("--gdn-state", choices=["fp32", "bf16", "fp16"], default="bf16")
    ap.add_argument("--spec", choices=["dflash2", "mtp", "lookup", "none"], default="dflash2")
    ap.add_argument("--draft-dtype", choices=["nvfp4", "fp8", "bf16"], default="nvfp4")
    ap.add_argument("--vision", choices=["off", "host", "fp8", "bf16"], default="off")
    ap.add_argument("--embed", choices=["host", "device"], default="host")
    ap.add_argument("--max-seqs", type=int, default=4)
    ap.add_argument("--cuda-context", type=float, default=0.35, help="GiB, measured in Phase 0")
    ap.add_argument("--arena", type=float, default=0.45, help="activation arena, GiB")
    ap.add_argument("--graphs", type=float, default=0.10, help="CUDA graphs, GiB")
    ap.add_argument("--checkpoints-gpu", type=float, default=0.30, help="GPU state-checkpoint cache, GiB")
    ap.add_argument("--bandwidth", type=float, default=1.65, help="effective DRAM bandwidth, TB/s")
    ap.add_argument("--draft-vocab", type=int, default=65536, help="pruned draft vocab rows (0 = full)")
    args = ap.parse_args()

    cfg = json.loads((snapshot(*TARGET) / "config.json").read_text())["text_config"]
    by_cat, fp8_params = load_target(snapshot(*TARGET))
    hidden, vocab = cfg["hidden_size"], cfg["vocab_size"]
    n_attn = cfg["layer_types"].count("full_attention")
    n_gdn = cfg["layer_types"].count("linear_attention")
    kv_heads, head_dim = cfg["num_key_value_heads"], cfg["head_dim"]
    v_heads, k_dim, v_dim = cfg["linear_num_value_heads"], cfg["linear_key_head_dim"], cfg["linear_value_head_dim"]
    conv_dim = 2 * cfg["linear_num_key_heads"] * k_dim + v_heads * v_dim

    # --- weights resident on the GPU ---
    gdn_attn = by_cat["gdn"] + by_cat["attn"]
    if args.weights == "allfp4":
        gdn_attn -= fp8_params * (1 - NVFP4)
    weights = {
        "mlp": by_cat["mlp"],
        "gdn+attn": gdn_attn,
        "lm_head": by_cat["lm_head"],
        "norms": by_cat["norms"],
        "embed": by_cat["embed"] if args.embed == "device" else 0,
        "vision": {"off": 0, "host": 0, "fp8": by_cat["vision"] / 2, "bf16": by_cat["vision"]}[args.vision],
    }
    per_byte = {"nvfp4": NVFP4, "fp8": 1.0, "bf16": 2.0}[args.draft_dtype]
    draft_bytes, draft_kv_per_seq, mtp_kv_per_token = 0, 0, 0
    if args.spec == "dflash2":
        dcfg = json.loads((snapshot(*DFLASH2) / "config.json").read_text())
        linear, other = load_draft(snapshot(*DFLASH2))
        draft_bytes = linear * per_byte + other
        window = dcfg["sliding_window"]
        draft_kv_per_seq = dcfg["num_hidden_layers"] * 2 * dcfg["num_key_value_heads"] * dcfg["head_dim"] * 2 * window
    elif args.spec == "mtp":
        draft_bytes = by_cat["mtp"] * per_byte / 2  # stored in BF16
        mtp_kv_per_token = 1  # one extra attention layer, added to the KV bytes below
    weights["draft"] = draft_bytes

    # --- per-sequence state ---
    state_elem = {"fp32": 4, "bf16": 2, "fp16": 2}[args.gdn_state]
    gdn_state = n_gdn * v_heads * k_dim * v_dim * state_elem
    conv_state = n_gdn * conv_dim * (cfg["linear_conv_kernel_dim"] - 1) * 2
    per_seq = gdn_state + conv_state + draft_kv_per_seq

    kv_elem = {"bf16": 2.0, "fp8": 1.0, "fp8k_nvfp4v": (1.0 + NVFP4) / 2, "nvfp4": NVFP4}[args.kv]
    kv_per_token = (n_attn + mtp_kv_per_token) * kv_heads * head_dim * 2 * kv_elem

    fixed = {
        "CUDA context": args.cuda_context * GiB,
        **{f"weights: {k}": v for k, v in weights.items() if v},
        "activation arena": args.arena * GiB,
        "CUDA graphs": args.graphs * GiB,
        f"per-seq state x{args.max_seqs}": per_seq * args.max_seqs,
        "state checkpoints (GPU)": args.checkpoints_gpu * GiB,
    }
    kv_bytes = args.budget * GiB - sum(fixed.values())
    pool = int(kv_bytes // kv_per_token)
    native = cfg["max_position_embeddings"]

    print(f"config: weights={args.weights} kv={args.kv} gdn_state={args.gdn_state} spec={args.spec}"
          f"/{args.draft_dtype} vision={args.vision} embed={args.embed} max_seqs={args.max_seqs}")
    print(f"\n{'memory plan':<34}{'GiB':>8}")
    for k, v in fixed.items():
        print(f"  {k:<32}{v / GiB:8.2f}")
    print(f"  {'KV pool':<32}{kv_bytes / GiB:8.2f}")
    print(f"  {'budget':<32}{args.budget:8.2f}")
    print(f"\nKV per token {kv_per_token / 1024:.1f} KiB -> pool {pool:,} tokens; "
          f"max context {min(pool, native):,} (native {native:,})")
    print(f"per-seq state {per_seq / 2**20:.0f} MiB (GDN {gdn_state / 2**20:.0f}, conv {conv_state / 2**20:.1f}, "
          f"draft KV {draft_kv_per_seq / 2**20:.0f})")

    # --- decode roofline (one step, one sequence) ---
    stream = weights["mlp"] + weights["gdn+attn"] + weights["lm_head"] + weights["norms"] + draft_bytes
    if args.spec == "dflash2":
        rows = args.draft_vocab or vocab
        stream += by_cat["lm_head"] * rows / vocab  # draft logits read the (pruned) LM head again
    stream += 2 * gdn_state  # state read + write
    print(f"\ndecode step at {args.bandwidth} TB/s (weights{' + draft' if draft_bytes else ''} + state + KV):")
    for ctx in (4096, 32768, 131072, 262144):
        if ctx > min(pool, native):
            continue
        b = stream + ctx * kv_per_token
        print(f"  ctx {ctx:>7,}: {b / 1e9:6.2f} GB  {b / 1e9 / args.bandwidth:6.2f} ms")
    nonspec = weights["mlp"] + weights["gdn+attn"] + weights["lm_head"] + 2 * gdn_state
    print(f"  no speculation, short ctx: {nonspec / 1e9 / args.bandwidth:.2f} ms/token "
          f"({args.bandwidth * 1e3 / (nonspec / 1e9):.0f} tok/s)")

    # --- prefill FLOPs per token (linear layers; attention grows with position) ---
    linear_params = (by_cat["mlp"] / NVFP4 + fp8_params)  # NVFP4 bytes -> params; FP8 bytes = params
    attn_per_pos = n_attn * cfg["num_attention_heads"] * head_dim * 4
    print(f"\nprefill: linear ~{2 * linear_params / 1e9:.1f} GFLOP/token; attention "
          f"{attn_per_pos / 1e6:.2f} MFLOP x position "
          f"(avg over a 194K prompt: {attn_per_pos * 97_000 / 1e9:.0f} GFLOP/token)")


if __name__ == "__main__":
    main()
