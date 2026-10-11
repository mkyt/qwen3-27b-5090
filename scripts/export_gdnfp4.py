"""Export the gdnfp4 weights as a self-contained modelopt checkpoint other engines can load (vLLM, SGLang).

The engine's model directory (cache/weights/allfp4-gdn) is one shard of NVFP4 GDN projections plus links to the
nvidia checkpoint, whose shards still hold the FP8 GDN projections under the same names (the engine takes the
first shard). Here the nvidia shards are rewritten without those tensors, and config.json / hf_quant_config.json
list the GDN projections as NVFP4 (group size 16) instead of FP8.

    python scripts/export_gdnfp4.py --out /hf/custom/qwen3.8-27b-nvfp4-gdnfp4
"""

import argparse
import json
import pathlib
import shutil
import sys

from safetensors import safe_open
from safetensors.torch import save_file

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "reference"))
from qwen27b.weights import engine_target  # noqa: E402

PROJ_SHARD = "model-00000-proj-nvfp4.safetensors"
SHARD_BYTES = 5 * 1024**3


def requant_config(q, nvfp4):
    """Move the layers in `nvfp4` from the FP8 group to the NVFP4 one (config.json and hf_quant_config.json)."""
    ql = q["quantized_layers"]
    for n in nvfp4:
        assert ql[n]["quant_algo"] == "FP8", n
        ql[n] = {"quant_algo": "NVFP4", "group_size": 16}
    if "config_groups" in q:
        groups = q["config_groups"]
        fp8 = next(g for g in groups.values() if g["weights"]["num_bits"] == 8)
        fp4 = next(g for g in groups.values() if g["weights"]["num_bits"] == 4)
        fp8["targets"] = [t for t in fp8["targets"] if t not in nvfp4]
        fp4["targets"] = sorted(fp4["targets"] + sorted(nvfp4))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", default="gdnfp4", help="weight profile or model directory (engine layout)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    src = engine_target(args.src)
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    with safe_open(src / PROJ_SHARD, "pt") as f:
        proj = {k: f.get_tensor(k) for k in f.keys()}
        meta = f.metadata()
    nvfp4 = sorted({k.rsplit(".", 1)[0] for k in proj})
    print(f"{len(nvfp4)} NVFP4 projections from {PROJ_SHARD}")
    # vLLM runs in_proj_qkv | in_proj_z (and q | k | v) as one GEMM with the largest of their weight global scales,
    # without rescaling the block scales: they must share one (quantize_proj.py build --shared-global).
    for p in nvfp4:
        if p.endswith(".in_proj_qkv") and p[:-4] + "_z" in nvfp4:
            a, b = float(proj[f"{p}.weight_scale_2"]), float(proj[p[:-4] + "_z.weight_scale_2"])
            if a != b:
                raise SystemExit(f"{p}: weight global scales differ from in_proj_z ({a} vs {b}); "
                                 "rebuild with quantize_proj.py build --shared-global")

    # Shards: the base shards without the replaced projections (re-split at 5 GiB), then the projections.
    base = sorted(p for p in src.iterdir() if p.name.endswith(".safetensors") and p.name != PROJ_SHARD)
    plan = []  # (source shard, kept names); tensors are read when written, one output shard at a time
    for p in base:
        with safe_open(p.resolve(), "pt") as f:
            names = [k for k in f.keys() if k.rsplit(".", 1)[0] not in nvfp4]
            dropped = [k for k in f.keys() if k.rsplit(".", 1)[0] in nvfp4]
        plan.append((p, names))
        print(f"{p.name}: keep {len(names)}, drop {len(dropped)} FP8 projection tensors")
    # Group names into output shards by size without loading everything at once.
    shards = []
    cur_names, cur_size = [], 0
    for p, names in plan:
        with safe_open(p.resolve(), "pt") as f:
            for k in names:
                sl = f.get_slice(k)
                n = 1
                for d in sl.get_shape():
                    n *= d
                nbytes = n * {"BF16": 2, "F32": 4, "F8_E4M3": 1, "U8": 1, "F16": 2, "I64": 8}[sl.get_dtype()]
                if cur_size and cur_size + nbytes > SHARD_BYTES:
                    shards.append(cur_names)
                    cur_names, cur_size = [], 0
                cur_names.append((p, k))
                cur_size += nbytes
    proj_items = [(None, k) for k in sorted(proj)]
    shards.append(cur_names)
    shards.append(proj_items)
    total = len(shards)
    weight_map, total_size = {}, 0
    for i, items in enumerate(shards, 1):
        name = f"model-{i:05d}-of-{total:05d}.safetensors"
        tensors, handles = {}, {}
        for p, k in items:
            if p is None:
                tensors[k] = proj[k]
            else:
                if p not in handles:
                    handles[p] = safe_open(p.resolve(), "pt")
                tensors[k] = handles[p].get_tensor(k)
        save_file(tensors, out / name, metadata={"format": "pt"})
        for k, t in tensors.items():
            weight_map[k] = name
            total_size += t.numel() * t.element_size()
        print(f"wrote {name}: {len(tensors)} tensors", flush=True)
        del tensors, handles

    (out / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {"total_size": total_size}, "weight_map": dict(sorted(weight_map.items()))}, indent=2))
    cfg = json.loads((src / "config.json").read_text())
    requant_config(cfg["quantization_config"], nvfp4)
    (out / "config.json").write_text(json.dumps(cfg, indent=2))
    hq = json.loads((src / "hf_quant_config.json").read_text())
    requant_config(hq["quantization"], nvfp4)
    (out / "hf_quant_config.json").write_text(json.dumps(hq, indent=2))
    for p in src.iterdir():  # tokenizer, templates, generation config, license
        if p.name in ("config.json", "hf_quant_config.json", "model.safetensors.index.json", "README.md",
                      "quant_stats.json") or p.name.endswith(".safetensors"):
            continue
        shutil.copyfile(p.resolve(), out / p.name)
    (out / "quantization_gdnfp4.json").write_text(json.dumps({"projection_shard_metadata": meta,
                                                              "nvfp4_projections": nvfp4}, indent=1))
    print(f"wrote {out}: {total} shards, {total_size / 1e9:.2f} GB")


if __name__ == "__main__":
    main()
