"""Lazy reader for the pinned safetensors checkpoints in the shared HF cache."""

import json
import os
import pathlib

import torch
from safetensors import safe_open

from .quant import QLinear, dequant_fp8, dequant_nvfp4

HF_HUB = pathlib.Path(os.environ.get("HF_HOME", pathlib.Path.home() / ".cache/huggingface")) / "hub"

TARGET = ("nvidia/Qwen3.8-27B-NVFP4", "dbb8f445b3145f8a4c18ddc769f032d57d32867c")
DFLASH2 = ("incoai/Qwen3.8-27B-DFlash2", "015e795645c74b1a0eeef3b570031fb62e769bc5")
# allfp4 candidate (Phase Q): every linear in NVFP4 (QAT), compressed-tensors; lm_head is the BF16 original,
# so the engine takes the NVFP4 lm_head of TARGET (Q27_LM_HEAD_CKPT).
QUASAR = ("QUASAR-QAT/Qwen3.8-27B-QUASAR-NVFP4", "15d2e47bffe5d8ad23928879f8f7d2f74909e259")
BF16 = ("Qwen/Qwen3.8-27B", "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0")  # the original (Phase Q quality reference)
# The gdnfp4 weight profile as published: scripts/build-gdnfp4.sh + scripts/export_gdnfp4.py (the GDN projections in
# NVFP4, the rest as TARGET). The serve default; scripts/download-weights.sh fetches it.
GDNFP4 = ("juzbox/Qwen3.8-27B-NVFP4-GDN", "3e686408cf61c2b579c67ab21696c9fc94e41e25")


# Development copy on the Linux file system (gitignored): the HF cache on /mnt/d loads at ~1.1 GB/s,
# a copy here comes from the page cache. Used when present, e.g.
#   cp -L <HF cache>/models--nvidia--Qwen3.8-27B-NVFP4/snapshots/<rev>/* cache/weights/nvidia--Qwen3.8-27B-NVFP4/<rev>/
LOCAL_COPIES = pathlib.Path(__file__).resolve().parents[2] / "cache/weights"
# A local build of the gdnfp4 profile (scripts/build-gdnfp4.sh: a shard of NVFP4 GDN projections + links to TARGET),
# used when the published GDNFP4 is not downloaded. Its outputs are the same as the published one's.
GDNFP4_DIR = "allfp4-gdn"


def snapshot(repo: str, rev: str) -> pathlib.Path:
    local = LOCAL_COPIES / repo.replace("/", "--") / rev
    if (local / "config.json").is_file():
        return local
    path = HF_HUB / f"models--{repo.replace('/', '--')}" / "snapshots" / rev
    if not path.is_dir():
        raise FileNotFoundError(f"{repo}@{rev} is not in {HF_HUB} (scripts/download-weights.sh)")
    return path


def engine_target(weights: str = "nvidia") -> pathlib.Path:
    """The model directory the engine opens for a weight profile (nvidia | gdnfp4 | quasar | a directory, e.g. one from
    scripts/quantize_proj.py). QUASAR keeps the BF16 original lm_head, so the engine reads TARGET's NVFP4 one
    (Q27_LM_HEAD_CKPT, set here)."""
    if weights == "gdnfp4":  # Phase Q: GDN projections in NVFP4 (GPTQ), attention FP8
        try:
            return snapshot(*GDNFP4)
        except FileNotFoundError:
            path = LOCAL_COPIES / GDNFP4_DIR
            if not (path / "config.json").is_file():
                raise FileNotFoundError(f"{GDNFP4[0]}@{GDNFP4[1]} is not in {HF_HUB}: "
                                        "run scripts/download-weights.sh") from None
            return path
    if weights not in ("nvidia", "quasar"):
        return pathlib.Path(weights)
    if weights == "quasar":
        os.environ["Q27_LM_HEAD_CKPT"] = str(snapshot(*TARGET))
        return snapshot(*QUASAR)
    return snapshot(*TARGET)


class Checkpoint:
    """Reads single tensors on demand; keeps shard handles open."""

    def __init__(self, path: pathlib.Path, lm_head_from: "Checkpoint | None" = None):
        self.path = pathlib.Path(path)
        self.lm_head_from = lm_head_from  # a BF16 lm_head is taken from here (as the engine's Q27_LM_HEAD_CKPT)
        index = self.path / "model.safetensors.index.json"
        if index.exists():
            self.weight_map = json.loads(index.read_text())["weight_map"]
        else:
            with safe_open(self.path / "model.safetensors", "pt") as f:
                self.weight_map = {k: "model.safetensors" for k in f.keys()}
        self._handles = {}
        self.config = json.loads((self.path / "config.json").read_text())

    def __contains__(self, name: str) -> bool:
        return name in self.weight_map

    def get(self, name: str, device="cpu") -> torch.Tensor:
        shard = self.weight_map[name]
        if shard not in self._handles:
            self._handles[shard] = safe_open(self.path / shard, "pt", device="cpu")
        return self._handles[shard].get_tensor(name).to(device)

    def linear(self, prefix: str, device="cpu", dtype=torch.float32, act_quant=True) -> QLinear:
        """Build a QLinear for `<prefix>.weight` in whatever format the checkpoint stores it."""
        if prefix == "lm_head" and self.lm_head_from is not None and self.get("lm_head.weight").dtype == torch.bfloat16:
            return self.lm_head_from.linear(prefix, device, dtype, act_quant)
        if f"{prefix}.weight_packed" in self:  # compressed-tensors NVFP4: the global scales are inverses
            dq = dequant_nvfp4(self.get(f"{prefix}.weight_packed", device), self.get(f"{prefix}.weight_scale", device),
                               1 / self.get(f"{prefix}.weight_global_scale", device))
            return QLinear("nvfp4", dq.to(dtype), 1 / self.get(f"{prefix}.input_global_scale", device), act_quant)
        w = self.get(f"{prefix}.weight", device)
        input_scale = self.get(f"{prefix}.input_scale", device) if f"{prefix}.input_scale" in self else None
        if w.dtype == torch.uint8:
            dq = dequant_nvfp4(w, self.get(f"{prefix}.weight_scale", device), self.get(f"{prefix}.weight_scale_2", device))
            kind = "nvfp4"
        elif w.dtype == torch.float8_e4m3fn:
            dq = dequant_fp8(w, self.get(f"{prefix}.weight_scale", device))
            kind = "fp8"
        else:
            dq, kind = w, "bf16"
        return QLinear(kind, dq.to(dtype), input_scale, act_quant)

    def bf16_weight(self, prefix: str, device="cpu") -> torch.Tensor:
        """`<prefix>.weight` in BF16; an NVFP4 one (compressed-tensors) dequantized and rounded to BF16, as
        the engine loads GDN in_proj_a / b."""
        if f"{prefix}.weight_packed" not in self:
            return self.get(f"{prefix}.weight", device)
        return dequant_nvfp4(self.get(f"{prefix}.weight_packed", device), self.get(f"{prefix}.weight_scale", device),
                             1 / self.get(f"{prefix}.weight_global_scale", device)).bfloat16()
