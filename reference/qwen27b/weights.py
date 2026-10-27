"""Lazy reader for the pinned safetensors checkpoints in the shared HF cache."""

import json
import pathlib

import torch
from safetensors import safe_open

from .paths import (BF16, DFLASH2, GDNFP4, GDNFP4_DIR, HF_HUB, LOCAL_COPIES, QUASAR, TARGET,  # noqa: F401
                    engine_target, snapshot)
from .quant import QLinear, dequant_fp8, dequant_nvfp4


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
