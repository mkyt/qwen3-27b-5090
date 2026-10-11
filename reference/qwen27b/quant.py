"""Numerics of the modelopt checkpoint formats, in plain PyTorch (fp32).

NVFP4 (MLP, lm_head; W4A4):
  weight        uint8 [out, in/2]   two e2m1 codes per byte, low nibble = even column
  weight_scale  e4m3  [out, in/16]  one scale per 16 input elements
  weight_scale_2 f32  []            global weight scale
  input_scale   f32   []            global activation scale (static, from calibration)
  W = e2m1(code) * weight_scale * weight_scale_2

  Activations are quantized per 16 elements at run time the way FlashInfer's fp4_quantize does:
  gs = 1 / input_scale, sf = e4m3(gs * amax / 6), q = rne_e2m1(x * gs / sf), x' = q * sf / gs.

FP8 (GDN and attention projections; W8A8, per tensor):
  weight        e4m3  [out, in],  weight_scale f32 [],  input_scale f32 []
  W = weight * weight_scale;  x' = e4m3(x / input_scale) * input_scale
"""

import torch

E2M1_VALUES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
FP4_BLOCK = 16
E4M3_MAX = 448.0
E2M1_MAX = 6.0


def unpack_e2m1(packed: torch.Tensor) -> torch.Tensor:
    """uint8 [..., n/2] -> fp32 [..., n] values in the e2m1 grid (signed)."""
    lut = torch.tensor(E2M1_VALUES, dtype=torch.float32, device=packed.device)
    codes = torch.stack((packed & 0x0F, packed >> 4), dim=-1).flatten(-2).long()
    mag = lut[codes & 0x7]
    return torch.where(codes & 0x8 != 0, -mag, mag)


def round_e2m1(x: torch.Tensor) -> torch.Tensor:
    """Round fp32 values to the nearest e2m1 value, ties to even (as cvt.rn.satfinite.e2m1x2), saturating at 6."""
    a = x.abs().clamp(max=E2M1_MAX)
    grid = torch.tensor(E2M1_VALUES, dtype=torch.float32, device=x.device)
    mids = (grid[1:] + grid[:-1]) / 2  # 0.25 0.75 1.25 1.75 2.5 3.5 5.0
    # bucketize puts a value equal to mids[i] in bin i, i.e. on the lower neighbour (code i).
    idx = torch.bucketize(a, mids)
    # Ties go to the even code (the even magnitudes are 0, 1, 2, 4): step up when the lower one is odd.
    tie = (a.unsqueeze(-1) == mids).any(-1)
    idx = torch.where(tie & (idx % 2 == 1), idx + 1, idx)
    return torch.copysign(grid[idx], x)


def to_e4m3(x: torch.Tensor) -> torch.Tensor:
    """Round to float8_e4m3fn with saturation, returned as fp32."""
    return x.clamp(-E4M3_MAX, E4M3_MAX).to(torch.float8_e4m3fn).to(torch.float32)


def dequant_nvfp4(weight: torch.Tensor, weight_scale: torch.Tensor, weight_scale_2: torch.Tensor) -> torch.Tensor:
    w = unpack_e2m1(weight)  # [out, in]
    s = weight_scale.to(torch.float32).repeat_interleave(FP4_BLOCK, dim=-1)
    return w * s * weight_scale_2.to(torch.float32)


def quant_act_nvfp4(x: torch.Tensor, input_scale: torch.Tensor) -> torch.Tensor:
    """Fake-quantize activations [..., k] (k % 16 == 0) to NVFP4 and back, in fp32."""
    shape = x.shape
    xb = x.to(torch.float32).reshape(*shape[:-1], -1, FP4_BLOCK)
    gs = 1.0 / input_scale.to(torch.float32)
    amax = xb.abs().amax(-1, keepdim=True)
    sf = to_e4m3(gs * amax / E2M1_MAX)
    inv = torch.where(sf != 0, gs / sf, torch.zeros_like(sf))
    q = round_e2m1(xb * inv)
    return (q * sf / gs).reshape(shape)


def dequant_fp8(weight: torch.Tensor, weight_scale: torch.Tensor) -> torch.Tensor:
    return weight.to(torch.float32) * weight_scale.to(torch.float32)


def quant_act_fp8(x: torch.Tensor, input_scale: torch.Tensor) -> torch.Tensor:
    s = input_scale.to(torch.float32)
    return to_e4m3(x.to(torch.float32) / s) * s


class QLinear:
    """y = x @ W^T with the checkpoint's weight format and (optionally) its activation quantization.

    kind: "nvfp4" | "fp8" | "bf16". Weights are dequantized once to `dtype` on `device`.
    """

    def __init__(self, kind, weight, input_scale=None, act_quant=True):
        self.kind = kind
        self.weight = weight  # dequantized [out, in]
        self.input_scale = input_scale
        self.act_quant = act_quant and input_scale is not None

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if self.act_quant:
            if self.kind == "nvfp4":
                x = quant_act_nvfp4(x, self.input_scale)
            elif self.kind == "fp8":
                x = quant_act_fp8(x, self.input_scale)
        return x.to(self.weight.dtype) @ self.weight.T
