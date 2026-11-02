"""The vision tower's input on the device (vision_pixels): the resized image's uint8 RGB, normalized and arranged
into patches there, against the image processor's fp32 pixel_values (reference/qwen27b/vision.py patchify) rounded
to BF16, and the fp32 path; bit for bit."""

import pytest
import torch

from q27lib import call
from qwen27b.vision import patchify

dev = "cuda"
# mean[3] + std[3] in 0..255 units as frontend/imageproc.py makes them: the checkpoint's 0.5 / 0.5, and CLIP's
NORMS = [(0.5,) * 6, (0.48145466, 0.4578275, 0.40821073, 0.26862954, 0.26130258, 0.27577711)]


# (the larger two take several chunks of the 4 MiB staging buffer on either path)
@pytest.mark.parametrize("gh, gw", [(2, 2), (4, 6), (40, 200), (128, 128)])
@pytest.mark.parametrize("norm", NORMS)
def test_rgb_patches(gh, gw, norm):
    rgb = torch.randint(0, 256, (gh * 16, gw * 16, 3), dtype=torch.uint8)
    rgb[0, :3] = torch.tensor([0, 255, 128])
    norm = (torch.tensor(norm, dtype=torch.float32) * 255).float()  # (fp32, as the server passes them)
    ref = patchify(rgb, norm[:3].tolist(), norm[3:].tolist())
    out = torch.empty(gh * gw, 1536, dtype=torch.bfloat16, device=dev)
    call("q27t_vision_pixels", None, rgb, norm, gh, gw, out)
    assert torch.equal(out.cpu().view(torch.int16), ref.bfloat16().view(torch.int16))
    out.zero_()
    call("q27t_vision_pixels", ref, None, None, gh, gw, out)
    assert torch.equal(out.cpu().view(torch.int16), ref.bfloat16().view(torch.int16))
