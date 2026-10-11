#!/usr/bin/env bash
# Build the gdnfp4 weight profile (Phase Q, PQ-14): the nvidia checkpoint with the 48 GDN layers' in_proj_qkv /
# in_proj_z / out_proj quantized from the BF16 original to NVFP4 with GPTQ; attention projections stay FP8.
#   1. Hessians of the projection inputs: the reference model (nvidia checkpoint) over the calibration set, 16
#      layers per pass (VRAM), about 16 minutes each -> cache/pq-hessian (15.6 GB)
#   2. GPTQ of the 144 GDN projections (about 10 s each) -> cache/weights/allfp4-gdn (+ cache/pq-parts-gptq)
# Needs the BF16 original (scripts/download-weights.sh --bf16) and the GPU to itself. Re-running skips finished steps.
# Arguments go to the hessian step: the calibration texts, e.g. `--docs my-notes/*.md --dumps my-api-dumps` or
# `@calib.args` (one argument per line). The repository's code is always included.
#   scripts/build-gdnfp4.sh --docs docs/*.md README.md
set -euo pipefail
cd "$(dirname "$0")/.."
DC="docker compose --profile dev run --rm -e PYTHONUNBUFFERED=1 -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True dev"
for g in 0 1 2 3; do
  last=$(printf "cache/pq-hessian/%02d-out.pt" $((g * 16 + 15)))
  if [ -f "$last" ]; then echo "hessians of layers $((g * 16))-$((g * 16 + 15)): present"; continue; fi
  $DC python scripts/quantize_proj.py hessian --long-len 8192 --layers "$(seq -s, $((g * 16)) $((g * 16 + 15)))" \
    --out cache/pq-hessian "$@"
done
$DC python scripts/quantize_proj.py build --method gptq --search 3 --hessians cache/pq-hessian \
  --parts cache/pq-parts-gptq-shared --shared-global --keep-fp8 self_attn --out cache/weights/allfp4-gdn
echo "built cache/weights/allfp4-gdn (WEIGHTS=gdnfp4)"
