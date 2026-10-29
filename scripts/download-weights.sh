#!/usr/bin/env bash
# Download the pinned checkpoints (revisions from reference/qwen27b/paths.py) into ./hf, the Hugging Face cache the
# compose services mount at /hf. To keep them elsewhere, make ./hf a symlink first. If a download fails or is
# interrupted, run the same command again: it resumes where it stopped (scripts/download_weights.py does the work).
#   scripts/download-weights.sh             # serving: gdnfp4 (19.5 GB) and the DFlash2 draft (3.5 GB)
#   scripts/download-weights.sh --dev       # also the NVIDIA checkpoint (20.4 GB): tests, reference model, golden
#   scripts/download-weights.sh --bf16      # also the BF16 original (55.6 GB): KL against it, building gdnfp4
#   scripts/download-weights.sh --quasar    # also QUASAR (19.9 GB, WEIGHTS=quasar)
#   scripts/download-weights.sh --repo ORG/NAME[@REVISION]  # also another checkpoint of the model (WEIGHTS=ORG/NAME)
# Requires Python 3.10+ (standard library only), else uv (https://docs.astral.sh/uv/) to fetch one. None of the repos
# is gated; HF_TOKEN only raises the rate limit.
set -euo pipefail
cd "$(dirname "$0")/.."

also=()
while [ $# -gt 0 ]; do
  case "$1" in
    --dev) also+=(--also nvidia) ;;
    --bf16) also+=(--also bf16) ;;
    --quasar) also+=(--also quasar) ;;
    --repo) [ $# -ge 2 ] || { echo "--repo ORG/NAME[@REVISION]" >&2; exit 2; }; also+=(--also "$2"); shift ;;
    *) echo "usage: $0 [--dev] [--bf16] [--quasar] [--repo ORG/NAME[@REVISION]]" >&2; exit 2 ;;
  esac
  shift
done

if python3 -c 'import sys; sys.exit(sys.version_info < (3, 10))' 2>/dev/null; then
  py=(python3)
else
  py=(uv run --no-project --python 3.12 python)
fi
mkdir -p hf
HF_HOME="$(cd hf && pwd -P)" "${py[@]}" scripts/download_weights.py --weights gdnfp4 "${also[@]}"
