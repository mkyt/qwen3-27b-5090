#!/usr/bin/env bash
# Download the pinned checkpoints (revisions from reference/qwen27b/weights.py) into ./hf, the Hugging Face cache the
# compose services mount at /hf. To keep them elsewhere, make ./hf a symlink first. Re-running verifies / resumes.
#   scripts/download-weights.sh             # serving: gdnfp4 (19.5 GB) and the DFlash2 draft (3.5 GB)
#   scripts/download-weights.sh --dev       # also the NVIDIA checkpoint (20.4 GB): tests, reference model, golden
#   scripts/download-weights.sh --bf16      # also the BF16 original (55.6 GB): KL against it, building gdnfp4
#   scripts/download-weights.sh --quasar    # also QUASAR (19.9 GB, WEIGHTS=quasar)
# Requires uv (https://docs.astral.sh/uv/). None of the repos is gated; HF_TOKEN only raises the rate limit.
set -euo pipefail
cd "$(dirname "$0")/.."

pin() {  # NAME -> "repo revision"
  sed -nE "s/^$1 = \(\"([^\"]+)\", \"([0-9a-f]{40})\"\).*/\1 \2/p" reference/qwen27b/weights.py
}
names=(GDNFP4 DFLASH2)
for arg in "$@"; do
  case "$arg" in
    --dev) names+=(TARGET) ;;
    --bf16) names+=(BF16) ;;
    --quasar) names+=(QUASAR) ;;
    *) echo "usage: $0 [--dev] [--bf16] [--quasar]" >&2; exit 2 ;;
  esac
done

mkdir -p hf
export HF_HOME="$(cd hf && pwd -P)"
for name in "${names[@]}"; do
  read -r repo rev <<<"$(pin "$name")"
  [ -n "$rev" ] || { echo "no pinned revision for $name in reference/qwen27b/weights.py" >&2; exit 1; }
  echo "==> $repo@$rev -> $HF_HOME"
  uvx --from 'huggingface_hub[hf_xet]==1.28.0' hf download "$repo" --revision "$rev"
done
echo "==> done"
