#!/usr/bin/env bash
# Download the pinned checkpoints (revisions from reference/qwen27b/paths.py) into ./hf, the Hugging Face cache the
# compose services mount at /hf. To keep them elsewhere, make ./hf a symlink first. Re-running verifies / resumes.
#   scripts/download-weights.sh             # serving: gdnfp4 (19.5 GB) and the DFlash2 draft (3.5 GB)
#   scripts/download-weights.sh --dev       # also the NVIDIA checkpoint (20.4 GB): tests, reference model, golden
#   scripts/download-weights.sh --bf16      # also the BF16 original (55.6 GB): KL against it, building gdnfp4
#   scripts/download-weights.sh --quasar    # also QUASAR (19.9 GB, WEIGHTS=quasar)
#   scripts/download-weights.sh --repo ORG/NAME[@REVISION]  # also another checkpoint of the model (WEIGHTS=ORG/NAME);
#                                           # scripts/download_weights.py checks first whether the engine takes it
# Requires uv (https://docs.astral.sh/uv/). None of the repos is gated; HF_TOKEN only raises the rate limit.
set -euo pipefail
cd "$(dirname "$0")/.."

pin() {  # NAME -> "repo revision"
  sed -nE "s/^$1 = \(\"([^\"]+)\", \"([0-9a-f]{40})\"\).*/\1 \2/p" reference/qwen27b/paths.py
}
names=(GDNFP4 DFLASH2)
repos=()
while [ $# -gt 0 ]; do
  case "$1" in
    --dev) names+=(TARGET) ;;
    --bf16) names+=(BF16) ;;
    --quasar) names+=(QUASAR) ;;
    --repo) [ $# -ge 2 ] || { echo "--repo ORG/NAME[@REVISION]" >&2; exit 2; }; repos+=("$2"); shift ;;
    *) echo "usage: $0 [--dev] [--bf16] [--quasar] [--repo ORG/NAME[@REVISION]]" >&2; exit 2 ;;
  esac
  shift
done

mkdir -p hf
export HF_HOME="$(cd hf && pwd -P)"
for name in "${names[@]}"; do
  read -r repo rev <<<"$(pin "$name")"
  [ -n "$rev" ] || { echo "no pinned revision for $name in reference/qwen27b/paths.py" >&2; exit 1; }
  echo "==> $repo@$rev -> $HF_HOME"
  uvx --from 'huggingface_hub[hf_xet]==1.28.0' hf download "$repo" --revision "$rev"
done
for r in "${repos[@]}"; do  # (a branch or tag also gets refs/<name>, which reference/qwen27b/paths.py reads)
  repo="${r%%@*}"; rev="main"; [ "$r" = "$repo" ] || rev="${r#*@}"
  echo "==> $repo@$rev -> $HF_HOME"
  uvx --from 'huggingface_hub[hf_xet]==1.28.0' hf download "$repo" --revision "$rev"
done
echo "==> done"
