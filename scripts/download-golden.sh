#!/usr/bin/env bash
# Download the golden reference recordings (next-token top-20 distributions of the PyTorch reference model over
# bench/golden/prompts.jsonl) from the repository's GitHub Releases into bench/golden/. Checks their SHA-256.
#   scripts/download-golden.sh
# Files: ref-gpu (NVIDIA's checkpoint, KV bf16: what the engine is compared with), ref-gpu-kvfp8 (the same with an
# FP8 KV cache: the noise floor), ref-gpu-bf16model (the BF16 original: quantization error), vision-ref (images).
# Uses the GitHub CLI when it is installed (works for private forks too), curl otherwise.
set -euo pipefail
cd "$(dirname "$0")/../bench/golden"
REPO=${GOLDEN_REPO:-mkyt/qwen3-27b-5090}
TAG=${GOLDEN_TAG:-golden-v2}
SUMS='f7b075d29eda6d64be59ef8bacce62743a033290caef790ae1b10cc9ba066d14  ref-gpu.jsonl.gz
e0bd067a96338d2bdbc1a7ef4ad5b74f5cfc951cc5c6ee457c32a727b160c999  ref-gpu-kvfp8.jsonl.gz
a4058d942f42be87347437c58c9772125ebcce20e2d74f3ed13704e2d1369814  ref-gpu-bf16model.jsonl.gz
95bde67ed96c05ea2cf034b5d38e3faedf6c9c5aca3818be18e712ef6dea54af  vision-ref.jsonl.gz'

while read -r sum file; do
  if [ -f "$file" ] && echo "$sum  $file" | sha256sum -c --status; then
    echo "$file: ok"; continue
  fi
  if command -v gh >/dev/null; then
    gh release download "$TAG" --repo "$REPO" --pattern "$file" --clobber
  else
    curl -fL --retry 3 -o "$file" "https://github.com/$REPO/releases/download/$TAG/$file"
  fi
  echo "$sum  $file" | sha256sum -c
done <<< "$SUMS"
