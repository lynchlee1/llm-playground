#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# convert_model.sh
#
# Downloads (or uses an already-present) Qwen3.5-9B checkpoint and converts
# it to a 5-bit MLX quantised model ready for local inference.
#
# Prerequisites:
#   • Python ≥ 3.10 with mlx-lm installed  (pip install mlx-lm)
#   • Enough disk space (~10 GB for the original weights + ~6 GB for Q5)
#
# Usage:
#   bash scripts/convert_model.sh
#   bash scripts/convert_model.sh --hf-repo Qwen/Qwen3.5-9B   # custom source
# ---------------------------------------------------------------------------
set -euo pipefail

HF_REPO="${1:-Qwen/Qwen3.5-9B}"
SOURCE_DIR="./Qwen3.5-9B"
OUTPUT_DIR="./Qwen3.5-9B-MLX-Q5"
Q_BITS=5

# ── Step 1: download weights from Hugging Face if not already present ───────
if [ ! -d "$SOURCE_DIR" ]; then
    echo ">>> Downloading $HF_REPO to $SOURCE_DIR …"
    python -c "
from huggingface_hub import snapshot_download
snapshot_download(repo_id='$HF_REPO', local_dir='$SOURCE_DIR', local_dir_use_symlinks=False)
"
else
    echo ">>> Found existing weights at $SOURCE_DIR — skipping download."
fi

# ── Step 2: quantise to MLX Q5 ───────────────────────────────────────────────
echo ">>> Converting $SOURCE_DIR → $OUTPUT_DIR (Q${Q_BITS}) …"
python -m mlx_lm.convert \
    --model "$SOURCE_DIR" \
    --quantize \
    --q-bits "$Q_BITS" \
    --output-path "$OUTPUT_DIR"

echo ">>> Done!  Quantised model is at: $OUTPUT_DIR"
