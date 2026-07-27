#!/bin/bash
# Download the moonshotai/Kimi-K3 source checkpoint (~1.56 TB).
# Experts ship as MXFP4 (weight_packed u8 + weight_scale e8m0, group 32);
# everything else is bf16. Resumable — snapshot_download skips complete files.
# Usage: scripts/download.sh [DEST]
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${PY:-python3}"
DEST="${1:-$PWD/Kimi-K3-src}"
WORKERS="${HF_WORKERS:-8}"

echo "[download] moonshotai/Kimi-K3 -> $DEST  (~1.56 TB, $WORKERS workers)"
DEST="$DEST" WORKERS="$WORKERS" "$PY" - <<'PY'
import os
from huggingface_hub import snapshot_download
p = snapshot_download(
    "moonshotai/Kimi-K3",
    local_dir=os.environ["DEST"],
    ignore_patterns=["assets/*"],
    max_workers=int(os.environ["WORKERS"]),
)
print("[download] complete ->", p)
PY
echo "[download] done: $(du -sh "$DEST" | cut -f1)"
