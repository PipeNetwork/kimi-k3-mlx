#!/bin/bash
# Register kimi_k3.py into the target interpreter's mlx_lm/models/ directory.
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${1:-python3}"
DST=$("$PY" -c "import os,mlx_lm;print(os.path.join(os.path.dirname(mlx_lm.__file__),'models','kimi_k3.py'))")
cp kimi_k3.py "$DST"
echo "[install] kimi_k3 -> $DST"
