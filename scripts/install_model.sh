#!/bin/bash
# Register both Kimi-K3 loaders in the target mlx-lm installation.
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${1:-python3}"
MODELS_DIR=$("$PY" -c "import os,mlx_lm;print(os.path.join(os.path.dirname(mlx_lm.__file__),'models'))")
cp kimi_k3.py "$MODELS_DIR/kimi_k3.py"
cp kimi_k3_uvmax.py "$MODELS_DIR/kimi_k3_uvmax.py"
mkdir -p "$MODELS_DIR/../tool_parsers"
cp mlx_lm_compat/tool_parsers/kimi_k3.py "$MODELS_DIR/../tool_parsers/kimi_k3.py"
echo "[install] Kimi-K3 loaders + XTML parser -> $MODELS_DIR"
