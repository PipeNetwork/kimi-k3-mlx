#!/bin/bash
# Build one half of the two-node Kimi-K3 2-bit JACCL pipeline checkpoint.
# Run rank 0 on Beast1 and rank 1 on Beast2. Each process downloads only the
# source shards for its own decoder layers and deletes them as conversion moves
# forward. The layer journal makes the command safe to restart verbatim.
set -euo pipefail
cd "$(dirname "$0")/.."

RANK="${1:?usage: scripts/build_pipeline_stage.sh RANK [MAX_LAYERS_THIS_RUN]}"
MAX_LAYERS="${2:-}"
SIZE=2
PY="$PWD/.venv/bin/python"
MODEL_NAME="Kimi-K3-MLX-2bit-pipeline-2"
SRC="$PWD/work/source-rank${RANK}"
OUT="$PWD/weights/$MODEL_NAME/rank${RANK}"
LOG_DIR="$PWD/work/logs"

read -r START END <<EOF
$($PY - "$RANK" "$SIZE" <<'PY'
import json, sys
rank, size = map(int, sys.argv[1:])
with open("reference/config.json") as f:
    n = json.load(f)["text_config"]["num_hidden_layers"]
base, extra = divmod(n, size)
stage = size - rank - 1
counts = [base + (i < extra) for i in range(size)]
start = sum(counts[:stage])
print(start, start + counts[stage])
PY
)
EOF

mkdir -p "$SRC" "$OUT" "$LOG_DIR"
scripts/install_model.sh "$PY"

ARGS=(
  --src "$SRC"
  --hf-repo moonshotai/Kimi-K3
  --evict-source-shards
  --out "$OUT"
  --profile 2bit
  --layer-start "$START"
  --layer-end "$END"
  --stage-rank "$RANK"
  --stage-size "$SIZE"
  --resume
)
if [ -n "$MAX_LAYERS" ]; then
  ARGS+=(--max-layers-this-run "$MAX_LAYERS")
fi

echo "[stage] rank=$RANK/$SIZE layers=[$START,$END) out=$OUT"
exec "$PY" scripts/convert.py "${ARGS[@]}" \
  2>&1 | tee -a "$LOG_DIR/convert-rank${RANK}.log"
