#!/bin/bash
# Wait for the source download to finish, verify it, then run REAP calibration.
#
# Chained so calibration starts the moment the download lands rather than
# whenever someone notices. Verification is not optional: a truncated download
# would otherwise surface as a confusing mid-run failure 40 layers deep.
#
# Usage: scripts/run_calibration.sh [DOWNLOAD_PID]
set -uo pipefail
cd "$(dirname "$0")/.." || exit
PY="${PY:-python3}"
SRC="${SRC:-$PWD/Kimi-K3-src}"
DL_PID="${1:-}"
LOG="$PWD/out/reap_calibrate.log"
SEQS="${SEQS:-64}"
SEQLEN="${SEQLEN:-2048}"
BATCH="${BATCH:-8}"

say() { echo "[chain $(date +%H:%M:%S)] $*"; }

if [ -n "$DL_PID" ]; then
  say "waiting for download pid $DL_PID"
  while kill -0 "$DL_PID" 2>/dev/null; do sleep 60; done
  say "download process exited"
fi

# --- verify the checkpoint is complete before spending hours on it
say "verifying source checkpoint"
"$PY" - "$SRC" <<'PY' || { echo "CALIB-FAILED: incomplete source"; exit 1; }
import json, os, sys
src = sys.argv[1]
missing = [f for f in ("config.json", "model.safetensors.index.json",
                       "tiktoken.model", "tokenization_kimi.py")
           if not os.path.exists(os.path.join(src, f))]
if missing:
    print("missing:", missing); sys.exit(1)
idx = json.load(open(os.path.join(src, "model.safetensors.index.json")))
wm = idx["weight_map"]
shards = sorted(set(wm.values()))
absent = [s for s in shards if not os.path.exists(os.path.join(src, s))]
if absent:
    print(f"missing {len(absent)}/{len(shards)} shards, e.g. {absent[:3]}"); sys.exit(1)
have = sum(os.path.getsize(os.path.join(src, s)) for s in shards)
want = idx["metadata"]["total_size"]
print(f"  {len(shards)} shards, {have/1e12:.4f} TB on disk vs {want/1e12:.4f} TB in index")
# safetensors files carry an 8-byte header length + JSON header, so on-disk is
# slightly larger than the tensor bytes the index totals.
if have < want:
    print("  ON-DISK SMALLER THAN INDEX -> truncated"); sys.exit(1)
print("  checkpoint complete")
PY

if [ ! -f out/calib.txt ]; then
  say "building calibration corpus"
  "$PY" scripts/make_calib.py --out out/calib.txt --mb 12 || exit 1
fi

scripts/install_model.sh "$PY" >/dev/null
say "starting REAP calibration ($SEQS x $SEQLEN tokens)"
"$PY" scripts/reap_calibrate.py \
    --src "$SRC" --out out/reap_saliency.npz \
    --calib-text out/calib.txt \
    --seqs "$SEQS" --seqlen "$SEQLEN" --batch "$BATCH" 2>&1 | tee "$LOG"
rc="${PIPESTATUS[0]}"
if [ "$rc" -ne 0 ]; then say "CALIB-FAILED rc=$rc"; exit "$rc"; fi
say "CALIB-DONE -> out/reap_saliency.npz"
