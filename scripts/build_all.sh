#!/bin/bash
# Build every Kimi-K3 MLX tier, sequentially, with per-tier logs and status files.
#
# Order matters: mxfp4 first because it is a pure byte copy (fastest, and it
# validates the whole pipeline against the real checkpoint before we spend hours
# on the lossy tiers).
#
# Usage: scripts/build_all.sh [profile ...]      (default: all four)
set -uo pipefail
cd "$(dirname "$0")/.."
PY="${PY:-python3}"
SRC="${SRC:-$PWD/Kimi-K3-src}"
OUTDIR="${OUTDIR:-$PWD/out}"
PROFILES=("$@")
[ ${#PROFILES[@]} -eq 0 ] && PROFILES=(mxfp4 3bit mixed2 2bit)

mkdir -p "$OUTDIR"
scripts/install_model.sh "$PY"

say() { echo "[build $(date +%H:%M:%S)] $*"; }

for p in "${PROFILES[@]}"; do
  OUT="$OUTDIR/Kimi-K3-MLX-$p"
  STATUS="$OUTDIR/build_$p.status"
  say "=== $p -> $OUT"
  rm -rf "$OUT"
  "$PY" scripts/convert.py --src "$SRC" --out "$OUT" --profile "$p" \
      2>&1 | tee "$OUTDIR/convert_$p.log"
  if [ "${PIPESTATUS[0]}" -ne 0 ]; then
    echo "FAILED: convert" > "$STATUS"; say "$p FAILED"; continue
  fi
  SIZE=$(du -sh "$OUT" | cut -f1)
  say "$p converted: $SIZE"

  # Structural verification. A real generation check is impossible on this box:
  # even the 2-bit tier is ~870 GB against 512 GB of unified memory.
  "$PY" scripts/verify.py --path "$OUT" --src "$SRC" 2>&1 | tee -a "$OUTDIR/convert_$p.log"
  if [ "${PIPESTATUS[0]}" -ne 0 ]; then
    echo "FAILED: verify measured=$SIZE" > "$STATUS"; say "$p VERIFY FAILED"; continue
  fi
  echo "OK measured=$SIZE" > "$STATUS"
  say "$p DONE -> $STATUS"
done

say "all done"
grep -H . "$OUTDIR"/build_*.status 2>/dev/null || true
