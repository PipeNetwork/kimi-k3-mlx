#!/bin/bash
# Plan -> prune -> convert -> verify, for a REAP'd Kimi-K3 tier.
#
# The 4-bit-class tier is mxfp4, not affine 4-bit: K3's experts ship as MXFP4 and
# MLX's mxfp4 mode is the same encoding, so mxfp4 is a bit-exact byte copy at
# 4.25 bpw while affine 4-bit is lossy AND larger at 4.5 bpw.
#
# Usage: scripts/build_reap.sh [KEEP_FRACTION] [MODE] [PROFILE]
#   scripts/build_reap.sh 0.27 uniform mxfp4
set -uo pipefail
cd "$(dirname "$0")/.."
PY="${PY:-python3}"
SRC="${SRC:-$PWD/Kimi-K3-src}"
KEEP="${1:-0.27}"
MODE="${2:-uniform}"
PROFILE="${3:-mxfp4}"
SAL="$PWD/out/reap_saliency.npz"
PLAN="$PWD/out/reap_plan_${MODE}_${KEEP}.json"
PCT=$($PY -c "print(f'{(1-$KEEP)*100:.0f}')")
OUT="$PWD/out/Kimi-K3-REAP${PCT}-MLX-${PROFILE}"

say() { echo "[reap-build $(date +%H:%M:%S)] $*"; }

[ -f "$SAL" ] || { say "no saliency at $SAL"; exit 1; }

say "planning: mode=$MODE keep=$KEEP (REAP-$PCT)"
"$PY" scripts/reap_plan.py --saliency "$SAL" --mode "$MODE" --keep "$KEEP" \
    --out "$PLAN" 2>&1 | tee out/reap_plan.log || exit 1

say "converting -> $OUT (profile=$PROFILE)"
rm -rf "$OUT"
"$PY" scripts/convert.py --src "$SRC" --out "$OUT" --profile "$PROFILE" \
    --prune-plan "$PLAN" 2>&1 | tee out/convert_reap.log
rc="${PIPESTATUS[0]}"
[ "$rc" -eq 0 ] || { say "BUILD-FAILED convert rc=$rc"; exit "$rc"; }
say "converted: $(du -sh "$OUT" | cut -f1)"

say "verifying"
"$PY" scripts/verify.py --path "$OUT" --src "$SRC" --samples 16 2>&1 | tee out/verify_reap.log
rc="${PIPESTATUS[0]}"
[ "$rc" -eq 0 ] || { say "BUILD-FAILED verify rc=$rc"; exit "$rc"; }

say "BUILD-DONE $OUT ($(du -sh "$OUT" | cut -f1))"
