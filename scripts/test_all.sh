#!/bin/bash
# Register the current kimi_k3.py into mlx-lm, then run every suite.
# The register step matters: tests import `mlx_lm.models.kimi_k3`, so editing
# the local file without reinstalling silently tests the previous version.
set -uo pipefail
cd "$(dirname "$0")/.."
PY="${PY:-python3}"
scripts/install_model.sh "$PY" >/dev/null
fail=0
for t in tests/test_*.py; do
  printf "%-34s " "$(basename "$t")"
  out=$("$PY" -W ignore "$t" 2>&1 | grep -E "^(OK|FAILED)|^Ran" | tr '\n' ' ')
  echo "$out"
  case "$out" in *FAILED*) fail=1;; esac
done
[ $fail -eq 0 ] && echo "ALL PASS" || echo "FAILURES"
exit $fail
