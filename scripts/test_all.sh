#!/bin/bash
# Register the current kimi_k3.py into mlx-lm, then run every suite.
# The register step matters: tests import `mlx_lm.models.kimi_k3`, so editing
# the local file without reinstalling silently tests the previous version.
set -uo pipefail
cd "$(dirname "$0")/.."
PY="${PY:-python3}"
scripts/install_model.sh "$PY" >/dev/null
# A suite that produces no result must FAIL, not pass quietly. This used to
# grep for ^Ran/^OK/^FAILED and test only for the word FAILED, so anything that
# emitted none of them -- a suite that died on import, or one that is not
# unittest-based -- contributed no output, tripped no check, and still counted
# toward "ALL PASS". The interpreter's exit code was discarded through the pipe,
# which is the signal that actually matters, so that is now the primary check.
fail=0
for t in tests/test_*.py; do
  printf "%-34s " "$(basename "$t")"
  raw=$("$PY" -W ignore "$t" 2>&1); rc=$?
  # unittest prints Ran/OK/FAILED. Suites that are plain scripts rather than
  # unittest (the mlx.launch pipeline checks) report a shouted verdict line such
  # as "PIPELINE PARITY OK" or "PARAM-KEY CHECK OK"; match that shape generally
  # rather than naming each one, or the next such suite is silently a failure.
  summary=$(printf '%s\n' "$raw" \
      | grep -E "^(OK|FAILED)|^Ran |^[A-Z][A-Z0-9 _-]*(OK|FAIL)$" | tr '\n' ' ')
  if [ $rc -ne 0 ]; then
    echo "${summary:-(no output)} <- EXIT $rc"
    printf '%s\n' "$raw" | tail -3 | sed 's/^/      /'
    fail=1
  elif printf '%s' "$summary" | grep -q "Ran 0 tests"; then
    echo "$summary <- ran 0 tests"
    fail=1
  elif [ -z "$summary" ]; then
    echo "(no test summary) <- exited 0 but reported nothing"
    fail=1
  else
    echo "$summary"
  fi
done
[ $fail -eq 0 ] && echo "ALL PASS" || echo "FAILURES"
exit $fail
