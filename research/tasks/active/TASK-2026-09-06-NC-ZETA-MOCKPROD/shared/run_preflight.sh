#!/bin/bash
# Run the preflight on every arm of TASK-2026-09-06-NC-ZETA-MOCKPROD.
# Read-only. Contains no submission command and cannot submit anything.
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
TASK="$(cd "$HERE/.." && pwd)"
PY="${PPSQJ_PYTHON:-$TASK/../../../../.venv/bin/python3}"
rc=0
for arm in "$TASK"/M_z*_nc* "$TASK"/E_dtau_* "$TASK"/conditional/M_z*_nc*; do
    [ -f "$arm/manifest.csv" ] || continue
    "$PY" "$HERE/preflight.py" "$arm" || rc=1
done
echo
if [ $rc -eq 0 ]; then
    echo "ALL ARMS PASS PREFLIGHT."
else
    echo "AT LEAST ONE ARM FAILED PREFLIGHT."
fi
echo "Nothing was submitted. research/RESOURCE_POLICY.md section 4."
exit $rc
