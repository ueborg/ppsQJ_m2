#!/bin/bash
# Preflight for a BLOCKED conditional arm.
#
# It refuses while the interlock is armed. That is the point: a
# "preflight everything" sweep must report this arm as BLOCKED, not READY.
# There is no scheduler-submission command in this file.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
RELEASE="../GATE_RELEASED_cond_M128_nc2048"
if [ ! -f "$RELEASE" ]; then
    echo "BLOCKED  cond_M128_nc2048"
    echo "  gate: CAMPAIGN D ADJUDICATION -- STRONGLY GATED"
    echo "  Release ONLY if campaign D's N_c = 2048 rung PASSES the frozen adequacy screen at L = 128. If it fails, the conditional N_c = 4096 central rung comes first and this arm stays blocked. An adequate N_c must be identified BEFORE a 9-point scan at this L is run at all."
    echo
    echo "  This arm is prepared and validated but is NOT ready for submission."
    echo "  See ../CONDITIONAL_SUBMISSION.md."
    exit 3
fi
PY=${PYTHON:-python3}
mkdir -p results
exec "$PY" preflight.py
