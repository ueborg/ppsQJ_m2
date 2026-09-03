#!/bin/bash
# Preflight for a BLOCKED conditional arm.
#
# It refuses while the interlock is armed. That is the point: a
# "preflight everything" sweep must report this arm as BLOCKED, not READY.
# There is no scheduler-submission command in this file.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
RELEASE="../GATE_RELEASED_cond_M128_nc4096"
if [ ! -f "$RELEASE" ]; then
    echo "BLOCKED  cond_M128_nc4096"
    echo "  gate: CAMPAIGN D AND cond_D2_L128_nc4096 ADJUDICATION"
    echo "  Release ONLY if N_c = 2048 FAILS the adequacy screen at L = 128 and the conditional N_c = 4096 central rung then PASSES it. Read the core-hour line before releasing: this is the most expensive object in the whole campaign by a wide margin and it should not be the first way the programme learns that L = 128 is unaffordable."
    echo
    echo "  This arm is prepared and validated but is NOT ready for submission."
    echo "  See ../CONDITIONAL_SUBMISSION.md."
    exit 3
fi
PY=${PYTHON:-python3}
mkdir -p results
exec "$PY" preflight.py
