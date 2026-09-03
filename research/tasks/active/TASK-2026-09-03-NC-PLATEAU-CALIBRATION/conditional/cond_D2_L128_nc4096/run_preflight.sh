#!/bin/bash
# Preflight for a BLOCKED conditional arm.
#
# It refuses while the interlock is armed. That is the point: a
# "preflight everything" sweep must report this arm as BLOCKED, not READY.
# There is no scheduler-submission command in this file.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
RELEASE="../GATE_RELEASED_cond_D2_L128_nc4096"
if [ ! -f "$RELEASE" ]; then
    echo "BLOCKED  cond_D2_L128_nc4096"
    echo "  gate: CAMPAIGN D ADJUDICATION"
    echo "  Recommend this arm if, on the L = 128 ladder completed by campaign D, EITHER |Delta_1024| = |I_2048 - I_1024| is resolved OUTSIDE the frozen material tolerance tau_I = 0.006 (i.e. the 95 % interval excludes [-tau_I, +tau_I]), OR no plateau criterion P1-P5 of ../SUCCESS_CRITERIA.yaml is satisfied at the top of that ladder. Do NOT recommend it because the observed Delta_1024 'looks large'."
    echo
    echo "  This arm is prepared and validated but is NOT ready for submission."
    echo "  See ../CONDITIONAL_SUBMISSION.md."
    exit 3
fi
PY=${PYTHON:-python3}
mkdir -p results
exec "$PY" preflight.py
