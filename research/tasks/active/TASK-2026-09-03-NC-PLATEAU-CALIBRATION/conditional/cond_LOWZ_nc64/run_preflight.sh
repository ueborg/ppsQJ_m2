#!/bin/bash
# Preflight for a BLOCKED conditional arm.
#
# It refuses while the interlock is armed. That is the point: a
# "preflight everything" sweep must report this arm as BLOCKED, not READY.
# There is no scheduler-submission command in this file.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
RELEASE="../GATE_RELEASED_cond_LOWZ_nc64"
if [ ! -f "$RELEASE" ]; then
    echo "BLOCKED  cond_LOWZ_nc64"
    echo "  gate: OPTIONAL -- NOT PART OF THE zeta = 0.35 CALIBRATION"
    echo "  Design 2 of TASK-2026-09-03-FINITE-NC-REQUIRED-SCALING. It is deliberately NOT in the immediate group: the programme wants the zeta = 0.35 calibration understood before it spends anything on a second zeta. Release only as an explicit decision to buy that one test now."
    echo
    echo "  This arm is prepared and validated but is NOT ready for submission."
    echo "  See ../CONDITIONAL_SUBMISSION.md."
    exit 3
fi
PY=${PYTHON:-python3}
mkdir -p results
exec "$PY" preflight.py
