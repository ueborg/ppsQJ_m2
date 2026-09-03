#!/bin/bash
# Preflight for a BLOCKED conditional arm.
#
# It refuses while the interlock is armed. That is the point: a
# "preflight everything" sweep must report this arm as BLOCKED, not READY.
# There is no scheduler-submission command in this file.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
RELEASE="../GATE_RELEASED_cond_LOWZ_nc256"
if [ ! -f "$RELEASE" ]; then
    echo "BLOCKED  cond_LOWZ_nc256"
    echo "  gate: OPTIONAL -- NOT PART OF THE zeta = 0.35 CALIBRATION"
    echo "  As cond_LOWZ_nc64. The pre-registered kill criterion needs BOTH population sizes: drift at zeta = 0.10 greater than or equal to drift at zeta = 0.35 kills the guided-residual mechanism and revives Born-rarity reasoning. Release both or neither."
    echo
    echo "  This arm is prepared and validated but is NOT ready for submission."
    echo "  See ../CONDITIONAL_SUBMISSION.md."
    exit 3
fi
PY=${PYTHON:-python3}
mkdir -p results
exec "$PY" preflight.py
