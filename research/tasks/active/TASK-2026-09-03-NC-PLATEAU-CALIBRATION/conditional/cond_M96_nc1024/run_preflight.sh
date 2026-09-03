#!/bin/bash
# Preflight for a BLOCKED conditional arm.
#
# It refuses while the interlock is armed. That is the point: a
# "preflight everything" sweep must report this arm as BLOCKED, not READY.
# There is no scheduler-submission command in this file.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
RELEASE="../GATE_RELEASED_cond_M96_nc1024"
if [ ! -f "$RELEASE" ]; then
    echo "BLOCKED  cond_M96_nc1024"
    echo "  gate: CAMPAIGN C ADJUDICATION -- AND ONLY ONE OF THE TWO M96 ARMS"
    echo "  Release ONLY if campaign C identifies N_c = 1024 as the smallest N_c meeting the frozen production adequacy criterion at L = 96. If it identifies 2048, release cond_M96_nc2048 INSTEAD. Never both: they are the same physical scan at two population sizes and running both is duplicated compute, not a robustness check."
    echo
    echo "  This arm is prepared and validated but is NOT ready for submission."
    echo "  See ../CONDITIONAL_SUBMISSION.md."
    exit 3
fi
PY=${PYTHON:-python3}
mkdir -p results
exec "$PY" preflight.py
