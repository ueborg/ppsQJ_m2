#!/bin/bash
# =============================================================================
# Run C: DIII Renyi drift — large-L sweep at fixed (lambda=0.05) in log phase.
# 2 zeta values × 6 L values = 12 tasks. PPS_RECORD_RENYI=1.
# Measures S_1, S_2, S_3 vs L to test whether c_2/c_1 drifts toward 0.75 (free Dirac).
# Output: $SCRATCH/pps_run_C/
# Test: c_2/c_1 vs L — if converging to 0.75, DIII marginal flow confirmed.
# L=384 requires regularlong partition (72h wall time).
# =============================================================================
set -euo pipefail

SCRATCH=/scratch/${USER}/pps_qj
LOG_DIR=$SCRATCH/logs
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
GENERIC=$SCRIPT_DIR/submit_array_generic.sh
mkdir -p $LOG_DIR

# ── fast ──
cat > $LOG_DIR/submit_run_C_fast_tasks.txt << 'TASKS'
64 0.0500 0.5000
64 0.0500 1.0000
96 0.0500 0.5000
96 0.0500 1.0000
128 0.0500 0.5000
128 0.0500 1.0000
TASKS

bash "$GENERIC" $LOG_DIR/submit_run_C_fast_tasks.txt $SCRATCH/pps_run_C 08:00:00 1 submit_run_C_fast "regular,parallel"

# ── slow ──
cat > $LOG_DIR/submit_run_C_slow_tasks.txt << 'TASKS'
192 0.0500 0.5000
192 0.0500 1.0000
256 0.0500 0.5000
256 0.0500 1.0000
TASKS

bash "$GENERIC" $LOG_DIR/submit_run_C_slow_tasks.txt $SCRATCH/pps_run_C 48:00:00 1 submit_run_C_slow "regular"

# ── vsl ──
cat > $LOG_DIR/submit_run_C_vsl_tasks.txt << 'TASKS'
384 0.0500 0.5000
384 0.0500 1.0000
TASKS

bash "$GENERIC" $LOG_DIR/submit_run_C_vsl_tasks.txt $SCRATCH/pps_run_C 72:00:00 1 submit_run_C_vsl "regularlong"

echo 'All sub-jobs submitted.'
