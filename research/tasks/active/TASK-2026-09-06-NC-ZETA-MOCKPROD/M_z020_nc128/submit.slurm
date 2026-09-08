#!/bin/bash
#SBATCH --job-name=mockprod-m_z020_nc128
#SBATCH --partition=cpu_med
#SBATCH --array=0-46%64
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=00:45:00
#SBATCH --output=logs/M_z020_nc128_%A_%a.out
#SBATCH --error=logs/M_z020_nc128_%A_%a.err
#
# ============================================================================
# M_z020_nc128 -- TASK-2026-09-06-NC-ZETA-MOCKPROD
#
# NOT SUBMITTED BY ANY AGENT. research/RESOURCE_POLICY.md section 4 forbids it
# unconditionally, at every stage, gate and approval level. The researcher types
# the submission command by hand. Neither this file nor the preflight contains
# one.
#
# PURPOSE
#   zeta = 0.20, N_c = 128: the full 9-point lambda grid at L = 32, 48, 64 (T = L), R = 16 independent populations per cell. One rung of the N_c ladder.
#
# # Committed arm. Submit per ../HUMAN_SUBMISSION.md.
#
# SCALE
#   432 populations packed into 47 array tasks so that no array task is
#   shorter than 600 s. packs.csv records the (start, count) of every pack.
#   ../shared/run_pack.py invokes ./run_cell.py -- the ARM-LOCAL copy, in this
#   directory -- once per row in a fresh process: packing changes the scheduling
#   and never the computation, and a row produced by a packed task is
#   EXACT-COMPATIBLE with one produced unpacked.
#
#   WHY THE EXECUTOR IS HERE AND NOT IN ../shared. run_cell.py takes no manifest
#   argument: it reads manifest.csv from the directory ITS OWN FILE sits in and
#   writes to results/ beside it. An executor invoked out of ../shared therefore
#   reads ../shared/manifest.csv, which does not exist, and every row dies with
#   FileNotFoundError before the sampler is reached. That is what happened to
#   this task's first Ruche submission (job IDs 1694328-1694621, zero result
#   JSONs, arrays cancelled -- see ../RUCHE_INCIDENT_2026-09-08.md). This arm's
#   run_cell.py is a BYTE-IDENTICAL copy of the frozen ../shared/run_cell.py,
#   written by tools/build_arms.py and re-verified at job start by run_pack.py.
#   Preflight P18/P19/P20 prove it; negative controls N15 and N16 break it
#   and require rejection, and tools/negative_controls.py reproduces the
#   original failure end to end before showing the repair.
#
#   The path to run_pack.py is ../shared, computed per arm rather than hard-coded:
#   the conditional arm lives one directory deeper and a literal ../shared would
#   not resolve there. Preflight P17 checks that it resolves; negative control
#   N14 breaks it and requires rejection.
#   Idempotent -- a completed row is never recomputed, so a requeued or
#   timed-out pack costs only what it has left to do.
#
# COST -- fitted to wall_s recorded by completed Ruche jobs of this code path.
#   8.7 core-hours (12.2 pessimistic at x1.40).
#   slowest single population 3.2 min; slowest pack 12.9 min
#   (18.0 min pessimistic); elapsed ~0.25 h at %64,
#   EXCLUDING queue wait, which is expected to dominate the short arms.
#
#   rate = rate35(L, N_c) * small_batch(N_c) * rho(zeta) * f_lam(lambda, zeta).
#   rate35 is Ruche-MEASURED and FLAT in N_c: the N_c^0.1871 growth law adopted
#   by TASK-2026-09-03-NC-PLATEAU-CALIBRATION is refuted by that campaign's own
#   returned rungs (L=64 N_c=4096/8192 measured 5.05/5.11 ms against 6.6/7.5
#   predicted; L=128 N_c=2048 measured 22.15 against 31.7). See ../COST_MODEL.md.
#
# --time is >= 1.6 x the PESSIMISTIC slowest PACK, snapped up. Partition is then
#   chosen to fit the time: cpu_med if it fits the 4 h MaxTime, else cpu_long.
#   cpu_short is NEVER used -- it is effectively serialised for this account by
#   QOSMaxJobsPerUserLimit, so its %N cap is not real
#   (TASK-2026-09-03-MOCK-LOWLAMBDA-EXTENSION/SCHEDULER_DECISION.md).
#
# --mem=1G against a modelled peak of 283 MB (1.45 x the inherited
#   formula; the formula alone under-predicts real peak RSS by up to 1.6x,
#   measured at 15 cells). zeta does not enter memory.
#
# --array=0-46 : ONE task per pack, exactly. preflight.py fails if this
#   stops matching packs.csv. %64 is a concurrency cap only.
# ============================================================================

set -euo pipefail

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs results

# run_cell.py defaults the repository root to five levels above ITS OWN file,
# which is right for an arm directly under the task directory and wrong for the
# conditional arm one level deeper. Derive it from this arm's own depth instead
# of relying on that default, and prove it before the array does any work.
PPSQJ_REPO="${PPSQJ_REPO:-$(cd "$SLURM_SUBMIT_DIR"/../../../../.. && pwd)}"
if [ ! -f "$PPSQJ_REPO/pps_qj/__init__.py" ]; then
    echo "PPSQJ_REPO does not contain the pps_qj package: $PPSQJ_REPO" >&2
    exit 2
fi
export PPSQJ_REPO

if [ ! -f "$SLURM_SUBMIT_DIR/run_cell.py" ]; then
    echo "no arm-local run_cell.py in $SLURM_SUBMIT_DIR -- rebuild the arms with" >&2
    echo "  tools/build_arms.py  (see ../RUCHE_INCIDENT_2026-09-08.md)" >&2
    exit 2
fi

PPSQJ_PYTHON="${PPSQJ_PYTHON:-/gpfs/workdir/ercetinut/envs/pps_qj/bin/python}"
if [ ! -x "$PPSQJ_PYTHON" ]; then
    echo "PPSQJ_PYTHON is not executable: $PPSQJ_PYTHON" >&2
    echo "Export PPSQJ_PYTHON to the validated interpreter, e.g." >&2
    echo "  export PPSQJ_PYTHON=\$WORKDIR/envs/pps_qj/bin/python" >&2
    exit 2
fi
export PATH="$(dirname "$PPSQJ_PYTHON"):$PATH"

echo "[M_z020_nc128] task ${SLURM_ARRAY_TASK_ID} on $(hostname) at $(date -u +%FT%TZ)"
echo "[M_z020_nc128] partition=${SLURM_JOB_PARTITION:-?}  python=$PPSQJ_PYTHON"
echo "[M_z020_nc128] repo=$PPSQJ_REPO  executor=$SLURM_SUBMIT_DIR/run_cell.py"
"$PPSQJ_PYTHON" -c 'import sys,numpy;print("[M_z020_nc128] resolved",sys.executable,"numpy",numpy.__version__)'

"$PPSQJ_PYTHON" ../shared/run_pack.py "${SLURM_ARRAY_TASK_ID}"
echo "[M_z020_nc128] task ${SLURM_ARRAY_TASK_ID} done at $(date -u +%FT%TZ)"
