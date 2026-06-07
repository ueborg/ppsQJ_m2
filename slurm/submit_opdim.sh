#!/bin/bash
# =============================================================================
# submit_opdim.sh -- measure Delta_B(lambda_c, zeta=1) + Jian/Foster multifractal
# exponents for the QJ Case-B Kitaev MIPT.
#
# zeta=1  =>  PPS weight = 1  =>  plain Born-rule quantum-jump trajectories,
# NO cloning / population dynamics.  One array task per (L, lam); each task
# samples N_traj trajectories in parallel across CPUS cores (single-thread BLAS).
#
# Default grid: L in {64,96,128} x lam in {0.46,0.48,0.50,0.52,0.54} = 15 tasks,
# bracketing lambda_c(zeta=1) ~ 0.5 so the fit also pins lambda_c as a byproduct.
#
# Output: $SCRATCH/pps_opdim/opdim_<task>.npz   (analyse with analysis/fit_opdim.py)
#
# Usage:
#   bash slurm/submit_opdim.sh                       # full grid, defaults
#   PPS_N_TRAJ=4000 CPUS=32 bash slurm/submit_opdim.sh
#   # cheap calibration first (1 point, few trajectories -> timing + saturation):
#   PPS_L_LIST=128 PPS_LAM_LIST=0.50 PPS_N_TRAJ=64 ARRAY=0-0 WALL=00:20:00 \
#       CPUS=16 bash slurm/submit_opdim.sh
#   # then a saturation sanity check by re-running that point at PPS_T_MULT=2 and 4
#   # and confirming Delta_B / S_half are stable.
# =============================================================================
set -euo pipefail

SCRATCH=/scratch/${USER}/pps_qj
OUTPUT_DIR=${OUTBASE:-$SCRATCH/pps_opdim}
LOG_DIR=$SCRATCH/logs
mkdir -p "$OUTPUT_DIR" "$LOG_DIR"

CPUS=${CPUS:-24}
CONC=${CONC:-15}
WALL=${WALL:-02:00:00}
PARTITION=${PARTITION:-"regular,parallel"}

PPS_L_LIST=${PPS_L_LIST:-"64,96,128"}
PPS_LAM_LIST=${PPS_LAM_LIST:-"0.46,0.48,0.50,0.52,0.54"}
PPS_N_TRAJ=${PPS_N_TRAJ:-2000}
PPS_T_MULT=${PPS_T_MULT:-3.0}
PPS_BULK_FRAC=${PPS_BULK_FRAC:-0.25}
PPS_SEED0=${PPS_SEED0:-20260607}

NL=$(echo "$PPS_L_LIST"   | awk -F, '{print NF}')
NLAM=$(echo "$PPS_LAM_LIST" | awk -F, '{print NF}')
NTASK=$(( NL * NLAM ))
ARRAY=${ARRAY:-0-$(( NTASK - 1 ))}

echo "Grid: L={$PPS_L_LIST} x lam={$PPS_LAM_LIST}  ->  $NTASK tasks"
echo "array=$ARRAY%$CONC  N_traj=$PPS_N_TRAJ  T_mult=$PPS_T_MULT  cpus=$CPUS  wall=$WALL"
echo "out=$OUTPUT_DIR"

sbatch <<SLURM
#!/bin/bash
#SBATCH --job-name=opdim
#SBATCH --array=${ARRAY}%${CONC}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=${CPUS}
#SBATCH --time=${WALL}
#SBATCH --partition=${PARTITION}
#SBATCH --output=${LOG_DIR}/opdim_%A_%a.out
#SBATCH --error=${LOG_DIR}/opdim_%A_%a.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=${USER}@rug.nl
set -euo pipefail

module purge
source \${HOME}/venvs/pps_qj/bin/activate
cd \${HOME}/pps_qj

export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PPS_N_WORKERS=${CPUS}
export PPS_L_LIST="${PPS_L_LIST}"
export PPS_LAM_LIST="${PPS_LAM_LIST}"
export PPS_N_TRAJ=${PPS_N_TRAJ}
export PPS_T_MULT=${PPS_T_MULT}
export PPS_BULK_FRAC=${PPS_BULK_FRAC}
export PPS_SEED0=${PPS_SEED0}
export PPS_FORCE_RERUN=\${PPS_FORCE_RERUN:-0}

echo "[\$(date +%H:%M:%S)] opdim task \${SLURM_ARRAY_TASK_ID} -> ${OUTPUT_DIR}"
python -m pps_qj.parallel.worker_opdim_pps \${SLURM_ARRAY_TASK_ID} ${OUTPUT_DIR}
echo "[\$(date +%H:%M:%S)] done"
SLURM
echo "Submitted opdim array ${ARRAY}%${CONC}."
