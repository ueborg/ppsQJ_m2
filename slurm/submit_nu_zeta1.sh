#!/bin/bash
# =============================================================================
# submit_nu_zeta1.sh -- zeta=1 (Born) FSS campaign to extract a clean(ish) nu.
#
# Reuses the VALIDATED worker_opdim_pps + submit_opdim.sh (zeta=1 => NO cloning,
# plain Born quantum-jump trajectories; records B_L crossing observable + S_half
# + the C_sc ingredients, so this ONE campaign also feeds the Delta_B opdim fit).
# nu is a clean single number only at zeta=1 (single Born fixed point); along the
# PPS line it is not -- so this is the only nu campaign worth running.
#
# WHY large L is feasible here but not at zeta<1: no cloning => trajectories are
# independent => embarrassingly parallel (split N_traj across cores), no ESS
# collapse, no N_c-ladder bias. So L up to 192 is reachable.
#
# THE T-LEVER (clean version): cost ~ N_traj * T * L^4 with T = T_MULT * L.
# opdim's default T_MULT=3 is conservative (entanglement saturates ~ballistic
# ~L/v). This script defaults T_MULT=2.0 (a ~1.5x saving) and PRINTS a saturation
# check. DO NOT drop below ~1.5 without confirming B_L is stable at lambda_c.
#
# WORKFLOW (two phases):
#   1) MODE=calib bash slurm/submit_nu_zeta1.sh
#      -> runs ONE cheap point (L=64, lam=0.50, few traj) to a calib dir.
#   2) (after calib finishes) MODE=prod bash slurm/submit_nu_zeta1.sh
#      -> reads the calib wall_time, sizes per-L walltimes from the L^5 cost
#         model, submits one array per L. L=192 is flagged if it exceeds 48h.
#   Analyse: python analysis/fit_nu_zeta1.py $SCRATCH/pps_qj/pps_nu_zeta1
#
# Saturation check before trusting prod (cheap, do once):
#   PPS_L_LIST=128 PPS_LAM_LIST=0.50 PPS_N_TRAJ=200 ARRAY=0-0 CPUS=24 \
#     PPS_T_MULT=3.0 OUTBASE=$SCRATCH/pps_qj/pps_sat3 bash slurm/submit_opdim.sh
#   ... and the same at PPS_T_MULT=2.0 ; confirm B_L_mean agrees within error.
# =============================================================================
set -euo pipefail

SCRATCH=/scratch/${USER}/pps_qj
OUT_PROD=${OUTBASE:-$SCRATCH/pps_nu_zeta1}
OUT_CALIB=$SCRATCH/pps_nu_zeta1_calib
HERE=$(cd "$(dirname "$0")" && pwd)
SUBMIT_OPDIM="$HERE/submit_opdim.sh"

MODE=${MODE:-prod}

# ---- campaign knobs ----
L_LIST=${PPS_L_LIST:-"64,96,128,160,192"}
LAM_LIST=${PPS_LAM_LIST:-"0.44,0.46,0.48,0.49,0.50,0.51,0.52,0.54,0.56"}
N_TRAJ=${PPS_N_TRAJ:-1500}
T_MULT=${PPS_T_MULT:-2.0}
SEED0=${PPS_SEED0:-20260620}            # distinct from the Delta_B opdim run (20260607)
PARTITION=${PARTITION:-"regular,parallel"}
MARGIN=${MARGIN:-1.6}                   # walltime safety factor
CALIB_NTRAJ=${CALIB_NTRAJ:-64}
CALIB_CPUS=${CALIB_CPUS:-16}

NLAM=$(echo "$LAM_LIST" | awk -F, '{print NF}')

# per-L cores: more cores at large L so N_traj parallelises (wall ~ N_traj/CPUS * t_traj)
cpus_for () { case "$1" in 64) echo 16;; 96) echo 24;; 128) echo 32;; 160) echo 48;; 192) echo 64;; *) echo 24;; esac; }

if [ "$MODE" = "calib" ]; then
  echo "[calib] L=64 lam=0.50 N_traj=$CALIB_NTRAJ T_mult=$T_MULT -> $OUT_CALIB"
  OUTBASE="$OUT_CALIB" PPS_L_LIST="64" PPS_LAM_LIST="0.50" PPS_N_TRAJ="$CALIB_NTRAJ" \
    PPS_T_MULT="$T_MULT" PPS_SEED0="$SEED0" PPS_FORCE_RERUN=1 \
    ARRAY="0-0" WALL="00:40:00" CPUS="$CALIB_CPUS" CONC=1 PARTITION="$PARTITION" \
    bash "$SUBMIT_OPDIM"
  echo "[calib] when it finishes: MODE=prod bash slurm/submit_nu_zeta1.sh"
  exit 0
fi

# ---- prod: size per-L walls from the calib wall_time ----
CALIB_WALL_S=$(./.venv/bin/python - "$OUT_CALIB" 2>/dev/null <<PY || true
import glob,sys,numpy as np
fs=sorted(glob.glob(sys.argv[1]+"/opdim_*.npz"))
print(float(np.load(fs[0])["wall_time"]) if fs else 0.0)
PY
)
CALIB_WALL_S=${CALIB_WALL_S:-0}
if ! awk "BEGIN{exit !($CALIB_WALL_S>0)}"; then
  echo "ERROR: no calib wall_time in $OUT_CALIB. Run 'MODE=calib bash slurm/submit_nu_zeta1.sh' first" >&2
  echo "(or pass CALIB_WALL_S=<seconds at L=64,N_traj=$CALIB_NTRAJ,CPUS=$CALIB_CPUS> explicitly)." >&2
  [ "${CALIB_WALL_S_OVERRIDE:-}" = "" ] && exit 1
fi
echo "calib anchor: L=64 N_traj=$CALIB_NTRAJ CPUS=$CALIB_CPUS T_mult=$T_MULT -> ${CALIB_WALL_S}s"
echo "grid: L={$L_LIST} x lam={$LAM_LIST} ($NLAM lam) ; N_traj=$N_TRAJ T_mult=$T_MULT seed0=$SEED0"
echo

IFS=',' read -ra LS <<< "$L_LIST"
Li=0
for L in "${LS[@]}"; do
  cpus=$(cpus_for "$L")
  # wall_s = calib * (N/Ncal) * (L/64)^5 * (CPUScal/cpus) * MARGIN   [T_mult equal both phases]
  read WALL OVER < <(./.venv/bin/python - "$CALIB_WALL_S" "$N_TRAJ" "$CALIB_NTRAJ" "$L" "$CALIB_CPUS" "$cpus" "$MARGIN" <<'PY'
import sys
cw,N,Nc,L,Ccal,cpus,margin=[float(x) for x in sys.argv[1:8]]
s=cw*(N/Nc)*((L/64.0)**5)*(Ccal/cpus)*margin
h=int(s//3600); m=int((s%3600)//60)
hh=min(h,72)
over=1 if s>48*3600 else 0
print(f"{hh:02d}:{m:02d}:00 {over}")
PY
)
  a0=$(( Li * NLAM )); a1=$(( Li*NLAM + NLAM - 1 ))
  flag=""; [ "$OVER" = "1" ] && flag="  [>48h: consider dropping L=$L or raising cpus]"
  echo "L=$L: array $a0-$a1  cpus=$cpus  wall=$WALL$flag"
  OUTBASE="$OUT_PROD" PPS_L_LIST="$L_LIST" PPS_LAM_LIST="$LAM_LIST" PPS_N_TRAJ="$N_TRAJ" \
    PPS_T_MULT="$T_MULT" PPS_SEED0="$SEED0" PPS_FORCE_RERUN="${PPS_FORCE_RERUN:-0}" \
    ARRAY="${a0}-${a1}" WALL="$WALL" CPUS="$cpus" CONC="$NLAM" PARTITION="$PARTITION" \
    bash "$SUBMIT_OPDIM"
  Li=$(( Li + 1 ))
done
echo
echo "submitted. analyse with: python analysis/fit_nu_zeta1.py $OUT_PROD"
