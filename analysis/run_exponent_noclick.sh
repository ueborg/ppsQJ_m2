#!/bin/bash
#SBATCH --job-name=noclick_exp
#SBATCH --output=noclick_exp_%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --partition=regular          # adjust to your Habrok partition

module purge
source ~/venvs/pps_qj/bin/activate
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK

cd "$SLURM_SUBMIT_DIR"
python exponent_noclick.py
