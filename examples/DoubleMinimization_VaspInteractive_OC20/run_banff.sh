#!/bin/bash
#SBATCH -N 3                       # size to ceil(n_structures / jobs_per_node)
#SBATCH --ntasks-per-node=224
#SBATCH -p amd
##SBATCH -q vip                    # vip ONLY inside the submit-before-cancel handover
#SBATCH -t 7-00:00:00
#SBATCH -J bdm_example
#SBATCH -o slurm_%j.log
#SBATCH --requeue
#SBATCH --signal=B:USR1@900
set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
echo "=== round $(cat .round 2>/dev/null || echo 0) | job $SLURM_JOB_ID | $(date) ==="
MAX_ROUNDS=${MAX_ROUNDS:-2}
ROUND=$(cat .round 2>/dev/null || echo 0)

CONDA_BASE=$(dirname $(dirname $CONDA_EXE))
source $CONDA_BASE/etc/profile.d/conda.sh
conda activate saddlemill
# Explicit: do not rely on ~/.bash_profile propagating through --export=ALL.
export PYTHONPATH=/home/sung/codes/SaddleMill:${PYTHONPATH:-}
export VASP_PP_PATH=/home/graeme/vasp
# vasp_nocb = cg-fixed build WITHOUT -CB (array bounds checking is banned)
export PATH=/home/sung/codes/vasp_nocb/bin:$PATH
export OMP_NUM_THREADS=1
which vasp_std

on_wall () {
  if [ "$ROUND" -ge "$MAX_ROUNDS" ]; then
    echo "WALL reached at round cap $MAX_ROUNDS -> STOPPING. $(date)"; touch .capped; return
  fi
  echo $((ROUND + 1)) > .round
  echo "WALL reached, work outstanding -> requeue as round $((ROUND + 1)). $(date)"
  scontrol requeue "$SLURM_JOB_ID"
}
trap 'echo "USR1: wall approaching"; on_wall' USR1

srun -N $SLURM_NNODES -n $SLURM_NNODES --mpi=pmi2 flux start \
    env -u PMI2_FD -u PMI_FD -u PMI2_RANK -u PMI_RANK -u PMI2_SIZE -u PMI_SIZE -u PMI2_SPROUTE \
    python -u -m saddlemill &
wait $!
echo "saddlemill exited rc=$? at $(date)"
