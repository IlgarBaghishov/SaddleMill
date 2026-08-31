#!/bin/bash
#SBATCH -N 4                       # size to ceil(n_structures / jobs_per_node)
#SBATCH --ntasks-per-node=128
#SBATCH -p wholenode
#SBATCH -A che190010
#SBATCH -t 4-00:00:00
#SBATCH -J bdma_example
#SBATCH -o slurm_%j.log
#SBATCH --requeue
#SBATCH --signal=B:USR1@900
set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
echo "=== round $(cat .round 2>/dev/null || echo 0) | job $SLURM_JOB_ID | $(date) ==="
MAX_ROUNDS=${MAX_ROUNDS:-4}
ROUND=$(cat .round 2>/dev/null || echo 0)

source $HOME/miniforge3/etc/profile.d/conda.sh
conda activate saddlemill
export VASP_PP_PATH=$HOME/vasp_pp
export PATH=/anvil/scratch/x-sjung3/genTS/vasp_cgfix/bin:$PATH
export PYTHONPATH=/anvil/scratch/x-sjung3/genTS/SaddleMill:${PYTHONPATH:-}
export OMP_NUM_THREADS=1
which vasp_std
# vasp-interactive must carry the version-regex fix (wiki-URL clobber kills VI)
python -c 'import vasp_interactive,os;s=open(os.path.join(os.path.dirname(vasp_interactive.__file__),"vasp_interactive.py")).read();print("FORK-OK",("POSITIONS AND LATTICE" in s))'

on_wall () {
  if [ "$ROUND" -ge "$MAX_ROUNDS" ]; then
    echo "WALL reached at round cap $MAX_ROUNDS -> STOPPING. $(date)"; touch .capped; return
  fi
  echo $((ROUND + 1)) > .round
  echo "WALL reached, work outstanding -> requeue as round $((ROUND + 1)). $(date)"
  scontrol requeue "$SLURM_JOB_ID"
}
trap 'echo "USR1: wall approaching"; on_wall' USR1

srun -N $SLURM_NNODES -n $SLURM_NNODES -c 128 --cpu-bind=none --mpi=pmi2 \
  flux start bash -c 'flux resource list; python -u -m saddlemill' &
wait $!
echo "saddlemill exited rc=$? at $(date)"
