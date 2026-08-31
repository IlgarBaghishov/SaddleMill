#!/bin/bash
#SBATCH -N 6                       # 1 side per 24-core node (jobs_per_node = 1)
#SBATCH --ntasks-per-node=24
#SBATCH -p intel
#SBATCH -J bdmfri_example
#SBATCH -o slurm_%j.log
set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
source ~/miniforge3/etc/profile.d/conda.sh
conda activate saddlemill
export VASP_PP_PATH=$HOME/vasp_pp
export PATH=$HOME/vasp_cgfix_fri/bin:$PATH
export PYTHONPATH=$HOME/SaddleMill:${PYTHONPATH:-}
export OMP_NUM_THREADS=1
which vasp_std
# fri has infinite wall, so no requeue/round machinery is needed here.
srun -N $SLURM_NNODES -n $SLURM_NNODES --mpi=pmi2 flux start \
    env -u PMI2_FD -u PMI_FD -u PMI2_RANK -u PMI_RANK -u PMI2_SIZE -u PMI_SIZE -u PMI2_SPROUTE \
    python -u -m saddlemill
echo "saddlemill exited rc=$? at $(date)"
