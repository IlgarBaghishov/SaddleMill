#!/bin/bash
#SBATCH -N 1
#SBATCH --ntasks-per-node=128
#SBATCH -p development
#SBATCH -t 01:00:00
#SBATCH -A YOUR_ALLOCATION
#SBATCH -J min_vasp_oc20

pwd; hostname -f; date

CONDA_BASE=$(dirname $(dirname $CONDA_EXE))
source $CONDA_BASE/etc/profile.d/conda.sh
conda activate saddlemill

ml unload xalt python3
ml load impi cuda/12.8

# executorlib=True in config.ini → SaddleMill uses FluxJobExecutor;
# the vasp_command spawns 64 MPI ranks per call.
srun -N $SLURM_NNODES -n $SLURM_NNODES --mpi=pmi2 flux start python -u -m saddlemill

date
