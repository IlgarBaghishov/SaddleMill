#!/bin/bash
#SBATCH -N 1
#SBATCH --ntasks-per-node=128
#SBATCH -p development
#SBATCH -t 01:00:00
#SBATCH -A YOUR_ALLOCATION
#SBATCH -J dimer_vasp_oc20

pwd; hostname -f; date

CONDA_BASE=$(dirname $(dirname $CONDA_EXE))
source $CONDA_BASE/etc/profile.d/conda.sh
conda activate saddlemill

ml unload xalt python3
ml load impi cuda/12.8

# Per-attempt VASP_{job}_{attempt}/ scratch dirs are created automatically.
srun -N $SLURM_NNODES -n $SLURM_NNODES --mpi=pmi2 flux start python -u -m saddlemill

date
