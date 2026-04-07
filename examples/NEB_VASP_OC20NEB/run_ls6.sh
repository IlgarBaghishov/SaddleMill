#!/bin/bash
##SBATCH -N 64
##SBATCH --ntasks-per-node=128
##SBATCH -p normal
##SBATCH -t 48:00:00
#SBATCH -N 2
#SBATCH --ntasks-per-node=128
#SBATCH -p development
#SBATCH -t 02:00:00
#SBATCH -A CHE23004
#SBATCH -J neb_oc20neb

pwd; hostname -f; date

CONDA_BASE=$(dirname $(dirname $CONDA_EXE))
source $CONDA_BASE/etc/profile.d/conda.sh
conda activate tsearch

ml unload xalt python3
ml load impi cuda/12.8

srun -N $SLURM_NNODES -n $SLURM_NNODES --mpi=pmi2 flux start python -u -m tsearch

date
