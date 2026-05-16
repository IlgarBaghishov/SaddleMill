#!/bin/bash
#SBATCH -N 1
#SBATCH --ntasks-per-node=128
#SBATCH -p development
#SBATCH -t 01:00:00
#SBATCH -A YOUR_ALLOCATION
#SBATCH -J dm_vasp_oc20

pwd; hostname -f; date

CONDA_BASE=$(dirname $(dirname $CONDA_EXE))
source $CONDA_BASE/etc/profile.d/conda.sh
conda activate saddlemill

ml unload xalt python3
ml load impi cuda/12.8

# DM creates three VASP_{job}_{side}/ scratch dirs (-1=min1, 0=ts, 1=min2).
# Sides run sequentially in the current implementation; each side's VASP call
# uses the full 64-core socket via vasp_command.
srun -N $SLURM_NNODES -n $SLURM_NNODES --mpi=pmi2 flux start python -u -m saddlemill

date
