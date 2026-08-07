#!/bin/bash
#SBATCH -N 2
#SBATCH --ntasks-per-node=224
#SBATCH -q debug
#SBATCH -t 02:00:00
#SBATCH -J sp_vtstdimer_mp20bat
#SBATCH -o slurm_%j.log

pwd; hostname -f; date

ml list;

CONDA_BASE=$(dirname $(dirname $CONDA_EXE))
source $CONDA_BASE/etc/profile.d/conda.sh
conda activate saddlemill

export VASP_PP_PATH=/home/graeme/vasp
export PATH=/home/sung/codes/vasp_earlystop/vasp.6.6.0/bin:$PATH
which vasp_std
srun -N $SLURM_NNODES -n $SLURM_NNODES --mpi=pmi2 flux start \
    env -u PMI2_FD -u PMI_FD -u PMI2_RANK -u PMI_RANK -u PMI2_SIZE -u PMI_SIZE -u PMI2_SPROUTE \
    python -u -m saddlemill

date
