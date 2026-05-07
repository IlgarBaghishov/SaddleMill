#!/bin/sh
#SBATCH -N 8
#SBATCH -n 8
#SBATCH -o ll_out
#SBATCH -p gpu-a100
#SBATCH -t 48:00:00
#SBATCH -A YOUR_ALLOCATION

module unload impi python3
module load cuda/12.8

srun -n $SLURM_NNODES --mpi=pmi2 flux start python -u -m saddlemill

