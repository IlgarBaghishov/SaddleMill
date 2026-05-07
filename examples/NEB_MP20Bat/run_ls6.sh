#!/bin/sh
#SBATCH -N 8
#SBATCH -n 8
#SBATCH -p gpu-a100
#SBATCH -t 48:00:00
#SBATCH -o slurm_%j.out
#SBATCH -A YOUR_ALLOCATION
#SBATCH -J neb_mp20bat

pwd; hostname -f; date

module unload impi python3
module load cuda/12.8

srun -N $SLURM_NNODES -n $SLURM_NNODES --mpi=pmi2 flux start python -u -m saddlemill

date
