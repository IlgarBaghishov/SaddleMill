#!/bin/sh
#SBATCH -N 2
#SBATCH -n 2
#SBATCH -o ll_out
#SBATCH -p gh-dev
#SBATCH -t 02:00:00
#SBATCH -A CHE23004
#SBATCH -J dimer1

# module unload xalt
# export LD_LIBRARY_PATH=/opt/apps/cuda/12.4/targets/sbsa-linux/lib/:$LD_LIBRARY_PATH

export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps-pipe-$USER
export CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-mps-log-$USER
srun -n $SLURM_NNODES mkdir -p $CUDA_MPS_PIPE_DIRECTORY
srun -n $SLURM_NNODES mkdir -p $CUDA_MPS_LOG_DIRECTORY

srun -n $SLURM_NNODES nvidia-cuda-mps-control -d

srun -n $SLURM_NNODES flux start python -u -m tsearch

srun -n $SLURM_NNODES bash -c "echo quit | nvidia-cuda-mps-control"
