#!/bin/bash
#SBATCH --nodes=256
#SBATCH --ntasks-per-node=1
#SBATCH --constraint=gpu
#SBATCH --gpus-per-node=4
#SBATCH -q regular
#SBATCH -t 48:00:00
#SBATCH --output=slurm_%j.log
#SBATCH --account=YOUR_ALLOCATION
#SBATCH -J doublemin_oc22

pwd; hostname -f; date

export MPICH_GPU_SUPPORT_ENABLED=1  # Turn on GTL; crucial if transferring data between GPUs on different nodes

# --- Library paths for pip-installed CUDA libs ---
export PY_SITE_PKGS=$(python -c "import site; print(site.getsitepackages()[0])")
export NVIDIA_DIR="${PY_SITE_PKGS}/nvidia"
export LD_LIBRARY_PATH="${NVIDIA_DIR}/cuda_runtime/lib:${NVIDIA_DIR}/nvjitlink/lib:${NVIDIA_DIR}/cusparse/lib:${NVIDIA_DIR}/cublas/lib:${NVIDIA_DIR}/cufft/lib:${NVIDIA_DIR}/cudnn/lib:${NVIDIA_DIR}/curand/lib:${NVIDIA_DIR}/cusolver/lib:${NVIDIA_DIR}/nccl/lib:${LD_LIBRARY_PATH}"

srun -N $SLURM_NNODES -n $SLURM_NNODES --gpus-per-node=4 flux start python -u -m saddlemill

date
