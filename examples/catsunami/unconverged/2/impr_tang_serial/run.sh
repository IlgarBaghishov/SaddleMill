#!/bin/sh

#SBATCH -N 1
#SBATCH -n 1
#SBATCH -o ll_out
#SBATCH -p gh-dev
#SBATCH -t 02:00:00
#SBATCH -A CHE23004

module unload xalt
export LD_LIBRARY_PATH=/opt/apps/cuda/12.4/targets/sbsa-linux/lib/:$LD_LIBRARY_PATH

python -u aseneb.py
