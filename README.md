# tsearch

## Installation on Lonestar6

### 1. Base Environment Setup

Load necessary modules and create the base Conda environment.

```bash
module unload impi python3
module load cuda/12.8

# Enter idev to get a GPU node
idev -A CHE23004 -p gpu-a100-dev -t 02:00:00

# Create base environment
conda create --prefix /work/08405/ilgar/ls6/conda_envs/executorlib -c conda-forge python=3.12 flux-core flux-sched openmpi=5.0.5 "libhwloc=*=cuda*" executorlib
conda activate executorlib

find $CONDA_PREFIX -name "sched-fluxion-*.so" -path "*feedstock_root*" -exec cp {} $CONDA_PREFIX/lib/flux/modules/ \;

```

## Installation on Vista

### 1. Base Environment Setup

Load necessary modules and create the base Conda environment.

```bash
# Create base environment
conda create -n executorlib -c conda-forge python=3.12 flux-core flux-sched "openmpi=5.0.5=external_*" executorlib
conda activate executorlib

# Install hardware locality
conda install "libhwloc=*=cuda*" -c conda-forge
export LD_LIBRARY_PATH=/opt/apps/cuda/12.4/targets/sbsa-linux/lib/:$LD_LIBRARY_PATH

```

**(Optional) MPI Support:**
If `mpi4py` is required, use the following (experimental):

```bash
MPICC=$(which mpicc) pip install --no-binary=mpi4py mpi4py

```

### 2. Verify Flux Resources

Check if flux detects all resources correctly:

```bash
srun -n 1 flux start flux resource list

```

### 3. Application Specifics

Clone the environment and install specific machine learning libraries.

```bash
conda create --prefix /work/08405/ilgar/vista/conda_libraries/tsearch --clone executorlib

pip install --cache-dir <scratch_cache_dir> fairchem-core
# This part below is only necessary for Vista and not for Lonestar6
pip uninstall torch
pip install "torch==2.9.0+cu128" --index-url https://download.pytorch.org/whl/cu128

```

## Configuration

### Hugging Face Login

The CLI login may encounter issues. Use the Python interface instead:

```bash
python -c "import huggingface_hub; huggingface_hub.login()"

```

Alternatively, export the token directly:

```bash
export HF_TOKEN="***"

```

### Environment Variables

Add these to your `.bashrc` or run before execution to manage cache and paths:

```bash
# Move cache to scratch to save home directory space
export FAIRCHEM_CACHE_DIR="$SCRATCH/.cache/fairchem"

```

## Usage

### Interactive Node

Request an interactive node:

```bash
idev -p gh-dev -N 1 -m 120 -A CHE23004
# or for Lonestar:
idev -p gpu-a100-dev -N 1 -m 120 -A CHE23004

```

### Runtime Requirements

Ensure you run these commands (or add to `.bashrc`) every time you log in or start a job:

```bash
# On Vista:
export PYTHONPATH=<tsearch_path>:$PYTHONPATH
export LD_LIBRARY_PATH=/opt/apps/cuda/12.4/targets/sbsa-linux/lib/:$LD_LIBRARY_PATH
export FAIRCHEM_CACHE_DIR="$SCRATCH/.cache/fairchem"

# On LS6:
export PYTHONPATH=<tsearch_path>:$PYTHONPATH
module unload impi python3
module load cuda/12.8
export FAIRCHEM_CACHE_DIR="$SCRATCH/.cache/fairchem"


```
