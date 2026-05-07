# SaddleMill

## Installation

### 1. Base Environment Setup

#### *Lonestar6 (TACC)*

Load necessary modules and create the base Conda environment.

```bash
module unload impi python3
module load cuda/12.8

# Enter idev to get a GPU node
idev -A YOUR_ALLOCATION -p gpu-a100-dev -t 02:00:00

# Create base environment
conda create --prefix $WORK/conda_envs/executorlib -c conda-forge python=3.12 flux-core flux-sched openmpi=5.0.5 "libhwloc=*=cuda*" executorlib
conda activate executorlib

find $CONDA_PREFIX -name "sched-fluxion-*.so" -path "*feedstock_root*" -exec cp {} $CONDA_PREFIX/lib/flux/modules/ \;

```

#### *Vista (TACC)*

Load necessary modules and create the base Conda environment.

```bash
# Create base environment
conda create -n executorlib -c conda-forge python=3.12 flux-core flux-sched "openmpi=5.0.5" executorlib
conda activate executorlib

# Install hardware locality
conda install "libhwloc=*=cuda*" -c conda-forge
export LD_LIBRARY_PATH=/opt/apps/cuda/12.4/targets/sbsa-linux/lib/:$LD_LIBRARY_PATH

find $CONDA_PREFIX -name "sched-fluxion-*.so" -path "*feedstock_root*" -exec cp {} $CONDA_PREFIX/lib/flux/modules/ \;
```

**(Optional) MPI Support:**
If `mpi4py` is required, use the following (experimental):

```bash
MPICC=$(which mpicc) pip install --no-binary=mpi4py mpi4py

```

#### *Perlmutter (NERSC)*

Necessary modules (should be loaded by default):
* `PrgEnv-gnu/8.5.0` (Compiler Suite)
* `cudatoolkit/12.4` (System CUDA headers)
* `craype-accel-nvidia80` (Links MPI to A100 GPUs)
* `cray-mpich/8.1.30` (High-Speed Network Library)
* `python/3.11` (System Python)
* `conda/Miniforge3-24.7.1-0` (Access to mamba)

Other modules (should be loaded by default):
* `craype-x86-milan`
* `libfabric/1.22.0`
* `craype-network-ofi`
* `xpmem/2.9.7...`
* `cray-dsmml/0.3.0 `
* `cray-libsci/24.07.0`
* `craype/2.7.32`
* `gcc-native/13.2`
* `perftools-base/24.07.0`
* `cpe/24.07`
* `gpu/1.0`
* `sqs/2.0`
* `darshan/default`

Create base environment:

```bash
# set mamba cache at scratch
mamba config --set pkgs_dirs $SCRATCH/.cache/conda

# Enter idev to get a GPU node
salloc --nodes 1 --qos interactive --time 04:00:00 --constraint gpu --gpus 4 --account YOUR_ALLOCATION

# Create base environment
mamba create -p $SCRATCH/envs/executorlib -c conda-forge python=3.12 \
  flux-core flux-sched executorlib "libhwloc=*=cuda*"  # warnings about cuda, ucx, nccl, etc. are ok
conda activate executorlib

```

**(Optional) MPI Support:**
If `mpi4py` is required, use the following to compile it from source with Cray wrappers (experimental):

```bash
MPICC="cc -shared" pip install --force-reinstall --no-cache-dir --no-binary=mpi4py mpi4py

```

### 2. Verify Flux Resources

Check if flux detects all resources correctly:

```bash
# For Vista and Perlmutter
srun -n 2 flux start flux resource list
# For LS6
srun -n 2 --mpi=pmi2 flux start flux resource list

```

### 3. Application Specifics

Clone the environment and install specific machine learning libraries.

```bash
conda create --prefix $WORK/conda_libraries/saddlemill --clone executorlib

pip config set global.cache-dir "/path/to/your/cache/directory"  # like $SCRATCH/.cache/pip

pip install "fairchem-core>=2.19.0" "ase>=3.26.0" scipy==1.16
# If you want Vasp inpute files to be created like in omat/oc20:
pip install fairchem-data-omat
# If you will need some of the catsunami functionality or create Vasp input files for oc20
pip install fairchem-data-oc
# If you will need VaspInteractive
pip install git+https://github.com/ulissigroup/vasp-interactive.git

# This part below is only necessary for Vista (not for Lonestar6 or Perlmutter)
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
export PYTHONPATH=<saddlemill_path>:$PYTHONPATH

```

## Usage

### Interactive Node

Request an interactive node:

```bash
idev -p gh-dev -N 1 -m 120 -A YOUR_ALLOCATION
# or for Lonestar:
idev -p gpu-a100-dev -N 1 -m 120 -A YOUR_ALLOCATION

```

### Runtime Requirements

Ensure you run these commands (or add to `.bashrc`) every time you log in or start a job:

```bash
# On Vista:
export PYTHONPATH=<saddlemill_path>:$PYTHONPATH
export FAIRCHEM_CACHE_DIR="$SCRATCH/.cache/fairchem"
export LD_LIBRARY_PATH=/opt/apps/cuda/12.4/targets/sbsa-linux/lib/:$LD_LIBRARY_PATH


# On LS6:
export PYTHONPATH=<saddlemill_path>:$PYTHONPATH
export FAIRCHEM_CACHE_DIR="$SCRATCH/.cache/fairchem"
module unload impi python3
module load cuda/12.8


# On Perlmutter
export PYTHONPATH=<saddlemill_path>:$PYTHONPATH
export FAIRCHEM_CACHE_DIR="$SCRATCH/.cache/fairchem"
# --- START: FIX LIBRARY PATHS ---
export PY_SITE_PKGS=$(python -c "import site; print(site.getsitepackages()[0])")
export NVIDIA_DIR="${PY_SITE_PKGS}/nvidia"
# Prepend ALL Nvidia libraries to the load path
for lib in cuda_runtime nvjitlink cusparse cublas cufft cudnn curand cusolver nccl; do
  export LD_LIBRARY_PATH="${NVIDIA_DIR}/${lib}/lib:${LD_LIBRARY_PATH}"
done
# --- END: FIX LIBRARY PATHS ---

```

### Running SaddleMill

Create a `config.ini` in your working directory (see `CLAUDE.md` for full reference), place your input `.traj` files in `dir_path` (subdirectories are scanned recursively), then launch:

```bash
# Distributed (multi-node, multi-GPU)
srun -N $SLURM_NNODES -n $SLURM_NNODES --gpus-per-node=4 flux start python -u -m saddlemill

# Serial (single GPU, useful for debugging)
# Set executorlib = False in config.ini, then:
python -u -m saddlemill
```

### Resume and Continuation

SaddleMill automatically handles resume if a job times out or is interrupted:

- **Resume unfinished jobs**: Just resubmit with the same `config.ini`. By default (`run_jobs = remaining`), only jobs that never ran are picked up. Already-completed jobs are skipped.

- **Redo specific categories**: Set `run_jobs` to target specific outcomes. Selection is per-line: a job with mixed results (e.g., 3 converged + 3 not_converged Dimer attempts) is selectable by both categories. Only matching entries are redone; the rest stay.
  ```ini
  run_jobs = not_converged    # Redo unconverged attempts/sub-bands/sides
  run_jobs = converged        # Redo converged entries (e.g., refine with VASP)
  run_jobs = errored          # Retry errored entries
  run_jobs = all              # Redo everything
  ```

- **Continue from previous result** (`continue_from_result = True`, default): When re-running, SaddleMill extracts the last result from output trajs and continues optimization from there. For Dimer, each matching attempt continues with its eigenmode. For NEB, matching sub-bands continue from their extracted images. For DoubleMinimization, unconverged sides continue from their last state. Errored entries fall back to fresh generation. Set `continue_from_result = False` to generate fresh replacements (same reaction types / same sub-band endpoints).

- **Archiving**: Old files are fully backed up as `previous_{N}.zip`. Only matching entries are removed from active CSVs and output trajectories (per-attempt for Dimer, per-sub-band for NEB, per-side for DoubleMinimization).

- **Fresh start**: Delete `traj_files_ordered.json` and the output directories to start completely from scratch.

### Dimer: `initial_guess` Reaction Type

The `initial_guess` reaction type is for running the dimer method on a **pre-prepared TS guess** from an external source (another code, a database, or a different SaddleMill method like NEB). It starts the dimer from the input geometry as-is with no displacement and no supercell expansion (even if `supercell = True`). If the input structure has an eigenmode in `atoms.info['eigenmode']`, it is used to seed the dimer instead of a random guess.

```ini
[ourDimer]
dataset_type = bulk          # or oc
reaction_types = initial_guess
```

This is different from `continue_from_result`, which is an automatic mechanism for continuing a previous SaddleMill run. Use `initial_guess` when bringing a TS from outside SaddleMill; `continue_from_result` handles the "pick up where I left off" case internally.

## Testing

Install test dependencies:

```bash
pip install pytest pytest-timeout
```

Run CPU-only unit tests (no GPU needed, works on login nodes):

```bash
pytest -m "not gpu and not flux" -v
```

Run all tests (requires a GPU node):

```bash
pytest -v --timeout=600
```

Run only GPU integration tests:

```bash
pytest -m gpu -v
```
