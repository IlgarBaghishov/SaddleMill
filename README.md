# SaddleMill

## Installation

### 1. Base Environment Setup

#### *3 A100 per node HPC*

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

#### *GH200 HPC*

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

#### *4 A100 per node HPC*

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

#### *In-house CPU cluster*

CPU-only (no GPU/CUDA), for VASP runs — no `torch`/`fairchem-core`. Create the env as `saddlemill` directly (no later clone):

```bash
conda create -n saddlemill -c conda-forge python=3.12 flux-core flux-sched "openmpi=5.0.5" executorlib
conda activate saddlemill

find $CONDA_PREFIX -name "sched-fluxion-*.so" -path "*feedstock_root*" -exec cp {} $CONDA_PREFIX/lib/flux/modules/ \;

```

### 2. Verify Flux Resources

Check if flux detects all resources correctly:

```bash
# For the GH200 HPC and the 4 A100 per node HPC
srun -n 2 flux start flux resource list
# For the 3 A100 per node HPC
srun -n 2 --mpi=pmi2 flux start flux resource list
# For the in-house CPU cluster (grab nodes first, e.g. salloc --nodes 2 --qos debug)
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

# This part below is only necessary for the GH200 HPC (not for the A100-based HPCs)
pip install "torch==2.9.0+cu128" --index-url https://download.pytorch.org/whl/cu128

```

**VTST + VASP (manual ASE patch required for now):** ASE ≤ 3.28 writes the integer INCAR tag `DROTMAX` as a float (`10.000000`), which VASP's VTST dimer silently ignores (falls back to `RotMax 4`). Until fixed upstream, patch the installed ASE — in `ase/calculators/vasp/create_input.py`, move `'drotmax'` from `float_keys` to `int_keys`. The `ase` version pin will be bumped to the fixed release once available. (Guarded by `tests/test_ase_vasp_incar.py`.)

**In-house CPU cluster (VASP-only):** the `saddlemill` env is already the app env (no `--clone`); VASP needs no `fairchem-core`/`torch`. Install only:

```bash
pip install "ase>=3.26.0" scipy==1.16 fairchem-data-omat fairchem-data-oc
```

Add `fairchem-core>=2.19.0` only for the FAIRChem calculator or `.aselmdb` I/O.

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
# or for the 3 A100 per node HPC:
idev -p gpu-a100-dev -N 1 -m 120 -A YOUR_ALLOCATION
# or for the in-house CPU cluster:
salloc --nodes 2 --qos debug

```

### Runtime Requirements

Ensure you run these commands (or add to `.bashrc`) every time you log in or start a job:

```bash
# On the GH200 HPC:
export PYTHONPATH=<saddlemill_path>:$PYTHONPATH
export FAIRCHEM_CACHE_DIR="$SCRATCH/.cache/fairchem"
export LD_LIBRARY_PATH=/opt/apps/cuda/12.4/targets/sbsa-linux/lib/:$LD_LIBRARY_PATH


# On the 3 A100 per node HPC:
export PYTHONPATH=<saddlemill_path>:$PYTHONPATH
export FAIRCHEM_CACHE_DIR="$SCRATCH/.cache/fairchem"
module unload impi python3
module load cuda/12.8


# On the 4 A100 per node HPC
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


# On the in-house CPU cluster (VASP-only; no CUDA/FAIRChem):
export PYTHONPATH=<saddlemill_path>:$PYTHONPATH
export VASP_PP_PATH=/path/to/vasp/potcars    # directory holding potpaw_PBE.<version>/

```

### Running SaddleMill

Create a `config.ini` in your working directory (see `CLAUDE.md` for full reference), place your input `.traj` files in `dir_path` (subdirectories are scanned recursively), then launch:

```bash
# Distributed (multi-node, multi-GPU)
srun -N $SLURM_NNODES -n $SLURM_NNODES --gpus-per-node=4 flux start python -u -m saddlemill

# Distributed on the in-house CPU cluster (no --gpus-per-node; --mpi=pmi2 + clear inherited
# PMI vars so the inner Flux/MPI doesn't clash with the outer srun):
srun -N $SLURM_NNODES -n $SLURM_NNODES --mpi=pmi2 flux start \
    env -u PMI2_FD -u PMI_FD -u PMI2_RANK -u PMI_RANK -u PMI2_SIZE -u PMI_SIZE -u PMI2_SPROUTE \
    python -u -m saddlemill

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

### VASP input generation (`input_generator`)

By default the `[Vasp]` section *is* your INCAR — every tag you list there is written to the calculator. When you want a standard input recipe (e.g. OMat24 or OC20) instead of hand-writing every tag, set `input_generator` in the `[ourVasp]` section. (`[ourVasp]` holds SaddleMill's VASP-input knobs; `[Vasp]` stays a pure ASE pass-through.) It works for **all methods** (NEB, Dimer, Minimization, DoubleMinimization, SinglePoint).

```ini
[ourVasp]
input_generator = omat24_static   # omat24_static | omat24_relax | cheap_omat
                                  #   | oc20 | cheap_oc20 | oc22 | cheap_oc22
                                  #   ...or 'my_pkg.my_module:my_func'
                                  #   ...or '/path/to/my_inputs.py:my_func'

[Vasp]
encut = 600                       # any [Vasp] tag overrides the generator
pp = PBE
xc = PBE
```

Per-tag precedence is **`[Vasp]` key → `input_generator` output → ASE/VASP default**: a tag you set in `[Vasp]` always wins; a tag the generator supplies is used when `[Vasp]` is silent; a tag in neither falls back to ASE's default. The generator contributes only electronic/accuracy settings — ionic-driver tags (`IBRION`/`NSW`/`POTIM`/`EDIFFG`) are stripped, since SaddleMill drives the geometry through ASE.

A custom generator is any function `generator(atoms) -> dict` returning ASE-`Vasp` kwargs (lowercased INCAR tags plus `kpts`/`gamma`/`setups`/`magmom`); point `input_generator` at it as `module:func` or `file.py:func`. The `omat24_*`/`cheap_omat` built-ins need `fairchem-data-omat` and `oc20`/`cheap_oc20` need `fairchem-data-oc` installed (see Installation above); `oc22`/`cheap_oc22` are self-contained (the exact Meta OC22 settings are frozen inside SaddleMill, since the upstream `OC22_dataset` code is deprecated and today's pymatgen has drifted from what Meta ran).

The `oc22` built-in reproduces the OC22 dataset's level of theory exactly (arXiv:2206.08917): PBE + Materials Project Hubbard U, spin-polarized with MP-default initial moments (magmoms already on the atoms win), ENCUT 500, Γ-centered `ceil(30/a) × ceil(30/b) × 1` meshes, 2022-era MPRelaxSet POTCAR setups, and dipole correction only when the structure has tag=2 adsorbate atoms (Meta applied it to adsorbate+slabs only). Meta used the PBE 5.4 POTCAR library — add `pp_version = 54` in `[Vasp]` if your `$VASP_PP_PATH` has versioned `potpaw_PBE.54` folders. The `cheap_oc20`/`cheap_oc22` variants follow the `cheap_omat` recipe: the in-plane k-point multiplier is halved (40→20 / 30→15) and POTCARs are lightened (`minimal` base, soft `_s` O/C/N, f-in-core lanthanides kept) so ENCUT can drop to ~300 via `[Vasp]` — spin, U and the functional are untouched, so the cheap pass stays on the same PES; reconverge with the exact preset afterwards.

### Extra input files (`extra_input_files`), e.g. VTST MODECAR

`input_generator` only sets INCAR/k-points/POTCAR. To drop **extra files** into the VASP working directory — most usefully a VTST `MODECAR` (initial dimer mode) — use `extra_input_files`:

```ini
[ourVasp]
extra_input_files = modecar       # built-in | 'module:func' | 'file.py:func' | space-separated list
```

The built-in `modecar` builds the file from `atoms.info['eigenmode']` (the same eigenmode NEB/Dimer/DoubleMinimization already stamp on their output frames), reordered to POSCAR order. A custom writer is any `writer(calc, atoms, directory) -> None` (it receives the calculator, so it can use `calc.sort`).

Symmetrically, **`extra_outputs`** parses files back *out* of the VASP directory after the run and merges the result into the output frame's `.info`. A parser is any `parser(calc, atoms, directory) -> dict`; the built-in `vtst_dimer` returns `eigenmode` (from `NEWMODECAR`, mapped back to atoms order via `calc.resort`) and `curvature` (from `DIMCAR`).

Together these enable a **VASP-internal VTST dimer driven by SaddleMill as a pure launcher**:

```ini
[ourVasp]
input_generator   = omat24_static   # base INCAR recipe
extra_input_files = modecar         # write MODECAR from each frame's eigenmode
extra_outputs     = vtst_dimer      # read NEWMODECAR/DIMCAR -> eigenmode/curvature onto output

[Vasp]                              # VTST dimer driver tags (plain pass-through)
ichain = 2
ibrion = 3
potim = 0
iopt = 3                            # QuickMin: force-driven step, no fixed jump
maxmove = 0.1
ddr = 0.005
nsw = 300
ediffg = -0.03
```

with `method = SinglePoint`, `Calculator = Vasp`. SaddleMill runs one VASP call per structure, VASP runs the whole dimer internally, and the converged saddle (geometry + E/F) **plus** the refined `eigenmode`/`curvature` are written to the output traj. Use plain `Vasp` (not `VaspInteractive`, which forces `ibrion = -1`). The input frames must carry `atoms.info['eigenmode']` (e.g. NEB-CI / Dimer / DoubleMinimization outputs).

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
