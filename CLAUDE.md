# tsearch - High-Throughput Transition State Search Library

## Overview

tsearch is a Python library for creating datasets of Transition States (TS) using neural network potentials (FAIRChemCalculator / Meta OCP models). It supports distributed GPU execution on HPC systems (NERSC Perlmutter, TACC Vista/LS6) via executorlib + Flux.

## Entry Point

```bash
srun -N $SLURM_NNODES -n $SLURM_NNODES --gpus-per-node=4 flux start python -u -m tsearch
```

The `__main__.py` reads `config.ini` from the current directory, loads the method, scans for `.traj` files, distributes jobs across GPUs, and collects results.

## Supported Methods

| Method | Config value | Module | Description |
|--------|-------------|--------|-------------|
| NEB | `NEB` | `nebopt.py` | Nudged Elastic Band (with optional DNEB switching) |
| Dimer | `Dimer` | `dimeropt.py` | Dimer method for saddle point search |
| Minimization | `Minimization` | `geomopt.py` | Single structure geometry optimization |
| DoubleMinimization | `DoubleMinimization` | `geomopt.py` | TS refinement: displaces along eigenmode in both directions, relaxes, checks for reaction |

## Architecture

```
config.ini
    |
    v
__main__.py  -->  init_function.py (per-worker GPU setup + calculator loading)
    |                    |
    v                    v
config.py            FluxJobExecutor (distributed) or serial mode
    |
    v
nebopt.py / dimeropt.py / geomopt.py  (method functions)
    |
    v
catsunami/ocpneb.py  (OCPNEB: batched NEB with swDNEB switching)
```

## Key Modules

### `config.py`
- `ConfigManager`: Reads `config.ini` with type inference (bool/int/float/list/string)
- `load_calculator()`: Creates FAIRChemCalculator from config
- `load_method()`: Imports the correct optimization function
- `load_optimizer()`: Returns optimizer class(es) - for NEB returns (endpoint_optimizer, neb_optimizer)
- `get_trajes_and_indices()`: Scans dir_path for .traj files, splits into job batches
- Resume support: `get_remaining_trajes()` skips completed jobs

### `nebopt.py` - NEB Workflow
1. **Endpoint relaxation** (optional): Relaxes reactant/product with configurable optimizer (e.g., LBFGS)
2. **Interpolation**: `ocp_idpp` (Meta's PBC-aware), `ase_idpp`, `ase_linear` (auto-falls back to IDPP on atom overlap), or `False` (use provided frames)
3. **NEB optimization**: Uses `OCPNEB` class with MDMin optimizer. Supports climbing image.
4. **Output**: Extracts critical image (TS candidate) with tangent vector as eigenmode, barrier height, and reaction energetics. Generates band plot PNG.

### `catsunami/ocpneb.py` - Core NEB Engine
- **`OCPNEB`** (extends DyNEB): Batch-evaluates intermediate images via FAIRChemCalculator for efficiency. Caches forces between calls. Handles constraints (fixed atoms by tag=0 or explicit constraints). Supports dynamic relaxation (skipping converged images).
- **`swDNEB`** (NEBMethod subclass): Implements the switched Doubly Nudged Elastic Band method:
  - Uses improved tangent vectors (energy-weighted at extrema)
  - Adds perpendicular spring force component to straighten the band
  - Switching function `sw = (2/pi) * arctan(|F_perp|^2 / |F_S_perp|^2)` turns off DNEB force as convergence is reached (preventing frustration)
  - Based on: Henkelman & Jonsson, J. Chem. Phys. (2000) and Trygubenko & Wales (2004)

### `dimeropt.py` - Dimer Method
- Generates displacement candidates via `dimertools/structure_edit.py`
- Supports `bulk` (random supercell vacancies) and `oc` (adsorbate-targeted) modes
- Convergence checks every 5 steps: participation ratio (delocalization) and desorption detection
- Extension check if initial convergence fails

### `geomopt.py` - Geometry Optimization
- `geomopt()`: Standard relaxation with optional cell relaxation (FrechetCellFilter)
- `doublegeomopt()`: Takes converged TS with eigenmode, displaces +/- 0.25*eigenmode, relaxes both directions, detects bond breaking/forming via `check_reaction()`

### `tools.py` - Utilities
- Bond detection via ASE neighbor_list with natural cutoffs
- `check_reaction()` / `check_adsorbate_reaction()`: Compare connectivity between structures

### `init_function.py` - Worker Initialization
- Assigns GPU to worker based on executorlib_worker_id and jobs_per_gpu
- Sets `CUDA_VISIBLE_DEVICES` for multi-job-per-GPU scenarios
- Returns `{calc, Optimizer}` dict passed to method functions

### `catsunami/autoframe.py` - NEB Frame Generation
- `AutoFrameDissociation` / `AutoFrameTransfer`: Generates NEB initial/final frames from reaction databases
- Anomaly detection (intercalation, desorption, surface changes)
- Adsorbate reordering for symmetric species

### `catsunami/reaction.py` - Reaction Definitions
- `Reaction` class: Represents dissociation/desorption/transfer reactions with atom mappings and edge lists

## Configuration Reference (`config.ini`)

```ini
[Main]
executorlib = True          # Use FluxJobExecutor (True) or serial mode (False)
method = NEB                # NEB | Dimer | Minimization | DoubleMinimization
dir_path = /path/to/trajs   # Directory containing .traj input files
Optimizer = MDMin           # MDMin | BFGS | LBFGS | FIRE (used for NEB band optimization)
fmax = 0.05                 # Force convergence criterion (eV/A)
steps = 6000                # Maximum optimization steps
jobs_per_gpu = 1            # Number of concurrent jobs per GPU
Calculator = FAIRChemCalculator
resume = False              # Resume from previous partial run
zip = True                  # Compress debug files

[FAIRChemCalculator]
device = cuda
model_name_or_path = uma-s-1p1   # Model checkpoint
task_name = oc20                  # Task type

[MDMin]
dt = 0.02                   # Time step (dimensionless, ASE default=0.2)
maxstep = 0.1               # Max displacement per step (Angstrom)

[LBFGS]
memory = 10
damping = 0.99
alpha = 200
maxstep = 0.1

[ourNEB]
relax_endpoints = True
endpoint_relax_Optimizer = LBFGS   # Separate optimizer for endpoints
endpoint_relax_fmax = 0.02
endpoint_relax_steps = 1000
interpolate_method = ase_linear    # ocp_idpp | ase_idpp | ase_linear | False
num_frames = 10                    # Number of NEB images
batch_size = 8                     # Batch size for FAIRChem inference
DNEB = True                        # Enable switched DNEB

[DyNEB]
k = 5                       # Spring constant (eV/A^2)
method = improvedtangent     # improvedtangent | aseneb
climb = True                 # Climbing image NEB
allow_shared_calculator = True
dynamic_relaxation = False   # Skip converged images during optimization

[ourDimer]
dataset_type = oc            # oc | bulk
num_attempts = 3
delocalization_threshold = 0.8

[ourMinimization]
relax_cell = False

[ourDoubleMinimization]
relax_cell = False
```

## Output Structure

For method `NEB`:
```
NEB_status_csvs/status_rank_*.csv    # job_id,rank,status (converged/not_converged/error)
NEB_trajes/collected_ts_rank_*.traj  # TS candidates with metadata in atoms.info
NEB_debug_zips/                      # Compressed log/traj/plot files
```

Each TS image in the output trajectory contains:
- `eigenmode`: Tangent vector at saddle point
- `barrier`: Forward barrier (eV)
- `dE`: Reaction energy (eV)
- `max_forces`: Max forces on each image
- `converged`: 1 or 0
- `reactant_positions` / `product_positions`

## Execution Modes

**Distributed (executorlib=True)**: Uses FluxJobExecutor with one worker per GPU (or jobs_per_gpu workers sharing GPUs). Each worker calls `init_function` once to load the calculator, then processes jobs sequentially.

**Serial (executorlib=False)**: Runs jobs one at a time on a single GPU. Useful for debugging.

## HPC Setup (Perlmutter)

```bash
#SBATCH --constraint=gpu
#SBATCH --gpus-per-node=4
#SBATCH --nodes=N
srun -N $SLURM_NNODES -n $SLURM_NNODES --gpus-per-node=4 flux start python -u -m tsearch
```

Set `FAIRCHEM_CACHE_DIR` for model caching. Requires CUDA libraries in `LD_LIBRARY_PATH`.

## DNEB Theory Notes

The Doubly Nudged Elastic Band adds a perpendicular spring force component to straighten the band during convergence. The switching function (Eq. 15 from Henkelman & Jonsson) turns off this force as `|F_perp|` drops below `|F_S_perp|`, preventing the frustration problem where the straightening force fights against convergence on curved MEPs. The switched DNEB (swDNEB) implementation is in `catsunami/ocpneb.py`.
