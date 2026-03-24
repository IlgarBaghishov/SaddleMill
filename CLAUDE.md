# tsearch - High-Throughput Transition State Search Library

## Overview

tsearch is a Python library for creating datasets of Transition States (TS) using neural network potentials (FAIRChemCalculator / Meta's UMA model) or DFT (VASP / VaspInteractive). It supports distributed GPU execution on HPC systems (NERSC Perlmutter, TACC Vista/LS6) via executorlib + Flux.

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
- `ConfigManager`: Reads `config.ini` with type inference (bool/int/float/list/string). Quoted strings (`"..."` or `'...'`) are preserved as literal strings (used for VASP commands containing spaces).
- `load_calculator()`: Returns calculator class/callable based on config (`FAIRChemCalculator`, `Vasp`, or `VaspInteractive`). For FAIRChem, returns `from_model_checkpoint` method; for VASP calculators, returns the class itself. Instantiation is deferred to `init_function.py` (for FAIRChem) or `nebopt.py` (for VASP, per-image).
- `load_method()`: Imports the correct optimization function
- `load_optimizer()`: Returns optimizer class(es) - for NEB returns (endpoint_optimizer, neb_optimizer)
- `get_trajes_and_indices()`: Scans dir_path for .traj files, splits into job batches
- Resume support: `get_remaining_trajes()` skips completed jobs

### `nebopt.py` - NEB Workflow
1. **Endpoint relaxation** (optional): Relaxes reactant/product with configurable optimizer (e.g., LBFGS). For VaspInteractive, calculators are finalized after relaxation and endpoints are frozen with `SinglePointCalculator`.
2. **Interpolation**: `ocp_idpp` (Meta's PBC-aware), `ase_idpp`, `ase_linear` (auto-falls back to IDPP on atom overlap), or `False` (use provided frames)
3. **NEB optimization**: Uses `OCPNEB` class with MDMin optimizer. Supports climbing image. For VASP/VaspInteractive, each image gets its own calculator instance with separate working directories (`VASP_{job_id}_{image_idx}/`), separate `command`/`ncore` settings for endpoints vs intermediates, and WAVECAR/CHG/CHGCAR cleanup after completion.
4. **Output**: Extracts critical image (TS candidate) with tangent vector as eigenmode, barrier height, and reaction energetics. Generates band plot PNG.

### `catsunami/ocpneb.py` - Core NEB Engine
- **`OCPNEB`** (extends BaseNEB): Two modes controlled by `vasp` flag:
  - **FAIRChem mode** (`vasp=False`): Batch-evaluates intermediate images via FAIRChemCalculator for efficiency. Caches forces between calls. fairchem imports are lazy (only loaded in this mode).
  - **VASP mode** (`vasp=True`): Delegates to parent `BaseNEB` for standard per-image force evaluation. Each image has its own VASP calculator. No batching, no caching, no fairchem dependency at runtime.
  - Both modes: Handles constraints (fixed atoms by tag=0 or explicit constraints). Stores full `real_forces` array `(nimages, natoms, 3)` including endpoint forces for uniform access via `real_forces[imax]`.
- **`swDNEB`** (NEBMethod subclass): Implements the switched Doubly Nudged Elastic Band method (works with both FAIRChem and VASP modes):
  - Uses improved tangent vectors (energy-weighted at extrema)
  - Adds perpendicular spring force component to straighten the band
  - Switching function `sw = (2/pi) * arctan(|F_perp|^2 / |F_S_perp|^2)` turns off DNEB force as convergence is reached (preventing frustration)
  - Based on: Henkelman & Jonsson, J. Chem. Phys. (2000) and Trygubenko & Wales (2004)

### `dimeropt.py` - Dimer Method
- Generates displacement candidates via `dimertools/structure_edit.py`
- Supports `bulk` (multiple reaction types, see below) and `oc` (adsorbate-targeted) modes
- Convergence checks every 5 steps: participation ratio (delocalization) and desorption detection
- Extension check if initial convergence fails
- Writes `reaction_type` to `atoms.info` for each attempt
- **Per-attempt error handling**: Each dimer attempt has its own try/except, so one failing attempt does not abort remaining attempts for the same structure
- **Consecutive error tracking**: Tracks structure-level errors via `consecutive_errors` counter (passed from `init_function`). If all attempts for a structure fail, counter increments; any successful attempt resets it to 0. When counter reaches `max_consecutive_errors`, worker calls `sys.exit(1)` to trigger executorlib restart (see Worker Health section)

### `dimertools/structure_edit.py` - Reaction Types for Dimer
Dimer mode supports 7 reaction types, configured via `reaction_types` (space-separated list):

| Type | Function | Description | Atoms displaced |
|------|----------|-------------|-----------------|
| `vacancy` | `get_vacancy_attempts()` | Remove atom, neighbor hops into vacancy | 1 (center-based) |
| `hop_reuse` | `get_hop_reuse_attempts()` | Existing atom relocated to interstitial site | 1 (vector) |
| `hop_insert` | `get_hop_insert_attempts()` | New small atom (H/C/N/O/B) inserted at interstitial site | 1 (vector) |
| `kickout_reuse` | `get_kickout_reuse_attempts()` | Existing atom placed at interstitial, kicks nearest lattice atom into another interstitial | 2 (vector) |
| `kickout_insert` | `get_kickout_insert_attempts()` | New similar-sized atom inserted at interstitial, kicks nearest lattice atom | 2 (vector) |
| `ring` | `get_ring_attempts()` | Ring of 2+ atoms rotate cooperatively; size randomly sampled from `ring_sizes` config. Use `ring_sizes = 2` for pairwise exchange. | N (vector) |
| `initial_guess` | `get_initial_guess_attempts()` | No displacement — dimer starts from input geometry as-is. For pre-prepared TS guesses. Exclusive: ignores other types with warning. Always 1 attempt. Works with both `bulk` and `oc` dataset types. | 0 (none) |

**Key infrastructure:**
- `find_interstitial_sites(atoms)`: Voronoi tessellation on 3x3x3 periodic images → filter by min distance from atoms → cluster within 0.5 Å. Uses `scipy.spatial.Voronoi` and `scipy.cluster.hierarchy`.
- `_mic_vector()` / `_nearest_site()`: Minimum image convention helpers for periodic distance calculations.
- `_find_ring(neighbors_dict, seed, ring_size)`: Finds closed rings of connected atoms in the neighbor graph via constrained random walk. Ring size=2 handles pairwise exchange, ring size>=3 handles cooperative ring rotations.
- Element sampling: `hop_insert` uses small atoms weighted by 1/covalent_radius (H heavily favored). `kickout_insert` uses Gaussian weight centered on host avg covalent radius (σ=0.2 Å) from a pool of 30 common metals/semiconductors.
- `turn_into_supercell(atoms, min_length=7.0)`: Preserves `.info` across `make_supercell()` (ASE's `make_supercell` drops `.info`). Enforces minimum cell dimension of 7 Å in each periodic direction to avoid self-interaction artifacts through PBC (important for fairchem's radius graph which uses a 5 Å cutoff). Called centrally in `get_attempts()` (controlled by `supercell` config option, default `True`).
- Dispatch: `_REACTION_TYPE_DISPATCH` dict maps type names to functions. `get_attempts()` iterates over configured types. `initial_guess` is handled early (before bulk/oc branch) since it works with both dataset types.
- `_safe_normalize(vec)`: Normalizes a vector; returns a random unit vector if norm is near zero (prevents division-by-zero in ring displacement calculations).
- `_build_neighbor_dict(atoms)`: Builds neighbor graph; skips self-interactions (`i == j`) which can occur in small periodic cells.
- Defensive guards throughout: zero-weight fallback in `_sample_kickout_insert_element`, empty `ring_sizes` early return, missing adsorbate atoms early return in OC mode.
- Backward compatible: default `reaction_types = vacancy` reproduces original behavior exactly.

### `geomopt.py` - Geometry Optimization
- `geomopt()`: Standard relaxation with optional cell relaxation (FrechetCellFilter)
- `doublegeomopt()`: Takes converged TS with eigenmode, displaces +/- 0.25*eigenmode, relaxes both directions, detects bond breaking/forming via `check_reaction()`. Reads `eigenmode`, `converged`, `src_index` from `atoms.info['orig_info']` (with fallback to `atoms.info` for backward compatibility).

### `tools.py` - Utilities
- `load_and_sanitize(traj, i, j)`: Loads atoms from trajectory and stashes original `.info` into `atoms.info = {"orig_info": <original_info>}`. This prevents per-atom array data (e.g. `forces`, `stress`) from causing size mismatches when atoms are later added/removed (e.g. vacancy in Dimer). Called in `__main__.py` for all methods uniformly.
- `clean_up_files(config_dict)`: Removes leftover temp files on resume. Method-aware: cleans NEB files (`neb_*.log`, `neb_*.traj`, `reactant_relaxation_*`, `product_relaxation_*`, `diffusion_barrier_*.png`), Dimer files (`dimer_control_*.log`, `dimer_opt_*.log`, `dimer_*.traj`), or Minimization files (`optimization_*.log`, `optimization_*.traj`). For VASP NEB, also removes `VASP_*_*/` per-image directories.
- Bond detection via ASE neighbor_list with natural cutoffs
- `check_reaction()` / `check_adsorbate_reaction()`: Compare connectivity between structures

### `init_function.py` - Worker Initialization
- Assigns GPU to worker based on executorlib_worker_id and jobs_per_gpu
- Sets `CUDA_VISIBLE_DEVICES` for multi-job-per-GPU scenarios (skipped for VASP calculators)
- For FAIRChem: instantiates calculator once (stored on GPU, shared across structures)
- For VASP/VaspInteractive: returns the class itself (instantiation deferred to per-image in `nebopt.py`)
- Returns `{calc, Optimizer, consecutive_errors}` dict passed to method functions. `consecutive_errors` is `[0]` (mutable list used as an in-memory counter); resets naturally on worker restart (fresh process)

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
Calculator = FAIRChemCalculator  # FAIRChemCalculator | Vasp | VaspInteractive
run_jobs = not_started       # Which job categories to process (see below)
zip = True                  # Compress debug files
max_consecutive_errors = 5  # Kill worker after N consecutive structures all-error (0 = disabled)
restart_limit = 3           # executorlib: max worker restarts before permanent death (0 = no restarts)

[FAIRChemCalculator]
device = cuda
name_or_path = uma-s-1p1   # Model checkpoint
task_name = oc20                  # Task type

[Vasp]                             # VASP INCAR parameters (used by both Vasp and VaspInteractive calculators)
setups = minimal
isif = 0
ispin = 1
isym = 0
lreal = Auto
ediff = 0.00001
ediffg = -0.03
symprec = 1e-10
encut = 350.0
gga = RP
pp = PBE
xc = PBE

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
batch_size = 8                     # Batch size for FAIRChem inference (ignored in VASP mode)
DNEB = True                        # Enable switched DNEB
# VASP-only settings (required when Calculator = Vasp or VaspInteractive):
vasp_command_endpoints = "srun --exclusive -n 64 vasp_std"
vasp_ncore_endpoints = 8           # NCORE for endpoint relaxation VASP jobs
vasp_command_intermediates = "srun --exclusive -n 16 vasp_std"
vasp_ncore_intermediates = 4       # NCORE for intermediate image VASP jobs

[BaseNEB]
k = 5                       # Spring constant (eV/A^2)
method = improvedtangent     # improvedtangent | aseneb
climb = True                 # Climbing image NEB
allow_shared_calculator = True

[ourDimer]
dataset_type = oc            # oc | bulk
num_attempts = 3             # Used for oc mode attempt count
reaction_types = vacancy     # Space-separated: vacancy hop_reuse hop_insert kickout_reuse kickout_insert ring initial_guess
num_attempts_per_type = 1    # Attempts per reaction type (bulk mode); total = len(types) * num_per_type
ring_sizes = 3 4             # Ring sizes to sample from for 'ring' reaction type
supercell = True             # Apply supercell expansion (min 7 Å) before generating attempts
delocalization_threshold = 0.8
extension_check_fmax = 0.4
extension_check_curvature = -0.2

[ourMinimization]
relax_cell = False

[ourDoubleMinimization]
relax_cell = False
```

### `run_jobs` — Flexible Job Selection

`run_jobs` specifies which categories of jobs to process. Fresh vs resume is determined implicitly by whether `traj_files_ordered.json` exists on disk.

**4 job categories:**

| Category | Meaning | CSV statuses that map here |
|---|---|---|
| `converged` | At least one attempt converged | `converged`, `converged_after_extension`, `converged_both`, `converged_min1`, `converged_min2` |
| `not_converged` | Ran without convergence | `not_converged`, `not_converged_after_extension`, `not_converged_StopRun`, `unconverged` |
| `error` | All attempts failed | `error`, `error: <message>` |
| `not_started` | No CSV row for this job_id | (absence of rows) |

**Examples:**
```ini
run_jobs = not_started                # Default. Fresh run: all jobs. Resume: continue unfinished.
run_jobs = not_started error          # Resume + retry errors
run_jobs = converged                  # Redo converged jobs (e.g., rerun with VASP)
run_jobs = not_converged              # Redo unconverged jobs
run_jobs = error                      # Retry only errors
run_jobs = all                        # Redo everything
```

**CSV archiving**: When redoing jobs that have existing CSV entries, old CSVs are archived as `{method}_status_csvs/previous_{N}.zip` (incrementing N) and entries for those job IDs are removed from the active CSVs. Jobs with no prior entries (e.g., `not_started`) don't trigger archiving.

**Fresh vs resume**: No explicit toggle — if `traj_files_ordered.json` doesn't exist, it's a fresh start; if it exists, it's a resume. To force a fresh start, delete the output directories and `traj_files_ordered.json`.

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
- `reaction_type`: (Dimer only) vacancy, hop_reuse, hop_insert, kickout_reuse, kickout_insert, ring, or unknown
- `orig_info`: Original `.info` dict from the input trajectory (stashed by `load_and_sanitize` to prevent per-atom array size mismatches)

## Execution Modes

**Distributed (executorlib=True)**: Uses FluxJobExecutor with one worker per GPU (or jobs_per_gpu workers sharing GPUs). Each worker calls `init_function` once to load the calculator, then processes jobs sequentially.

**Serial (executorlib=False)**: Runs jobs one at a time on a single GPU. Useful for debugging.

## Worker Health and Automatic Restart

CUDA device-side asserts (e.g., from fairchem's `radius_graph_pbc_v2` on degenerate structures) permanently poison the GPU context — every subsequent CUDA operation fails. To handle this:

1. **In-memory tracking**: `init_function` creates `consecutive_errors = [0]`. Each method function increments it when all attempts for a structure error, resets to 0 on any success.
2. **Worker self-kill**: When `consecutive_errors[0] >= max_consecutive_errors`, the method function calls `sys.exit(1)` before processing the next structure.
3. **executorlib restart**: `restart_limit` (default 3) in the `resource_dict` tells executorlib to restart the crashed worker. A new Flux job is submitted (potentially on a different node/GPU), `init_function` re-runs (fresh CUDA context, fresh calculator, `consecutive_errors` reset to `[0]`), and the failed task is automatically requeued.
4. **Exhaustion**: After `restart_limit` restarts, the worker is permanently dead and remaining tasks assigned to it receive `ExecutorlibSocketError`.

**Note**: On restart, Flux may allocate a different node/GPU. The `worker_id` is preserved, but `CUDA_VISIBLE_DEVICES` assignment in `init_function` may compute incorrectly if `jobs_per_gpu > 1` and the new node has a different GPU count. For `jobs_per_gpu = 1` (the common case), this is not an issue since executorlib's own `gpus_per_core=1` handles GPU allocation.

## HPC Setup (Perlmutter)

```bash
#SBATCH --constraint=gpu
#SBATCH --gpus-per-node=4
#SBATCH --nodes=N
srun -N $SLURM_NNODES -n $SLURM_NNODES --gpus-per-node=4 flux start python -u -m tsearch
```

Set `FAIRCHEM_CACHE_DIR` for model caching. Requires CUDA libraries in `LD_LIBRARY_PATH`.

## VASP / VaspInteractive Calculator Support

The NEB method supports VASP and VaspInteractive as alternatives to FAIRChemCalculator. Key design differences:

- **No batching**: Unlike FAIRChem (which batch-evaluates all intermediate images on GPU), VASP runs each image as a separate process. `OCPNEB` delegates to the parent `BaseNEB` force evaluation when `vasp=True`.
- **Per-image calculator instances**: Each NEB image gets its own calculator with a unique working directory (`VASP_{job_id}_{image_idx}/`). The `VASP_` prefix makes these directories easy to identify and safely glob for cleanup on resume. Endpoints and intermediates can use different `command`/`ncore` settings (e.g., more cores for endpoint relaxation).
- **Deferred instantiation**: `load_calculator()` returns the VASP class (not an instance). Calculators are instantiated per-image in `nebopt.py` with directory, command, ncore, and `[Vasp]` INCAR parameters.
- **VaspInteractive lifecycle**: VaspInteractive keeps a persistent VASP process. Endpoint calculators are finalized after relaxation (before replacing with `SinglePointCalculator`). Intermediate calculators are finalized after the NEB run completes.
- **Cleanup**: WAVECAR, CHG, CHGCAR files are removed after each job (both success and error paths) to avoid disk bloat. Image directories are added to `temp_files` for zip archival.
- **VASP config**: INCAR parameters live in the `[Vasp]` section regardless of whether the calculator is `Vasp` or `VaspInteractive`. VASP-specific NEB settings (`vasp_command_endpoints`, `vasp_ncore_endpoints`, `vasp_command_intermediates`, `vasp_ncore_intermediates`) live in `[ourNEB]`.
- **Current scope**: VASP support is implemented for NEB only. Dimer, Minimization, and DoubleMinimization still require FAIRChemCalculator.

## DNEB Theory Notes

The Doubly Nudged Elastic Band adds a perpendicular spring force component to straighten the band during convergence. The switching function (Eq. 15 from Henkelman & Jonsson) turns off this force as `|F_perp|` drops below `|F_S_perp|`, preventing the frustration problem where the straightening force fights against convergence on curved MEPs. The switched DNEB (swDNEB) implementation is in `catsunami/ocpneb.py`.
