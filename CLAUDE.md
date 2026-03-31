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
- `get_trajes_and_indices()`: Recursively scans dir_path (including subdirectories) for .traj files, splits into job batches
- Resume support: `get_remaining_trajes()` skips completed jobs

### `nebopt.py` - NEB Workflow
1. **Continuation check**: If images carry `_continuation` flag in `orig_info` (set by `continue_from_result`), locally overrides `relax_endpoints = False` and `interpolate_method = False` — skips steps 2-3 below and goes straight to NEB optimization from the extracted band.
2. **Endpoint relaxation** (optional): Relaxes reactant/product with configurable optimizer (e.g., LBFGS). For VaspInteractive, calculators are finalized after relaxation and endpoints are frozen with `SinglePointCalculator`.
3. **Interpolation**: `ocp_idpp` (Meta's PBC-aware), `ase_idpp`, `ase_linear` (auto-falls back to IDPP on atom overlap), or `False` (use provided frames)
4. **NEB optimization**: Uses `OCPNEB` class with MDMin optimizer. Supports climbing image and intermediate minima detection (per-segment climbing). For VASP/VaspInteractive, each image gets its own calculator instance with separate working directories (`VASP_{job_id}_{image_idx}/`), separate `command`/`ncore` settings for endpoints vs intermediates, and WAVECAR/CHG/CHGCAR cleanup after completion. Intermediate minima reclassification is periodic (every `intermediate_minima_check_interval` steps), not per-step, to prevent oscillation. Automatic image addition: when `max_num_frames > num_frames`, the optimization periodically checks for unconverged segments and doubles their images (FAIRChem only).
5. **Output**: For each sub-band (segment between intermediate minima), extracts the climbing image (TS candidate) with tangent vector as eigenmode, per-segment barrier/dE, and full sub-band data (`NEB_images` positions, `image_energies`, `image_fmax`). Generates band plot PNG. Per-image effective fmax is tracked throughout optimization and stored in debug trajs.

### `catsunami/ocpneb.py` - Core NEB Engine
- **`OCPNEB`** (extends BaseNEB): Two modes controlled by `vasp` flag:
  - **FAIRChem mode** (`vasp=False`): Batch-evaluates intermediate images via FAIRChemCalculator for efficiency. Caches forces between calls. fairchem imports are lazy (only loaded in this mode).
  - **VASP mode** (`vasp=True`): Delegates to parent `BaseNEB` for standard per-image force evaluation. Each image has its own VASP calculator. No batching, no caching, no fairchem dependency at runtime. When `intermediate_minima=True`, VASP mode bypasses `BaseNEB.get_forces()` and evaluates images individually so it can route through the custom `get_precon_forces()` where the intermediate minima logic lives.
  - Both modes: Handles constraints (fixed atoms by tag=0 or explicit constraints). Stores full `real_forces` array `(nimages, natoms, 3)` including endpoint forces for uniform access via `real_forces[imax]`.
- **Intermediate minima support**: When `intermediate_minima=True`, `get_precon_forces()` periodically (every `intermediate_minima_check_interval` force calls) scans images 2 through `nimages-3` for local energy minima (energy lower than both neighbors by at least `intermediate_minima_min_depth`). Between checks, the classification is frozen. Detected minima receive full PES forces (no spring forces, no tangent projection) so they relax freely into the energy basin. The band is split into segments at these minima, and each segment gets its own climbing image (highest-energy interior image in the segment, recomputed every step from current energies). `imax` is set to the global highest-energy climbing image. Endpoint-adjacent images (1 and `nimages-2`) are excluded from minima detection to ensure each segment has room for a climbing image.
- **Per-image effective fmax**: After applying NEB force modifications in `get_precon_forces()`, computes `max|F|` for each image (regular: NEB-modified force with spring; climbing: PES force with doubled reversed tangential; imin: pure PES force). Stored in `self.image_fmax` (array) and on each `image.info['effective_fmax']` (float, written to debug traj for history tracking).
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
- **Continuation mode**: When `atoms_orig` is a list (from `continue_from_result`), `dimeropt` bypasses `get_attempts()` and uses `_continuation_iter()` instead. This generator extracts each attempt's eigenmode, reaction_type, and selected_index from `orig_info`, builds a negligible displacement, and yields the same `(attempt, (atoms, disp_dict, selected_index))` tuples as normal mode. The existing loop body handles both modes transparently — eigenmode is passed to `MinModeAtoms` via `eigenmodes=` kwarg when present in `atoms.info`.

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
| `initial_guess` | `get_initial_guess_attempts()` | No displacement — dimer starts from input geometry as-is (supercell expansion is skipped). For pre-prepared TS guesses. Exclusive: ignores other types with warning. Always 1 attempt. Works with both `bulk` and `oc` dataset types. If the input structure has an eigenmode (in `atoms.info['eigenmode']` or `atoms.info['orig_info']['eigenmode']`), it is passed to `MinModeAtoms` to seed the dimer instead of a random guess. | 0 (none) |

**Key infrastructure:**
- `find_interstitial_sites(atoms)`: Voronoi tessellation on 3x3x3 periodic images → filter by min distance from atoms → cluster within 0.5 Å. Uses `scipy.spatial.Voronoi` and `scipy.cluster.hierarchy`.
- `_mic_vector()` / `_nearest_site()`: Minimum image convention helpers for periodic distance calculations.
- `_find_ring(neighbors_dict, seed, ring_size)`: Finds closed rings of connected atoms in the neighbor graph via constrained random walk. Ring size=2 handles pairwise exchange, ring size>=3 handles cooperative ring rotations.
- Element sampling: `hop_insert` uses small atoms weighted by 1/covalent_radius (H heavily favored). `kickout_insert` uses Gaussian weight centered on host avg covalent radius (σ=0.2 Å) from a pool of 30 common metals/semiconductors.
- `turn_into_supercell(atoms, min_length=7.0)`: Preserves `.info` across `make_supercell()` (ASE's `make_supercell` drops `.info`). Enforces minimum cell dimension of 7 Å in each periodic direction to avoid self-interaction artifacts through PBC (important for fairchem's radius graph which uses a 5 Å cutoff). Called centrally in `get_attempts()` for all reaction types except `initial_guess` (controlled by `supercell` config option, default `True`).
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
- **Continue-from-result extraction** (used by `__main__.py` when `continue_from_result = True`):
  - `extract_previous_results(job_ids, config_dict)`: Dispatcher that calls method-specific extractors. Returns `{job_id: images}` dict. Jobs with no extractable result are omitted (fall back to original input).
  - `_extract_neb_band(job_id, num_frames, debug_zip_index)`: Extracts the final NEB band (last `num_frames` frames) from debug traj — tries loose file first, then searches debug zips.
  - `_extract_dimer_attempts(job_id, output_traj_index)`: Extracts ALL successful attempt results for a job_id from output trajectories (one Atoms per attempt that produced output).
  - `_extract_minimization_structure(job_id, output_traj_index)`: Extracts the relaxed structure from output trajectory.
  - `_build_debug_zip_index(method_name)`: Scans all debug zips once, builds `{filename: zip_path}` map.
  - `_build_output_traj_index(method_name)`: Scans output trajectories, builds `{src_index: [(traj_path, frame_idx, info), ...]}` map.
  - `_sanitize_with_continuation(atoms)`: Wraps `.info` into `orig_info` (like `load_and_sanitize`) and sets `_continuation = True` flag so method functions can detect continuation images.
- Bond detection via ASE neighbor_list with natural cutoffs
- `check_reaction()` / `check_adsorbate_reaction()`: Compare connectivity between structures

### `init_function.py` - Worker Initialization
- Assigns GPU to worker based on executorlib_worker_id and jobs_per_gpu
- For multi-job-per-GPU (`jobs_per_gpu > 1`): auto-detects per-GPU MPS daemons (`/tmp/mps_{gpu}/control`). If MPS is running, sets `CUDA_MPS_PIPE_DIRECTORY` and `CUDA_VISIBLE_DEVICES=0`; otherwise falls back to direct `CUDA_VISIBLE_DEVICES` assignment. Skipped for VASP calculators.
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
continue_from_result = True # On resume, start from previous result instead of original input (see below)
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
intermediate_minima = True         # Detect intermediate minima and do per-segment climbing image NEB
intermediate_minima_check_interval = 100  # Re-evaluate intermediate minima classification every N force calls (first check always runs on the very first force call)
intermediate_minima_min_depth = 0.05   # Min energy dip (eV) below both neighbors to count as intermediate minimum
max_num_frames = 80                # Max total band size (enables automatic image addition if > num_frames, FAIRChem only)
add_images_check_interval = 100    # Check for image addition every N optimizer steps
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
| `converged` | At least one attempt converged | `converged`, `converged_after_extension`, `converged_both`, `converged_min1`, `converged_min2`, `converged_only_CI` |
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

**Archiving on resume**: When redoing jobs that have existing results, all output files are archived and cleaned:
- **CSVs**: Archived as `{method}_status_csvs/previous_{N}.zip`, entries for re-run job IDs removed from active CSVs.
- **Output trajectories**: Archived as `{method}_trajes/previous_{N}.zip`, frames for re-run job IDs removed (filtered by `src_index`).
- **Debug zips**: Archived as `{method}_debug_zips/previous_{N}.zip`, entries for re-run job IDs removed (filtered by job_id in filename via regex).

Jobs with no prior entries (e.g., `not_started`) don't trigger archiving. The `previous_*.zip` archives in debug_zips are excluded from extraction scans (`_build_debug_zip_index` skips them).

**Fresh vs resume**: No explicit toggle — if `traj_files_ordered.json` doesn't exist, it's a fresh start; if it exists, it's a resume. To force a fresh start, delete the output directories and `traj_files_ordered.json`.

### `continue_from_result` — Continue from Previous Result

When `continue_from_result = True` (default) and re-running previously completed jobs (any `run_jobs` category except `not_started`), the system extracts previous results and uses them as starting points instead of the original input trajectories.

Continuation is handled **per-job** inside each method function — no global config mutation. This means fresh jobs (`not_started`) and continuation jobs can coexist in the same batch without interference.

**How it works per method:**

| Method | What is extracted | What happens |
|---|---|---|
| NEB | Full band from debug traj (`neb_{job_id}.traj` in debug zips) | `nebopt` detects `_continuation` flag, locally skips endpoint relaxation and interpolation, continues NEB optimization from the extracted band |
| Dimer | All successful attempt results from output traj (one per attempt) | `dimeropt` detects list input, bypasses `get_attempts()`, continues each attempt from its previous structure with its original eigenmode and reaction_type |
| Minimization | Relaxed structure from output traj | Optimizer continues from previous positions |
| DoubleMinimization | Not supported (raises error) | — |

**Dimer per-attempt continuation**: Each attempt that produced output (converged or not_converged) is continued individually. The original `reaction_type` (e.g., `"vacancy"`) is preserved in the output — not replaced with `"initial_guess"`. The previous eigenmode is passed to `MinModeAtoms` to seed the dimer. Errored attempts (which have no output frame) are skipped. The configured `reaction_types` in `[ourDimer]` is ignored for continuation jobs.

**Fallback**: If previous results cannot be extracted (missing debug files, all attempts errored), the job silently falls back to the original input and runs from scratch. In `__main__.py`, this happens naturally: jobs missing from `previous_results` dict use `load_and_sanitize()` to load the original input.

**`initial_guess` vs `continue_from_result`**: These are separate features. `initial_guess` is a Dimer reaction type for fresh runs with **external** TS guesses (from another code, a database, or a different tsearch method like NEB→Dimer). It requires the user to set `reaction_types = initial_guess` and provide input `.traj` files. `continue_from_result` is an automatic internal mechanism that extracts results from a **previous tsearch run** and feeds them back. It bypasses `get_attempts()` entirely and does not use `initial_guess`. Use `initial_guess` when you bring a TS from outside tsearch; use `continue_from_result` when you want tsearch to pick up where it left off.

**Examples:**
```ini
# Continue unconverged NEB jobs with more steps
run_jobs = not_converged
steps = 10000

# Refine converged dimer results with a better calculator
run_jobs = converged
Calculator = VaspInteractive

# Re-run from scratch (ignore previous results)
run_jobs = not_converged
continue_from_result = False
```

## Output Structure

For method `NEB`:
```
NEB_status_csvs/status_rank_*.csv    # job_id,rank,sub_band_id,status (converged/converged_only_CI/not_converged/error)
NEB_trajes/collected_ts_rank_*.traj  # TS candidates (one per sub-band) with metadata in atoms.info
NEB_debug_zips/                      # Compressed log/traj/plot files (images have effective_fmax in .info)
```

Each TS image (one per sub-band/segment) in the output trajectory contains:
- `eigenmode`: Tangent vector at saddle point
- `barrier`: Forward barrier within segment (eV)
- `dE`: Reaction energy within segment (eV)
- `converged`: 1 if all images in sub-band below fmax, 0 otherwise
- `ci_converged`: 1 if this CI specifically below fmax, 0 otherwise
- `segment_id`: Sub-band index (0 when no intermediate minima)
- `NEB_images`: 3D array `(n_subband_images, natoms, 3)` — positions of all images in the sub-band
- `image_energies`: List of floats — per-image energies in the sub-band
- `image_fmax`: List of floats — per-image effective fmax in the sub-band
- `nimages`: Total band size (for continue_from_result extraction)
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
