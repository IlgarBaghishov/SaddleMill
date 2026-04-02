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
1. **Continuation/sub-band check**: If `entries_to_run` is set (per-sub-band redo), overrides starting images from `continuation_data`. All NEB settings (relax_endpoints, interpolate_method, num_frames, etc.) always come from config.ini. If `continue_from_result=True`, uses extracted images directly (sub-band or de-duplicated full band). If `False`, sub-bands re-interpolate between extracted endpoints using config's `num_frames`; full-band falls through to fresh run with original input. Intermediate minima detection is only enabled on fresh full-band runs; full-band `continue_from_result=True` seeds the imin set from extracted `image_type` metadata (frozen, no re-scanning); sub-band reruns disable imin entirely. Image addition respects config but sub-band total is capped at `num_frames`.
2. **Endpoint relaxation** (optional): Relaxes reactant/product with configurable optimizer (e.g., LBFGS). For VaspInteractive, calculators are finalized after relaxation and endpoints are frozen with `SinglePointCalculator`.
3. **Interpolation**: `ocp_idpp` (Meta's PBC-aware), `ase_idpp`, `ase_linear` (auto-falls back to IDPP on atom overlap), or `False` (use provided frames)
4. **NEB optimization**: Uses `OCPNEB` class with MDMin optimizer. Supports climbing image and intermediate minima detection (per-segment climbing). For VASP/VaspInteractive, each image gets its own calculator instance with separate working directories (`VASP_{job_id}_{image_idx}/`), separate `command`/`ncore` settings for endpoints vs intermediates, and WAVECAR/CHG/CHGCAR cleanup after completion. Intermediate minima reclassification is periodic (every `intermediate_minima_check_interval` steps), not per-step, to prevent oscillation. Automatic image addition: when `max_num_frames > num_frames`, the optimization periodically checks for unconverged segments and doubles their images (FAIRChem only). **Auto-freeze** (FAIRChem only, always on): at every step, OCPNEB checks for converged sub-bands/imin/CIs and adds them to `_frozen_set`. Frozen images get zero NEB forces (don't move), are skipped in FAIRChem batching after first evaluation (cached energy/forces), and act as segment boundaries for climbing selection. Only sub-bands (all images below threshold), individual imin (below `endpoint_fmax`), and individual CIs (below NEB `fmax`) can be frozen — never individual regular images.
5. **Post-NEB refinement** (optional, FAIRChem only): If the band is not fully converged: (a) `dimer_refine_ci=True`: runs a quick ASE dimer on each unconverged CI, seeded with the NEB tangent vector as eigenmode. Uses `[DimerControl]` config for dimer parameters. (b) `refine_band_steps > 0` with `intermediate_minima=True`: relaxes unconverged intermediate minima using the endpoint optimizer to `endpoint_fmax`. (c) If `refine_band_steps > 0` and any new extrema converged: continues NEB for `refine_band_steps` steps with `climb=False`, `intermediate_minima=False`, and the frozen set carrying over (newly converged extrema are frozen). Uses `_setup_dimer()` from `dimeropt.py` (shared helper, no code duplication).
6. **Output**: Writes ALL band images as separate frames to the output trajectory. Each image gets `src_index`, `image_idx`, `subband_idx`, `image_type` (endpoint/intermediate_minimum/climbing/regular), `effective_fmax`, `image_converged`, `band_converged`, `band_converged_CI`. CI images additionally get `eigenmode`, `barrier`, `dE`. Intermediate minima images are duplicated so each sub-band is self-contained. Generates band plot PNG. Per-sub-band NEB runs use temp files with `_sub{subband_idx}` suffix.

### `catsunami/ocpneb.py` - Core NEB Engine
- **`OCPNEB`** (extends BaseNEB): Two modes controlled by `vasp` flag:
  - **FAIRChem mode** (`vasp=False`): Batch-evaluates intermediate images via FAIRChemCalculator for efficiency. Caches forces between calls. fairchem imports are lazy (only loaded in this mode).
  - **VASP mode** (`vasp=True`): Delegates to parent `BaseNEB` for standard per-image force evaluation. Each image has its own VASP calculator. No batching, no caching, no fairchem dependency at runtime. When `intermediate_minima=True` or `initial_imin_set` is provided, VASP mode bypasses `BaseNEB.get_forces()` and evaluates images individually so it can route through the custom `get_precon_forces()` where the intermediate minima logic lives.
  - Both modes: Handles constraints (fixed atoms by tag=0 or explicit constraints). Stores full `real_forces` array `(nimages, natoms, 3)` including endpoint forces for uniform access via `real_forces[imax]`.
- **Intermediate minima support**: When `intermediate_minima=True`, `get_precon_forces()` periodically (every `intermediate_minima_check_interval` force calls) scans images 2 through `nimages-3` for local energy minima (energy lower than both neighbors by at least `intermediate_minima_min_depth`). Between checks, the classification is frozen. Detected minima receive full PES forces (no spring forces, no tangent projection) so they relax freely into the energy basin. The band is split into segments at these minima, and each segment gets its own climbing image (highest-energy interior image in the segment, recomputed every step from current energies). `imax` is set to the global highest-energy climbing image. Endpoint-adjacent images (1 and `nimages-2`) are excluded from minima detection to ensure each segment has room for a climbing image. Alternatively, `initial_imin_set` can seed known imin positions (used for full-band `continue_from_result=True`); the seeded set is frozen (no periodic re-scanning) and imin images receive the same pure PES force treatment.
- **Frozen images support**: `frozen_images` constructor param provides an initial set; `freeze_fmax`/`freeze_endpoint_fmax` enable auto-freeze at every step via `_auto_freeze()`. Frozen images: (1) get zero NEB forces in `get_precon_forces()`, (2) are skipped in FAIRChem batching after first evaluation (energy/forces cached in `_frozen_energies`/`_frozen_pbc_forces`), (3) are excluded from intermediate minima detection along with their neighbors, (4) act as segment boundaries for climbing image selection. Only entire converged sub-bands, individual converged imin, and individual converged CIs can be frozen — never individual regular images. `_auto_freeze()` runs after fmax computation; newly frozen images take effect on the next force call.
- **Per-image effective fmax**: After applying NEB force modifications in `get_precon_forces()`, computes `max|F|` for each image (regular: NEB-modified force with spring; climbing: PES force with doubled reversed tangential; imin: pure PES force; frozen: 0.0). Stored in `self.image_fmax` (array) and on each `image.info['effective_fmax']` (float, written to debug traj for history tracking).
- **`swDNEB`** (NEBMethod subclass): Implements the switched Doubly Nudged Elastic Band method (works with both FAIRChem and VASP modes):
  - Uses improved tangent vectors (energy-weighted at extrema)
  - Adds perpendicular spring force component to straighten the band
  - Switching function `sw = (2/pi) * arctan(|F_perp|^2 / |F_S_perp|^2)` turns off DNEB force as convergence is reached (preventing frustration)
  - Based on: Henkelman & Jonsson, J. Chem. Phys. (2000) and Trygubenko & Wales (2004)

### `dimeropt.py` - Dimer Method
- **`_setup_dimer()`**: Shared helper that creates `MinModeAtoms` + `MinModeTranslate` from ASE's dimer API. Used by both `dimeropt()` (full dimer workflow) and `nebopt()` (quick CI refinement). Does not run the optimization — returns `(d_atoms, dim_rlx)` for the caller to attach callbacks and run.
- Generates displacement candidates via `dimertools/structure_edit.py`
- Supports `bulk` (multiple reaction types, see below) and `oc` (adsorbate-targeted, see below) modes. Both use `reaction_types` + `num_attempts_per_type` config.
- Convergence checks every 5 steps: participation ratio (delocalization) and desorption detection
- **Desorption flagging**: When the desorption check triggers a `StopRun`, the `reaction_type` in `atoms.info` is overridden to `'desorption'` (regardless of the initialization type). This allows distinguishing desorption events from other stop reasons (e.g., delocalization) in the output trajectories.
- Extension check if initial convergence fails
- Writes `reaction_type` to `atoms.info` for each attempt
- **Per-attempt error handling**: Each dimer attempt has its own try/except, so one failing attempt does not abort remaining attempts for the same structure
- **Consecutive error tracking**: Tracks structure-level errors via `consecutive_errors` counter (passed from `init_function`). If all attempts for a structure fail, counter increments; any successful attempt resets it to 0. When counter reaches `max_consecutive_errors`, worker calls `sys.exit(1)` to trigger executorlib restart (see Worker Health section)
- **Per-attempt execution**: `dimeropt` accepts `entries_to_run` (set of attempt_ids) and `continuation_data` (dict: attempt_id → Atoms) via kwargs. Always calls `get_attempts()` on the original input, then for each attempt: skips if not in `entries_to_run`, uses `continuation_data[attempt_id]` if available (with near-zero displacement), otherwise runs the fresh-generated attempt. Eigenmode and reaction_type are read from top-level `.info` first, then `orig_info` fallback (following the `.info` handling rule).

### `dimertools/structure_edit.py` - Reaction Types for Dimer
Dimer mode supports named reaction types, configured via `reaction_types` (space-separated list). Bulk and OC dataset types have separate type sets dispatched via `_BULK_REACTION_TYPE_DISPATCH` and `_OC_REACTION_TYPE_DISPATCH` respectively.

**Bulk reaction types** (`dataset_type = bulk`):

| Type | Function | Description | Atoms displaced |
|------|----------|-------------|-----------------|
| `vacancy` | `get_vacancy_attempts()` | Remove atom, neighbor hops into vacancy | 1 (center-based) |
| `hop_reuse` | `get_hop_reuse_attempts()` | Existing atom relocated to interstitial site | 1 (vector) |
| `hop_insert` | `get_hop_insert_attempts()` | New small atom (H/C/N/O/B) inserted at interstitial site | 1 (vector) |
| `kickout_reuse` | `get_kickout_reuse_attempts()` | Existing atom placed at interstitial, kicks nearest lattice atom into another interstitial | 2 (vector) |
| `kickout_insert` | `get_kickout_insert_attempts()` | New similar-sized atom inserted at interstitial, kicks nearest lattice atom | 2 (vector) |
| `ring` | `get_ring_attempts()` | Ring of 2+ atoms rotate cooperatively; size randomly sampled from `ring_sizes` config. Use `ring_sizes = 2` for pairwise exchange. | N (vector) |
| `initial_guess` | `get_initial_guess_attempts()` | No displacement — dimer starts from input geometry as-is (supercell expansion is skipped). For pre-prepared TS guesses. Exclusive: ignores other types with warning. Always 1 attempt. Works with both `bulk` and `oc` dataset types. If the input structure has an eigenmode (in `atoms.info['eigenmode']` or `atoms.info['orig_info']['eigenmode']`), it is passed to `MinModeAtoms` to seed the dimer instead of a random guess. | 0 (none) |

**OC reaction types** (`dataset_type = oc`):

Adsorbate atoms are identified by tag=2 (fallback tag=1). Substrate atoms (tag=0) are always fixed via `FixAtoms`. Both bulk and OC use `reaction_types` + `num_attempts_per_type` for configuration.

| Type | Function | Description | Displacement mechanism |
|------|----------|-------------|----------------------|
| `adsorbate_atom` | `get_adsorbate_atom_attempts()` | Tight Gaussian on one adsorbate atom — only that atom moves | `displacement_center` + `gauss_std=0.2, number_of_atoms=1` |
| `adsorbate_atom_neighbors` | `get_adsorbate_atom_neighbors_attempts()` | Broad Gaussian on one adsorbate atom — nearby atoms also displaced | `displacement_center` (default DimerControl std) |
| `adsorbate` | `get_adsorbate_attempts()` | Random noise on all adsorbate atoms (internal rearrangement, isomerization) | Adsorbate-only mask |
| `diffusion` | `get_diffusion_attempts()` | Uniform translation of all adsorbate atoms in a random 3D direction (molecular migration, desorption) | `displacement_vector` (same direction for all adsorbate atoms, magnitude ~0.1 A) |
| `rotation` | `get_rotation_attempts()` | Rigid-body rotation of adsorbate around center of mass (molecular reorientation). Skips with warning for single-atom adsorbates. | `displacement_vector` (tangential, random axis, ~0.05 rad) |
| `adsorbate_surface` | `get_adsorbate_surface_attempts()` | Random noise on adsorbate + neighboring substrate atoms (interface reactions, subsurface penetration) | Adsorbate+neighbors mask (natural_cutoffs * 1.25) |
| `surface` | `get_surface_attempts()` | Broad Gaussian on one surface atom (tag=1) — surface reconstruction | `displacement_center` (default DimerControl std) |
| `custom` | `get_custom_attempts()` | No overrides — displacement fully controlled by `[DimerControl]` settings (gauss_std, displacement_radius, displacement_method, etc.) | Empty dict (pure DimerControl defaults) |
| `initial_guess` | `get_initial_guess_attempts()` | Same as bulk `initial_guess` (see above) | 0 (none) |

**OC helper functions:**
- `_get_oc_adsorbate_indices(atoms)`: Returns indices of adsorbate atoms (tag=2, fallback tag=1).
- `_get_oc_neighbor_mask(atoms, adsorbate_indices)`: Builds neighbor mask using natural_cutoffs * 1.25.
- `_sample_adsorbate_atoms(adsorbate_indices, num_needed)`: Samples N atom indices with cycling if fewer available.

**Key infrastructure:**
- `find_interstitial_sites(atoms)`: Voronoi tessellation on 3x3x3 periodic images → filter by min distance from atoms → cluster within 0.5 Å. Uses `scipy.spatial.Voronoi` and `scipy.cluster.hierarchy`.
- `_mic_vector()` / `_nearest_site()`: Minimum image convention helpers for periodic distance calculations.
- `_find_ring(neighbors_dict, seed, ring_size)`: Finds closed rings of connected atoms in the neighbor graph via constrained random walk. Ring size=2 handles pairwise exchange, ring size>=3 handles cooperative ring rotations.
- Element sampling: `hop_insert` uses small atoms weighted by 1/covalent_radius (H heavily favored). `kickout_insert` uses Gaussian weight centered on host avg covalent radius (σ=0.2 Å) from a pool of 30 common metals/semiconductors.
- `turn_into_supercell(atoms, min_length=7.0)`: Preserves `.info` across `make_supercell()` (ASE's `make_supercell` drops `.info`). Enforces minimum cell dimension of 7 Å in each periodic direction to avoid self-interaction artifacts through PBC (important for fairchem's radius graph which uses a 5 Å cutoff). Called centrally in `get_attempts()` for all reaction types except `initial_guess` (controlled by `supercell` config option, default `True`).
- Dispatch: `_BULK_REACTION_TYPE_DISPATCH` and `_OC_REACTION_TYPE_DISPATCH` dicts map type names to functions. `get_attempts()` selects the appropriate dict based on `dataset_type`. `initial_guess` is handled early (before bulk/oc branch) since it works with both dataset types.
- `_safe_normalize(vec)`: Normalizes a vector; returns a random unit vector if norm is near zero (prevents division-by-zero in ring displacement calculations).
- `_build_neighbor_dict(atoms)`: Builds neighbor graph; skips self-interactions (`i == j`) which can occur in small periodic cells.
- Defensive guards throughout: zero-weight fallback in `_sample_kickout_insert_element`, empty `ring_sizes` early return, missing adsorbate atoms early return in OC mode.

### `geomopt.py` - Geometry Optimization
- `geomopt()`: Standard relaxation with optional cell relaxation (FrechetCellFilter)
- `doublegeomopt()`: Takes converged TS with eigenmode, displaces +/- 0.25*eigenmode, relaxes both directions, detects bond breaking/forming via `check_reaction()`. Reads `eigenmode`, `converged`, `src_index` from `atoms.info['orig_info']` (with fallback to `atoms.info`). Each frame gets `side` in `.info` (-1=min1, 0=ts, 1=min2). Writes 2 CSV status lines per job (one per side). Accepts `entries_to_run` (set of side_ids) and `continuation_data` (dict: side → Atoms) for per-side execution — skips converged sides and uses them from `continuation_data` for reaction check.

### `tools.py` - Utilities
- `load_and_sanitize(traj, i, j)`: Loads atoms from trajectory and stashes original `.info` into `atoms.info = {"orig_info": <original_info>}`. This prevents per-atom array data (e.g. `forces`, `stress`) from causing size mismatches when atoms are later added/removed (e.g. vacancy in Dimer). Called in `__main__.py` for all methods uniformly.
- `clean_up_files(config_dict)`: Removes leftover temp files on resume. Method-aware: cleans NEB files (`neb_*.log`, `neb_*.traj`, `reactant_relaxation_*`, `product_relaxation_*`, `diffusion_barrier_*.png`), Dimer files (`dimer_control_*.log`, `dimer_opt_*.log`, `dimer_*.traj`), or Minimization files (`optimization_*.log`, `optimization_*.traj`). For VASP NEB, also removes `VASP_*_*/` per-image directories.
- **Previous result extraction** (used by `__main__.py` when redoing previously-run jobs):
  - `extract_previous_results(job_ids, config_dict, redo_info)`: Unified extraction from output trajs for ALL methods. Returns `{job_id: continuation_data}` where continuation_data is: Dimer `{attempt_id: Atoms}`, NEB `{subband_idx: [Atoms sorted by image_idx]}`, DoubleMinimization `{side: Atoms}`, Minimization `Atoms`.
  - `_build_output_traj_index(method_name)`: Scans output trajectories, builds `{src_index: [Atoms]}` map.
  - `_sanitize_with_continuation(atoms)`: Wraps `.info` into `orig_info` (like `load_and_sanitize`) for extracted results.
- Bond detection via ASE neighbor_list with natural cutoffs
- `check_reaction()` / `check_adsorbate_reaction()`: Compare connectivity between structures

### `.info` Handling Rule (applies to all methods)

`orig_info` is set once at the entry point (`load_and_sanitize` for fresh runs, `_sanitize_with_continuation` for continuations) and **never modified after that**. It contains whatever `.info` was before this run started. Methods write their output keys to the top level of `.info`. When a method needs data that may come from a previous run (e.g., eigenmode, reaction_type), it checks the top level first, then falls back to `orig_info`. On continuation runs, `orig_info` nests (each layer contains the previous run's full `.info`, which itself has an `orig_info` from the run before it).

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
run_jobs = remaining         # Which job categories to process (see below)
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
dimer_refine_ci = False            # After NEB: run quick dimer on unconverged CIs (FAIRChem only)
dimer_refine_steps = 300           # Max dimer optimization steps per CI
refine_band_steps = 0              # Steps for continuation NEB after extrema refinement (0 = disabled)
                                   # When > 0 with intermediate_minima=True, also relaxes unconverged imin
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
reaction_types = vacancy     # Bulk: vacancy hop_reuse hop_insert kickout_reuse kickout_insert ring initial_guess
                             # OC: adsorbate_atom adsorbate_atom_neighbors adsorbate diffusion rotation adsorbate_surface surface custom initial_guess
num_attempts_per_type = 1    # Attempts per reaction type; total = len(types) * num_per_type
ring_sizes = 3 4             # Ring sizes to sample from for 'ring' reaction type (bulk only)
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

**4 categories** (each CSV status line is independently categorized):

| Category | CSV statuses that map here |
|---|---|
| `converged` | `converged`, `converged_after_extension`, `converged_CI` |
| `not_converged` | `not_converged`, `not_converged_after_extension`, `not_converged_StopRun` |
| `errored` | `error`, `error: <message>` |
| `remaining` | No CSV row for this job_id |

**Per-line selection**: A job is included if ANY of its CSV lines match the requested categories. This means a Dimer job with 3 converged + 3 not_converged attempts is selectable by both `run_jobs = converged` and `run_jobs = not_converged`. Only the matching attempts/sub-bands/sides are redone; the rest stay in the active output.

**Examples:**
```ini
run_jobs = remaining                  # Default. Fresh run: all jobs. Resume: continue unfinished.
run_jobs = remaining errored          # Resume + retry errors
run_jobs = converged                  # Redo converged entries (e.g., refine with VASP)
run_jobs = not_converged              # Redo not-converged entries
run_jobs = errored                    # Retry only errors
run_jobs = all                        # Redo everything
```

**Archiving on resume**: Archive is always a full backup. Cleaning is per-entry:
- **CSVs**: Archived as `{method}_status_csvs/previous_{N}.zip`. Only CSV lines matching `run_jobs` categories are removed from active CSVs (per-line cleaning).
- **Output trajectories**: Archived as `{method}_trajes/previous_{N}.zip`. Only frames matching cleaned entries are removed (by `attempt_id` for Dimer, `subband_idx` for NEB, `src_index` for Minimization). For DoubleMinimization, all 3 frames (min1+TS+min2) are removed and re-written together since they share reaction check metadata.
- **Debug zips**: Archived as `{method}_debug_zips/previous_{N}.zip`. Per-entry cleaning where filenames contain subunit info (Dimer: `attempt_id`, DoubleMinimization: side via `-0`/`-1` suffix, NEB sub-band reruns: `_sub{idx}` suffix). NEB full-band debug files (no `_sub` suffix) are removed as a whole when any sub-band is redone since they lack per-sub-band granularity.

Jobs with no prior entries (e.g., `remaining`) don't trigger archiving.

**Fresh vs resume**: No explicit toggle — if `traj_files_ordered.json` doesn't exist, it's a fresh start; if it exists, it's a resume. To force a fresh start, delete the output directories and `traj_files_ordered.json`.

### `continue_from_result` — Continue from Previous Result

When re-running previously completed entries (any `run_jobs` category except `remaining`), the system always extracts previous results from output trajs (needed for sub-band endpoints, kept sides, etc.). The `continue_from_result` flag controls how extracted data is used:

- `True` (default): continue optimization from the extracted structure.
- `False`: start fresh. For Dimer: generates fresh attempts at the same `attempt_id` positions (same reaction types). For NEB partial sub-bands: re-interpolates between sub-band endpoints. For NEB all sub-bands: uses original input. For DoubleMinimization: re-displaces from TS.

Errored entries always fall back to fresh generation (no output to continue from).

Continuation is handled **per-entry** inside each method function — no global config mutation. Fresh jobs (`remaining`) and continuation jobs coexist in the same batch.

**How it works per method:**

| Method | Granularity | `continue_from_result=True` | `continue_from_result=False` |
|---|---|---|---|
| Dimer | per-attempt | Continue from extracted attempt structure with its eigenmode | Generate fresh attempt at same `attempt_id` (same reaction type, new random displacement) |
| NEB | per-sub-band | Continue from extracted sub-band images. Full-band: seeds imin from previous `image_type`. All NEB settings from config. | Partial: re-interpolate between sub-band endpoints using config's `num_frames`. All sub-bands: use original input. All NEB settings from config. |
| DoubleMinimization | per-side | Continue not-converged side from extracted last state | Re-displace from TS (ts ± eigenmode) |
| Minimization | per-job | Continue from extracted relaxed structure | Use original input |

All methods extract from output trajs uniformly via `extract_previous_results`. Errored entries always fall back to fresh (no output to extract).

**`initial_guess` vs `continue_from_result`**: These are separate features. `initial_guess` is a Dimer reaction type for fresh runs with **external** TS guesses (from another code, a database, or a different tsearch method like NEB→Dimer). `continue_from_result` is an automatic internal mechanism that extracts results from a **previous tsearch run**. Use `initial_guess` when you bring a TS from outside tsearch; use `continue_from_result` when you want tsearch to pick up where it left off.

**Examples:**
```ini
# Continue not-converged NEB jobs with more steps
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
NEB_status_csvs/status_rank_*.csv    # job_id,rank,sub_band_id,status (converged/converged_CI/not_converged/error)
NEB_trajes/collected_ts_rank_*.traj  # All band images (full band, one frame per image) with metadata in atoms.info
NEB_debug_zips/                      # Compressed log/traj/plot files
```

Each band image in the output trajectory contains:
- `src_index`: Job ID
- `image_idx`: Position in full band (0 to N-1)
- `subband_idx`: Which sub-band this image belongs to (0, 1, 2, ...)
- `image_type`: `"endpoint"` / `"intermediate_minimum"` / `"climbing"` / `"regular"`
- `effective_fmax`: Per-image force max (float)
- `image_converged`: Whether this image's fmax < threshold (bool)
- `band_converged`: Whether all images in this sub-band converged (bool)
- `band_converged_CI`: Whether the CI of this sub-band converged (bool)
- `nimages`: Total band size
- `interpolation_method`: Interpolation method used
- `eigenmode`: (CI images only) Tangent vector at saddle point
- `barrier`: (CI images only) Forward barrier within segment (eV)
- `dE`: (CI images only) Reaction energy within segment (eV)
- `orig_info`: Original `.info` dict from the input trajectory

Intermediate minima images are duplicated so each sub-band is self-contained (has its own endpoints). Per-sub-band redo temp files use `_sub{subband_idx}` suffix.

For method `Dimer`, each output frame contains: `eigenmode`, `curvature`, `converged`, `src_index`, `attempt_id`, `stoprun`, `selected_index`, `reaction_type`, `orig_info`.

For method `DoubleMinimization`, each output frame contains: `side` (-1=min1, 0=ts, 1=min2), `parent_ts_index`, `converged`, `src_index`, `is_reaction`, `n_formed_bonds`, `n_broken_bonds`, `orig_info`. CSV writes 2 lines per job (one per side): `{job_id},{rank},{side_id},{parent_ts_idx},{status}`.

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

## Testing

The project uses pytest with ~196 tests across 9 test files. Tests are split into CPU-only unit tests (run anywhere) and GPU integration tests (require CUDA).

### Setup

```bash
pip install pytest pytest-timeout
```

### Running Tests

```bash
# CPU-only unit tests (login nodes, no GPU needed, ~40s)
pytest -m "not gpu and not flux" -v

# All tests (GPU node required, ~6 min)
pytest -v --timeout=600

# Only GPU integration tests
pytest -m gpu -v

# Single test file
pytest tests/test_config.py -v
```

### Test Structure

```
tests/
├── conftest.py                  # Shared fixtures, skip logic, make_config_dict() helper
├── fixtures/                    # Small .traj files for tests
│   ├── bulk_crystal.traj        # 3-atom FCC crystal (bulk dimer tests)
│   ├── minimization_input.traj  # 68-atom slab+adsorbate (minimization tests)
│   ├── oc_adsorbate_slab.traj   # 131-atom slab+adsorbate (OC dimer tests)
│   └── oc_neb_pair.traj         # 10-frame NEB band, 68 atoms (NEB tests)
├── test_config.py               # ConfigManager, load_*, run_jobs, archive/clean (~74 tests, CPU)
├── test_tools.py                # load_and_sanitize, check_reaction, extract_previous_results (~28 tests, CPU)
├── test_structure_edit.py       # All bulk & OC reaction types, supercell, interstitials (~35 tests, CPU)
├── test_ocpneb.py               # swDNEB, _find_segment_ci, OCPNEB forces, frozen images (~20 tests, mixed)
├── test_nebopt_integration.py   # Full NEB runs: interpolation, DNEB, imin, dimer_refine (8 tests, GPU)
├── test_dimeropt_integration.py # Dimer: OC + bulk types, entries_to_run, error tracking (7 tests, GPU)
├── test_geomopt_integration.py  # geomopt + doublegeomopt: cell relax, continuation (5 tests, GPU)
├── test_init_function.py        # init_function return structure (5 tests, GPU)
└── test_main_integration.py     # End-to-end serial pipeline + resume logic (4 tests, GPU)
```

### Markers

- `@pytest.mark.gpu` — requires CUDA GPU (auto-skipped without one)
- `@pytest.mark.flux` — requires Flux scheduler
- `@pytest.mark.slow` — tests taking >60 seconds

### Writing New Tests

- Use `make_config_dict(method="NEB", **overrides)` from `tests/conftest.py` to build config dicts
- GPU tests use the session-scoped `fairchem_calc` fixture (loads model once, shared across tests)
- The `converged_ts_atoms` fixture generates a TS via Dimer for doublegeomopt tests
- All method integration tests create output dirs in `tmp_path` and `monkeypatch.chdir()` there
- CPU unit tests for structure_edit.py seed randomness with `random.seed(42)` + `np.random.seed(42)`

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
