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
- `ConfigManager`: Reads `config.ini` with type inference (bool/int/float/list/string). Quoted strings (`"..."` or `'...'`) preserved as literal strings (for VASP commands with spaces).
- `load_calculator()`: Returns calculator class/callable (`FAIRChemCalculator`, `Vasp`, or `VaspInteractive`). For FAIRChem, returns `from_model_checkpoint`; for VASP, returns the class itself. Instantiation deferred to `init_function.py` (FAIRChem) or `nebopt.py` (VASP, per-image). VASP support is NEB only; Dimer/Minimization/DoubleMinimization require FAIRChemCalculator.
- `load_method()`: Imports the correct optimization function
- `load_optimizer()`: Returns optimizer class(es) - for NEB returns (endpoint_optimizer, neb_optimizer)
- `get_trajes_and_indices()`: Recursively scans dir_path (including subdirectories) for .traj files, splits into job batches
- Resume support: `get_remaining_trajes()` skips completed jobs

### `nebopt.py` - NEB Workflow
1. **Continuation/sub-band check**: When `entries_to_run` is set, overrides starting images from `continuation_data`. All NEB settings always come from config.ini. `continue_from_result=True` uses extracted images directly; `False` re-interpolates sub-bands between endpoints or falls through to fresh run for full-band. Imin detection: enabled only on fresh full-band runs; full-band `continue_from_result=True` seeds imin from extracted `image_type` (frozen, no re-scanning); sub-band reruns disable imin. Image addition respects config but sub-band total capped at `num_frames`.
2. **Endpoint relaxation** (optional): Relaxes reactant/product with configurable optimizer (e.g., LBFGS). For VaspInteractive, calculators finalized after relaxation and endpoints frozen with `SinglePointCalculator`.
3. **Interpolation**: `ocp_idpp` (Meta's PBC-aware), `ase_idpp`, `ase_linear` (auto-falls back to IDPP on atom overlap), or `False` (use provided frames)
4. **NEB optimization**: Uses `OCPNEB` with MDMin optimizer. Supports climbing image and intermediate minima (per-segment climbing). For VASP/VaspInteractive: per-image calculators in `VASP_{job_id}_{image_idx}/` dirs, separate `command`/`ncore` for endpoints vs intermediates, WAVECAR/CHG/CHGCAR cleanup after completion. VaspInteractive endpoint calculators finalized before `SinglePointCalculator` replacement; intermediate calculators finalized after NEB completes. Imin reclassification periodic (every `intermediate_minima_check_interval` steps) to prevent oscillation. Auto image addition: when `max_num_frames > num_frames`, periodically doubles images in unconverged segments (FAIRChem only). **Auto-freeze** (FAIRChem only, always on): OCPNEB checks for converged sub-bands/imin/CIs each step, adds to `_frozen_set`. Frozen images get zero NEB forces, are skipped in FAIRChem batching after first eval (cached), and act as segment boundaries. Only sub-bands (all below threshold), individual imin (below `endpoint_fmax`), and individual CIs (below `fmax`) can freeze — never regular images.
5. **Post-NEB refinement** (optional, FAIRChem only): If band not fully converged: (a) `dimer_refine_ci=True`: quick ASE dimer on each unconverged CI, seeded with NEB tangent as eigenmode, using `[DimerControl]` config. (b) `refine_band_steps > 0` with `intermediate_minima=True`: relaxes unconverged imin to `endpoint_fmax`. (c) If `refine_band_steps > 0` and new extrema converged: continues NEB for `refine_band_steps` steps with `climb=False`, `intermediate_minima=False`, frozen set carrying over. Uses `_setup_dimer()` from `dimeropt.py` (shared helper).
6. **Output**: Writes ALL band images as separate frames to output traj. Each image gets `src_index`, `image_idx`, `subband_idx`, `image_type` (endpoint/intermediate_minimum/climbing/regular), `effective_fmax`, `image_converged`, `band_converged`, `band_converged_CI`. CI images additionally get `eigenmode`, `barrier`, `dE`. Imin images duplicated so each sub-band is self-contained. Generates band plot PNG. Per-sub-band runs use temp files with `_sub{subband_idx}` suffix.

### `catsunami/ocpneb.py` - Core NEB Engine
- **`OCPNEB`** (extends BaseNEB): Two modes controlled by `vasp` flag:
  - **FAIRChem mode** (`vasp=False`): Batch-evaluates intermediate images via FAIRChemCalculator. Caches forces between calls. fairchem imports are lazy.
  - **VASP mode** (`vasp=True`): Delegates to parent `BaseNEB` for per-image force evaluation. Each image has its own VASP calculator. No batching/caching/fairchem dependency. When `intermediate_minima=True` or `initial_imin_set` provided, bypasses `BaseNEB.get_forces()` and evaluates individually to route through custom `get_precon_forces()`.
  - Both modes: Handles constraints (fixed atoms by tag=0 or explicit). Stores full `real_forces` array `(nimages, natoms, 3)` including endpoints for uniform access via `real_forces[imax]`.
- **Intermediate minima support**: `get_precon_forces()` periodically (every `intermediate_minima_check_interval` force calls) scans images 2 through `nimages-3` for local energy minima (energy lower than both neighbors by ≥ `intermediate_minima_min_depth`). Between checks, classification is frozen. Detected minima receive pure PES forces (no spring/tangent projection) to relax into the basin. Band splits into segments at minima; each segment gets its own climbing image (highest-energy interior, recomputed every step). `imax` = global highest-energy CI. Images 1 and `nimages-2` excluded from detection to ensure room for CIs. `initial_imin_set` can seed known positions (for full-band `continue_from_result=True`); seeded set is frozen (no re-scanning), same PES force treatment.
- **Frozen images**: `frozen_images` constructor param provides initial set; `freeze_fmax`/`freeze_endpoint_fmax` enable auto-freeze via `_auto_freeze()`. Frozen images: (1) zero NEB forces, (2) skipped in FAIRChem batching after first eval (cached in `_frozen_energies`/`_frozen_pbc_forces`), (3) excluded from imin detection (with neighbors), (4) segment boundaries for CI selection. Only converged sub-bands, individual converged imin, and individual converged CIs can freeze — never regular images. `_auto_freeze()` runs after fmax computation; new freezes take effect next force call.
- **Per-image effective fmax**: After NEB force modifications in `get_precon_forces()`, computes `max|F|` per image (regular: NEB-modified; climbing: PES + doubled reversed tangential; imin: pure PES; frozen: 0.0). Stored in `self.image_fmax` and `image.info['effective_fmax']`.
- **`swDNEB`** (NEBMethod subclass): Switched Doubly Nudged Elastic Band (works with FAIRChem and VASP):
  - Improved tangent vectors (energy-weighted at extrema)
  - Perpendicular spring force to straighten band
  - Switching function `sw = (2/pi) * arctan(|F_perp|^2 / |F_S_perp|^2)` turns off DNEB as convergence is reached (preventing frustration on curved MEPs)
  - Refs: Henkelman & Jonsson, J. Chem. Phys. (2000); Trygubenko & Wales (2004)

### `dimeropt.py` - Dimer Method
- **`_setup_dimer()`**: Shared helper creating `MinModeAtoms` + `MinModeTranslate` from ASE's dimer API. Used by `dimeropt()` and `nebopt()` (CI refinement). Returns `(d_atoms, dim_rlx)` without running.
- Generates displacement candidates via `dimertools/structure_edit.py`
- Supports `bulk` and `oc` modes, both using `reaction_types` + `num_attempts_per_type` config.
- Convergence checks every 5 steps: participation ratio (delocalization) and desorption detection
- **Desorption flagging**: When desorption triggers `StopRun`, `reaction_type` overridden to `'desorption'` regardless of initialization type.
- Extension check if initial convergence fails
- Writes `reaction_type` to `atoms.info` for each attempt
- **Per-attempt error handling**: Each attempt has its own try/except; one failure doesn't abort remaining attempts.
- **Consecutive error tracking**: Structure-level `consecutive_errors` counter (from `init_function`). All attempts fail → increment; any success → reset to 0. At `max_consecutive_errors`, worker calls `sys.exit(1)` for executorlib restart.
- **Per-attempt execution**: Accepts `entries_to_run` (set of attempt_ids) and `continuation_data` (dict: attempt_id → Atoms). Calls `get_attempts()` on original input, then per attempt: skips if not in `entries_to_run`, uses `continuation_data[attempt_id]` if available (near-zero displacement), else fresh attempt. Eigenmode/reaction_type read from top-level `.info` first, `orig_info` fallback.

### `dimertools/structure_edit.py` - Reaction Types for Dimer
Reaction types configured via `reaction_types` (space-separated list). Bulk and OC dispatched via `_BULK_REACTION_TYPE_DISPATCH` / `_OC_REACTION_TYPE_DISPATCH`.

**Bulk reaction types** (`dataset_type = bulk`):

| Type | Function | Description | Atoms displaced |
|------|----------|-------------|-----------------|
| `vacancy` | `get_vacancy_attempts()` | Remove atom, neighbor hops into vacancy | 1 (center-based) |
| `hop_reuse` | `get_hop_reuse_attempts()` | Existing atom relocated to interstitial site | 1 (vector) |
| `hop_insert` | `get_hop_insert_attempts()` | New small atom (H/C/N/O/B) inserted at interstitial | 1 (vector) |
| `kickout_reuse` | `get_kickout_reuse_attempts()` | Existing atom to interstitial, kicks nearest into another | 2 (vector) |
| `kickout_insert` | `get_kickout_insert_attempts()` | New similar-sized atom at interstitial, kicks nearest lattice atom | 2 (vector) |
| `ring` | `get_ring_attempts()` | Ring of 2+ atoms rotate cooperatively; size from `ring_sizes` config. `ring_sizes = 2` for pairwise exchange. | N (vector) |
| `initial_guess` | `get_initial_guess_attempts()` | No displacement — starts from input as-is (supercell skipped). For pre-prepared TS guesses. Exclusive: ignores other types with warning. Always 1 attempt. Works with both `bulk` and `oc`. Uses eigenmode from `atoms.info['eigenmode']` or `atoms.info['orig_info']['eigenmode']` if present. | 0 (none) |

**OC reaction types** (`dataset_type = oc`):

Adsorbate atoms: tag=2 (fallback tag=1). Substrate (tag=0) fixed via `FixAtoms`.

| Type | Function | Description | Displacement mechanism |
|------|----------|-------------|----------------------|
| `adsorbate_atom` | `get_adsorbate_atom_attempts()` | Tight Gaussian on one adsorbate atom — only that atom moves | `displacement_center` + `gauss_std=0.2, number_of_atoms=1` |
| `adsorbate_atom_neighbors` | `get_adsorbate_atom_neighbors_attempts()` | Broad Gaussian on one adsorbate atom — nearby atoms also displaced | `displacement_center` (default DimerControl std) |
| `adsorbate` | `get_adsorbate_attempts()` | Random noise on all adsorbate atoms (isomerization) | Adsorbate-only mask |
| `diffusion` | `get_diffusion_attempts()` | Uniform translation of all adsorbate atoms in random 3D direction | `displacement_vector` (same direction, ~0.1 Å) |
| `rotation` | `get_rotation_attempts()` | Rigid-body rotation around adsorbate COM. Skips single-atom adsorbates. | `displacement_vector` (tangential, ~0.05 rad) |
| `adsorbate_surface` | `get_adsorbate_surface_attempts()` | Random noise on adsorbate + neighboring substrate | Adsorbate+neighbors mask (natural_cutoffs * 1.25) |
| `surface` | `get_surface_attempts()` | Broad Gaussian on one surface atom (tag=1) — surface reconstruction | `displacement_center` (default DimerControl std) |
| `custom` | `get_custom_attempts()` | Displacement fully controlled by `[DimerControl]` settings | Empty dict (pure DimerControl defaults) |
| `initial_guess` | `get_initial_guess_attempts()` | Same as bulk `initial_guess` | 0 (none) |

**OC helpers:** `_get_oc_adsorbate_indices(atoms)` (tag=2, fallback tag=1), `_get_oc_neighbor_mask(atoms, adsorbate_indices)` (natural_cutoffs * 1.25), `_sample_adsorbate_atoms(adsorbate_indices, num_needed)` (cycles if fewer available).

**Key infrastructure:**
- `find_interstitial_sites(atoms)`: Voronoi on 3x3x3 periodic images → filter by min distance → cluster within 0.5 Å. Uses `scipy.spatial.Voronoi` and `scipy.cluster.hierarchy`.
- `_mic_vector()` / `_nearest_site()`: Minimum image convention helpers.
- `_find_ring(neighbors_dict, seed, ring_size)`: Finds closed rings via constrained random walk. Size=2 → pairwise exchange; ≥3 → cooperative rotation.
- Element sampling: `hop_insert` weights by 1/covalent_radius (H favored). `kickout_insert` uses Gaussian weight centered on host avg covalent radius (σ=0.2 Å) from 30 metals/semiconductors.
- `turn_into_supercell(atoms, min_length=7.0)`: Preserves `.info` across `make_supercell()`. Enforces min 7 Å cell dimensions (fairchem uses 5 Å radius graph cutoff). Called in `get_attempts()` for all types except `initial_guess` (controlled by `supercell` config, default `True`).
- Dispatch: `get_attempts()` selects dispatch dict by `dataset_type`. `initial_guess` handled early (before bulk/oc branch).
- `_safe_normalize(vec)`: Returns random unit vector if norm ≈ 0.
- `_build_neighbor_dict(atoms)`: Skips self-interactions (`i == j`) in small periodic cells.
- Defensive guards: zero-weight fallback in `_sample_kickout_insert_element`, empty `ring_sizes` early return, missing adsorbate early return in OC mode.

### `geomopt.py` - Geometry Optimization
- `geomopt()`: Standard relaxation with optional cell relaxation (FrechetCellFilter)
- `doublegeomopt()`: Takes converged TS with eigenmode, displaces ±0.25*eigenmode, relaxes both directions, detects bond breaking/forming via `check_reaction()`. Reads `eigenmode`, `converged`, `src_index` from `atoms.info['orig_info']` (fallback `atoms.info`). Each frame gets `side` in `.info` (-1=min1, 0=ts, 1=min2). Writes 2 CSV lines per job. Accepts `entries_to_run`/`continuation_data` for per-side execution.

### `tools.py` - Utilities
- `load_and_sanitize(traj, i, j)`: Loads atoms and stashes original `.info` into `atoms.info = {"orig_info": <original_info>}`. Prevents per-atom array data from causing size mismatches on atom add/remove. Called in `__main__.py` for all methods.
- `clean_up_files(config_dict)`: Removes leftover temp files on resume. Method-aware: NEB (`neb_*.log/traj`, `reactant/product_relaxation_*`, `diffusion_barrier_*.png`, `VASP_*_*/`), Dimer (`dimer_*.log/traj`), Minimization (`optimization_*.log/traj`).
- **Previous result extraction**: `extract_previous_results(job_ids, config_dict, redo_info)` — unified extraction from output trajs for all methods. Returns `{job_id: continuation_data}` (Dimer: `{attempt_id: Atoms}`, NEB: `{subband_idx: [Atoms]}`, DoubleMinimization: `{side: Atoms}`, Minimization: `Atoms`). Helpers: `_build_output_traj_index()`, `_sanitize_with_continuation()`.
- `check_reaction()` / `check_adsorbate_reaction()`: Bond detection via ASE neighbor_list with natural cutoffs, compare connectivity between structures.

### `.info` Handling Rule (applies to all methods)

`orig_info` is set once at entry (`load_and_sanitize` for fresh, `_sanitize_with_continuation` for continuations) and **never modified**. Methods write output keys to top-level `.info`. When needing data from a previous run (eigenmode, reaction_type), check top level first, then `orig_info` fallback. On continuations, `orig_info` nests recursively.

### `init_function.py` - Worker Initialization
- Assigns GPU based on executorlib_worker_id and jobs_per_gpu
- Multi-GPU sharing (`jobs_per_gpu > 1`): auto-detects MPS daemons (`/tmp/mps_{gpu}/control`). If MPS running, sets `CUDA_MPS_PIPE_DIRECTORY` + `CUDA_VISIBLE_DEVICES=0`; else direct assignment. Skipped for VASP.
- FAIRChem: instantiates calculator once (on GPU, shared across structures)
- VASP/VaspInteractive: returns class (instantiation deferred to per-image in `nebopt.py`)
- Returns `{calc, Optimizer, consecutive_errors}`. `consecutive_errors` is `[0]` (mutable list); resets on worker restart.

### `catsunami/autoframe.py` - NEB Frame Generation
- `AutoFrameDissociation` / `AutoFrameTransfer`: Generates NEB frames from reaction databases
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
Optimizer = MDMin           # MDMin | BFGS | LBFGS | FIRE (NEB band optimizer)
fmax = 0.05                 # Force convergence (eV/Å)
steps = 6000                # Max optimization steps
jobs_per_gpu = 1            # Concurrent jobs per GPU
Calculator = FAIRChemCalculator  # FAIRChemCalculator | Vasp | VaspInteractive
run_jobs = remaining         # Job categories to process (see below)
continue_from_result = True # Resume from previous result (see below)
zip = True                  # Compress debug files
max_consecutive_errors = 5  # Kill worker after N consecutive all-error structures (0 = disabled)
restart_limit = 3           # executorlib: max worker restarts (0 = no restarts)

[FAIRChemCalculator]
device = cuda
name_or_path = uma-s-1p1
task_name = oc20

[Vasp]                             # INCAR params (for both Vasp and VaspInteractive)
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
dt = 0.02
maxstep = 0.1

[LBFGS]
memory = 10
damping = 0.99
alpha = 200
maxstep = 0.1

[ourNEB]
relax_endpoints = True
endpoint_relax_Optimizer = LBFGS
endpoint_relax_fmax = 0.02
endpoint_relax_steps = 1000
interpolate_method = ase_linear    # ocp_idpp | ase_idpp | ase_linear | False
num_frames = 10
batch_size = 8                     # FAIRChem only
DNEB = True
intermediate_minima = True
intermediate_minima_check_interval = 100
intermediate_minima_min_depth = 0.05
max_num_frames = 80                # Enables auto image addition if > num_frames (FAIRChem only)
add_images_check_interval = 100
dimer_refine_ci = False            # Post-NEB dimer on unconverged CIs (FAIRChem only)
dimer_refine_steps = 300
refine_band_steps = 0              # Continuation NEB after extrema refinement; also relaxes imin when > 0
# VASP-only (required when Calculator = Vasp or VaspInteractive):
vasp_command_endpoints = "srun --exclusive -n 64 vasp_std"
vasp_ncore_endpoints = 8
vasp_command_intermediates = "srun --exclusive -n 16 vasp_std"
vasp_ncore_intermediates = 4

[BaseNEB]
k = 5
method = improvedtangent
climb = True
allow_shared_calculator = True

[ourDimer]
dataset_type = oc            # oc | bulk
reaction_types = vacancy     # Bulk: vacancy hop_reuse hop_insert kickout_reuse kickout_insert ring initial_guess
                             # OC: adsorbate_atom adsorbate_atom_neighbors adsorbate diffusion rotation adsorbate_surface surface custom initial_guess
num_attempts_per_type = 1
ring_sizes = 3 4             # For 'ring' type (bulk only)
supercell = True             # Min 7 Å expansion
delocalization_threshold = 0.8
extension_check_fmax = 0.4
extension_check_curvature = -0.2

[ourMinimization]
relax_cell = False

[ourDoubleMinimization]
relax_cell = False
```

### `run_jobs` — Flexible Job Selection

`run_jobs` specifies job categories to process. Fresh vs resume determined by whether `traj_files_ordered.json` exists (no explicit toggle; delete output dirs + json to force fresh start).

**4 categories** (each CSV status line independently categorized):

| Category | CSV statuses |
|---|---|
| `converged` | `converged`, `converged_after_extension`, `converged_CI` |
| `not_converged` | `not_converged`, `not_converged_after_extension`, `not_converged_StopRun` |
| `errored` | `error`, `error: <message>` |
| `remaining` | No CSV row for this job_id |

**Per-line selection**: A job is included if ANY of its CSV lines match requested categories. Only matching attempts/sub-bands/sides are redone; the rest stay in active output.

**Examples:**
```ini
run_jobs = remaining                  # Default. Fresh: all jobs. Resume: unfinished only.
run_jobs = remaining errored          # Resume + retry errors
run_jobs = converged                  # Redo converged (e.g., refine with VASP)
run_jobs = not_converged              # Redo not-converged
run_jobs = errored                    # Retry errors only
run_jobs = all                        # Redo everything
```

**Archiving on resume**: Full backup, per-entry cleaning:
- **CSVs**: Archived as `{method}_status_csvs/previous_{N}.zip`. Only matching lines removed.
- **Output trajs**: Archived as `{method}_trajes/previous_{N}.zip`. Frames removed by `attempt_id` (Dimer), `subband_idx` (NEB), `src_index` (Minimization). DoubleMinimization removes all 3 frames (min1+TS+min2) together since they share reaction check metadata.
- **Debug zips**: Archived as `{method}_debug_zips/previous_{N}.zip`. Per-entry cleaning by subunit info in filenames (Dimer: `attempt_id`, DoubleMinimization: `-0`/`-1` suffix, NEB sub-band: `_sub{idx}` suffix). NEB full-band debug files removed whole when any sub-band is redone.

Jobs with no prior entries don't trigger archiving.

### `continue_from_result` — Continue from Previous Result

When re-running previously completed entries, the system extracts previous results from output trajs. `continue_from_result` controls usage:

- `True` (default): continue from extracted structure.
- `False`: start fresh (Dimer: fresh attempts at same positions; NEB partial: re-interpolate; NEB all: original input; DoubleMinimization: re-displace from TS).

Errored entries always fall back to fresh. Continuation handled per-entry inside method functions; fresh and continuation jobs coexist in same batch.

| Method | Granularity | `True` | `False` |
|---|---|---|---|
| Dimer | per-attempt | Continue from extracted structure with eigenmode | Fresh attempt at same `attempt_id` (same type, new displacement) |
| NEB | per-sub-band | Continue from extracted images. Full-band: seed imin from `image_type`. | Partial: re-interpolate. All sub-bands: original input. |
| DoubleMinimization | per-side | Continue not-converged side | Re-displace from TS |
| Minimization | per-job | Continue from extracted structure | Original input |

All methods extract via `extract_previous_results`. **`initial_guess` vs `continue_from_result`**: separate features — `initial_guess` is for **external** TS guesses (from another code/method); `continue_from_result` extracts from a **previous tsearch run**.

**Examples:**
```ini
run_jobs = not_converged
steps = 10000                     # More steps for not-converged NEB

run_jobs = converged
Calculator = VaspInteractive      # Refine with VASP

run_jobs = not_converged
continue_from_result = False      # Re-run from scratch
```

## Output Structure

For method `NEB`:
```
NEB_status_csvs/status_rank_*.csv    # job_id,rank,sub_band_id,status
NEB_trajes/collected_ts_rank_*.traj  # All band images with metadata in atoms.info
NEB_debug_zips/                      # Compressed log/traj/plot files
```

NEB output image metadata: `src_index`, `image_idx`, `subband_idx`, `image_type` (endpoint/intermediate_minimum/climbing/regular), `effective_fmax`, `image_converged`, `band_converged`, `band_converged_CI`, `nimages`, `interpolation_method`, `orig_info`. CI images also get `eigenmode`, `barrier`, `dE`. Imin images duplicated for sub-band self-containment. Per-sub-band redo temp files use `_sub{subband_idx}` suffix.

Dimer output: `eigenmode`, `curvature`, `converged`, `src_index`, `attempt_id`, `stoprun`, `selected_index`, `reaction_type`, `orig_info`.

DoubleMinimization output: `side` (-1/0/1), `parent_ts_index`, `converged`, `src_index`, `is_reaction`, `n_formed_bonds`, `n_broken_bonds`, `orig_info`. CSV: 2 lines per job `{job_id},{rank},{side_id},{parent_ts_idx},{status}`.

## Execution Modes

**Distributed (executorlib=True)**: FluxJobExecutor, one worker per GPU (or jobs_per_gpu sharing). Each worker calls `init_function` once, then processes jobs sequentially.

**Serial (executorlib=False)**: Single GPU, one job at a time. For debugging.

## Worker Health and Automatic Restart

CUDA device-side asserts (e.g., fairchem's `radius_graph_pbc_v2` on degenerate structures) permanently poison the GPU context.

1. **Tracking**: `init_function` creates `consecutive_errors = [0]`. Methods increment on all-error structures, reset on any success.
2. **Self-kill**: At `max_consecutive_errors`, method calls `sys.exit(1)`.
3. **Restart**: `restart_limit` tells executorlib to restart crashed workers (new Flux job, potentially different node/GPU, fresh CUDA context, `init_function` re-runs).
4. **Exhaustion**: After `restart_limit` restarts, worker dies permanently; remaining tasks get `ExecutorlibSocketError`.

**Note**: On restart with `jobs_per_gpu > 1`, GPU assignment may be incorrect if new node has different GPU count. Not an issue for `jobs_per_gpu = 1` (executorlib handles allocation).

## Testing

```bash
pip install pytest pytest-timeout

# CPU-only unit tests (~40s, no GPU)
pytest -m "not gpu and not flux" -v

# All tests (GPU node, ~6 min)
pytest -v --timeout=600

# GPU integration tests only
pytest -m gpu -v
```

### Test Structure

```
tests/
├── conftest.py                  # Fixtures, skip logic, make_config_dict()
├── fixtures/                    # Small .traj files
│   ├── bulk_crystal.traj        # 3-atom FCC crystal
│   ├── minimization_input.traj  # 68-atom slab+adsorbate
│   ├── oc_adsorbate_slab.traj   # 131-atom slab+adsorbate
│   └── oc_neb_pair.traj         # 10-frame NEB, 68 atoms
├── test_config.py               # ConfigManager, load_*, run_jobs, archive/clean (~74, CPU)
├── test_tools.py                # load_and_sanitize, check_reaction, extraction (~28, CPU)
├── test_structure_edit.py       # Bulk & OC reaction types, supercell (~35, CPU)
├── test_ocpneb.py               # swDNEB, forces, frozen images (~20, mixed)
├── test_nebopt_integration.py   # Full NEB runs (8, GPU)
├── test_dimeropt_integration.py # Dimer runs (7, GPU)
├── test_geomopt_integration.py  # geomopt + doublegeomopt (5, GPU)
├── test_init_function.py        # init_function (5, GPU)
└── test_main_integration.py     # End-to-end pipeline + resume (4, GPU)
```

Markers: `@pytest.mark.gpu` (CUDA, auto-skipped), `@pytest.mark.flux` (Flux scheduler), `@pytest.mark.slow` (>60s)

### Writing New Tests

- `make_config_dict(method="NEB", **overrides)` from `conftest.py`
- Session-scoped `fairchem_calc` fixture for GPU tests
- `converged_ts_atoms` fixture for doublegeomopt tests
- Integration tests use `tmp_path` + `monkeypatch.chdir()`
- CPU structure_edit tests seed with `random.seed(42)` + `np.random.seed(42)`

## HPC Setup (Perlmutter)

```bash
#SBATCH --constraint=gpu
#SBATCH --gpus-per-node=4
#SBATCH --nodes=N
srun -N $SLURM_NNODES -n $SLURM_NNODES --gpus-per-node=4 flux start python -u -m tsearch
```

Set `FAIRCHEM_CACHE_DIR` for model caching. Requires CUDA libraries in `LD_LIBRARY_PATH`.
