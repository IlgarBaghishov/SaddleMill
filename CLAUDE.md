# SaddleMill - High-Throughput Transition State Search Library

## Overview

SaddleMill is a Python library for creating datasets of Transition States (TS) using neural network potentials (FAIRChemCalculator / Meta's UMA model) or DFT (VASP / VaspInteractive). It supports distributed GPU execution on HPC systems (4 A100 per node, GH200, 3 A100 per node) via executorlib + Flux.

## Dependencies

Minimum required versions (baseline, enforced in `pyproject.toml`):
- `ase >= 3.26.0`
- `fairchem-core >= 2.19.0`

Both are required by the package — `fairchem-core` registers the LMDB backend that `ase.db.connect` uses for `.aselmdb` files, and `ase >= 3.26.0` is the first ASE release whose `db` layer wires up that backend correctly. Any code path that does ASE LMDB I/O must `import fairchem.core.datasets` before calling `ase.db.connect`.

## Entry Point

```bash
srun -N $SLURM_NNODES -n $SLURM_NNODES --gpus-per-node=4 flux start python -u -m saddlemill
```

The `__main__.py` reads `config.ini` from the current directory, loads the method, scans for `.traj` files, distributes jobs across GPUs, and collects results.

## Supported Methods

| Method | Config value | Module | Description |
|--------|-------------|--------|-------------|
| NEB | `NEB` | `nebopt.py` | Nudged Elastic Band (with optional DNEB switching) |
| Dimer | `Dimer` | `dimeropt.py` | Dimer method for saddle point search |
| Minimization | `Minimization` | `geomopt.py` | Single structure geometry optimization |
| DoubleMinimization | `DoubleMinimization` | `geomopt.py` | TS refinement: displaces along eigenmode in both directions, relaxes, checks for reaction |
| SinglePoint | `SinglePoint` | `geomopt.py` | One E/F call per frame; only method that accepts `.aselmdb` input. `frames_per_job = N` enables a batched FAIRChem forward pass for N frames per job. With VASP/VaspInteractive, `frames_per_job` is forced to 1 (no batched DFT). |

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
catsunami/ocpneb.py  (OCPNEB: batched NEB with optional swDNEB)
```

Auxiliary entry points:
- `python -m saddlemill.status [dir]` — summarize completion state of a results directory (see `status.py`)
- `python -m saddlemill.analyze_neb [dir]` — generate plots + text reports for not-converged NEB jobs (see `analyze_neb.py`)

## Key Modules

### `config.py`
- `ConfigManager`: Reads `config.ini` with type inference (bool/int/float/list/string). Quoted strings (`"..."` or `'...'`) preserved as literal strings (for VASP commands with spaces). Warns on unrecognized keys in known sections.
- **Backward-compat key migration** (`_RENAMED_KEYS` + `_migrate_renamed_keys`): silently renames `intermediate_minima_check_interval` → `intermediate_minima_check_step`, `add_images_check_interval` → `add_images_step`. The removed bool `intermediate_minima = True` is auto-converted to `intermediate_minima_check_step = 100`.
- `load_calculator()`: Returns calculator class/callable (`FAIRChemCalculator`, `Vasp`, or `VaspInteractive`). For FAIRChem, returns `from_model_checkpoint`; for VASP, returns the class itself. Instantiation deferred to `init_function.py` (FAIRChem) or to the method modules (VASP, per-job-unit). All five methods (NEB, Dimer, Minimization, DoubleMinimization, SinglePoint) accept VASP/VaspInteractive. NEB uses `[ourNEB] vasp_command_endpoints/intermediates`; other methods use a single `vasp_command` in their own `[ourXxx]` section. `vasp_command` is required when Calculator is VASP — `load_method` raises if it is missing. SinglePoint with VASP requires `frames_per_job = 1`.
- `load_method()`: Imports the correct optimization function
- `load_optimizer()`: Returns optimizer class(es) - for NEB returns (endpoint_optimizer, neb_optimizer)
- `get_trajes_and_indices()`: Recursively scans dir_path (including subdirectories) for .traj files, splits into job batches
- Resume support: `get_remaining_trajes()` skips completed jobs; `build_redo_info()`, `archive_and_clean_csvs()`, `archive_and_clean_outputs()` implement the `run_jobs` archive-and-redo flow
- `create_results_directories()`, `read_status_csv_rows()`, `get_flux_resources()` round out the helper surface used by `__main__.py`

### `nebopt.py` - NEB Workflow
1. **Continuation check**: When `entries_to_run` is set, overrides starting images from `continuation_data`. All NEB settings always come from config.ini. `continue_from_result=True` uses extracted images directly and seeds imin/frozen/frozen_fmax from previous result; config controls whether new imin/images are detected on top of seeded state. `False` falls through to fresh run. Always full-band (no sub-band reruns).
2. **Endpoint relaxation** (optional): Relaxes reactant/product with `endpoint_relax_Optimizer` (defaults to `Main.Optimizer` if `None`), to `endpoint_relax_fmax` within `endpoint_relax_steps`. For VaspInteractive, calculators finalized after relaxation and endpoints frozen with `SinglePointCalculator`.
3. **Interpolation**: `ocp_idpp` (Meta's PBC-aware), `ase_idpp`, `ase_linear` (auto-falls back to IDPP on atom overlap), or `False` (use provided frames)
4. **NEB optimization**: Uses `OCPNEB` with the configured band optimizer (default MDMin). Supports climbing image and intermediate minima (per-segment climbing). For VASP/VaspInteractive: per-image calculators in `VASP_{job_id}_{image_idx}/` dirs, separate `command`/`ncore` for endpoints vs intermediates, WAVECAR/CHG/CHGCAR cleanup after completion. VaspInteractive endpoint calculators finalized before `SinglePointCalculator` replacement; intermediate calculators finalized after NEB completes. The optimization is split into phases driven by **one-shot events**: `intermediate_minima_check_step` triggers imin detection at a specific step (one-shot, not periodic; 0 = disabled); `add_images_step` triggers image addition at a specific step (one-shot). When an event fires, NEB stops, the event runs, and a fresh optimizer instance resumes the same NEB object until the next event or the global step budget runs out.
   - **`_detect_imin()`**: Standalone function in nebopt.py. One-shot detection with exclusion based on existing imin +/-1 positions. Merges with existing imin_set.
   - **`_relax_imin()`**: After detection, each newly detected imin is relaxed individually with `endpoint_relax_Optimizer` (or `Main.Optimizer`) to `endpoint_relax_fmax` within `endpoint_relax_steps`. Relaxed imin below `endpoint_relax_fmax` are added to `neb._frozen_set` so they act as segment boundaries from then on. Logs/trajs `imin_relax_{file_tag}_img{idx}.{log,traj}` are tracked in temp files for cleanup/zip.
   - **Auto image addition** (FAIRChem only): when `max_num_frames > num_frames`, `_expand_band()` doubles images in unconverged segments at `add_images_step` via IDPP interpolation; `_remap_indices()` keeps imin/frozen/climbing indices consistent across the expansion.
   - **Auto-freeze** (FAIRChem only, always on): OCPNEB checks for converged sub-bands/imin/CIs each step, adds to `_frozen_set`. Frozen images are skipped in FAIRChem batching after first eval (cached), report cached NEB-modified fmax (not 0), and act as segment boundaries. Only sub-bands (all below threshold), individual imin (below `endpoint_fmax`), and individual CIs (below `fmax`) can freeze — never regular images.
5. **Post-NEB refinement** (optional, FAIRChem only): If band not fully converged: (a) `dimer_refine_ci=True`: quick ASE dimer on each unconverged CI, seeded with NEB tangent as eigenmode, using `[DimerControl]` config. Updates CI position in the band, then calls `neb.get_forces()` to obtain proper NEB fmax, and freezes with cached NEB fmax. (b) `refine_band_steps > 0`: always runs if band not converged. Reuses the SAME NEB object with a new optimizer — all imin/CI/frozen state preserved (no separate OCPNEB instance). Uses `_setup_dimer()` from `dimeropt.py` (shared helper).
6. **Output**: Writes ALL band images as separate frames to output traj. Each image gets `src_index`, `image_idx`, `subband_idx`, `image_type` (endpoint/intermediate_minimum/climbing/regular), `effective_fmax`, `image_converged`, `band_converged`, `band_converged_CI`, `nimages`, `interpolation_method`, `task_name`, and the per-step band-wide sets `imin_set`/`climbing_set`/`frozen_set` are stamped on every image (set by OCPNEB). CI images additionally get `eigenmode`, `barrier`, `dE`. Imin images duplicated so each sub-band is self-contained. Generates band plot PNG. Result extraction uses the same NEB object's state directly (no state loss).

### `catsunami/ocpneb.py` - Core NEB Engine
- **`OCPNEB`** (extends BaseNEB): Two modes controlled by `vasp` flag:
  - **FAIRChem mode** (`vasp=False`): Batch-evaluates intermediate images via FAIRChemCalculator. Caches forces between calls. fairchem imports are lazy.
  - **VASP mode** (`vasp=True`): Delegates to parent `BaseNEB` for per-image force evaluation. Each image has its own VASP calculator. No batching/caching/fairchem dependency. When `initial_imin_set` or `frozen_images` provided, bypasses `BaseNEB.get_forces()` and evaluates individually to route through custom `get_precon_forces()`.
  - Both modes: Handles constraints (fixed atoms by tag=0 or explicit). Stores full `real_forces` array `(nimages, natoms, 3)` including endpoints for uniform access via `real_forces[imax]`.
  - `dneb=True` swaps the band's NEBMethod for `swDNEB` (default `False` → standard `improvedtangent` from BaseNEB).
- **Intermediate minima support**: OCPNEB no longer detects intermediate minima internally — imin_set is externally managed by nebopt.py (via `_detect_imin()`). `initial_imin_set` seeds known positions. Detected minima receive pure PES forces (no spring/tangent projection) to relax into the basin. Band splits into segments at minima; each segment gets its own climbing image (highest-energy interior, recomputed every step via `_find_segment_ci()`). `imax` = global highest-energy CI.
- **Frozen images**: `frozen_images` constructor param provides initial set; `frozen_fmax` constructor param (dict: img_idx -> cached NEB fmax) provides initial fmax cache; `freeze_fmax`/`freeze_endpoint_fmax` enable auto-freeze via `_auto_freeze()`. Frozen images: (1) report cached NEB-modified fmax (not 0), (2) skipped in FAIRChem batching after first eval (cached in `_frozen_energies`/`_frozen_pbc_forces`), (3) segment boundaries for CI selection. Imin detection exclusion zone is based on existing imin +/-1 (NOT frozen set) — new imin can be next to frozen CIs but not other imin. Only converged sub-bands, individual converged imin, and individual converged CIs can freeze — never regular images. `_auto_freeze()` runs after fmax computation; caches NEB fmax for newly frozen images in `_frozen_fmax_cache`; new freezes take effect next force call. Frozen CIs are "sticky" — they remain the climbing image for their segment even if a later image becomes higher-energy.
- **Per-image effective fmax**: After NEB force modifications in `get_precon_forces()`, computes `max|F|` per image (regular: NEB-modified; climbing: PES + doubled reversed tangential; imin: pure PES; frozen: cached NEB fmax from freeze time). Stored in `self.image_fmax` and `image.info['effective_fmax']`. Each force call also stamps `nimages` on every image and the band-wide sets `imin_set`/`climbing_set`/`frozen_set` on `image[0]`.
- **`swDNEB`** (NEBMethod subclass): Switched Doubly Nudged Elastic Band (works with FAIRChem and VASP):
  - Improved tangent vectors (energy-weighted at extrema)
  - Perpendicular spring force to straighten band
  - Switching function `sw = (2/pi) * arctan(|F_perp|^2 / |F_S_perp|^2)` turns off DNEB as convergence is reached (preventing frustration on curved MEPs)
  - Refs: Henkelman & Jonsson, J. Chem. Phys. (2000); Trygubenko & Wales (2004)

### `dimeropt.py` - Dimer Method
- **`_setup_dimer()`**: Shared helper creating `MinModeAtoms` + `MinModeTranslate` from ASE's dimer API. Used by `dimeropt()` and `nebopt()` (CI refinement). Returns `(d_atoms, dim_rlx)` without running.
- **`_refine_eigenmode()`**: Rotation-only dimer helper. Creates `MinModeAtoms` on a copy of atoms, triggers eigenmode rotation via `get_forces()`, returns `(refined_eigenmode, curvature)` without translating. Used by `doublegeomopt()` for optional pre-displacement eigenmode refinement.
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
| `vacancy` | `get_vacancy_attempts()` | Remove atom, then one of three sub-mechanisms (NN hop into vacancy, NNN hop, concerted 2-atom chain) chosen at random per attempt | 1–N (center-based) |
| `hop_reuse` | `get_hop_reuse_attempts()` | Existing atom relocated to interstitial site | 1 (vector) |
| `hop_insert` | `get_hop_insert_attempts()` | New small atom (H/C/N/O/B) inserted at interstitial | 1 (vector) |
| `kickout_reuse` | `get_kickout_reuse_attempts()` | Existing atom to interstitial, kicks nearest into another | 2 (vector) |
| `kickout_insert` | `get_kickout_insert_attempts()` | New similar-sized atom at interstitial, kicks nearest lattice atom | 2 (vector) |
| `ring` | `get_ring_attempts()` | Ring of 2+ atoms rotate cooperatively; size from `ring_sizes` config. `ring_sizes = 2` for pairwise exchange. | N (vector) |
| `initial_guess` | `get_initial_guess_attempts()` | No displacement — starts from input as-is (supercell skipped). For pre-prepared TS guesses. Exclusive: ignores other types with warning. Always 1 attempt. Works with both `bulk` and `oc`. Uses eigenmode from `atoms.info['eigenmode']` or `atoms.info['orig_info']['eigenmode']` if present. | 0 (none) |

All bulk types (except `initial_guess`) route directed displacement candidates through `_maybe_gaussian()`, which with 10% probability replaces the directed vector with broad isotropic Gaussian noise on the same atom set — keeps a stochastic exploration tail on top of the geometric heuristics.

**OC reaction types** (`dataset_type = oc`):

Adsorbate atoms: tag=2 only (no fallback). Substrate (tag=0) fixed via `FixAtoms`. If no tag=2 atoms exist, OC reaction-type generators emit a warning and return no attempts.

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

**OC helpers:** `_get_oc_adsorbate_indices(atoms)` (tag=2 only), `_get_oc_neighbor_mask(atoms, adsorbate_indices)` (natural_cutoffs * 1.25), `_sample_adsorbate_atoms(adsorbate_indices, num_needed)` (cycles if fewer available).

**Key infrastructure:**
- `find_interstitial_sites(atoms, min_dist_frac=0.4)`: Voronoi on 3x3x3 periodic images → filter by min distance (`min_dist_frac * avg_nn_dist`) → cluster within 0.5 Å. Uses `scipy.spatial.Voronoi` and `scipy.cluster.hierarchy`.
- `_mic_vector()` / `_nearest_site()`: Minimum image convention helpers.
- `_find_ring(neighbors_dict, seed, ring_size)`: Finds closed rings via constrained random walk. Size=2 → pairwise exchange (with perpendicular dodge vectors so the two atoms don't collide along the bond axis); ≥3 → cooperative rotation.
- Element sampling: `hop_insert` weights by 1/covalent_radius via `_sample_hop_insert_element` (H favored). `kickout_insert` uses Gaussian weight centered on host avg covalent radius (σ=0.2 Å) from 30 metals/semiconductors via `_sample_kickout_insert_element`.
- `_get_atom_selection_weights(atoms)`: inverse-covalent-radius weights for picking which existing atom to displace (small atoms favored).
- `_shuffled_site_indices(num_sites)`: cycles through interstitial sites without repeats, generator pattern.
- `_maybe_gaussian(displacement, atoms, gauss_prob=0.1, gauss_std=0.4)`: bulk-types pass directed displacements through this — with `gauss_prob` it returns broad Gaussian noise instead, adding an exploration tail.
- `turn_into_supercell(atoms, min_length=7.0)`: Preserves `.info` across `make_supercell()`. Enforces min 7 Å cell dimensions (fairchem uses 5 Å radius graph cutoff). Called in `get_attempts()` for all types except `initial_guess` (controlled by `supercell` config, default `True`).
- Dispatch: `get_attempts()` selects dispatch dict by `dataset_type`. `initial_guess` handled early (before bulk/oc branch) and is also registered in both `_BULK_REACTION_TYPE_DISPATCH` and `_OC_REACTION_TYPE_DISPATCH`.
- `_safe_normalize(vec)`: Returns random unit vector if norm ≈ 0.
- `_build_neighbor_dict(atoms)`: Skips self-interactions (`i == j`) in small periodic cells.
- Defensive guards: zero-weight fallback in `_sample_kickout_insert_element`, empty `ring_sizes` early return, missing adsorbate early return in OC mode.

### `geomopt.py` - Geometry Optimization
- `geomopt()`: Standard relaxation with optional cell relaxation (FrechetCellFilter). Output frames carry `task_name` (from `get_task_name`).
- `doublegeomopt()`: Takes converged TS with eigenmode, displaces ±0.25*eigenmode, relaxes both directions, detects bond breaking/forming via `check_reaction()` and `check_adsorbate_reaction()`. Reads `eigenmode`, `converged`, `src_index` from `atoms.info['orig_info']` (fallback `atoms.info`). Each frame gets `side` in `.info` (-1=min1, 0=ts, 1=min2). Writes 2 CSV lines per job in the form `{job_id},{rank},{side_id},{parent_source_idx},"{status_msg}"`. Accepts `entries_to_run`/`continuation_data` for per-side execution. Optional `pre_dimer_refine=True` (default False) runs a rotation-only dimer step via `_refine_eigenmode()` from `dimeropt.py` to refine the eigenmode direction before displacement. Rotation parameters controlled by `[DimerControl]` section (especially `max_num_rot`, `dimer_separation`). Stores refined eigenmode and `curvature` on the TS output frame's `.info`. The full reaction-detection metadata dict (`is_reaction`, `n_formed_bonds`, `n_broken_bonds`, `broken_bonds`, `formed_bonds`, plus `is_ads_reaction` / `n_ads_formed_bonds` / `n_ads_broken_bonds` / `ads_broken_bonds` / `ads_formed_bonds` for OC inputs) is copied onto every emitted frame (min1, TS, min2) along with `parent_ts_index` and `task_name`.
- **Desorption-skip logic**: when `check_adsorbate_reaction()` flags the bond change as a desorption, the higher-energy side is skipped to avoid relaxing into vacuum; that side gets `side_statuses[side] = "converged_desorption_skipped"` while the other side still runs normally. The TS frame's `status` always reflects the converged TS itself.
- `singlepoint()`: One E/F call per frame. Only method that supports `.aselmdb` input (via `[Main] input_format = lmdb`). With `[ourSinglePoint] frames_per_job = N` (any positive integer) it bundles N frames per executorlib job and computes their energies and forces in a single batched FAIRChem forward pass via `fairchem.core.datasets.atomic_data.atomicdata_list_to_batch` + `calc.predictor.predict`, then writes the frames contiguously in input order to the same rank shard. Per-frame `natoms` may vary within a batch — forces are sliced by cumulative `natoms` offsets. The last batch in each shard may be smaller than N (no divisibility check). For triplet-respecting use cases (e.g. DM output min1/TS/min2), pick N as a multiple of 3 so triplets stay intact. No optimizer, no temp log/traj files, no debug zips. Output goes to `SinglePoint_trajes/collected_sp_rank_{rank}.traj` (traj input) or `SinglePoint_lmdbs/collected_sp_rank_{rank}.aselmdb` (lmdb input). LMDB output preserves the source row's `key_value_pairs` and `data` (`info` + `traj_path`) verbatim and populates `row.energy` / `row.forces` via `SinglePointCalculator`. Does **not** call `atoms.wrap()` — the structure is untouched, only E/F are added. **VASP/VaspInteractive**: supported with `frames_per_job = 1` only (no batched DFT). Each job uses one `VASP_{i}/` scratch dir which is `shutil.rmtree`'d at job end — SP has no `_debug_zips/`. LMDB input restricted to `run_jobs = remaining`.

### `tools.py` - Utilities
- `load_and_sanitize(traj, i, j)`: Loads atoms and stashes original `.info` into `atoms.info = {"orig_info": <original_info>}`. Prevents per-atom array data from causing size mismatches on atom add/remove. Called in `__main__.py` for all methods.
- `passes_input_filter(images, config_dict)`: Applies the `input_statuses` fnmatch filter to a sanitized input — checks `orig_info['status']`, returns False if a filter is set and the frame's status doesn't match (or `status` is missing). Returns True for `input_statuses = all`.
- `save_ordered_traj_names()` / `read_ordered_traj_names()`: Persist/read the `traj_files_ordered.json` that anchors the resume contract.
- `clean_up_files(config_dict)`: Removes leftover temp files on resume. Method-aware: NEB (`neb_*.log/traj`, `reactant/product_relaxation_*`, `imin_relax_*`, `diffusion_barrier_*.png`, `VASP_*_*/`), Dimer (`dimer_*.log/traj`), Minimization (`optimization_*.log/traj`).
- `backup_flux_logs(worker_id)`: Snapshots flux log files for a worker before it self-kills, so the post-restart logs don't overwrite the diagnostic trail.
- `get_task_name(config_dict)`: Returns `[FAIRChemCalculator] task_name` if FAIRChem is the calculator, else `None`. Used by all methods to stamp `task_name` on output frames so downstream consumers know which UMA task generated them.
- **Previous result extraction**: `extract_previous_results(job_ids, config_dict, redo_info)` — unified extraction from output trajs for all methods. Returns `{job_id: continuation_data}` (Dimer: `{attempt_id: Atoms}`, NEB: `{subband_idx: [Atoms]}`, DoubleMinimization: `{side: Atoms}`, Minimization: `Atoms`). Helpers: `_build_output_traj_index()`, `_sanitize_with_continuation()`.
- `get_bond_set()`, `check_reaction()`, `check_adsorbate_reaction()`: Bond detection via ASE neighbor_list with natural cutoffs, compare connectivity between structures.

### `.info` Handling Rule (applies to all methods)

`orig_info` is set once at entry (`load_and_sanitize` for fresh, `_sanitize_with_continuation` for continuations) and **never modified**. Methods write output keys to top-level `.info`. When needing data from a previous run (eigenmode, reaction_type), check top level first, then `orig_info` fallback. On continuations, `orig_info` nests recursively.

### `init_function.py` - Worker Initialization
- Assigns GPU based on executorlib_worker_id and jobs_per_gpu
- Multi-GPU sharing (`jobs_per_gpu > 1`): auto-detects MPS daemons (`/tmp/mps_{gpu}/control`). If MPS running, sets `CUDA_MPS_PIPE_DIRECTORY` + `CUDA_VISIBLE_DEVICES=0`; else direct assignment. Skipped for VASP.
- FAIRChem: instantiates calculator once (on GPU, shared across structures)
- VASP/VaspInteractive: returns class — instantiation is deferred to the per-job-unit `resolve_vasp_calc()` helper in `tools.py`, called by `nebopt.py` (per-image), `dimeropt.py` (per-attempt), and `geomopt.py` (per-job / per-side / per-frame).
- Returns `{calc, Optimizer, consecutive_errors}`. `consecutive_errors` is `[0]` (mutable list); resets on worker restart.

### `catsunami/autoframe.py` - NEB Frame Generation
- `AutoFrameDissociation` / `AutoFrameTransfer`: Generates NEB frames from reaction databases
- Anomaly detection (intercalation, desorption, surface changes)
- Adsorbate reordering for symmetric species

### `catsunami/reaction.py` - Reaction Definitions
- `Reaction` class: Represents dissociation/desorption/transfer reactions with atom mappings and edge lists

### `status.py` - Job Status Reporter
Run with `python -m saddlemill.status [dir]` (default: current dir). Reads `config.ini`, `traj_files_ordered.json`, and `{method}_status_csvs/status_rank_*.csv` to print:
- Total/started/remaining job counts (honors `input_statuses` filter via the same `passes_input_filter` logic as `__main__.py`)
- Expected-vs-completed entries per job (Dimer: `len(reaction_types) * num_attempts_per_type`; DoubleMinimization: 2; Minimization: 1; NEB: unknown — band-level only)
- Per-status percentage breakdown
- Method-specific summaries: Dimer per-reaction-type table (`_print_dimer_reaction_type_table`); NEB per-job convergence + sub-band count distribution (`_print_neb_job_summary`)

### `analyze_neb.py` - NEB Diagnostics
Run with `python -m saddlemill.analyze_neb [dir]` (default: current dir). Reads `NEB_status_csvs/` + `NEB_debug_zips/` and emits a per-job analysis bundle into `neb_analysis/`:
- `job_{job_id}_overview.png` — band fmax vs step, per-image fmax heatmap, energy profile, image-type counts
- `job_{job_id}_per_image_fmax.png` — grid of per-image fmax evolution
- `job_{job_id}_refinement.png` — dimer-CI / imin-relax / refine-NEB refinement results
- `job_{job_id}_energy_evolution.png` — band energy profiles at regular checkpoints
- `job_{job_id}_fmax_timeline.png` — stacked image-type timeline + per-image fmax lines
- `analysis_report.txt` — combined plain-text report with event timeline, per-image convergence, sub-band analysis, and a "why didn't it converge" summary

Module-level config: `ANALYZE_STATUSES = {"not_converged", "converged"}` (set to `None`/empty to include every status), `FMAX_THRESHOLD = 0.05`, `ENDPOINT_FMAX = 0.02`, `FROZEN_THRESHOLD = 1e-6`. Internal helpers: `parse_optimizer_log`, `parse_dimer_log`, `parse_neb_trajectory` (reconstructs per-step `imin_set`/`climbing_set`/`frozen_set` events), `extract_job_files`, `compute_path_distances`, `load_status_csv`, plus the `plot_*` and `generate_text_report` functions.

### `nebtools/`
Standalone scripts for preparing NEB inputs. Currently `create_endpoints_for_MP_batteries.py` (Materials Project battery dataset endpoint generation by ion substitution).

## Configuration Reference (`config.ini`)

```ini
[Main]
executorlib = True          # Use FluxJobExecutor (True) or serial mode (False)
method = NEB                # NEB | Dimer | Minimization | DoubleMinimization | SinglePoint
dir_path = /path/to/trajs   # Directory containing .traj (or .aselmdb if input_format=lmdb) input files
input_format = traj         # traj (default) | lmdb. lmdb only supported for method=SinglePoint.
Optimizer = MDMin           # MDMin | BFGS | LBFGS | FIRE (NEB band optimizer)
fmax = 0.05                 # Force convergence (eV/Å)
steps = 6000                # Max optimization steps
jobs_per_gpu = 1            # Concurrent jobs per GPU
Calculator = FAIRChemCalculator  # FAIRChemCalculator | Vasp | VaspInteractive
run_jobs = remaining         # Job categories to process (see below)
input_statuses = all         # Filter input frames by stored source status (see below)
continue_from_result = True # Resume from previous result (see below)
zip = True                  # Compress debug files
max_consecutive_errors = 5  # Kill worker after N consecutive all-error structures (0 = disabled)
restart_limit = 3           # executorlib: max worker restarts (0 = no restarts)

[FAIRChemCalculator]
device = cuda
name_or_path = uma-s-1p2
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
intermediate_minima_check_step = 1600  # 0 = disabled; >0 = one-shot imin detection at this optimizer step
intermediate_minima_min_depth = 0.05
max_num_frames = 80                # Enables one-shot image addition if > num_frames (FAIRChem only)
add_images_step = 0                # 0 = disabled; >0 = one-shot image addition at this optimizer step
dimer_refine_ci = False            # Post-NEB dimer on unconverged CIs (FAIRChem only)
dimer_refine_steps = 300
refine_band_steps = 0              # Continue NEB with same object if not converged; reuses all imin/CI/frozen state
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
pre_dimer_refine = False   # Rotation-only dimer to refine eigenmode before displacement (requires [DimerControl])

[ourSinglePoint]
frames_per_job = 1   # 1 (default) | N. With N>1, each executorlib job processes N frames in a single batched FAIRChem forward pass. VASP/VaspInteractive forces N=1 (no batched DFT).
# Required when Calculator = Vasp or VaspInteractive (same pattern in [ourDimer], [ourMinimization], [ourDoubleMinimization]):
vasp_command = "srun --exclusive -n 64 vasp_std"
vasp_ncore = 8
```

**VASP/VaspInteractive across all methods.** Every method instantiates per-job-unit VASP working directories via the shared `tools.resolve_vasp_calc()` helper:

| Method | Working dir | Notes |
|--------|------------|-------|
| NEB | `VASP_{job}_{image_idx}/` | Separate `vasp_command_endpoints`/`_intermediates` commands. |
| Dimer | `VASP_{job}_{attempt_id}/` | One dir per attempt — clean isolation across reaction-type displacements. |
| Minimization | `VASP_{job}/` | Single dir per job. |
| DoubleMinimization | `VASP_{job}_-1/`, `VASP_{job}_0/`, `VASP_{job}_1/` | Side ints match `info['side']`. TS dir reused for `pre_dimer_refine`; side dirs reused for desorption-check single-point + relaxation (WAVECAR warm-start). |
| SinglePoint | `VASP_{job}/` | `shutil.rmtree`'d at job end — SP has no `_debug_zips/`. |

`VaspInteractive.finalize()` is always called before swapping in `SinglePointCalculator`, and in error handlers, so the VASP subprocess never outlives the Python job. WAVECAR/CHG/CHGCAR are removed from each dir on success; remaining VASP files are zipped into `{method}_debug_zips/` (except SP). Resume cleanup picks up stale `VASP_*` dirs through `tools.clean_up_files()` for every method when Calculator is VASP/VaspInteractive.

### `run_jobs` — Flexible Job Selection

`run_jobs` specifies job categories to process. Fresh vs resume determined by whether `traj_files_ordered.json` exists (no explicit toggle; delete output dirs + json to force fresh start).

**4 categories** (each CSV status line independently categorized):

| Category | CSV statuses |
|---|---|
| `converged` | `converged`, `converged_after_extension`, `converged_CI` |
| `not_converged` | `not_converged`, `not_converged_after_extension`, `not_converged_StopRun` |
| `errored` | `error`, `error: <message>` |
| `remaining` | No CSV row for this job_id |

**Per-line selection**: A job is included if ANY of its CSV lines match requested categories. Only matching attempts/sides are redone; the rest stay in active output. **NEB exception**: categorization is at band level — "converged" only if ALL sub-bands have converged/converged_CI status. When a NEB job is selected, the entire band is re-run (no partial sub-band reruns).

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
- **Output trajs**: Archived as `{method}_trajes/previous_{N}.zip`. Frames removed by `attempt_id` (Dimer), `src_index` (NEB removes all band images, Minimization). DoubleMinimization removes all 3 frames (min1+TS+min2) together since they share reaction check metadata.
- **Debug zips**: Archived as `{method}_debug_zips/previous_{N}.zip`. Per-entry cleaning by subunit info in filenames (Dimer: `attempt_id`, DoubleMinimization: `-0`/`-1` suffix). NEB debug files removed whole when the band is redone.

Jobs with no prior entries don't trigger archiving.

### `continue_from_result` — Continue from Previous Result

When re-running previously completed entries, the system extracts previous results from output trajs. `continue_from_result` controls usage:

- `True` (default): continue from extracted structure.
- `False`: start fresh (Dimer: fresh attempts at same positions; NEB: original input; DoubleMinimization: re-displace from TS).

Errored entries always fall back to fresh. Continuation handled per-entry inside method functions; fresh and continuation jobs coexist in same batch.

| Method | Granularity | `True` | `False` |
|---|---|---|---|
| Dimer | per-attempt | Continue from extracted structure with eigenmode | Fresh attempt at same `attempt_id` (same type, new displacement) |
| NEB | full-band | Continue from extracted images. Seeds imin/frozen/frozen_fmax from previous result. Config controls new imin/image detection on top of seeded state. | Fresh run from original input. |
| DoubleMinimization | per-side | Continue not-converged side | Re-displace from TS |
| Minimization | per-job | Continue from extracted structure | Original input |

All methods extract via `extract_previous_results`. **`initial_guess` vs `continue_from_result`**: separate features — `initial_guess` is for **external** TS guesses (from another code/method); `continue_from_result` extracts from a **previous SaddleMill run**.

**Examples:**
```ini
run_jobs = not_converged
steps = 10000                     # More steps for not-converged NEB

run_jobs = converged
Calculator = VaspInteractive      # Refine with VASP

run_jobs = not_converged
continue_from_result = False      # Re-run from scratch
```

### `input_statuses` — Filter Input Frames by Source Status

Every output frame carries its CSV status in `.info['status']`. `input_statuses` filters *input* frames by that stored status, applied in `__main__.py` right after `load_and_sanitize` via `tools.passes_input_filter`. Matched frames submit; unmatched frames are silently skipped (no CSV row) so they remain "remaining" on a subsequent resume with a broader filter.

Patterns use `fnmatch` wildcards — e.g. `converged*` matches `converged`, `converged_CI`, `converged_after_extension`, `converged_to_desorption`, `converged_desorption_skipped`. Values are a single pattern or a space-separated list. The special value `all` (the default) bypasses the filter entirely.

When an explicit filter is set, frames whose `orig_info` lacks a `status` field (raw `.traj` files that have never been through SaddleMill) are rejected — an explicit filter is a deliberate "only these statuses" request. Use the default `all` for first-time runs on user-prepared inputs.

Status strings by source:
- **Dimer**: `converged`, `converged_after_extension`, `converged_to_desorption`, `not_converged`, `not_converged_after_extension`, `not_converged_StopRun`, `error: ...`
- **NEB** (per sub-band, tagged onto every image in the segment): `converged`, `converged_CI`, `not_converged`
- **Minimization**: `converged`, `not_converged`
- **DoubleMinimization**: `converged`, `converged_desorption_skipped`, `not_converged` (TS frame always `converged`)

**Examples:**
```ini
method = DoubleMinimization
dir_path = /path/to/NEB_trajes
input_statuses = converged_CI                       # only CI-only converged sub-bands
input_statuses = converged                          # only fully-converged sub-bands
input_statuses = converged converged_CI             # both together
input_statuses = converged*                         # any converged* variant
input_statuses = all                                # no filtering (default)

dir_path = /path/to/Dimer_trajes
input_statuses = converged converged_after_extension  # Dimer converged, no desorption
```

## Output Structure

For method `NEB`:
```
NEB_status_csvs/status_rank_*.csv    # job_id,rank,sub_band_id,status
NEB_trajes/collected_ts_rank_*.traj  # All band images with metadata in atoms.info
NEB_debug_zips/                      # Compressed log/traj/plot files
```

Every output frame (all methods) carries `status` — the CSV status string that was written for that entry, used by `input_statuses` for downstream filtering.

All output frames also carry `task_name` (the FAIRChem task that produced them, or `None` for VASP) so downstream filtering/training can keep tasks separated.

NEB output image metadata: `src_index`, `image_idx`, `subband_idx`, `image_type` (endpoint/intermediate_minimum/climbing/regular), `effective_fmax`, `image_converged`, `band_converged`, `band_converged_CI`, `status`, `nimages`, `interpolation_method`, `imin_set`/`climbing_set`/`frozen_set` (band-wide, stamped on each image), `task_name`, `orig_info`. CI images also get `eigenmode`, `barrier`, `dE`. Every image in a sub-band shares the sub-band's `status` (`converged` / `converged_CI` / `not_converged`). Imin images duplicated for sub-band self-containment. NEB CSV is still per-sub-band lines; band-level run_jobs categorization requires ALL sub-bands converged/converged_CI.

Dimer output: `eigenmode`, `curvature`, `converged`, `src_index`, `attempt_id`, `stoprun`, `selected_index`, `reaction_type`, `status`, `task_name`, `orig_info`.

DoubleMinimization output: `side` (-1/0/1), `parent_ts_index`, `converged`, `src_index`, full reaction-detection dict (`is_reaction`, `n_formed_bonds`, `n_broken_bonds`, `broken_bonds`, `formed_bonds`, plus `is_ads_reaction` / `n_ads_*` / `ads_*_bonds` for OC inputs), `status` (`converged` / `converged_desorption_skipped` / `not_converged`; TS frame always `converged`), `task_name`, optional `curvature` on the TS frame when `pre_dimer_refine=True`, `orig_info`. CSV: 2 lines per job `{job_id},{rank},{side_id},{parent_ts_idx},"{status}"`.

For method `SinglePoint`:
```
SinglePoint_status_csvs/status_rank_*.csv          # job_id,rank,status (one line per job; triplets log once)
SinglePoint_trajes/collected_sp_rank_*.traj        # only when input_format=traj
SinglePoint_lmdbs/collected_sp_rank_*.aselmdb      # only when input_format=lmdb
```
SP creates only the output directory matching the active `input_format` (plus `_status_csvs/`). No `_debug_zips/`. SP output frames carry `src_index`, `status` (always `converged` on success), `task_name`, `orig_info` (for .traj input via `load_and_sanitize`). For LMDB input, each output row mirrors the source row's `key_value_pairs` and `data` (`info` + `traj_path`) verbatim, with `row.energy` / `row.forces` newly populated via `SinglePointCalculator` on the atoms object — i.e. byte-equivalent to a fresh `build_lmdb_parallel.py` run on E/F-bearing trajectories. With `frames_per_job = 3`, the three frames of each triplet are evaluated in one batched FAIRChem forward pass and written contiguously in input order to the same rank shard.

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
│   ├── converged_ts.traj        # Pre-converged TS for doublegeomopt tests
│   ├── minimization_input.traj  # 68-atom slab+adsorbate
│   ├── oc_adsorbate_slab.traj   # 131-atom slab+adsorbate
│   └── oc_neb_pair.traj         # 10-frame NEB, 68 atoms
├── test_config.py               # ConfigManager, load_*, run_jobs, archive/clean (77, CPU)
├── test_tools.py                # load_and_sanitize, check_reaction, extraction, get_task_name, passes_input_filter (42, CPU)
├── test_structure_edit.py       # Bulk & OC reaction types, supercell (42, CPU)
├── test_ocpneb.py               # swDNEB, forces, frozen images (20, mixed)
├── test_spc_wrap_ordering.py    # SinglePointCalculator vs Atoms.wrap() ordering (4, CPU)
├── test_nebopt_integration.py   # Full NEB runs (8, GPU)
├── test_dimeropt_integration.py # Dimer runs (7, GPU)
├── test_geomopt_integration.py  # geomopt + doublegeomopt (5, GPU)
├── test_init_function.py        # init_function (5, GPU)
└── test_main_integration.py     # End-to-end pipeline + resume (6, GPU)
```

Markers: `@pytest.mark.gpu` (CUDA, auto-skipped), `@pytest.mark.flux` (Flux scheduler), `@pytest.mark.slow` (>60s)

### Writing New Tests

- `make_config_dict(method="NEB", **overrides)` from `conftest.py`
- Session-scoped `fairchem_calc` fixture for GPU tests
- `converged_ts_atoms` fixture for doublegeomopt tests
- Integration tests use `tmp_path` + `monkeypatch.chdir()`
- CPU structure_edit tests seed with `random.seed(42)` + `np.random.seed(42)`

## HPC Setup (4 A100 per node HPC)

```bash
#SBATCH --constraint=gpu
#SBATCH --gpus-per-node=4
#SBATCH --nodes=N
srun -N $SLURM_NNODES -n $SLURM_NNODES --gpus-per-node=4 flux start python -u -m saddlemill
```

Set `FAIRCHEM_CACHE_DIR` for model caching. Requires CUDA libraries in `LD_LIBRARY_PATH`.
