# SaddleMill - High-Throughput Transition State Search Library

## Overview

SaddleMill is a Python library for creating datasets of Transition States (TS) using neural network potentials (FAIRChemCalculator / Meta's UMA model) or DFT (VASP / VaspInteractive). It supports distributed GPU execution on HPC systems (4 A100 per node, GH200, 3 A100 per node) via executorlib + Flux.


> **Configuration source-of-truth rule.** Current checked-out SaddleMill source, `config.py`, and this `CLAUDE.md` define the accepted schema. Historical campaign configs are scientific records, not schema authority; validate them against the current parser before reuse and stop on changed/invalid semantics rather than silently translating them.

## Dependencies

Minimum required versions (baseline, enforced in `pyproject.toml`):
- `ase >= 3.26.0`
- `fairchem-core >= 2.19.0`

Both are required by the package — `fairchem-core` registers the LMDB backend that `ase.db.connect` uses for `.aselmdb` files, and `ase >= 3.26.0` is the first ASE release whose `db` layer wires up that backend correctly. Any code path that does ASE LMDB I/O must `import fairchem.core.datasets` before calling `ase.db.connect`.

**VaspInteractive**: always use the ulissigroup implementation (`from vasp_interactive import VaspInteractive`, installed via the git URL in `README.md`). **Never use ASE's bundled `ase.calculators.vasp.interactive.VaspInteractive` — it is broken.** Optional VASP-input-generation and VaspInteractive install steps live in `README.md`; refer there rather than hardcoding install commands.

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

> **Config-section philosophy (do not violate).** A config section named after an
> external class/library — `[Vasp]`, `[FAIRChemCalculator]`, `[DimerControl]`,
> `[BaseNEB]`, `[MDMin]`/`[LBFGS]`/`[FIRE]`/`[Kappa]` — is a **pure pass-through**: every key
> in it is handed verbatim as `**kwargs` to that class (`Vasp(**...)`,
> `DimerControl(**...)`, etc.). **Never** add a SaddleMill-specific key to one of
> these — it would be forwarded to the external class (e.g. ASE's `Vasp` writes
> unknown keys straight into INCAR via its type-based fallback) and corrupt its
> input. All SaddleMill knobs live in `[Main]` or in an `our*` section
> (`[ourNEB]`, `[ourDimer]`, `[ourMinimization]`, `[ourDoubleMinimization]`,
> `[ourSinglePoint]`, `[ourVasp]`). This is why `input_generator` /
> `extra_input_files` are in `[ourVasp]`, not `[Vasp]`. The section name signals
> ownership: a bare class name = "the user is configuring that class"; an `our`
> prefix = "SaddleMill owns this." Keep that boundary clean.

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
4. **NEB optimization**: Uses `OCPNEB` with the configured band optimizer (default MDMin). Supports climbing image and intermediate minima (per-segment climbing). For VASP/VaspInteractive: per-image calculators in `VASP_{job_id}_{image_idx}/` dirs, separate `command`/`ncore` for endpoints vs intermediates, WAVECAR/CHG/CHGCAR cleanup after completion. VaspInteractive endpoint calculators finalized before `SinglePointCalculator` replacement; intermediate calculators finalized after NEB completes. The optimization is split into phases driven by **one-shot events**: `intermediate_minima_check_step` triggers imin detection at a specific step (one-shot, not periodic; 0 = disabled); `add_images_step` triggers image addition at a specific step (one-shot). When an event fires, NEB stops, the event runs, and a fresh optimizer instance resumes the same NEB object until the next event or the global step budget runs out. **On-the-fly imin detection (`intermediate_minima_check_step`) is FAIRChem-only** — like auto image-addition, auto-freeze, and post-NEB refinement, it is skipped for VASP/VaspInteractive (with a warning) because `_relax_imin()` needs the shared calculator instance, which VASP modes don't have per-image. Seeded imin from a continuation (`initial_imin_set`) still work in VASP mode. So VASP NEB is effectively plain climbing-image NEB with seeded-only intermediate-minima segmentation.
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
- **Per-attempt execution**: Accepts `entries_to_run` (set of attempt_ids) and `continuation_data` (dict: attempt_id → Atoms). Calls `get_attempts()` on original input, then per attempt: skips if not in `entries_to_run`, uses `continuation_data[attempt_id]` if available (near-zero displacement),
- else fresh attempt. Eigenmode/reaction_type read from top-level `.info` first, `orig_info` fallback.
- **Engine Selection**: `[ourDimer] engine` supports standard ASE Dimer (`ase`, default), Kappa Dimer (`kappa`), and Sella (`sella`). Sella is dispatched through `sella_engine.py`; ASE/Kappa Dimer behavior remains on the existing path unless `engine = sella` is explicitly selected.
- **Corrected Kappa regime**: Phase B is run only after Phase A finds non-positive curvature **and** the real center `fmax >= kappa_recover_fmax`. Positive Phase-A curvature never uses Kappa gamma weighting; translation is the stock Dimer drag-up force `-F_parallel`. Below `kappa_recover_fmax`, Kappa is inactive and translation is the normal Dimer force. Trial positions do not independently flip the accepted center's Kappa regime.
- **Independent Dimer optimizers**: `rotation_optimizer = ase | lbfgs` and `translation_optimizer = ase | lbfgs`. These selectors apply only to ASE/Kappa Dimer. Selecting either L-BFGS option is a scientific-method change; the default remains ASE. No separate Kappa-LBFGS module exists: Kappa rotation selection is implemented in the single authoritative `dimertools/kappa_dimer.py`.
- **Reaction Attempts Mapping**: `num_attempts_per_type` accepts either a single integer (applied globally to all configured `reaction_types`) or a space-separated list of integers mapping 1:1 to the `reaction_types` list.
- **Status CSV Tracking**: The Dimer status CSV now records the total number of **force calls** for each attempt, providing a direct metric for computational cost alongside the convergence status.
<!-- BEGIN DIMER_LBFGS_DOCS -->
### `dimertools/lbfgs_dimer.py` - Configurable Dimer Rotation and Translation
- **Independent solver selection**: `_setup_dimer()` treats the dimer engine, rotational optimizer, and translational optimizer as separate choices. `[ourDimer] engine = ase | kappa | sella`, `rotation_optimizer = ase | lbfgs`, and `translation_optimizer = ase | lbfgs | fire_lbfgs` (`hybrid` remains a compatibility alias for `fire_lbfgs`). Defaults remain ASE rotation and stock ASE dimer translation.
- **L-BFGS rotation remains specialized**: `LimitedMemoryInverseHessian` is now used only by the custom rotational search. `LBFGSDimerEigenmodeSearch` changes the rotational search direction in the tangent space of the normalized dimer mode while retaining ASE's trial-angle evaluation, Fourier interpolation, endpoint-force extrapolation, curvature definition, and physical rotational-force stopping criteria. Rotation history exists only within the rotations at one centre geometry and is reset after translation because a new search object is created.
- **Kappa rotation histories**: With `engine = kappa` and `rotation_optimizer = lbfgs`, Phase A and constrained Phase B each receive a separate rotational history. This changes only the rotational direction solver; it does not remove or redefine the constrained Phase-B curvature calculation.
- **Pure L-BFGS translation is ASE-backed**: `LBFGSMinModeTranslate` subclasses `ase.optimize.LBFGS` and passes the live `MinModeAtoms` object directly to it. `MinModeAtoms.get_forces()` therefore supplies ASE LBFGS with the projected minimum-mode-following force. SaddleMill does not implement a second translation two-loop recursion. `use_line_search=False` because the physical potential energy is not the scalar objective whose gradient is the projected saddle-search force.
- **Translation scaling**: `[ourDimerLBFGS] translation_initial_hessian` maps directly to ASE LBFGS `alpha`. ASE initializes the inverse Hessian as `1/alpha`; therefore a larger value gives a smaller, more conservative initial step. The default is `70.0`, matching ASE. `translation_memory` and `translation_damping` map to ASE `memory` and `damping`. `translation_dynamic_h0` is retained only for config compatibility and is inert for ASE-backed translation.
- **Step-limit semantics**: ASE LBFGS applies `maximum_translation` as its `maxstep`, limiting the largest individual atomic displacement. Stock `MinModeTranslate` and ASE FIRE instead cap the global Cartesian displacement norm. This difference is part of the selected translation algorithm and can change trajectories; it must not be described as a performance-only change.
- **Warm FIRE -> L-BFGS translation**: `HybridMinModeTranslate` delegates FIRE steps to `ase.optimize.FIRE` and L-BFGS steps to `ase.optimize.LBFGS`. With `warm_start_history=True`, SaddleMill stores accepted generalized coordinates and projected forces, then rebuilds ASE's native L-BFGS state using the same update/iteration sequence as ASE trajectory replay. No custom translation two-loop recursion is used. With `warm_start_history=False`, the L-BFGS phase starts from ASE's `1/alpha` initial inverse Hessian.
- **Dimer hysteresis**: FIRE -> L-BFGS entry requires projected dimer `fmax <= enter_fmax`, curvature `<= enter_curvature`, and (for a warm start) at least `minimum_history_pairs`, sustained for `enter_stable_steps`. L-BFGS -> FIRE exit requires either `fmax >= exit_fmax` or curvature `>= exit_curvature` for `exit_stable_steps`.
- **Kappa translation regime changes**: FIRE and L-BFGS consume the force returned by `KappaMinModeAtoms.get_projected_forces()`. With `reset_translation_on_regime_change=True`, a `kappa` <-> `standard` switch discards the ASE L-BFGS state and warm snapshot buffer before further updates.
- **DimerControl isolation**: all SaddleMill-created dimer controls use `IsolatedDimerControl`, which copies ASE's class-level `parameters` dictionary before applying instance overrides. One attempt's `[DimerControl]` or `[Kappa]` values therefore cannot mutate defaults seen by later control instances.
- **Scientific-method boundary**: Selecting L-BFGS rotation, ASE L-BFGS translation, or FIRE/L-BFGS translation changes the optimization trajectory and can change which saddle is reached. The translation rewrite from the former custom recursion to ASE LBFGS and the default `translation_initial_hessian` change from 1 to 70 are scientific-method changes. Old custom-L-BFGS benchmark arms are not directly comparable to the corrected ASE-backed arms.

**Dimer diagnostic shards** (all append by worker rank; reruns are separated by `trace_id`):
- `Dimer_status_csvs/reaction_rank_<rank>.csv`: configured, initial, and final reaction labels plus convergence/status and exact dimer force-call total. This is a diagnostic shard, not a canonical status shard; status readers must match only `status_rank_*.csv`.
- `Dimer_mode_csvs/mode_rank_<rank>.csv`: one row at translation state 0 and after each applied translation, including cumulative force calls, curvature, participation ratio, projective mode angles, mode coherence, and atom-participation overlap over 5- and 10-state windows.
- `Dimer_optimizer_csvs/optimizer_rank_<rank>.csv`: one `step` row per applied Cartesian translation plus one `summary` row. The CSV columns are preserved. Translation-history counts now report ASE's stored/update state; ASE 3.29 does not apply the former custom positive-`s dot y` pair-rejection rule. `step_norm` remains the actual global displacement norm, while `step_clipped` follows the active ASE optimizer's native max-step rule.
<!-- END DIMER_LBFGS_DOCS -->


<!-- BEGIN SELLA_ENGINE_DOCS -->
### `sella_engine.py` - Optional Sella Saddle Engine
- `[ourDimer] engine = sella` dispatches the existing Dimer attempt workflow to Sella 2.5.0 while preserving attempt generation, resume identity, status CSVs, collected trajectories, calculator lifecycle, and per-attempt error handling.
- Sella is a separate first-order saddle-search algorithm. `rotation_optimizer` and `translation_optimizer` apply only to ASE/Kappa Dimer and must remain `ase` when `engine = sella`; Sella options live in `[ourSella]`.
- Default/validated mode is Cartesian (`internal = False`). Convergence may optionally require exactly one negative eigenvalue in Sella's final approximate constrained model Hessian.
- Canonical Dimer metadata stores Sella's PES evaluation counter (`optimizer.pes.neval`) as `n_force_calls`. This is also written to the status CSV; benchmark scripts should independently count calculator evaluations when exact cross-engine accounting matters.
<!-- END SELLA_ENGINE_DOCS -->

### `dimertools/structure_edit.py` - Reaction Attempts for Dimer
`get_attempts()` returns three position-aligned lists: generated `Atoms`, ASE Dimer displacement dictionaries, and selected atom indices. Their list position is the global Dimer `attempt_id`, so every configured reaction type must retain exactly its requested number of slots. A mechanism that cannot generate a candidate returns `None` in that slot; it must not shorten the list and shift later reaction types.

**Bulk reaction types** (`dataset_type = bulk`): `vacancy`, `hop_reuse`, `hop_insert`, `kickout_reuse`, `displace_kickout_reuse`, `kickout_insert`, `ring`, `initial_guess`, `all_atoms`, and `random_bubble`.

- `hop_reuse`, `kickout_reuse`, `displace_kickout_reuse`, and `ring` enumerate deterministic ranked candidates. `[ourDimer] bulk_reuse_offset` selects the first ranked candidate; `sm_offset` and `SM_OFFSET` are compatibility aliases. `SM_SEED_OFFSET` is deliberately unrelated: it changes the random realization in `dimeropt.py`, not the ranked candidate index.
- Ranked-candidate exhaustion never creates Gaussian filler. `[ourDimer] reuse_exhaustion = stop` is the only accepted policy. When candidates run out, the mechanism pads the remaining configured slots with `None`, preserving attempt IDs and configured reaction-type accounting.
- `kickout_reuse` is the site-driven existing-atom mechanism. `displace_kickout_reuse` explicitly enumerates `(void, kicked, kicker)` collision chains, moving the kicker toward the kicked atom and the kicked atom toward the selected void.
- `ring` enumerates simple cycles for `ring_sizes`, ranks them deterministically, and supports `ring_mode = arc | chord | legacy`. `ring_frac` is the fraction of a full `2*pi/n` exchange rotation; the default arc mode preserves ring bond lengths during the initial displacement.
- `initial_guess` is exclusive, skips supercell expansion, and always returns one near-zero-displacement attempt. A top-level `atoms.info['eigenmode']` is authoritative; nested `orig_info['eigenmode']` is used only when the top-level mode is absent.

**OC reaction types** (`dataset_type = oc`): `all_movable`, `adsorbate_atom`, `adsorbate_atom_neighbors`, `adsorbate`, `diffusion`, `rotation`, `adsorbate_surface`, `surface`, `custom`, `initial_guess`, and `random_bubble`. Tag-0 substrate atoms are fixed. Adsorbate-specific generators require tag 2.

**Gaussian and concentrated replacements:**
- `[ourDimer] gaussian_normal_attempts = N` gives a deterministic sequence of `N` directed attempts followed by one Gaussian replacement; `0` disables scheduled Gaussian replacements. The first slot is always directed. Scheduled Gaussian slots preserve the directed mechanism's atom set and do not consume its ranked candidate.
- `gaussian_swap_prob` is a deprecated integer-only alias for `gaussian_normal_attempts`; decimal probabilities fail loudly instead of silently changing meaning.
- `[ourDimer] concentrate_prob` independently replaces a Gaussian displacement with the power-law-concentrated vector on the same eligible atom set. Its default is `0.0` (off). `concentrate_power`, `concentrate_std`, `concentrate_max_disp`, and `concentrate_envelope` control the concentration shape and normalization.

**Other infrastructure:**
- `turn_into_supercell(atoms, min_length=7.0)` preserves `.info` and is centralized in `get_attempts()`. OC systems containing an adsorbate are not expanded because that would duplicate the adsorbate.
- `find_interstitial_sites()` uses Voronoi vertices from periodic images, a nearest-atom distance filter, and 0.5 A clustering.
- `num_attempts_per_type` may be one integer for all configured types or a position-aligned list with one count per type.

### `geomopt.py` - Geometry Optimization
- `geomopt()`: Standard relaxation with optional cell relaxation (FrechetCellFilter). Output frames carry `task_name` (from `get_task_name`).
- `doublegeomopt()`: Takes converged TS with eigenmode, displaces ±0.25*eigenmode, relaxes both directions, detects bond breaking/forming via `check_reaction()` and `check_adsorbate_reaction()`. Reads `eigenmode`, `converged`, `src_index` from `atoms.info['orig_info']` (fallback `atoms.info`). Each frame gets `side` in `.info` (-1=min1, 0=ts, 1=min2). Writes 2 CSV lines per job in the form `{job_id},{rank},{side_id},{parent_source_idx},"{status_msg}"`. Accepts `entries_to_run`/`continuation_data` for per-side execution. Optional `pre_dimer_refine=True` (default False) runs a rotation-only dimer step via `_refine_eigenmode()` from `dimeropt.py` to refine the eigenmode direction before displacement. Rotation parameters controlled by `[DimerControl]` section (especially `max_num_rot`, `dimer_separation`). Stores refined eigenmode and `curvature` on the TS output frame's `.info`. The full reaction-detection metadata dict (`is_reaction`, `n_formed_bonds`, `n_broken_bonds`, `broken_bonds`, `formed_bonds`, plus `is_ads_reaction` / `n_ads_formed_bonds` / `n_ads_broken_bonds` / `ads_broken_bonds` / `ads_formed_bonds` for OC inputs) is copied onto every emitted frame (min1, TS, min2) along with `parent_ts_index` and `task_name`.
- **Desorption-skip logic**: when `check_adsorbate_reaction()` flags the bond change as a desorption, the higher-energy side is skipped to avoid relaxing into vacuum; that side gets `side_statuses[side] = "converged_desorption_skipped"` while the other side still runs normally. The TS frame's `status` always reflects the converged TS itself.
- `singlepoint()`: One E/F call per frame. Only method that supports `.aselmdb` input (via `[Main] input_format = lmdb`). With `[ourSinglePoint] frames_per_job = N` (any positive integer) it bundles N frames per executorlib job and computes their energies and forces in a single batched FAIRChem forward pass via `fairchem.core.datasets.atomic_data.atomicdata_list_to_batch` + `calc.predictor.predict`, then writes the frames contiguously in input order to the same rank shard. Per-frame `natoms` may vary within a batch — forces are sliced by cumulative `natoms` offsets. The last batch in each shard may be smaller than N (no divisibility check). For triplet-respecting use cases (e.g. DM output min1/TS/min2), pick N as a multiple of 3 so triplets stay intact. No optimizer and no temp log/traj files; no debug zips for FAIRChem (VASP SP keeps them — see the VASP note below). Output goes to `SinglePoint_trajes/collected_sp_rank_{rank}.traj` (traj input) or `SinglePoint_lmdbs/collected_sp_rank_{rank}.aselmdb` (lmdb input). LMDB output preserves the source row's `key_value_pairs` and `data` (`info` + `traj_path`) verbatim and populates `row.energy` / `row.forces` via `SinglePointCalculator`. Does **not** call `atoms.wrap()` — the structure is untouched, only E/F are added. The SinglePoint contract is to leave the source structure and its info untouched and only add E/F (plus any opted-in `extra_outputs`); because `ase.db` never serializes `atoms.info` (only the explicit `data=` blob), the `src_index`/`status`/`task_name` bookkeeping that traj output stamps onto `.info` is traj-output-only — the lmdb exceptions are opted-in `extra_outputs` and (for VASP) a `converged` flag, merged into `data['info']` (see VASP note below). **VASP/VaspInteractive**: supported with `frames_per_job = 1` only (no batched DFT). Each job uses one `VASP_{i}/` scratch dir; on success the heavies (WAVECAR/CHG/CHGCAR) are dropped and the rest is archived into `SinglePoint_debug_zips/structure_rank_{rank}_data.zip` exactly like the other VASP methods (gated on `[Main] zip`; `zip = False` just deletes the dir — the old behavior), and on error the dir is archived with an `ERROR_` prefix. After the run, anything an `[ourVasp] extra_outputs` parser left on `calc.sm_extra_outputs` (e.g. VTST `eigenmode`/`curvature`) is merged into the output frame's `.info` (traj output) or the output row's `data['info']` (lmdb output) before that archiving — the opted-in extras ride along regardless of output format. **For VASP the per-frame `status` is the real convergence verdict** — `calc.converged` (ASE's OUTCAR 'reached required accuracy', valid for `ibrion in [1,2,3]`/`nsw!=0`) OR the EDIFFG force criterion VTST itself uses (max per-atom `|F| <= |EDIFFG|` on the true force, *not* the DIMCAR total-norm `Force`) — never an unconditional `converged`; a final ionic step whose SCF didn't reach EDIFF logs `error: scf_not_converged` (`tools.vasp_final_scf_converged`), and `info['converged']` / lmdb `data['info']['converged']` records 1/0. A VTST dimer that drifts toward a minimum simply never meets EDIFFG → `not_converged` (it cannot converge to a minimum: `dimer.F` `ProjectDimer` climbs uphill when curvature ≥ 0). FAIRChem SP (and any `nsw=0` single-point) stays unconditional `converged`. LMDB input restricted to `run_jobs = remaining`.

<!-- BEGIN FIRE_LBFGS_MINIMIZER_DOCS -->
### `fire_lbfgs.py` - Warm-Started ASE FIRE/L-BFGS Minimizer
- `FIRELBFGS` is opt-in for ordinary `Minimization` and each independently relaxed `DoubleMinimization` side. Select it with `[Main] Optimizer = FIRELBFGS`.
- The outer controller operates through ASE's generalized `optimizable` coordinates, so normal `Atoms` and `FrechetCellFilter` follow the same implementation.
- FIRE steps are delegated to `ase.optimize.FIRE`. L-BFGS steps are delegated to `ase.optimize.LBFGS`; SaddleMill no longer supplies a separate minimizer two-loop recursion.
- With `warm_start_history=True`, accepted generalized-coordinate/projected-force snapshots from FIRE are replayed into ASE's native L-BFGS update state before the first L-BFGS step. `warm_start_history=False` gives an ASE cold start. `lbfgs_initial_hessian` is ASE's `alpha`, so the initial inverse-Hessian scale is `1/alpha`; the default remains 70.
- The controller uses fmax hysteresis: FIRE -> L-BFGS after `fmax <= enter_fmax` for `enter_stable_steps` with enough available history; L-BFGS -> FIRE after `fmax >= exit_fmax` for `exit_stable_steps`. `reset_history_on_exit` controls whether the buffered warm history and ASE state are discarded on a threshold exit.
- `lbfgs_dynamic_h0` and `lbfgs_curvature_epsilon` remain accepted for config compatibility but are inert in the ASE-backed implementation. ASE restart files remain unsupported; SaddleMill continuation resumes from the saved structure with new in-memory optimizer state.
- Existing optimizer diagnostic field names are preserved. They now report ASE L-BFGS history/reset state and the active ASE FIRE/L-BFGS step.
- **Legacy accounting warning**: existing Minimization/DoubleMinimization output metadata named `n_force_calls` remains populated from optimizer steps, not exact calculator force calls.
<!-- END FIRE_LBFGS_MINIMIZER_DOCS -->

### `tools.py` - Utilities
- `load_and_sanitize(traj, i, j)`: Loads atoms and stashes original `.info` into `atoms.info = {"orig_info": <original_info>}`. Prevents per-atom array data from causing size mismatches on atom add/remove. Called in `__main__.py` for all methods.
- `passes_input_filter(images, config_dict)`: Applies the `input_statuses` fnmatch filter to a sanitized input — checks `orig_info['status']`, returns False if a filter is set and the frame's status doesn't match (or `status` is missing). Returns True for `input_statuses = all`.
- `save_ordered_traj_names()` / `read_ordered_traj_names()`: Persist/read the `traj_files_ordered.json` that anchors the resume contract.
- `clean_up_files(config_dict)`: Removes leftover temp files on resume. Method-aware: NEB (`neb_*.log/traj`, `reactant/product_relaxation_*`, `imin_relax_*`, `diffusion_barrier_*.png`, `VASP_*_*/`), Dimer (`dimer_*.log/traj`), Minimization (`optimization_*.log/traj`).
- `backup_flux_logs(worker_id)`: Snapshots flux log files for a worker before it self-kills, so the post-restart logs don't overwrite the diagnostic trail.
- `get_task_name(config_dict)`: Returns `[FAIRChemCalculator] task_name` if FAIRChem is the calculator, else `None`. Used by all methods to stamp `task_name` on output frames so downstream consumers know which UMA task generated them.
- **Previous result extraction**: `extract_previous_results(job_ids, config_dict, redo_info)` — unified extraction from output trajs for all methods. Returns `{job_id: continuation_data}` (Dimer: `{attempt_id: Atoms}`, NEB: `{subband_idx: [Atoms]}`, DoubleMinimization: `{side: Atoms}`, Minimization: `Atoms`). Helpers: `_build_output_traj_index()`, `_sanitize_with_continuation()`.
- `get_bond_set()`, `check_reaction()`, `check_adsorbate_reaction()`: Bond detection via ASE neighbor_list with natural cutoffs, compare connectivity between structures.
- **VASP helpers**: `resolve_vasp_calc(config_dict, calc, i, subunit_id, section, atoms=None)` builds a per-job-unit `Vasp`/`VaspInteractive` instance (FAIRChem path returns the shared instance). `vasp_incar_kwargs(config_dict, atoms=None)` computes the calculator's INCAR/k-point/setup kwargs: it evaluates an optional `[ourVasp] input_generator` on `atoms`, then layers the explicit `[Vasp]` keys on top so **`[Vasp]` always wins** (per-tag precedence: `[Vasp]` > generator > ASE default). `[Vasp]` is passed verbatim to ASE (the orchestration keys live in `[ourVasp]`, never in `[Vasp]`); with no generator or no `atoms` the result is just the `[Vasp]` section. `resolve_vasp_calc_class(config_dict, calc)` wraps the calc class via `_with_extra_io()` when `[ourVasp] extra_input_files` and/or `extra_outputs` are set: the subclass's `write_input` drops extra files in (e.g. a VTST MODECAR), and its `read_results` runs output parsers and stashes their merged dict on `calc.sm_extra_outputs`. Used by both `resolve_vasp_calc` and `nebopt._build_neb_vasp_calc` so the hooks are uniform across all methods. `remove_vasp_heavies()` / `finalize_if_vasp_interactive()` / `archive_and_clear_temp_files()` round out the VASP lifecycle helpers.

### `vasp_io.py` - Pluggable VASP input generation + extra input/output hooks
A *generator* is a callable `generator(atoms) -> dict` of ASE-`Vasp` kwargs (lowercased INCAR tags + `kpts`/`gamma`/`setups`/`magmom`). Selected via `[ourVasp] input_generator` (a SaddleMill section — **not** `[Vasp]`, which is a pure ASE pass-through), which may be a built-in name (`omat24_static`, `omat24_relax`, `cheap_omat`, `oc20`, `cheap_oc20`, `oc22`, `cheap_oc22`), a dotted `module:func`, or a `/path/file.py:func`. SaddleMill never lets the generator write files — it only computes *settings*, which `vasp_incar_kwargs` merges under `[Vasp]` and hands to ASE, so atom sorting / force resort / POTCAR / VaspInteractive interactive-flags stay correct. Applies to **all five methods** (every per-job-unit calc routes through `vasp_incar_kwargs`). `load_input_generator(spec)` resolves the spec (validated fail-fast in `load_method`; `_import_callable` handles the `module:func` / `file.py:func` forms, shared with the writer loader below). Built-in adapters convert a pymatgen `VaspInputSet` (`AseAtomsAdaptor.get_structure` → `OMat24*Set(struct, sort_structure=False)`) into ASE kwargs via `_pmg_set_to_ase_kwargs` (which also rewrites pymatgen's *positional* `LDAUU`/`LDAUL`/`LDAUJ` lists — aligned to pymatgen's `poscar.site_symbols` — into ASE's element-keyed `ldau_luj` dict, so DFT+U survives ASE's alphabetical atom re-sort and lands on the right species rather than being silently misassigned; `MAGMOM` needs no such fix since it is per-atom and ASE re-sorts it). `cheap_omat` is a reduced-cost OMat24 variant for a first-pass saddle search — `reciprocal_density` 16 (vs 64) plus `minimal`-base light POTCARs (plain transition metals, soft `_s` O/C/N) so ENCUT can drop to ~300; reconverge the result with `omat24_static`. `oc20` uses `fairchem.data.oc`'s `VASP_FLAGS` + `calculate_surface_k_points` — exactly Meta's OC20 settings (RPBE, ENCUT 350, no spin, `minimal` setups). `oc22` reproduces Meta's OC22 settings exactly (arXiv:2206.08917 — WhereWulff `MOSurfaceSet` as rendered by 2022-era pymatgen, cross-checked against the paper SI): PBE + MP Hubbard U (element-keyed `ldau_luj`, applied iff O/F present AND a U-metal present), ISPIN 2 with MP-default MAGMOM (atoms' own magmoms win), ENCUT 500, EDIFF 1e-4, ISMEAR 0/SIGMA 0.05, ALGO Fast, PREC Accurate, LREAL False, LASPH/LORBIT 11, NELM 60/NELMIN 8, ISTART 1, LWAVE/LVTOT True, LMAXMIX 4/6 by d/f-block, Γ-centered `ceil(30/a)×ceil(30/b)×1` k-mesh, 2022 MPRelaxSet POTCAR table with the W_sv fix (PBE 5.4 library — `pp_version = 54` in `[Vasp]`), and dipole correction (`ldipol`/`idipol=3`/`dipol`=mass-weighted COM) iff tag==2 adsorbate atoms exist (Meta: adslabs only). The settings are **hardcoded** (tables `_OC22_U`/`_OC22_MAGMOM`/`_OC22_SETUPS`, helper `_oc22_kwargs`), NOT derived via pymatgen — the upstream `OC22_dataset` branch is deprecated and modern pymatgen has drifted from what Meta ran (injects ENAUG=4000/EDIFF 1e-5, changed LDAU applicability, updated POTCAR maps); do not "simplify" oc22 back onto pymatgen sets. `cheap_oc20`/`cheap_oc22` mirror `cheap_omat`: in-plane k-multiplier halved (40→20 / 30→15), `minimal`-base setups with soft `_s` O/C/N and f-in-core lanthanides kept (module-level `_LANTH_FCORE`; cheap_oc22 keeps its parent's `_3`/Yb `_2` picks), everything else — including spin/U/functional — untouched so the cheap pass stays on the same PES; drop ENCUT to ~300 via `[Vasp]`, reconverge with the exact preset. **Ionic-driver tags `IBRION`/`NSW`/`POTIM`/`EDIFFG` (`_DRIVER_KEYS`) are stripped** from generator output — SaddleMill drives geometry through ASE optimizers and VaspInteractive forbids overriding `IBRION`/`POTIM`. Built-ins need `fairchem-data-omat` (omat24_*) or `fairchem-data-oc` (oc20/cheap_oc20); `oc22`/`cheap_oc22` are dependency-free; imports are lazy so the loader works without them.

**Extra-input-file writers** (`[ourVasp] extra_input_files`) and **extra-output parsers** (`[ourVasp] extra_outputs`): a symmetric pair around the VASP run, both applied by `resolve_vasp_calc_class` via the `tools._with_extra_io` calc subclass.
- *Writers* `writer(calc, atoms, directory) -> None` drop files INTO the dir; they run in the subclass's `write_input` (after ASE writes its inputs, so the dir exists and `calc.sort` is set, before VASP runs). Built-in `write_modecar` reads `atoms.info['eigenmode']` (`orig_info` fallback), reorders to POSCAR order via `calc.sort`, normalizes, writes `MODECAR` (no-op + warning if no eigenmode).
- *Parsers* `parser(calc, atoms, directory) -> dict` read FROM the dir; they run in the subclass's `read_results` (after VASP finishes, `calc.resort` set) and the merged dict is stashed on `calc.sm_extra_outputs` for the method to stamp onto output frames. Built-in `read_vtst_dimer` returns `eigenmode` (from `NEWMODECAR`, POSCAR→atoms order via `calc.resort`) and `curvature` (last `DIMCAR` row). **SinglePoint** stamps `calc.sm_extra_outputs` onto the output frame's `.info` after the run; other methods leave it unused (they compute their own).

Both share the selection grammar (built-in / `module:func` / `file.py:func`) plus a space-separated **list**, resolved by `load_extra_input_writer` / `load_extra_output_parser` (fail-fast validated in `load_method`). Together they enable a **VASP-internal VTST dimer driven by SaddleMill as a launcher**: `method = SinglePoint`, `Calculator = Vasp`, `[ourVasp] input_generator = omat24_static`, `extra_input_files = modecar`, `extra_outputs = vtst_dimer`, plus VTST INCAR tags in `[Vasp]` (`ichain`, `iopt`, `ibrion=3`, `potim=0`, `nsw`, `ediffg`) — ASE writes even those non-standard tags via its type-based fallback. The SinglePoint output frame then carries the converged saddle geometry + E/F **and** `eigenmode`/`curvature`. Use plain `Vasp` (not `VaspInteractive`, which forces `ibrion=-1`). **Requires a manual ASE patch** until fixed upstream: ASE ≤ 3.28 files `drotmax` under `float_keys` and writes `DROTMAX` as a float (`10.000000`), which VTST silently rejects — move `'drotmax'` to `int_keys` in `ase/calculators/vasp/create_input.py` (guarded by `tests/test_ase_vasp_incar.py`; the `ase` pin will be bumped once fixed).

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

### `scripts/` - Downstream dataset utilities
Standalone post-processing on a finished campaign's output dirs (not part of `python -m saddlemill`). The Dimer → DoubleMinimization → initbasinopt → SaddleFlow-dataset pipeline:
- `build_lmdb_parallel.py` - parallel multi-shard ASE-LMDB build from DoubleMinimization `(R,S,P)` triplet `.traj`s (converged-only filter, per-frame `ms_id`, `data['info']`+`traj_path`); the "no-EF" triplet dataset (SaddleFlow dataset 1, positional `MaterialsSaddlesDataset` format).
- `consolidate_dimer_basin_entry.py` - extracts each converged Dimer attempt's basin-entry frame (first step of the last negative-curvature stretch, from `dimer_opt_*.log`) into `dimer_basin_entry_*.traj`; the input to an `initbasinopt` Minimization run.
- `compare_basin_entry_vs_doublemin.py` - joins each converged initbasinopt minimum to the two DoubleMinimization endpoints by the Dimer `(src_index, attempt_id)` (read from the `orig_info` level holding both - top-level `src_index` is reassigned per run), scoring periodic RMSD + max-atom-displacement (MIC, translation-removed) to each side → `comparison_rmsd.csv` + `matched_attempts.csv` + histogram. `_key()` is the canonical cross-run Dimer-identity join reused by the scripts below.
- `make_comparison_trajs.py` - for sampled matched/unmatched saddles, stitches a 4-section "story" `.traj` (DM saddle→reactant, DM saddle→product, dimer reversed saddle→basin-entry, initbasinopt basin-entry→min) and copies the verbatim `dimer_opt_*.log`; self-checks that section-3 end == section-4 start.
- `make_traj_datasets.py` - builds the SaddleFlow "trajectory→saddle" LMDB datasets: D3 (all both-converged saddles, 18 rows) and D2 (matched, +4 initbasinopt rows = 22). Each saddle = one `group_id` (`= src*16+att`, consistent across datasets); rows = saddle (flow target, energy + eigenmode) + reactant/product minima (E+F) + each DM/dimer/initbasin trajectory resampled to 6 RMSD-uniform nodes by interpolation (positions wrapped, E/F on the 3 core rows only); deterministic 90/5/5 split. Phases `locmap` / `build` (multi-node, per-zip-batched, resumable via `.done`) / `verify` / `d1split` (emit a D1 split manifest on the same `group_id` split). On-disk schema documented in `datasets/DATASETS_FOR_SADDLEFLOW.md` in the campaign dir.
- `measure_dataset23.py` - sizing/timing measurement for the `make_traj_datasets` build (samples saddles, runs the real 6-node interpolation, writes a throwaway LMDB to estimate bytes/frame and build time).
- `backfill_status.py`, `extract_neb_triplets.py` - status-CSV backfill / NEB-triplet extraction helpers.

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

[Vasp]                             # INCAR params — pure pass-through to ASE's Vasp calculator; no SaddleMill keys here
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

[ourVasp]                          # SaddleMill-side VASP orchestration (optional; only for Calculator = Vasp/VaspInteractive)
input_generator =                  # INCAR recipe: built-in (omat24_static | omat24_relax | cheap_omat | oc20 | cheap_oc20 | oc22 | cheap_oc22), 'module:func', or '/path/file.py:func'
extra_input_files =                # files written INTO the VASP dir: built-in 'modecar' (VTST mode from atoms.info['eigenmode']), 'module:func', '/path/file.py:func', or a space-separated list
extra_outputs =                    # parsers read FROM the VASP dir -> merged into output .info: built-in 'vtst_dimer' (eigenmode from NEWMODECAR + curvature from DIMCAR), 'module:func', '/path/file.py:func', or a list. Currently consumed by SinglePoint — stamped onto output `.info` (traj) / merged into `data['info']` (lmdb).
# Per-tag precedence: [Vasp] key > input_generator output > ASE/VASP default.
# input_generator yields electronic/accuracy settings only; ionic-driver tags
# (IBRION/NSW/POTIM/EDIFFG) are stripped (ASE drives geometry). Needs
# fairchem-data-omat (omat24_*) or fairchem-data-oc (oc20/cheap_oc20) — see
# README; oc22/cheap_oc22 are self-contained (exact Meta settings hardcoded).

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
reaction_types = vacancy     # Bulk includes vacancy, hop_reuse, hop_insert, kickout_reuse,
                             # displace_kickout_reuse, kickout_insert, ring, all_atoms,
                             # random_bubble, and initial_guess.
                             # OC includes all_movable, adsorbate*, diffusion, rotation,
                             # surface, custom, random_bubble, and initial_guess.
num_attempts_per_type = 1    # One count for all types, or one position-aligned count per type.
gaussian_normal_attempts = 0 # 0=off; N gives N directed attempts then one Gaussian replacement.
reuse_exhaustion = stop      # Only supported policy; exhausted ranked slots become explicit None attempts.
bulk_reuse_offset =          # Effective default 0; independent of SM_SEED_OFFSET.
ring_sizes = 3 4
ring_mode = arc              # arc | chord | legacy
ring_frac = 0.2              # Fraction of a full 2*pi/n exchange rotation.
ring_neighbor_mult = 1.20
ring_neighbor_cutoff =       # Optional flat cutoff override in A.
ring_max_cycles = 20000
concentrate_prob = 0.0       # Off by default; fraction of Gaussian slots made concentrated.
concentrate_power = 1.5
concentrate_std = 0.2
concentrate_max_disp = 0.0
concentrate_envelope = 0.0
supercell = True             # Min 7 A expansion; skipped for OC structures with adsorbates.
delocalization_threshold = 0.8
extension_check_fmax = 0.4
extension_check_curvature = -0.2
engine = ase                 # ase (default) | kappa | sella
rotation_optimizer = ase     # ase (default) | lbfgs; leave ase for sella
translation_optimizer = ase  # ase (default) | lbfgs; leave ase for sella
kappa_beta = 5.0             # only used when engine = kappa
kappa_recover_fmax = 0.3     # only used when engine = kappa

[DimerControl]
max_num_rot = 1
f_rot_max = 1.0
f_rot_min = 0.1
initial_eigenmode_method = gauss
maximum_translation = 0.1
dimer_separation = 0.01
extrapolate_forces = True
displacement_radius = 3.5 # need to add displacement_radius or number_of_displacement_atoms to avoid errors in generating biased reaction guesses
number_of_displacement_atoms = 10

[Kappa]
# Acts identically to [DimerControl] but configures the Phase B constrained rotation 
# for KappaMinModeAtoms (only used when engine = kappa).
max_num_rot = 8
f_rot_min = 0.01
f_rot_max = 0.50
dimer_separation = 0.01

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

**VASP/VaspInteractive across all methods.** Both `Vasp` and `VaspInteractive` are accepted by **every** method with **no code-level distinction** — `is_vasp` checks always test `("Vasp", "VaspInteractive")` together, and nothing blocks any (method, calculator) pair. Do **not** add such a block. The Dimer example defaults to plain `Vasp` purely as an untested-conservative choice (ASE `MinModeAtoms` does rotated/displaced evals that may interact awkwardly with VaspInteractive's persistent process); VaspInteractive + Dimer is allowed and is to be validated during real testing. Every method instantiates per-job-unit VASP working directories via the shared `tools.resolve_vasp_calc()` helper:

| Method | Working dir | Notes |
|--------|------------|-------|
| NEB | `VASP_{job}_{image_idx}/` | Separate `vasp_command_endpoints`/`_intermediates` commands. |
| Dimer | `VASP_{job}_{attempt_id}/` | One dir per attempt — clean isolation across reaction-type displacements. |
| Minimization | `VASP_{job}/` | Single dir per job. |
| DoubleMinimization | `VASP_{job}_-1/`, `VASP_{job}_0/`, `VASP_{job}_1/` | Side ints match `info['side']`. TS dir reused for `pre_dimer_refine`; side dirs reused for desorption-check single-point + relaxation (WAVECAR warm-start). |
| SinglePoint | `VASP_{job}/` | On success: heavies dropped + dir archived to `SinglePoint_debug_zips/` (`ERROR_`-prefixed on error), like the other methods; `zip = False` just deletes. FAIRChem SP keeps nothing. |

`VaspInteractive.finalize()` is always called before swapping in `SinglePointCalculator`, and in error handlers, so the VASP subprocess never outlives the Python job. WAVECAR/CHG/CHGCAR are removed from each dir on success; remaining VASP files are zipped into `{method}_debug_zips/` (SP included, when Calculator is VASP/VaspInteractive). Resume cleanup picks up stale `VASP_*` dirs through `tools.clean_up_files()` for every method when Calculator is VASP/VaspInteractive.

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
- **SinglePoint**: `converged`, `not_converged`, `error: scf_not_converged` — FAIRChem (and any true `nsw=0` single-point) is always `converged`; for a VASP relaxation (e.g. VTST dimer) the status is the real verdict (`calc.converged` ∨ max per-atom `|F| <= |EDIFFG|`), and a final-step SCF that didn't reach EDIFF errors out.

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
SinglePoint_debug_zips/structure_rank_*.zip        # only when Calculator is VASP/VaspInteractive
```
SP creates the output directory matching the active `input_format` (plus `_status_csvs/`), and — only when Calculator is VASP/VaspInteractive — a `SinglePoint_debug_zips/` (FAIRChem SP has none). SP output frames carry `src_index`, `status`, `task_name`, `orig_info` (for .traj input via `load_and_sanitize`), plus `converged` (0/1) for VASP. `status` is `converged` for FAIRChem / any true single-point, but for a VASP relaxation (e.g. VTST dimer) it is the real verdict — `converged` / `not_converged` / `error: scf_not_converged`. For LMDB input, each output row mirrors the source row's `key_value_pairs` and `data` (`info` + `traj_path`) verbatim, with `row.energy` / `row.forces` newly populated via `SinglePointCalculator` on the atoms object — i.e. byte-equivalent to a fresh `build_lmdb_parallel.py` run on E/F-bearing trajectories (the only additions to `data['info']` are opted-in `[ourVasp] extra_outputs` such as VTST `eigenmode`/`curvature`, plus a `converged` flag for VASP; with none — e.g. all-FAIRChem SP — the row stays byte-equivalent). The `src_index`/`status`/`task_name` bookkeeping is traj-output-only — `ase.db` serializes only the `data=` blob, never `atoms.info`. With `frames_per_job = 3`, the three frames of each triplet are evaluated in one batched FAIRChem forward pass and written contiguously in input order to the same rank shard.

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
