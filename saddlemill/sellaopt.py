"""Sella (P-RFO) saddle search - a drop-in sibling of the Dimer method.

`method = Sella` runs the *same* attempt-generation machinery as `method = Dimer`
(same reaction types, same continuation/resume contract, same output `.info` keys
and status-CSV schema) and swaps only the saddle optimizer: Sella's partitioned
rational-function optimization in place of ASE's `MinModeTranslate`.

The two differ in how they find the unstable direction. The dimer converges a
*rotation* to the lowest mode and translates along it, paying force calls per
rotation. Sella instead carries a quasi-Newton Hessian model (TS-BFGS updates
plus iterative Rayleigh-Ritz diagonalization); it never builds a Hessian, but it
does re-diagonalize every `nsteps_per_diag` steps.

Force-call accounting (important). This module records `dyn.pes.neval`, the true
energy/force evaluation count - NOT `dyn.get_number_of_steps()`. The step count
undercounts real cost by roughly 2x, because each periodic re-diagonalization
spends extra gradient evaluations that are not optimizer steps. Recording
`neval` is what makes the `n_force_calls` CSV column directly comparable to the
Dimer's `forcecalls` counter, so the two methods can be benchmarked head to head.

Eigenmode seeding. `[ourDimer]`/`[ourSella]` runs on an existing saddle carry a
stored eigenmode; the Dimer seeds it via `eigenmodes=[...]`. The Sella analogue
is the initial Rayleigh-Ritz guess `pes.v0`, which lives in Sella's *free-DOF*
subspace rather than raw 3N - so the stored mode is projected through
`pes.get_Ufree()` before being installed. Without that projection the seed is
silently the wrong shape on any structure carrying `FixAtoms` (all OC20/OC22
slabs).
"""
import os
import sys
import traceback
import random
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from ase.neighborlist import natural_cutoffs, neighbor_list
from ase.io import Trajectory
from ase.mep import DimerControl, MinModeAtoms
from ase.calculators.singlepoint import SinglePointCalculator
from saddlemill.dimertools.structure_edit import get_attempts
from saddlemill.tools import (backup_flux_logs, get_task_name, resolve_vasp_calc,
                              remove_vasp_heavies, finalize_if_vasp_interactive,
                              archive_and_clear_temp_files, hessian_index)


class StopRun(Exception):
    pass


def _apply_displacement(atoms, displacement_dict, dimer_control_kwargs=None,
                        control_logfile=None):
    """Apply a `get_attempts()` displacement dict to *atoms*, in place.

    The displacement dicts produced by `dimertools.structure_edit` are ASE
    `MinModeAtoms.displace()` kwargs, so the displacement is applied through a
    throwaway `MinModeAtoms` wrapper. `displace()` is pure geometry - it reads
    and writes positions and costs no force calls - so this reuses the dimer
    toolkit's displacement machinery verbatim and guarantees that Dimer and
    Sella see *identical* attempt geometries for the same seed. That equality is
    what makes a head-to-head benchmark of the two optimizers meaningful.
    """
    if not displacement_dict:
        return
    d_control = DimerControl(logfile=control_logfile,
                             **(dimer_control_kwargs or {}))
    wrapper = MinModeAtoms(atoms, d_control)
    wrapper.displace(**displacement_dict)
    # MinModeAtoms mutates the wrapped Atoms in place; read back defensively so
    # this stays correct even if that ever stops being true.
    atoms.set_positions(wrapper.get_positions())


def _seed_eigenmode(dyn, eigenmode):
    """Install a stored eigenmode as Sella's initial Rayleigh-Ritz guess.

    `pes.v0` must live in the free-DOF subspace spanned by `pes.get_Ufree()`
    (shape `(3N, nfree)`), not in raw 3N Cartesian space - on a slab with
    `FixAtoms` those differ. Returns True if the seed was installed.
    """
    if eigenmode is None:
        return False
    mode = np.asarray(eigenmode, dtype=float)
    if mode.size == 0:
        return False
    try:
        Ufree = dyn.pes.get_Ufree()            # (3N, nfree), no force calls
        if mode.size != Ufree.shape[0]:
            return False
        v0 = mode.ravel() @ Ufree              # -> (nfree,)
        norm = np.linalg.norm(v0)
        if norm < 1e-12:
            # The stored mode is entirely inside the constrained subspace
            # (e.g. it only moves atoms that are now fixed) - useless as a seed.
            return False
        dyn.pes.v0 = v0 / norm
        return True
    except Exception:
        return False


def _mode_and_curvature(dyn, natoms):
    """Current lowest eigenvector (N,3) and eigenvalue from Sella's Hessian model.

    `pes.H.evecs` is stored in full 3N Cartesian space (rows for constrained
    atoms come back exactly zero), so column 0 reshapes straight to (N, 3).
    Returns (None, None) before the first diagonalization - which is the normal
    state when a structure was already converged and Sella took zero steps.
    """
    try:
        H = dyn.pes.H
        if H is None or H.evals is None or H.evecs is None:
            return None, None
        evals = np.asarray(H.evals)
        evecs = np.asarray(H.evecs)
        if evals.size == 0 or evecs.shape[0] != 3 * natoms:
            return None, None
        return evecs[:, 0].reshape(natoms, 3), float(evals[0])
    except Exception:
        return None, None


def _setup_sella(atoms, calc, eigenmode=None, displacement_dict=None,
                 sella_kwargs=None, dimer_control_kwargs=None,
                 control_logfile=None, logfile=None, trajectory=None):
    """Create a Sella optimizer for a saddle search. Mirrors `_setup_dimer()`.

    Does not run the optimization. The caller attaches callbacks and calls
    `dyn.run()`. Returns `(dyn, seeded)`; *seeded* records whether the stored
    eigenmode was installed as the initial mode guess.
    """
    from sella import Sella

    _apply_displacement(atoms, displacement_dict,
                        dimer_control_kwargs=dimer_control_kwargs,
                        control_logfile=control_logfile)
    atoms.calc = calc

    kw = dict(sella_kwargs or {})
    kw.setdefault("order", 1)        # first-order saddle
    # Cartesian, not internal coordinates: these are periodic solids and slabs,
    # and Sella's internal-coordinate machinery targets molecular systems.
    kw.setdefault("internal", False)
    dyn = Sella(atoms, logfile=logfile, trajectory=trajectory, **kw)

    seeded = _seed_eigenmode(dyn, eigenmode)
    return dyn, seeded


def sellaopt(i, config_dict, atoms_orig, calc, consecutive_errors=None,
             executorlib_worker_id=None, **kwargs):

    rank = executorlib_worker_id

    run_offset = int(os.environ.get("SM_RUN_OFFSET", "0"))
    seed = i + run_offset * 1000

    random.seed(seed)
    np.random.seed(seed)

    method_name = config_dict["Main"]["method"]
    status_file = f"{method_name}_status_csvs/status_rank_{rank}.csv"
    my_output_file = f"{method_name}_trajes/collected_ts_rank_{rank}.traj"
    zip_name = f"{method_name}_debug_zips/structure_rank_{rank}_data.zip"
    task_name = get_task_name(config_dict)
    is_vasp = config_dict["Main"]["Calculator"] in ("Vasp", "VaspInteractive")

    our = config_dict["ourSella"]
    sella_kwargs = config_dict.get("Sella") or {}
    check_index = our.get("check_index", False)

    max_consecutive_errors = config_dict["Main"]["max_consecutive_errors"]
    if consecutive_errors is not None and consecutive_errors[0] >= max_consecutive_errors > 0:
        print(f"Rank {rank}: {consecutive_errors[0]} consecutive structures errored. Killing worker for restart.", flush=True)
        backup_flux_logs(rank)
        sys.exit(1)

    def log_status(attempt, slctd_indx, status_msg, n_force_calls=0):
        with open(status_file, 'a') as f:
            f.write(f'{i},{rank},{attempt},{slctd_indx},{n_force_calls},"{status_msg}"\n')

    # --- MAIN LOOP ---
    any_attempt_succeeded = False
    all_attempts_none = False

    continuation_data = kwargs.get('continuation_data')  # {attempt_id: Atoms} or None
    entries_to_run = kwargs.get('entries_to_run')        # set of attempt_ids or None

    with Trajectory(my_output_file, 'a') as writer:

        attempt = "init"
        slctd_indx = -1
        temp_files = []

        generated = get_attempts(atoms_orig, config_dict)
        all_attempts_none = all(a is None for a in generated[0])
        if all_attempts_none:
            print(f"Rank {rank} WARNING on structure {i}: "
                  "All attempts failed to generate.", flush=True)

        attempts_iter = enumerate(zip(*generated))

        for attempt, (atoms, displacement_dict, slctd_indx) in attempts_iter:

            if entries_to_run is not None and attempt not in entries_to_run:
                continue

            if atoms is None:
                log_status(attempt, -1, "error: failed to generate attempt")
                continue

            # Use continuation structure if available for this attempt. Unlike
            # the Dimer, Sella needs no symmetry-breaking kick to restart - it
            # rebuilds its own curvature model - so the geometry is reused as is.
            if continuation_data and attempt in continuation_data:
                atoms = continuation_data[attempt]
                displacement_dict = None

            temp_log = f'sella_control_{i}_{attempt}_{slctd_indx}.log'
            temp_opt_log = f'sella_opt_{i}_{attempt}_{slctd_indx}.log'
            temp_traj = f'sella_{i}_{attempt}_{slctd_indx}.traj'
            temp_files = [temp_log, temp_opt_log, temp_traj]
            attempt_vasp_dir = f"VASP_{i}_{attempt}" if is_vasp else None
            if attempt_vasp_dir is not None:
                temp_files.append(attempt_vasp_dir)
            attempt_calc = None

            try:
                # Handle constraints:
                if atoms.constraints:
                    free_indices = [atom.index for atom in atoms if atom.index not in atoms.constraints[0].get_indices()]
                else:
                    free_indices = [atom.index for atom in atoms]

                # Use existing eigenmode if available (top level from
                # get_attempts/initial_guess, or orig_info from continuation),
                # otherwise let Sella derive its own from the gradient.
                eigenmode = atoms.info.get('eigenmode')
                if eigenmode is None:
                    eigenmode = atoms.info.get('orig_info', {}).get('eigenmode')
                if eigenmode is not None:
                    eigenmode = np.array(eigenmode)

                attempt_calc = resolve_vasp_calc(config_dict, calc, i, attempt, "ourSella", atoms=atoms)
                dyn, seeded = _setup_sella(
                    atoms, attempt_calc, eigenmode=eigenmode,
                    displacement_dict=displacement_dict,
                    sella_kwargs=sella_kwargs,
                    dimer_control_kwargs=config_dict.get("DimerControl"),
                    control_logfile=temp_log,
                    logfile=temp_opt_log, trajectory=temp_traj,
                )

                # PR Check - skip early steps to let Sella's Hessian model pick
                # up the unstable mode (the initial displacement can look
                # delocalized, especially for diffusion/rotation types).
                delocalization_start_step = max(1, int(0.1 * config_dict["Main"]["steps"]))

                def check_delocalization():
                    if dyn.nsteps < delocalization_start_step:
                        return
                    mode, _ = _mode_and_curvature(dyn, len(atoms))
                    if mode is None:
                        return
                    v2 = (mode**2).sum(axis=1)
                    v2 = v2[free_indices]
                    sum_v2 = np.sum(v2)
                    if sum_v2 < 1e-12: return
                    pr = (sum_v2**2) / (len(v2) * np.sum(v2**2))
                    if pr > our["delocalization_threshold"]:
                        raise StopRun(f"Eigenmode Delocalized (PR={pr:.3f})")

                def check_desorption():
                    check_atoms = atoms
                    cutoffs = natural_cutoffs(check_atoms, mult=2.0)
                    nl_i, nl_j = neighbor_list('ij', check_atoms, cutoffs)
                    adjacency = csr_matrix((np.ones(len(nl_i)), (nl_i, nl_j)), shape=(len(check_atoms), len(check_atoms)))
                    n_components, labels = connected_components(adjacency, connection='weak')
                    if n_components > 1:
                        raise StopRun(f"Adsorbate desorbed")

                dyn.attach(check_delocalization, interval=5)
                dyn.attach(check_desorption, interval=5)

                stop_reason = None
                stopped_early = False
                converged = False
                try:
                    converged = dyn.run(fmax=config_dict["Main"]["fmax"], steps=config_dict["Main"]["steps"])
                except StopRun as e:
                    stopped_early = True
                    stop_reason = str(e)
                    converged = False

                eigenmode_out, curvature = _mode_and_curvature(dyn, len(atoms))

                if converged:
                    status = "converged"
                elif not converged and not stopped_early:
                    # Extension check - same gate as the Dimer: close in force
                    # and still on a genuinely downhill mode.
                    fmax_check = np.sqrt((atoms.get_forces()**2).sum(axis=1).max()) < our['extension_check_fmax']
                    curvature_check = (curvature is not None
                                       and curvature < our['extension_check_curvature'])
                    if fmax_check and curvature_check:
                        try:
                            converged = dyn.run(fmax=config_dict["Main"]["fmax"], steps=150)
                        except StopRun as e:
                            stopped_early = True
                            stop_reason = str(e)
                            converged = False

                        if converged:
                            status = "converged_after_extension"
                        else:
                            status = "not_converged_after_extension"
                        eigenmode_out, curvature = _mode_and_curvature(dyn, len(atoms))
                    else:
                        status = "not_converged"
                else:
                    status = "not_converged_StopRun"

                # Metadata. n_force_calls is the TRUE evaluation count
                # (pes.neval), not the optimizer step count - see module docstring.
                n_force_calls = int(getattr(dyn.pes, "neval", 0))
                n_steps = int(dyn.get_number_of_steps())
                # Sella may converge at step 0 without ever diagonalizing; keep
                # the seeded mode rather than emitting nothing.
                if eigenmode_out is None:
                    eigenmode_out = eigenmode
                energy = atoms.get_potential_energy()
                forces = atoms.get_forces()

                # Optional independent index check: the quasi-Newton Hessian is
                # only a model, so verifying "is this really a first-order
                # saddle" needs real curvature - finite-difference Lanczos.
                nneg, eigs = None, None
                # Deliberately NOT gated on convergence: a non-converged search
                # still leaves a geometry behind, and whether that geometry is
                # better or worse than the label it started from is the whole
                # question behind the status filter. Gating here would make it
                # unmeasurable.
                if check_index:
                    eigs, nneg = hessian_index(
                        atoms,
                        nev=our.get("index_nev", 4),
                        eps=our.get("index_eps", 2e-3),
                        tol=our.get("index_tol", 1e-2),
                    )
                    n_force_calls = int(getattr(dyn.pes, "neval", 0))

                finalize_if_vasp_interactive(config_dict, attempt_calc)
                if attempt_vasp_dir is not None:
                    remove_vasp_heavies(attempt_vasp_dir)

                atoms.info['eigenmode'] = eigenmode_out
                if curvature is not None:
                    atoms.info['curvature'] = float(curvature)
                atoms.info['n_force_calls'] = int(n_force_calls)
                atoms.info['n_steps'] = n_steps
                atoms.info['eigenmode_seeded'] = 1 if seeded else 0
                atoms.info['converged'] = 1 if converged else 0
                atoms.info['src_index'] = i
                atoms.info['attempt_id'] = attempt
                atoms.info['stoprun'] = 1 if stopped_early else 0
                atoms.info['selected_index'] = slctd_indx
                if nneg is not None:
                    atoms.info['nneg'] = int(nneg)
                    atoms.info['eigenvalues'] = list(eigs) if eigs is not None else None
                orig = atoms.info.get('orig_info', {})
                atoms.info['reaction_type'] = atoms.info.get('reaction_type', orig.get('reaction_type', 'unknown'))
                if stop_reason and "desorbed" in stop_reason:
                    status = "converged_to_desorption"
                    atoms.info['converged'] = 1
                    atoms.info['reaction_type'] = 'desorption'
                atoms.info['status'] = status
                atoms.info['task_name'] = task_name
                atoms.wrap()
                atoms.calc = SinglePointCalculator(atoms, energy=energy, forces=forces)

                writer.write(atoms)

                archive_and_clear_temp_files(temp_files, zip_name, prefix="",
                                   enabled=config_dict['Main']['zip'])

                log_status(attempt, slctd_indx, status, n_force_calls)
                any_attempt_succeeded = True

            except Exception as e:
                print(f"Rank {rank} FAILED on structure {i}, attempt {attempt}: {e}", flush=True)
                print(f"\nTraceback details:\n{traceback.format_exc()}", flush=True)
                if attempt_calc is not None:
                    finalize_if_vasp_interactive(config_dict, attempt_calc)
                archive_and_clear_temp_files(temp_files, zip_name, prefix="ERROR_",
                                   enabled=config_dict['Main']['zip'])
                log_status(attempt, slctd_indx, f"error: {str(e)}")

    # Track consecutive structure-level errors for worker health
    if consecutive_errors is not None:
        if any_attempt_succeeded:
            consecutive_errors[0] = 0
        elif all_attempts_none:
            pass  # Data issue (e.g., no adsorbate atoms), not a worker error
        else:
            consecutive_errors[0] += 1
