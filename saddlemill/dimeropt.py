import os
import sys
import traceback
import random
import zipfile
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from ase.neighborlist import natural_cutoffs, neighbor_list
from ase.io import Trajectory
from ase.mep import DimerControl, MinModeAtoms, MinModeTranslate
from ase.calculators.singlepoint import SinglePointCalculator
from saddlemill.dimertools.structure_edit import get_attempts
from saddlemill.tools import (backup_flux_logs, get_task_name, resolve_vasp_calc,
                              remove_vasp_heavies, finalize_if_vasp_interactive,
                              archive_and_clear_temp_files)


class StopRun(Exception):
    pass


def _setup_dimer(atoms, calc, eigenmode=None, displacement_dict=None,
                 dimer_control_kwargs=None, control_logfile=None,
                 logfile=None, trajectory=None,
                 engine="ase", kappa_kwargs=None, kappa_control_kwargs=None,
                 rotation_optimizer="ase", translation_optimizer="ase",
                 rotation_lbfgs_options=None, translation_lbfgs_options=None):
    """Create the selected dimer engine, rotation solver, and translator.

    The default ase/ase/ase path intentionally uses ASE's stock
    DimerControl -> MinModeAtoms -> MinModeTranslate classes. L-BFGS is
    selected only when explicitly requested.
    """
    atoms.calc = calc
    eig_kw = {"eigenmodes": [np.array(eigenmode)]} if eigenmode is not None else {}
    rotation_optimizer = str(rotation_optimizer).lower()
    translation_optimizer = str(translation_optimizer).lower()

    if engine == "kappa":
        from saddlemill.dimertools.kappa_dimer import (
            IsolatedDimerControl, KappaMinModeAtoms,
        )
        d_control = IsolatedDimerControl(
            logfile=control_logfile, **(dimer_control_kwargs or {})
        )
        kw = dict(kappa_kwargs or {})
        kw.update({
            "rotation_optimizer": rotation_optimizer,
            "rotation_lbfgs_options": rotation_lbfgs_options or {},
        })
        if kappa_control_kwargs:
            kw["kappa_control"] = IsolatedDimerControl(
                logfile=control_logfile, **kappa_control_kwargs
            )
        d_atoms = KappaMinModeAtoms(atoms, control=d_control, **eig_kw, **kw)
    elif engine == "ase":
        d_control = DimerControl(
            logfile=control_logfile, **(dimer_control_kwargs or {})
        )
        if rotation_optimizer == "ase":
            # Exact legacy/default class path.
            d_atoms = MinModeAtoms(atoms, d_control, **eig_kw)
        elif rotation_optimizer == "lbfgs":
            from saddlemill.dimertools.lbfgs_dimer import (
                ConfigurableRotationMinModeAtoms,
            )
            d_atoms = ConfigurableRotationMinModeAtoms(
                atoms,
                control=d_control,
                rotation_optimizer="lbfgs",
                rotation_lbfgs_options=rotation_lbfgs_options or {},
                **eig_kw,
            )
        else:
            raise ValueError(
                "[ourDimer] rotation_optimizer must be ase or lbfgs; "
                f"got {rotation_optimizer!r}."
            )
    else:
        raise ValueError(
            f"Unknown [ourDimer] engine={engine!r}; expected 'ase' or 'kappa'."
        )

    if displacement_dict:
        d_atoms.displace(**displacement_dict)
    else:
        d_atoms.displace(
            displacement_vector=np.random.randn(len(atoms), 3) * 1e-10,
            method="vector",
        )

    if translation_optimizer == "ase":
        # Exact legacy/default translator.
        dim_rlx = MinModeTranslate(
            d_atoms, trajectory=trajectory, logfile=logfile
        )
    elif translation_optimizer == "lbfgs":
        from saddlemill.dimertools.lbfgs_dimer import LBFGSMinModeTranslate
        dim_rlx = LBFGSMinModeTranslate(
            d_atoms,
            trajectory=trajectory,
            logfile=logfile,
            lbfgs_options=translation_lbfgs_options or {},
        )
    else:
        raise ValueError(
            "[ourDimer] translation_optimizer must be ase or lbfgs; "
            f"got {translation_optimizer!r}."
        )
    return d_atoms, dim_rlx

def _refine_eigenmode(atoms, calc, eigenmode, dimer_control_kwargs=None,
                      control_logfile=None):
    """Refine eigenmode via dimer rotation only (no translation).

    Works on a copy of *atoms* — the original is never modified.
    Returns (refined_eigenmode, curvature).
    """
    refine_atoms = atoms.copy()
    refine_atoms.calc = calc
    d_control = DimerControl(logfile=control_logfile,
                             **(dimer_control_kwargs or {}))
    d_atoms = MinModeAtoms(refine_atoms, d_control,
                           eigenmodes=[np.array(eigenmode)])
    d_atoms.displace(displacement_vector=np.random.randn(len(refine_atoms), 3) * 1e-10,
                     method='vector')
    # get_forces() triggers eigenmode rotation (up to max_num_rot iterations).
    # No translation — only the eigenmode direction and curvature are updated.
    d_atoms.get_forces()
    return d_atoms.get_eigenmode(), float(d_atoms.get_curvature())


def dimeropt(i, config_dict, atoms_orig, calc, consecutive_errors=None, executorlib_worker_id=None, **kwargs):

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
    saddle_engine = str(config_dict["ourDimer"].get("engine", "ase")).lower()
    sella_options = None
    if saddle_engine == "sella":
        from saddlemill.sella_engine import sella_options_from_config
        sella_options = sella_options_from_config(config_dict)

    lbfgs_cfg = config_dict.get("ourDimerLBFGS", {}) or {}
    rotation_lbfgs_options = {
        "memory": int(lbfgs_cfg.get("rotation_memory", 10)),
        "initial_hessian": float(lbfgs_cfg.get("rotation_initial_hessian", 1.0)),
        "dynamic_h0": bool(lbfgs_cfg.get("rotation_dynamic_h0", False)),
        "curvature_epsilon": float(lbfgs_cfg.get("curvature_epsilon", 1.0e-12)),
    }
    translation_lbfgs_options = {
        "memory": int(lbfgs_cfg.get("translation_memory", 10)),
        "initial_hessian": float(lbfgs_cfg.get("translation_initial_hessian", 70.0)),
        "dynamic_h0": bool(lbfgs_cfg.get("translation_dynamic_h0", False)),
        "curvature_epsilon": float(lbfgs_cfg.get("curvature_epsilon", 1.0e-12)),
        "damping": float(lbfgs_cfg.get("translation_damping", 1.0)),
        "reset_on_regime_change": bool(
            lbfgs_cfg.get("reset_translation_on_regime_change", True)
        ),
    }

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

            # Use continuation structure if available for this attempt
            if continuation_data and attempt in continuation_data:
                atoms = continuation_data[attempt]
                displacement_dict = {"displacement_vector": np.random.randn(len(atoms), 3) * 1e-10, "method": "vector"}

            temp_log = f'dimer_control_{i}_{attempt}_{slctd_indx}.log'
            if saddle_engine == "sella":
                temp_opt_log = f'dimer_sella_opt_{i}_{attempt}_{slctd_indx}.log'
                temp_traj = f'dimer_sella_{i}_{attempt}_{slctd_indx}.traj'
            else:
                temp_opt_log = f'dimer_opt_{i}_{attempt}_{slctd_indx}.log'
                temp_traj = f'dimer_{i}_{attempt}_{slctd_indx}.traj'
            temp_files = [temp_log, temp_opt_log, temp_traj]
            attempt_vasp_dir = f"VASP_{i}_{attempt}" if is_vasp else None
            if attempt_vasp_dir is not None:
                temp_files.append(attempt_vasp_dir)
            attempt_calc = None
            d_atoms = None
            dim_rlx = None

            try:
                # Handle constraints:
                if atoms.constraints:
                    free_indices = [atom.index for atom in atoms if atom.index not in atoms.constraints[0].get_indices()]
                else:
                    free_indices = [atom.index for atom in atoms]

                # Use existing eigenmode if available (top level from
                # get_attempts/initial_guess, or orig_info from continuation),
                # otherwise let ASE derive one from the displacement.
                eigenmode = atoms.info.get('eigenmode')
                if eigenmode is None:
                    eigenmode = atoms.info.get('orig_info', {}).get('eigenmode')
                if eigenmode is not None:
                    eigenmode = np.array(eigenmode)

                attempt_calc = resolve_vasp_calc(config_dict, calc, i, attempt, "ourDimer", atoms=atoms)
                if saddle_engine == "sella":
                    from saddlemill.sella_engine import setup_sella
                    atoms, dim_rlx = setup_sella(
                        atoms, attempt_calc, eigenmode=eigenmode,
                        displacement_dict=displacement_dict,
                        dimer_control_kwargs=config_dict["DimerControl"],
                        logfile=temp_opt_log, trajectory=temp_traj,
                        sella_options=sella_options,
                    )
                else:
                    d_atoms, dim_rlx = _setup_dimer(
                        atoms, attempt_calc, eigenmode=eigenmode,
                        displacement_dict=displacement_dict,
                        dimer_control_kwargs=config_dict["DimerControl"],
                        control_logfile=temp_log,
                        logfile=temp_opt_log, trajectory=temp_traj,
                        engine=saddle_engine,
                        kappa_kwargs={
                            "beta": config_dict["ourDimer"]["kappa_beta"],
                            "recover_fmax": config_dict["ourDimer"]["kappa_recover_fmax"],
                        },
                        kappa_control_kwargs=(config_dict.get("Kappa") or None),
                        rotation_optimizer=config_dict["ourDimer"].get(
                            "rotation_optimizer", "ase"
                        ),
                        translation_optimizer=config_dict["ourDimer"].get(
                            "translation_optimizer", "ase"
                        ),
                        rotation_lbfgs_options=rotation_lbfgs_options,
                        translation_lbfgs_options=translation_lbfgs_options,
                    )

                sella_cfg = config_dict.get("ourSella", {}) or {}
                check_interval = (
                    int(sella_cfg.get("check_interval", 5))
                    if saddle_engine == "sella" else 5
                )

                # PR Check — skip early steps to let the dimer rotate
                # the eigenmode (initial displacement can look delocalized,
                # especially for diffusion/rotation types).
                delocalization_start_step = max(1, int(0.1 * config_dict["Main"]["steps"]))

                def check_delocalization():
                    if dim_rlx.nsteps < delocalization_start_step:
                        return
                    if saddle_engine == "sella":
                        if not bool(sella_cfg.get("check_delocalization", False)):
                            return
                        from saddlemill.sella_engine import extract_lowest_mode
                        mode, _, _ = extract_lowest_mode(dim_rlx)
                    else:
                        mode = d_atoms.get_eigenmode()
                    v2 = (mode**2).sum(axis=1)
                    v2 = v2[free_indices]
                    sum_v2 = np.sum(v2)
                    if sum_v2 < 1e-12: return
                    pr = (sum_v2**2) / (len(v2) * np.sum(v2**2))
                    if pr > config_dict["ourDimer"]["delocalization_threshold"]:
                        raise StopRun(f"Eigenmode Delocalized (PR={pr:.3f})")

                def check_desorption():
                    if saddle_engine == "sella" and not bool(
                        sella_cfg.get("check_desorption", True)
                    ):
                        return
                    check_atoms = atoms if saddle_engine == "sella" else d_atoms.atoms
                    cutoffs = natural_cutoffs(check_atoms, mult=2.0)
                    i, j = neighbor_list('ij', check_atoms, cutoffs)
                    adjacency = csr_matrix((np.ones(len(i)), (i, j)), shape=(len(check_atoms), len(check_atoms)))
                    n_components, labels = connected_components(adjacency, connection='weak')
                    if n_components > 1:
                        raise StopRun(f"Adsorbate desorbed")

                if saddle_engine == "sella":
                    if bool(sella_cfg.get("check_delocalization", False)):
                        dim_rlx.attach(check_delocalization, interval=check_interval)
                    if bool(sella_cfg.get("check_desorption", True)):
                        dim_rlx.attach(check_desorption, interval=check_interval)
                else:
                    dim_rlx.attach(check_delocalization, interval=5)
                    dim_rlx.attach(check_desorption, interval=5)

                stop_reason = None
                stopped_early = False
                converged = False
                try:
                    converged = dim_rlx.run(fmax=config_dict["Main"]["fmax"], steps=config_dict["Main"]["steps"])
                except StopRun as e:
                    stopped_early = True
                    stop_reason = str(e)
                    converged = False

                sella_eigenvalues = None
                sella_negative_modes = None
                sella_stationary_converged = None
                if saddle_engine == "sella":
                    from saddlemill.sella_engine import (
                        classify_sella_convergence, extract_lowest_mode,
                        sella_force_calls,
                    )
                    try:
                        eigenmode, curvature, sella_eigenvalues = extract_lowest_mode(dim_rlx)
                    except Exception:
                        if not stopped_early:
                            raise
                        if eigenmode is None:
                            eigenmode = np.zeros((len(atoms), 3), dtype=float)
                        curvature = float("nan")
                        sella_eigenvalues = np.array([], dtype=float)
                    sella_stationary_converged = bool(converged)
                    converged, status, sella_negative_modes = classify_sella_convergence(
                        sella_stationary_converged, sella_eigenvalues,
                        negative_eigenvalue_tolerance=float(
                            sella_cfg.get("negative_eigenvalue_tolerance", 1.0e-6)
                        ),
                        require_first_order_model=bool(
                            sella_cfg.get("require_first_order_model", True)
                        ),
                    )
                    if stopped_early:
                        converged = False
                        status = "not_converged_StopRun"
                    n_force_calls = sella_force_calls(dim_rlx)
                else:
                    if converged:
                        status = "converged"
                    elif not converged and not stopped_early:
                        # Extension check
                        fmax_check = np.sqrt((d_atoms.get_forces()**2).sum(axis=1).max()) < config_dict['ourDimer']['extension_check_fmax']
                        curvature_check = d_atoms.get_curvature() < config_dict['ourDimer']['extension_check_curvature']
                        if fmax_check and curvature_check:
                            try:
                                converged = dim_rlx.run(fmax=config_dict["Main"]["fmax"], steps=150)
                            except StopRun as e:
                                stopped_early = True
                                stop_reason = str(e)
                                converged = False
                            if converged:
                                status = "converged_after_extension"
                            else:
                                status = "not_converged_after_extension"
                        else:
                            status = "not_converged"
                    else:
                        status = "not_converged_StopRun"
                    eigenmode = d_atoms.get_eigenmode()
                    curvature = d_atoms.get_curvature()
                    n_force_calls = d_atoms.control.get_counter('forcecalls')
                energy = atoms.get_potential_energy()
                forces = atoms.get_forces()
                finalize_if_vasp_interactive(config_dict, attempt_calc)
                if attempt_vasp_dir is not None:
                    remove_vasp_heavies(attempt_vasp_dir)

                atoms.info['eigenmode'] = eigenmode
                atoms.info['curvature'] = float(curvature)
                atoms.info['n_force_calls'] = int(n_force_calls)
                atoms.info['converged'] = 1 if converged else 0
                atoms.info['src_index'] = i
                atoms.info['attempt_id'] = attempt
                atoms.info['stoprun'] = 1 if stopped_early else 0
                atoms.info['selected_index'] = slctd_indx
                orig = atoms.info.get('orig_info', {})
                atoms.info['reaction_type'] = atoms.info.get('reaction_type', orig.get('reaction_type', 'unknown'))
                if saddle_engine == "sella":
                    atoms.info['saddle_engine'] = 'sella'
                    atoms.info['sella_version'] = getattr(dim_rlx, 'sm_sella_version', 'unknown')
                    atoms.info['sella_used_input_mode'] = int(
                        bool(getattr(dim_rlx, 'sm_used_input_mode', False))
                    )
                    atoms.info['sella_stationary_converged'] = int(
                        bool(sella_stationary_converged)
                    )
                    atoms.info['sella_model_negative_modes'] = int(
                        sella_negative_modes or 0
                    )
                    atoms.info['sella_order_check'] = 'approximate_model_hessian'
                    atoms.info['sella_order'] = 1
                if stop_reason and "desorbed" in stop_reason:
                    status = "converged_to_desorption"
                    atoms.info['converged'] = 1
                    atoms.info['reaction_type'] = 'desorption'
                atoms.info['status'] = status
                atoms.info['task_name'] = task_name
                atoms.wrap()
                atoms.calc = SinglePointCalculator(atoms, energy=energy, forces=forces)

                writer.write(atoms)
                if saddle_engine == "sella":
                    dim_rlx.close()

                # Clean up temp files (the zip block below walks directories too,
                # so the per-attempt VASP dir is captured automatically).
                archive_and_clear_temp_files(temp_files, zip_name, prefix="",
                                   enabled=config_dict['Main']['zip'])

                log_status(attempt, slctd_indx, status, n_force_calls)
                any_attempt_succeeded = True

            except Exception as e:
                print(f"Rank {rank} FAILED on structure {i}, attempt {attempt}: {e}", flush=True)
                print(f"\nTraceback details:\n{traceback.format_exc()}", flush=True)
                if attempt_calc is not None:
                    finalize_if_vasp_interactive(config_dict, attempt_calc)
                if saddle_engine == "sella" and dim_rlx is not None:
                    try:
                        dim_rlx.close()
                    except Exception:
                        pass
                archive_and_clear_temp_files(temp_files, zip_name, prefix="ERROR_",
                                   enabled=config_dict['Main']['zip'])
                try:
                    if saddle_engine == "sella" and dim_rlx is not None:
                        from saddlemill.sella_engine import sella_force_calls
                        error_force_calls = sella_force_calls(dim_rlx)
                    else:
                        error_force_calls = (
                            d_atoms.control.get_counter('forcecalls')
                            if d_atoms is not None else 0
                        )
                except Exception:
                    error_force_calls = 0
                log_status(attempt, slctd_indx, f"error: {str(e)}", error_force_calls)

    # Track consecutive structure-level errors for worker health
    if consecutive_errors is not None:
        if any_attempt_succeeded:
            consecutive_errors[0] = 0
        elif all_attempts_none:
            pass  # Data issue (e.g., no adsorbate atoms), not a worker error
        else:
            consecutive_errors[0] += 1
