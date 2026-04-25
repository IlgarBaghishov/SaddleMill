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
from tsearch.dimertools.structure_edit import get_attempts
from tsearch.tools import backup_flux_logs, get_task_name


class StopRun(Exception):
    pass


def _setup_dimer(atoms, calc, eigenmode=None, displacement_dict=None,
                 dimer_control_kwargs=None, control_logfile=None,
                 logfile=None, trajectory=None):
    """Create MinModeAtoms and MinModeTranslate optimizer for dimer method.

    Does not run the optimization. Caller attaches callbacks and calls
    dim_rlx.run().

    Returns (d_atoms, dim_rlx).
    """
    atoms.calc = calc
    d_control = DimerControl(logfile=control_logfile, **(dimer_control_kwargs or {}))
    eig_kw = {'eigenmodes': [np.array(eigenmode)]} if eigenmode is not None else {}
    d_atoms = MinModeAtoms(atoms, d_control, **eig_kw)
    if displacement_dict:
        d_atoms.displace(**displacement_dict)
    else:
        d_atoms.displace(displacement_vector=np.random.randn(len(atoms), 3) * 1e-10,
                         method='vector')
    dim_rlx = MinModeTranslate(d_atoms, trajectory=trajectory, logfile=logfile)
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
    random.seed(i)
    np.random.seed(i)

    method_name = config_dict["Main"]["method"]
    status_file = f"{method_name}_status_csvs/status_rank_{rank}.csv"
    my_output_file = f"{method_name}_trajes/collected_ts_rank_{rank}.traj"
    zip_name = f"{method_name}_debug_zips/structure_rank_{rank}_data.zip"
    task_name = get_task_name(config_dict)

    max_consecutive_errors = config_dict["Main"]["max_consecutive_errors"]
    if consecutive_errors is not None and consecutive_errors[0] >= max_consecutive_errors > 0:
        print(f"Rank {rank}: {consecutive_errors[0]} consecutive structures errored. Killing worker for restart.", flush=True)
        backup_flux_logs(rank)
        sys.exit(1)

    def log_status(attempt, slctd_indx, status_msg):
        with open(status_file, 'a') as f:
            f.write(f'{i},{rank},{attempt},{slctd_indx},"{status_msg}"\n')

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
            temp_opt_log = f'dimer_opt_{i}_{attempt}_{slctd_indx}.log'
            temp_traj = f'dimer_{i}_{attempt}_{slctd_indx}.traj'
            temp_files = [temp_log, temp_opt_log, temp_traj]

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

                d_atoms, dim_rlx = _setup_dimer(
                    atoms, calc, eigenmode=eigenmode,
                    displacement_dict=displacement_dict,
                    dimer_control_kwargs=config_dict["DimerControl"],
                    control_logfile=temp_log,
                    logfile=temp_opt_log, trajectory=temp_traj,
                )

                # PR Check — skip early steps to let the dimer rotate
                # the eigenmode (initial displacement can look delocalized,
                # especially for diffusion/rotation types).
                delocalization_start_step = max(1, int(0.1 * config_dict["Main"]["steps"]))

                def check_delocalization():
                    if dim_rlx.nsteps < delocalization_start_step:
                        return
                    mode = d_atoms.get_eigenmode()
                    v2 = (mode**2).sum(axis=1)
                    v2 = v2[free_indices]
                    sum_v2 = np.sum(v2)
                    if sum_v2 < 1e-12: return
                    pr = (sum_v2**2) / (len(v2) * np.sum(v2**2))
                    if pr > config_dict["ourDimer"]["delocalization_threshold"]:
                        raise StopRun(f"Eigenmode Delocalized (PR={pr:.3f})")

                def check_desorption():
                    check_atoms = d_atoms.atoms
                    cutoffs = natural_cutoffs(check_atoms, mult=2.0)
                    i, j = neighbor_list('ij', check_atoms, cutoffs)
                    adjacency = csr_matrix((np.ones(len(i)), (i, j)), shape=(len(check_atoms), len(check_atoms)))
                    n_components, labels = connected_components(adjacency, connection='weak')
                    if n_components > 1:
                        raise StopRun(f"Adsorbate desorbed")

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

                # Metadata
                eigenmode = d_atoms.get_eigenmode()
                curvature = d_atoms.get_curvature()
                energy = atoms.get_potential_energy()
                forces = atoms.get_forces()

                atoms.info['eigenmode'] = eigenmode
                atoms.info['curvature'] = float(curvature)
                atoms.calc = SinglePointCalculator(atoms, energy=energy, forces=forces)
                # atoms.info['converged'] = 1 if converged else 0
                atoms.info['src_index'] = i
                atoms.info['attempt_id'] = attempt
                atoms.info['stoprun'] = 1 if stopped_early else 0
                atoms.info['selected_index'] = slctd_indx
                orig = atoms.info.get('orig_info', {})
                atoms.info['reaction_type'] = atoms.info.get('reaction_type', orig.get('reaction_type', 'unknown'))
                if stop_reason and "desorbed" in stop_reason:
                    status = "converged_to_desorption"
                    atoms.info['converged'] = 1
                    atoms.info['reaction_type'] = 'desorption'
                atoms.info['status'] = status
                atoms.info['task_name'] = task_name
                atoms.wrap()

                writer.write(atoms)

                # Clean up temp files
                existing_files = [f for f in temp_files if os.path.exists(f)]
                if existing_files and config_dict['Main']['zip']:
                    with zipfile.ZipFile(zip_name, 'a', zipfile.ZIP_DEFLATED) as zf:
                        for f_name in existing_files:
                            zf.write(f_name, arcname=f"{f_name}")
                    for f_name in existing_files:
                        os.remove(f_name)

                log_status(attempt, slctd_indx, status)
                any_attempt_succeeded = True

            except Exception as e:
                print(f"Rank {rank} FAILED on structure {i}, attempt {attempt}: {e}", flush=True)
                print(f"\nTraceback details:\n{traceback.format_exc()}", flush=True)
                existing_files = [f for f in temp_files if os.path.exists(f)]
                if existing_files and config_dict['Main']['zip']:
                    with zipfile.ZipFile(zip_name, 'a', zipfile.ZIP_DEFLATED) as zf:
                        for f_name in existing_files:
                            zf.write(f_name, arcname=f"ERROR_{f_name}")
                    for f_name in existing_files:
                        os.remove(f_name)
                log_status(attempt, slctd_indx, f"error: {str(e)}")

    # Track consecutive structure-level errors for worker health
    if consecutive_errors is not None:
        if any_attempt_succeeded:
            consecutive_errors[0] = 0
        elif all_attempts_none:
            pass  # Data issue (e.g., no adsorbate atoms), not a worker error
        else:
            consecutive_errors[0] += 1
