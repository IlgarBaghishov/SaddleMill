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
from tsearch.tools import backup_flux_logs


class StopRun(Exception):
    pass



def dimeropt(i, config_dict, atoms_orig, calc, consecutive_errors=None, executorlib_worker_id=None, **kwargs):

    rank = executorlib_worker_id
    random.seed(i)
    np.random.seed(i)

    method_name = config_dict["Main"]["method"]
    status_file = f"{method_name}_status_csvs/status_rank_{rank}.csv"
    my_output_file = f"{method_name}_trajes/collected_ts_rank_{rank}.traj"
    zip_name = f"{method_name}_debug_zips/structure_rank_{rank}_data.zip"

    max_consecutive_errors = config_dict["Main"]["max_consecutive_errors"]
    if consecutive_errors is not None and consecutive_errors[0] >= max_consecutive_errors > 0:
        print(f"Rank {rank}: {consecutive_errors[0]} consecutive structures errored. Killing worker for restart.", flush=True)
        backup_flux_logs(rank)
        sys.exit(1)

    def log_status(attempt, slctd_indx, status_msg):
        with open(status_file, 'a') as f:
            f.write(f"{i},{rank},{attempt},{slctd_indx},{status_msg}\n")

    # --- MAIN LOOP ---
    any_attempt_succeeded = False

    continuation_data = kwargs.get('continuation_data')  # {attempt_id: Atoms} or None
    entries_to_run = kwargs.get('entries_to_run')        # set of attempt_ids or None

    with Trajectory(my_output_file, 'a') as writer:

        attempt = "init"
        slctd_indx = -1
        temp_files = []

        attempts_iter = enumerate(zip(*get_attempts(atoms_orig, config_dict)))

        for attempt, (atoms, displacement_dict, slctd_indx) in attempts_iter:

            if entries_to_run is not None and attempt not in entries_to_run:
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
                atoms.calc = calc

                # Handle constraints:
                if atoms.constraints:
                    free_indices = [atom.index for atom in atoms if atom.index not in atoms.constraints[0].get_indices()]
                else:
                    free_indices = [atom.index for atom in atoms]

                d_control = DimerControl(
                    logfile = temp_log,
                    **config_dict["DimerControl"],
                )

                # Use existing eigenmode if available (top level from
                # get_attempts/initial_guess, or orig_info from continuation),
                # otherwise let ASE derive one from the displacement.
                eigenmode_kwarg = {}
                orig = atoms.info.get('orig_info', {})
                if 'eigenmode' in atoms.info:
                    eigenmode_kwarg['eigenmodes'] = [np.array(atoms.info['eigenmode'])]
                elif 'eigenmode' in orig:
                    eigenmode_kwarg['eigenmodes'] = [np.array(orig['eigenmode'])]

                d_atoms = MinModeAtoms(atoms, d_control, **eigenmode_kwarg)
                d_atoms.displace(**displacement_dict)

                dim_rlx = MinModeTranslate(
                    d_atoms,
                    trajectory = temp_traj,
                    logfile = temp_opt_log
                )

                # PR Check
                def check_delocalization():
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
                energy = atoms.get_potential_energy()
                forces = atoms.get_forces()

                atoms.info['eigenmode'] = eigenmode
                atoms.calc = SinglePointCalculator(atoms, energy=energy, forces=forces)
                atoms.info['converged'] = 1 if converged else 0
                atoms.info['src_index'] = i
                atoms.info['attempt_id'] = attempt
                atoms.info['stoprun'] = 1 if stopped_early else 0
                atoms.info['selected_index'] = slctd_indx
                orig = atoms.info.get('orig_info', {})
                atoms.info['reaction_type'] = atoms.info.get('reaction_type', orig.get('reaction_type', 'unknown'))
                if stop_reason and "desorbed" in stop_reason:
                    atoms.info['reaction_type'] = 'desorption'
                atoms.wrap()

                writer.write(atoms)

                log_status(attempt, slctd_indx, status)
                any_attempt_succeeded = True

                # Clean up temp files
                existing_files = [f for f in temp_files if os.path.exists(f)]
                if existing_files and config_dict['Main']['zip']:
                    with zipfile.ZipFile(zip_name, 'a', zipfile.ZIP_DEFLATED) as zf:
                        for f_name in existing_files:
                            zf.write(f_name, arcname=f"{f_name}")
                    for f_name in existing_files:
                        os.remove(f_name)

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
        else:
            consecutive_errors[0] += 1
