import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import io
import random
import zipfile
import numpy as np
import pandas as pd
from ase.io import Trajectory
from ase.mep import DimerControl, MinModeAtoms, MinModeTranslate
from ase.calculators.singlepoint import SinglePointCalculator
from tsearch.config import load_config, load_calculator
from tsearch.dimertools.structure_edit import get_attempts


config_dict = load_config("config.ini")
calc = load_calculator(config_dict)


class StopRun(Exception):
    pass


def dimeropt(i, config_dict, atoms_orig, executorlib_worker_id=None):

    rank = executorlib_worker_id
    random.seed(i)
    np.random.seed(i)

    method_name = config_dict["Main"]["method"]
    status_file = f"{method_name}_status_csvs/status_rank_{rank}.csv"
    my_output_file = f"{method_name}_trajes/collected_ts_rank_{rank}.traj"
    zip_name = f"{method_name}_debug_zips/structure_rank_{rank}_data.zip"

    def log_status(attempt, slctd_indx, status_msg):
        with open(status_file, 'a') as f:
            f.write(f"{i},{rank},{attempt},{slctd_indx},{status_msg}\n")

    # --- MAIN LOOP ---
    with Trajectory(my_output_file, 'a') as writer:

        attempt = "init"
        slctd_indx = -1
        temp_files = []

        try:
            for attempt, (atoms, displacement_dict, slctd_indx) in enumerate(zip(*get_attempts(atoms_orig, config_dict))):

                temp_log = f'dimer_control_{i}_{attempt}_{slctd_indx}.log'
                temp_opt_log = f'dimer_opt_{i}_{attempt}_{slctd_indx}.log'
                temp_traj = f'dimer_{i}_{attempt}_{slctd_indx}.traj'
                temp_files = [temp_log, temp_opt_log, temp_traj]

                atoms.calc = calc

                d_control = DimerControl(
                    logfile = temp_log,
                    **config_dict["DimerControl"],
                )

                d_atoms = MinModeAtoms(atoms, d_control)
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
                    sum_v2 = np.sum(v2)
                    if sum_v2 < 1e-12: return
                    pr = (sum_v2**2) / (len(atoms) * np.sum(v2**2))
                    if pr > config_dict["ourDimer"]["delocalization_threshold"]:
                        raise StopRun(f"Eigenmode Delocalized (PR={pr:.3f})")

                dim_rlx.attach(check_delocalization, interval=5)

                stopped_early = False
                converged = False
                try:
                    converged = dim_rlx.run(fmax=config_dict["Main"]["fmax"], steps=config_dict["Main"]["steps"])
                except StopRun:
                    stopped_early = True
                    converged = False

                if converged:
                    log_status(attempt, slctd_indx, "converged")
                elif not converged and not stopped_early:
                    # Extension check
                    fmax_check = np.sqrt((d_atoms.get_forces()**2).sum(axis=1).max()) < config_dict['ourDimer']['extension_check_fmax']
                    curvature_check = d_atoms.get_curvature() < config_dict['ourDimer']['extension_check_curvature']
                    if fmax_check and curvature_check:
                        try:
                            converged = dim_rlx.run(fmax=config_dict["Main"]["fmax"], steps=150)
                        except StopRun:
                            stopped_early = True
                            converged = False

                        if converged:
                            log_status(attempt, slctd_indx, "converged_after_extension")
                        else:
                            log_status(attempt, slctd_indx, "not_converged_after_extension")
                    else:
                        log_status(attempt, slctd_indx, "not_converged")
                else:
                    log_status(attempt, slctd_indx, "not_converged_delocalized")

                # Metadata
                eigenmode = d_atoms.get_eigenmode()
                energy = atoms.get_potential_energy()
                forces = atoms.get_forces()

                atoms.info['eigenmode'] = eigenmode
                atoms.calc = SinglePointCalculator(atoms, energy=energy, forces=forces)
                atoms.info['converged'] = 1 if converged else 0
                atoms.info['src_index'] = i
                atoms.info['attempt_id'] = attempt
                atoms.info['delocalized'] = 1 if stopped_early else 0
                atoms.info['selected_index'] = slctd_indx

                writer.write(atoms)

                # Clean up temp files
                existing_files = [f for f in temp_files if os.path.exists(f)]
                if existing_files:
                    with zipfile.ZipFile(zip_name, 'a', zipfile.ZIP_DEFLATED) as zf:
                        for f_name in existing_files:
                            zf.write(f_name, arcname=f"{f_name}")
                    for f_name in existing_files:
                        os.remove(f_name)

        except Exception as e:
            print(f"Rank {rank} FAILED on structure {i}: {e}")
            existing_files = [f for f in temp_files if os.path.exists(f)]
            if existing_files:
                with zipfile.ZipFile(zip_name, 'a', zipfile.ZIP_DEFLATED) as zf:
                    for f_name in existing_files:
                        zf.write(f_name, arcname=f"ERROR_{f_name}")
                for f_name in existing_files:
                    os.remove(f_name)
            log_status(attempt, slctd_indx, "error")
