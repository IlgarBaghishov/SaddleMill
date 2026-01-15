import os
import io
import time
import random
import sys
import zipfile
import numpy as np
import pandas as pd
from ase.io import Trajectory, read
from ase.build import make_supercell
from ase.mep import DimerControl, MinModeAtoms, MinModeTranslate
from ase.calculators.singlepoint import SinglePointCalculator
from ase.neighborlist import neighbor_list
from tsearch.tools import parse_inputfile, load_calculator


config_dict = parse_inputfile("config.ini")
calc = load_calculator(config_dict)
rank = "DetermineRankHereFromExecutorlibOrFlux"

csv_path = config_dict["ourDimer"]["csv_path"]
df = pd.read_csv(csv_path, index_col=0)


class StopRun(Exception):
    pass


def turn_into_supercell(atoms):
    n_atoms = len(atoms)
    M = [1, 1, 1]
    if n_atoms == 1: M = [3, 3, 3]
    if n_atoms <= 4: M = [2, 2, 2]
    elif 5 <= n_atoms <= 8:
        lengths = atoms.cell.lengths()
        M[np.argsort(lengths)[0]] = 2
        M[np.argsort(lengths)[1]] = 2
    elif 9 <= n_atoms <= 16:
        M[np.argmin(atoms.cell.lengths())] = 2
    if M != [1, 1, 1]:
        atoms = make_supercell(atoms, np.diag(M))
    return atoms


def dimeropt(i, config_dict):

    random.seed(i)
    np.random.seed(i)

    # restart_file = "restart_todo.csv"
    status_file = f"status_csvs/status_rank_{rank}.csv"
    my_output_file = f"ts_trajes/collected_ts_rank_{rank}.traj"
    zip_name = f"debug_zips/structure_rank_{rank}_data.zip"

    def log_status(idx, attempt, rm_idx, status_msg):
        with open(status_file, 'a') as f:
            f.write(f"{idx},{attempt},{rm_idx},{status_msg}\n")

    # --- MAIN LOOP ---
    with Trajectory(my_output_file, 'a') as writer:

        attempt = "init"
        rm_idx = -1
        temp_files = []

        try:
            # Look up structure from main DF using the index
            atoms_orig = read(io.StringIO(df.at[i,'cif']), format='cif')
            atoms_orig = turn_into_supercell(atoms_orig)

            # Fresh random sampling for this restart
            remove_indices = random.sample(range(len(atoms_orig)), config_dict["ourDimer"]["num_attempts"])

            for attempt, rm_idx in enumerate(remove_indices):

                temp_log = f'dimer_control_r{rank}_{i}_{attempt}.log'
                temp_opt_log = f'dimer_opt_r{rank}_{i}_{attempt}.log'
                temp_traj = f'dimer_r{rank}_{i}_{attempt}.traj'
                temp_files = [temp_log, temp_opt_log, temp_traj]

                # Neighbor Logic
                i_idx, j_idx = neighbor_list('ij', atoms_orig, 3.5)
                neighbor_indices = j_idx[i_idx == rm_idx]

                if len(neighbor_indices) == 0:
                    neighbor_indices = [x for x in range(len(atoms_orig)) if x != rm_idx]

                chosen_neighbor = random.choice(neighbor_indices)

                atoms = atoms_orig.copy()
                del atoms[rm_idx]

                new_center_idx = chosen_neighbor if chosen_neighbor < rm_idx else chosen_neighbor - 1

                atoms.calc = calc

                d_control = DimerControl(
                    displacement_center = int(new_center_idx),
                    # displacement_center = atoms_orig[rm_idx].position.tolist(),
                    logfile = temp_log,
                    **config_dict["ourDimer"],
                )

                d_atoms = MinModeAtoms(atoms, d_control)
                d_atoms.displace()

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
                    converged = dim_rlx.run(fmax=config_dict["Main"]["fmax"], steps=config_dict["Main"]["fmax"]-150)
                except StopRun:
                    stopped_early = True
                    converged = False

                if converged:
                    log_status(i, attempt, rm_idx, "converged")
                elif not converged and not stopped_early:
                    # Extension check
                    if np.sqrt((d_atoms.get_forces()**2).sum(axis=1).max()) < 0.4 and d_atoms.get_curvature() < -0.2:
                        try:
                            converged = dim_rlx.run(fmax=config_dict["Main"]["fmax"], steps=150)
                        except StopRun:
                            stopped_early = True
                            converged = False

                        if converged:
                            log_status(i, attempt, rm_idx, "converged_after_extension")
                        else:
                            log_status(i, attempt, rm_idx, "not_converged_after_extension")
                    else:
                        log_status(i, attempt, rm_idx, "not_converged")
                else:
                    log_status(i, attempt, rm_idx, "not_converged_delocalized")

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
                atoms.info['removed_atom_index'] = rm_idx

                writer.write(atoms)

                # Clean up temp files
                existing_files = [f for f in temp_files if os.path.exists(f)]
                if existing_files:
                    if not converged:
                        with zipfile.ZipFile(zip_name, 'a', zipfile.ZIP_DEFLATED) as zf:
                            for f_name in existing_files:
                                zf.write(f_name, arcname=f"attempt_{attempt}_{f_name}")
                    for f_name in existing_files:
                        os.remove(f_name)

        except Exception as e:
            print(f"Rank {rank} FAILED on structure {i}: {e}")
            existing_files = [f for f in temp_files if os.path.exists(f)]
            if existing_files:
                with zipfile.ZipFile(zip_name, 'a', zipfile.ZIP_DEFLATED) as zf:
                    for f_name in existing_files:
                        zf.write(f_name, arcname=f"attempt_{attempt}_ERROR_{f_name}")
                for f_name in existing_files:
                    os.remove(f_name)
            log_status(i, attempt, rm_idx, "error")

    print(f"Rank {rank}: Finished.")
