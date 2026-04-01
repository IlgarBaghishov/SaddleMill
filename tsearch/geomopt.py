import os
import sys
import traceback
import zipfile
from ase.io import Trajectory
from ase.filters import FrechetCellFilter
from ase.calculators.singlepoint import SinglePointCalculator
from tsearch.tools import check_reaction, check_adsorbate_reaction, backup_flux_logs


def relax_structure(config_dict, optimizable, logfile, trajfile, Optimizer):
    opt = Optimizer(optimizable, logfile=logfile, trajectory=trajfile,
                    **config_dict[config_dict["Main"]["Optimizer"]])
    converged = opt.run(fmax=config_dict["Main"]["fmax"], steps=config_dict["Main"]["steps"])
    return converged


def geomopt(i, config_dict, atoms, calc, Optimizer, consecutive_errors=None, executorlib_worker_id=None, **kwargs):

    rank = executorlib_worker_id

    max_consecutive_errors = config_dict["Main"]["max_consecutive_errors"]
    if consecutive_errors is not None and consecutive_errors[0] >= max_consecutive_errors > 0:
        print(f"Rank {rank}: {consecutive_errors[0]} consecutive structures errored. Killing worker for restart.", flush=True)
        backup_flux_logs(rank)
        sys.exit(1)

    continuation_data = kwargs.get('continuation_data')
    if continuation_data is not None and config_dict["Main"]["continue_from_result"]:
        atoms = continuation_data

    atoms.calc = calc

    method_name = config_dict["Main"]["method"]
    status_file = f"{method_name}_status_csvs/status_rank_{rank}.csv"
    my_output_file = f"{method_name}_trajes/collected_opt_rank_{rank}.traj"
    zip_name = f"{method_name}_debug_zips/structure_rank_{rank}_data.zip"

    def log_status(status_msg):
        with open(status_file, 'a') as f:
            f.write(f"{i},{rank},{status_msg}\n")

    # --- MAIN LOOP ---
    with Trajectory(my_output_file, 'a') as writer:

        temp_opt_log = f'optimization_r{rank}_{i}.log'
        temp_traj = f'optimization_r{rank}_{i}.traj'
        temp_files = [temp_opt_log, temp_traj]

        try:
            optimizable = FrechetCellFilter(atoms) if config_dict['our'+method_name]['relax_cell'] else atoms
            converged = relax_structure(config_dict, optimizable, temp_opt_log, temp_traj, Optimizer)
            atoms.calc = SinglePointCalculator(atoms, energy=atoms.get_potential_energy(), forces=atoms.get_forces())

            if converged:
                log_status("converged")
                atoms.info['converged'] = 1
            else:
                log_status("not_converged")
                atoms.info['converged'] = 0
            atoms.info['src_index'] = i
            atoms.wrap()

            writer.write(atoms)

            if consecutive_errors is not None:
                consecutive_errors[0] = 0

            # Clean up temp files
            existing_files = [f for f in temp_files if os.path.exists(f)]
            if existing_files and config_dict['Main']['zip']:
                with zipfile.ZipFile(zip_name, 'a', zipfile.ZIP_DEFLATED) as zf:
                    for f_name in existing_files:
                        zf.write(f_name, arcname=f"{f_name}")
                for f_name in existing_files:
                    os.remove(f_name)

        except Exception as e:
            print(f"Rank {rank} FAILED on structure {i}: {e}", flush=True)
            print(f"\nTraceback details:\n{traceback.format_exc()}", flush=True)
            if consecutive_errors is not None:
                consecutive_errors[0] += 1
            existing_files = [f for f in temp_files if os.path.exists(f)]
            if existing_files and config_dict['Main']['zip']:
                with zipfile.ZipFile(zip_name, 'a', zipfile.ZIP_DEFLATED) as zf:
                    for f_name in existing_files:
                        zf.write(f_name, arcname=f"ERROR_{f_name}")
                for f_name in existing_files:
                    os.remove(f_name)
            log_status(f"error: {str(e)}")


def doublegeomopt(i, config_dict, atoms, calc, Optimizer, consecutive_errors=None, executorlib_worker_id=None, **kwargs):

    rank = executorlib_worker_id

    max_consecutive_errors = config_dict["Main"]["max_consecutive_errors"]
    if consecutive_errors is not None and consecutive_errors[0] >= max_consecutive_errors > 0:
        print(f"Rank {rank}: {consecutive_errors[0]} consecutive structures errored. Killing worker for restart.", flush=True)
        backup_flux_logs(rank)
        sys.exit(1)

    atoms.calc = calc

    method_name = config_dict["Main"]["method"]
    status_file = f"{method_name}_status_csvs/status_rank_{rank}.csv"
    my_output_file = f"{method_name}_trajes/collected_opt_rank_{rank}.traj"
    zip_name = f"{method_name}_debug_zips/structure_rank_{rank}_data.zip"

    def log_status(side_id, parent_source_idx, status_msg):
        with open(status_file, 'a') as f:
            f.write(f"{i},{rank},{side_id},{parent_source_idx},{status_msg}\n")

    # 3. Initialize list to track temp files from BOTH optimizations
    temp_files = []
    continuation_data = kwargs.get('continuation_data')  # {side: Atoms} or None
    entries_to_run = kwargs.get('entries_to_run')        # set of side_ids (-1, 1) or None
    with Trajectory(my_output_file, 'a') as writer:
        orig = atoms.info.get('orig_info', {})
        parent_source_idx = orig['src_index']
        try:
            if not orig['converged']:
                raise Exception("Input structure marked as unconverged.")

            if 'eigenmode' not in orig:
                 raise Exception("Input structure missing 'eigenmode' in info.")
            if 'src_index' not in orig:
                 raise Exception("Input structure missing 'src_index' in info.")

            # Identify IDs
            refined_eigenmode = orig['eigenmode']
            continue_from_result = config_dict["Main"]["continue_from_result"]

            # --- PREPARE TS (Middle Image) ---
            ts_atoms = atoms.copy()
            ts_atoms.info = atoms.info.copy()
            ts_atoms.calc = SinglePointCalculator(
                ts_atoms,
                energy=atoms.get_potential_energy(),
                forces=atoms.get_forces()
            )

            # --- MINIMIZE BOTH SIDES ---
            mins = {}  # side -> (atoms, converged)
            displacement = 0.25
            for side in [-1, 1]:
                should_run = entries_to_run is None or side in entries_to_run
                file_idx = 0 if side == -1 else 1

                if should_run:
                    if continuation_data and side in continuation_data and continue_from_result:
                        min_atoms = continuation_data[side].copy()
                        min_atoms.calc = calc
                    else:
                        min_atoms = ts_atoms.copy()
                        min_atoms.calc = calc
                        min_atoms.positions += -side * displacement * refined_eigenmode

                    log_f = f'optimization_r{rank}_{i}-{file_idx}.log'
                    traj_f = f'optimization_r{rank}_{i}-{file_idx}.traj'
                    temp_files.extend([log_f, traj_f])

                    optimizable = FrechetCellFilter(min_atoms) if config_dict['our'+method_name]['relax_cell'] else min_atoms
                    conv = relax_structure(config_dict, optimizable, log_f, traj_f, Optimizer)
                    min_atoms.calc = SinglePointCalculator(min_atoms, energy=min_atoms.get_potential_energy(), forces=min_atoms.get_forces())
                else:
                    if not (continuation_data and side in continuation_data):
                        raise ValueError(f"Missing continuation data for kept side={side}")
                    min_atoms = continuation_data[side].copy()
                    conv = bool(min_atoms.info.get('orig_info', {}).get('converged'))

                min_atoms.info['side'] = side
                min_atoms.info['parent_ts_index'] = parent_source_idx
                min_atoms.info['converged'] = conv
                min_atoms.info['src_index'] = i
                mins[side] = (min_atoms, conv)

            min1, conv1 = mins[-1]
            min2, conv2 = mins[1]

            # --- CHECK REACTION ---
            neighbor_fudge = 1.25
            res  = check_reaction(min1, min2, neighbor_fudge=neighbor_fudge)
            is_reaction = res["occurred"]
            broken_bonds = res["broken_bonds"]
            formed_bonds = res["formed_bonds"]
            n_broken = res["n_broken"]
            n_formed = res["n_formed"]
            res = check_adsorbate_reaction(min1, min2, neighbor_fudge=neighbor_fudge,
                                           target_tag=2)
            is_ads_reaction = res["occurred"]
            ads_broken_bonds = res["broken_bonds"]
            ads_formed_bonds = res["formed_bonds"]
            ads_n_broken = res["n_broken"]
            ads_n_formed = res["n_formed"]
            min1.info['is_reaction'] = is_reaction
            min1.info['n_formed_bonds'] = n_formed
            min1.info['n_broken_bonds'] = n_broken
            min1.info['is_ads_reaction'] = is_ads_reaction
            min1.info['n_ads_formed_bonds'] = ads_n_formed
            min1.info['n_ads_broken_bonds'] = ads_n_broken

            min2.info['is_reaction'] = is_reaction
            min2.info['n_formed_bonds'] = n_formed
            min2.info['n_broken_bonds'] = n_broken
            min2.info['is_ads_reaction'] = is_ads_reaction
            min2.info['n_ads_formed_bonds'] = ads_n_formed
            min2.info['n_ads_broken_bonds'] = ads_n_broken

            ts_atoms.info['is_reaction'] = is_reaction
            ts_atoms.info['n_formed_bonds'] = n_formed
            ts_atoms.info['n_broken_bonds'] = n_broken
            ts_atoms.info['is_ads_reaction'] = is_ads_reaction
            ts_atoms.info['n_ads_formed_bonds'] = ads_n_formed
            ts_atoms.info['n_ads_broken_bonds'] = ads_n_broken
            ts_atoms.info['side'] = 0
            ts_atoms.info['src_index'] = i

            # --- WRITE FRAMES (Min1, TS, Min2) ---
            min1.wrap()
            ts_atoms.wrap()
            min2.wrap()
            writer.write(min1)
            writer.write(ts_atoms)
            writer.write(min2)

            for side, (_, conv) in mins.items():
                if entries_to_run is None or side in entries_to_run:
                    log_status(side, parent_source_idx, "converged" if conv else "not_converged")

            if consecutive_errors is not None:
                consecutive_errors[0] = 0

            # --- CLEANUP (Success Case) ---
            existing_files = [f for f in temp_files if os.path.exists(f)]
            if existing_files and config_dict['Main']['zip']:
                with zipfile.ZipFile(zip_name, 'a', zipfile.ZIP_DEFLATED) as zf:
                    for f_name in existing_files:
                        zf.write(f_name, arcname=f_name)
                for f_name in existing_files:
                    os.remove(f_name)

        except Exception as e:
            # --- CLEANUP (Error Case) ---
            print(f"Rank {rank} FAILED on structure {i}: {e}", flush=True)
            print(f"\nTraceback details:\n{traceback.format_exc()}", flush=True)
            if consecutive_errors is not None:
                consecutive_errors[0] += 1
            for side in [-1, 1]:
                if entries_to_run is None or side in entries_to_run:
                    log_status(side, parent_source_idx, f"error: {str(e)}")

            existing_files = [f for f in temp_files if os.path.exists(f)]
            if existing_files and config_dict['Main']['zip']:
                with zipfile.ZipFile(zip_name, 'a', zipfile.ZIP_DEFLATED) as zf:
                    for f_name in existing_files:
                        zf.write(f_name, arcname=f"ERROR_{f_name}")
                for f_name in existing_files:
                    os.remove(f_name)

