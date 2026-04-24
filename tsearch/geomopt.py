import os
import sys
import traceback
import zipfile
from ase.io import Trajectory
from ase.filters import FrechetCellFilter
from ase.calculators.singlepoint import SinglePointCalculator
from tsearch.tools import check_reaction, check_adsorbate_reaction, backup_flux_logs, get_task_name
from tsearch.dimeropt import _refine_eigenmode


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
    task_name = get_task_name(config_dict)

    def log_status(status_msg):
        with open(status_file, 'a') as f:
            f.write(f'{i},{rank},"{status_msg}"\n')

    # --- MAIN LOOP ---
    with Trajectory(my_output_file, 'a') as writer:

        temp_opt_log = f'optimization_{i}.log'
        temp_traj = f'optimization_{i}.traj'
        temp_files = [temp_opt_log, temp_traj]

        try:
            optimizable = FrechetCellFilter(atoms) if config_dict['our'+method_name]['relax_cell'] else atoms
            converged = relax_structure(config_dict, optimizable, temp_opt_log, temp_traj, Optimizer)
            atoms.calc = SinglePointCalculator(atoms, energy=atoms.get_potential_energy(), forces=atoms.get_forces())

            if converged:
                status = "converged"
                atoms.info['converged'] = 1
            else:
                status = "not_converged"
                atoms.info['converged'] = 0
            atoms.info['status'] = status
            atoms.info['task_name'] = task_name
            atoms.info['src_index'] = i
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

            log_status(status)
            if consecutive_errors is not None:
                consecutive_errors[0] = 0

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
    task_name = get_task_name(config_dict)

    def log_status(side_id, parent_source_idx, status_msg):
        with open(status_file, 'a') as f:
            f.write(f'{i},{rank},{side_id},{parent_source_idx},"{status_msg}"\n')

    # 3. Initialize list to track temp files from BOTH optimizations
    temp_files = []
    continuation_data = kwargs.get('continuation_data')  # {side: Atoms} or None
    entries_to_run = kwargs.get('entries_to_run')        # set of side_ids (-1, 1) or None
    with Trajectory(my_output_file, 'a') as writer:
        orig = atoms.info.get('orig_info', {})
        parent_source_idx = orig['src_index']
        try:
            if 'eigenmode' not in orig:
                raise Exception("Input structure missing 'eigenmode' in info.")
            if 'src_index' not in orig:
                raise Exception("Input structure missing 'src_index' in info.")

            # Identify IDs
            refined_eigenmode = orig['eigenmode']

            # --- OPTIONAL: Refine eigenmode via dimer rotation ---
            curvature = orig.get('curvature')
            if config_dict['ourDoubleMinimization']['pre_dimer_refine']:
                dimer_log = f'dimer_refine_{i}.log'
                temp_files.append(dimer_log)
                refined_eigenmode, curvature = _refine_eigenmode(
                    atoms, calc, refined_eigenmode,
                    dimer_control_kwargs=config_dict.get("DimerControl", {}),
                    control_logfile=dimer_log,
                )

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

            # Desorption: only optimize toward bound state, skip desorption direction
            skip_side = None
            if orig.get('reaction_type') == 'desorption':
                energies = {}
                for test_side in [-1, 1]:
                    test_atoms = ts_atoms.copy()
                    test_atoms.calc = calc
                    test_atoms.positions += -test_side * displacement * refined_eigenmode
                    energies[test_side] = test_atoms.get_potential_energy()
                skip_side = max(energies, key=energies.get)

            for side in [-1, 1]:
                should_run = entries_to_run is None or side in entries_to_run

                if side == skip_side:
                    # Desorption direction: use TS as placeholder, no optimization
                    min_atoms = ts_atoms.copy()
                    min_atoms.calc = SinglePointCalculator(min_atoms,
                        energy=ts_atoms.get_potential_energy(),
                        forces=ts_atoms.get_forces())
                    conv = True
                elif should_run:
                    if continuation_data and side in continuation_data and continue_from_result:
                        min_atoms = continuation_data[side].copy()
                        min_atoms.calc = calc
                    else:
                        min_atoms = ts_atoms.copy()
                        min_atoms.calc = calc
                        min_atoms.positions += -side * displacement * refined_eigenmode

                    log_f = f'optimization_{i}_{side}.log'
                    traj_f = f'optimization_{i}_{side}.traj'
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
            res = check_reaction(min1, min2, neighbor_fudge=neighbor_fudge)
            ads_res = check_adsorbate_reaction(min1, min2, neighbor_fudge=neighbor_fudge,
                                               target_tag=2)
            reaction_info = {
                'is_reaction': res['occurred'],
                'broken_bonds': sorted(res['broken_bonds']),
                'formed_bonds': sorted(res['formed_bonds']),
                'n_formed_bonds': res['n_formed'],
                'n_broken_bonds': res['n_broken'],
                'is_ads_reaction': ads_res['occurred'],
                'ads_broken_bonds': sorted(ads_res['broken_bonds']),
                'ads_formed_bonds': sorted(ads_res['formed_bonds']),
                'n_ads_formed_bonds': ads_res['n_formed'],
                'n_ads_broken_bonds': ads_res['n_broken'],
            }
            for obj in [min1, min2, ts_atoms]:
                obj.info.update(reaction_info)
            ts_atoms.info['side'] = 0
            ts_atoms.info['src_index'] = i
            ts_atoms.info['eigenmode'] = refined_eigenmode
            if curvature is not None:
                ts_atoms.info['curvature'] = curvature

            # --- WRITE FRAMES (Min1, TS, Min2) ---
            side_statuses = {}
            for side in [-1, 1]:
                if side == skip_side:
                    side_statuses[side] = "converged_desorption_skipped"
                else:
                    side_statuses[side] = "converged" if mins[side][1] else "not_converged"
            min1.info['status'] = side_statuses[-1]
            min2.info['status'] = side_statuses[1]
            ts_atoms.info['status'] = "converged"
            min1.info['task_name'] = task_name
            min2.info['task_name'] = task_name
            ts_atoms.info['task_name'] = task_name
            min1.wrap()
            ts_atoms.wrap()
            min2.wrap()
            writer.write(min1)
            writer.write(ts_atoms)
            writer.write(min2)

            # --- CLEANUP (Success Case) ---
            existing_files = [f for f in temp_files if os.path.exists(f)]
            if existing_files and config_dict['Main']['zip']:
                with zipfile.ZipFile(zip_name, 'a', zipfile.ZIP_DEFLATED) as zf:
                    for f_name in existing_files:
                        zf.write(f_name, arcname=f_name)
                for f_name in existing_files:
                    os.remove(f_name)

            for side in mins:
                if entries_to_run is None or side in entries_to_run:
                    log_status(side, parent_source_idx, side_statuses[side])

            if consecutive_errors is not None:
                consecutive_errors[0] = 0

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

