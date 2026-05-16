import os
import sys
import traceback
import zipfile
from ase.io import Trajectory
from ase.filters import FrechetCellFilter
from ase.calculators.singlepoint import SinglePointCalculator
from saddlemill.tools import (check_reaction, check_adsorbate_reaction, backup_flux_logs,
                              get_task_name, resolve_vasp_calc, remove_vasp_heavies,
                              finalize_if_vasp_interactive, archive_and_clear_temp_files)
from saddlemill.dimeropt import _refine_eigenmode


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

    method_name = config_dict["Main"]["method"]
    is_vasp = config_dict["Main"]["Calculator"] in ("Vasp", "VaspInteractive")
    vasp_calc = resolve_vasp_calc(config_dict, calc, i, None, "ourMinimization")
    atoms.calc = vasp_calc

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
        if is_vasp:
            temp_files.append(f"VASP_{i}")
        orig = atoms.info.get('orig_info', {})
        parent_source_idx = orig.get('src_index')

        try:
            optimizable = FrechetCellFilter(atoms) if config_dict['our'+method_name]['relax_cell'] else atoms
            converged = relax_structure(config_dict, optimizable, temp_opt_log, temp_traj, Optimizer)
            energy = atoms.get_potential_energy()
            forces = atoms.get_forces()
            finalize_if_vasp_interactive(config_dict, vasp_calc)
            if is_vasp:
                remove_vasp_heavies(f"VASP_{i}")

            if converged:
                status = "converged"
                atoms.info['converged'] = 1
            else:
                status = "not_converged"
                atoms.info['converged'] = 0
            atoms.info['status'] = status
            atoms.info['task_name'] = task_name
            atoms.info['parent_ts_index'] = parent_source_idx
            atoms.info['src_index'] = i
            atoms.wrap()
            atoms.calc = SinglePointCalculator(atoms, energy=energy, forces=forces)

            writer.write(atoms)

            archive_and_clear_temp_files(temp_files, zip_name, prefix="",
                                         enabled=config_dict['Main']['zip'])

            log_status(status)
            if consecutive_errors is not None:
                consecutive_errors[0] = 0

        except Exception as e:
            print(f"Rank {rank} FAILED on structure {i}: {e}", flush=True)
            print(f"\nTraceback details:\n{traceback.format_exc()}", flush=True)
            if consecutive_errors is not None:
                consecutive_errors[0] += 1
            finalize_if_vasp_interactive(config_dict, vasp_calc)
            archive_and_clear_temp_files(temp_files, zip_name, prefix="ERROR_",
                                         enabled=config_dict['Main']['zip'])
            log_status(f"error: {str(e)}")


def doublegeomopt(i, config_dict, atoms, calc, Optimizer, consecutive_errors=None, executorlib_worker_id=None, **kwargs):

    rank = executorlib_worker_id

    max_consecutive_errors = config_dict["Main"]["max_consecutive_errors"]
    if consecutive_errors is not None and consecutive_errors[0] >= max_consecutive_errors > 0:
        print(f"Rank {rank}: {consecutive_errors[0]} consecutive structures errored. Killing worker for restart.", flush=True)
        backup_flux_logs(rank)
        sys.exit(1)

    method_name = config_dict["Main"]["method"]
    is_vasp = config_dict["Main"]["Calculator"] in ("Vasp", "VaspInteractive")
    # TS-side calculator: used for TS E/F and optional pre_dimer_refine.
    ts_calc = resolve_vasp_calc(config_dict, calc, i, 0 if is_vasp else None, "ourDoubleMinimization")
    active_vasp_calcs = [ts_calc] if is_vasp else []
    atoms.calc = ts_calc

    status_file = f"{method_name}_status_csvs/status_rank_{rank}.csv"
    my_output_file = f"{method_name}_trajes/collected_opt_rank_{rank}.traj"
    zip_name = f"{method_name}_debug_zips/structure_rank_{rank}_data.zip"
    task_name = get_task_name(config_dict)

    def log_status(side_id, parent_source_idx, status_msg):
        with open(status_file, 'a') as f:
            f.write(f'{i},{rank},{side_id},{parent_source_idx},"{status_msg}"\n')

    # 3. Initialize list to track temp files from BOTH optimizations
    temp_files = []
    if is_vasp:
        temp_files.extend([f"VASP_{i}_{s}" for s in (-1, 0, 1)])
    continuation_data = kwargs.get('continuation_data')  # {side: Atoms} or None
    entries_to_run = kwargs.get('entries_to_run')        # set of side_ids (-1, 1) or None
    with Trajectory(my_output_file, 'a') as writer:
        orig = atoms.info.get('orig_info', {})
        parent_source_idx = orig.get('src_index')
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
                    atoms, ts_calc, refined_eigenmode,
                    dimer_control_kwargs=config_dict.get("DimerControl", {}),
                    control_logfile=dimer_log,
                )

            continue_from_result = config_dict["Main"]["continue_from_result"]

            # --- PREPARE TS (Middle Image) ---
            ts_atoms = atoms.copy()
            ts_atoms.info = atoms.info.copy()
            ts_energy = atoms.get_potential_energy()
            ts_forces = atoms.get_forces()
            finalize_if_vasp_interactive(config_dict, ts_calc)
            if is_vasp:
                remove_vasp_heavies(f"VASP_{i}_0")

            # --- MINIMIZE BOTH SIDES ---
            mins = {}  # side -> (atoms, converged)
            displacement = 0.25

            # Per-side VASP calc cache: instantiated lazily, reused for both
            # the desorption-check single-point AND the subsequent relaxation
            # (so WAVECAR warm-starts the relax).
            side_calcs = {}

            def _side_calc(side):
                if not is_vasp:
                    return calc
                if side not in side_calcs:
                    side_calcs[side] = resolve_vasp_calc(
                        config_dict, calc, i, side, "ourDoubleMinimization")
                    active_vasp_calcs.append(side_calcs[side])
                return side_calcs[side]

            # Desorption: only optimize toward bound state, skip desorption direction
            skip_side = None
            if orig.get('reaction_type') == 'desorption':
                energies = {}
                for test_side in [-1, 1]:
                    test_atoms = ts_atoms.copy()
                    test_atoms.calc = _side_calc(test_side)
                    test_atoms.positions += -test_side * displacement * refined_eigenmode
                    energies[test_side] = test_atoms.get_potential_energy()
                skip_side = max(energies, key=energies.get)

            for side in [-1, 1]:
                should_run = entries_to_run is None or side in entries_to_run

                if side == skip_side:
                    # Desorption direction: use TS as placeholder, no optimization
                    min_atoms = ts_atoms.copy()
                    energy = ts_energy
                    forces = ts_forces
                    conv = True
                    # The skip-side dir got one single-point during the desorption
                    # check above; finalize + clean WAVECAR so nothing leaks.
                    if is_vasp and side in side_calcs:
                        finalize_if_vasp_interactive(config_dict, side_calcs[side])
                        remove_vasp_heavies(f"VASP_{i}_{side}")
                elif should_run:
                    if continuation_data and side in continuation_data and continue_from_result:
                        min_atoms = continuation_data[side].copy()
                        min_atoms.calc = _side_calc(side)
                    else:
                        min_atoms = ts_atoms.copy()
                        min_atoms.calc = _side_calc(side)
                        min_atoms.positions += -side * displacement * refined_eigenmode

                    log_f = f'optimization_{i}_{side}.log'
                    traj_f = f'optimization_{i}_{side}.traj'
                    temp_files.extend([log_f, traj_f])

                    optimizable = FrechetCellFilter(min_atoms) if config_dict['our'+method_name]['relax_cell'] else min_atoms
                    conv = relax_structure(config_dict, optimizable, log_f, traj_f, Optimizer)
                    energy = min_atoms.get_potential_energy()
                    forces = min_atoms.get_forces()
                    if is_vasp:
                        finalize_if_vasp_interactive(config_dict, side_calcs[side])
                        remove_vasp_heavies(f"VASP_{i}_{side}")
                else:
                    if not (continuation_data and side in continuation_data):
                        raise ValueError(f"Missing continuation data for kept side={side}")
                    min_atoms = continuation_data[side].copy()
                    conv = bool(min_atoms.info.get('orig_info', {}).get('converged'))
                    energy = min_atoms.get_potential_energy()
                    forces = min_atoms.get_forces()

                min_atoms.info['side'] = side
                min_atoms.info['parent_ts_index'] = parent_source_idx
                min_atoms.info['converged'] = conv
                min_atoms.info['src_index'] = i
                mins[side] = (min_atoms, conv, energy, forces)

            min1, conv1, min1_energy, min1_forces = mins[-1]
            min2, conv2, min2_energy, min2_forces = mins[1]

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
            min1.calc = SinglePointCalculator(min1, energy=min1_energy, forces=min1_forces)
            ts_atoms.calc = SinglePointCalculator(ts_atoms, energy=ts_energy, forces=ts_forces)
            min2.calc = SinglePointCalculator(min2, energy=min2_energy, forces=min2_forces)
            writer.write(min1)
            writer.write(ts_atoms)
            writer.write(min2)

            # --- CLEANUP (Success Case) ---
            archive_and_clear_temp_files(temp_files, zip_name, prefix="",
                                         enabled=config_dict['Main']['zip'])

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
            for vc in active_vasp_calcs:
                finalize_if_vasp_interactive(config_dict, vc)
            for side in [-1, 1]:
                if entries_to_run is None or side in entries_to_run:
                    log_status(side, parent_source_idx, f"error: {str(e)}")

            archive_and_clear_temp_files(temp_files, zip_name, prefix="ERROR_",
                                         enabled=config_dict['Main']['zip'])


def singlepoint(i, config_dict, atoms, calc, consecutive_errors=None,
                executorlib_worker_id=None, **kwargs):
    """Single-point energy/force calculation. Writes results to traj or LMDB.

    With frames_per_job>1, `atoms` arrives as a list of frames and all of them
    are fused into one batched FAIRChem forward pass, then written in input
    order to the same rank shard.
    """
    rank = executorlib_worker_id

    max_consecutive_errors = config_dict["Main"]["max_consecutive_errors"]
    if consecutive_errors is not None and consecutive_errors[0] >= max_consecutive_errors > 0:
        print(f"Rank {rank}: {consecutive_errors[0]} consecutive structures errored. "
              f"Killing worker for restart.", flush=True)
        backup_flux_logs(rank)
        sys.exit(1)

    method_name = config_dict["Main"]["method"]   # "SinglePoint"
    input_format = config_dict["Main"]["input_format"]
    is_vasp = config_dict["Main"]["Calculator"] in ("Vasp", "VaspInteractive")
    status_file = f"{method_name}_status_csvs/status_rank_{rank}.csv"
    task_name = get_task_name(config_dict)

    frames = atoms if isinstance(atoms, list) else [atoms]
    extras = kwargs.get('extras') or [{} for _ in frames]

    def log_status(status_msg):
        with open(status_file, 'a') as f:
            f.write(f'{i},{rank},"{status_msg}"\n')

    vasp_calc = None
    vasp_dir = f"VASP_{i}" if is_vasp else None
    try:
        if is_vasp:
            # Config-level guard enforces frames_per_job=1 for VASP; defensive
            # assert here so a misconfigured call doesn't silently drop frames.
            if len(frames) != 1:
                raise ValueError(
                    f"SinglePoint+VASP requires frames_per_job=1; got {len(frames)} frames.")
            a = frames[0]
            vasp_calc = resolve_vasp_calc(config_dict, calc, i, None, "ourSinglePoint")
            a.calc = vasp_calc
            energy_v = a.get_potential_energy()
            forces_v = a.get_forces()
            finalize_if_vasp_interactive(config_dict, vasp_calc)
            ef_pairs = [(energy_v, forces_v)]
        elif len(frames) > 1:
            # Single batched FAIRChem forward pass for all frames.
            # Pattern from catsunami/ocpneb.py:168-175. Frames may have different
            # natoms (different parent structures in the same batch) — slice the
            # concatenated forces by per-frame natoms.
            import numpy as _np
            from fairchem.core.datasets.atomic_data import atomicdata_list_to_batch
            data_list = [calc.a2g(a) for a in frames]
            batch = atomicdata_list_to_batch(data_list)
            preds = calc.predictor.predict(batch)
            energies = preds["energy"].detach().cpu().flatten().tolist()
            forces_flat = preds["forces"].detach().cpu().numpy()
            offsets = _np.cumsum([0] + [len(a) for a in frames])
            ef_pairs = [(energies[k], forces_flat[offsets[k]:offsets[k+1]])
                        for k in range(len(frames))]
        else:
            a = frames[0]
            a.calc = calc
            ef_pairs = [(a.get_potential_energy(), a.get_forces())]

        if input_format == "lmdb":
            import fairchem.core.datasets  # noqa: F401  (register aselmdb backend)
            from ase.db import connect
            out_path = f"{method_name}_lmdbs/collected_sp_rank_{rank}.aselmdb"
            with connect(out_path, type='aselmdb') as db:
                for a, (e, f_arr), extra in zip(frames, ef_pairs, extras):
                    a.info['src_index'] = i
                    a.info['status'] = 'converged'
                    a.info['task_name'] = task_name
                    a.calc = SinglePointCalculator(a, energy=e, forces=f_arr)
                    db.write(a, **(extra.get('kvp') or {}),
                             data=(extra.get('row_data') or {}))
        else:
            out_path = f"{method_name}_trajes/collected_sp_rank_{rank}.traj"
            with Trajectory(out_path, 'a') as writer:
                for a, (e, f_arr) in zip(frames, ef_pairs):
                    a.info['src_index'] = i
                    a.info['status'] = 'converged'
                    a.info['task_name'] = task_name
                    a.calc = SinglePointCalculator(a, energy=e, forces=f_arr)
                    writer.write(a)

        if vasp_dir is not None and os.path.isdir(vasp_dir):
            # SP has no _debug_zips/ — just drop the VASP scratch dir.
            import shutil as _shutil
            _shutil.rmtree(vasp_dir, ignore_errors=True)

        log_status("converged")
        if consecutive_errors is not None:
            consecutive_errors[0] = 0

    except Exception as e:
        print(f"Rank {rank} FAILED on structure {i}: {e}", flush=True)
        print(f"\nTraceback details:\n{traceback.format_exc()}", flush=True)
        if consecutive_errors is not None:
            consecutive_errors[0] += 1
        if vasp_calc is not None:
            finalize_if_vasp_interactive(config_dict, vasp_calc)
        if vasp_dir is not None and os.path.isdir(vasp_dir):
            import shutil as _shutil
            _shutil.rmtree(vasp_dir, ignore_errors=True)
        log_status(f"error: {str(e)}")
