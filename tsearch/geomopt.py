import sys, os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import zipfile
from ase.io import Trajectory
from ase.filters import FrechetCellFilter
from ase.calculators.singlepoint import SinglePointCalculator
from tsearch.config import load_config, load_calculator, load_optimizer


config_dict = load_config("config.ini")
calc = load_calculator(config_dict)
Optimizer = load_optimizer(config_dict)


def relax_structure(config_dict, atoms, logfile, trajfile):
    opt = Optimizer(FrechetCellFilter(atoms), logfile=logfile, trajectory=trajfile,
                    **config_dict[config_dict["Main"]["Optimizer"]])
    converged = opt.run(fmax=config_dict["Main"]["fmax"], steps=config_dict["Main"]["steps"])
    return converged


def geomopt(i, config_dict, atoms, executorlib_worker_id=None):
    
    rank = executorlib_worker_id
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
            converged = relax_structure(config_dict, atoms, temp_opt_log, temp_traj)
            
            if converged:
                log_status("converged")
                atoms.info['converged'] = 1
            else:
                log_status("not_converged")
                atoms.info['converged'] = 0
            atoms.info['src_index'] = i

            writer.write(atoms)

            # Clean up temp files
            existing_files = [f for f in temp_files if os.path.exists(f)]
            if existing_files:
                #if not converged:
                if True:  # converged or unconverged
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
            log_status("error") 


def doublegeomopt(i, config_dict, atoms, executorlib_worker_id=None):
    
    rank = executorlib_worker_id
    atoms.calc = calc

    method_name = config_dict["Main"]["method"]
    status_file = f"{method_name}_status_csvs/status_rank_{rank}.csv"
    my_output_file = f"{method_name}_trajes/collected_opt_rank_{rank}.traj"
    zip_name = f"{method_name}_debug_zips/structure_rank_{rank}_data.zip"

    def log_status(parent_source_idx, status_msg):
        with open(status_file, 'a') as f:
            f.write(f"{i},{rank},{parent_source_idx},{status_msg}\n")

    # 3. Initialize list to track temp files from BOTH optimizations
    temp_files = []

    with Trajectory(my_output_file, 'a') as writer:
        try:
            if not atoms.info['converged']:
                raise Exception("Input structure marked as unconverged.")
            
            if 'eigenmode' not in atoms.info:
                 raise Exception("Input structure missing 'eigenmode' in info.")
            if 'src_index' not in atoms.info:
                 raise Exception("Input structure missing 'eigenmode' in info.")


            # Identify IDs
            parent_source_idx = atoms.info['src_index']
            refined_eigenmode = atoms.info['eigenmode']

            # --- PREPARE TS (Middle Image) ---
            ts_atoms = atoms.copy()
            ts_atoms.info = atoms.info.copy()
            ts_atoms.calc = SinglePointCalculator(
                ts_atoms,
                energy=atoms.get_potential_energy(),
                forces=atoms.get_forces()
            )

            # --- MINIMIZATION 1 (Forward) ---
            min1 = ts_atoms.copy()
            min1.calc = calc
            displacement = 0.25
            min1.positions += displacement * refined_eigenmode
            
            # Define temp files for step 1
            log1 = f'optimization_r{rank}_{i}-0.log'
            traj1 = f'optimization_r{rank}_{i}-0.traj'
            temp_files.extend([log1, traj1])
            
            conv1 = relax_structure(config_dict, min1, log1, traj1)
            
            # Freeze results
            min1.calc = SinglePointCalculator(min1, energy=min1.get_potential_energy(), forces=min1.get_forces())
            min1.info['type'] = 'minimum_1'
            min1.info['parent_ts_index'] = parent_source_idx
            min1.info['converged'] = conv1

            # --- MINIMIZATION 2 (Backward) ---
            min2 = ts_atoms.copy()
            min2.calc = calc
            min2.positions -= displacement * refined_eigenmode
            
            # Define temp files for step 2
            log2 = f'optimization_r{rank}_{i}-1.log'
            traj2 = f'optimization_r{rank}_{i}-1.traj'
            temp_files.extend([log2, traj2])
            
            conv2 = relax_structure(config_dict, min2, log2, traj2)
            
            # Freeze results
            min2.calc = SinglePointCalculator(min2, energy=min2.get_potential_energy(), forces=min2.get_forces())
            min2.info['type'] = 'minimum_2'
            min2.info['parent_ts_index'] = parent_source_idx
            min2.info['converged'] = conv2

            # --- WRITE TRIPLET (Min1 -> TS -> Min2) ---
            writer.write(min1)
            writer.write(ts_atoms)
            writer.write(min2)
            
            log_status(parent_source_idx, "done")

            # --- CLEANUP (Success Case) ---
            existing_files = [f for f in temp_files if os.path.exists(f)]
            if existing_files:
                with zipfile.ZipFile(zip_name, 'a', zipfile.ZIP_DEFLATED) as zf:
                    for f_name in existing_files:
                        zf.write(f_name, arcname=f_name)
                for f_name in existing_files:
                    os.remove(f_name)

        except Exception as e:
            # --- CLEANUP (Error Case) ---
            print(f"Rank {rank} FAILED on structure {i}: {e}")
            log_status(parent_source_idx, f"error: {str(e)}")
            
            existing_files = [f for f in temp_files if os.path.exists(f)]
            if existing_files:
                with zipfile.ZipFile(zip_name, 'a', zipfile.ZIP_DEFLATED) as zf:
                    for f_name in existing_files:
                        zf.write(f_name, arcname=f"ERROR_{f_name}")
                for f_name in existing_files:
                    os.remove(f_name)
