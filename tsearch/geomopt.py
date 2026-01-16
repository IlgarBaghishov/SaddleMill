import sys, os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from ase.io import read
from ase.filters import FrechetCellFilter
from tsearch.tools import parse_inputfile, load_calculator, load_optimizer
from flux import Flux, resource

handle = Flux()
rs = resource.status.ResourceStatusRPC(handle).get()
rl = resource.list.resource_list(handle).get()
all_ncores = rl.all.ncores
all_ngpus = rl.all.ngpus
print("in geomopt", all_ncores, all_ngpus)


config_dict = parse_inputfile("config.ini")
calc = load_calculator(config_dict)
Optimizer = load_optimizer(config_dict)


def geomopt(i, config_dict, atoms, executorlib_worker_id=None):
    
    rank = executorlib_worker_id
    atoms.calc = calc

    method = load_method(config_dict)
    status_file = f"{method}_status_csvs/status_rank_{rank}.csv"
    my_output_file = f"{method}_trajes/collected_opt_rank_{rank}.traj"
    zip_name = f"{method}_debug_zips/structure_rank_{rank}_data.zip"

    opt = Optimizer(FrechetCellFilter(atoms), logfile=f'optimization{rank}_{i}.log', **config_dict[config_dict["Main"]["Optimizer"]])
    opt.run(fmax=config_dict["Main"]["fmax"], steps=config_dict["Main"]["steps"])


def doublegeomopt(i, config_dict, atoms, executorlib_worker_id=None):
    from ase.mep import DimerControl, MinModeAtoms
    from ase.calculators.singlepoint import SinglePointCalculator

    rank = executorlib_worker_id
    atoms.calc = calc

    opt = Optimizer(FrechetCellFilter(atoms), logfile=f'optimization{rank}_{i}.log', **config_dict[config_dict["Main"]["Optimizer"]])
    opt.run(fmax=config_dict["Main"]["fmax"], steps=config_dict["Main"]["steps"])

    method_name = config_dict["Main"]["method"]
    status_file = f"{method_name}_status_csvs/status_rank_{rank}.csv"
    my_output_file = f"{method_name}_trajes/collected_opt_rank_{rank}.traj"
    zip_name = f"{method_name}_debug_zips/structure_rank_{rank}_data.zip"


    def relax_structure(atoms_obj, suffix="min"):
        """
        Helper function to relax a structure using FIRE.
        Returns the converged atoms object and a boolean for convergence.
        """
        # Create a unique log filename to avoid conflicts in parallel jobs
        log_file = f"opt_{suffix}_r{rank}_{os.getpid()}.log"
        
        #opt = FIRE(atoms_obj, logfile=log_file)
        opt = FIRE(atoms_obj, logfile=None)  # to prevent file io bottlenecks
        converged = opt.run(fmax=FMAX_TOL, steps=500)
        
        # Cleanup log file to save inode space
        #if os.path.exists(log_file):
        #    os.remove(log_file)
            
        return atoms_obj, converged

    # --- MAIN PROCESSING LOOP ---
    for filenum, filename in enumerate(my_files):
        print(f"Rank {rank}: Processing {filename}... ({filenum+1}/{len(my_files)})", flush=True)
        
        try:
            # Read all images in the trajectory
            traj_images = read(filename, index=':')
        except Exception as e:
            print(f"Rank {rank}: Error reading {filename}: {e}")
            continue

        # Prepare output filename
        # Strip the parent path (../) to get just the filename
        base_name = os.path.splitext(os.path.basename(filename))[0]
        output_filename = os.path.join(output_dir, f"{base_name}_with-mins.traj")

        with Trajectory(output_filename, 'w') as writer:
            # Open output trajectory
            for idx, atoms in enumerate(traj_images):
                try:
                    if not atoms.info['converged']:
                        continue
                    parent_source_idx = atoms.info['src_index']

                    # 1. SETUP & REFINE MODE (No Translation)
                    atoms.calc = calc
                    
                    # Configure DimerControl for finding the mode.
                    # initial_eigenmode_method='gauss' will automatically:
                    # 1. Displace atoms randomly by gauss_std (0.1)
                    # 2. Calculate the vector
                    # 3. Move atoms BACK to original position
                    # 4. Then start rotating to refine the mode.
                    #d_control = DimerControl(
                    #    initial_eigenmode_method='gauss', 
                    #    displacement_method='gauss', 
                    #    gauss_std=0.1,        # Must be > 0 to generate initial vector
                    #    max_num_rot=200,      # Allow many rotations to find true mode
                    #    f_rot_min=FMAX_TOL,      # Tight convergence for the rotation
                    #    f_rot_max=FMAX_TOL,
                    #    dimer_separation=0.01,
                    #    logfile=None          # Suppress detailed rotation logs
                    #)
                    #
                    #d_atoms = MinModeAtoms(atoms, d_control)
                    #
                    ## Calling get_forces() triggers the mode initialization and rotation loop.
                    ## Since we are NOT using MinModeTranslate, the atoms stay put.
                    #d_atoms.get_forces()
                    #
                    ## Retrieve the refined eigenmode
                    #refined_eigenmode = d_atoms.get_eigenmode()
                    #atoms.info['eigenmode'] = refined_eigenmode
                    #print("SHAPE:", refined_eigenmode.shape)
                    refined_eigenmode = atoms.info['eigenmode']
                    
                    # Save the refined TS structure data (update info but keep position)
                    ts_atoms = atoms.copy()
                    ts_atoms.info = atoms.info.copy()
                    ts_atoms.calc = SinglePointCalculator(
                        ts_atoms, 
                        energy=atoms.get_potential_energy(), 
                        forces=atoms.get_forces()
                    )

                    # 2. MINIMIZATION 1 (Forward along mode)
                    min1 = ts_atoms.copy()
                    min1.calc = calc
                    displacement = 0.25 # Angstrom (Small push)
                    min1.positions += displacement * refined_eigenmode
                    min1, conv1 = relax_structure(min1, suffix=f"{parent_source_idx}_pos")
                    e, f = min1.get_potential_energy(), min1.get_forces()
                    min1.calc = SinglePointCalculator(
                        min1,
                        energy=e,
                        forces=f
                    )
                    min1.info['type'] = 'minimum_1'
                    min1.info['parent_ts_index'] = parent_source_idx
                    min1.info['converged'] = conv1

                    # 3. MINIMIZATION 2 (Backward along mode)
                    # Note: We start from atoms (the TS), NOT min1
                    min2 = ts_atoms.copy()
                    min2.calc = calc
                    min2.positions -= displacement * refined_eigenmode
                    min2, conv2 = relax_structure(min2, suffix=f"{parent_source_idx}_neg")
                    e, f = min2.get_potential_energy(), min2.get_forces()
                    min2.calc = SinglePointCalculator(
                        min2,
                        energy=e,
                        forces=f
                    )
                    min2.info['type'] = 'minimum_2'
                    min2.info['parent_ts_index'] = parent_source_idx
                    min2.info['converged'] = conv2

                    # 4. WRITE TRIPLET (Min1 -> TS -> Min2)
                    writer.write(min1)
                    writer.write(ts_atoms)
                    writer.write(min2)
                    
                    print(f"\tRank {rank}: File {filename} Img {idx} -> Done ({idx+1}/{len(traj_images)}).", flush=True)

                except Exception as e:
                    print(f"Rank {rank}: Failed on {filename} image {idx}. Error: {e}", flush=True)
                    continue

    print(f"Rank {rank}: Finished processing assigned files.")