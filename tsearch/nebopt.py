import os
from tsearch.config import load_config, load_calculator, load_optimizer
config_dict = load_config("config.ini")
if config_dict["Main"]["jobs_per_gpu"] != 1: os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import traceback
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import zipfile, os
import numpy as np
from ase.data import covalent_radii
from ase.calculators.singlepoint import SinglePointCalculator
from ase.io import Trajectory
from ase.mep.neb import NEB, NEBTools, NEBState
from tsearch.catsunami.ocpneb import OCPNEB


calc = load_calculator(config_dict)
Optimizer = load_optimizer(config_dict)


def nebopt(i, config_dict, images, executorlib_worker_id=None):

    rank = executorlib_worker_id
    zip_name = f"{config_dict['Main']['method']}_debug_zips/structure_rank_{rank}_data.zip"
    status_file = f"{config_dict['Main']['method']}_status_csvs/status_rank_{rank}.csv"
    my_output_file = f"{config_dict['Main']['method']}_trajes/collected_ts_rank_{rank}.traj"
    temp_log = f'neb_{i}.log'
    temp_traj = f'neb_{i}.traj'
    temp_plot = f'diffusion_barrier_{i}.png'
    temp_react_relax_log = f'reactant_relaxation_{i}.log'
    temp_prod_relax_log = f'product_relaxation_{i}.log'
    temp_react_relax = f'reactant_relaxation_{i}.traj'
    temp_prod_relax = f'product_relaxation_{i}.traj'
    temp_files = [temp_log, temp_traj, temp_plot, temp_react_relax_log, temp_prod_relax_log, temp_react_relax, temp_prod_relax]

    def log_status(status_msg):
        with open(status_file, 'a') as f:
            f.write(f"{i},{rank},{status_msg}\n")

    relax_endpoints = config_dict["ourNEB"]["relax_endpoints"]
    interpolate_method = config_dict["ourNEB"]["interpolate_method"]  # this is idpp implementation from Meta OCP, other choises are "ase_idpp" and "ase_linear" or False if you already have a frame set
    perform_aseidpp = False
    num_frames = config_dict["ourNEB"]["num_frames"]

    try:
        reactant = images[0]
        product = images[-1]

        if relax_endpoints:
            if not interpolate_method: print("Are you sure you want to relax end points while keeping the intermediate inages from your traj?")
            reactant.calc = calc
            opt = Optimizer(reactant, logfile=temp_react_relax_log, trajectory=temp_react_relax, **config_dict[config_dict["Main"]["Optimizer"]])
            opt.run(config_dict["ourNEB"]["endpoint_relax_fmax"], config_dict["ourNEB"]["endpoint_relax_steps"])
            product.calc = calc
            opt = Optimizer(product, logfile=temp_prod_relax_log, trajectory=temp_prod_relax, **config_dict[config_dict["Main"]["Optimizer"]])
            opt.run(config_dict["ourNEB"]["endpoint_relax_fmax"], config_dict["ourNEB"]["endpoint_relax_steps"])

        if interpolate_method:
            if interpolate_method == "ocp_idpp":
                # `interpolate` function Meta implemented is very similar to idpp but not sensative to periodic boundary crossings. 
                # Alternatively you can adopt whatever interpolation scheme you prefer. The `interpolate` function lacks some of the extra protections implemented 
                # in the `interpolate_and_correct_frames` which is used in the CatTSunami enumeration workflow. Care should be taken to ensure the results are reasonable.
                # 
                # IMPORTANT NOTES: 
                # 1. Make sure the indices in the initial and final frame map to the same atoms
                # 2. Ensure you have the proper constraints on subsurface atoms
                # 
                """
                The approach uses ase, so you must provide ase.Atoms objects
                with the appropriate constraints (i.e. fixed subsurface atoms).
                """
                from tsearch.catsunami.autoframe import interpolate
                images = interpolate(reactant, product, num_frames)

            elif interpolate_method[:4] == "ase_":
                images = [reactant]
                images += [reactant.copy() for i in range(num_frames-2)]
                images += [product]

                neb0 = NEB(images, **config_dict["DyNEB"])

                if interpolate_method[4:] == "idpp":
                    perform_aseidpp = True
                else:
                    neb0.interpolate(method="linear", mic=True)

                    # Array of covalent radii for the system
                    radii = np.array([covalent_radii[z] for z in reactant.numbers])
                    radii_sum = radii[:, None] + radii[None, :]

                    for atoms in neb0.images[1:-1]:
                        dists = atoms.get_all_distances(mic=True)
                        np.fill_diagonal(dists, np.inf)

                        if np.any(dists < 0.6 * radii_sum):
                            perform_aseidpp = True
                            break

                if perform_aseidpp:
                    neb0.interpolate(method="idpp", mic=True)

        for image in images:
            image.calc = calc

        neb = OCPNEB(
            images,
            batch_size = config_dict["ourNEB"]["batch_size"], # If you get a memory error, try reducing it to 4
            dneb = config_dict["ourNEB"]["DNEB"],
            **config_dict["DyNEB"],
        )

        opt = Optimizer(neb,
                        logfile = temp_log,
                        trajectory = temp_traj,
                        **config_dict[config_dict["Main"]["Optimizer"]],
                        )
        converged = opt.run(fmax = config_dict["Main"]["fmax"], steps = config_dict["Main"]["steps"])

        ci_image = neb.images[neb.imax].copy()
        energy = neb.intermediate_energies[neb.imax]
        forces = neb.intermediate_forces[len(ci_image)*(neb.imax-1):len(ci_image)*neb.imax]
        state = NEBState(neb, neb.images, neb.intermediate_energies)
        spring1 = state.spring(neb.imax-1)
        spring2 = state.spring(neb.imax)
        tangent = neb.neb_method.get_tangent(state, spring1, spring2, neb.imax)

        nebtools = NEBTools(neb.images)
        Ef, dE = nebtools.get_barrier()
        max_forces = nebtools.get_fmax(**config_dict["DyNEB"])
        fig = nebtools.plot_band()
        fig.savefig(temp_plot)
        plt.close(fig)

        with Trajectory(my_output_file, 'a') as writer:
            ci_image.info['eigenmode'] = tangent
            ci_image.calc = SinglePointCalculator(ci_image, energy=energy, forces=forces)
            ci_image.info['converged'] = 1 if converged else 0
            ci_image.info['src_index'] = i
            ci_image.info['barrier'] = Ef
            ci_image.info['dE'] = dE
            ci_image.info['max_forces'] = max_forces
            ci_image.info['reactant_positions'] = neb.images[0].positions
            ci_image.info['product_positions'] = neb.images[-1].positions
            ci_image.info['interpolation_method'] = interpolate_method
            if isinstance(interpolate_method, str) and interpolate_method.startswith("ase_") and perform_aseidpp:
                ci_image.info['interpolation_method'] = "ase_idpp"
            ci_image.wrap()
            writer.write(ci_image)

        if converged:
            log_status("converged")
        else:
            log_status("not_converged")

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
        print(f"\nTraceback details:\n{traceback.format_exc()}")
        existing_files = [f for f in temp_files if os.path.exists(f)]
        if existing_files:
            with zipfile.ZipFile(zip_name, 'a', zipfile.ZIP_DEFLATED) as zf:
                for f_name in existing_files:
                    zf.write(f_name, arcname=f"{f_name}")
            for f_name in existing_files:
                os.remove(f_name)
        log_status("error")

