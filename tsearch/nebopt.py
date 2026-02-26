import os
import traceback
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import zipfile, os
import numpy as np
from ase.data import covalent_radii
from ase.calculators.singlepoint import SinglePointCalculator
from ase.io import Trajectory
from ase.mep.neb import NEB, NEBTools, NEBState
from tsearch.catsunami.ocpneb import OCPNEB


def nebopt(i, config_dict, images, calc, Optimizer, executorlib_worker_id=None):

    rank = executorlib_worker_id
    relax_endpoints = config_dict["ourNEB"]["relax_endpoints"]
    interpolate_method = config_dict["ourNEB"]["interpolate_method"]  # this is idpp implementation from Meta OCP, other choises are "ase_idpp" and "ase_linear" or False if you already have a frame set
    perform_aseidpp = False
    num_frames = config_dict["ourNEB"]["num_frames"]
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
    if config_dict["Main"]["Calculator"] in ("Vasp", "VaspInteractive"):
        temp_files.extend([f"{i}_{image_idx}" for image_idx in range(num_frames)])

    def log_status(status_msg):
        with open(status_file, 'a') as f:
            f.write(f"{i},{rank},{status_msg}\n")

    try:
        reactant = images[0]
        if config_dict["Main"]["Calculator"] in ("Vasp", "VaspInteractive"):
            reactant.calc = calc(
                directory=f"{i}_{0}",
                command=config_dict["ourNEB"]["vasp_command_endpoints"],
                ncore=int(config_dict["ourNEB"]["vasp_ncore_endpoints"]),
                **config_dict["Vasp"],
                )
        else:
            reactant.calc = calc

        product = images[-1]
        if config_dict["Main"]["Calculator"] in ("Vasp", "VaspInteractive"):
            product.calc = calc(
                directory=f"{i}_{num_frames-1}",
                command=config_dict["ourNEB"]["vasp_command_endpoints"],
                ncore=int(config_dict["ourNEB"]["vasp_ncore_endpoints"]),
                **config_dict["Vasp"],
                )
        else:
            product.calc = calc

        if relax_endpoints:
            if not interpolate_method: print("Are you sure you want to relax end points while keeping the intermediate images from your traj?")
            if config_dict["ourNEB"]["endpoint_relax_Optimizer"] is None:
                endpoint_relax_optimizer_name = config_dict["Main"]["Optimizer"]
            else:
                endpoint_relax_optimizer_name = config_dict["ourNEB"]["endpoint_relax_Optimizer"]

            opt = Optimizer[0](reactant, logfile=temp_react_relax_log, trajectory=temp_react_relax, **config_dict[endpoint_relax_optimizer_name])
            opt.run(config_dict["ourNEB"]["endpoint_relax_fmax"], config_dict["ourNEB"]["endpoint_relax_steps"])
            energy, forces = reactant.get_potential_energy(), reactant.get_forces()
            if config_dict["Main"]["Calculator"] == "VaspInteractive": reactant.calc.finalize()
            reactant.calc = SinglePointCalculator(reactant, energy=energy, forces=forces)

            opt = Optimizer[0](product, logfile=temp_prod_relax_log, trajectory=temp_prod_relax, **config_dict[endpoint_relax_optimizer_name])
            opt.run(config_dict["ourNEB"]["endpoint_relax_fmax"], config_dict["ourNEB"]["endpoint_relax_steps"])
            energy, forces = product.get_potential_energy(), product.get_forces()
            if config_dict["Main"]["Calculator"] == "VaspInteractive": product.calc.finalize()
            product.calc = SinglePointCalculator(product, energy=energy, forces=forces)

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

        for image_idx in range(1,num_frames-1):
            if config_dict["Main"]["Calculator"] in ("Vasp", "VaspInteractive"):
                images[image_idx].calc = calc(
                    directory=f"{i}_{image_idx}",
                    command=config_dict["ourNEB"]["vasp_command_intermediates"],
                    ncore=int(config_dict["ourNEB"]["vasp_ncore_intermediates"]),
                    **config_dict["Vasp"],
                    )
            else:
                images[image_idx].calc = calc

        neb = OCPNEB(
            images,
            batch_size = config_dict["ourNEB"]["batch_size"], # If you get a memory error, try reducing it to 4
            dneb = config_dict["ourNEB"]["DNEB"],
            vasp = config_dict["Main"]["Calculator"] in ("Vasp", "VaspInteractive"),
            **config_dict["DyNEB"],
        )

        opt = Optimizer[1](neb,
                           logfile = temp_log,
                           trajectory = temp_traj,
                           **config_dict[config_dict["Main"]["Optimizer"]],
                           )
        converged = opt.run(fmax = config_dict["Main"]["fmax"], steps = config_dict["Main"]["steps"])

        if config_dict["Main"]["Calculator"] == "VaspInteractive":
            for img in neb.images[1:-1]:
                img.calc.finalize()

        ci_image = neb.images[neb.imax].copy()
        energy = neb.energies[neb.imax]
        forces = neb.real_forces[neb.imax]
        state = NEBState(neb, neb.images, neb.energies)
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
        if config_dict["Main"]["Calculator"] in ("Vasp", "VaspInteractive"):
            for image_idx in range(num_frames):
                for vasp_heavy_files in [f'{i}_{image_idx}/WAVECAR',f'{i}_{image_idx}/CHG',f'{i}_{image_idx}/CHGCAR']:
                    if os.path.exists(vasp_heavy_files): os.remove(vasp_heavy_files)
        existing_files = [f for f in temp_files if os.path.exists(f)]
        if existing_files and config_dict['Main']['zip']:
            with zipfile.ZipFile(zip_name, 'a', zipfile.ZIP_DEFLATED) as zf:
                for f_name in existing_files:
                    zf.write(f_name, arcname=f"{f_name}")
            for f_name in existing_files:
                os.remove(f_name)

    except Exception as e:
        print(f"Rank {rank} FAILED on structure {i}: {e}")
        print(f"\nTraceback details:\n{traceback.format_exc()}")
        if config_dict["Main"]["Calculator"] in ("Vasp", "VaspInteractive"):
            for image_idx in range(num_frames):
                for vasp_heavy_files in [f'{i}_{image_idx}/WAVECAR',f'{i}_{image_idx}/CHG',f'{i}_{image_idx}/CHGCAR']:
                    if os.path.exists(vasp_heavy_files): os.remove(vasp_heavy_files)
        existing_files = [f for f in temp_files if os.path.exists(f)]
        if existing_files and config_dict['Main']['zip']:
            with zipfile.ZipFile(zip_name, 'a', zipfile.ZIP_DEFLATED) as zf:
                for f_name in existing_files:
                    zf.write(f_name, arcname=f"{f_name}")
            for f_name in existing_files:
                os.remove(f_name)
        log_status("error")

