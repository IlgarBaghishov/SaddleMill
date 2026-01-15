import sys
import numpy as np
from ase.optimize import BFGS, FIRE, LBFGS, MDMin
from ase.io import read
from ase.mep import NEB, NEBTools
from tsearch.catsunami.ocpneb import OCPNEB
from tsearch.catsunami.autoframe import interpolate
from tsearch.tools import parse_inputfile, load_calculator


config_dict = parse_inputfile("config.ini")
calc = load_calculator(config_dict)


def nebopt(i, config_dict):
    relax_endpoints = config_dict["ourNEB"]["relax_endpoints"]
    interpolate_method = config_dict["ourNEB"]["interpolate_method"]  # this is idpp implementation from Meta OCP, other choises are "ase_idpp" and "ase_linear" or False if you already have a frame set
    num_frames = config_dict["ourNEB"]["num_frames"]

    if not interpolate_method:
        """
        The approach uses ase, so you must provide a list of ase.Atoms objects
        with the appropriate constraints.
        """
        images = read(config_dict["ourNEB"]["interpolated_images_path"], f"0:{num_frames}")  # Change to the path to your atoms of the frame set
        reactant = images[0]
        product = images[-1]
    else:
        reactant = read(config_dict["ourNEB"]["reactant_path"])
        product = read(config_dict["ourNEB"]["product_path"])

    if relax_endpoints:
        if not interpolate_method: print("Are you sure you want to relax end points while keeping the intermediate inages from your traj?")
        reactant.calc = calc
        opt = BFGS(reactant, trajectory=f'reactant_relaxation{i}.traj')
        opt.run(0.05, 300)
        reactant.calc = calc
        opt = BFGS(product, trajectory=f'product_relaxation{i}.traj')
        opt.run(0.05, 300)

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
        images = interpolate(reactant, product, num_frames)

    elif interpolate_method[:4] == "ase_":
        images = [reactant]
        images += [reactant.copy() for i in range(num_frames-2)]
        images += [product]

        neb0 = NEB(images, **config_dict["DyNEB"])
        neb0.interpolate(method=interpolate_method[4:], mic=True)

    for image in images:
        image.calc = calc

    neb = OCPNEB(
        images,
        batch_size = config_dict["ourNEB"]["batch_size"], # If you get a memory error, try reducing it to 4
        **config_dict["DyNEB"],
    )

    optimizer = MDMin(neb, dt=0.02, maxstep=0.1, trajectory=f"your-neb{i}.traj")
    conv = optimizer.run(fmax=config_dict["Main"]["fmax"], steps=config_dict["Main"]["steps"])

    # optimizer = MDMin(neb, dt=0.02, maxstep=0.1, trajectory=f"your-neb.traj")
    # conv = optimizer.run(fmax=fmax + delta_fmax_climb, steps=500)
    # if conv:
    #     print("initial NEB optimization is done, starting climbing image")
    #     neb.climb = True
    #     conv = optimizer.run(fmax=fmax, steps=1000)


    # Final analysis
    nebtools = NEBTools(images)
    Ef, dE = nebtools.get_barrier()

    # Get the actual maximum force at this point in the simulation.
    max_force = nebtools.get_fmax(vars(neb))

    print(f'Diffusion barrier: {Ef:.3f} eV and {dE:.3f} eV')
    print(f'Maximum force: {np.array2string(max_force, precision=3)} eV/Å')

    # Create a figure like that coming from ASE-GUI.
    fig = nebtools.plot_band()
    fig.savefig('diffusion-barrier.png')

    # # Create a figure with custom parameters.
    # fig = plt.figure(figsize=(5.5, 4.0))
    # ax = fig.add_axes((0.15, 0.15, 0.8, 0.75))
    # nebtools.plot_band(ax)
    # fig.savefig('diffusion-barrier.png')
