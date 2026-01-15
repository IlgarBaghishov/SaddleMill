import numpy as np
from ase.optimize import BFGS, FIRE, LBFGS, MDMin
from ase.io import read
from ase.mep import NEB, NEBTools
from tsearch.catsunami.ocpneb import OCPNEB
from tsearch.catsunami.autoframe import interpolate
from fairchem.core import FAIRChemCalculator


relax_endpoints = True
neb_method = "improvedtangent"
interpolate_method = "ocp_idpp"  # this is idpp implementation from Meta OCP, other choises are "ase_idpp" and "ase_linear" or False if you already have a frame set
fmax = 0.05
k = 5
num_frames = 10

device = "cuda"
model_name_or_path = "uma-s-1p1"
task_name = "oc20"

calc = FAIRChemCalculator.from_model_checkpoint(model_name_or_path, task_name, device=device)

if not interpolate_method:
    """
    The approach uses ase, so you must provide a list of ase.Atoms objects
    with the appropriate constraints.
    """
    # file_path = "../../../datasets/all_dft_neb_trajs/transfer_id_0_2342_21_2-1-1-3_neb1.traj"
    file_path = "../../../datasets/all_dft_neb_trajs/dissociation_ood_595_3492_21_211-2_neb1.traj"
    # file_path = "../../../datasets/all_dft_neb_trajs/desorption_id_106_6686_7_111-3_neb1.traj"
    images = read(file_path, f"0:{num_frames}")  # Change to the path to your atoms of the frame set
    reactant = images[0]
    product = images[-1]
else:
    reactant = read("optimized_reactant.vasp")
    product = read("optimized_product.vasp")

reactant.calc = FAIRChemCalculator.from_model_checkpoint(model_name_or_path, task_name, device=device)
product.calc = FAIRChemCalculator.from_model_checkpoint(model_name_or_path, task_name, device=device)

if relax_endpoints:
    if not interpolate_method: print("Are you sure you want to relax end points while keeping the intermediate inages from your traj?")
    opt = BFGS(reactant, trajectory='reactant_relaxation.traj')
    opt.run(0.05, 300)
    opt = BFGS(product, trajectory='product_relaxation.traj')
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

    neb0 = NEB(
        images,
        k=k,
        climb=True,
        method=neb_method,
        allow_shared_calculator=False,
        dynamic_relaxation=False,
    )
    neb0.interpolate(method=interpolate_method[4:], mic=True)

for image in images[1:-1]:
    image.calc = calc

neb = OCPNEB(
    images,
    k=k,
    climb=True,
    method=neb_method,
    allow_shared_calculator=True,
    dynamic_relaxation=False,
    batch_size=8, # If you get a memory error, try reducing it to 4
)

optimizer = MDMin(neb, dt=0.02, maxstep=0.1, trajectory=f"your-neb.traj")
conv = optimizer.run(fmax=fmax, steps=500)

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