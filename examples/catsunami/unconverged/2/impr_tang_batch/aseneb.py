import numpy as np
from ase.optimize import BFGS, FIRE, LBFGS, MDMin
from ase.io import read
from ase.mep import NEB, NEBTools
from tsearch.catsunami.ocpneb import OCPNEB
from tsearch.catsunami.autoframe import interpolate
from fairchem.core import FAIRChemCalculator


relax_endpoints = True
neb_method = "improvedtangent"
interpolate_method = "ase_linear"  # this is idpp implementation from Meta OCP, other choises are "ase_idpp" and "ase_linear" or False if you already have a frame set
fmax = 0.05
k = 5
num_frames = 10

device = "cuda"
name_or_path = "uma-s-1p1"
task_name = "oc20"

calc = FAIRChemCalculator.from_model_checkpoint(name_or_path, task_name, device=device)

reactant = read("optimized_reactant.vasp")
product = read("optimized_product.vasp")

reactant.calc = calc
product.calc = calc

if relax_endpoints:
    if not interpolate_method: print("Are you sure you want to relax end points while keeping the intermediate inages from your traj?")
    opt = BFGS(reactant, trajectory='reactant_relaxation.traj')
    opt.run(0.05, 300)
    opt = BFGS(product, trajectory='product_relaxation.traj')
    opt.run(0.05, 300)


images = [reactant]
images += [reactant.copy() for i in range(num_frames-2)]
images += [product]

neb0 = NEB(
    images,
    k=k,
    climb=True,
    method=neb_method,
    allow_shared_calculator=False,
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
    batch_size=8, # If you get a memory error, try reducing it to 4
)

optimizer = MDMin(neb, dt=0.02, maxstep=0.1, logfile="neb.log", trajectory=f"your-neb.traj")
conv = optimizer.run(fmax=fmax, steps=3000)

# optimizer = MDMin(neb, dt=0.02, maxstep=0.1, trajectory=f"your-neb.traj")
# conv = optimizer.run(fmax=fmax + delta_fmax_climb, steps=500)
# if conv:
#     print("initial NEB optimization is done, starting climbing image")
#     neb.climb = True
#     conv = optimizer.run(fmax=fmax, steps=1000)


# Final analysis
nebtools = NEBTools(neb.images)
Ef, dE = nebtools.get_barrier()

# Get the actual maximum force at this point in the simulation.
max_force = nebtools.get_fmax(
    k=k,
    climb=True,
    method=neb_method,
    allow_shared_calculator=True,
    )

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