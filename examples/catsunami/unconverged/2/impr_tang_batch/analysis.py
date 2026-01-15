import matplotlib.pyplot as plt
from fairchem.core import FAIRChemCalculator
from ase.io import read
from ase.mep import NEBTools
import numpy as np

images = read('your-neb.traj', '-10:')

calc = FAIRChemCalculator.from_model_checkpoint("uma-s-1p1", "oc20", device="cuda")

for image in images:
    image.calc = calc

nebtools = NEBTools(images)
Ef, dE = nebtools.get_barrier()

# Get the actual maximum force at this point in the simulation.
max_force = nebtools.get_fmax(
    k=5,
    climb=True,
    method="aseneb",
    allow_shared_calculator=True,
    dynamic_relaxation=False,
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