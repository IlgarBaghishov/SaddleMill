from ase.optimize import BFGS, FIRE, LBFGS, MDMin
from ase.io import read
from ase.mep import NEB
from fairchem.core import FAIRChemCalculator


fmax = 0.001
k = 5
num_frames = 10
device = "cuda"
model_name_or_path = "uma-s-1p1"
task_name = "oc20"

reactant = read("optimized_reactant.vasp")
product = read("optimized_product.vasp")

images = [reactant]
images += [reactant.copy() for i in range(num_frames-2)]
images += [product]

neb = NEB(
    images,
    k=k,
    climb=True,
    method="improvedtangent",
    allow_shared_calculator=False,
    dynamic_relaxation=False,
)

neb.interpolate()

for image in images:
    image.calc = FAIRChemCalculator.from_model_checkpoint(model_name_or_path, task_name, device=device)

optimizer = MDMin(neb, dt=0.02, maxstep=0.1, trajectory=f"your-neb.traj")
conv = optimizer.run(fmax=fmax, steps=500)

