import glob
from ase.io import read, write
import numpy as np
from ase.optimize import BFGS
from ase.build import sort
from fairchem.core import FAIRChemCalculator

calc = FAIRChemCalculator.from_model_checkpoint("uma-s-1p1", "oc20", device="cuda")

traj_file = glob.glob("*.traj")[0]
band = read(traj_file, "0:10")

reactant = band[0]
print(reactant)
reactant.calc = calc
opt = BFGS(reactant)
opt.run(0.01, 100)
print(reactant)

product = band[-1]
print(product)
product.calc = calc
opt = BFGS(product)
opt.run(0.01, 100)
print(product)

print(f"Reactant max force is {np.max(reactant.get_forces())}")
print(f"Product max force is {np.max(product.get_forces())}")

write("reactant.vasp", reactant, format='vasp')
write("product.vasp", product, format='vasp')

reactant.rotate(reactant.cell[0], 'x', rotate_cell=True)
angle = np.arctan2(reactant.cell[1][2], reactant.cell[1][1])
reactant.rotate(-np.degrees(angle), 'x', rotate_cell=True)
print(reactant)

product.rotate(product.cell[0], 'x', rotate_cell=True)
angle = np.arctan2(product.cell[1][2], product.cell[1][1])
product.rotate(-np.degrees(angle), 'x', rotate_cell=True)
print(product)

print(f"Reactant max force is {np.max(reactant.get_forces())}")
print(f"Product max force is {np.max(product.get_forces())}")

write("rotated_reactant.vasp", reactant, format='vasp')
write("rotated_product.vasp", product, format='vasp')

sorted_reactant = sort(reactant)
sorted_reactant.calc = calc
print(sorted_reactant)
sorted_product = sort(product)
sorted_product.calc = calc
print(sorted_product)

print(f"Reactant max force is {np.max(sorted_reactant.get_forces())}")
print(f"Product max force is {np.max(sorted_product.get_forces())}")

write("sorted_reactant.vasp", sorted_reactant, format='vasp')
write("sorted_product.vasp", sorted_product, format='vasp')
