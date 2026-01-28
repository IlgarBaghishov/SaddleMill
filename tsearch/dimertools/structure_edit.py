import numpy as np
import random
from ase.constraints import FixAtoms
from ase.neighborlist import NeighborList, natural_cutoffs, neighbor_list
from ase.build import make_supercell


def turn_into_supercell(atoms):
    n_atoms = len(atoms)
    M = [1, 1, 1]
    if n_atoms == 1: M = [3, 3, 3]
    elif n_atoms <= 4: M = [2, 2, 2]
    elif 5 <= n_atoms <= 8:
        lengths = atoms.cell.lengths()
        M[np.argsort(lengths)[0]] = 2
        M[np.argsort(lengths)[1]] = 2
    elif 9 <= n_atoms <= 16:
        M[np.argmin(atoms.cell.lengths())] = 2
    if M != [1, 1, 1]:
        atoms = make_supercell(atoms, np.diag(M))
    return atoms


def get_attempts(atoms, config_dict):

    atoms = atoms.copy()

    images = []
    displacement_dicts = []
    selected_indices = []

    if config_dict["ourDimer"]["dataset_type"] == "bulk":

        atoms = turn_into_supercell(atoms)
        i_idx, j_idx = neighbor_list('ij', atoms, 3.5)

        remove_indices = random.sample(range(len(atoms)), config_dict["ourDimer"]["num_attempts"])
        selected_indices = remove_indices

        for rm_idx in remove_indices:

            neighbor_indices = j_idx[i_idx == rm_idx]
            if len(neighbor_indices) == 0:
                neighbor_indices = [x for x in range(len(atoms)) if x != rm_idx]

            chosen_neighbor = random.choice(neighbor_indices)

            atoms_new = atoms.copy()
            del atoms_new[rm_idx]
            images.append(atoms_new)

            new_center_idx = chosen_neighbor if chosen_neighbor < rm_idx else chosen_neighbor - 1
            displacement_dicts.append({"displacement_center": int(new_center_idx)})

    elif config_dict["ourDimer"]["dataset_type"] == "oc":

        tags = atoms.get_tags()
        indices = np.where(tags == 0)[0]
        atoms.set_constraint(FixAtoms(indices=indices))

        cutoffs = natural_cutoffs(atoms, mult=1.25)
        nl = NeighborList(cutoffs, self_interaction=False, bothways=True)
        nl.update(atoms)

        adsorbate_indices = np.where(tags == 2)[0]
        if adsorbate_indices.shape[0] == 0:
            adsorbate_indices = np.where(tags == 1)[0]

        neighbor_indices = set()
        for idx in adsorbate_indices:
            indices, offsets = nl.get_neighbors(idx)
            neighbor_indices.update(indices)
        unique_neighbors = np.array(list(neighbor_indices))
        mask = np.zeros(len(atoms), dtype=bool)
        mask[unique_neighbors] = True
        mask = mask.tolist()

        num_needed = config_dict["ourDimer"]["num_attempts"]//3
        if len(adsorbate_indices) >= num_needed:
            chosen_indices = random.sample(list(adsorbate_indices), num_needed)
        else:
            chosen_indices = list(adsorbate_indices) * int(num_needed//len(adsorbate_indices))
            remainder = num_needed % len(adsorbate_indices)
            chosen_indices.extend(random.sample(list(adsorbate_indices), remainder))

        for i in range(config_dict["ourDimer"]["num_attempts"]):
            images.append(atoms.copy())

            if i < num_needed:
                displacement_dicts.append({"displacement_center": int(chosen_indices[i])})
                selected_indices.append(int(chosen_indices[i]))
            elif num_needed <= i < 2*num_needed:
                displacement_dicts.append({"displacement_center": int(chosen_indices[i-num_needed]), "gauss_std":0.2, "number_of_atoms":1})
                selected_indices.append(int(chosen_indices[i-num_needed]))
            else:
                displacement_dicts.append({"mask": mask})
                selected_indices.append(-1)

    else:
        raise Exception("dataset_type in ourDimer must be set")

    return images, displacement_dicts, selected_indices
