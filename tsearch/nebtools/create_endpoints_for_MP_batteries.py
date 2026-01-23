import numpy as np
import random
from ase.io import Trajectory
from ase.build import make_supercell

def get_variants(atoms):
    w_ion = atoms.info['working_ion']
    
    monovalents = ['Li', 'Na', 'K', 'Rb', 'Cs']
    divalents = ['Mg', 'Ca', 'Zn']
    trivalent = ['Al', 'Y']
    targets = []

    if w_ion in monovalents:
        targets = monovalents
    elif w_ion in divalents:
        targets = monovalents + divalents
    elif w_ion in trivalent:
        targets = monovalents + divalents + trivalent
    else:
        targets = monovalents + [w_ion]

    variants = []
    for new_ion in targets:
        new_atoms = atoms.copy()
        # Update symbols
        for atom in new_atoms:
            if atom.symbol == w_ion:
                atom.symbol = new_ion
        new_atoms.info['working_ion'] = new_ion
        variants.append(new_atoms)
    
    return variants

def process_structure(atoms_orig, old_w_ion):
    w_ion = atoms_orig.info['working_ion']
    w_indices = [a.index for a in atoms_orig if a.symbol == w_ion]
    n_ions = len(w_indices)

    # Supercell Logic
    M = [1, 1, 1]
    lengths = atoms_orig.cell.lengths()
    
    if n_ions == 1:
        # Double in 2 smallest directions
        sorted_dirs = np.argsort(lengths)
        M[sorted_dirs[0]] = 2
        M[sorted_dirs[1]] = 2
    elif n_ions == 2:
        # Double in shortest direction
        M[np.argmin(lengths)] = 2
    
    if M != [1, 1, 1]:
        # make_supercell loses .info dictionary. Save it first.
        original_info = atoms_orig.info.copy()
        atoms_orig = make_supercell(atoms_orig, np.diag(M))
        atoms_orig.info = original_info # Restore info
        w_indices = [a.index for a in atoms_orig if a.symbol == w_ion]

    if len(w_indices) < 2:
        return # Cannot create pair if fewer than 2 ions exist even after supercell

    # NEB Pair Generation
    idx_A_list = random.sample(w_indices, 2)
    for idx_A in idx_A_list:
        atoms = atoms_orig.copy()
        idx_A_orig = idx_A
        pos_A = atoms.positions[idx_A].copy()
        
        # Find closest working ion (B)
        min_dist = float('inf')
        idx_B = -1
        
        for idx in w_indices:
            if idx == idx_A:
                continue
            dist = atoms.get_distance(idx_A, idx, mic=True)
            if dist < min_dist:
                min_dist = dist
                idx_B = idx
                
        pos_B = atoms.positions[idx_B].copy()

        # Identify candidates for removal (excluding the active pair A and B)
        removable = [i for i in w_indices if i not in (idx_A, idx_B)]
        to_remove = []
        
        if removable:
            # Pick random quantity to remove (0 to all available)
            n_remove = random.randint(0, len(removable))
            to_remove = random.sample(removable, n_remove)
            
            # Update indices of A and B because deleting atoms shifts indices
            idx_A -= sum(1 for x in to_remove if x < idx_A)
            idx_B -= sum(1 for x in to_remove if x < idx_B)
            
            # Delete in reverse sorted order to preserve indices during deletion
            del atoms[sorted(to_remove, reverse=True)]

        # Create base structure with A removed
        del atoms[idx_A]
        
        # Handle index shift for B
        # If B was after A, its index decreased by 1
        idx_B_new = idx_B - 1 if idx_B > idx_A else idx_B

        img1 = atoms.copy()
        img2 = atoms.copy()
        img1.info["removed_ion_idxs"] = to_remove

        # Image 1: B stays at pos_B (already there)
        # Image 2: B moves to pos_A
        img2.positions[idx_B_new] = pos_A

        filename = f"{atoms.info.get('discharge_id', 'struct')}_{old_w_ion}_{w_ion}_{idx_A_orig}.traj"
        with Trajectory(filename, 'w') as traj_out:
            traj_out.write(img1)
            traj_out.write(img2)

# Main Execution
traj_reader = Trajectory('../../MP_batteries_fully_ionated_structures.traj', 'r')

count = 0
for atoms in traj_reader:
    print(count)
    variants = get_variants(atoms)
    orig_ion = atoms.info['working_ion']
    for var in variants:
        process_structure(var, orig_ion)
    count += 1