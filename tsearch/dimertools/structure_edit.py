import warnings
import numpy as np
import random
from ase import Atom
from ase.constraints import FixAtoms
from ase.neighborlist import NeighborList, natural_cutoffs, neighbor_list, mic
from ase.build import make_supercell
from ase.data import covalent_radii, atomic_numbers


def turn_into_supercell(atoms, min_length=7.0):
    """Ensure sufficient atoms AND cell dimensions to avoid self-interaction."""
    n_atoms = len(atoms)
    lengths = atoms.cell.lengths()
    M = [1, 1, 1]

    # Rule 1: Minimum atoms
    if n_atoms == 1:
        M = [3, 3, 3]
    elif n_atoms <= 4:
        M = [2, 2, 2]
    elif 5 <= n_atoms <= 8:
        sorted_indices = np.argsort(lengths)
        M[sorted_indices[0]] = 2
        M[sorted_indices[1]] = 2
    elif 9 <= n_atoms <= 16:
        M[np.argmin(lengths)] = 2

    # Rule 2: Minimum length (ensures no self-interaction via PBC)
    for i in range(3):
        if lengths[i] > 1e-6 and lengths[i] * M[i] < min_length:
            M[i] = int(np.ceil(min_length / lengths[i]))

    if M != [1, 1, 1]:
        saved_info = dict(atoms.info)
        atoms = make_supercell(atoms, np.diag(M))
        atoms.info.update(saved_info)
    return atoms


def find_interstitial_sites(atoms, min_dist_frac=0.4):
    """Find interstitial sites using Voronoi tessellation on periodic images.

    Parameters
    ----------
    atoms : ase.Atoms
        Structure to find interstitial sites in.
    min_dist_frac : float
        Minimum distance from any atom as a fraction of the nearest-neighbor
        distance. Sites closer than this are discarded.

    Returns
    -------
    sites : np.ndarray, shape (N_sites, 3)
        Cartesian coordinates of interstitial sites.
    """
    from scipy.spatial import Voronoi
    from scipy.cluster.hierarchy import fcluster, linkage

    cell = atoms.get_cell()
    positions = atoms.get_positions()

    # Build 3x3x3 periodic images
    shifts = np.array(
        [[i, j, k] for i in range(-1, 2) for j in range(-1, 2) for k in range(-1, 2)]
    )
    all_positions = np.vstack([positions + s @ cell for s in shifts])

    if len(all_positions) < 4:
        return np.empty((0, 3))

    vor = Voronoi(all_positions)
    vertices = vor.vertices

    # Keep vertices inside the unit cell (fractional coords in [0, 1))
    inv_cell = np.linalg.inv(cell)
    frac_coords = vertices @ inv_cell
    inside = np.all((frac_coords >= -1e-6) & (frac_coords < 1.0 - 1e-6), axis=1)
    candidates = vertices[inside]

    if len(candidates) == 0:
        return np.empty((0, 3))

    # Compute nearest-neighbor distance in the original structure
    if len(atoms) >= 2:
        _, _, d = neighbor_list('ijd', atoms, 5.0)
        nn_dist = d.min() if len(d) > 0 else 2.5
    else:
        nn_dist = 2.5

    min_dist = min_dist_frac * nn_dist

    # Filter: keep sites far enough from any real atom (vectorized)
    # deltas shape: (N_sites, N_atoms, 3) → flatten for mic, then restore
    deltas = (candidates[:, None, :] - positions[None, :, :]).reshape(-1, 3)
    min_dists = np.linalg.norm(mic(deltas, cell).reshape(len(candidates), len(positions), 3), axis=-1).min(axis=1)
    candidates = candidates[min_dists > min_dist]

    if len(candidates) == 0:
        return np.empty((0, 3))

    # Cluster nearby sites within 0.5 Å
    if len(candidates) > 1:
        Z = linkage(candidates, method='single')
        labels = fcluster(Z, t=0.5, criterion='distance')
        unique_labels = np.unique(labels)
        candidates = np.array([candidates[labels == lbl].mean(axis=0) for lbl in unique_labels])

    return candidates


def _safe_normalize(vec):
    """Normalize a vector, returning a random unit vector if norm is near zero."""
    norm = np.linalg.norm(vec)
    if norm < 1e-12:
        vec = np.random.randn(3)
        norm = np.linalg.norm(vec)
    return vec / norm


def _nearest_site(site_a, other_sites, cell):
    """Return the nearest site from other_sites to site_a under MIC."""
    deltas = mic(other_sites - site_a, cell)
    dists = np.linalg.norm(deltas, axis=1)
    idx = np.argmin(dists)
    return other_sites[idx], deltas[idx]


def _get_atom_selection_weights(atoms):
    """Return probability weights for atom selection based on inverse covalent radii."""
    numbers = atoms.get_atomic_numbers()
    weights = np.array([1.0 / covalent_radii[z] for z in numbers])
    weights /= weights.sum()
    return weights


def _shuffled_site_indices(n_sites, n_attempts):
    """Return n_attempts site indices cycling through a shuffled list (no repeats per cycle)."""
    indices = list(range(n_sites))
    random.shuffle(indices)
    return [indices[i % n_sites] for i in range(n_attempts)]


def _maybe_gaussian(disp_dict, center_idx, p=0.1):
    """With probability p, replace a directed displacement with ASE Gaussian noise.

    Returning {"displacement_center": center_idx} tells ASE's MinModeAtoms to apply
    a random Gaussian displacement centred on that atom, which lets the dimer discover
    reaction types that the directed guess might miss.
    """
    if random.random() < p:
        return {"displacement_center": int(center_idx)}
    return disp_dict


# --- Element sampling pools ---

_HOP_INSERT_ELEMENTS = [1, 6, 7, 8, 5]  # H, C, N, O, B (small interstitial species)
_HOP_INSERT_WEIGHTS = np.array([1.0 / covalent_radii[z] for z in _HOP_INSERT_ELEMENTS])
_HOP_INSERT_WEIGHTS /= _HOP_INSERT_WEIGHTS.sum()


def _sample_hop_insert_element():
    """Sample a common small interstitial element weighted by inverse covalent radius."""
    idx = np.random.choice(len(_HOP_INSERT_ELEMENTS), p=_HOP_INSERT_WEIGHTS)
    return _HOP_INSERT_ELEMENTS[idx]


# Common metallic / bulk elements for kickout_insert (similar-sized substitutional candidates)
_KICKOUT_INSERT_POOL = [
    'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn',
    'Zr', 'Nb', 'Mo', 'Ru', 'Rh', 'Pd', 'Ag',
    'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au',
    'Al', 'Si', 'Ge', 'Sn', 'Ga', 'In',
]
_KICKOUT_INSERT_POOL_Z = [atomic_numbers[s] for s in _KICKOUT_INSERT_POOL]
_KICKOUT_INSERT_POOL_RADII = np.array([covalent_radii[z] for z in _KICKOUT_INSERT_POOL_Z])


def _sample_kickout_insert_element(atoms, sigma=0.2):
    """Sample an element with covalent radius similar to the host atoms.

    Uses a Gaussian weight: exp(-(r_candidate - r_host_avg)^2 / (2*sigma^2)).
    """
    host_radii = np.array([covalent_radii[z] for z in atoms.get_atomic_numbers()])
    r_avg = host_radii.mean()
    weights = np.exp(-(_KICKOUT_INSERT_POOL_RADII - r_avg) ** 2 / (2 * sigma ** 2))
    if weights.sum() < 1e-12:
        weights = np.ones_like(weights)
    weights /= weights.sum()
    idx = np.random.choice(len(_KICKOUT_INSERT_POOL_Z), p=weights)
    return _KICKOUT_INSERT_POOL_Z[idx]


# --- Vacancy attempts ---

def get_vacancy_attempts(atoms, config_dict, num_attempts):
    """Vacancy-mediated diffusion with three sub-mechanisms sampled with equal probability:

    0. NN hop: a nearest-neighbor atom hops directly into the vacancy.
    1. NNN hop: a second-nearest-neighbor atom hops directly into the vacancy.
    2. Concerted 2-atom chain: NN hops into the vacancy while its NNN simultaneously
       hops into the NN's original site.

    All mechanisms displace atoms halfway along the hop vector (good saddle-point guess).
    With 10% probability, the directed displacement is replaced by Gaussian noise centred
    on the primary atom (allows the dimer to discover unexpected reaction paths).
    """
    cell = atoms.get_cell()
    i_idx, j_idx = neighbor_list('ij', atoms, 3.5)

    num_attempts = min(num_attempts, len(atoms))
    remove_indices = random.sample(range(len(atoms)), num_attempts)

    images = []
    displacement_dicts = []
    selected_indices = []

    for rm_idx in remove_indices:
        vacancy_pos = atoms.positions[rm_idx].copy()

        nn_indices = list(j_idx[i_idx == rm_idx])
        if len(nn_indices) == 0:
            nn_indices = [x for x in range(len(atoms)) if x != rm_idx]

        mechanism = random.randint(0, 2)

        if mechanism == 0:
            # Direct NN hop: NN atom moves into the vacancy.
            chosen_nn = random.choice(nn_indices)
            nn_pos = atoms.positions[chosen_nn].copy()

            atoms_new = atoms.copy()
            del atoms_new[rm_idx]
            new_nn_idx = chosen_nn if chosen_nn < rm_idx else chosen_nn - 1

            disp_vector = np.zeros((len(atoms_new), 3))
            disp_vector[new_nn_idx] = 0.5 * mic(vacancy_pos - nn_pos, cell)

            atoms_new.info['reaction_type'] = 'vacancy'
            images.append(atoms_new)
            displacement_dicts.append(_maybe_gaussian(
                {"displacement_vector": disp_vector, "method": "vector"}, new_nn_idx))
            selected_indices.append(rm_idx)

        elif mechanism == 1:
            # NNN hop: NNN atom hops directly toward the vacancy.
            nn_set = set(nn_indices)
            nnn_pairs = [
                (int(nnn), int(nn))
                for nn in nn_indices
                for nnn in j_idx[i_idx == nn]
                if nnn != rm_idx and nnn not in nn_set
            ]

            if len(nnn_pairs) == 0:
                # Fallback to NN hop when no NNN exists (tiny cells, etc.)
                chosen_nn = random.choice(nn_indices)
                nn_pos = atoms.positions[chosen_nn].copy()

                atoms_new = atoms.copy()
                del atoms_new[rm_idx]
                new_nn_idx = chosen_nn if chosen_nn < rm_idx else chosen_nn - 1

                disp_vector = np.zeros((len(atoms_new), 3))
                disp_vector[new_nn_idx] = 0.5 * mic(vacancy_pos - nn_pos, cell)
                atoms_new.info['reaction_type'] = 'vacancy'
                images.append(atoms_new)
                displacement_dicts.append(_maybe_gaussian(
                    {"displacement_vector": disp_vector, "method": "vector"}, new_nn_idx))
                selected_indices.append(rm_idx)
                continue

            chosen_nnn, _ = random.choice(nnn_pairs)
            nnn_pos = atoms.positions[chosen_nnn].copy()

            atoms_new = atoms.copy()
            del atoms_new[rm_idx]
            new_nnn_idx = chosen_nnn if chosen_nnn < rm_idx else chosen_nnn - 1

            disp_vector = np.zeros((len(atoms_new), 3))
            # NNN hops directly toward the vacancy (not toward via-NN)
            disp_vector[new_nnn_idx] = 0.5 * mic(vacancy_pos - nnn_pos, cell)

            atoms_new.info['reaction_type'] = 'vacancy'
            images.append(atoms_new)
            displacement_dicts.append(_maybe_gaussian(
                {"displacement_vector": disp_vector, "method": "vector"}, new_nnn_idx))
            selected_indices.append(rm_idx)

        else:  # mechanism == 2
            # Concerted 2-atom chain: NN→vacancy AND NNN→NN simultaneously.
            chosen_nn = random.choice(nn_indices)
            nn_pos = atoms.positions[chosen_nn].copy()

            nn_set = set(nn_indices)
            nnn_candidates = [int(n) for n in j_idx[i_idx == chosen_nn]
                              if n != rm_idx and n not in nn_set]

            atoms_new = atoms.copy()
            del atoms_new[rm_idx]
            new_nn_idx = chosen_nn if chosen_nn < rm_idx else chosen_nn - 1

            disp_vector = np.zeros((len(atoms_new), 3))
            disp_vector[new_nn_idx] = 0.5 * mic(vacancy_pos - nn_pos, cell)

            if len(nnn_candidates) > 0:
                chosen_nnn = random.choice(nnn_candidates)
                nnn_pos = atoms.positions[chosen_nnn].copy()
                new_nnn_idx = chosen_nnn if chosen_nnn < rm_idx else chosen_nnn - 1
                disp_vector[new_nnn_idx] = 0.5 * mic(nn_pos - nnn_pos, cell)

            atoms_new.info['reaction_type'] = 'vacancy'
            images.append(atoms_new)
            displacement_dicts.append(_maybe_gaussian(
                {"displacement_vector": disp_vector, "method": "vector"}, new_nn_idx))
            selected_indices.append(rm_idx)

    return images, displacement_dicts, selected_indices


# --- Hop attempts (interstitial mechanism) ---

def get_hop_reuse_attempts(atoms, num_attempts):
    """Displace an existing lattice atom halfway toward its nearest interstitial site.

    No atoms are added or removed. The directed displacement gives the dimer a
    physically motivated starting point for the interstitial hop transition state.
    With 10% probability, Gaussian noise is used instead to allow discovery of
    unexpected paths.
    """
    sites = find_interstitial_sites(atoms)

    if len(sites) == 0:
        warnings.warn("Found no interstitial sites; skipping hop_reuse.")
        return [], [], []

    cell = atoms.get_cell()
    positions = atoms.get_positions()
    weights = _get_atom_selection_weights(atoms)
    site_idx_list = _shuffled_site_indices(len(sites), num_attempts)

    images = []
    displacement_dicts = []
    selected_indices = []

    for attempt in range(num_attempts):
        atom_idx = np.random.choice(len(atoms), p=weights)
        atom_pos = positions[atom_idx]

        # Use pre-shuffled site for diversity across attempts
        site_a = sites[site_idx_list[attempt]]
        delta = mic(site_a - atom_pos, cell)

        atoms_new = atoms.copy()
        atoms_new.info['reaction_type'] = 'hop_reuse'

        disp_vector = np.zeros((len(atoms_new), 3))
        disp_vector[atom_idx] = 0.5 * delta

        images.append(atoms_new)
        displacement_dicts.append(_maybe_gaussian(
            {"displacement_vector": disp_vector, "method": "vector"}, atom_idx))
        selected_indices.append(int(atom_idx))

    return images, displacement_dicts, selected_indices


def get_hop_insert_attempts(atoms, num_attempts):
    """Insert a new small atom at an interstitial site, displace halfway to nearest neighbor site.

    With 10% probability, Gaussian noise is used instead.
    """
    sites = find_interstitial_sites(atoms)

    if len(sites) < 2:
        warnings.warn("Found fewer than 2 interstitial sites; skipping hop_insert.")
        return [], [], []

    cell = atoms.get_cell()
    site_idx_list = _shuffled_site_indices(len(sites), num_attempts)

    images = []
    displacement_dicts = []
    selected_indices = []

    for attempt in range(num_attempts):
        element_z = _sample_hop_insert_element()

        site_a_idx = site_idx_list[attempt]
        site_a = sites[site_a_idx]

        other_sites = np.delete(sites, site_a_idx, axis=0)
        site_b, delta_ab = _nearest_site(site_a, other_sites, cell)

        atoms_new = atoms.copy()
        atoms_new.append(Atom(element_z, position=site_a))
        new_atom_idx = len(atoms_new) - 1
        atoms_new.info['reaction_type'] = 'hop_insert'

        disp_vector = np.zeros((len(atoms_new), 3))
        disp_vector[new_atom_idx] = 0.5 * delta_ab

        images.append(atoms_new)
        displacement_dicts.append(_maybe_gaussian(
            {"displacement_vector": disp_vector, "method": "vector"}, new_atom_idx))
        selected_indices.append(int(new_atom_idx))

    return images, displacement_dicts, selected_indices


# --- Kickout attempts (interstitialcy / kick-out mechanism) ---

def get_kickout_reuse_attempts(atoms, num_attempts):
    """Interstitialcy kick-out using only existing atoms (no atoms added or removed).

    For each attempt:
    1. Pick an interstitial site A.
    2. The atom nearest to site A is the 'kicked' atom — it will be displaced toward
       site B (nearest other interstitial site), as if pushed out of the way.
    3. A separate 'kicker' atom (randomly selected, weighted by inverse covalent radius,
       excluding the kicked atom) is displaced toward site A.

    The original lattice structure is preserved in frame 0; only the displacement
    vectors encode the mechanism. With 10% probability, Gaussian noise is used instead.
    """
    sites = find_interstitial_sites(atoms)

    if len(sites) < 2:
        warnings.warn("Found fewer than 2 interstitial sites; skipping kickout_reuse.")
        return [], [], []

    cell = atoms.get_cell()
    positions = atoms.get_positions()
    weights = _get_atom_selection_weights(atoms)
    site_idx_list = _shuffled_site_indices(len(sites), num_attempts)

    images = []
    displacement_dicts = []
    selected_indices = []

    for attempt in range(num_attempts):
        site_a_idx = site_idx_list[attempt]
        site_a = sites[site_a_idx]

        # Sort all atoms by distance to site A.
        # Nearest  → kicked (gets displaced when the kicker arrives at site A).
        # 2nd nearest → kicker (hops into site A, physically close so displacement is sensible).
        dists_to_a = np.linalg.norm(mic(positions - site_a, cell), axis=1)
        sorted_by_dist = np.argsort(dists_to_a)
        kicked_idx = int(sorted_by_dist[0])
        kicker_idx = int(sorted_by_dist[1])
        kicked_pos = positions[kicked_idx]
        kicker_pos = positions[kicker_idx]

        # Site B: nearest interstitial to kicked atom (exclude site A)
        other_sites = np.delete(sites, site_a_idx, axis=0)
        site_b, _ = _nearest_site(kicked_pos, other_sites, cell)

        atoms_new = atoms.copy()
        atoms_new.info['reaction_type'] = 'kickout_reuse'

        disp_vector = np.zeros((len(atoms_new), 3))
        disp_vector[kicker_idx] = 0.5 * mic(site_a - kicker_pos, cell)
        disp_vector[kicked_idx] = 0.5 * mic(site_b - kicked_pos, cell)

        images.append(atoms_new)
        displacement_dicts.append(_maybe_gaussian(
            {"displacement_vector": disp_vector, "method": "vector"}, kicker_idx))
        selected_indices.append(int(kicker_idx))

    return images, displacement_dicts, selected_indices


def get_kickout_insert_attempts(atoms, num_attempts):
    """Insert a new similar-sized atom at interstitial site; it kicks nearest lattice atom out.

    1. Sample a new element with covalent radius similar to host (Gaussian weight).
    2. Insert it at interstitial site A.
    3. Find the nearest lattice atom — this is the atom being kicked.
    4. Inserted atom displaced halfway toward kicked atom's position.
    5. Kicked atom displaced halfway toward site B.
    With 10% probability, Gaussian noise is used instead.
    """
    sites = find_interstitial_sites(atoms)

    if len(sites) < 2:
        warnings.warn("Found fewer than 2 interstitial sites; skipping kickout_insert.")
        return [], [], []

    cell = atoms.get_cell()
    positions = atoms.get_positions()
    site_idx_list = _shuffled_site_indices(len(sites), num_attempts)

    images = []
    displacement_dicts = []
    selected_indices = []

    for attempt in range(num_attempts):
        element_z = _sample_kickout_insert_element(atoms)

        site_a_idx = site_idx_list[attempt]
        site_a = sites[site_a_idx]

        # Kicked: nearest lattice atom to site A
        dists_to_a = np.linalg.norm(mic(positions - site_a, cell), axis=1)
        kicked_idx = int(np.argmin(dists_to_a))
        kicked_pos = positions[kicked_idx]

        other_sites = np.delete(sites, site_a_idx, axis=0)
        site_b, _ = _nearest_site(kicked_pos, other_sites, cell)

        atoms_new = atoms.copy()
        atoms_new.append(Atom(element_z, position=site_a))
        inserted_idx = len(atoms_new) - 1
        atoms_new.info['reaction_type'] = 'kickout_insert'

        disp_vector = np.zeros((len(atoms_new), 3))
        disp_vector[inserted_idx] = 0.5 * mic(kicked_pos - site_a, cell)
        disp_vector[kicked_idx] = 0.5 * mic(site_b - kicked_pos, cell)

        images.append(atoms_new)
        displacement_dicts.append(_maybe_gaussian(
            {"displacement_vector": disp_vector, "method": "vector"}, inserted_idx))
        selected_indices.append(int(inserted_idx))

    return images, displacement_dicts, selected_indices


# --- Ring attempts (coordinated position swaps; ring_size=2 covers pairwise exchange) ---

def _find_ring(neighbors_dict, seed, ring_size, max_retries=50):
    """Find a ring of connected atoms starting from seed.

    For ring_size=2, simply pick a random neighbor.
    For ring_size>=3, random walk through neighbors, where the last atom
    must be a neighbor of seed to close the ring.

    Returns list of ring_size unique atom indices forming the ring, or None.
    """
    if ring_size == 2:
        nbrs = neighbors_dict.get(seed, [])
        if len(nbrs) == 0:
            return None
        return [seed, random.choice(nbrs)]

    seed_neighbors = set(neighbors_dict.get(seed, []))

    for _ in range(max_retries):
        path = [seed]
        for step in range(ring_size - 1):
            current = path[-1]
            nbrs = neighbors_dict.get(current, [])
            if step == ring_size - 2:
                # Last step: must close the ring back to seed
                candidates = [n for n in nbrs if n not in path and n in seed_neighbors]
            else:
                candidates = [n for n in nbrs if n not in path]
            if not candidates:
                break
            path.append(random.choice(candidates))
        else:
            return path

    return None


def _build_neighbor_dict(atoms, cutoff=3.5):
    """Build a dict mapping atom index -> list of neighbor indices."""
    i_idx, j_idx = neighbor_list('ij', atoms, cutoff)
    neighbors_dict = {}
    for i, j in zip(i_idx, j_idx):
        if i == j: continue  # FIX: Skip self-interactions
        neighbors_dict.setdefault(int(i), []).append(int(j))
    return neighbors_dict


def _make_ring_attempt(atoms, neighbors_dict, cell, ring_size, reaction_type):
    """Create a single ring swap attempt. Returns (image, disp_dict, index) or None."""
    seed = random.randrange(len(atoms))
    ring = _find_ring(neighbors_dict, seed, ring_size)

    if ring is None:
        warnings.warn(f"Could not find ring of size {ring_size}; skipping one {reaction_type} attempt.")
        return None

    atoms_new = atoms.copy()
    positions = atoms_new.get_positions()
    disp_vector = np.zeros_like(positions)

    if len(ring) == 2:
        # Exchange: atoms head directly toward each other along the same line.
        # Add a perpendicular component so they dodge to OPPOSITE sides — otherwise
        # both atoms end up at the same midpoint and overlap.
        delta = mic(positions[ring[1]] - positions[ring[0]], cell)
        delta_hat = _safe_normalize(delta)
        ref = np.array([1.0, 0.0, 0.0])
        if abs(np.dot(delta_hat, ref)) > 0.9:
            ref = np.array([0.0, 1.0, 0.0])
        perp = np.cross(delta_hat, ref)
        perp = (_safe_normalize(perp)) * 0.15 * np.linalg.norm(delta) * random.choice([-1, 1])
        # ring[0] gets +perp, ring[1] gets -perp → they separate at the midpoint
        disp_vector[ring[0]] = 0.5 * delta + perp
        disp_vector[ring[1]] = -0.5 * delta - perp
    else:
        for k in range(len(ring)):
            src = ring[k]
            dst = ring[(k + 1) % len(ring)]
            disp_vector[src] = 0.5 * mic(positions[dst] - positions[src], cell)

    atoms_new.info['reaction_type'] = reaction_type
    disp_dict = {"displacement_vector": disp_vector, "method": "vector"}
    return (atoms_new, _maybe_gaussian(disp_dict, int(ring[0])), int(ring[0]))


def get_ring_attempts(atoms, config_dict, num_attempts):
    """Cooperative ring rotation. Ring size is randomly sampled from config ring_sizes.

    ring_sizes = 2 3 4   # includes 2 → also covers pairwise exchange
    ring_sizes = 3 4     # only true rings (no exchange)
    """
    ring_sizes = config_dict["ourDimer"].get("ring_sizes", [3, 4])
    if isinstance(ring_sizes, (int, float)):
        ring_sizes = [int(ring_sizes)]
    elif isinstance(ring_sizes, str):
        ring_sizes = [int(x) for x in ring_sizes.split()]
    else:
        ring_sizes = [int(x) for x in ring_sizes]

    if not ring_sizes:
        return [], [], []

    neighbors_dict = _build_neighbor_dict(atoms)
    cell = atoms.get_cell()

    images, displacement_dicts, selected_indices = [], [], []
    for _ in range(num_attempts):
        size = random.choice(ring_sizes)
        result = _make_ring_attempt(atoms, neighbors_dict, cell, size, 'ring')
        if result:
            images.append(result[0])
            displacement_dicts.append(result[1])
            selected_indices.append(result[2])
    return images, displacement_dicts, selected_indices


# --- Initial guess (no displacement) ---

def get_initial_guess_attempts(atoms):
    """Return the structure as-is with a negligible displacement for eigenmode initialization.

    Used when the input geometry is already a good TS guess and only dimer
    refinement (rotation + translation) is needed — no random perturbation.
    Always produces exactly 1 attempt.
    """
    atoms_new = atoms.copy()
    atoms_new.info['reaction_type'] = 'initial_guess'
    disp_vector = np.random.randn(len(atoms_new), 3) * 1e-10
    return [atoms_new], [{"displacement_vector": disp_vector, "method": "vector"}], [-1]


# --- Main dispatch ---

_REACTION_TYPE_DISPATCH = {
    "vacancy": lambda atoms, config_dict, n: get_vacancy_attempts(atoms, config_dict, n),
    "hop_reuse": lambda atoms, config_dict, n: get_hop_reuse_attempts(atoms, n),
    "hop_insert": lambda atoms, config_dict, n: get_hop_insert_attempts(atoms, n),
    "kickout_reuse": lambda atoms, config_dict, n: get_kickout_reuse_attempts(atoms, n),
    "kickout_insert": lambda atoms, config_dict, n: get_kickout_insert_attempts(atoms, n),
    "ring": lambda atoms, config_dict, n: get_ring_attempts(atoms, config_dict, n),
    "initial_guess": lambda atoms, config_dict, n: get_initial_guess_attempts(atoms),
}


def get_attempts(atoms, config_dict):

    atoms = atoms.copy()

    # Centralized supercell expansion (controlled by config, default True)
    if config_dict["ourDimer"].get("supercell", True):
        atoms = turn_into_supercell(atoms)

    # --- Handle initial_guess early (works for both bulk and oc) ---
    reaction_types = config_dict["ourDimer"].get("reaction_types")
    if isinstance(reaction_types, str):
        reaction_types_list = reaction_types.split() if ' ' in reaction_types else [reaction_types]
    elif isinstance(reaction_types, list):
        reaction_types_list = reaction_types
    else:
        reaction_types_list = []

    if "initial_guess" in reaction_types_list:
        other_types = [rt for rt in reaction_types_list if rt != "initial_guess"]
        if other_types:
            warnings.warn(f"'initial_guess' is exclusive — ignoring other reaction types: {other_types}")

        dataset_type = config_dict["ourDimer"]["dataset_type"]
        if dataset_type == "bulk":
            num_per_type = config_dict["ourDimer"].get("num_attempts_per_type", 1)
            if num_per_type is not None and num_per_type > 1:
                warnings.warn(f"'initial_guess' always produces 1 attempt — ignoring num_attempts_per_type={num_per_type}")
        elif dataset_type == "oc":
            num_attempts = config_dict["ourDimer"].get("num_attempts", 1)
            if num_attempts is not None and num_attempts > 1:
                warnings.warn(f"'initial_guess' always produces 1 attempt — ignoring num_attempts={num_attempts}")
            # Apply OC constraints (fix substrate atoms)
            tags = atoms.get_tags()
            indices = np.where(tags == 0)[0]
            atoms.set_constraint(FixAtoms(indices=indices))

        return get_initial_guess_attempts(atoms)

    # --- Normal dispatch ---

    images = []
    displacement_dicts = []
    selected_indices = []

    if config_dict["ourDimer"]["dataset_type"] == "bulk":

        if reaction_types is None:
            raise ValueError("Configuration error: 'ourDimer' -> 'reaction_types' is not set. "
                             "Please specify reaction types (e.g., 'vacancy') in config.ini")

        num_per_type = config_dict["ourDimer"].get("num_attempts_per_type", 1)

        for rtype in reaction_types_list:
            if rtype not in _REACTION_TYPE_DISPATCH:
                supported = ", ".join(_REACTION_TYPE_DISPATCH.keys())
                raise ValueError(f"Unknown reaction type: '{rtype}'. "
                                 f"Supported types: {supported}")
            imgs, dds, idxs = _REACTION_TYPE_DISPATCH[rtype](atoms, config_dict, num_per_type)
            images.extend(imgs)
            displacement_dicts.extend(dds)
            selected_indices.extend(idxs)

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

        if len(adsorbate_indices) == 0:
            warnings.warn("No adsorbate atoms (tag 1 or 2) found for 'oc' dataset; skipping.")
            return [], [], []

        neighbor_indices = set()
        for idx in adsorbate_indices:
            indices, offsets = nl.get_neighbors(idx)
            neighbor_indices.update(indices)
        unique_neighbors = np.array(list(neighbor_indices))
        mask = np.zeros(len(atoms), dtype=bool)
        mask[unique_neighbors] = True
        mask = mask.tolist()

        num_needed = config_dict["ourDimer"]["num_attempts"] // 3
        if len(adsorbate_indices) >= num_needed:
            chosen_indices = random.sample(list(adsorbate_indices), num_needed)
        else:
            chosen_indices = list(adsorbate_indices) * int(num_needed // len(adsorbate_indices))
            remainder = num_needed % len(adsorbate_indices)
            chosen_indices.extend(random.sample(list(adsorbate_indices), remainder))

        for i in range(config_dict["ourDimer"]["num_attempts"]):
            images.append(atoms.copy())

            if i < num_needed:
                displacement_dicts.append({"displacement_center": int(chosen_indices[i])})
                selected_indices.append(int(chosen_indices[i]))
            elif num_needed <= i < 2 * num_needed:
                displacement_dicts.append({"displacement_center": int(chosen_indices[i - num_needed]), "gauss_std": 0.2, "number_of_atoms": 1})
                selected_indices.append(int(chosen_indices[i - num_needed]))
            else:
                displacement_dicts.append({"mask": mask})
                selected_indices.append(-1)

    else:
        raise Exception("dataset_type in ourDimer must be set")

    return images, displacement_dicts, selected_indices
