import warnings
import numpy as np
import random
from ase import Atom
from ase.constraints import FixAtoms
from ase.neighborlist import NeighborList, natural_cutoffs, neighbor_list, mic
from ase.build import make_supercell
from ase.data import covalent_radii, atomic_numbers

REUSE_MAX_DIST = 5.0  # Å — max hop/collision distance for *_reuse; tighten to taste


def _is_gauss_slot(g, p_percent):
    """Deterministic, evenly-spread ~p fraction of global attempt indices."""
    if p_percent <= 0:
        return False
    if p_percent >= 100:
        return True
    return (g * p_percent) % 100 < p_percent


def _gauss_count_before(g, p_percent):
    """How many global indices in [0, g) are gaussian slots.

    Lets the directed pointer skip gaussian slots so they never consume a
    ranked pair. Pattern has period 100 in g, so count whole blocks + remainder.
    """
    if p_percent <= 0:
        return 0
    if p_percent >= 100:
        return g
    per_block = sum(1 for k in range(100) if (k * p_percent) % 100 < p_percent)
    full_blocks, rem = divmod(g, 100)
    partial = sum(1 for k in range(rem) if (k * p_percent) % 100 < p_percent)
    return full_blocks * per_block + partial

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

def _swap_prob(config_dict):
    """Probability of swapping a directed displacement for ASE Gaussian noise.
    Reads [ourDimer] gaussian_swap_prob; default 0.1 reproduces the old hardcoded
    value. 0.0 disables the swap entirely, 1.0 always uses Gaussian noise.
    """
    if config_dict is None:
        return 0.1
    return config_dict["ourDimer"].get("gaussian_swap_prob", 0.1)

def _concentrate_params(config_dict):
    """(prob, power, std) for power-law-concentrated Gaussian swaps.
    concentrate_prob=0 (default) disables the feature — no behavior change."""
    d = config_dict["ourDimer"]
    return (float(d.get("concentrate_prob", 0.0)),
            float(d.get("concentrate_power", 1.5)),
            float(d.get("concentrate_std", 0.2)))


def _displacement_radius(config_dict, default=3.0):
    if config_dict is None:
        return default
    dc = config_dict.get("DimerControl", {}) or {}
    try:
        return float(dc.get("displacement_radius", default))
    except (TypeError, ValueError):
        return default


def _movable_oc(atoms):
    """OC movable atoms (tag != 0); substrate (tag 0) is fixed and excluded."""
    return set(int(i) for i in np.where(atoms.get_tags() != 0)[0])


def _atoms_within_radius(atoms, center_idx, radius):
    cell = atoms.get_cell()
    deltas = mic(atoms.get_positions() - atoms.positions[center_idx], cell)
    return np.where(np.linalg.norm(deltas, axis=1) <= radius)[0]


def _power_law_vector(natoms, eligible_indices, power, std):
    """Concentrated random displacement: iid Gaussian on the eligible atoms,
    each atom's magnitude raised to `power` (ratios sharpen), then renormalized
    so the total norm equals std*sqrt(3*n_eligible) — the same total an iid
    Gaussian of this std would inject, just redistributed. power=1 == plain."""
    d = np.zeros((natoms, 3))
    idx = np.asarray(sorted(set(int(i) for i in eligible_indices)), dtype=int)
    if len(idx) == 0:
        return d
    d[idx] = np.random.standard_normal((len(idx), 3))
    mag = np.linalg.norm(d[idx], axis=1, keepdims=True)
    d[idx] = np.where(mag > 1e-12, d[idx] * mag ** (power - 1), 0.0)
    total = np.linalg.norm(d)
    if total > 1e-12:
        d *= (std * np.sqrt(3 * len(idx))) / total
    return d


def _maybe_concentrate(disp_dict, eligible_indices, natoms, config_dict):
    """With probability concentrate_prob, replace an ASE Gaussian dict with an
    explicit power-law-concentrated vector. Returns (dict, was_concentrated)."""
    q, power, std = _concentrate_params(config_dict)
    if q <= 0.0 or random.random() >= q:
        return disp_dict, False
    vec = _power_law_vector(natoms, eligible_indices, power, std)
    return {"displacement_vector": vec, "method": "vector"}, True

def _resolve_attempts_per_type(num_per_type, reaction_types_list):
    """Map num_attempts_per_type onto a per-type count list.

    - int or None  -> original behavior: same count for every type.
    - list         -> new behavior: one count per type, aligned by position;
                      must match the number of reaction types.
    """
    if isinstance(num_per_type, list):
        counts = [int(x) for x in num_per_type]
        if len(counts) != len(reaction_types_list):
            raise ValueError(
                f"[ourDimer] num_attempts_per_type has {len(counts)} values but "
                f"reaction_types has {len(reaction_types_list)} entries. Give one "
                f"count per type (aligned by position), or a single integer for all.")
        return counts
    n = int(num_per_type) if num_per_type is not None else 1
    return [n] * len(reaction_types_list)


def _attempt_count_max(num_per_type):
    """Largest attempt count implied by num_attempts_per_type (int or list)."""
    if isinstance(num_per_type, list):
        return max((int(x) for x in num_per_type), default=1)
    return int(num_per_type) if num_per_type is not None else 1

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
    p = _swap_prob(config_dict)
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
                {"displacement_vector": disp_vector, "method": "vector"}, new_nn_idx, p=p))
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
                    {"displacement_vector": disp_vector, "method": "vector"}, new_nn_idx, p=p))
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
                {"displacement_vector": disp_vector, "method": "vector"}, new_nnn_idx, p=p))
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
                {"displacement_vector": disp_vector, "method": "vector"}, new_nn_idx, p=p))
            selected_indices.append(rm_idx)

    return images, displacement_dicts, selected_indices


# --- Hop attempts (interstitial mechanism) ---

def get_hop_reuse_attempts(atoms, num_attempts, config_dict=None):
    """Displace an existing lattice atom halfway toward an interstitial site.

    (atom, site) pairs are ranked by dist*radius (short hops of small atoms
    first) and consumed deterministically in order. Pairs farther apart than
    REUSE_MAX_DIST are discarded. A deterministic ~p fraction of attempts are
    swapped for a Gaussian displacement; those swaps do NOT advance the ranked
    pointer, so they never steal a directed guess. If the ranked list is
    exhausted, remaining attempts fall back to Gaussian.
    """
    sites = find_interstitial_sites(atoms)
    if len(sites) == 0:
        warnings.warn("Found no interstitial sites; skipping hop_reuse.")
        return [None] * num_attempts, [None] * num_attempts, [-1] * num_attempts

    cell = atoms.get_cell()
    p = _swap_prob(config_dict)
    p_percent = int(round(p * 100))
    positions = atoms.get_positions()
    numbers = atoms.get_atomic_numbers()

    offset = 0
    if config_dict is not None and "ourDimer" in config_dict:
        offset = int(config_dict["ourDimer"].get("bulk_reuse_offset", 0))

    # 1. Score every in-range (atom, site) pair; rank best (lowest) first.
    scored_pairs = []
    for atom_idx, atom_pos in enumerate(positions):
        radius = covalent_radii[numbers[atom_idx]]
        for site_pos in sites:
            delta = mic(site_pos - atom_pos, cell)
            dist = np.linalg.norm(delta)
            if dist > REUSE_MAX_DIST:
                continue
            scored_pairs.append((dist * radius, atom_idx, delta))
    scored_pairs.sort(key=lambda x: x[0])
    n_valid = len(scored_pairs)
    if n_valid == 0:
        warnings.warn("hop_reuse: no (atom, site) pairs within REUSE_MAX_DIST; "
                      "using Gaussian fallback.")

    images, displacement_dicts, selected_indices = [], [], []

    for i in range(num_attempts):
        g = offset + i
        ptr = g - _gauss_count_before(g, p_percent)
        use_gauss = (ptr >= n_valid) or _is_gauss_slot(g, p_percent)

        atoms_new = atoms.copy()

        if use_gauss:
            # Center on the next directed atom if one exists, else a random atom.
            if n_valid > 0:
                center_idx = scored_pairs[ptr % n_valid][1]
            else:
                center_idx = random.randrange(len(atoms))
            atoms_new.info['reaction_type'] = 'hop_reuse_gauss'
            images.append(atoms_new)
            displacement_dicts.append({"displacement_center": int(center_idx)})
            selected_indices.append(int(center_idx))
        else:
            _, atom_idx, delta = scored_pairs[ptr]
            atoms_new.info['reaction_type'] = 'hop_reuse'
            disp_vector = np.zeros((len(atoms_new), 3))
            disp_vector[atom_idx] = 0.5 * delta
            images.append(atoms_new)
            displacement_dicts.append({"displacement_vector": disp_vector, "method": "vector"})
            selected_indices.append(int(atom_idx))

    return images, displacement_dicts, selected_indices


def get_hop_insert_attempts(atoms, num_attempts, config_dict=None):
    """Insert a new small atom at an interstitial site, displace halfway to nearest neighbor site.

    With 10% probability, Gaussian noise is used instead.
    """
    sites = find_interstitial_sites(atoms)

    if len(sites) < 2:
        warnings.warn("Found fewer than 2 interstitial sites; skipping hop_insert.")
        return [None] * num_attempts, [None] * num_attempts, [-1] * num_attempts

    cell = atoms.get_cell()
    p = _swap_prob(config_dict)
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
            {"displacement_vector": disp_vector, "method": "vector"}, new_atom_idx, p=p))
        selected_indices.append(int(new_atom_idx))

    return images, displacement_dicts, selected_indices


# --- Kickout attempts (interstitialcy / kick-out mechanism) ---

def get_kickout_reuse_attempts(atoms, num_attempts, config_dict=None):
    """Interstitialcy knock-on (kick-out) using only existing atoms.

    Triplets (void=site_b, kicked, kicker) are ranked and consumed
    deterministically. A triplet is dropped if the kicked atom is farther than
    REUSE_MAX_DIST from the void, or the kicker is farther than REUSE_MAX_DIST
    from the kicked atom. Gaussian swaps do NOT advance the ranked pointer;
    exhaustion falls back to Gaussian.
    """
    sites = find_interstitial_sites(atoms)
    if len(sites) < 2:
        warnings.warn("Found fewer than 2 interstitial sites; skipping kickout_reuse.")
        return [None] * num_attempts, [None] * num_attempts, [-1] * num_attempts

    cell = atoms.get_cell()
    p = _swap_prob(config_dict)
    p_percent = int(round(p * 100))
    positions = atoms.get_positions()
    numbers = atoms.get_atomic_numbers()

    offset = 0
    if config_dict is not None and "ourDimer" in config_dict:
        offset = int(config_dict["ourDimer"].get("bulk_reuse_offset", 0))

    # 1. Score in-range triplets (true kick-out); rank best (lowest) first.
    scored_triplets = []
    for site_b in sites:
        dists_to_void = np.linalg.norm(mic(positions - site_b, cell), axis=1)
        kicked_idx = int(np.argmin(dists_to_void))
        dist_void = dists_to_void[kicked_idx]
        if dist_void > REUSE_MAX_DIST:
            continue
        kicked_pos = positions[kicked_idx]

        dists_to_kicked = np.linalg.norm(mic(positions - kicked_pos, cell), axis=1)
        sorted_by_dist_to_kicked = np.argsort(dists_to_kicked)

        for kicker_idx in sorted_by_dist_to_kicked[1:6]:  # skip self at index 0
            kicker_idx = int(kicker_idx)
            dist_collision = dists_to_kicked[kicker_idx]
            if dist_collision > REUSE_MAX_DIST:
                continue
            r_kicked = covalent_radii[numbers[kicked_idx]]
            r_kicker = covalent_radii[numbers[kicker_idx]]
            score = (dist_void * r_kicked) + (dist_collision * r_kicker)
            scored_triplets.append((score, site_b, kicked_idx, kicker_idx))
    scored_triplets.sort(key=lambda x: x[0])
    n_valid = len(scored_triplets)
    if n_valid == 0:
        warnings.warn("kickout_reuse: no triplets within REUSE_MAX_DIST; "
                      "using Gaussian fallback.")

    images, displacement_dicts, selected_indices = [], [], []

    for i in range(num_attempts):
        g = offset + i
        ptr = g - _gauss_count_before(g, p_percent)
        use_gauss = (ptr >= n_valid) or _is_gauss_slot(g, p_percent)

        atoms_new = atoms.copy()

        if use_gauss:
            if n_valid > 0:
                center_idx = scored_triplets[ptr % n_valid][3]  # kicker
            else:
                center_idx = random.randrange(len(atoms))
            atoms_new.info['reaction_type'] = 'kickout_reuse_gauss'
            images.append(atoms_new)
            displacement_dicts.append({"displacement_center": int(center_idx)})
            selected_indices.append(int(center_idx))
        else:
            _, site_b, kicked_idx, kicker_idx = scored_triplets[ptr]
            kicked_pos = positions[kicked_idx]
            kicker_pos = positions[kicker_idx]
            atoms_new.info['reaction_type'] = 'kickout_reuse'
            disp_vector = np.zeros((len(atoms_new), 3))
            disp_vector[kicker_idx] = 0.5 * mic(kicked_pos - kicker_pos, cell)
            disp_vector[kicked_idx] = 0.5 * mic(site_b - kicked_pos, cell)
            images.append(atoms_new)
            displacement_dicts.append({"displacement_vector": disp_vector, "method": "vector"})
            selected_indices.append(int(kicker_idx))

    return images, displacement_dicts, selected_indices


def get_kickout_insert_attempts(atoms, num_attempts, config_dict=None):
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
        return [None] * num_attempts, [None] * num_attempts, [-1] * num_attempts

    cell = atoms.get_cell()
    p = _swap_prob(config_dict)
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
            {"displacement_vector": disp_vector, "method": "vector"}, inserted_idx, p=p))
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


def _make_ring_attempt(atoms, neighbors_dict, cell, ring_size, reaction_type, p=0.1):
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
    return (atoms_new, _maybe_gaussian(disp_dict, int(ring[0]), p=p), int(ring[0]))


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
        return [None] * num_attempts, [None] * num_attempts, [-1] * num_attempts

    neighbors_dict = _build_neighbor_dict(atoms)
    cell = atoms.get_cell()
    p = _swap_prob(config_dict)

    images, displacement_dicts, selected_indices = [], [], []
    for _ in range(num_attempts):
        size = random.choice(ring_sizes)
        result = _make_ring_attempt(atoms, neighbors_dict, cell, size, 'ring', p=p)
        if result:
            images.append(result[0])
            displacement_dicts.append(result[1])
            selected_indices.append(result[2])
        else:
            images.append(None)
            displacement_dicts.append(None)
            selected_indices.append(-1)
    return images, displacement_dicts, selected_indices


# --- OC helpers ---

def _get_oc_adsorbate_indices(atoms):
    """Return indices of adsorbate atoms (tag=2)."""
    tags = atoms.get_tags()
    return np.where(tags == 2)[0]


def _get_oc_neighbor_mask(atoms, adsorbate_indices):
    """Return a boolean list mask of all atoms neighboring the adsorbate."""
    cutoffs = natural_cutoffs(atoms, mult=1.25)
    nl = NeighborList(cutoffs, self_interaction=False, bothways=True)
    nl.update(atoms)

    neighbor_indices = set()
    for idx in adsorbate_indices:
        indices, offsets = nl.get_neighbors(idx)
        neighbor_indices.update(indices)
    unique_neighbors = np.array(list(neighbor_indices))
    mask = np.zeros(len(atoms), dtype=bool)
    mask[unique_neighbors] = True
    return mask.tolist()


def _sample_adsorbate_atoms(adsorbate_indices, num_needed):
    """Sample num_needed adsorbate atom indices, cycling if needed."""
    if len(adsorbate_indices) >= num_needed:
        return random.sample(list(adsorbate_indices), num_needed)
    else:
        chosen = list(adsorbate_indices) * (num_needed // len(adsorbate_indices))
        remainder = num_needed % len(adsorbate_indices)
        chosen.extend(random.sample(list(adsorbate_indices), remainder))
        return chosen


# --- OC reaction types ---

def get_adsorbate_attempts(atoms, config_dict, num_attempts):
    """Noise on all adsorbate atoms; concentrate_prob swaps in a power-law kick."""
    adsorbate_indices = _get_oc_adsorbate_indices(atoms)
    if len(adsorbate_indices) == 0:
        warnings.warn("No adsorbate atoms (tag=2) found; skipping 'adsorbate'.")
        return [None] * num_attempts, [None] * num_attempts, [-1] * num_attempts

    mask = np.zeros(len(atoms), dtype=bool)
    mask[adsorbate_indices] = True
    mask = mask.tolist()
    eligible = [int(i) for i in adsorbate_indices]

    images, displacement_dicts, selected_indices = [], [], []
    for _ in range(num_attempts):
        atoms_new = atoms.copy()
        disp, conc = _maybe_concentrate({"mask": mask}, eligible, len(atoms_new), config_dict)
        atoms_new.info['reaction_type'] = 'adsorbate_conc' if conc else 'adsorbate'
        images.append(atoms_new)
        displacement_dicts.append(disp)
        selected_indices.append(-1)
    return images, displacement_dicts, selected_indices


def get_adsorbate_surface_attempts(atoms, config_dict, num_attempts):
    """Noise on adsorbate + neighboring substrate; concentrate over movable atoms."""
    adsorbate_indices = _get_oc_adsorbate_indices(atoms)
    if len(adsorbate_indices) == 0:
        warnings.warn("No adsorbate atoms (tag=2) found; skipping 'adsorbate_surface'.")
        return [None] * num_attempts, [None] * num_attempts, [-1] * num_attempts

    mask = _get_oc_neighbor_mask(atoms, adsorbate_indices)
    movable = _movable_oc(atoms)
    eligible = [i for i, m in enumerate(mask) if m and i in movable]

    images, displacement_dicts, selected_indices = [], [], []
    for _ in range(num_attempts):
        atoms_new = atoms.copy()
        disp, conc = _maybe_concentrate({"mask": mask}, eligible, len(atoms_new), config_dict)
        atoms_new.info['reaction_type'] = 'adsorbate_surface_conc' if conc else 'adsorbate_surface'
        images.append(atoms_new)
        displacement_dicts.append(disp)
        selected_indices.append(-1)
    return images, displacement_dicts, selected_indices


def get_adsorbate_atom_neighbors_attempts(atoms, config_dict, num_attempts):
    """Broad Gaussian on one adsorbate atom, neighbors dragged."""
    adsorbate_indices = _get_oc_adsorbate_indices(atoms)
    if len(adsorbate_indices) == 0:
        warnings.warn("No adsorbate atoms (tag=2) found; skipping 'adsorbate_atom_neighbors'.")
        return [None] * num_attempts, [None] * num_attempts, [-1] * num_attempts

    q = _concentrate_params(config_dict)[0]
    radius = _displacement_radius(config_dict)
    movable = _movable_oc(atoms)

    chosen = _sample_adsorbate_atoms(adsorbate_indices, num_attempts)
    images, displacement_dicts, selected_indices = [], [], []
    for idx in chosen:
        atoms_new = atoms.copy()
        disp = {"displacement_center": int(idx)}
        if q > 0.0:
            near = _atoms_within_radius(atoms_new, int(idx), radius)
            eligible = [int(i) for i in near if int(i) in movable]
        else:
            eligible = [int(idx)]
        disp, conc = _maybe_concentrate(disp, eligible, len(atoms_new), config_dict)
        atoms_new.info['reaction_type'] = 'adsorbate_atom_neighbors_conc' if conc else 'adsorbate_atom_neighbors'
        images.append(atoms_new)
        displacement_dicts.append(disp)
        selected_indices.append(int(idx))
    return images, displacement_dicts, selected_indices


def get_surface_attempts(atoms, config_dict, num_attempts):
    """Broad Gaussian on one surface atom (tag=1), neighbors dragged."""
    tags = atoms.get_tags()
    surface_indices = np.where(tags == 1)[0]
    if len(surface_indices) == 0:
        warnings.warn("No surface atoms (tag=1) found; skipping 'surface'.")
        return [None] * num_attempts, [None] * num_attempts, [-1] * num_attempts

    q = _concentrate_params(config_dict)[0]
    radius = _displacement_radius(config_dict)
    movable = _movable_oc(atoms)

    chosen = _sample_adsorbate_atoms(surface_indices, num_attempts)
    images, displacement_dicts, selected_indices = [], [], []
    for idx in chosen:
        atoms_new = atoms.copy()
        disp = {"displacement_center": int(idx)}
        if q > 0.0:
            near = _atoms_within_radius(atoms_new, int(idx), radius)
            eligible = [int(i) for i in near if int(i) in movable]
        else:
            eligible = [int(idx)]
        disp, conc = _maybe_concentrate(disp, eligible, len(atoms_new), config_dict)
        atoms_new.info['reaction_type'] = 'surface_conc' if conc else 'surface'
        images.append(atoms_new)
        displacement_dicts.append(disp)
        selected_indices.append(int(idx))
    return images, displacement_dicts, selected_indices
    
def get_adsorbate_atom_attempts(atoms, config_dict, num_attempts):
    """Tight Gaussian on one adsorbate atom (gauss_std=0.2, single atom)."""
    adsorbate_indices = _get_oc_adsorbate_indices(atoms)
    if len(adsorbate_indices) == 0:
        warnings.warn("No adsorbate atoms (tag=2) found; skipping 'adsorbate_atom'.")
        return [None] * num_attempts, [None] * num_attempts, [-1] * num_attempts

    chosen = _sample_adsorbate_atoms(adsorbate_indices, num_attempts)
    images, displacement_dicts, selected_indices = [], [], []
    for idx in chosen:
        atoms_new = atoms.copy()
        atoms_new.info['reaction_type'] = 'adsorbate_atom'
        images.append(atoms_new)
        displacement_dicts.append({"displacement_center": int(idx), "gauss_std": 0.2, "number_of_atoms": 1})
        selected_indices.append(int(idx))
    return images, displacement_dicts, selected_indices

def get_diffusion_attempts(atoms, config_dict, num_attempts):
    """Uniform translation of all adsorbate atoms in a random 3D direction."""
    adsorbate_indices = _get_oc_adsorbate_indices(atoms)
    if len(adsorbate_indices) == 0:
        warnings.warn("No adsorbate atoms (tag=2) found; skipping 'diffusion'.")
        return [None] * num_attempts, [None] * num_attempts, [-1] * num_attempts

    images, displacement_dicts, selected_indices = [], [], []
    for _ in range(num_attempts):
        atoms_new = atoms.copy()
        atoms_new.info['reaction_type'] = 'diffusion'

        direction = np.random.randn(3)
        direction /= np.linalg.norm(direction)

        disp = np.zeros((len(atoms), 3))
        disp[adsorbate_indices] = direction * 0.1
        displacement_dicts.append({"displacement_vector": disp, "method": "vector"})
        selected_indices.append(-1)
        images.append(atoms_new)
    return images, displacement_dicts, selected_indices

def get_all_movable_attempts(atoms, config_dict, num_attempts):
    """Noise on all movable atoms (tag != 0); concentrate_prob swaps in a power-law kick."""
    movable = _movable_oc(atoms)
    if not movable:
        warnings.warn("No movable atoms (tag != 0) found; skipping 'all_movable'.")
        return [None] * num_attempts, [None] * num_attempts, [-1] * num_attempts

    # Boolean mask for standard Gaussian noise if concentrate_prob fails
    mask = [i in movable for i in range(len(atoms))]
    eligible = list(movable)

    images, displacement_dicts, selected_indices = [], [], []
    for _ in range(num_attempts):
        atoms_new = atoms.copy()
        
        # This will return a power-law vector if concentrate_prob triggers, 
        # otherwise it returns the standard boolean mask for ASE to use.
        disp, conc = _maybe_concentrate({"mask": mask}, eligible, len(atoms_new), config_dict)
        
        atoms_new.info['reaction_type'] = 'all_movable_conc' if conc else 'all_movable'
        images.append(atoms_new)
        displacement_dicts.append(disp)
        selected_indices.append(-1)
        
    return images, displacement_dicts, selected_indices

def get_rotation_attempts(atoms, config_dict, num_attempts):
    """Rigid-body rotation of adsorbate around its center of mass."""
    adsorbate_indices = _get_oc_adsorbate_indices(atoms)
    if len(adsorbate_indices) == 0:
        warnings.warn("No adsorbate atoms (tag=2) found; skipping 'rotation'.")
        return [None] * num_attempts, [None] * num_attempts, [-1] * num_attempts
    if len(adsorbate_indices) < 2:
        warnings.warn("Rotation requires at least 2 adsorbate atoms; skipping.")
        return [None] * num_attempts, [None] * num_attempts, [-1] * num_attempts

    positions = atoms.get_positions()
    masses = atoms.get_masses()
    ads_positions = positions[adsorbate_indices]
    ads_masses = masses[adsorbate_indices]
    com = np.average(ads_positions, weights=ads_masses, axis=0)

    images, displacement_dicts, selected_indices = [], [], []
    for _ in range(num_attempts):
        atoms_new = atoms.copy()
        atoms_new.info['reaction_type'] = 'rotation'

        axis = np.random.randn(3)
        axis /= np.linalg.norm(axis)
        angle = 0.05  # radians

        disp = np.zeros((len(atoms), 3))
        for idx in adsorbate_indices:
            r = positions[idx] - com
            disp[idx] = angle * np.cross(axis, r)

        displacement_dicts.append({"displacement_vector": disp, "method": "vector"})
        selected_indices.append(-1)
        images.append(atoms_new)
    return images, displacement_dicts, selected_indices

def get_custom_attempts(atoms, config_dict, num_attempts):
    """No overrides — displacement fully controlled by [DimerControl] settings."""
    images, displacement_dicts, selected_indices = [], [], []
    for _ in range(num_attempts):
        atoms_new = atoms.copy()
        atoms_new.info['reaction_type'] = 'custom'
        images.append(atoms_new)
        displacement_dicts.append({})
        selected_indices.append(-1)
    return images, displacement_dicts, selected_indices


# --- Initial guess (no displacement) ---

def get_initial_guess_attempts(atoms):
    """Return the structure as-is with a negligible displacement for eigenmode initialization.

    Used when the input geometry is already a good TS guess and only dimer
    refinement (rotation + translation) is needed — no random perturbation.
    Always produces exactly 1 attempt.

    If the input atoms carry an eigenmode (in atoms.info['eigenmode'] or
    atoms.info['orig_info']['eigenmode']), it is preserved in the output so
    that dimeropt can seed the dimer with it instead of a random guess.
    """
    atoms_new = atoms.copy()
    atoms_new.info['reaction_type'] = 'initial_guess'

    # Propagate eigenmode from orig_info if present
    orig = atoms.info.get('orig_info', {})
    if 'eigenmode' in orig:
        atoms_new.info['eigenmode'] = np.array(orig['eigenmode'])

    disp_vector = np.random.randn(len(atoms_new), 3) * 1e-10
    return [atoms_new], [{"displacement_vector": disp_vector, "method": "vector"}], [-1]


# --- Main dispatch ---

_BULK_REACTION_TYPE_DISPATCH = {
    "vacancy": lambda atoms, config_dict, n: get_vacancy_attempts(atoms, config_dict, n),
    "hop_reuse": lambda atoms, config_dict, n: get_hop_reuse_attempts(atoms, n, config_dict),
    "hop_insert": lambda atoms, config_dict, n: get_hop_insert_attempts(atoms, n, config_dict),
    "kickout_reuse": lambda atoms, config_dict, n: get_kickout_reuse_attempts(atoms, n, config_dict),
    "kickout_insert": lambda atoms, config_dict, n: get_kickout_insert_attempts(atoms, n, config_dict),
    "ring": lambda atoms, config_dict, n: get_ring_attempts(atoms, config_dict, n),
    "initial_guess": lambda atoms, config_dict, n: get_initial_guess_attempts(atoms),
}

_OC_REACTION_TYPE_DISPATCH = {
    "all_movable": lambda atoms, config_dict, n: get_all_movable_attempts(atoms, config_dict, n),
    "adsorbate_atom": lambda atoms, config_dict, n: get_adsorbate_atom_attempts(atoms, config_dict, n),
    "adsorbate_atom_neighbors": lambda atoms, config_dict, n: get_adsorbate_atom_neighbors_attempts(atoms, config_dict, n),
    "adsorbate": lambda atoms, config_dict, n: get_adsorbate_attempts(atoms, config_dict, n),
    "diffusion": lambda atoms, config_dict, n: get_diffusion_attempts(atoms, config_dict, n),
    "rotation": lambda atoms, config_dict, n: get_rotation_attempts(atoms, config_dict, n),
    "adsorbate_surface": lambda atoms, config_dict, n: get_adsorbate_surface_attempts(atoms, config_dict, n),
    "surface": lambda atoms, config_dict, n: get_surface_attempts(atoms, config_dict, n),
    "custom": lambda atoms, config_dict, n: get_custom_attempts(atoms, config_dict, n),
    "initial_guess": lambda atoms, config_dict, n: get_initial_guess_attempts(atoms),
}

# Backward-compat alias
_REACTION_TYPE_DISPATCH = _BULK_REACTION_TYPE_DISPATCH


def get_attempts(atoms, config_dict):

    atoms = atoms.copy()

    # --- Handle initial_guess early (no supercell, works for both bulk and oc) ---
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
            if _attempt_count_max(num_per_type) > 1:
                warnings.warn(f"'initial_guess' always produces 1 attempt — ignoring num_attempts_per_type={num_per_type}")
        elif dataset_type == "oc":
            num_per_type = config_dict["ourDimer"].get("num_attempts_per_type", 1)
            if _attempt_count_max(num_per_type) > 1:
                warnings.warn(f"'initial_guess' always produces 1 attempt — ignoring num_attempts_per_type={num_per_type}")
            # Apply OC constraints (fix substrate atoms)
            tags = atoms.get_tags()
            indices = np.where(tags == 0)[0]
            atoms.set_constraint(FixAtoms(indices=indices))

        return get_initial_guess_attempts(atoms)

    # Centralized supercell expansion (controlled by config, default True)
    if config_dict["ourDimer"].get("supercell", True):
        atoms = turn_into_supercell(atoms)

    # --- Normal dispatch ---

    images = []
    displacement_dicts = []
    selected_indices = []

    if config_dict["ourDimer"]["dataset_type"] == "bulk":

        if reaction_types is None:
            raise ValueError("Configuration error: 'ourDimer' -> 'reaction_types' is not set. "
                             "Please specify reaction types (e.g., 'vacancy') in config.ini")

        num_per_type = config_dict["ourDimer"].get("num_attempts_per_type", 1)
        counts = _resolve_attempts_per_type(num_per_type, reaction_types_list)

        for rtype, n_attempts in zip(reaction_types_list, counts):
            if rtype not in _BULK_REACTION_TYPE_DISPATCH:
                supported = ", ".join(_BULK_REACTION_TYPE_DISPATCH.keys())
                raise ValueError(f"Unknown bulk reaction type: '{rtype}'. "
                                 f"Supported types: {supported}")
            imgs, dds, idxs = _BULK_REACTION_TYPE_DISPATCH[rtype](atoms, config_dict, n_attempts)
            images.extend(imgs)
            displacement_dicts.extend(dds)
            selected_indices.extend(idxs)

    elif config_dict["ourDimer"]["dataset_type"] == "oc":

        # Fix substrate atoms (tag=0)
        tags = atoms.get_tags()
        substrate_indices = np.where(tags == 0)[0]
        atoms.set_constraint(FixAtoms(indices=substrate_indices))

        if reaction_types is None:
            raise ValueError(
                "Configuration error: 'ourDimer' -> 'reaction_types' is not set. "
                "Please specify reaction types (e.g., 'adsorbate_atom adsorbate diffusion') in config.ini. "
                "Supported OC types: " + ", ".join(_OC_REACTION_TYPE_DISPATCH.keys()))

        num_per_type = config_dict["ourDimer"].get("num_attempts_per_type", 1)
        counts = _resolve_attempts_per_type(num_per_type, reaction_types_list)

        for rtype, n_attempts in zip(reaction_types_list, counts):
            if rtype not in _OC_REACTION_TYPE_DISPATCH:
                supported = ", ".join(_OC_REACTION_TYPE_DISPATCH.keys())
                raise ValueError(f"Unknown OC reaction type: '{rtype}'. "
                                 f"Supported types: {supported}")
            imgs, dds, idxs = _OC_REACTION_TYPE_DISPATCH[rtype](atoms, config_dict, n_attempts)
            images.extend(imgs)
            displacement_dicts.extend(dds)
            selected_indices.extend(idxs)

    else:
        raise Exception("dataset_type in ourDimer must be set")

    return images, displacement_dicts, selected_indices
