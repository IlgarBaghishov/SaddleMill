import os
import warnings
import numpy as np
import random
from ase import Atom
from ase.constraints import FixAtoms
from ase.neighborlist import NeighborList, natural_cutoffs, neighbor_list, mic
from ase.build import make_supercell
from ase.data import covalent_radii, atomic_numbers

REUSE_MAX_DIST = 5.0  # Angstrom: max atom-to-site distance for hop_reuse.
DISPLACE_KICKOUT_VOID_DIST = 5.0  # Angstrom: kicked atom to target void.
DISPLACE_KICKOUT_COLL_DIST = 5.0  # Angstrom: kicker to kicked atom.

# Ring defaults. ring_frac is the fraction of a FULL exchange rotation, so 0.5
# is the old "halfway" convention and 0.2 is a fifth of the way round.
RING_DEFAULT_MODE = "arc"
RING_DEFAULT_FRAC = 0.2
RING_DEFAULT_NBR_MULT = 1.20
RING_DEFAULT_MAX_CYCLES = 20000


def _is_gauss_slot(g, normal_attempts_per_gaussian):
    """Return whether global attempt index ``g`` is the Gaussian slot.

    ``normal_attempts_per_gaussian`` is an integer ratio:
      0 -> Gaussian swaps disabled
      1 -> one normal, then one Gaussian
      4 -> four normal, then one Gaussian

    The first slot is always normal. The pattern repeats every ``N + 1``
    attempts, with the Gaussian at the end of each block.
    """
    n = int(normal_attempts_per_gaussian)
    if n <= 0:
        return False
    return g % (n + 1) == n


def _gauss_count_before(g, normal_attempts_per_gaussian):
    """Count deterministic Gaussian slots in global indices ``[0, g)``."""
    n = int(normal_attempts_per_gaussian)
    if n <= 0:
        return 0
    return g // (n + 1)


def _reuse_offset(config_dict):
    """Return the ranked-candidate offset for bulk reuse mechanisms.

    This offset is deliberately independent of the Dimer random-seed offset.
    ``SM_SEED_OFFSET`` controls the random realization in ``dimeropt.py`` and is
    not consulted here. Precedence is:

      [ourDimer] bulk_reuse_offset, [ourDimer] sm_offset, SM_OFFSET, then 0.

    ``sm_offset`` and ``SM_OFFSET`` are retained as compatibility aliases for
    ranked-candidate campaigns.
    """
    if config_dict is not None:
        dimer_config = config_dict.get("ourDimer", {}) or {}
        for key in ("bulk_reuse_offset", "sm_offset"):
            value = dimer_config.get(key)
            if value not in (None, ""):
                offset = int(value)
                if offset < 0:
                    raise ValueError(
                        f"[ourDimer] {key} must be a nonnegative integer; got {value!r}"
                    )
                return offset

    value = os.environ.get("SM_OFFSET")
    if value not in (None, ""):
        offset = int(value)
        if offset < 0:
            raise ValueError(f"SM_OFFSET must be nonnegative; got {value!r}")
        return offset

    return 0


def _reuse_exhaustion(config_dict):
    """Validate the ranked-candidate exhaustion policy.

    Gaussian exhaustion is disabled. Ranked mechanisms stop consuming directed
    candidates when the ranked list ends, and their callers pad the remaining
    configured attempt slots with ``None`` so later reaction types keep stable
    attempt IDs. The explicit ``stop`` value is retained for readable configs;
    the former ``gaussian`` value now fails loudly.
    """
    if config_dict is None:
        return "stop"
    d = config_dict.get("ourDimer", {}) or {}
    value = str(d.get("reuse_exhaustion", "stop")).strip().lower()
    if value != "stop":
        raise ValueError(
            "[ourDimer] Gaussian exhaustion is disabled; "
            f"reuse_exhaustion must be 'stop', got {d.get('reuse_exhaustion')!r}"
        )
    return value


def _ranked_slots(num_attempts, n_valid, config_dict, mech_name):
    """Yield ``(attempt_index, ptr, use_gauss)`` for a ranked mechanism.

    ``ptr`` indexes the ranked candidate list with scheduled Gaussian slots
    skipped, so a scheduled Gaussian replacement never consumes a directed
    candidate. Once the ranked list is exhausted, iteration stops. The caller
    pads the ungenerated configured slots with ``None`` to preserve the global
    attempt-ID contract used by ``dimeropt.py`` and resume/analysis code.
    """
    _reuse_exhaustion(config_dict)  # validate that Gaussian exhaustion is off
    normal_per_gaussian = _gaussian_normal_attempts(config_dict)
    offset = _reuse_offset(config_dict)
    produced = 0

    for i in range(num_attempts):
        g = offset + i
        ptr = g - _gauss_count_before(g, normal_per_gaussian)
        if ptr >= n_valid:
            if n_valid == 0:
                warnings.warn(
                    f"{mech_name}: no directed candidates exist for this "
                    f"structure; padding all {num_attempts} configured slots."
                )
            elif produced == 0:
                warnings.warn(
                    f"{mech_name}: offset {offset} is past the end of the "
                    f"{n_valid}-candidate ranked list; padding all "
                    f"{num_attempts} configured slots."
                )
            else:
                warnings.warn(
                    f"{mech_name}: exhausted {n_valid} ranked candidates after "
                    f"{produced} of {num_attempts} requested attempts; padding "
                    f"the remaining configured slots."
                )
            return

        yield i, ptr, _is_gauss_slot(g, normal_per_gaussian)
        produced += 1


def _pad_attempt_slots(images, displacement_dicts, selected_indices, num_attempts):
    """Pad a generator result to exactly ``num_attempts`` aligned slots.

    ``dimeropt.py`` assigns global attempt IDs by list position and independently
    expands configured reaction-type counts. Returning a shorter list would shift
    every later reaction type. ``None`` slots are therefore explicit generation
    failures for the exhausted mechanism rather than omitted attempts.
    """
    if not (len(images) == len(displacement_dicts) == len(selected_indices)):
        raise ValueError("Attempt generator returned misaligned result lists")
    if len(images) > num_attempts:
        raise ValueError(
            f"Attempt generator produced {len(images)} slots for "
            f"num_attempts={num_attempts}"
        )
    missing = num_attempts - len(images)
    if missing:
        images.extend([None] * missing)
        displacement_dicts.extend([None] * missing)
        selected_indices.extend([-1] * missing)
    return images, displacement_dicts, selected_indices


def _empty_attempt_slots(num_attempts):
    """Return exactly ``num_attempts`` explicit ungenerated slots."""
    return ([None] * num_attempts,
            [None] * num_attempts,
            [-1] * num_attempts)

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
    # deltas shape: (N_sites, N_atoms, 3) -> flatten for mic, then restore
    deltas = (candidates[:, None, :] - positions[None, :, :]).reshape(-1, 3)
    min_dists = np.linalg.norm(
        mic(deltas, cell).reshape(len(candidates), len(positions), 3), axis=-1
    ).min(axis=1)
    candidates = candidates[min_dists > min_dist]

    if len(candidates) == 0:
        return np.empty((0, 3))

    # Cluster nearby sites within 0.5 A
    if len(candidates) > 1:
        Z = linkage(candidates, method='single')
        labels = fcluster(Z, t=0.5, criterion='distance')
        unique_labels = np.unique(labels)
        candidates = np.array(
            [candidates[labels == lbl].mean(axis=0) for lbl in unique_labels]
        )

    return candidates


def _safe_normalize(vec):
    """Normalize a vector, returning a random unit vector if norm is near zero."""
    norm = np.linalg.norm(vec)
    if norm < 1e-12:
        vec = np.random.randn(3)
        norm = np.linalg.norm(vec)
    return vec / norm


def _rotation_matrix(axis, theta):
    """Rodrigues rotation matrix for ``theta`` radians about ``axis``."""
    a = _safe_normalize(np.asarray(axis, dtype=float))
    K = np.array([[0.0, -a[2], a[1]], [a[2], 0.0, -a[0]], [-a[1], a[0], 0.0]])
    return np.eye(3) + np.sin(theta) * K + (1.0 - np.cos(theta)) * (K @ K)


def _nearest_site(site_a, other_sites, cell):
    """Return the nearest site from other_sites to site_a under MIC."""
    deltas = mic(other_sites - site_a, cell)
    dists = np.linalg.norm(deltas, axis=1)
    idx = np.argmin(dists)
    return other_sites[idx], deltas[idx]


def _shuffled_site_indices(n_sites, n_attempts):
    """Return n_attempts site indices cycling through a shuffled list."""
    indices = list(range(n_sites))
    random.shuffle(indices)
    return [indices[i % n_sites] for i in range(n_attempts)]


def _parse_nonnegative_int(value, key):
    """Parse a nonnegative integer config value without silently rounding."""
    if isinstance(value, bool):
        raise ValueError(f"[ourDimer] {key} must be a nonnegative integer, not bool")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"[ourDimer] {key} must be a nonnegative integer; got {value!r}"
        ) from exc
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        numeric = float(parsed)
    if numeric != parsed or parsed < 0:
        raise ValueError(
            f"[ourDimer] {key} must be a nonnegative whole number; got {value!r}"
        )
    return parsed


def _gaussian_normal_attempts(config_dict):
    """Number of normal attempts scheduled before each Gaussian replacement.

    Preferred key: ``[ourDimer] gaussian_normal_attempts``.
    ``gaussian_swap_prob`` is a deprecated integer-only compatibility alias. A
    decimal probability fails loudly because the behavior is now deterministic:
    0 disables scheduled replacements, 1 gives 1 normal : 1 Gaussian, and 4
    gives 4 normal : 1 Gaussian.
    """
    if config_dict is None:
        return 0
    d = config_dict.get("ourDimer", {}) or {}
    alias = d.get("gaussian_swap_prob")
    preferred = d.get("gaussian_normal_attempts", 0)
    if alias not in (None, ""):
        if preferred not in (None, "", 0, "0"):
            raise ValueError(
                "Set only one of [ourDimer] gaussian_normal_attempts and the "
                "deprecated gaussian_swap_prob alias"
            )
        return _parse_nonnegative_int(alias, "gaussian_swap_prob")
    return _parse_nonnegative_int(
        0 if preferred in (None, "") else preferred,
        "gaussian_normal_attempts",
    )


def _indices_to_mask(natoms, indices):
    """Return a boolean mask for the unique valid atom indices."""
    mask = np.zeros(natoms, dtype=bool)
    idx = np.asarray(sorted(set(int(i) for i in indices)), dtype=int)
    if len(idx):
        if idx[0] < 0 or idx[-1] >= natoms:
            raise IndexError(f"Atom-mask index outside [0, {natoms}): {idx.tolist()}")
        mask[idx] = True
    return mask.tolist(), idx.tolist()


def _displacement_atom_mask(disp_dict, natoms, center_idx=None):
    """Derive the exact atom set represented by a displacement instruction.

    A supplied mask is preserved. For an explicit displacement vector, every
    nonzero row is selected. A center is only a final one-atom fallback.
    """
    if "mask" in disp_dict:
        raw = np.asarray(disp_dict["mask"], dtype=bool)
        if raw.shape != (natoms,):
            raise ValueError(
                f"Displacement mask has shape {raw.shape}; expected ({natoms},)"
            )
        eligible = np.where(raw)[0].astype(int).tolist()
        return raw.tolist(), eligible

    if "displacement_vector" in disp_dict:
        vec = np.asarray(disp_dict["displacement_vector"], dtype=float)
        if vec.shape != (natoms, 3):
            raise ValueError(
                f"Displacement vector has shape {vec.shape}; expected ({natoms}, 3)"
            )
        eligible = np.where(np.linalg.norm(vec, axis=1) > 1e-12)[0].astype(int).tolist()
        if eligible:
            return _indices_to_mask(natoms, eligible)

    if center_idx is None:
        raise ValueError("Cannot derive Gaussian atom mask without a center atom")
    return _indices_to_mask(natoms, [int(center_idx)])


def _concentrate_params(config_dict):
    """(prob, power, std, max_disp, envelope) for power-law-concentrated swaps.
    concentrate_prob=0 (default) disables the feature -- no behavior change."""
    if config_dict is None:
        return (0.0, 1.5, 0.2, 0.0, 0.0)
    d = config_dict.get("ourDimer", {}) or {}
    return (float(d.get("concentrate_prob", 0.0)),
            float(d.get("concentrate_power", 1.5)),
            float(d.get("concentrate_std", 0.2)),
            float(d.get("concentrate_max_disp", 0.0)),
            float(d.get("concentrate_envelope", 0.0)))


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


def _power_law_vector(natoms, eligible_indices, power, std, max_disp=0.0,
                      envelope=0.0, atoms=None, center_idx=None):
    """Concentrated random displacement: iid Gaussian on the eligible atoms,
    each atom's magnitude raised to `power` (ratios sharpen), then renormalized.

    Normalization, pick one:
      max_disp <= 0  legacy: total norm = std*sqrt(3*n_eligible), the same total
                     an iid Gaussian of this std would inject, just redistributed.
                     power=1 == plain. Size-EXTENSIVE: the largest single-atom
                     displacement grows like n_eligible**0.25, so one std means a
                     different kick size on lemat bulk vs an OC22 slab.
      max_disp  > 0  largest single-atom displacement = max_disp exactly. Size-
                     intensive; `std` is unused on this path.

    envelope > 0 weights each eligible atom by exp(-r^2/(2*envelope^2)) about
    center_idx before the power is applied, so the surviving displacement is a
    contiguous cluster rather than atoms scattered through the cell. Ignored
    unless both `atoms` and `center_idx` are supplied."""
    d = np.zeros((natoms, 3))
    idx = np.asarray(sorted(set(int(i) for i in eligible_indices)), dtype=int)
    if len(idx) == 0:
        return d
    d[idx] = np.random.standard_normal((len(idx), 3))
    if envelope > 0.0 and atoms is not None and center_idx is not None:
        deltas = mic(atoms.positions[idx] - atoms.positions[int(center_idx)],
                     atoms.get_cell())
        w = np.exp(-np.sum(deltas ** 2, axis=1) / (2.0 * envelope ** 2))
        d[idx] *= w[:, None]
    mag = np.linalg.norm(d[idx], axis=1, keepdims=True)
    d[idx] = np.where(mag > 1e-12, d[idx] * mag ** (power - 1), 0.0)
    if max_disp > 0.0:
        peak = np.linalg.norm(d[idx], axis=1).max()
        if peak > 1e-12:
            d *= max_disp / peak
    else:
        total = np.linalg.norm(d)
        if total > 1e-12:
            d *= (std * np.sqrt(3 * len(idx))) / total
    return d


def _maybe_concentrate(disp_dict, eligible_indices, natoms, config_dict,
                       atoms=None, center_idx=None):
    """With probability concentrate_prob, replace an ASE Gaussian dict with an
    explicit power-law-concentrated vector. Returns (dict, was_concentrated).

    `atoms`/`center_idx` are only consulted when concentrate_envelope > 0. If no
    center is given, a random eligible atom becomes the envelope center."""
    q, power, std, max_disp, envelope = _concentrate_params(config_dict)
    if q <= 0.0 or random.random() >= q:
        return disp_dict, False
    if envelope > 0.0 and atoms is not None and center_idx is None:
        pool = [int(i) for i in eligible_indices]
        if pool:
            center_idx = random.choice(pool)
    vec = _power_law_vector(natoms, eligible_indices, power, std,
                            max_disp=max_disp, envelope=envelope,
                            atoms=atoms, center_idx=center_idx)
    return {"displacement_vector": vec, "method": "vector"}, True


def _gauss_or_concentrate(atoms_new, gaussian_dict, eligible_indices,
                          config_dict, center_idx=None):
    """Apply Gaussian or concentration to one explicit mechanism atom set.

    ``gaussian_dict`` describes ordinary Gaussian noise on ``eligible_indices``.
    If concentration triggers, the power-law vector is generated over that exact
    same set. No displacement-radius expansion is performed here.
    """
    disp, concentrated = _maybe_concentrate(
        gaussian_dict,
        eligible_indices,
        len(atoms_new),
        config_dict,
        atoms=atoms_new,
        center_idx=center_idx,
    )
    return disp, "_conc" if concentrated else ""


def _maybe_gauss_or_concentrate(disp_dict, center_idx, atoms_new, config_dict,
                                attempt_index):
    """Deterministically replace a directed attempt using the configured ratio.

    The plain Gaussian and concentrated alternatives both operate on the exact
    atoms moved by ``disp_dict``. The mechanistic directions are discarded, but
    the selected mechanism atom set is preserved.
    """
    n_normal = _gaussian_normal_attempts(config_dict)
    g = _reuse_offset(config_dict) + int(attempt_index)
    if not _is_gauss_slot(g, n_normal):
        return disp_dict, ""

    mask, eligible = _displacement_atom_mask(
        disp_dict, len(atoms_new), center_idx=center_idx
    )
    gaussian_dict = {"mask": mask}
    gdict, gsuffix = _gauss_or_concentrate(
        atoms_new,
        gaussian_dict,
        eligible,
        config_dict,
        center_idx=int(center_idx),
    )
    return gdict, "_gauss" + gsuffix


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


# Common metallic / bulk elements for kickout_insert
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

def _vacancy_nn_hop(atoms, rm_idx, chosen_nn, cell, config_dict, attempt):
    """Build one direct NN-into-vacancy hop attempt."""
    vacancy_pos = atoms.positions[rm_idx].copy()
    nn_pos = atoms.positions[chosen_nn].copy()

    atoms_new = atoms.copy()
    del atoms_new[rm_idx]
    new_nn_idx = chosen_nn if chosen_nn < rm_idx else chosen_nn - 1

    disp_vector = np.zeros((len(atoms_new), 3))
    disp_vector[new_nn_idx] = 0.5 * mic(vacancy_pos - nn_pos, cell)

    disp, suffix = _maybe_gauss_or_concentrate(
        {"displacement_vector": disp_vector, "method": "vector"},
        new_nn_idx, atoms_new, config_dict, attempt)
    atoms_new.info['reaction_type'] = 'vacancy' + suffix
    return atoms_new, disp, rm_idx


def get_vacancy_attempts(atoms, config_dict, num_attempts):
    """Vacancy-mediated diffusion with three sub-mechanisms sampled with equal
    probability:

    0. NN hop: a nearest-neighbor atom hops directly into the vacancy.
    1. NNN hop: a second-nearest-neighbor atom hops directly into the vacancy.
    2. Concerted 2-atom chain: NN hops into the vacancy while its NNN simultaneously
       hops into the NN's original site.

    All mechanisms displace atoms halfway along the hop vector. Every
    ``gaussian_normal_attempts`` normal attempts, one deterministic Gaussian
    replacement is inserted, using the exact atoms moved by the directed
    mechanism.

    There are at most ``len(atoms)`` distinct vacancies, so requests beyond that
    are padded with ``None`` rather than silently returning a shorter list.
    """
    cell = atoms.get_cell()
    i_idx, j_idx = neighbor_list('ij', atoms, 3.5)

    n_real = min(num_attempts, len(atoms))
    if n_real < num_attempts:
        warnings.warn(
            f"vacancy: only {len(atoms)} distinct vacancies exist but "
            f"{num_attempts} attempts were requested; padding the remainder."
        )
    remove_indices = random.sample(range(len(atoms)), n_real)

    images = []
    displacement_dicts = []
    selected_indices = []

    for attempt, rm_idx in enumerate(remove_indices):
        vacancy_pos = atoms.positions[rm_idx].copy()

        nn_indices = list(j_idx[i_idx == rm_idx])
        if len(nn_indices) == 0:
            nn_indices = [x for x in range(len(atoms)) if x != rm_idx]

        mechanism = random.randint(0, 2)

        if mechanism == 0:
            image, disp, idx = _vacancy_nn_hop(
                atoms, rm_idx, random.choice(nn_indices), cell, config_dict, attempt)
            images.append(image)
            displacement_dicts.append(disp)
            selected_indices.append(idx)

        elif mechanism == 1:
            nn_set = set(nn_indices)
            nnn_pairs = [
                (int(nnn), int(nn))
                for nn in nn_indices
                for nnn in j_idx[i_idx == nn]
                if nnn != rm_idx and nnn not in nn_set
            ]

            if len(nnn_pairs) == 0:
                # Fallback to NN hop when no NNN exists (tiny cells, etc.)
                image, disp, idx = _vacancy_nn_hop(
                    atoms, rm_idx, random.choice(nn_indices), cell,
                    config_dict, attempt)
                images.append(image)
                displacement_dicts.append(disp)
                selected_indices.append(idx)
                continue

            chosen_nnn, _ = random.choice(nnn_pairs)
            nnn_pos = atoms.positions[chosen_nnn].copy()

            atoms_new = atoms.copy()
            del atoms_new[rm_idx]
            new_nnn_idx = chosen_nnn if chosen_nnn < rm_idx else chosen_nnn - 1

            disp_vector = np.zeros((len(atoms_new), 3))
            disp_vector[new_nnn_idx] = 0.5 * mic(vacancy_pos - nnn_pos, cell)

            disp, suffix = _maybe_gauss_or_concentrate(
                {"displacement_vector": disp_vector, "method": "vector"},
                new_nnn_idx, atoms_new, config_dict, attempt)
            atoms_new.info['reaction_type'] = 'vacancy' + suffix
            images.append(atoms_new)
            displacement_dicts.append(disp)
            selected_indices.append(rm_idx)

        else:  # mechanism == 2
            # Concerted 2-atom chain: NN->vacancy AND NNN->NN simultaneously.
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

            disp, suffix = _maybe_gauss_or_concentrate(
                {"displacement_vector": disp_vector, "method": "vector"},
                new_nn_idx, atoms_new, config_dict, attempt)
            atoms_new.info['reaction_type'] = 'vacancy' + suffix
            images.append(atoms_new)
            displacement_dicts.append(disp)
            selected_indices.append(rm_idx)

    pad = num_attempts - len(images)
    if pad > 0:
        images.extend([None] * pad)
        displacement_dicts.extend([None] * pad)
        selected_indices.extend([-1] * pad)

    return images, displacement_dicts, selected_indices


# --- Hop attempts (interstitial mechanism) ---

def get_hop_reuse_attempts(atoms, num_attempts, config_dict=None):
    """Displace an existing lattice atom halfway toward an interstitial site.

    (atom, site) pairs are ranked by dist*radius (short hops of small atoms
    first) and consumed deterministically in order. Pairs farther apart than
    REUSE_MAX_DIST are discarded. After every configured number of normal
    attempts, one Gaussian displacement is scheduled; those slots do NOT advance
    the ranked pointer, so they never consume a directed candidate.

    When the ranked list runs out, directed generation stops and the remaining
    configured slots are returned as explicit ``None`` entries. This preserves
    stable global attempt IDs; Gaussian exhaustion is disabled.
    """
    sites = find_interstitial_sites(atoms)
    if len(sites) == 0:
        warnings.warn("Found no interstitial sites; skipping hop_reuse.")
        return _empty_attempt_slots(num_attempts)

    cell = atoms.get_cell()
    positions = atoms.get_positions()
    numbers = atoms.get_atomic_numbers()

    # ONE vectorized mic() call for all pairs. Per-pair mic() calls re-run the
    # full general_find_mic path (Minkowski reduction + image enumeration) on
    # EVERY call for triclinic cells -- that was the bulk Phase-2 stall.
    n_at, n_st = len(positions), len(sites)
    deltas = mic((sites[None, :, :] - positions[:, None, :]).reshape(-1, 3),
                 cell).reshape(n_at, n_st, 3)
    dists = np.linalg.norm(deltas, axis=-1)
    scores = dists * covalent_radii[numbers][:, None]
    a_ix, s_ix = np.nonzero(dists <= REUSE_MAX_DIST)
    order = np.argsort(scores[a_ix, s_ix], kind="stable")
    scored_pairs = [(float(scores[a_ix[k], s_ix[k]]), int(a_ix[k]),
                     deltas[a_ix[k], s_ix[k]]) for k in order]
    n_valid = len(scored_pairs)

    images, displacement_dicts, selected_indices = [], [], []

    for i, ptr, use_gauss in _ranked_slots(num_attempts, n_valid, config_dict,
                                           "hop_reuse"):
        atoms_new = atoms.copy()

        if use_gauss:
            # Preserve the next ranked hop's atom set, but randomize direction.
            if n_valid > 0:
                center_idx = scored_pairs[ptr % n_valid][1]
            else:
                center_idx = random.randrange(len(atoms))
            mask, eligible = _indices_to_mask(len(atoms_new), [center_idx])
            disp, suffix = _gauss_or_concentrate(
                atoms_new, {"mask": mask}, eligible, config_dict,
                center_idx=int(center_idx)
            )
            atoms_new.info['reaction_type'] = 'hop_reuse_gauss' + suffix
            images.append(atoms_new)
            displacement_dicts.append(disp)
            selected_indices.append(int(center_idx))
        else:
            _, atom_idx, delta = scored_pairs[ptr]
            atoms_new.info['reaction_type'] = 'hop_reuse'
            disp_vector = np.zeros((len(atoms_new), 3))
            disp_vector[atom_idx] = 0.5 * delta
            images.append(atoms_new)
            displacement_dicts.append(
                {"displacement_vector": disp_vector, "method": "vector"})
            selected_indices.append(int(atom_idx))

    return _pad_attempt_slots(
        images, displacement_dicts, selected_indices, num_attempts
    )


def get_hop_insert_attempts(atoms, num_attempts, config_dict=None):
    """Insert a new small atom at an interstitial site, displace halfway to
    nearest neighbor site.

    Every ``gaussian_normal_attempts`` normal attempts, a Gaussian or
    concentrated replacement is used on the exact directed-mechanism atom set.
    """
    sites = find_interstitial_sites(atoms)

    if len(sites) < 2:
        warnings.warn("Found fewer than 2 interstitial sites; skipping hop_insert.")
        return [None] * num_attempts, [None] * num_attempts, [-1] * num_attempts

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

        disp_vector = np.zeros((len(atoms_new), 3))
        disp_vector[new_atom_idx] = 0.5 * delta_ab

        disp, suffix = _maybe_gauss_or_concentrate(
            {"displacement_vector": disp_vector, "method": "vector"},
            new_atom_idx, atoms_new, config_dict, attempt)
        atoms_new.info['reaction_type'] = 'hop_insert' + suffix

        images.append(atoms_new)
        displacement_dicts.append(disp)
        selected_indices.append(int(new_atom_idx))

    return images, displacement_dicts, selected_indices


# --- Kickout attempts (interstitialcy / kick-out mechanism) ---

def get_kickout_reuse_attempts(atoms, num_attempts, config_dict=None):
    """Original site-driven interstitialcy kick-out using existing atoms.

    For each interstitial site A, the nearest atom is kicked, the second-nearest
    atom is the kicker, and the kicked atom targets the nearest other
    interstitial site B. Candidates are ranked by the covalent-radius-weighted
    total directed travel distance and consumed deterministically using the
    shared reuse offset.

    Gaussian slots replace the directed kick-out directions without consuming a
    ranked candidate. Exhausting the ranked list pads the remaining configured
    slots with ``None``; Gaussian exhaustion is disabled.
    """
    sites = find_interstitial_sites(atoms)
    if len(sites) < 2:
        warnings.warn("Found fewer than 2 interstitial sites; skipping kickout_reuse.")
        return _empty_attempt_slots(num_attempts)
    if len(atoms) < 2:
        warnings.warn("kickout_reuse requires at least 2 atoms; using no attempts.")
        return _empty_attempt_slots(num_attempts)

    cell = atoms.get_cell()
    positions = atoms.get_positions()
    radii = covalent_radii[atoms.get_atomic_numbers()]

    scored_candidates = []
    for site_a_idx, site_a in enumerate(sites):
        deltas_to_a = mic(positions - site_a, cell)
        dists_to_a = np.linalg.norm(deltas_to_a, axis=1)
        sorted_by_dist = np.argsort(dists_to_a, kind="stable")
        kicked_idx = int(sorted_by_dist[0])
        kicker_idx = int(sorted_by_dist[1])

        kicked_pos = positions[kicked_idx]
        kicker_pos = positions[kicker_idx]
        other_sites = np.delete(sites, site_a_idx, axis=0)
        _, kicked_delta = _nearest_site(kicked_pos, other_sites, cell)
        kicker_delta = mic(site_a - kicker_pos, cell)

        score = (
            np.linalg.norm(kicker_delta) * radii[kicker_idx]
            + np.linalg.norm(kicked_delta) * radii[kicked_idx]
        )
        scored_candidates.append(
            (float(score), int(site_a_idx), kicked_idx, kicker_idx,
             kicker_delta, kicked_delta)
        )

    scored_candidates.sort(key=lambda item: item[0])
    n_valid = len(scored_candidates)

    images, displacement_dicts, selected_indices = [], [], []
    for i, ptr, use_gauss in _ranked_slots(num_attempts, n_valid, config_dict,
                                           "kickout_reuse"):
        atoms_new = atoms.copy()
        if use_gauss:
            if n_valid > 0:
                candidate = scored_candidates[ptr % n_valid]
                kicked_idx, kicker_idx = candidate[2], candidate[3]
                center_idx = kicker_idx
                eligible = [kicker_idx, kicked_idx]
            else:
                center_idx = random.randrange(len(atoms))
                eligible = [center_idx]
            mask, eligible = _indices_to_mask(len(atoms_new), eligible)
            disp, suffix = _gauss_or_concentrate(
                atoms_new, {"mask": mask}, eligible, config_dict,
                center_idx=int(center_idx)
            )
            atoms_new.info["reaction_type"] = "kickout_reuse_gauss" + suffix
            images.append(atoms_new)
            displacement_dicts.append(disp)
            selected_indices.append(int(center_idx))
            continue

        _, _, kicked_idx, kicker_idx, kicker_delta, kicked_delta = (
            scored_candidates[ptr]
        )
        disp_vector = np.zeros((len(atoms_new), 3))
        disp_vector[kicker_idx] = 0.5 * kicker_delta
        disp_vector[kicked_idx] = 0.5 * kicked_delta
        atoms_new.info["reaction_type"] = "kickout_reuse"
        images.append(atoms_new)
        displacement_dicts.append(
            {"displacement_vector": disp_vector, "method": "vector"}
        )
        selected_indices.append(int(kicker_idx))

    return _pad_attempt_slots(
        images, displacement_dicts, selected_indices, num_attempts
    )


def get_displace_kickout_reuse_attempts(atoms, num_attempts, config_dict=None):
    """Ranked collision-chain kick-out using only existing atoms.

    Enumerate every in-range (void, kicked, kicker) triplet. The kicker moves
    toward the kicked atom's lattice site, while the kicked atom moves toward
    the selected void. Exhausting the ranked list pads the remaining configured
    slots with ``None``; Gaussian exhaustion is disabled.
    """
    sites = find_interstitial_sites(atoms)
    if len(sites) < 2:
        warnings.warn(
            "Found fewer than 2 interstitial sites; skipping "
            "displace_kickout_reuse."
        )
        return _empty_attempt_slots(num_attempts)

    cell = atoms.get_cell()
    positions = atoms.get_positions()
    numbers = atoms.get_atomic_numbers()
    n = len(atoms)
    radii = covalent_radii[numbers]

    atom_dist = np.linalg.norm(
        mic((positions[:, None, :] - positions[None, :, :]).reshape(-1, 3), cell),
        axis=1,
    ).reshape(n, n)
    void_dist = np.linalg.norm(
        mic((positions[:, None, :] - sites[None, :, :]).reshape(-1, 3), cell),
        axis=1,
    ).reshape(n, len(sites))

    scored_triplets = []
    for site_idx in range(len(sites)):
        kicked_candidates = np.where(
            void_dist[:, site_idx] <= DISPLACE_KICKOUT_VOID_DIST
        )[0]
        for kicked_idx in kicked_candidates:
            dist_void = void_dist[kicked_idx, site_idx]
            kicker_candidates = np.where(
                (atom_dist[kicked_idx] <= DISPLACE_KICKOUT_COLL_DIST)
                & (atom_dist[kicked_idx] > 1e-6)
            )[0]
            for kicker_idx in kicker_candidates:
                score = (
                    dist_void * radii[kicked_idx]
                    + atom_dist[kicked_idx, kicker_idx] * radii[kicker_idx]
                )
                scored_triplets.append(
                    (float(score), site_idx, int(kicked_idx), int(kicker_idx))
                )

    scored_triplets.sort(key=lambda item: item[0])
    n_valid = len(scored_triplets)

    images, displacement_dicts, selected_indices = [], [], []
    for i, ptr, use_gauss in _ranked_slots(num_attempts, n_valid, config_dict,
                                           "displace_kickout_reuse"):
        atoms_new = atoms.copy()
        if use_gauss:
            if n_valid > 0:
                candidate = scored_triplets[ptr % n_valid]
                kicked_idx, kicker_idx = candidate[2], candidate[3]
                center_idx = kicker_idx
                eligible = [kicker_idx, kicked_idx]
            else:
                center_idx = random.randrange(n)
                eligible = [center_idx]
            mask, eligible = _indices_to_mask(len(atoms_new), eligible)
            disp, suffix = _gauss_or_concentrate(
                atoms_new, {"mask": mask}, eligible, config_dict,
                center_idx=int(center_idx)
            )
            atoms_new.info["reaction_type"] = (
                "displace_kickout_reuse_gauss" + suffix
            )
            images.append(atoms_new)
            displacement_dicts.append(disp)
            selected_indices.append(int(center_idx))
            continue

        _, site_idx, kicked_idx, kicker_idx = scored_triplets[ptr]
        target_void = sites[site_idx]
        kicked_pos = positions[kicked_idx]
        kicker_pos = positions[kicker_idx]
        disp_vector = np.zeros((n, 3))
        disp_vector[kicker_idx] = 0.5 * mic(kicked_pos - kicker_pos, cell)
        disp_vector[kicked_idx] = 0.5 * mic(target_void - kicked_pos, cell)
        atoms_new.info["reaction_type"] = "displace_kickout_reuse"
        images.append(atoms_new)
        displacement_dicts.append(
            {"displacement_vector": disp_vector, "method": "vector"}
        )
        selected_indices.append(int(kicker_idx))

    return _pad_attempt_slots(
        images, displacement_dicts, selected_indices, num_attempts
    )


def get_kickout_insert_attempts(atoms, num_attempts, config_dict=None):
    """Insert a new similar-sized atom at interstitial site; it kicks nearest
    lattice atom out.

    1. Sample a new element with covalent radius similar to host.
    2. Insert it at interstitial site A.
    3. Find the nearest lattice atom -- this is the atom being kicked.
    4. Inserted atom displaced halfway toward kicked atom's position.
    5. Kicked atom displaced halfway toward site B.
    """
    sites = find_interstitial_sites(atoms)

    if len(sites) < 2:
        warnings.warn("Found fewer than 2 interstitial sites; skipping kickout_insert.")
        return [None] * num_attempts, [None] * num_attempts, [-1] * num_attempts

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

        dists_to_a = np.linalg.norm(mic(positions - site_a, cell), axis=1)
        kicked_idx = int(np.argmin(dists_to_a))
        kicked_pos = positions[kicked_idx]

        other_sites = np.delete(sites, site_a_idx, axis=0)
        site_b, _ = _nearest_site(kicked_pos, other_sites, cell)

        atoms_new = atoms.copy()
        atoms_new.append(Atom(element_z, position=site_a))
        inserted_idx = len(atoms_new) - 1

        disp_vector = np.zeros((len(atoms_new), 3))
        disp_vector[inserted_idx] = 0.5 * mic(kicked_pos - site_a, cell)
        disp_vector[kicked_idx] = 0.5 * mic(site_b - kicked_pos, cell)

        disp, suffix = _maybe_gauss_or_concentrate(
            {"displacement_vector": disp_vector, "method": "vector"},
            inserted_idx, atoms_new, config_dict, attempt)
        atoms_new.info['reaction_type'] = 'kickout_insert' + suffix

        images.append(atoms_new)
        displacement_dicts.append(disp)
        selected_indices.append(int(inserted_idx))

    return images, displacement_dicts, selected_indices


# --- Ring attempts (deterministic; ring_size=2 covers pairwise exchange) ---

def _ring_params(config_dict):
    """Ring configuration.

    [ourDimer] keys, all optional:
      ring_sizes            sizes to enumerate               default "3 4"
      ring_mode             arc | chord | legacy             default arc
      ring_frac             fraction of a FULL exchange      default 0.2
      ring_neighbor_mult    natural_cutoffs multiplier       default 1.20
      ring_neighbor_cutoff  flat cutoff override, Angstrom   default unset
      ring_max_cycles       enumeration cap per size         default 20000

    ``arc`` rotates the ring about its own axis and preserves every ring bond
    length exactly. ``chord`` moves each atom a fraction of the way along the
    straight line to its successor, contracting the ring by cos(pi/n) at
    frac=0.5. ``legacy`` reproduces the previous hard-coded behaviour:
    0.5*chord for n>=3, and 0.5*chord plus a 0.15*|d| perpendicular kick at n=2.
    """
    d = {} if config_dict is None else (config_dict.get("ourDimer", {}) or {})

    raw_sizes = d.get("ring_sizes", [3, 4])
    if isinstance(raw_sizes, (int, float)):
        sizes = [int(raw_sizes)]
    elif isinstance(raw_sizes, str):
        sizes = [int(x) for x in raw_sizes.split()]
    else:
        sizes = [int(x) for x in raw_sizes]

    bad = [s for s in sizes if s < 2]
    if bad:
        raise ValueError(
            f"[ourDimer] ring_sizes must all be >= 2; got {bad}. A size below 2 "
            f"produces a zero displacement and a silently wasted attempt."
        )
    sizes = sorted(set(sizes))

    mode = str(d.get("ring_mode", RING_DEFAULT_MODE)).strip().lower()
    if mode not in ("arc", "chord", "legacy"):
        raise ValueError(
            f"[ourDimer] ring_mode must be arc, chord, or legacy; got {mode!r}")

    frac = float(d.get("ring_frac", RING_DEFAULT_FRAC))
    if not 0.0 < frac <= 1.0:
        raise ValueError(f"[ourDimer] ring_frac must be in (0, 1]; got {frac}")

    cutoff = d.get("ring_neighbor_cutoff", None)
    cutoff = None if cutoff in (None, "") else float(cutoff)

    return {
        "sizes": sizes,
        "mode": mode,
        "frac": frac,
        "mult": float(d.get("ring_neighbor_mult", RING_DEFAULT_NBR_MULT)),
        "cutoff": cutoff,
        "max_cycles": int(d.get("ring_max_cycles", RING_DEFAULT_MAX_CYCLES)),
    }


def _build_neighbor_dict(atoms, cutoff=None, mult=RING_DEFAULT_NBR_MULT):
    """Adjacency dict for the ring graph.

    Defaults to per-element ``natural_cutoffs`` scaled by ``mult``. A single
    flat cutoff cannot serve every structure: on an open lattice it leaves atoms
    with no neighbors at all, and on a dense one it reaches past the first shell
    so that "rings" run through non-bonded pairs.
    """
    cut = float(cutoff) if cutoff is not None else natural_cutoffs(atoms, mult=mult)
    i_idx, j_idx = neighbor_list('ij', atoms, cut)
    neighbors_dict = {}
    for i, j in zip(i_idx, j_idx):
        i, j = int(i), int(j)
        if i == j:
            continue  # self-image pair; a ring through one atom is meaningless
        neighbors_dict.setdefault(i, set()).add(j)
    return {k: sorted(v) for k, v in neighbors_dict.items()}


def _canonical_cycle(path):
    """Smallest rotation/reflection of a cycle, for deduplication."""
    n = len(path)
    best = None
    for seq in (list(path), list(path)[::-1]):
        for r in range(n):
            cand = tuple(seq[r:] + seq[:r])
            if best is None or cand < best:
                best = cand
    return best


def _enumerate_cycles(neighbors_dict, n, max_cycles):
    """Every simple cycle of length exactly ``n``, deduped up to rotation and
    reflection. Cost grows like N * c^(n-1) in the coordination number, so the
    caller supplies a hard cap."""
    if n == 2:
        return [[i, j] for i, js in sorted(neighbors_dict.items())
                for j in js if i < j][:max_cycles]

    found = {}
    for start in sorted(neighbors_dict):
        stack = [(start, [start])]
        while stack:
            cur, path = stack.pop()
            if len(path) == n:
                if start in neighbors_dict.get(cur, ()):
                    found.setdefault(_canonical_cycle(path), list(path))
                    if len(found) >= max_cycles:
                        return list(found.values())
                continue
            for nb in neighbors_dict.get(cur, ()):
                # Only build cycles whose smallest member is `start`.
                if nb > start and nb not in path:
                    stack.append((nb, path + [nb]))
    return list(found.values())


def _unwrap_ring(atoms, ring):
    """Ring positions in one continuous frame, chained by MIC from ring[0].

    Also returns the closure error. A large error means the cycle wraps a whole
    lattice vector and is not a local ring.
    """
    cell = atoms.get_cell()
    P = atoms.get_positions()
    out = [P[ring[0]]]
    for k in range(1, len(ring)):
        out.append(out[-1] + mic(P[ring[k]] - P[ring[k - 1]], cell))
    out = np.array(out)
    closed = out[-1] + mic(P[ring[0]] - P[ring[-1]], cell)
    return out, float(np.linalg.norm(closed - out[0]))


def _rank_rings(atoms, config_dict):
    """Enumerate and rank every valid ring across all configured sizes.

    Scoring matches the other reuse mechanisms: sum over ring bonds of
    length * covalent radius, so short rings of small atoms come first. Ties are
    broken by bond-length regularity, then by the sorted index tuple, so the
    ordering is fully deterministic and independent of any RNG.
    """
    params = _ring_params(config_dict)
    neighbors = _build_neighbor_dict(
        atoms, cutoff=params["cutoff"], mult=params["mult"])
    radii = covalent_radii[atoms.get_atomic_numbers()]

    ranked = []
    for n in params["sizes"]:
        for ring in _enumerate_cycles(neighbors, n, params["max_cycles"]):
            pos, closure = _unwrap_ring(atoms, ring)
            if closure > 0.5:
                continue
            bonds = np.array([
                np.linalg.norm(pos[(k + 1) % n] - pos[k]) for k in range(n)
            ])
            if bonds.min() < 1e-6:
                continue
            score = float(sum(bonds[k] * radii[ring[k]] for k in range(n)))
            regularity = float(bonds.std() / bonds.mean())
            ranked.append((round(score, 9), round(regularity, 9),
                           tuple(sorted(int(i) for i in ring)),
                           [int(i) for i in ring]))

    ranked.sort(key=lambda t: (t[0], t[1], t[2]))
    return [r[3] for r in ranked], params


def _ring_perp_axis(atoms, ring, pos, ndirs=12):
    """Rotation axis for a 2-ring. Any direction perpendicular to the bond gives
    a valid exchange, so pick the one that keeps the pair farthest from the
    surrounding atoms instead of whatever the cell orientation happens to give.
    """
    cell = atoms.get_cell()
    P = atoms.get_positions()
    rset = set(int(i) for i in ring)
    others = [i for i in range(len(atoms)) if i not in rset]

    centroid = pos.mean(axis=0)
    C = pos - centroid
    bhat = _safe_normalize(C[1] - C[0])
    ref = np.array([1.0, 0.0, 0.0])
    if abs(float(bhat @ ref)) > 0.9:
        ref = np.array([0.0, 1.0, 0.0])
    e1 = _safe_normalize(np.cross(bhat, ref))
    e2 = np.cross(bhat, e1)

    if not others:
        return e1

    best, best_clearance = e1, -np.inf
    for t in np.linspace(0.0, np.pi, ndirs, endpoint=False):
        axis = np.cos(t) * e1 + np.sin(t) * e2
        trial = (C @ _rotation_matrix(axis, np.pi / 2).T) + centroid
        clearance = min(
            float(np.linalg.norm(mic(P[others] - q, cell), axis=1).min())
            for q in trial
        )
        if clearance > best_clearance:
            best, best_clearance = axis, clearance
    return best


def _orient_ring_axis(C, axis):
    """Flip the axis if needed so a positive rotation carries k toward k+1."""
    moved = C[0] @ _rotation_matrix(axis, 0.05).T
    if np.linalg.norm(moved - C[1]) < np.linalg.norm(C[0] - C[1]):
        return axis
    return -axis


def _ring_displacement(atoms, ring, mode, frac):
    """Displacement vector for one ring.

    ``frac`` is the fraction of a full exchange rotation (2*pi/n), so both modes
    mean the same thing at the endpoints and are directly comparable.
    """
    n = len(ring)
    cell = atoms.get_cell()
    P = atoms.get_positions()
    disp = np.zeros((len(atoms), 3))

    if mode == "legacy":
        if n == 2:
            delta = mic(P[ring[1]] - P[ring[0]], cell)
            dhat = _safe_normalize(delta)
            ref = np.array([1.0, 0.0, 0.0])
            if abs(float(dhat @ ref)) > 0.9:
                ref = np.array([0.0, 1.0, 0.0])
            perp = (_safe_normalize(np.cross(dhat, ref)) * 0.15
                    * np.linalg.norm(delta) * random.choice([-1, 1]))
            disp[ring[0]] = 0.5 * delta + perp
            disp[ring[1]] = -0.5 * delta - perp
        else:
            for k in range(n):
                src, dst = ring[k], ring[(k + 1) % n]
                disp[src] = 0.5 * mic(P[dst] - P[src], cell)
        return disp

    if mode == "chord":
        for k in range(n):
            src, dst = ring[k], ring[(k + 1) % n]
            disp[src] = frac * mic(P[dst] - P[src], cell)
        return disp

    # arc: rigid rotation about the ring's own axis, preserves bond lengths
    pos, _ = _unwrap_ring(atoms, ring)
    centroid = pos.mean(axis=0)
    C = pos - centroid
    if n == 2:
        axis = _ring_perp_axis(atoms, ring, pos)
    else:
        # Least-variance direction of the ring points; for a non-planar ring
        # this is the best-fit plane normal rather than an exact normal.
        _, _, vt = np.linalg.svd(C)
        axis = _orient_ring_axis(C, vt[2])
    new = (C @ _rotation_matrix(axis, frac * 2.0 * np.pi / n).T) + centroid
    for k in range(n):
        disp[ring[k]] = new[k] - pos[k]
    return disp


def get_ring_attempts(atoms, config_dict, num_attempts):
    """Cooperative ring rotation, deterministic.

    Every ring of every configured size is enumerated once and ranked by
    bond-length * covalent-radius, then consumed in order using the shared reuse
    offset. Ring size is therefore chosen by rank, not at random, which removes
    the old failure where a random size landed on a seed that could not close a
    cycle and the attempt was wasted.

    Scheduled Gaussian slots do not consume a ranked ring. Exhausting the ranked
    list pads the remaining configured slots with ``None``; Gaussian exhaustion
    is disabled.
    """
    ranked_rings, params = _rank_rings(atoms, config_dict)
    n_valid = len(ranked_rings)
    if n_valid == 0:
        warnings.warn(
            f"ring: no valid cycles of size(s) {params['sizes']} found "
            f"(neighbor cutoff "
            f"{'flat ' + str(params['cutoff']) if params['cutoff'] else 'natural*' + str(params['mult'])}"
            f"); producing no attempts."
        )
        return _empty_attempt_slots(num_attempts)

    images, displacement_dicts, selected_indices = [], [], []
    for i, ptr, use_gauss in _ranked_slots(num_attempts, n_valid, config_dict,
                                           "ring"):
        ring = ranked_rings[ptr % n_valid]
        atoms_new = atoms.copy()
        atoms_new.info['ring_size'] = len(ring)
        atoms_new.info['ring_indices'] = list(ring)

        if use_gauss:
            mask, eligible = _indices_to_mask(len(atoms_new), ring)
            disp, suffix = _gauss_or_concentrate(
                atoms_new, {"mask": mask}, eligible, config_dict,
                center_idx=int(ring[0])
            )
            atoms_new.info['reaction_type'] = f"ring{len(ring)}_gauss{suffix}"
        else:
            vec = _ring_displacement(atoms_new, ring, params["mode"], params["frac"])
            disp = {"displacement_vector": vec, "method": "vector"}
            atoms_new.info['reaction_type'] = f"ring{len(ring)}"

        images.append(atoms_new)
        displacement_dicts.append(disp)
        selected_indices.append(int(ring[0]))

    return _pad_attempt_slots(
        images, displacement_dicts, selected_indices, num_attempts
    )


# --- OC helpers ---

def _get_oc_adsorbate_indices(atoms):
    """Return indices of adsorbate atoms (tag=2)."""
    tags = atoms.get_tags()
    return np.where(tags == 2)[0]


def _get_oc_neighbor_mask(atoms, adsorbate_indices):
    """Boolean mask covering the adsorbate and everything neighboring it.

    The adsorbate indices are seeded into the set explicitly. NeighborList runs
    with self_interaction=False, so a single-atom adsorbate would otherwise
    never appear in its own mask and would be the one atom left undisplaced.
    """
    cutoffs = natural_cutoffs(atoms, mult=1.25)
    nl = NeighborList(cutoffs, self_interaction=False, bothways=True)
    nl.update(atoms)

    neighbor_indices = set(int(i) for i in adsorbate_indices)
    for idx in adsorbate_indices:
        indices, offsets = nl.get_neighbors(idx)
        neighbor_indices.update(int(i) for i in indices)

    mask = np.zeros(len(atoms), dtype=bool)
    if neighbor_indices:
        mask[np.array(sorted(neighbor_indices), dtype=int)] = True
    return mask.tolist()


def _sample_adsorbate_atoms(adsorbate_indices, num_needed):
    """Sample num_needed adsorbate atom indices, cycling if needed."""
    if len(adsorbate_indices) >= num_needed:
        return random.sample(list(adsorbate_indices), num_needed)
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
        disp, conc = _maybe_concentrate({"mask": mask}, eligible, len(atoms_new),
                                        config_dict, atoms_new)
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

    neighbor_mask = _get_oc_neighbor_mask(atoms, adsorbate_indices)
    movable = _movable_oc(atoms)
    eligible = [i for i, m in enumerate(neighbor_mask) if m and i in movable]
    mask, eligible = _indices_to_mask(len(atoms), eligible)

    images, displacement_dicts, selected_indices = [], [], []
    for _ in range(num_attempts):
        atoms_new = atoms.copy()
        disp, conc = _maybe_concentrate({"mask": mask}, eligible, len(atoms_new),
                                        config_dict, atoms_new)
        atoms_new.info['reaction_type'] = (
            'adsorbate_surface_conc' if conc else 'adsorbate_surface')
        images.append(atoms_new)
        displacement_dicts.append(disp)
        selected_indices.append(-1)
    return images, displacement_dicts, selected_indices


def get_adsorbate_atom_neighbors_attempts(atoms, config_dict, num_attempts):
    """Broad Gaussian on one adsorbate atom, neighbors dragged."""
    adsorbate_indices = _get_oc_adsorbate_indices(atoms)
    if len(adsorbate_indices) == 0:
        warnings.warn(
            "No adsorbate atoms (tag=2) found; skipping 'adsorbate_atom_neighbors'.")
        return [None] * num_attempts, [None] * num_attempts, [-1] * num_attempts

    radius = _displacement_radius(config_dict)
    movable = _movable_oc(atoms)

    chosen = _sample_adsorbate_atoms(adsorbate_indices, num_attempts)
    images, displacement_dicts, selected_indices = [], [], []
    for idx in chosen:
        atoms_new = atoms.copy()
        near = _atoms_within_radius(atoms_new, int(idx), radius)
        eligible = [int(i) for i in near if int(i) in movable]
        mask, eligible = _indices_to_mask(len(atoms_new), eligible)
        disp, conc = _maybe_concentrate(
            {"mask": mask}, eligible, len(atoms_new),
            config_dict, atoms_new, int(idx)
        )
        atoms_new.info['reaction_type'] = (
            'adsorbate_atom_neighbors_conc' if conc else 'adsorbate_atom_neighbors')
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

    radius = _displacement_radius(config_dict)
    movable = _movable_oc(atoms)

    chosen = _sample_adsorbate_atoms(surface_indices, num_attempts)
    images, displacement_dicts, selected_indices = [], [], []
    for idx in chosen:
        atoms_new = atoms.copy()
        near = _atoms_within_radius(atoms_new, int(idx), radius)
        eligible = [int(i) for i in near if int(i) in movable]
        mask, eligible = _indices_to_mask(len(atoms_new), eligible)
        disp, conc = _maybe_concentrate(
            {"mask": mask}, eligible, len(atoms_new),
            config_dict, atoms_new, int(idx)
        )
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
        mask, eligible = _indices_to_mask(len(atoms_new), [int(idx)])
        base_disp = {"mask": mask, "gauss_std": 0.2, "number_of_atoms": 1}
        disp, conc = _maybe_concentrate(
            base_disp, eligible, len(atoms_new),
            config_dict, atoms_new, int(idx)
        )
        atoms_new.info['reaction_type'] = (
            'adsorbate_atom_conc' if conc else 'adsorbate_atom'
        )
        images.append(atoms_new)
        displacement_dicts.append(disp)
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
    """Noise on all movable atoms (tag != 0)."""
    movable = _movable_oc(atoms)
    if not movable:
        warnings.warn("No movable atoms (tag != 0) found; skipping 'all_movable'.")
        return [None] * num_attempts, [None] * num_attempts, [-1] * num_attempts

    mask = [i in movable for i in range(len(atoms))]
    eligible = list(movable)

    images, displacement_dicts, selected_indices = [], [], []
    for _ in range(num_attempts):
        atoms_new = atoms.copy()
        disp, conc = _maybe_concentrate({"mask": mask}, eligible, len(atoms_new),
                                        config_dict, atoms_new)
        atoms_new.info['reaction_type'] = 'all_movable_conc' if conc else 'all_movable'
        images.append(atoms_new)
        displacement_dicts.append(disp)
        selected_indices.append(-1)

    return images, displacement_dicts, selected_indices


def get_all_atoms_attempts(atoms, config_dict, num_attempts):
    """Full random Gaussian over ALL atoms (bulk; no fixed substrate)."""
    eligible = list(range(len(atoms)))
    mask = [True] * len(atoms)
    images, displacement_dicts, selected_indices = [], [], []
    for _ in range(num_attempts):
        atoms_new = atoms.copy()
        disp, conc = _maybe_concentrate({"mask": mask}, eligible, len(atoms_new),
                                        config_dict, atoms_new)
        atoms_new.info['reaction_type'] = 'all_atoms_conc' if conc else 'all_atoms'
        images.append(atoms_new)
        displacement_dicts.append(disp)
        selected_indices.append(-1)
    return images, displacement_dicts, selected_indices


def get_random_bubble_attempts(atoms, config_dict, num_attempts):
    """Localized noise on a random atom and its neighbors within displacement_radius."""
    radius = _displacement_radius(config_dict, default=4.0)

    dataset_type = (config_dict or {}).get("ourDimer", {}).get("dataset_type")
    movable = _movable_oc(atoms) if dataset_type == "oc" else set(range(len(atoms)))

    if not movable:
        warnings.warn("No movable atoms found; skipping 'random_bubble'.")
        return [None] * num_attempts, [None] * num_attempts, [-1] * num_attempts

    movable_list = sorted(movable)
    images, displacement_dicts, selected_indices = [], [], []

    for _ in range(num_attempts):
        atoms_new = atoms.copy()
        center_idx = random.choice(movable_list)

        near_indices = _atoms_within_radius(atoms_new, center_idx, radius)
        eligible = [int(i) for i in near_indices if int(i) in movable]

        mask, eligible = _indices_to_mask(len(atoms_new), eligible)
        base_disp = {"mask": mask}
        disp, conc = _maybe_concentrate(
            base_disp, eligible, len(atoms_new),
            config_dict, atoms_new, int(center_idx)
        )

        atoms_new.info['reaction_type'] = (
            'random_bubble_conc' if conc else 'random_bubble')
        images.append(atoms_new)
        displacement_dicts.append(disp)
        selected_indices.append(int(center_idx))

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
    """No overrides -- displacement fully controlled by [DimerControl] settings."""
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
    """Return the structure as-is with a negligible displacement for eigenmode
    initialization.

    Used when the input geometry is already a good TS guess and only dimer
    refinement (rotation + translation) is needed. Always produces exactly 1
    attempt.

    If the input atoms carry an eigenmode (in atoms.info['eigenmode'] or
    atoms.info['orig_info']['eigenmode']), it is preserved in the output so
    that dimeropt can seed the dimer with it instead of a random guess.
    """
    atoms_new = atoms.copy()
    atoms_new.info['reaction_type'] = 'initial_guess'

    orig = atoms.info.get('orig_info', {})
    if ('eigenmode' not in atoms_new.info
            and isinstance(orig, dict) and 'eigenmode' in orig):
        atoms_new.info['eigenmode'] = np.array(orig['eigenmode'])

    disp_vector = np.random.randn(len(atoms_new), 3) * 1e-10
    return [atoms_new], [{"displacement_vector": disp_vector, "method": "vector"}], [-1]


# --- Main dispatch ---

_BULK_REACTION_TYPE_DISPATCH = {
    "vacancy": lambda atoms, config_dict, n: get_vacancy_attempts(atoms, config_dict, n),
    "hop_reuse": lambda atoms, config_dict, n: get_hop_reuse_attempts(atoms, n, config_dict),
    "hop_insert": lambda atoms, config_dict, n: get_hop_insert_attempts(atoms, n, config_dict),
    "kickout_reuse": lambda atoms, config_dict, n: get_kickout_reuse_attempts(atoms, n, config_dict),
    "displace_kickout_reuse": lambda atoms, config_dict, n: get_displace_kickout_reuse_attempts(atoms, n, config_dict),
    "kickout_insert": lambda atoms, config_dict, n: get_kickout_insert_attempts(atoms, n, config_dict),
    "ring": lambda atoms, config_dict, n: get_ring_attempts(atoms, config_dict, n),
    "initial_guess": lambda atoms, config_dict, n: get_initial_guess_attempts(atoms),
    "all_atoms": lambda atoms, config_dict, n: get_all_atoms_attempts(atoms, config_dict, n),
    "random_bubble": lambda atoms, config_dict, n: get_random_bubble_attempts(atoms, config_dict, n),
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
    "random_bubble": lambda atoms, config_dict, n: get_random_bubble_attempts(atoms, config_dict, n),
}

# Backward-compat alias
_REACTION_TYPE_DISPATCH = _BULK_REACTION_TYPE_DISPATCH


def _parse_reaction_types(reaction_types):
    """Split the configured reaction types on any whitespace."""
    if isinstance(reaction_types, str):
        return reaction_types.split()
    if isinstance(reaction_types, list):
        return list(reaction_types)
    return []


def get_attempts(atoms, config_dict):

    atoms = atoms.copy()

    # --- Handle initial_guess early (no supercell, works for both bulk and oc) ---
    reaction_types = config_dict["ourDimer"].get("reaction_types")
    reaction_types_list = _parse_reaction_types(reaction_types)

    if "initial_guess" in reaction_types_list:
        other_types = [rt for rt in reaction_types_list if rt != "initial_guess"]
        if other_types:
            warnings.warn(
                f"'initial_guess' is exclusive -- ignoring other reaction types: "
                f"{other_types}")

        dataset_type = config_dict["ourDimer"]["dataset_type"]
        num_per_type = config_dict["ourDimer"].get("num_attempts_per_type", 1)
        if _attempt_count_max(num_per_type) > 1:
            warnings.warn(
                f"'initial_guess' always produces 1 attempt -- ignoring "
                f"num_attempts_per_type={num_per_type}")
        if dataset_type == "oc":
            tags = atoms.get_tags()
            indices = np.where(tags == 0)[0]
            atoms.set_constraint(FixAtoms(indices=indices))

        return get_initial_guess_attempts(atoms)

    # Centralized supercell expansion (controlled by config, default True).
    # An OC slab carries exactly one adsorbate; expanding it would duplicate
    # that adsorbate and silently change the system being studied.
    if config_dict["ourDimer"].get("supercell", True):
        is_oc = config_dict["ourDimer"]["dataset_type"] == "oc"
        if is_oc and np.any(atoms.get_tags() == 2):
            expanded = turn_into_supercell(atoms.copy())
            if len(expanded) != len(atoms):
                warnings.warn(
                    "Skipping supercell expansion for an OC structure with "
                    "adsorbate atoms: expansion would duplicate the adsorbate. "
                    "Set [ourDimer] supercell = False to silence this."
                )
        else:
            atoms = turn_into_supercell(atoms)

    # --- Normal dispatch ---

    images = []
    displacement_dicts = []
    selected_indices = []

    if config_dict["ourDimer"]["dataset_type"] == "bulk":

        if reaction_types is None:
            raise ValueError(
                "Configuration error: 'ourDimer' -> 'reaction_types' is not set. "
                "Please specify reaction types (e.g., 'vacancy') in config.ini")

        num_per_type = config_dict["ourDimer"].get("num_attempts_per_type", 1)
        counts = _resolve_attempts_per_type(num_per_type, reaction_types_list)

        for rtype, n_attempts in zip(reaction_types_list, counts):
            if rtype not in _BULK_REACTION_TYPE_DISPATCH:
                supported = ", ".join(_BULK_REACTION_TYPE_DISPATCH.keys())
                raise ValueError(f"Unknown bulk reaction type: '{rtype}'. "
                                 f"Supported types: {supported}")
            imgs, dds, idxs = _BULK_REACTION_TYPE_DISPATCH[rtype](
                atoms, config_dict, n_attempts)
            images.extend(imgs)
            displacement_dicts.extend(dds)
            selected_indices.extend(idxs)

    elif config_dict["ourDimer"]["dataset_type"] == "oc":

        tags = atoms.get_tags()
        substrate_indices = np.where(tags == 0)[0]
        atoms.set_constraint(FixAtoms(indices=substrate_indices))

        if reaction_types is None:
            raise ValueError(
                "Configuration error: 'ourDimer' -> 'reaction_types' is not set. "
                "Please specify reaction types (e.g., 'adsorbate_atom adsorbate "
                "diffusion') in config.ini. Supported OC types: "
                + ", ".join(_OC_REACTION_TYPE_DISPATCH.keys()))

        num_per_type = config_dict["ourDimer"].get("num_attempts_per_type", 1)
        counts = _resolve_attempts_per_type(num_per_type, reaction_types_list)

        for rtype, n_attempts in zip(reaction_types_list, counts):
            if rtype not in _OC_REACTION_TYPE_DISPATCH:
                supported = ", ".join(_OC_REACTION_TYPE_DISPATCH.keys())
                raise ValueError(f"Unknown OC reaction type: '{rtype}'. "
                                 f"Supported types: {supported}")
            imgs, dds, idxs = _OC_REACTION_TYPE_DISPATCH[rtype](
                atoms, config_dict, n_attempts)
            images.extend(imgs)
            displacement_dicts.extend(dds)
            selected_indices.extend(idxs)

    else:
        raise Exception("dataset_type in ourDimer must be set")

    return images, displacement_dicts, selected_indices