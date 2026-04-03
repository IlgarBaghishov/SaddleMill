"""Comprehensive tests for tsearch/dimertools/structure_edit.py."""

import random
import warnings

import numpy as np
import pytest
from ase import Atoms
from ase.build import bulk, fcc111, add_adsorbate
from ase.calculators.emt import EMT
from ase.constraints import FixAtoms

from tsearch.dimertools.structure_edit import (
    turn_into_supercell,
    find_interstitial_sites,
    get_vacancy_attempts,
    get_hop_reuse_attempts,
    get_hop_insert_attempts,
    get_kickout_reuse_attempts,
    get_kickout_insert_attempts,
    get_ring_attempts,
    get_initial_guess_attempts,
    get_adsorbate_atom_attempts,
    get_adsorbate_atom_neighbors_attempts,
    get_adsorbate_attempts,
    get_diffusion_attempts,
    get_rotation_attempts,
    get_adsorbate_surface_attempts,
    get_surface_attempts,
    get_custom_attempts,
    get_attempts,
    _get_oc_adsorbate_indices,
)

# Re-use conftest helpers
from tests.conftest import make_config_dict


# =========================================================================
# Helpers
# =========================================================================

def _seed():
    """Seed both random modules for reproducibility."""
    random.seed(42)
    np.random.seed(42)


def _bulk_config(**overrides):
    """Shortcut for a bulk Dimer config dict."""
    defaults = dict(
        method="Dimer",
        dataset_type="bulk",
        reaction_types="vacancy",
        num_attempts_per_type=2,
    )
    defaults.update(overrides)
    return make_config_dict(**defaults)


def _oc_config(**overrides):
    """Shortcut for an OC Dimer config dict."""
    defaults = dict(
        method="Dimer",
        dataset_type="oc",
        reaction_types="adsorbate_atom",
        num_attempts_per_type=2,
    )
    defaults.update(overrides)
    return make_config_dict(**defaults)


# =========================================================================
# TestTurnIntoSupercell
# =========================================================================

class TestTurnIntoSupercell:

    def test_small_cell_expanded(self):
        """A 1-atom cell (< 7 A per axis) must be expanded to at least 7 A in every direction."""
        _seed()
        atoms = bulk("Cu", "fcc", a=3.6)  # 1 atom, ~3.6 A cell
        result = turn_into_supercell(atoms, min_length=7.0)
        lengths = result.cell.lengths()
        assert all(L >= 7.0 - 0.01 for L in lengths), (
            f"All cell lengths must be >= 7.0, got {lengths}"
        )
        # 1 atom -> 3x3x3 base => 27 atoms minimum, but min_length may force more
        assert len(result) >= 27

    def test_large_cell_unchanged(self):
        """A cell already >= 7 A in all directions with >= 17 atoms should not be expanded."""
        _seed()
        atoms = bulk("Cu", "fcc", a=3.6, cubic=True) * (2, 2, 2)  # 32 atoms, ~7.2 A
        n_orig = len(atoms)
        result = turn_into_supercell(atoms, min_length=7.0)
        assert len(result) == n_orig

    def test_info_preserved(self):
        """atoms.info must survive the supercell expansion."""
        _seed()
        atoms = bulk("Cu", "fcc", a=3.6)  # will be expanded
        atoms.info["test_key"] = "hello"
        atoms.info["number"] = 42
        result = turn_into_supercell(atoms, min_length=7.0)
        assert result.info.get("test_key") == "hello"
        assert result.info.get("number") == 42


# =========================================================================
# TestFindInterstitialSites
# =========================================================================

class TestFindInterstitialSites:

    def test_fcc_has_sites(self, emt_cu_bulk):
        """FCC copper should have a non-trivial number of interstitial sites."""
        _seed()
        sites = find_interstitial_sites(emt_cu_bulk)
        assert isinstance(sites, np.ndarray)
        assert sites.ndim == 2 and sites.shape[1] == 3
        assert len(sites) > 0, "FCC Cu must have interstitial sites"

    def test_sites_inside_cell(self, emt_cu_bulk):
        """All returned sites must lie inside the unit cell (fractional in [0, 1))."""
        _seed()
        sites = find_interstitial_sites(emt_cu_bulk)
        inv_cell = np.linalg.inv(emt_cu_bulk.get_cell())
        frac = sites @ inv_cell
        assert np.all(frac >= -0.05) and np.all(frac < 1.05), (
            "Sites should be approximately inside the unit cell"
        )

    def test_sites_far_from_atoms(self, emt_cu_bulk):
        """Every interstitial site must be at least min_dist_frac * nn_dist from any atom."""
        _seed()
        from ase.neighborlist import neighbor_list, mic
        sites = find_interstitial_sites(emt_cu_bulk, min_dist_frac=0.4)
        cell = emt_cu_bulk.get_cell()
        positions = emt_cu_bulk.get_positions()
        _, _, d = neighbor_list('ijd', emt_cu_bulk, 5.0)
        nn_dist = d.min()
        min_dist = 0.4 * nn_dist

        for site in sites:
            deltas = mic(site - positions, cell)
            dists = np.linalg.norm(deltas, axis=1)
            assert dists.min() >= min_dist - 0.01, (
                f"Site too close to an atom: {dists.min():.3f} < {min_dist:.3f}"
            )


# =========================================================================
# TestBulkVacancy
# =========================================================================

class TestBulkVacancy:

    def test_removes_one_atom(self, emt_cu_bulk):
        """Each vacancy attempt removes exactly 1 atom."""
        _seed()
        config = _bulk_config(reaction_types="vacancy", num_attempts_per_type=3)
        images, dds, idxs = get_vacancy_attempts(emt_cu_bulk, config, 3)
        assert len(images) == 3
        for img in images:
            assert len(img) == len(emt_cu_bulk) - 1

    def test_correct_return_count(self, emt_cu_bulk):
        """Returns matching-length lists for images, displacement_dicts, selected_indices."""
        _seed()
        config = _bulk_config(reaction_types="vacancy", num_attempts_per_type=5)
        images, dds, idxs = get_vacancy_attempts(emt_cu_bulk, config, 5)
        assert len(images) == len(dds) == len(idxs)

    def test_reaction_type_set(self, emt_cu_bulk):
        """Each image must have reaction_type='vacancy' in .info."""
        _seed()
        config = _bulk_config(reaction_types="vacancy", num_attempts_per_type=2)
        images, dds, idxs = get_vacancy_attempts(emt_cu_bulk, config, 2)
        for img in images:
            assert img.info.get("reaction_type") == "vacancy"


# =========================================================================
# TestBulkHopReuse
# =========================================================================

class TestBulkHopReuse:

    def test_no_atoms_changed(self, emt_cu_bulk):
        """hop_reuse should not add or remove atoms."""
        _seed()
        images, dds, idxs = get_hop_reuse_attempts(emt_cu_bulk, 3)
        for img in images:
            assert len(img) == len(emt_cu_bulk)

    def test_displacement_shape(self, emt_cu_bulk):
        """displacement_vector must have shape (N_atoms, 3)."""
        _seed()
        images, dds, idxs = get_hop_reuse_attempts(emt_cu_bulk, 2)
        for dd in dds:
            if "displacement_vector" in dd:
                assert dd["displacement_vector"].shape == (len(emt_cu_bulk), 3)


# =========================================================================
# TestBulkHopInsert
# =========================================================================

class TestBulkHopInsert:

    def test_atom_added(self, emt_cu_bulk):
        """hop_insert must add exactly 1 atom per attempt."""
        _seed()
        images, dds, idxs = get_hop_insert_attempts(emt_cu_bulk, 3)
        for img in images:
            assert len(img) == len(emt_cu_bulk) + 1

    def test_inserted_element_is_small(self, emt_cu_bulk):
        """The inserted atom should be one of H, B, C, N, O (Z in {1,5,6,7,8})."""
        _seed()
        allowed_z = {1, 5, 6, 7, 8}
        images, dds, idxs = get_hop_insert_attempts(emt_cu_bulk, 5)
        for img in images:
            new_z = img.get_atomic_numbers()[-1]
            assert new_z in allowed_z, f"Inserted element Z={new_z} not in {allowed_z}"


# =========================================================================
# TestBulkKickoutReuse
# =========================================================================

class TestBulkKickoutReuse:

    def test_no_atoms_changed(self, emt_cu_bulk):
        """kickout_reuse should not add or remove atoms."""
        _seed()
        images, dds, idxs = get_kickout_reuse_attempts(emt_cu_bulk, 3)
        for img in images:
            assert len(img) == len(emt_cu_bulk)

    def test_two_atoms_displaced(self, emt_cu_bulk):
        """The displacement vector should have exactly 2 non-zero rows."""
        _seed()
        images, dds, idxs = get_kickout_reuse_attempts(emt_cu_bulk, 3)
        for dd in dds:
            if "displacement_vector" in dd:
                dv = dd["displacement_vector"]
                nonzero_rows = np.any(np.abs(dv) > 1e-14, axis=1).sum()
                assert nonzero_rows == 2, f"Expected 2 displaced atoms, got {nonzero_rows}"


# =========================================================================
# TestBulkKickoutInsert
# =========================================================================

class TestBulkKickoutInsert:

    def test_atom_added(self, emt_cu_bulk):
        """kickout_insert must add exactly 1 atom."""
        _seed()
        images, dds, idxs = get_kickout_insert_attempts(emt_cu_bulk, 3)
        for img in images:
            assert len(img) == len(emt_cu_bulk) + 1

    def test_element_similar_radius(self, emt_cu_bulk):
        """The inserted element should be a metal/semiconductor (from the pool)."""
        _seed()
        from tsearch.dimertools.structure_edit import _KICKOUT_INSERT_POOL_Z
        images, dds, idxs = get_kickout_insert_attempts(emt_cu_bulk, 5)
        for img in images:
            new_z = img.get_atomic_numbers()[-1]
            assert new_z in _KICKOUT_INSERT_POOL_Z, (
                f"Inserted element Z={new_z} not in kickout insert pool"
            )


# =========================================================================
# TestBulkRing
# =========================================================================

class TestBulkRing:

    def test_pairwise_exchange(self, emt_cu_bulk):
        """ring_sizes=[2] should produce pairwise exchanges (2 non-zero displacements)."""
        _seed()
        config = _bulk_config(reaction_types="ring", ring_sizes=[2])
        images, dds, idxs = get_ring_attempts(emt_cu_bulk, config, 5)
        assert len(images) > 0, "Should produce at least one attempt"
        for img in images:
            assert len(img) == len(emt_cu_bulk)
            assert img.info["reaction_type"] == "ring"
        for dd in dds:
            if "displacement_vector" in dd:
                dv = dd["displacement_vector"]
                nonzero = np.any(np.abs(dv) > 1e-14, axis=1).sum()
                assert nonzero == 2, f"Pairwise exchange should displace 2 atoms, got {nonzero}"

    def test_ring_size_3(self, emt_cu_bulk):
        """ring_sizes=[3] should produce 3-atom ring rotations."""
        _seed()
        config = _bulk_config(reaction_types="ring", ring_sizes=[3])
        images, dds, idxs = get_ring_attempts(emt_cu_bulk, config, 5)
        assert len(images) > 0, "Should produce at least one ring-3 attempt"
        for dd in dds:
            if "displacement_vector" in dd:
                dv = dd["displacement_vector"]
                nonzero = np.any(np.abs(dv) > 1e-14, axis=1).sum()
                assert nonzero == 3, f"Ring-3 should displace 3 atoms, got {nonzero}"

    def test_empty_ring_sizes_returns_none_entries(self, emt_cu_bulk):
        """ring_sizes=[] should return None entries for each requested attempt."""
        _seed()
        config = _bulk_config(reaction_types="ring", ring_sizes=[])
        images, dds, idxs = get_ring_attempts(emt_cu_bulk, config, 5)
        assert len(images) == 5 and all(img is None for img in images)
        assert len(dds) == 5 and all(dd is None for dd in dds)
        assert idxs == [-1] * 5


# =========================================================================
# TestInitialGuess
# =========================================================================

class TestInitialGuess:

    def test_returns_one_attempt(self, emt_cu_bulk):
        """initial_guess always produces exactly 1 attempt."""
        _seed()
        images, dds, idxs = get_initial_guess_attempts(emt_cu_bulk)
        assert len(images) == 1
        assert len(dds) == 1
        assert len(idxs) == 1

    def test_near_zero_displacement(self, emt_cu_bulk):
        """The displacement vector should be essentially zero (1e-10 scale)."""
        _seed()
        images, dds, idxs = get_initial_guess_attempts(emt_cu_bulk)
        dv = dds[0]["displacement_vector"]
        assert np.max(np.abs(dv)) < 1e-8, (
            f"Displacement too large: max |dv| = {np.max(np.abs(dv))}"
        )

    def test_preserves_eigenmode_from_orig_info(self):
        """If atoms.info['orig_info']['eigenmode'] exists, it should be copied to output."""
        _seed()
        atoms = bulk("Cu", "fcc", a=3.6, cubic=True) * (2, 2, 2)
        eigenmode = np.random.randn(len(atoms), 3)
        atoms.info["orig_info"] = {"eigenmode": eigenmode}
        images, dds, idxs = get_initial_guess_attempts(atoms)
        assert "eigenmode" in images[0].info
        np.testing.assert_array_almost_equal(images[0].info["eigenmode"], eigenmode)
        assert images[0].info["reaction_type"] == "initial_guess"


# =========================================================================
# TestOCReactionTypes
# =========================================================================

class TestOCReactionTypes:

    def test_adsorbate_atom_targets_tag2(self, emt_cu_slab_with_adsorbate):
        """adsorbate_atom should target tag=2 atoms with tight Gaussian."""
        _seed()
        atoms = emt_cu_slab_with_adsorbate
        config = _oc_config(reaction_types="adsorbate_atom", num_attempts_per_type=3)
        images, dds, idxs = get_adsorbate_atom_attempts(atoms, config, 3)
        assert len(images) == 3
        tag2_indices = set(np.where(atoms.get_tags() == 2)[0])
        for dd, idx in zip(dds, idxs):
            assert dd.get("gauss_std") == 0.2
            assert dd.get("number_of_atoms") == 1
            assert dd["displacement_center"] in tag2_indices

    def test_adsorbate_atom_neighbors_targets_tag2(self, emt_cu_slab_with_adsorbate):
        """adsorbate_atom_neighbors targets tag=2 with default std (no gauss_std override)."""
        _seed()
        atoms = emt_cu_slab_with_adsorbate
        config = _oc_config(reaction_types="adsorbate_atom_neighbors")
        images, dds, idxs = get_adsorbate_atom_neighbors_attempts(atoms, config, 2)
        assert len(images) == 2
        tag2_indices = set(np.where(atoms.get_tags() == 2)[0])
        for dd, idx in zip(dds, idxs):
            assert "displacement_center" in dd
            assert dd["displacement_center"] in tag2_indices
            assert "gauss_std" not in dd  # uses DimerControl default

    def test_adsorbate_mask(self, emt_cu_slab_with_adsorbate):
        """adsorbate mode should mask only tag=2 atoms."""
        _seed()
        atoms = emt_cu_slab_with_adsorbate
        config = _oc_config(reaction_types="adsorbate")
        images, dds, idxs = get_adsorbate_attempts(atoms, config, 2)
        assert len(images) == 2
        tags = atoms.get_tags()
        expected_mask = (tags == 2).tolist()
        for dd in dds:
            assert dd["mask"] == expected_mask

    def test_diffusion_uniform_translation(self, emt_cu_slab_with_adsorbate):
        """All adsorbate atoms get the same displacement direction in diffusion."""
        _seed()
        atoms = emt_cu_slab_with_adsorbate
        config = _oc_config(reaction_types="diffusion")
        images, dds, idxs = get_diffusion_attempts(atoms, config, 2)
        ads_indices = np.where(atoms.get_tags() == 2)[0]
        for dd in dds:
            dv = dd["displacement_vector"]
            # All adsorbate atoms should get the same displacement vector
            ads_disps = dv[ads_indices]
            for i in range(1, len(ads_disps)):
                np.testing.assert_array_almost_equal(ads_disps[i], ads_disps[0])
            # Non-adsorbate atoms should have zero displacement
            non_ads = np.delete(np.arange(len(atoms)), ads_indices)
            assert np.max(np.abs(dv[non_ads])) < 1e-14
            # Magnitude should be ~0.1 A
            assert 0.05 < np.linalg.norm(ads_disps[0]) < 0.2

    def test_rotation_single_atom_skips(self, emt_cu_slab_with_adsorbate):
        """Rotation with a single adsorbate atom should return empty with a warning."""
        _seed()
        # Create a slab with only 1 adsorbate atom
        slab = fcc111("Cu", size=(3, 3, 3), vacuum=10.0)
        tags = np.zeros(len(slab), dtype=int)
        z_positions = slab.positions[:, 2]
        z_unique = np.sort(np.unique(np.round(z_positions, 2)))
        for idx, atom in enumerate(slab):
            z_rounded = round(atom.position[2], 2)
            if z_rounded >= z_unique[-1]:
                tags[idx] = 1
        slab.set_tags(tags)
        add_adsorbate(slab, "C", height=1.8, position="ontop")
        new_tags = slab.get_tags().copy()
        new_tags[-1] = 2  # only 1 adsorbate atom
        slab.set_tags(new_tags)
        slab.calc = EMT()

        config = _oc_config(reaction_types="rotation")
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            images, dds, idxs = get_rotation_attempts(slab, config, 3)
        assert len(images) == 3 and all(img is None for img in images)
        assert any("at least 2 adsorbate" in str(warning.message) for warning in w)

    def test_rotation_multi_atom(self, emt_cu_slab_with_adsorbate):
        """Rotation with 2+ adsorbate atoms should produce tangential displacements."""
        _seed()
        atoms = emt_cu_slab_with_adsorbate
        config = _oc_config(reaction_types="rotation")
        images, dds, idxs = get_rotation_attempts(atoms, config, 2)
        ads_indices = np.where(atoms.get_tags() == 2)[0]
        assert len(images) == 2
        for dd in dds:
            dv = dd["displacement_vector"]
            # Non-adsorbate atoms should have zero displacement
            non_ads = np.delete(np.arange(len(atoms)), ads_indices)
            assert np.max(np.abs(dv[non_ads])) < 1e-14
            # Adsorbate atoms should have non-zero displacement
            assert np.max(np.abs(dv[ads_indices])) > 1e-10
            assert dd["method"] == "vector"

    def test_surface_targets_tag1(self, emt_cu_slab_with_adsorbate):
        """surface mode should target tag=1 atoms."""
        _seed()
        atoms = emt_cu_slab_with_adsorbate
        config = _oc_config(reaction_types="surface")
        images, dds, idxs = get_surface_attempts(atoms, config, 3)
        tag1_indices = set(np.where(atoms.get_tags() == 1)[0])
        assert len(images) == 3
        for dd, idx in zip(dds, idxs):
            assert dd["displacement_center"] in tag1_indices

    def test_custom_empty_dict(self, emt_cu_slab_with_adsorbate):
        """custom should return empty displacement dicts."""
        _seed()
        atoms = emt_cu_slab_with_adsorbate
        config = _oc_config(reaction_types="custom")
        images, dds, idxs = get_custom_attempts(atoms, config, 3)
        assert len(images) == 3
        for dd in dds:
            assert dd == {}
        for img in images:
            assert img.info["reaction_type"] == "custom"

    def test_no_adsorbate_returns_empty(self):
        """When no atoms have tag=1 or tag=2, adsorbate functions should return empty."""
        _seed()
        atoms = bulk("Cu", "fcc", a=3.6, cubic=True)  # all tag=0
        config = _oc_config(reaction_types="adsorbate_atom")
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            images, dds, idxs = get_adsorbate_atom_attempts(atoms, config, 2)
        assert len(images) == 2 and all(img is None for img in images)
        assert any("No adsorbate atoms" in str(warning.message) for warning in w)

    def test_adsorbate_surface_includes_neighbors(self, emt_cu_slab_with_adsorbate):
        """adsorbate_surface mask should include adsorbate atoms and neighboring substrate."""
        _seed()
        atoms = emt_cu_slab_with_adsorbate
        config = _oc_config(reaction_types="adsorbate_surface")
        images, dds, idxs = get_adsorbate_surface_attempts(atoms, config, 2)
        assert len(images) == 2
        tags = atoms.get_tags()
        ads_indices = np.where(tags == 2)[0]
        for dd in dds:
            mask = dd["mask"]
            # All adsorbate atoms must be in the mask
            for idx in ads_indices:
                assert mask[idx] is True or mask[idx] == True
            # At least some non-adsorbate atoms should also be True (neighbors)
            non_ads_masked = sum(
                1 for i, m in enumerate(mask)
                if m and i not in set(ads_indices)
            )
            assert non_ads_masked > 0, "Should include neighboring substrate atoms"


# =========================================================================
# TestGetAttempts (dispatcher)
# =========================================================================

class TestGetAttempts:

    def test_bulk_vacancy_dispatch(self, emt_cu_bulk):
        """get_attempts dispatches to vacancy for bulk dataset_type."""
        _seed()
        config = _bulk_config(reaction_types="vacancy", num_attempts_per_type=2)
        images, dds, idxs = get_attempts(emt_cu_bulk, config)
        assert len(images) == 2
        for img in images:
            assert img.info["reaction_type"] == "vacancy"
            assert len(img) == len(emt_cu_bulk) - 1

    def test_oc_dispatch(self, emt_cu_slab_with_adsorbate):
        """get_attempts dispatches to adsorbate_atom for oc dataset_type."""
        _seed()
        config = _oc_config(reaction_types="adsorbate_atom", num_attempts_per_type=2)
        images, dds, idxs = get_attempts(emt_cu_slab_with_adsorbate, config)
        assert len(images) == 2
        for img in images:
            assert img.info["reaction_type"] == "adsorbate_atom"

    def test_initial_guess_skips_supercell(self):
        """initial_guess should not expand the cell, regardless of supercell config."""
        _seed()
        atoms = bulk("Cu", "fcc", a=3.6)  # 1 atom, tiny cell
        n_orig = len(atoms)
        config = _bulk_config(
            reaction_types="initial_guess",
            num_attempts_per_type=1,
            supercell=True,
        )
        images, dds, idxs = get_attempts(atoms, config)
        assert len(images) == 1
        # Should NOT have been expanded because initial_guess bypasses supercell
        assert len(images[0]) == n_orig

    def test_initial_guess_exclusive_warning(self, emt_cu_bulk):
        """initial_guess with other types should warn and ignore the others."""
        _seed()
        config = _bulk_config(
            reaction_types="initial_guess vacancy",
            num_attempts_per_type=3,
        )
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            images, dds, idxs = get_attempts(emt_cu_bulk, config)
        assert len(images) == 1  # Only initial_guess produced
        assert any("exclusive" in str(warning.message).lower() for warning in w)

    def test_oc_fixes_substrate(self, emt_cu_slab_with_adsorbate):
        """OC dispatch should apply FixAtoms to tag=0 atoms."""
        _seed()
        # Create a slab without pre-existing constraints
        slab = emt_cu_slab_with_adsorbate.copy()
        slab.set_constraint()  # clear existing constraints
        config = _oc_config(reaction_types="custom", num_attempts_per_type=1)
        images, dds, idxs = get_attempts(slab, config)
        assert len(images) == 1
        img = images[0]
        constraints = img.constraints
        assert len(constraints) > 0, "OC dispatch should add FixAtoms"
        # Find FixAtoms constraint and check it fixes tag=0 atoms
        fix_atoms = [c for c in constraints if isinstance(c, FixAtoms)]
        assert len(fix_atoms) > 0
        fixed_indices = set(fix_atoms[0].index)
        tag0_indices = set(np.where(img.get_tags() == 0)[0])
        assert fixed_indices == tag0_indices

    def test_unknown_type_raises(self, emt_cu_bulk):
        """An unrecognized reaction type should raise ValueError."""
        _seed()
        config = _bulk_config(reaction_types="totally_fake_type")
        with pytest.raises(ValueError, match="Unknown bulk reaction type"):
            get_attempts(emt_cu_bulk, config)

    def test_unknown_oc_type_raises(self, emt_cu_slab_with_adsorbate):
        """An unrecognized OC reaction type should raise ValueError."""
        _seed()
        config = _oc_config(reaction_types="totally_fake_type")
        with pytest.raises(ValueError, match="Unknown OC reaction type"):
            get_attempts(emt_cu_slab_with_adsorbate, config)

    def test_multiple_types(self, emt_cu_slab_with_adsorbate):
        """Multiple reaction types should all contribute attempts."""
        _seed()
        config = _oc_config(
            reaction_types="adsorbate_atom custom",
            num_attempts_per_type=2,
        )
        images, dds, idxs = get_attempts(emt_cu_slab_with_adsorbate, config)
        # 2 adsorbate_atom + 2 custom = 4 total
        assert len(images) == 4
        types = [img.info["reaction_type"] for img in images]
        assert types.count("adsorbate_atom") == 2
        assert types.count("custom") == 2

    def test_supercell_false_no_expansion(self):
        """When supercell=False, small cells should not be expanded."""
        _seed()
        atoms = bulk("Cu", "fcc", a=3.6, cubic=True)  # 4 atoms
        n_orig = len(atoms)
        config = _bulk_config(
            reaction_types="hop_reuse",
            num_attempts_per_type=1,
            supercell=False,
        )
        images, dds, idxs = get_attempts(atoms, config)
        # hop_reuse doesn't add/remove atoms, and supercell=False means no expansion
        if len(images) > 0:
            assert len(images[0]) == n_orig
