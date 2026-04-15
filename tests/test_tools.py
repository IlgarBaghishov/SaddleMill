"""Comprehensive tests for tsearch/tools.py."""

import os
import json
import numpy as np
import pytest
from ase import Atoms
from ase.build import bulk, fcc111, add_adsorbate
from ase.io import Trajectory
from ase.calculators.emt import EMT
from ase.calculators.singlepoint import SinglePointCalculator
from ase.constraints import FixAtoms
from ase.neighborlist import natural_cutoffs

from tests.conftest import make_config_dict
from tsearch.tools import (
    load_and_sanitize,
    passes_input_filter,
    save_ordered_traj_names,
    read_ordered_traj_names,
    clean_up_files,
    get_bond_set,
    check_reaction,
    check_adsorbate_reaction,
    extract_previous_results,
    _build_output_traj_index,
    _sanitize_with_continuation,
)


# =========================================================================
# TestLoadAndSanitize
# =========================================================================

class TestLoadAndSanitize:
    """Tests for load_and_sanitize: wraps .info into orig_info."""

    def _write_traj(self, path, atoms_list):
        """Helper: write a list of Atoms to a trajectory file."""
        with Trajectory(str(path), "w") as traj:
            for atoms in atoms_list:
                traj.write(atoms)

    def test_single_frame_wraps_info(self, tmp_path):
        """Single frame (j == i+1) wraps .info into orig_info."""
        atoms = bulk("Cu", "fcc", a=3.6)
        atoms.info = {"key1": "value1", "key2": 42}
        traj_path = tmp_path / "test.traj"
        self._write_traj(traj_path, [atoms])

        with Trajectory(str(traj_path), "r") as traj:
            result = load_and_sanitize(traj, 0, 1)

        assert isinstance(result, Atoms)
        assert "orig_info" in result.info
        assert result.info["orig_info"]["key1"] == "value1"
        assert result.info["orig_info"]["key2"] == 42
        # Top-level should only have orig_info
        assert set(result.info.keys()) == {"orig_info"}

    def test_multi_frame_wraps_each(self, tmp_path):
        """Multi frame (j != i+1) returns list, each with orig_info."""
        atoms_list = []
        for idx in range(5):
            a = bulk("Cu", "fcc", a=3.6)
            a.info = {"frame": idx}
            atoms_list.append(a)
        traj_path = tmp_path / "multi.traj"
        self._write_traj(traj_path, atoms_list)

        with Trajectory(str(traj_path), "r") as traj:
            result = load_and_sanitize(traj, 1, 4)

        assert isinstance(result, list)
        assert len(result) == 3
        for i, img in enumerate(result):
            assert "orig_info" in img.info
            assert img.info["orig_info"]["frame"] == i + 1
            assert set(img.info.keys()) == {"orig_info"}

    def test_preserves_original_info_keys(self, tmp_path):
        """Original info keys are preserved inside orig_info, not lost."""
        atoms = bulk("Cu", "fcc", a=3.6)
        atoms.info = {"eigenmode": [[1, 0, 0]], "converged": True, "src_index": 7}
        traj_path = tmp_path / "info.traj"
        self._write_traj(traj_path, [atoms])

        with Trajectory(str(traj_path), "r") as traj:
            result = load_and_sanitize(traj, 0, 1)

        orig = result.info["orig_info"]
        assert orig["eigenmode"] == [[1, 0, 0]]
        assert orig["converged"] is True
        assert orig["src_index"] == 7

    def test_empty_info_works(self, tmp_path):
        """Atoms with empty .info should still wrap into orig_info = {}."""
        atoms = bulk("Cu", "fcc", a=3.6)
        atoms.info = {}
        traj_path = tmp_path / "empty.traj"
        self._write_traj(traj_path, [atoms])

        with Trajectory(str(traj_path), "r") as traj:
            result = load_and_sanitize(traj, 0, 1)

        assert result.info == {"orig_info": {}}

    def test_info_with_numpy_arrays_preserved(self, tmp_path):
        """Numpy arrays in .info should be preserved inside orig_info."""
        atoms = bulk("Cu", "fcc", a=3.6)
        arr = np.array([1.0, 2.0, 3.0])
        atoms.info = {"forces_max": arr, "label": "test"}
        traj_path = tmp_path / "numpy.traj"
        self._write_traj(traj_path, [atoms])

        with Trajectory(str(traj_path), "r") as traj:
            result = load_and_sanitize(traj, 0, 1)

        orig = result.info["orig_info"]
        assert np.array_equal(orig["forces_max"], arr)
        assert orig["label"] == "test"


# =========================================================================
# TestPassesInputFilter
# =========================================================================

class TestPassesInputFilter:
    """Tests for passes_input_filter: fnmatch-based status filtering."""

    def _sanitized(self, **info_kwargs):
        """Build an Atoms with orig_info populated like load_and_sanitize does."""
        atoms = bulk("Cu", "fcc", a=3.6)
        atoms.info = {"orig_info": dict(info_kwargs)}
        return atoms

    def test_all_bypasses_filter(self):
        """input_statuses='all' accepts any frame, even one without a status."""
        atoms = self._sanitized()
        config = make_config_dict(method="Dimer", input_statuses="all")
        assert passes_input_filter(atoms, config) is True

    def test_string_filter_exact_match_passes(self):
        """Single-string filter matches exact status."""
        atoms = self._sanitized(status="converged_CI")
        config = make_config_dict(method="Dimer", input_statuses="converged_CI")
        assert passes_input_filter(atoms, config) is True

    def test_string_filter_mismatch_skips(self):
        """Single-string filter skips frames with a different status."""
        atoms = self._sanitized(status="converged")
        config = make_config_dict(method="Dimer", input_statuses="converged_CI")
        assert passes_input_filter(atoms, config) is False

    def test_list_filter_any_match_passes(self):
        """List filter accepts frames whose status is in the list."""
        atoms = self._sanitized(status="converged_CI")
        config = make_config_dict(
            method="Dimer",
            input_statuses=["converged", "converged_CI"],
        )
        assert passes_input_filter(atoms, config) is True

    def test_list_filter_excludes_unlisted_status(self):
        """List filter rejects statuses not enumerated."""
        atoms = self._sanitized(status="converged_to_desorption")
        config = make_config_dict(
            method="Dimer",
            input_statuses=["converged", "converged_after_extension"],
        )
        assert passes_input_filter(atoms, config) is False

    def test_missing_status_rejected_when_filter_set(self):
        """Frames lacking 'status' (raw inputs) are rejected by an explicit filter."""
        atoms = self._sanitized()
        config = make_config_dict(method="Dimer", input_statuses="converged")
        assert passes_input_filter(atoms, config) is False

    def test_missing_status_passes_when_default_all(self):
        """Default 'all' accepts raw inputs with no prior status."""
        atoms = self._sanitized()
        config = make_config_dict(method="Dimer")
        assert config["Main"]["input_statuses"] == "all"
        assert passes_input_filter(atoms, config) is True

    @pytest.mark.parametrize("status", [
        "converged",
        "converged_CI",
        "converged_after_extension",
        "converged_to_desorption",
        "converged_desorption_skipped",
    ])
    def test_wildcard_matches_all_converged_variants(self, status):
        """'converged*' matches every converged-prefixed status."""
        atoms = self._sanitized(status=status)
        config = make_config_dict(method="Dimer", input_statuses="converged*")
        assert passes_input_filter(atoms, config) is True

    @pytest.mark.parametrize("status", [
        "not_converged",
        "not_converged_after_extension",
        "not_converged_StopRun",
        "error: kaboom",
    ])
    def test_wildcard_rejects_non_converged(self, status):
        """'converged*' does not match statuses that don't start with 'converged'."""
        atoms = self._sanitized(status=status)
        config = make_config_dict(method="Dimer", input_statuses="converged*")
        assert passes_input_filter(atoms, config) is False

    def test_wildcard_in_list(self):
        """Wildcards work inside a list; any pattern match accepts."""
        atoms = self._sanitized(status="not_converged_StopRun")
        config = make_config_dict(
            method="Dimer",
            input_statuses=["converged*", "not_converged_*"],
        )
        assert passes_input_filter(atoms, config) is True

    def test_list_of_images_uses_first(self):
        """For multi-image jobs (NEB input), the first image drives filtering."""
        head = self._sanitized(status="converged")
        tail = self._sanitized(status="converged_CI")
        config = make_config_dict(method="NEB", input_statuses="converged")
        assert passes_input_filter([head, tail], config) is True
        config2 = make_config_dict(method="NEB", input_statuses="converged_CI")
        assert passes_input_filter([head, tail], config2) is False


# =========================================================================
# TestCheckReaction
# =========================================================================

class TestCheckReaction:
    """Tests for check_reaction: bond-breaking/forming detection."""

    def test_identical_structures_no_reaction(self, emt_cu_bulk):
        """Two identical structures should show no reaction."""
        atoms1 = emt_cu_bulk.copy()
        atoms2 = emt_cu_bulk.copy()
        result = check_reaction(atoms1, atoms2)
        assert result["occurred"] is False
        assert result["n_broken"] == 0
        assert result["n_formed"] == 0
        assert len(result["broken_bonds"]) == 0
        assert len(result["formed_bonds"]) == 0

    def test_displaced_atom_reaction_detected(self, emt_cu_bulk):
        """Displacing an atom far enough should break/form bonds."""
        atoms1 = emt_cu_bulk.copy()
        atoms2 = emt_cu_bulk.copy()
        # Move atom 0 far away to break its bonds
        atoms2.positions[0] += [5.0, 5.0, 5.0]
        result = check_reaction(atoms1, atoms2)
        assert result["occurred"] is True
        assert result["n_broken"] > 0 or result["n_formed"] > 0

    def test_different_species_raises(self, emt_cu_bulk):
        """Structures with different atomic numbers should raise AssertionError."""
        atoms1 = emt_cu_bulk.copy()
        atoms2 = emt_cu_bulk.copy()
        atoms2.numbers[0] = 79  # Change Cu to Au
        with pytest.raises(AssertionError):
            check_reaction(atoms1, atoms2)


# =========================================================================
# TestCheckAdsorbateReaction
# =========================================================================

class TestCheckAdsorbateReaction:
    """Tests for check_adsorbate_reaction: tag-filtered bond detection."""

    def test_only_tag2_bonds_considered(self, emt_cu_slab_with_adsorbate):
        """Only bonds where BOTH atoms have tag=2 should be considered."""
        atoms1 = emt_cu_slab_with_adsorbate.copy()
        atoms2 = emt_cu_slab_with_adsorbate.copy()
        # Displace a substrate atom (tag=0) - should NOT register as adsorbate reaction
        substrate_idx = np.where(atoms2.get_tags() == 0)[0]
        if len(substrate_idx) > 0:
            atoms2.positions[substrate_idx[0]] += [3.0, 3.0, 0.0]
        result = check_adsorbate_reaction(atoms1, atoms2, target_tag=2)
        assert result["occurred"] is False

    def test_tag2_bond_break_detected(self, emt_cu_slab_with_adsorbate):
        """Breaking a bond between tag=2 atoms should be detected."""
        atoms1 = emt_cu_slab_with_adsorbate.copy()
        atoms2 = emt_cu_slab_with_adsorbate.copy()
        # Find tag=2 atoms (C and O adsorbate)
        ads_idx = np.where(atoms2.get_tags() == 2)[0]
        assert len(ads_idx) >= 2, "Need at least 2 adsorbate atoms"
        # Check if there's already a bond between them
        cutoffs = natural_cutoffs(atoms1, mult=1.25)
        bonds_before = get_bond_set(atoms1, cutoffs, tag_filter=2)
        if len(bonds_before) > 0:
            # There's a bond; break it by moving one adsorbate far away
            atoms2.positions[ads_idx[-1]] += [0.0, 0.0, 8.0]
            result = check_adsorbate_reaction(atoms1, atoms2, target_tag=2)
            assert result["occurred"] is True
            assert result["n_broken"] > 0
        else:
            # No existing tag=2 bond; form one by bringing them close
            atoms2.positions[ads_idx[-1]] = atoms2.positions[ads_idx[0]] + [0.5, 0.0, 0.0]
            result = check_adsorbate_reaction(atoms1, atoms2, target_tag=2)
            assert result["occurred"] is True
            assert result["n_formed"] > 0

    def test_no_adsorbate_atoms_no_reaction(self):
        """Structure with no tag=2 atoms should show no adsorbate reaction."""
        atoms1 = bulk("Cu", "fcc", a=3.6, cubic=True) * (2, 2, 2)
        atoms2 = atoms1.copy()
        # All tags default to 0
        result = check_adsorbate_reaction(atoms1, atoms2, target_tag=2)
        assert result["occurred"] is False
        assert result["n_broken"] == 0
        assert result["n_formed"] == 0


# =========================================================================
# TestGetBondSet
# =========================================================================

class TestGetBondSet:
    """Tests for get_bond_set: bond set construction."""

    def test_returns_set_of_sorted_tuples(self, emt_cu_bulk):
        """Bonds should be set of sorted (a, b) tuples with a < b."""
        atoms = emt_cu_bulk.copy()
        cutoffs = natural_cutoffs(atoms, mult=1.25)
        bonds = get_bond_set(atoms, cutoffs)
        assert isinstance(bonds, set)
        for bond in bonds:
            assert isinstance(bond, tuple)
            assert len(bond) == 2
            assert bond[0] < bond[1]

    def test_tag_filter_excludes_others(self, emt_cu_slab_with_adsorbate):
        """tag_filter should only include bonds where BOTH atoms have that tag."""
        atoms = emt_cu_slab_with_adsorbate.copy()
        cutoffs = natural_cutoffs(atoms, mult=1.25)
        bonds_all = get_bond_set(atoms, cutoffs, tag_filter=None)
        bonds_tag0 = get_bond_set(atoms, cutoffs, tag_filter=0)
        bonds_tag2 = get_bond_set(atoms, cutoffs, tag_filter=2)
        # Filtered bonds should be subsets of all bonds
        assert bonds_tag0.issubset(bonds_all)
        assert bonds_tag2.issubset(bonds_all)
        # tag=0 bonds should not overlap with tag=2 bonds
        assert bonds_tag0.isdisjoint(bonds_tag2)

    def test_symmetric_bonds_collapsed(self, emt_cu_bulk):
        """Bond (i,j) and (j,i) should be collapsed into one sorted tuple."""
        atoms = emt_cu_bulk.copy()
        cutoffs = natural_cutoffs(atoms, mult=1.25)
        bonds = get_bond_set(atoms, cutoffs)
        # Verify no duplicates by checking sorted property
        for bond in bonds:
            reverse = (bond[1], bond[0])
            assert reverse not in bonds


# =========================================================================
# TestCleanUpFiles
# =========================================================================

class TestCleanUpFiles:
    """Tests for clean_up_files: temp file removal per method."""

    def test_neb_cleanup(self, tmp_path, monkeypatch):
        """NEB cleanup removes NEB-specific temp files."""
        monkeypatch.chdir(tmp_path)
        # Create NEB temp files
        files = [
            "neb_0.log", "neb_0.traj", "neb_1.log",
            "reactant_relaxation_0.log", "reactant_relaxation_0.traj",
            "product_relaxation_0.log", "product_relaxation_0.traj",
            "diffusion_barrier_0.png",
            "flux_0.out", "flux_0.err",
        ]
        for f in files:
            (tmp_path / f).write_text("dummy")
        # Keep file that should not be removed
        (tmp_path / "important.txt").write_text("keep me")

        config = make_config_dict(method="NEB")
        clean_up_files(config)

        for f in files:
            assert not (tmp_path / f).exists(), f"Expected {f} to be removed"
        assert (tmp_path / "important.txt").exists()

    def test_neb_vasp_cleanup(self, tmp_path, monkeypatch):
        """NEB with VASP calculator also removes VASP_*_*/ directories."""
        monkeypatch.chdir(tmp_path)
        (tmp_path / "neb_0.log").write_text("dummy")
        vasp_dir = tmp_path / "VASP_0_1"
        vasp_dir.mkdir()
        (vasp_dir / "INCAR").write_text("dummy")

        config = make_config_dict(method="NEB", Calculator="Vasp")
        clean_up_files(config)

        assert not (tmp_path / "neb_0.log").exists()
        assert not vasp_dir.exists()

    def test_dimer_cleanup(self, tmp_path, monkeypatch):
        """Dimer cleanup removes dimer-specific temp files."""
        monkeypatch.chdir(tmp_path)
        files = [
            "dimer_control_0.log", "dimer_opt_0.log", "dimer_0.traj",
            "flux_0.out", "flux_0.err",
        ]
        for f in files:
            (tmp_path / f).write_text("dummy")
        (tmp_path / "keep.txt").write_text("keep")

        config = make_config_dict(method="Dimer")
        clean_up_files(config)

        for f in files:
            assert not (tmp_path / f).exists(), f"Expected {f} to be removed"
        assert (tmp_path / "keep.txt").exists()

    def test_minimization_cleanup(self, tmp_path, monkeypatch):
        """Minimization cleanup removes optimization_* files."""
        monkeypatch.chdir(tmp_path)
        files = [
            "optimization_0.log", "optimization_0.traj",
            "flux_0.out", "flux_0.err",
        ]
        for f in files:
            (tmp_path / f).write_text("dummy")

        config = make_config_dict(method="Minimization")
        clean_up_files(config)

        for f in files:
            assert not (tmp_path / f).exists(), f"Expected {f} to be removed"

    def test_double_minimization_cleanup(self, tmp_path, monkeypatch):
        """DoubleMinimization shares same patterns as Minimization."""
        monkeypatch.chdir(tmp_path)
        files = ["optimization_0.log", "optimization_0.traj"]
        for f in files:
            (tmp_path / f).write_text("dummy")

        config = make_config_dict(method="DoubleMinimization")
        clean_up_files(config)

        for f in files:
            assert not (tmp_path / f).exists()


# =========================================================================
# TestExtractPreviousResults
# =========================================================================

class TestExtractPreviousResults:
    """Tests for extract_previous_results and helpers."""

    def _make_atoms(self, **info_kwargs):
        """Create a simple Atoms with specified .info keys."""
        atoms = bulk("Cu", "fcc", a=3.6)
        atoms.info.update(info_kwargs)
        return atoms

    def _write_output_traj(self, traj_dir, filename, atoms_list):
        """Write atoms to an output trajectory."""
        os.makedirs(traj_dir, exist_ok=True)
        path = os.path.join(traj_dir, filename)
        with Trajectory(path, "w") as traj:
            for a in atoms_list:
                traj.write(a)

    def test_minimization_extraction(self, tmp_path, monkeypatch):
        """Minimization: extracts single Atoms per job_id."""
        monkeypatch.chdir(tmp_path)
        atoms = self._make_atoms(src_index=0, converged=True)
        self._write_output_traj("Minimization_trajes", "collected_ts_rank_0.traj", [atoms])

        config = make_config_dict(method="Minimization")
        redo_info = {0: {None}}
        results = extract_previous_results([0], config, redo_info)

        assert 0 in results
        assert isinstance(results[0], Atoms)
        assert "orig_info" in results[0].info

    def test_dimer_extraction(self, tmp_path, monkeypatch):
        """Dimer: extracts {attempt_id: Atoms} per job_id."""
        monkeypatch.chdir(tmp_path)
        frames = []
        for aid in range(3):
            frames.append(self._make_atoms(src_index=5, attempt_id=aid, reaction_type="vacancy"))
        self._write_output_traj("Dimer_trajes", "collected_ts_rank_0.traj", frames)

        config = make_config_dict(method="Dimer")
        redo_info = {5: {0, 1, 2}}
        results = extract_previous_results([5], config, redo_info)

        assert 5 in results
        grouped = results[5]
        assert isinstance(grouped, dict)
        assert set(grouped.keys()) == {0, 1, 2}
        for aid, atoms in grouped.items():
            assert isinstance(atoms, Atoms)
            assert "orig_info" in atoms.info

    def test_neb_extraction(self, tmp_path, monkeypatch):
        """NEB: extracts {subband_idx: [Atoms sorted by image_idx]}."""
        monkeypatch.chdir(tmp_path)
        frames = []
        # Sub-band 0 with 5 images (write out of order to test sorting)
        for img_idx in [4, 2, 0, 1, 3]:
            frames.append(self._make_atoms(
                src_index=10, subband_idx=0, image_idx=img_idx,
                image_type="regular",
            ))
        # Sub-band 1 with 3 images
        for img_idx in [2, 0, 1]:
            frames.append(self._make_atoms(
                src_index=10, subband_idx=1, image_idx=img_idx,
                image_type="regular",
            ))
        self._write_output_traj("NEB_trajes", "collected_ts_rank_0.traj", frames)

        config = make_config_dict(method="NEB")
        redo_info = {10: {0, 1}}
        results = extract_previous_results([10], config, redo_info)

        assert 10 in results
        grouped = results[10]
        assert set(grouped.keys()) == {0, 1}
        # Sub-band 0: sorted by image_idx
        sb0 = grouped[0]
        assert len(sb0) == 5
        for i, atoms in enumerate(sb0):
            assert atoms.info["orig_info"]["image_idx"] == i
        # Sub-band 1: sorted by image_idx
        sb1 = grouped[1]
        assert len(sb1) == 3
        for i, atoms in enumerate(sb1):
            assert atoms.info["orig_info"]["image_idx"] == i

    def test_double_minimization_extraction(self, tmp_path, monkeypatch):
        """DoubleMinimization: extracts {side: Atoms} per job_id."""
        monkeypatch.chdir(tmp_path)
        frames = []
        for side in [-1, 0, 1]:
            frames.append(self._make_atoms(src_index=3, side=side, converged=True))
        self._write_output_traj(
            "DoubleMinimization_trajes", "collected_ts_rank_0.traj", frames
        )

        config = make_config_dict(method="DoubleMinimization")
        redo_info = {3: {-1, 1}}
        results = extract_previous_results([3], config, redo_info)

        assert 3 in results
        grouped = results[3]
        assert isinstance(grouped, dict)
        # All sides present, each flattened to a single Atoms (not a list)
        assert -1 in grouped and 0 in grouped and 1 in grouped
        for side, atoms in grouped.items():
            assert not isinstance(atoms, list)
            assert "orig_info" in atoms.info

    def test_missing_job_skipped(self, tmp_path, monkeypatch):
        """Jobs not in redo_info or with no output frames are skipped."""
        monkeypatch.chdir(tmp_path)
        atoms = self._make_atoms(src_index=0)
        self._write_output_traj("Minimization_trajes", "collected_ts_rank_0.traj", [atoms])

        config = make_config_dict(method="Minimization")
        # job_id=99 is not in the output traj
        redo_info = {99: {None}}
        results = extract_previous_results([99], config, redo_info)
        assert 99 not in results

    def test_job_not_in_redo_info_skipped(self, tmp_path, monkeypatch):
        """Jobs present in output but not in redo_info are skipped."""
        monkeypatch.chdir(tmp_path)
        atoms = self._make_atoms(src_index=0)
        self._write_output_traj("Minimization_trajes", "collected_ts_rank_0.traj", [atoms])

        config = make_config_dict(method="Minimization")
        # redo_info is empty -- job_id=0 not requested
        results = extract_previous_results([0], config, redo_info={})
        assert 0 not in results


# =========================================================================
# TestBuildOutputTrajIndex
# =========================================================================

class TestBuildOutputTrajIndex:
    """Tests for _build_output_traj_index helper."""

    def test_builds_index_from_output_trajs(self, tmp_path, monkeypatch):
        """Index maps src_index to list of Atoms."""
        monkeypatch.chdir(tmp_path)
        traj_dir = tmp_path / "Minimization_trajes"
        traj_dir.mkdir()
        atoms1 = bulk("Cu", "fcc", a=3.6)
        atoms1.info = {"src_index": 0}
        atoms2 = bulk("Cu", "fcc", a=3.6)
        atoms2.info = {"src_index": 1}
        with Trajectory(str(traj_dir / "out.traj"), "w") as t:
            t.write(atoms1)
            t.write(atoms2)

        index = _build_output_traj_index("Minimization")
        assert 0 in index
        assert 1 in index
        assert len(index[0]) == 1
        assert len(index[1]) == 1

    def test_missing_src_index_skipped(self, tmp_path, monkeypatch):
        """Frames without src_index are not indexed."""
        monkeypatch.chdir(tmp_path)
        traj_dir = tmp_path / "NEB_trajes"
        traj_dir.mkdir()
        atoms = bulk("Cu", "fcc", a=3.6)
        atoms.info = {}  # No src_index
        with Trajectory(str(traj_dir / "out.traj"), "w") as t:
            t.write(atoms)

        index = _build_output_traj_index("NEB")
        assert len(index) == 0


# =========================================================================
# TestSanitizeWithContinuation
# =========================================================================

class TestSanitizeWithContinuation:
    """Tests for _sanitize_with_continuation."""

    def test_wraps_info(self):
        """Should wrap .info into orig_info."""
        atoms = bulk("Cu", "fcc", a=3.6)
        atoms.info = {"key": "value", "num": 42}
        result = _sanitize_with_continuation(atoms)
        assert result is atoms  # mutates in place
        assert set(atoms.info.keys()) == {"orig_info"}
        assert atoms.info["orig_info"]["key"] == "value"
        assert atoms.info["orig_info"]["num"] == 42


# =========================================================================
# TestSaveAndReadOrderedTrajNames
# =========================================================================

class TestSaveAndReadOrderedTrajNames:
    """Tests for save_ordered_traj_names / read_ordered_traj_names roundtrip."""

    def test_roundtrip(self, tmp_path, monkeypatch):
        """Data written by save should be read back identically."""
        monkeypatch.chdir(tmp_path)
        data = [
            ["/path/to/traj1.traj", 0, 10],
            ["/path/to/traj2.traj", 10, 20],
            ["/path/to/traj3.traj", 0, 1],
        ]
        save_ordered_traj_names(data)
        result = read_ordered_traj_names()
        assert result == data

    def test_roundtrip_empty(self, tmp_path, monkeypatch):
        """Empty list roundtrips correctly."""
        monkeypatch.chdir(tmp_path)
        save_ordered_traj_names([])
        result = read_ordered_traj_names()
        assert result == []

    def test_file_created(self, tmp_path, monkeypatch):
        """save_ordered_traj_names creates traj_files_ordered.json."""
        monkeypatch.chdir(tmp_path)
        save_ordered_traj_names([["a.traj", 0, 1]])
        assert (tmp_path / "traj_files_ordered.json").exists()
        # Verify it's valid JSON
        with open(tmp_path / "traj_files_ordered.json") as f:
            data = json.load(f)
        assert data == [["a.traj", 0, 1]]
