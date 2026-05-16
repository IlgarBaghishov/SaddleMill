"""Comprehensive tests for saddlemill/config.py."""

import copy
import json
import os
import textwrap
import zipfile

import numpy as np
import pytest
from ase import Atoms
from ase.io import Trajectory

from saddlemill.config import (
    VALID_RUN_CATEGORIES,
    ConfigManager,
    _categorize_status,
    _normalize_run_jobs,
    archive_and_clean_csvs,
    get_trajes_and_indices,
    load_calculator,
    load_config,
    load_method,
    load_optimizer,
    _load_optimizer,
)
from tests.conftest import make_config_dict


# ===================================================================
# 1. TestConfigManager
# ===================================================================


class TestConfigManager:
    """Tests for ConfigManager initialization, parsing, and access."""

    def test_defaults_structure(self):
        """DEFAULTS contains expected top-level sections with correct types."""
        d = ConfigManager.DEFAULTS
        assert "Main" in d
        assert "ourNEB" in d
        assert "ourDimer" in d
        assert "ourMinimization" in d
        assert "ourDoubleMinimization" in d
        # Spot-check some values and types
        assert d["Main"]["fmax"] == 0.05
        assert d["Main"]["executorlib"] is True
        assert d["Main"]["method"] is None
        assert d["Main"]["input_statuses"] == "all"
        assert d["ourNEB"]["num_frames"] == 10
        assert d["ourDimer"]["num_attempts_per_type"] == 1

    def test_input_statuses_parses_as_list(self, tmp_path):
        """Space-separated input_statuses becomes a list via _parse_value."""
        ini = tmp_path / "config.ini"
        ini.write_text(textwrap.dedent("""\
            [Main]
            method = DoubleMinimization
            input_statuses = converged converged_CI
        """))
        cm = ConfigManager(str(ini))
        assert cm["Main"]["input_statuses"] == ["converged", "converged_CI"]

    def test_input_statuses_parses_single_string(self, tmp_path):
        """Single-token input_statuses stays a string."""
        ini = tmp_path / "config.ini"
        ini.write_text(textwrap.dedent("""\
            [Main]
            method = DoubleMinimization
            input_statuses = converged_CI
        """))
        cm = ConfigManager(str(ini))
        assert cm["Main"]["input_statuses"] == "converged_CI"

    def test_defaults_not_mutated_by_instance(self):
        """Creating a ConfigManager does not mutate the class-level DEFAULTS."""
        original = copy.deepcopy(ConfigManager.DEFAULTS)
        _ = ConfigManager("/nonexistent/config_xyz.ini")
        assert ConfigManager.DEFAULTS == original

    def test_missing_file_uses_defaults(self, capsys):
        """When config file does not exist, uses defaults and prints warning."""
        cm = ConfigManager("/nonexistent/path/config_abc.ini")
        captured = capsys.readouterr()
        assert "Warning" in captured.out
        assert "not found" in captured.out
        # Values should be defaults
        assert cm["Main"]["fmax"] == 0.05
        assert cm["Main"]["Optimizer"] == "MDMin"

    def test_load_from_file(self, tmp_path):
        """Load a config.ini and verify values are parsed correctly."""
        ini = tmp_path / "config.ini"
        ini.write_text(textwrap.dedent("""\
            [Main]
            method = NEB
            fmax = 0.03
            steps = 5000
            executorlib = False
            [ourNEB]
            num_frames = 15
            DNEB = True
        """))
        cm = ConfigManager(str(ini))
        assert cm["Main"]["method"] == "NEB"
        assert cm["Main"]["fmax"] == 0.03
        assert cm["Main"]["steps"] == 5000
        assert cm["Main"]["executorlib"] is False
        assert cm["ourNEB"]["num_frames"] == 15
        assert cm["ourNEB"]["DNEB"] is True

    def test_load_config_function(self, tmp_path):
        """load_config() returns a ConfigManager."""
        ini = tmp_path / "config.ini"
        ini.write_text("[Main]\nmethod = Dimer\n")
        cm = load_config(str(ini))
        assert isinstance(cm, ConfigManager)
        assert cm["Main"]["method"] == "Dimer"

    # ---- _parse_value tests ----

    def test_parse_value_bool_true(self):
        cm = ConfigManager("/nonexistent.ini")
        assert cm._parse_value("True") is True
        assert cm._parse_value("true") is True
        assert cm._parse_value("TRUE") is True

    def test_parse_value_bool_false(self):
        cm = ConfigManager("/nonexistent.ini")
        assert cm._parse_value("False") is False
        assert cm._parse_value("false") is False

    def test_parse_value_int(self):
        cm = ConfigManager("/nonexistent.ini")
        assert cm._parse_value("42") == 42
        assert isinstance(cm._parse_value("42"), int)

    def test_parse_value_negative_int(self):
        cm = ConfigManager("/nonexistent.ini")
        assert cm._parse_value("-1") == -1
        assert isinstance(cm._parse_value("-1"), int)

    def test_parse_value_float(self):
        cm = ConfigManager("/nonexistent.ini")
        assert cm._parse_value("0.05") == 0.05
        assert isinstance(cm._parse_value("0.05"), float)

    def test_parse_value_string(self):
        cm = ConfigManager("/nonexistent.ini")
        assert cm._parse_value("MDMin") == "MDMin"

    def test_parse_value_quoted_string_double(self):
        cm = ConfigManager("/nonexistent.ini")
        result = cm._parse_value('"srun --exclusive -n 64 vasp_std"')
        assert result == "srun --exclusive -n 64 vasp_std"
        assert isinstance(result, str)

    def test_parse_value_quoted_string_single(self):
        cm = ConfigManager("/nonexistent.ini")
        result = cm._parse_value("'srun -n 16 vasp_std'")
        assert result == "srun -n 16 vasp_std"

    def test_parse_value_space_separated_list(self):
        cm = ConfigManager("/nonexistent.ini")
        result = cm._parse_value("3 4 5")
        assert result == [3, 4, 5]

    def test_parse_value_space_separated_mixed(self):
        cm = ConfigManager("/nonexistent.ini")
        result = cm._parse_value("vacancy hop_reuse ring")
        assert result == ["vacancy", "hop_reuse", "ring"]

    # ---- Access method tests ----

    def test_getitem_existing_section(self):
        cm = ConfigManager("/nonexistent.ini")
        main = cm["Main"]
        assert isinstance(main, dict)
        assert "fmax" in main

    def test_getitem_missing_section(self):
        cm = ConfigManager("/nonexistent.ini")
        assert cm["NonexistentSection"] == {}

    def test_get_existing(self):
        cm = ConfigManager("/nonexistent.ini")
        main = cm.get("Main")
        assert main is not None
        assert "method" in main

    def test_get_missing_with_fallback(self):
        cm = ConfigManager("/nonexistent.ini")
        result = cm.get("NoSuchSection", "fallback_value")
        assert result == "fallback_value"

    def test_get_value(self):
        cm = ConfigManager("/nonexistent.ini")
        assert cm.get_value("Main", "fmax") == 0.05

    def test_get_value_missing_key(self):
        cm = ConfigManager("/nonexistent.ini")
        assert cm.get_value("Main", "nonexistent_key", 999) == 999

    def test_get_value_missing_section(self):
        cm = ConfigManager("/nonexistent.ini")
        assert cm.get_value("NoSuch", "key", "default") == "default"

    def test_as_dict(self):
        cm = ConfigManager("/nonexistent.ini")
        d = cm.as_dict
        assert isinstance(d, dict)
        assert "Main" in d
        assert d["Main"]["fmax"] == 0.05

    def test_str_is_json(self):
        cm = ConfigManager("/nonexistent.ini")
        s = str(cm)
        parsed = json.loads(s)
        assert "Main" in parsed
        assert parsed["Main"]["fmax"] == 0.05

    def test_unrecognized_key_warning(self, tmp_path, capsys):
        """Config file with unknown key in a known section triggers warning."""
        ini = tmp_path / "config.ini"
        ini.write_text(textwrap.dedent("""\
            [Main]
            method = NEB
            bogus_key = 123
        """))
        _ = ConfigManager(str(ini))
        captured = capsys.readouterr()
        assert "Unrecognized key" in captured.out
        assert "bogus_key" in captured.out

    def test_extra_section_no_warning(self, tmp_path, capsys):
        """A section not in DEFAULTS (e.g. FAIRChemCalculator) is allowed without warning."""
        ini = tmp_path / "config.ini"
        ini.write_text(textwrap.dedent("""\
            [FAIRChemCalculator]
            device = cuda
            name_or_path = uma-s-1p2
        """))
        cm = ConfigManager(str(ini))
        captured = capsys.readouterr()
        # No "Unrecognized key" warning for sections not in DEFAULTS
        assert "Unrecognized key" not in captured.out
        assert cm["FAIRChemCalculator"]["device"] == "cuda"


# ===================================================================
# 2. TestLoadCalculator
# ===================================================================


class TestLoadCalculator:
    """Tests for load_calculator()."""

    def test_fairchem_returns_callable(self):
        """FAIRChemCalculator dispatch returns a callable (mocked import)."""
        from unittest.mock import MagicMock, patch

        mock_module = MagicMock()
        mock_calc_class = MagicMock()
        mock_calc_class.from_model_checkpoint = MagicMock()
        mock_module.FAIRChemCalculator = mock_calc_class

        config = make_config_dict(Calculator="FAIRChemCalculator")

        with patch.dict("sys.modules", {"fairchem.core": mock_module, "fairchem": MagicMock()}):
            result = load_calculator(config)
        assert result is mock_calc_class.from_model_checkpoint
        assert callable(result)

    def test_unknown_calculator_raises(self):
        """Unknown calculator name raises ValueError."""
        config = make_config_dict(Calculator="UnknownCalc")
        with pytest.raises(ValueError, match="Unknown calculator"):
            load_calculator(config)


# ===================================================================
# 3. TestLoadMethod
# ===================================================================


class TestLoadMethod:
    """Tests for load_method()."""

    def test_neb(self):
        config = make_config_dict(method="NEB")
        fn = load_method(config)
        from saddlemill.nebopt import nebopt
        assert fn is nebopt

    def test_dimer(self):
        config = make_config_dict(method="Dimer")
        fn = load_method(config)
        from saddlemill.dimeropt import dimeropt
        assert fn is dimeropt

    def test_minimization(self):
        config = make_config_dict(method="Minimization")
        fn = load_method(config)
        from saddlemill.geomopt import geomopt
        assert fn is geomopt

    def test_double_minimization(self):
        config = make_config_dict(method="DoubleMinimization")
        fn = load_method(config)
        from saddlemill.geomopt import doublegeomopt
        assert fn is doublegeomopt

    def test_none_raises_value_error(self):
        config = make_config_dict(method=None)
        with pytest.raises(ValueError, match="method.*not set"):
            load_method(config)

    def test_unknown_raises_not_implemented(self):
        config = make_config_dict(method="FooBarMethod")
        with pytest.raises(NotImplementedError, match="not implemented"):
            load_method(config)

    def test_singlepoint_vasp_requires_fpj_1(self):
        """SinglePoint + VASP rejects frames_per_job != 1."""
        config = make_config_dict(method="SinglePoint", Calculator="Vasp",
                                  frames_per_job=3, vasp_command="mpirun vasp_std")
        with pytest.raises(NotImplementedError, match="frames_per_job=1"):
            load_method(config)

    def test_singlepoint_vasp_fpj_1_loads(self):
        """SinglePoint + VASP with frames_per_job=1 loads cleanly."""
        config = make_config_dict(method="SinglePoint", Calculator="Vasp",
                                  frames_per_job=1, vasp_command="mpirun vasp_std")
        fn = load_method(config)
        from saddlemill.geomopt import singlepoint
        assert fn is singlepoint

    def test_singlepoint_vaspinteractive_requires_fpj_1(self):
        """SinglePoint + VaspInteractive also rejects frames_per_job > 1."""
        config = make_config_dict(method="SinglePoint", Calculator="VaspInteractive",
                                  frames_per_job=2, vasp_command="mpirun vasp_std")
        with pytest.raises(NotImplementedError, match="frames_per_job=1"):
            load_method(config)

    @pytest.mark.parametrize("method,section,required_key", [
        ("NEB", "ourNEB", "vasp_command_endpoints"),
        ("Dimer", "ourDimer", "vasp_command"),
        ("Minimization", "ourMinimization", "vasp_command"),
        ("DoubleMinimization", "ourDoubleMinimization", "vasp_command"),
        ("SinglePoint", "ourSinglePoint", "vasp_command"),
    ])
    def test_vasp_command_required(self, method, section, required_key):
        """Missing vasp_command for any VASP method raises a clear ValueError."""
        config = make_config_dict(method=method, Calculator="Vasp")
        # Ensure the required key is not set (None is the default — leave it).
        config[section][required_key] = None
        with pytest.raises(ValueError, match=required_key):
            load_method(config)

    def test_vasp_command_present_loads(self):
        """A method with all required VASP command keys loads cleanly."""
        config = make_config_dict(method="Dimer", Calculator="Vasp",
                                  vasp_command="mpirun -n 64 vasp_std")
        fn = load_method(config)
        from saddlemill.dimeropt import dimeropt
        assert fn is dimeropt


class TestVaspDebugPatterns:
    """Tests for the per-method VASP debug-zip filename regexes added in
    `_get_debug_filename_patterns` (config.py)."""

    def _match_in(self, patterns, name):
        for pat in patterns:
            m = pat.match(name)
            if m:
                return m
        return None

    def test_dimer_vasp_pattern_captures_job_and_attempt(self):
        from saddlemill.config import _get_debug_filename_patterns
        patterns = _get_debug_filename_patterns("Dimer")
        m = self._match_in(patterns, "VASP_3_5/OUTCAR")
        assert m is not None and m.group(1) == "3" and m.group(2) == "5"
        # ERROR_-prefixed entries also match.
        m = self._match_in(patterns, "ERROR_VASP_7_0/INCAR")
        assert m is not None and m.group(1) == "7" and m.group(2) == "0"

    def test_minimization_vasp_pattern_captures_job_only(self):
        from saddlemill.config import _get_debug_filename_patterns
        patterns = _get_debug_filename_patterns("Minimization")
        m = self._match_in(patterns, "VASP_12/OUTCAR")
        assert m is not None and m.group(1) == "12"
        # subunit_id is not captured (lastindex==1).
        assert m.lastindex == 1

    def test_doublemin_vasp_pattern_accepts_negative_side(self):
        from saddlemill.config import _get_debug_filename_patterns
        patterns = _get_debug_filename_patterns("DoubleMinimization")
        # Side -1
        m = self._match_in(patterns, "VASP_4_-1/OUTCAR")
        assert m is not None and m.group(1) == "4"
        # Side 0 (TS)
        m = self._match_in(patterns, "VASP_4_0/INCAR")
        assert m is not None and m.group(1) == "4"
        # Side 1
        m = self._match_in(patterns, "VASP_4_1/POSCAR")
        assert m is not None and m.group(1) == "4"
        # subunit_id (side) is NOT captured — that lets DM fall through to
        # "remove all for matching job" since DM always re-runs all 3 sides.
        assert m.lastindex == 1

    def test_dimer_should_remove_debug_per_attempt(self):
        """Dimer cleans VASP files only for the matching (job, attempt)."""
        from saddlemill.config import (_get_debug_filename_patterns,
                                       _should_remove_debug)
        patterns = _get_debug_filename_patterns("Dimer")
        cleaned = {3: {5}}
        assert _should_remove_debug("VASP_3_5/OUTCAR", patterns, cleaned, "Dimer")
        assert not _should_remove_debug("VASP_3_6/OUTCAR", patterns, cleaned, "Dimer")
        assert not _should_remove_debug("VASP_4_5/OUTCAR", patterns, cleaned, "Dimer")

    def test_doublemin_should_remove_debug_all_sides(self):
        """DM cleans every VASP side dir for the matching job."""
        from saddlemill.config import (_get_debug_filename_patterns,
                                       _should_remove_debug)
        patterns = _get_debug_filename_patterns("DoubleMinimization")
        cleaned = {2: {-1, 1}}  # cleaned subunit_ids = sides
        for side in (-1, 0, 1):
            assert _should_remove_debug(f"VASP_2_{side}/OUTCAR", patterns,
                                        cleaned, "DoubleMinimization"), (
                f"VASP_2_{side} should be removed (DM re-runs all sides)")
        assert not _should_remove_debug("VASP_3_-1/OUTCAR", patterns, cleaned,
                                        "DoubleMinimization")

    def test_minimization_should_remove_debug_per_job(self):
        from saddlemill.config import (_get_debug_filename_patterns,
                                       _should_remove_debug)
        patterns = _get_debug_filename_patterns("Minimization")
        cleaned = {7: {None}}
        assert _should_remove_debug("VASP_7/INCAR", patterns, cleaned, "Minimization")
        assert not _should_remove_debug("VASP_8/INCAR", patterns, cleaned, "Minimization")


# ===================================================================
# 4. TestLoadOptimizer
# ===================================================================


class TestLoadOptimizer:
    """Tests for _load_optimizer() and load_optimizer()."""

    def test_mdmin(self):
        from ase.optimize import MDMin
        assert _load_optimizer("MDMin") is MDMin

    def test_bfgs(self):
        from ase.optimize import BFGS
        assert _load_optimizer("BFGS") is BFGS

    def test_lbfgs(self):
        from ase.optimize import LBFGS
        assert _load_optimizer("LBFGS") is LBFGS

    def test_fire(self):
        from ase.optimize import FIRE
        assert _load_optimizer("FIRE") is FIRE

    def test_case_insensitive(self):
        from ase.optimize import MDMin
        assert _load_optimizer("mdmin") is MDMin
        assert _load_optimizer("MDMIN") is MDMin

    def test_unknown_raises_not_implemented(self):
        with pytest.raises(NotImplementedError, match="not implemented"):
            _load_optimizer("ConjugateGradient")

    def test_neb_returns_tuple(self):
        """NEB method returns (endpoint_opt, neb_opt) tuple."""
        config = make_config_dict(method="NEB", Optimizer="MDMin")
        result = load_optimizer(config)
        assert isinstance(result, tuple)
        assert len(result) == 2
        from ase.optimize import MDMin
        assert result[0] is MDMin
        assert result[1] is MDMin

    def test_neb_separate_endpoint_optimizer(self):
        """NEB with separate endpoint_relax_Optimizer returns different first element."""
        config = make_config_dict(method="NEB", Optimizer="MDMin",
                                  endpoint_relax_Optimizer="LBFGS")
        result = load_optimizer(config)
        from ase.optimize import MDMin, LBFGS
        assert result[0] is LBFGS
        assert result[1] is MDMin

    def test_non_neb_returns_single_optimizer(self):
        """Non-NEB method returns a single optimizer class (not a tuple)."""
        config = make_config_dict(method="Dimer", Optimizer="BFGS")
        result = load_optimizer(config)
        from ase.optimize import BFGS
        assert result is BFGS


# ===================================================================
# 5. TestNormalizeRunJobs
# ===================================================================


class TestNormalizeRunJobs:
    """Tests for _normalize_run_jobs()."""

    def test_single_string(self):
        assert _normalize_run_jobs("remaining") == {"remaining"}

    def test_all_expands(self):
        result = _normalize_run_jobs("all")
        assert result == set(VALID_RUN_CATEGORIES)
        assert result == {"converged", "not_converged", "errored", "remaining"}

    def test_list_input(self):
        result = _normalize_run_jobs(["remaining", "errored"])
        assert result == {"remaining", "errored"}

    def test_alias_not_started(self):
        result = _normalize_run_jobs("not_started")
        assert result == {"remaining"}

    def test_alias_error(self):
        result = _normalize_run_jobs("error")
        assert result == {"errored"}

    def test_alias_in_list(self):
        result = _normalize_run_jobs(["not_started", "error"])
        assert result == {"remaining", "errored"}

    def test_invalid_string_raises(self):
        with pytest.raises(ValueError, match="Invalid run_jobs"):
            _normalize_run_jobs("bogus")

    def test_invalid_in_list_raises(self):
        with pytest.raises(ValueError, match="Invalid run_jobs"):
            _normalize_run_jobs(["remaining", "bogus"])

    def test_invalid_type_raises(self):
        with pytest.raises(ValueError, match="Invalid run_jobs"):
            _normalize_run_jobs(42)


# ===================================================================
# 6. TestCategorizeStatus
# ===================================================================


class TestCategorizeStatus:
    """Tests for _categorize_status()."""

    def test_converged(self):
        assert _categorize_status("converged") == "converged"

    def test_converged_ci(self):
        assert _categorize_status("converged_CI") == "converged"

    def test_converged_after_extension(self):
        assert _categorize_status("converged_after_extension") == "converged"

    def test_error(self):
        assert _categorize_status("error") == "errored"

    def test_error_with_message(self):
        assert _categorize_status("error: CUDA OOM") == "errored"

    def test_error_prefix(self):
        assert _categorize_status("error:something") == "errored"

    def test_not_converged(self):
        assert _categorize_status("not_converged") == "not_converged"

    def test_not_converged_after_extension(self):
        assert _categorize_status("not_converged_after_extension") == "not_converged"

    def test_not_converged_stoprun(self):
        assert _categorize_status("not_converged_StopRun") == "not_converged"

    def test_unknown_status_defaults_to_errored(self):
        assert _categorize_status("garbage") == "errored"


# ===================================================================
# 7. TestGetTrajesAndIndices
# ===================================================================


def _write_traj(path, n_frames):
    """Write a dummy trajectory with n_frames to path."""
    with Trajectory(str(path), "w") as traj:
        for i in range(n_frames):
            atoms = Atoms("H", positions=[[0, 0, float(i)]], cell=[5, 5, 5], pbc=True)
            traj.write(atoms)


class TestGetTrajesAndIndices:
    """Tests for get_trajes_and_indices()."""

    def test_minimization_single_frame(self, tmp_path):
        """Minimization: 1 frame per traj -> 1 job each."""
        traj_dir = tmp_path / "inputs"
        traj_dir.mkdir()
        _write_traj(traj_dir / "a.traj", 1)
        _write_traj(traj_dir / "b.traj", 1)

        config = make_config_dict(method="Minimization", dir_path=str(traj_dir))
        result = get_trajes_and_indices(config)
        assert len(result) == 2
        for item in result:
            assert item[1] == 0  # start_idx
            assert item[2] == 1  # end_idx (nimages=1)

    def test_neb_colon_mode_splitting(self, tmp_path):
        """NEB colon mode: 10 frames / num_frames=5 -> 2 batches."""
        traj_dir = tmp_path / "inputs"
        traj_dir.mkdir()
        _write_traj(traj_dir / "band.traj", 10)

        config = make_config_dict(method="NEB", dir_path=str(traj_dir),
                                  num_frames=5,
                                  images_location_in_input_traj=":")
        result = get_trajes_and_indices(config)
        assert len(result) == 2
        assert result[0][1] == 0 and result[0][2] == 5   # batch 1: [0, 5)
        assert result[1][1] == 5 and result[1][2] == 10  # batch 2: [5, 10)

    def test_neb_colon_mode_indivisible_raises(self, tmp_path):
        """NEB colon mode: frames not divisible by num_frames raises ValueError."""
        traj_dir = tmp_path / "inputs"
        traj_dir.mkdir()
        _write_traj(traj_dir / "band.traj", 7)

        config = make_config_dict(method="NEB", dir_path=str(traj_dir),
                                  num_frames=5,
                                  images_location_in_input_traj=":")
        with pytest.raises(ValueError, match="Can't divide"):
            get_trajes_and_indices(config)

    def test_neb_mode_0(self, tmp_path):
        """NEB mode 0: takes first num_frames images."""
        traj_dir = tmp_path / "inputs"
        traj_dir.mkdir()
        _write_traj(traj_dir / "band.traj", 20)

        config = make_config_dict(method="NEB", dir_path=str(traj_dir),
                                  num_frames=10,
                                  images_location_in_input_traj=0)
        result = get_trajes_and_indices(config)
        assert len(result) == 1
        assert result[0][1] == 0
        assert result[0][2] == 10

    def test_neb_mode_minus1(self, tmp_path):
        """NEB mode -1: takes last num_frames images."""
        traj_dir = tmp_path / "inputs"
        traj_dir.mkdir()
        _write_traj(traj_dir / "band.traj", 20)

        config = make_config_dict(method="NEB", dir_path=str(traj_dir),
                                  num_frames=10,
                                  images_location_in_input_traj=-1)
        result = get_trajes_and_indices(config)
        assert len(result) == 1
        assert result[0][1] == 10  # 20 - 10
        assert result[0][2] == 20

    def test_recursive_scan(self, tmp_path):
        """Traj files in subdirectories are found."""
        sub = tmp_path / "inputs" / "subdir"
        sub.mkdir(parents=True)
        _write_traj(tmp_path / "inputs" / "a.traj", 1)
        _write_traj(sub / "b.traj", 1)

        config = make_config_dict(method="Minimization",
                                  dir_path=str(tmp_path / "inputs"))
        result = get_trajes_and_indices(config)
        assert len(result) == 2

    def test_empty_directory(self, tmp_path):
        """No traj files returns empty list."""
        traj_dir = tmp_path / "empty"
        traj_dir.mkdir()
        config = make_config_dict(method="Minimization", dir_path=str(traj_dir))
        result = get_trajes_and_indices(config)
        assert result == []


# ===================================================================
# 8. TestArchiveAndClean
# ===================================================================


class TestArchiveAndClean:
    """Tests for archive_and_clean_csvs()."""

    def _setup_csv_dir(self, tmp_path, method_name, csv_rows):
        """Create a status CSV dir with given rows and return (config_dict, csv_path).

        csv_rows: list of strings (one per CSV line), each line is "job_id,rank,..."
        """
        status_dir = tmp_path / f"{method_name}_status_csvs"
        status_dir.mkdir()
        csv_path = status_dir / "status_rank_0.csv"
        csv_path.write_text("\n".join(csv_rows) + "\n")
        return csv_path

    def test_archive_creates_zip(self, tmp_path, monkeypatch):
        """Archiving creates a previous_0.zip with the original CSV."""
        monkeypatch.chdir(tmp_path)
        method = "Dimer"
        rows = [
            "0,0,0,converged",
            "0,0,1,not_converged",
            "1,0,0,converged",
        ]
        csv_path = self._setup_csv_dir(tmp_path, method, rows)

        config = make_config_dict(method=method, run_jobs="converged")
        # Clean job_id=0 entries matching "converged"
        cleaned = archive_and_clean_csvs(config, [0], {"converged"})

        zip_path = tmp_path / f"{method}_status_csvs" / "previous_0.zip"
        assert zip_path.exists()
        with zipfile.ZipFile(str(zip_path), "r") as zf:
            names = zf.namelist()
            assert "status_rank_0.csv" in names

    def test_cleaning_removes_matching_rows(self, tmp_path, monkeypatch):
        """Only rows matching categories_to_clean for selected job_ids are removed."""
        monkeypatch.chdir(tmp_path)
        method = "Dimer"
        rows = [
            "0,0,0,converged",
            "0,0,1,not_converged",
            "1,0,0,converged",
        ]
        csv_path = self._setup_csv_dir(tmp_path, method, rows)

        config = make_config_dict(method=method, run_jobs="converged")
        cleaned = archive_and_clean_csvs(config, [0], {"converged"})

        # job_id=0, attempt_id=0 was converged -> cleaned
        assert 0 in cleaned
        assert 0 in cleaned[0]

        # CSV should still have the not_converged row for job 0 and the row for job 1
        import csv
        with open(str(csv_path)) as f:
            remaining_rows = [row for row in csv.reader(f) if row]
        assert len(remaining_rows) == 2
        # job_id=0, attempt=1 (not_converged) should remain
        job0_rows = [r for r in remaining_rows if int(r[0]) == 0]
        assert len(job0_rows) == 1
        assert int(job0_rows[0][2]) == 1  # attempt_id=1

    def test_incremental_numbering(self, tmp_path, monkeypatch):
        """Second archive creates previous_1.zip (not overwriting previous_0.zip)."""
        monkeypatch.chdir(tmp_path)
        method = "NEB"
        rows = [
            "0,0,0,converged",
            "1,0,0,not_converged",
        ]
        self._setup_csv_dir(tmp_path, method, rows)
        config = make_config_dict(method=method, run_jobs="converged")

        # First archive
        archive_and_clean_csvs(config, [0], {"converged"})
        zip0 = tmp_path / f"{method}_status_csvs" / "previous_0.zip"
        assert zip0.exists()

        # Re-create CSV (simulating a new run that produced results)
        csv_path = tmp_path / f"{method}_status_csvs" / "status_rank_0.csv"
        csv_path.write_text("1,0,0,not_converged\n")

        # Second archive
        archive_and_clean_csvs(config, [1], {"not_converged"})
        zip1 = tmp_path / f"{method}_status_csvs" / "previous_1.zip"
        assert zip1.exists()
        assert zip0.exists()  # previous_0 still there

    def test_no_job_ids_returns_empty(self, tmp_path, monkeypatch):
        """Empty job_ids list returns empty dict without creating archive."""
        monkeypatch.chdir(tmp_path)
        method = "Minimization"
        self._setup_csv_dir(tmp_path, method, ["0,0,converged"])
        config = make_config_dict(method=method)
        cleaned = archive_and_clean_csvs(config, [], {"converged"})
        assert cleaned == {}

    def test_no_matching_entries_returns_empty(self, tmp_path, monkeypatch):
        """When no CSV rows match the given job_ids, return empty without archiving."""
        monkeypatch.chdir(tmp_path)
        method = "Minimization"
        self._setup_csv_dir(tmp_path, method, ["0,0,converged"])
        config = make_config_dict(method=method)
        cleaned = archive_and_clean_csvs(config, [99], {"converged"})
        assert cleaned == {}
        # No archive created
        zip0 = tmp_path / f"{method}_status_csvs" / "previous_0.zip"
        assert not zip0.exists()

    def test_all_rows_removed_deletes_csv(self, tmp_path, monkeypatch):
        """When all rows in a CSV are cleaned, the CSV file is deleted."""
        monkeypatch.chdir(tmp_path)
        method = "Minimization"
        csv_path = self._setup_csv_dir(tmp_path, method, ["0,0,converged"])
        config = make_config_dict(method=method)
        cleaned = archive_and_clean_csvs(config, [0], {"converged"})
        assert not csv_path.exists()
        assert 0 in cleaned
