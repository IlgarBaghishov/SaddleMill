"""GPU integration tests for saddlemill/geomopt.py using FAIRChem calculator."""

import copy

import numpy as np
import pytest
from ase.calculators.singlepoint import SinglePointCalculator
from ase.io import Trajectory
from ase.optimize import MDMin

from saddlemill.geomopt import geomopt, doublegeomopt
from tests.conftest import make_config_dict


@pytest.mark.gpu
class TestGeomopt:
    """Integration tests that run geomopt() with FAIRChem on real structures."""

    def _setup_dirs(self, tmp_path, monkeypatch):
        """Create output directories and chdir to tmp_path."""
        monkeypatch.chdir(tmp_path)
        (tmp_path / "Minimization_status_csvs").mkdir()
        (tmp_path / "Minimization_trajes").mkdir()
        (tmp_path / "Minimization_debug_zips").mkdir()

    def _make_config(self, steps=50, relax_cell=False, **overrides):
        config = make_config_dict(method="Minimization", steps=steps, fmax=0.05,
                                  Optimizer="MDMin")
        config["ourMinimization"]["relax_cell"] = relax_cell
        for k, v in overrides.items():
            if k in config["Main"]:
                config["Main"][k] = v
        return config

    def _prepare_atoms(self, atoms):
        """Deep-copy atoms and wrap info into orig_info."""
        atoms = atoms.copy()
        atoms.info = {"orig_info": dict(atoms.info)}
        return atoms

    def _read_output_traj(self, tmp_path):
        """Read output trajectory frames.

        Asserts every frame round-trips with energy/forces in calc.results
        (regression guard for the SinglePointCalculator + atoms.wrap() race).
        """
        traj_path = tmp_path / "Minimization_trajes" / "collected_opt_rank_0.traj"
        assert traj_path.exists(), f"Output trajectory not found at {traj_path}"
        traj = Trajectory(str(traj_path), "r")
        for idx, frame in enumerate(traj):
            assert frame.calc is not None, f"Frame {idx} has no calculator after read-back"
            assert 'energy' in frame.calc.results, (
                f"Frame {idx} missing energy — SPC+wrap race regression")
            assert 'forces' in frame.calc.results, (
                f"Frame {idx} missing forces — SPC+wrap race regression")
        return traj

    def _read_csv(self, tmp_path):
        """Read CSV status lines."""
        csv_path = tmp_path / "Minimization_status_csvs" / "status_rank_0.csv"
        assert csv_path.exists(), f"Status CSV not found at {csv_path}"
        return csv_path.read_text().strip().splitlines()

    # ----- Test: basic minimization -----

    def test_basic_minimization(self, tmp_path, monkeypatch, fairchem_calc, minimization_input):
        """minimization_input, 50 steps.

        Verify: output traj with converged key, energy is finite.
        """
        self._setup_dirs(tmp_path, monkeypatch)
        config = self._make_config(steps=50)
        atoms = self._prepare_atoms(minimization_input)

        geomopt(0, config, atoms, fairchem_calc, MDMin,
                consecutive_errors=[0], executorlib_worker_id=0)

        csv_lines = self._read_csv(tmp_path)
        assert len(csv_lines) == 1

        with self._read_output_traj(tmp_path) as traj:
            assert len(traj) == 1
            frame = traj[0]
            assert "converged" in frame.info
            assert "src_index" in frame.info
            assert frame.info["src_index"] == 0
            assert frame.info["converged"] in (0, 1)

    # ----- Test: cell relaxation -----

    def test_cell_relaxation(self, tmp_path, monkeypatch, fairchem_calc, bulk_crystal):
        """bulk_crystal, relax_cell=True, 30 steps."""
        self._setup_dirs(tmp_path, monkeypatch)
        config = self._make_config(steps=30, relax_cell=True)
        atoms = self._prepare_atoms(bulk_crystal)

        geomopt(0, config, atoms, fairchem_calc, MDMin,
                consecutive_errors=[0], executorlib_worker_id=0)

        csv_lines = self._read_csv(tmp_path)
        assert len(csv_lines) == 1

        with self._read_output_traj(tmp_path) as traj:
            assert len(traj) == 1
            frame = traj[0]
            assert "converged" in frame.info

    # ----- Test: continuation data -----

    def test_continuation_data(self, tmp_path, monkeypatch, fairchem_calc, minimization_input):
        """Run geomopt once, read result, pass as continuation_data in second run."""
        self._setup_dirs(tmp_path, monkeypatch)
        config = self._make_config(steps=20)
        atoms = self._prepare_atoms(minimization_input)

        # First run
        geomopt(0, config, atoms, fairchem_calc, MDMin,
                consecutive_errors=[0], executorlib_worker_id=0)

        # Read back the result
        with self._read_output_traj(tmp_path) as traj:
            result_atoms = traj[0].copy()

        # Prepare continuation: wrap into orig_info (as _sanitize_with_continuation does)
        result_atoms.info = {"orig_info": dict(result_atoms.info)}

        # Second run with continuation_data
        # Need fresh output dirs (or append mode handles it)
        config2 = self._make_config(steps=20)
        geomopt(1, config2, self._prepare_atoms(minimization_input), fairchem_calc, MDMin,
                consecutive_errors=[0], executorlib_worker_id=0,
                continuation_data=result_atoms)

        # Should now have 2 frames total in the traj (appended)
        with self._read_output_traj(tmp_path) as traj:
            assert len(traj) == 2
            # Second frame should have src_index=1
            assert traj[1].info["src_index"] == 1


@pytest.mark.gpu
class TestDoubleGeomopt:
    """Integration tests that run doublegeomopt() with FAIRChem."""

    def _setup_dirs(self, tmp_path, monkeypatch):
        """Create output directories and chdir to tmp_path."""
        monkeypatch.chdir(tmp_path)
        (tmp_path / "DoubleMinimization_status_csvs").mkdir()
        (tmp_path / "DoubleMinimization_trajes").mkdir()
        (tmp_path / "DoubleMinimization_debug_zips").mkdir()

    def _make_config(self, steps=50, relax_cell=False, **overrides):
        config = make_config_dict(method="DoubleMinimization", steps=steps, fmax=0.05,
                                  Optimizer="MDMin")
        config["ourDoubleMinimization"]["relax_cell"] = relax_cell
        for k, v in overrides.items():
            if k in config["Main"]:
                config["Main"][k] = v
        return config

    def _read_output_traj(self, tmp_path):
        """Read output trajectory frames.

        Asserts every frame round-trips with energy/forces in calc.results
        (regression guard for the SinglePointCalculator + atoms.wrap() race).
        """
        traj_path = tmp_path / "DoubleMinimization_trajes" / "collected_opt_rank_0.traj"
        assert traj_path.exists(), f"Output trajectory not found at {traj_path}"
        traj = Trajectory(str(traj_path), "r")
        for idx, frame in enumerate(traj):
            assert frame.calc is not None, f"Frame {idx} has no calculator after read-back"
            assert 'energy' in frame.calc.results, (
                f"Frame {idx} missing energy — SPC+wrap race regression")
            assert 'forces' in frame.calc.results, (
                f"Frame {idx} missing forces — SPC+wrap race regression")
        return traj

    def _read_csv(self, tmp_path):
        """Read CSV status lines."""
        csv_path = tmp_path / "DoubleMinimization_status_csvs" / "status_rank_0.csv"
        assert csv_path.exists(), f"Status CSV not found at {csv_path}"
        return csv_path.read_text().strip().splitlines()

    # ----- Test: basic double minimization -----

    def test_basic_double_minimization(self, tmp_path, monkeypatch, fairchem_calc, converged_ts_atoms):
        """converged_ts_atoms, 50 steps.

        Verify: 3 frames (side -1, 0, 1), is_reaction key, bond counts present.
        """
        self._setup_dirs(tmp_path, monkeypatch)
        config = self._make_config(steps=50)

        # Prepare atoms: wrap current info into orig_info
        # converged_ts_atoms has eigenmode, converged, src_index at top level
        atoms = converged_ts_atoms.copy()
        atoms.info = {"orig_info": dict(converged_ts_atoms.info)}

        doublegeomopt(0, config, atoms, fairchem_calc, MDMin,
                      consecutive_errors=[0], executorlib_worker_id=0)

        csv_lines = self._read_csv(tmp_path)
        assert len(csv_lines) == 2  # one per side

        with self._read_output_traj(tmp_path) as traj:
            assert len(traj) == 3  # min1, TS, min2
            sides = [traj[idx].info["side"] for idx in range(3)]
            assert sorted(sides) == [-1, 0, 1]

            for idx in range(3):
                frame = traj[idx]
                assert "is_reaction" in frame.info
                assert "n_formed_bonds" in frame.info
                assert "n_broken_bonds" in frame.info
                assert "src_index" in frame.info
                assert frame.info["src_index"] == 0

    # ----- Test: entries_to_run one side -----

    def test_entries_to_run_one_side(self, tmp_path, monkeypatch, fairchem_calc, converged_ts_atoms):
        """Run full doublegeomopt first, then rerun with entries_to_run={1}.

        Verify: second run only processes side=1 but still writes all 3 frames.
        """
        self._setup_dirs(tmp_path, monkeypatch)
        config = self._make_config(steps=50)

        # First full run
        atoms = converged_ts_atoms.copy()
        atoms.info = {"orig_info": dict(converged_ts_atoms.info)}

        doublegeomopt(0, config, atoms, fairchem_calc, MDMin,
                      consecutive_errors=[0], executorlib_worker_id=0)

        # Read back results for continuation_data
        with self._read_output_traj(tmp_path) as traj:
            all_frames = [traj[idx] for idx in range(len(traj))]

        continuation_data = {}
        for frame in all_frames:
            side = frame.info["side"]
            if side != 0:  # only min1 and min2, not TS
                cont_frame = frame.copy()
                # Mirror production: extract_previous_results hands frames that
                # still carry their SinglePointCalculator (only .info is
                # rewrapped). Regression guard: the kept-side branch must read
                # E/F from this frame BEFORE .copy() drops the calculator.
                cont_frame.calc = SinglePointCalculator(
                    cont_frame, energy=frame.get_potential_energy(),
                    forces=frame.get_forces())
                cont_frame.info = {"orig_info": dict(frame.info)}
                continuation_data[side] = cont_frame

        # Second run: only rerun side=1, keep side=-1 from continuation_data
        atoms2 = converged_ts_atoms.copy()
        atoms2.info = {"orig_info": dict(converged_ts_atoms.info)}

        doublegeomopt(1, config, atoms2, fairchem_calc, MDMin,
                      consecutive_errors=[0], executorlib_worker_id=0,
                      entries_to_run={1},
                      continuation_data=continuation_data)

        # Should have appended 3 more frames (min1+TS+min2 for job_id=1)
        with self._read_output_traj(tmp_path) as traj:
            assert len(traj) == 6  # 3 from first run + 3 from second run
            # Second batch (last 3 frames) should have src_index=1
            for idx in range(3, 6):
                assert traj[idx].info["src_index"] == 1

        # CSV should have 2 lines from first run + 1 from second run (only side=1)
        csv_lines = self._read_csv(tmp_path)
        assert len(csv_lines) == 3  # 2 from first + 1 from second
