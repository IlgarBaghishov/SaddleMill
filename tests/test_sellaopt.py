"""CPU tests for the Sella (P-RFO) saddle-search method.

These run on EMT rather than FAIRChem so the whole method — config plumbing,
attempt generation, eigenmode seeding, force-call accounting, output schema —
is covered without a GPU. The GPU/FAIRChem path is exercised by the same code.
"""
import csv

import numpy as np
import pytest
from ase.build import fcc100, add_adsorbate
from ase.calculators.emt import EMT
from ase.constraints import FixAtoms
from ase.io import Trajectory

from tests.conftest import make_config_dict

sella = pytest.importorskip("sella", reason="sella is the [sella] extra")


def _cu_slab():
    """Small Cu(100) slab with an Au adatom and a fixed bottom layer."""
    slab = fcc100("Cu", size=(2, 2, 3), vacuum=8.0)
    add_adsorbate(slab, "Au", 1.7, "hollow")
    zs = slab.positions[:, 2]
    fixed = [i for i in range(len(slab)) if zs[i] < zs.min() + 1.0]
    slab.set_constraint(FixAtoms(indices=fixed))
    return slab


def _prepare(atoms):
    """Mirror load_and_sanitize(): stash the original info under orig_info."""
    atoms = atoms.copy()
    atoms.info = {"orig_info": dict(atoms.info)}
    return atoms


def _make_config(**overrides):
    config = make_config_dict(method="Sella", steps=25, fmax=0.05)
    config["ourSella"]["dataset_type"] = "bulk"
    config["ourSella"]["reaction_types"] = "initial_guess"
    config["ourSella"]["num_attempts_per_type"] = 1
    config["ourSella"]["supercell"] = False
    for k, v in overrides.items():
        if k in config["ourSella"]:
            config["ourSella"][k] = v
        elif k in config["Main"]:
            config["Main"][k] = v
        else:
            config.setdefault("Sella", {})[k] = v
    return config


def _setup_dirs(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    for d in ("Sella_status_csvs", "Sella_trajes", "Sella_debug_zips"):
        (tmp_path / d).mkdir()


# --------------------------------------------------------------------------
# Config plumbing
# --------------------------------------------------------------------------

class TestSellaConfig:
    def test_load_method_returns_sellaopt(self):
        from saddlemill.config import load_method
        config = _make_config()
        assert load_method(config).__name__ == "sellaopt"

    def test_oursella_defaults_exist(self):
        from saddlemill.config import ConfigManager
        our = ConfigManager.DEFAULTS["ourSella"]
        for key in ("dataset_type", "reaction_types", "num_attempts_per_type",
                    "supercell", "delocalization_threshold", "check_index",
                    "vasp_command", "vasp_ncore"):
            assert key in our, f"[ourSella] missing {key}"

    def test_oursella_mirrors_ourdimer_shared_knobs(self):
        """The two methods share the attempt machinery, so the knobs that drive
        it must exist in both sections under the same names."""
        from saddlemill.config import ConfigManager
        shared = {"dataset_type", "reaction_types", "num_attempts_per_type",
                  "gaussian_swap_prob", "ring_sizes", "supercell",
                  "delocalization_threshold", "extension_check_fmax",
                  "extension_check_curvature", "vasp_command", "vasp_ncore"}
        dimer = set(ConfigManager.DEFAULTS["ourDimer"])
        sella_keys = set(ConfigManager.DEFAULTS["ourSella"])
        assert shared <= dimer and shared <= sella_keys

    def test_dimer_only_knobs_absent_from_oursella(self):
        """Rotation-engine knobs are meaningless for P-RFO and must not leak in."""
        from saddlemill.config import ConfigManager
        sella_keys = set(ConfigManager.DEFAULTS["ourSella"])
        assert not ({"engine", "kappa_beta", "kappa_recover_fmax"} & sella_keys)

    def test_vasp_requires_command(self):
        from saddlemill.config import load_method
        config = _make_config()
        config["Main"]["Calculator"] = "Vasp"
        with pytest.raises(ValueError, match="ourSella"):
            load_method(config)

    def test_subunit_info_matches_dimer(self):
        from saddlemill.config import _get_subunit_config
        assert _get_subunit_config("Sella") == _get_subunit_config("Dimer")


class TestAttemptSectionResolution:
    """get_attempts() must read the *active* method's our*-section."""

    def test_sella_reads_oursella(self):
        from saddlemill.dimertools.structure_edit import _attempts_cfg
        config = _make_config()
        config["ourSella"]["dataset_type"] = "oc"
        config["ourDimer"]["dataset_type"] = "bulk"
        assert _attempts_cfg(config)["dataset_type"] == "oc"

    def test_dimer_still_reads_ourdimer(self):
        from saddlemill.dimertools.structure_edit import _attempts_cfg
        config = make_config_dict(method="Dimer")
        config["ourDimer"]["dataset_type"] = "bulk"
        config.setdefault("ourSella", {})["dataset_type"] = "oc"
        assert _attempts_cfg(config)["dataset_type"] == "bulk"

    def test_error_message_names_active_section(self):
        from saddlemill.dimertools.structure_edit import get_attempts
        config = _make_config()
        config["ourSella"]["reaction_types"] = "vacancy"  # not initial_guess
        config["ourSella"]["dataset_type"] = "bulk"
        config["ourSella"]["reaction_types"] = None
        with pytest.raises(ValueError, match="ourSella"):
            get_attempts(_prepare(_cu_slab()), config)


# --------------------------------------------------------------------------
# Eigenmode seeding — the load-bearing throughput trick
# --------------------------------------------------------------------------

class TestEigenmodeSeeding:
    def test_seed_projects_into_free_dof_subspace(self):
        """The stored mode is 3N, but Sella's v0 lives in the free-DOF basis."""
        from sella import Sella
        from saddlemill.sellaopt import _seed_eigenmode

        atoms = _cu_slab()
        atoms.calc = EMT()
        dyn = Sella(atoms, order=1, internal=False, logfile=None)

        mode = np.zeros((len(atoms), 3))
        mode[-1, 2] = 1.0  # move the adatom — a genuinely free DOF
        assert _seed_eigenmode(dyn, mode) is True

        nfree = dyn.pes.get_Ufree().shape[1]
        assert dyn.pes.v0.shape == (nfree,)
        assert nfree < 3 * len(atoms), "FixAtoms should reduce the free-DOF count"
        assert np.isclose(np.linalg.norm(dyn.pes.v0), 1.0)

    def test_seed_rejected_when_mode_is_entirely_constrained(self):
        """A mode that only moves fixed atoms is useless and must not be installed."""
        from sella import Sella
        from saddlemill.sellaopt import _seed_eigenmode

        atoms = _cu_slab()
        atoms.calc = EMT()
        dyn = Sella(atoms, order=1, internal=False, logfile=None)

        fixed = atoms.constraints[0].get_indices()
        mode = np.zeros((len(atoms), 3))
        mode[fixed, 2] = 1.0
        assert _seed_eigenmode(dyn, mode) is False

    def test_seed_rejected_on_shape_mismatch(self):
        from sella import Sella
        from saddlemill.sellaopt import _seed_eigenmode

        atoms = _cu_slab()
        atoms.calc = EMT()
        dyn = Sella(atoms, order=1, internal=False, logfile=None)
        assert _seed_eigenmode(dyn, np.zeros((len(atoms) + 5, 3))) is False

    def test_none_eigenmode_is_a_noop(self):
        from sella import Sella
        from saddlemill.sellaopt import _seed_eigenmode

        atoms = _cu_slab()
        atoms.calc = EMT()
        dyn = Sella(atoms, order=1, internal=False, logfile=None)
        assert _seed_eigenmode(dyn, None) is False


# --------------------------------------------------------------------------
# Force-call accounting
# --------------------------------------------------------------------------

class TestForceCallAccounting:
    def test_neval_exceeds_step_count(self):
        """`n_force_calls` must be the true evaluation count (pes.neval), which
        is strictly larger than the optimizer step count because Sella
        re-diagonalizes periodically. Recording steps instead would understate
        Sella's cost and make a Dimer-vs-Sella comparison invalid."""
        from sella import Sella

        atoms = _cu_slab()
        atoms.calc = EMT()
        dyn = Sella(atoms, order=1, internal=False, logfile=None)
        dyn.run(fmax=0.05, steps=10)

        steps = dyn.get_number_of_steps()
        neval = dyn.pes.neval
        assert steps > 0
        assert neval > steps, (
            f"expected true force calls ({neval}) > optimizer steps ({steps})")


# --------------------------------------------------------------------------
# Index verification
# --------------------------------------------------------------------------

class TestHessianIndex:
    def test_minimum_has_index_zero(self):
        """A relaxed minimum must have no negative curvature."""
        from ase.optimize import BFGS
        from saddlemill.tools import hessian_index

        atoms = _cu_slab()
        atoms.calc = EMT()
        BFGS(atoms, logfile=None).run(fmax=0.01, steps=200)

        eigs, nneg = hessian_index(atoms, nev=4)
        assert nneg == 0, f"relaxed minimum reported index {nneg} (eigs={eigs})"

    def test_positions_restored_after_check(self):
        """The finite-difference probe must leave the geometry untouched."""
        from saddlemill.tools import hessian_index

        atoms = _cu_slab()
        atoms.calc = EMT()
        before = atoms.get_positions().copy()
        hessian_index(atoms, nev=2)
        assert np.allclose(atoms.get_positions(), before)


# --------------------------------------------------------------------------
# End-to-end
# --------------------------------------------------------------------------

class TestSellaoptEndToEnd:
    def test_run_writes_traj_and_csv(self, tmp_path, monkeypatch):
        from saddlemill.sellaopt import sellaopt

        _setup_dirs(tmp_path, monkeypatch)
        config = _make_config(steps=15)
        atoms = _prepare(_cu_slab())

        sellaopt(0, config, atoms, EMT(), consecutive_errors=[0],
                 executorlib_worker_id=0)

        traj_path = tmp_path / "Sella_trajes" / "collected_ts_rank_0.traj"
        assert traj_path.exists()
        frames = list(Trajectory(str(traj_path)))
        assert len(frames) == 1

        f = frames[0]
        # Output schema parity with Dimer.
        for key in ("eigenmode", "n_force_calls", "converged", "src_index",
                    "attempt_id", "stoprun", "selected_index", "reaction_type",
                    "status", "task_name"):
            assert key in f.info, f"missing .info[{key!r}]"
        # Sella-specific additions.
        assert "n_steps" in f.info and "eigenmode_seeded" in f.info
        assert f.info["n_force_calls"] > 0
        assert f.calc is not None and "energy" in f.calc.results

        csv_path = tmp_path / "Sella_status_csvs" / "status_rank_0.csv"
        rows = list(csv.reader(csv_path.open()))
        assert len(rows) == 1
        # job_id, rank, attempt, selected_index, n_force_calls, status
        assert len(rows[0]) == 6
        assert int(rows[0][4]) > 0, "CSV must record true force calls"

    def test_stored_eigenmode_is_seeded(self, tmp_path, monkeypatch):
        """A run on an existing saddle carries an eigenmode; it must be used."""
        from saddlemill.sellaopt import sellaopt

        _setup_dirs(tmp_path, monkeypatch)
        config = _make_config(steps=5)

        slab = _cu_slab()
        mode = np.zeros((len(slab), 3))
        mode[-1, 2] = 1.0
        slab.info["eigenmode"] = mode
        atoms = _prepare(slab)

        sellaopt(0, config, atoms, EMT(), consecutive_errors=[0],
                 executorlib_worker_id=0)

        frames = list(Trajectory(str(tmp_path / "Sella_trajes" / "collected_ts_rank_0.traj")))
        assert frames[0].info["eigenmode_seeded"] == 1

    def test_index_check_records_nneg(self, tmp_path, monkeypatch):
        from saddlemill.sellaopt import sellaopt

        _setup_dirs(tmp_path, monkeypatch)
        # A large fmax makes the very first geometry "converged", so the index
        # check runs on a well-defined structure and the test stays fast.
        config = _make_config(steps=5, fmax=50.0, check_index=True)
        config["ourSella"]["index_nev"] = 2
        atoms = _prepare(_cu_slab())

        sellaopt(0, config, atoms, EMT(), consecutive_errors=[0],
                 executorlib_worker_id=0)

        frames = list(Trajectory(str(tmp_path / "Sella_trajes" / "collected_ts_rank_0.traj")))
        assert frames[0].info["status"].startswith("converged")
        assert "nneg" in frames[0].info

    def test_error_is_logged_not_raised(self, tmp_path, monkeypatch):
        """A failing attempt must be caught per-attempt, like the Dimer's."""
        from saddlemill.sellaopt import sellaopt

        _setup_dirs(tmp_path, monkeypatch)
        config = _make_config(steps=5)

        class Boom(EMT):
            def calculate(self, *a, **k):
                raise RuntimeError("boom")

        atoms = _prepare(_cu_slab())
        sellaopt(0, config, atoms, Boom(), consecutive_errors=[0],
                 executorlib_worker_id=0)

        rows = list(csv.reader((tmp_path / "Sella_status_csvs" / "status_rank_0.csv").open()))
        assert len(rows) == 1
        assert rows[0][-1].startswith("error")

    def test_consecutive_errors_increment_then_reset(self, tmp_path, monkeypatch):
        from saddlemill.sellaopt import sellaopt

        _setup_dirs(tmp_path, monkeypatch)
        config = _make_config(steps=5)

        class Boom(EMT):
            def calculate(self, *a, **k):
                raise RuntimeError("boom")

        counter = [0]
        sellaopt(0, config, _prepare(_cu_slab()), Boom(), consecutive_errors=counter,
                 executorlib_worker_id=0)
        assert counter[0] == 1

        sellaopt(1, config, _prepare(_cu_slab()), EMT(), consecutive_errors=counter,
                 executorlib_worker_id=0)
        assert counter[0] == 0


class TestAttemptGeometryParity:
    """Dimer and Sella must see identical attempt geometries for the same seed.

    This is what makes a head-to-head throughput comparison of the two
    optimizers valid: any difference in outcome is then the optimizer, not a
    different starting structure.
    """

    def test_same_seed_same_geometry(self):
        import random
        from saddlemill.dimertools.structure_edit import get_attempts

        base = _prepare(_cu_slab())

        def gen(method, section):
            config = make_config_dict(method=method)
            config[section]["dataset_type"] = "oc"
            config[section]["reaction_types"] = "adsorbate_atom"
            config[section]["num_attempts_per_type"] = 2
            config[section]["supercell"] = False
            config["DimerControl"]["number_of_displacement_atoms"] = 2
            random.seed(7); np.random.seed(7)
            return get_attempts(base.copy(), config)

        # tag=2 marks the adsorbate for the OC reaction types.
        tags = np.zeros(len(base), dtype=int); tags[-1] = 2
        base.set_tags(tags)

        d_imgs, d_disp, d_idx = gen("Dimer", "ourDimer")
        s_imgs, s_disp, s_idx = gen("Sella", "ourSella")

        assert d_idx == s_idx
        assert len(d_imgs) == len(s_imgs)
        for a, b in zip(d_imgs, s_imgs):
            if a is None or b is None:
                assert a is None and b is None
                continue
            assert np.allclose(a.get_positions(), b.get_positions())
