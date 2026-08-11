"""Tests for SinglePoint VASP resume: banking mid-run VTST-dimer state on a
wall-kill and seeding the resumed run's POSCAR/MODECAR from it.

Covers tools.bank_singlepoint_vasp_restarts / make_sp_resume_seed_writer /
resolve_vasp_calc_class(extra_writers=...) and geomopt._sp_resume_seed_writers.
"""

import os
import numpy as np
import pytest
from ase.build import bulk

from tests.conftest import make_config_dict
from saddlemill.tools import (
    _poscar_natoms,
    _modecar_natoms,
    _count_oszicar_ionic_steps,
    bank_singlepoint_vasp_restarts,
    make_sp_resume_seed_writer,
    resolve_vasp_calc_class,
)
from saddlemill.geomopt import _sp_resume_seed_writers


# ---------------------------------------------------------------------------
# Minimal VASP-file writers (POSCAR/CONTCAR/CENTCAR share the same format).
# ---------------------------------------------------------------------------

def _write_poscar(path, natoms, tag="A"):
    """Write a minimal valid VASP5 POSCAR with `natoms` X atoms.

    `tag` is embedded in the comment/coords so callers can prove which file's
    content ended up where.
    """
    lines = [f"seed-{tag}", "1.0",
             "10.0 0.0 0.0", "0.0 10.0 0.0", "0.0 0.0 10.0",
             "X", str(natoms), "Cartesian"]
    for k in range(natoms):
        lines.append(f"{0.1 * k:.6f} 0.000000 0.000000")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def _write_modecar(path, natoms, tag=1.0):
    with open(path, "w") as f:
        for _ in range(natoms):
            f.write(f"{tag:.6f} 0.000000 0.000000\n")


def _write_oszicar(path, nsteps):
    with open(path, "w") as f:
        for s in range(1, nsteps + 1):
            f.write(f"   {s} F= -.1E+03 E0= -.1E+03  d E =-.1E+00  mag=  1.0\n")


def _mk_workdir(tmp_path, job_id, files):
    """Create VASP_{job_id}/ under tmp_path and return its path.

    `files` maps a filename to either bytes/str (written verbatim) or an int
    (POSCAR-like: that many atoms) via the helpers above -- but here we just
    write raw content so tests stay explicit.
    """
    d = tmp_path / f"VASP_{job_id}"
    d.mkdir()
    return d


# ---------------------------------------------------------------------------
# Parsers
# ---------------------------------------------------------------------------

class TestParsers:
    def test_poscar_natoms_vasp5(self, tmp_path):
        p = tmp_path / "POSCAR"
        _write_poscar(p, 7)
        assert _poscar_natoms(str(p)) == 7

    def test_poscar_natoms_multispecies(self, tmp_path):
        p = tmp_path / "CONTCAR"
        # species line + counts line "8 2 6 16 1" == 33 atoms (real VASP_5 shape)
        p.write_text("c\n1.0\n1 0 0\n0 1 0\n0 0 1\nK Er Np Te Ag\n8 2 6 16 1\nDirect\n"
                     + "0 0 0\n" * 33)
        assert _poscar_natoms(str(p)) == 33

    def test_poscar_natoms_empty_or_missing(self, tmp_path):
        empty = tmp_path / "empty"
        empty.write_text("")
        assert _poscar_natoms(str(empty)) is None
        assert _poscar_natoms(str(tmp_path / "nope")) is None
        assert _poscar_natoms(None) is None

    def test_modecar_natoms(self, tmp_path):
        p = tmp_path / "MODECAR"
        _write_modecar(p, 5)
        assert _modecar_natoms(str(p)) == 5

    def test_modecar_natoms_bad_rows(self, tmp_path):
        p = tmp_path / "MODECAR"
        p.write_text("0.1 0.2\n0.3 0.4\n")  # 2 cols, not 3 -> unusable
        assert _modecar_natoms(str(p)) is None

    def test_modecar_natoms_empty(self, tmp_path):
        p = tmp_path / "MODECAR"
        p.write_text("")
        assert _modecar_natoms(str(p)) is None

    def test_oszicar_ionic_steps(self, tmp_path):
        p = tmp_path / "OSZICAR"
        _write_oszicar(p, 12)
        assert _count_oszicar_ionic_steps(str(tmp_path)) == 12

    def test_oszicar_missing(self, tmp_path):
        assert _count_oszicar_ionic_steps(str(tmp_path)) is None


# ---------------------------------------------------------------------------
# bank_singlepoint_vasp_restarts
# ---------------------------------------------------------------------------

class TestBanking:
    def _config(self):
        return make_config_dict(method="SinglePoint", Calculator="Vasp")

    def test_banks_centcar_and_newmodecar(self, tmp_path, monkeypatch):
        """Full mid-run dir: POSCAR<-CENTCAR, MODECAR<-NEWMODECAR, steps counted."""
        monkeypatch.chdir(tmp_path)
        d = _mk_workdir(tmp_path, 5, None)
        _write_poscar(d / "POSCAR", 33, tag="input")     # original ASE POSCAR
        _write_poscar(d / "CONTCAR", 33, tag="contcar")  # has velocity block IRL
        _write_poscar(d / "CENTCAR", 33, tag="centcar")  # dimer center (preferred)
        _write_modecar(d / "MODECAR", 33, tag=1.0)       # input mode
        _write_modecar(d / "NEWMODECAR", 33, tag=2.0)    # updated mode (preferred)
        _write_oszicar(d / "OSZICAR", 42)

        res = bank_singlepoint_vasp_restarts([5], self._config())

        assert set(res) == {5}
        info = res[5]
        assert info["geom_src"] == "CENTCAR"
        assert info["mode_src"] == "NEWMODECAR"
        assert info["natoms"] == 33
        assert info["banked_steps"] == 42
        bank = info["resume_dir"]
        assert "seed-centcar" in open(os.path.join(bank, "POSCAR")).read()
        assert open(os.path.join(bank, "MODECAR")).read().startswith("2.000000")

    def test_falls_back_to_contcar_and_modecar(self, tmp_path, monkeypatch):
        """No CENTCAR/NEWMODECAR -> CONTCAR + MODECAR are used."""
        monkeypatch.chdir(tmp_path)
        d = _mk_workdir(tmp_path, 8, None)
        _write_poscar(d / "CONTCAR", 20, tag="contcar")
        _write_modecar(d / "MODECAR", 20, tag=1.0)
        res = bank_singlepoint_vasp_restarts([8], self._config())
        assert res[8]["geom_src"] == "CONTCAR"
        assert res[8]["mode_src"] == "MODECAR"
        assert res[8]["natoms"] == 20

    def test_first_scf_death_not_banked(self, tmp_path, monkeypatch):
        """0-byte CONTCAR, no CENTCAR/NEWMODECAR -> skipped (falls back to scratch)."""
        monkeypatch.chdir(tmp_path)
        d = _mk_workdir(tmp_path, 939, None)
        _write_poscar(d / "POSCAR", 104, tag="input")
        (d / "CONTCAR").write_text("")           # 0-byte
        _write_modecar(d / "MODECAR", 104)        # mode exists but no geometry
        res = bank_singlepoint_vasp_restarts([939], self._config())
        assert res == {}
        assert not (tmp_path / "SinglePoint_resume_states" / "939").exists()

    def test_no_workdir_not_banked(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        assert bank_singlepoint_vasp_restarts([0, 1, 2], self._config()) == {}

    def test_atom_count_mismatch_not_banked(self, tmp_path, monkeypatch):
        """Geometry natoms != mode natoms -> skipped."""
        monkeypatch.chdir(tmp_path)
        d = _mk_workdir(tmp_path, 3, None)
        _write_poscar(d / "CENTCAR", 10, tag="centcar")
        _write_modecar(d / "NEWMODECAR", 9)  # mismatch
        assert bank_singlepoint_vasp_restarts([3], self._config()) == {}

    def test_rebank_refreshes_stale_bank(self, tmp_path, monkeypatch):
        """A pre-existing bank dir is replaced by the latest wall-kill state."""
        monkeypatch.chdir(tmp_path)
        stale = tmp_path / "SinglePoint_resume_states" / "5"
        stale.mkdir(parents=True)
        (stale / "POSCAR").write_text("STALE")
        (stale / "leftover").write_text("junk")
        d = _mk_workdir(tmp_path, 5, None)
        _write_poscar(d / "CENTCAR", 12, tag="fresh")
        _write_modecar(d / "NEWMODECAR", 12)
        res = bank_singlepoint_vasp_restarts([5], self._config())
        bank = res[5]["resume_dir"]
        assert "seed-fresh" in open(os.path.join(bank, "POSCAR")).read()
        assert not os.path.exists(os.path.join(bank, "leftover"))  # rmtree'd


# ---------------------------------------------------------------------------
# make_sp_resume_seed_writer  (post-ASE-write overwrite)
# ---------------------------------------------------------------------------

class TestSeedWriter:
    def test_seed_overwrites_poscar_and_modecar(self, tmp_path):
        bank = tmp_path / "bank"
        bank.mkdir()
        _write_poscar(bank / "POSCAR", 6, tag="banked")
        _write_modecar(bank / "MODECAR", 6, tag=9.0)

        rundir = tmp_path / "VASP_0"
        rundir.mkdir()
        _write_poscar(rundir / "POSCAR", 6, tag="fresh_ase")   # what ASE wrote
        _write_modecar(rundir / "MODECAR", 6, tag=1.0)          # what modecar wrote

        writer = make_sp_resume_seed_writer(str(bank))
        writer(calc=None, atoms=None, directory=str(rundir))

        assert "seed-banked" in open(rundir / "POSCAR").read()
        assert open(rundir / "MODECAR").read().startswith("9.000000")

    def test_seed_skips_absent_bank_files(self, tmp_path):
        """Missing/empty bank files leave the fresh inputs untouched (no crash)."""
        bank = tmp_path / "bank"
        bank.mkdir()
        (bank / "MODECAR").write_text("")  # empty -> skip
        rundir = tmp_path / "VASP_0"
        rundir.mkdir()
        _write_poscar(rundir / "POSCAR", 4, tag="fresh_ase")
        _write_modecar(rundir / "MODECAR", 4, tag=1.0)
        make_sp_resume_seed_writer(str(bank))(None, None, str(rundir))
        assert "seed-fresh_ase" in open(rundir / "POSCAR").read()   # untouched
        assert open(rundir / "MODECAR").read().startswith("1.000000")  # untouched


# ---------------------------------------------------------------------------
# resolve_vasp_calc_class extra_writers plumbing
# ---------------------------------------------------------------------------

class _DummyCalc:
    """Stand-in ASE-Vasp-like calc: write_input drops sentinel POSCAR/MODECAR."""
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.sort = None

    def write_input(self, atoms, *args, **kwargs):
        directory = kwargs.get("directory", ".")
        _write_poscar(os.path.join(directory, "POSCAR"), len(atoms), tag="fresh_ase")


class TestResolveWithExtraWriters:
    def test_no_specs_no_writers_returns_base(self):
        """Non-SP path: no extra_input_files/outputs and no extra_writers -> unwrapped."""
        cfg = make_config_dict(method="Dimer", Calculator="Vasp")
        assert resolve_vasp_calc_class(cfg, _DummyCalc) is _DummyCalc

    def test_extra_writers_forces_wrap_and_runs_last(self, tmp_path):
        cfg = make_config_dict(method="SinglePoint", Calculator="Vasp")
        bank = tmp_path / "bank"
        bank.mkdir()
        _write_poscar(bank / "POSCAR", 3, tag="banked")
        _write_modecar(bank / "MODECAR", 3, tag=7.0)
        seed = make_sp_resume_seed_writer(str(bank))

        wrapped = resolve_vasp_calc_class(cfg, _DummyCalc, extra_writers=[seed])
        assert wrapped is not _DummyCalc  # got wrapped

        rundir = tmp_path / "VASP_0"
        rundir.mkdir()
        atoms = bulk("Cu", "fcc", a=3.6) * (1, 1, 3)  # 3 atoms
        wrapped().write_input(atoms, directory=str(rundir))
        # DummyCalc wrote a fresh POSCAR; the seed writer (running last) replaced it.
        assert "seed-banked" in open(rundir / "POSCAR").read()
        assert open(rundir / "MODECAR").read().startswith("7.000000")


# ---------------------------------------------------------------------------
# geomopt._sp_resume_seed_writers gating
# ---------------------------------------------------------------------------

class TestSpResumeGating:
    def _bank(self, tmp_path, natoms):
        bank = tmp_path / "bank"
        bank.mkdir()
        _write_poscar(bank / "POSCAR", natoms, tag="banked")
        _write_modecar(bank / "MODECAR", natoms)
        return {"resume_dir": str(bank), "banked_steps": 11,
                "geom_src": "CENTCAR", "mode_src": "NEWMODECAR", "natoms": natoms}

    def test_none_continuation_returns_none(self):
        cfg = make_config_dict(method="SinglePoint", Calculator="Vasp")
        atoms = bulk("Cu", "fcc", a=3.6)
        assert _sp_resume_seed_writers(None, atoms, cfg, 0, 0) == (None, None)

    def test_continue_flag_off_returns_none(self, tmp_path):
        cfg = make_config_dict(method="SinglePoint", Calculator="Vasp",
                               continue_from_result=False)
        atoms = bulk("Cu", "fcc", a=3.6)
        data = self._bank(tmp_path, 1)
        assert _sp_resume_seed_writers(data, atoms, cfg, 0, 0) == (None, None)

    def test_natoms_mismatch_returns_none(self, tmp_path):
        cfg = make_config_dict(method="SinglePoint", Calculator="Vasp")
        atoms = bulk("Cu", "fcc", a=3.6)  # 1 atom
        data = self._bank(tmp_path, 5)    # banked 5 atoms -> mismatch
        assert _sp_resume_seed_writers(data, atoms, cfg, 0, 0) == (None, None)

    def test_missing_bank_file_returns_none(self, tmp_path):
        cfg = make_config_dict(method="SinglePoint", Calculator="Vasp")
        atoms = bulk("Cu", "fcc", a=3.6)
        data = self._bank(tmp_path, 1)
        os.remove(os.path.join(data["resume_dir"], "MODECAR"))
        assert _sp_resume_seed_writers(data, atoms, cfg, 0, 0) == (None, None)

    def test_valid_returns_writer_and_steps(self, tmp_path):
        cfg = make_config_dict(method="SinglePoint", Calculator="Vasp")
        atoms = bulk("Cu", "fcc", a=3.6)  # 1 atom
        data = self._bank(tmp_path, 1)
        writers, steps = _sp_resume_seed_writers(data, atoms, cfg, 0, 0)
        assert steps == 11
        assert writers is not None and len(writers) == 1
        assert callable(writers[0])
