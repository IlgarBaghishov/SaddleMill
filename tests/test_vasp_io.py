"""Tests for the pluggable VASP input-generation layer.

The loader / precedence tests use a file-based custom generator and run CPU-only
(no fairchem-data-omat / pymatgen needed). The OMat24 / OC20 translation tests
skip automatically when those optional packages are absent.
"""
import os

import numpy as np
import pytest
from ase import Atoms

from saddlemill.vasp_io import (
    load_input_generator, load_extra_input_writer, load_extra_output_parser,
    write_modecar, read_vtst_dimer, _pmg_set_to_ase_kwargs, _DRIVER_KEYS,
)
from saddlemill.tools import (vasp_incar_kwargs, resolve_vasp_calc_class,
                              _with_extra_io)
from ase.calculators.singlepoint import SinglePointCalculator
from tests.conftest import make_config_dict


# A tiny custom generator written to a temp file for the loader/precedence tests.
_CUSTOM_GEN_SRC = """
def gen(atoms):
    # Returns a couple of INCAR-ish keys plus a driver key that callers strip.
    return {"encut": 520, "ismear": 0, "ibrion": 2}
"""


@pytest.fixture
def custom_gen_file(tmp_path):
    p = tmp_path / "my_gen.py"
    p.write_text(_CUSTOM_GEN_SRC)
    return p


class TestLoadInputGenerator:
    def test_builtin_names_resolve_to_callables(self):
        for name in ("omat24_static", "omat24_relax", "cheap_omat", "oc20",
                     "cheap_oc20", "oc22", "cheap_oc22"):
            gen = load_input_generator(name)
            assert callable(gen)

    def test_callable_passthrough(self):
        f = lambda atoms: {"encut": 1}
        assert load_input_generator(f) is f

    def test_module_func_path(self):
        # Resolve (not call) a real module:func without heavy imports.
        gen = load_input_generator("saddlemill.vasp_io:oc20")
        from saddlemill.vasp_io import oc20
        assert gen is oc20

    def test_file_path_func(self, custom_gen_file):
        gen = load_input_generator(f"{custom_gen_file}:gen")
        assert gen(None) == {"encut": 520, "ismear": 0, "ibrion": 2}

    def test_unknown_name_raises(self):
        with pytest.raises(ValueError, match="Unknown input_generator"):
            load_input_generator("nonsense")

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_input_generator(f"{tmp_path/'nope.py'}:gen")

    def test_missing_func_in_module_raises(self):
        with pytest.raises(AttributeError):
            load_input_generator("saddlemill.vasp_io:does_not_exist")


class TestPrecedence:
    """vasp_incar_kwargs: [Vasp] overrides generator; generator fills the rest.

    input_generator lives in [ourVasp]; [Vasp] is a pure ASE-Vasp pass-through,
    so neither orchestration key ever reaches the calculator.
    """

    def test_vasp_section_overrides_generator(self, custom_gen_file):
        cfg = {"Vasp": {"encut": 999, "xc": "PBE"},
               "ourVasp": {"input_generator": f"{custom_gen_file}:gen"}}
        kw = vasp_incar_kwargs(cfg, atoms=object())  # custom gen ignores atoms
        assert kw["encut"] == 999          # [Vasp] wins
        assert kw["ismear"] == 0           # from generator (absent in [Vasp])
        assert kw["xc"] == "PBE"           # [Vasp]-only key
        assert "input_generator" not in kw and "extra_input_files" not in kw

    def test_no_atoms_skips_generator(self, custom_gen_file):
        cfg = {"Vasp": {"encut": 350},
               "ourVasp": {"input_generator": f"{custom_gen_file}:gen"}}
        kw = vasp_incar_kwargs(cfg, atoms=None)
        assert kw == {"encut": 350}        # generator skipped without atoms

    def test_no_generator_is_plain_vasp_section(self):
        cfg = {"Vasp": {"encut": 350, "xc": "PBE"}}
        assert vasp_incar_kwargs(cfg, atoms=object()) == {"encut": 350, "xc": "PBE"}

    def test_missing_vasp_key_is_ase_default(self, custom_gen_file):
        # A tag in neither [Vasp] nor the generator is simply absent (-> ASE default).
        cfg = {"Vasp": {}, "ourVasp": {"input_generator": f"{custom_gen_file}:gen"}}
        kw = vasp_incar_kwargs(cfg, atoms=object())
        assert "nelmin" not in kw


@pytest.mark.parametrize("set_name", ["omat24_static", "omat24_relax"])
def test_omat24_translation(set_name):
    pytest.importorskip("pymatgen")
    pytest.importorskip("fairchem.data.omat")
    from ase.build import bulk
    atoms = bulk("Fe", "bcc", a=2.87, cubic=True)
    kw = load_input_generator(set_name)(atoms)

    # Lowercased INCAR keys, no ionic-driver tags.
    assert _DRIVER_KEYS.isdisjoint(kw)
    # MAGMOM aligned to atom order (ASE re-sorts internally).
    assert "magmom" in kw and len(kw["magmom"]) == len(atoms)
    # Explicit k-mesh + per-element POTCAR setups.
    assert "kpts" in kw and len(kw["kpts"]) == 3
    assert kw.get("setups") == {"Fe": "_pv"}
    assert kw["ispin"] == 2
    # Pure-metal Fe (no O/F) -> no DFT+U emitted at all.
    assert "ldau_luj" not in kw and "ldauu" not in kw


def test_omat24_ldau_element_keyed():
    """DFT+U must be emitted as element-keyed ``ldau_luj`` (not the positional
    ldauu/ldaul/ldauj lists), so ASE's alphabetical atom re-sort cannot land U
    on the wrong element."""
    pytest.importorskip("pymatgen")
    pytest.importorskip("fairchem.data.omat")
    from ase.build import bulk
    atoms = bulk("FeO", "rocksalt", a=4.3)                # Fe + O -> Hubbard U on Fe
    kw = load_input_generator("omat24_static")(atoms)

    assert "ldau_luj" in kw                               # element-keyed dict
    assert not any(k in kw for k in ("ldauu", "ldaul", "ldauj"))  # raw lists dropped
    assert kw["ldau_luj"]["Fe"] == {"L": 2, "U": 5.3, "J": 0.0}   # U on Fe, d-shell
    assert kw["ldau_luj"]["O"]["U"] == 0.0               # not on O
    assert len(kw["magmom"]) == len(atoms)               # magmom stays per-atom


def test_cheap_omat():
    """cheap_omat = OMat24 static with reciprocal_density 16 + minimal-base light
    POTCARs (plain metals, soft O/C/N) for a cheap first-pass saddle search."""
    pytest.importorskip("pymatgen")
    pytest.importorskip("fairchem.data.omat")
    import numpy as np
    from ase.build import bulk
    atoms = bulk("Fe", "bcc", a=2.87, cubic=True)
    cheap = load_input_generator("cheap_omat")(atoms)
    full = load_input_generator("omat24_static")(atoms)

    assert np.prod(cheap["kpts"]) < np.prod(full["kpts"])          # lighter mesh (rd 16 < 64)
    assert cheap["setups"] == {"base": "minimal", "O": "_s", "C": "_s", "N": "_s"}
    assert cheap["encut"] == full["encut"] and cheap["ismear"] == -5  # electronics left to [Vasp]


def test_cheap_omat_lanthanide_fcore():
    """cheap_omat must keep the f-in-core ``_3`` lanthanide POTCARs (same as omat24_static);
    the minimal base would otherwise pick f-in-valence potentials that don't converge."""
    pytest.importorskip("pymatgen")
    pytest.importorskip("fairchem.data.omat")
    from ase import Atoms
    # Nd (uses _3) + Gd (stays standard) + Fe (lightened to minimal)
    atoms = Atoms("NdGdFe", positions=[(0, 0, 0), (3, 0, 0), (6, 0, 0)], cell=[12, 8, 8], pbc=True)
    cheap = load_input_generator("cheap_omat")(atoms)
    full = load_input_generator("omat24_static")(atoms)

    assert cheap["setups"]["Nd"] == "_3"          # f-in-core kept, not reverted to valence
    assert "Gd" not in cheap["setups"]            # Gd stays standard (matches OMat24StaticSet)
    assert cheap["setups"]["base"] == "minimal"   # non-f elements still lightened
    # lanthanide choice agrees with omat24_static
    assert cheap["setups"].get("Nd") == full.get("setups", {}).get("Nd")


def test_oc20_translation():
    pytest.importorskip("fairchem.data.oc")
    from ase.build import fcc111, add_adsorbate
    slab = fcc111("Cu", size=(2, 2, 3), vacuum=8.0)
    add_adsorbate(slab, "C", height=1.8, position="ontop")
    kw = load_input_generator("oc20")(slab)

    assert _DRIVER_KEYS.isdisjoint(kw)
    assert "kpts" in kw and len(kw["kpts"]) == 3
    assert kw["kpts"][2] == 1            # surface: single k-point along vacuum axis
    assert kw["gga"] == "RP"             # RPBE
    assert kw.get("setups") == "minimal"


def test_cheap_oc20():
    """cheap_oc20 = OC20 with the k-point multiplier halved (40 -> 20) + soft
    _s O/C/N on the minimal base; everything else (RPBE, ENCUT 350) untouched."""
    pytest.importorskip("fairchem.data.oc")
    from ase.build import fcc111
    slab = fcc111("Cu", size=(2, 2, 3), vacuum=8.0)
    cheap = load_input_generator("cheap_oc20")(slab)
    full = load_input_generator("oc20")(slab)

    assert np.prod(cheap["kpts"]) < np.prod(full["kpts"])   # lighter in-plane mesh
    assert cheap["kpts"][2] == 1
    assert cheap["setups"] == {"base": "minimal", "O": "_s", "C": "_s", "N": "_s"}
    assert cheap["gga"] == "RP" and cheap["encut"] == full["encut"]  # physics untouched
    assert _DRIVER_KEYS.isdisjoint(cheap)


# --- OC22 (exact Meta settings, hardcoded -- no pymatgen/fairchem needed) ----

def _oc22_adslab(with_adsorbate=True, magmoms=None):
    """Small Fe-oxide slab; tag=2 marks the OH adsorbate (the OC convention)."""
    slab = Atoms("Fe4O4",
                 positions=[(x * 2.0, y * 2.0, 8.0) for x in range(2) for y in range(2)]
                 + [(x * 2.0 + 1.0, y * 2.0 + 1.0, 9.5) for x in range(2) for y in range(2)],
                 cell=[8.7, 8.7, 30.0], pbc=True)
    slab.set_tags([1] * 4 + [0] * 4)
    if with_adsorbate:
        ads = Atoms("OH", positions=[(2.0, 2.0, 11.0), (2.0, 2.0, 12.0)])
        tags = list(slab.get_tags()) + [2, 2]
        slab = slab + ads
        slab.set_tags(tags)
    if magmoms is not None:
        slab.set_initial_magnetic_moments(magmoms)
    return slab


def test_oc22_exact_settings():
    slab = _oc22_adslab()
    kw = load_input_generator("oc22")(slab)

    assert _DRIVER_KEYS.isdisjoint(kw)
    # Level of theory: PBE, spin-polarized, ENCUT 500, EDIFF 1e-4.
    assert kw["gga"] == "PE" and kw["ispin"] == 2
    assert kw["encut"] == 500.0 and kw["ediff"] == 1e-4
    assert kw["ismear"] == 0 and kw["sigma"] == 0.05
    assert kw["algo"] == "Fast" and kw["prec"] == "Accurate"
    assert kw["lreal"] is False and kw["lasph"] is True
    assert kw["nelm"] == 60 and kw["nelmin"] == 8
    # Gamma-centered ceil(30/a) x ceil(30/b) x 1: ceil(30/8.7) = 4.
    assert kw["kpts"] == (4, 4, 1) and kw["gamma"] is True
    # MP Hubbard U on Fe (element-keyed, so ASE's re-sort can't misassign it).
    assert kw["ldau"] is True and kw["ldautype"] == 2
    assert kw["ldau_luj"] == {"Fe": {"L": 2, "U": 5.3, "J": 0.0}}
    assert kw["lmaxmix"] == 4                       # d-block present
    # MP default moments: ferromagnetic Fe (5), 0.6 elsewhere, atoms order.
    assert kw["magmom"] == [5] * 4 + [0.6] * 6
    # 2022 MPRelaxSet POTCAR mapping (suffixed elements only; O stays bare).
    assert kw["setups"] == {"Fe": "_pv"}
    # No pymatgen-era-drift keys.
    assert "enaug" not in kw


def test_oc22_dipole_only_with_tag2_adsorbate():
    with_ads = load_input_generator("oc22")(_oc22_adslab(True))
    clean = load_input_generator("oc22")(_oc22_adslab(False))

    assert with_ads["ldipol"] is True and with_ads["idipol"] == 3
    assert len(with_ads["dipol"]) == 3              # mass-weighted COM, fractional
    assert all(k not in clean for k in ("ldipol", "idipol", "dipol"))


def test_oc22_no_u_without_u_element_or_anion():
    # Ti has no MP U value -> oxide still gets no LDAU tags at all.
    tio = Atoms("TiO2", positions=[(0, 0, 0), (1.5, 1.5, 1.5), (3, 3, 3)],
                cell=[6, 6, 30], pbc=True)
    kw = load_input_generator("oc22")(tio)
    assert not any(k.startswith("ldau") for k in kw)
    assert kw["lmaxmix"] == 4                       # LMAXMIX rule is U-independent

    # Fe without O/F (most electronegative element is not O/F) -> no U either.
    fe = Atoms("Fe2", positions=[(0, 0, 0), (2, 0, 0)], cell=[8, 8, 30], pbc=True)
    assert not any(k.startswith("ldau") for k in load_input_generator("oc22")(fe))


def test_oc22_atoms_magmoms_win():
    # Structures carrying initial moments keep them (ASE writes MAGMOM itself).
    slab = _oc22_adslab(magmoms=[2.0] * 10)
    assert "magmom" not in load_input_generator("oc22")(slab)


def test_cheap_oc22():
    """cheap_oc22 = OC22 with k-product halved (30 -> 15) + minimal-base light
    POTCARs (soft O/C/N); spin, U and all electronic knobs untouched."""
    slab = _oc22_adslab()
    cheap = load_input_generator("cheap_oc22")(slab)
    full = load_input_generator("oc22")(slab)

    assert np.prod(cheap["kpts"]) < np.prod(full["kpts"])
    assert cheap["setups"] == {"base": "minimal", "O": "_s", "C": "_s", "N": "_s"}
    assert cheap["ispin"] == 2 and cheap["ldau_luj"] == full["ldau_luj"]  # same PES
    assert cheap["encut"] == full["encut"]          # electronics left to [Vasp]


def test_cheap_oc22_lanthanide_fcore():
    """cheap_oc22 keeps OC22's f-in-core lanthanide POTCARs (_3, Yb _2); the
    minimal base would otherwise pick f-in-valence ones that don't converge."""
    atoms = Atoms("YbNdO", positions=[(0, 0, 0), (3, 0, 0), (6, 0, 0)],
                  cell=[12, 8, 30], pbc=True)
    cheap = load_input_generator("cheap_oc22")(atoms)
    full = load_input_generator("oc22")(atoms)

    assert cheap["setups"]["Nd"] == "_3" == full["setups"]["Nd"]
    assert cheap["setups"]["Yb"] == "_2" == full["setups"]["Yb"]  # OC22-era choice
    assert cheap["setups"]["base"] == "minimal"


# --- extra_input_files / MODECAR -------------------------------------------

# A tiny custom writer written to a temp file for the loader test.
_CUSTOM_WRITER_SRC = """
import os
def write_iconst(calc, atoms, directory):
    with open(os.path.join(directory, "ICONST"), "w") as f:
        f.write("LR 1 0\\n")
"""


class _FakeCalc:
    """Stand-in for a VASP calc: holds the sort/resort maps ASE would build."""
    def __init__(self, directory=".", sort=None, resort=None):
        self.directory = directory
        self.sort = sort
        self.resort = resort

    def write_input(self, atoms, *args, **kwargs):
        self.wrote_input = True

    def read_results(self):
        self.read_called = True


class TestLoadExtraInputWriter:
    def test_builtin_modecar_resolves(self):
        assert load_extra_input_writer("modecar") is write_modecar

    def test_callable_passthrough(self):
        f = lambda calc, atoms, directory: None
        assert load_extra_input_writer(f) is f

    def test_file_path_writer(self, tmp_path):
        p = tmp_path / "w.py"
        p.write_text(_CUSTOM_WRITER_SRC)
        w = load_extra_input_writer(f"{p}:write_iconst")
        w(_FakeCalc(directory=str(tmp_path)), Atoms("H"), str(tmp_path))
        assert (tmp_path / "ICONST").read_text().startswith("LR")

    def test_unknown_name_raises(self):
        with pytest.raises(ValueError, match="Unknown extra_input_files writer"):
            load_extra_input_writer("nonsense")


class TestWriteModecar:
    def _atoms(self, eigenmode, info_key="eigenmode"):
        a = Atoms("H3", positions=[[0, 0, 0], [1, 0, 0], [2, 0, 0]])
        if info_key == "orig":
            a.info["orig_info"] = {"eigenmode": eigenmode}
        else:
            a.info["eigenmode"] = eigenmode
        return a

    def test_ordering_uses_calc_sort(self, tmp_path):
        eig = [[1., 0., 0.], [0., 2., 0.], [0., 0., 3.]]   # atoms order
        atoms = self._atoms(eig)
        calc = _FakeCalc(directory=str(tmp_path), sort=[2, 0, 1])  # POSCAR = atoms[sort]
        write_modecar(calc, atoms, str(tmp_path))

        rows = np.loadtxt(tmp_path / "MODECAR")
        expected = np.array(eig)[[2, 0, 1]]
        expected = expected / np.linalg.norm(expected)         # normalized full vector
        assert np.allclose(rows, expected)

    def test_orig_info_fallback(self, tmp_path):
        atoms = self._atoms([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]], info_key="orig")
        write_modecar(_FakeCalc(directory=str(tmp_path)), atoms, str(tmp_path))
        assert (tmp_path / "MODECAR").exists()

    def test_missing_eigenmode_warns_and_skips(self, tmp_path):
        atoms = Atoms("H3", positions=[[0, 0, 0], [1, 0, 0], [2, 0, 0]])
        with pytest.warns(UserWarning, match="no 'eigenmode'"):
            write_modecar(_FakeCalc(directory=str(tmp_path)), atoms, str(tmp_path))
        assert not (tmp_path / "MODECAR").exists()


class TestExtraInputFilesWiring:
    def test_with_extra_input_files_runs_writers(self, tmp_path):
        wrapped = _with_extra_io(_FakeCalc, [write_modecar], [])
        inst = wrapped(directory=str(tmp_path))
        atoms = Atoms("H2", positions=[[0, 0, 0], [1, 0, 0]])
        atoms.info["eigenmode"] = [[1., 0., 0.], [0., 1., 0.]]
        inst.write_input(atoms)
        assert inst.wrote_input is True             # super().write_input ran
        assert (tmp_path / "MODECAR").exists()      # writer ran after it

    def test_resolve_class_wraps_only_when_configured(self):
        base = {"Main": {"Calculator": "Vasp"}, "ourVasp": {}}
        # No extra_input_files -> class returned unchanged.
        assert resolve_vasp_calc_class(base, _FakeCalc) is _FakeCalc
        # Configured -> wrapped subclass.
        cfg = {"Main": {"Calculator": "Vasp"}, "ourVasp": {"extra_input_files": "modecar"}}
        wrapped = resolve_vasp_calc_class(cfg, _FakeCalc)
        assert wrapped is not _FakeCalc and issubclass(wrapped, _FakeCalc)
        # FAIRChem -> never wrapped even if set.
        fc = {"Main": {"Calculator": "FAIRChemCalculator"},
              "ourVasp": {"extra_input_files": "modecar"}}
        assert resolve_vasp_calc_class(fc, _FakeCalc) is _FakeCalc

    def test_list_of_writers(self, tmp_path):
        p = tmp_path / "w.py"
        p.write_text(_CUSTOM_WRITER_SRC)
        cfg = {"Main": {"Calculator": "Vasp"},
               "ourVasp": {"extra_input_files": ["modecar", f"{p}:write_iconst"]}}
        wrapped = resolve_vasp_calc_class(cfg, _FakeCalc)
        inst = wrapped(directory=str(tmp_path))
        atoms = Atoms("H2", positions=[[0, 0, 0], [1, 0, 0]])
        atoms.info["eigenmode"] = [[1., 0., 0.], [0., 1., 0.]]
        inst.write_input(atoms)
        assert (tmp_path / "MODECAR").exists() and (tmp_path / "ICONST").exists()


class TestReadVtstDimer:
    def test_parses_newmodecar_and_dimcar(self, tmp_path):
        (tmp_path / "NEWMODECAR").write_text("0 0 1\n1 0 0\n0 1 0\n")  # POSCAR order
        (tmp_path / "DIMCAR").write_text(
            "Step Force Torque Energy Curvature Angle\n"
            "    1   0.5   0.20  -10.0   -0.30   5.0\n"
            "    2   0.1   0.05  -10.1   -0.45   1.0\n")
        atoms = Atoms("H3", positions=[[0, 0, 0], [1, 0, 0], [2, 0, 0]])
        calc = _FakeCalc(directory=str(tmp_path), resort=[2, 0, 1])  # POSCAR -> atoms
        info = read_vtst_dimer(calc, atoms, str(tmp_path))

        assert info["curvature"] == -0.45                       # last DIMCAR row, col 4
        mode_poscar = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]], float)
        assert np.allclose(info["eigenmode"], mode_poscar[[2, 0, 1]])  # resorted

    def test_converged_dimcar_dashes_reads_row_above(self, tmp_path):
        # On convergence VTST's Dimer_Fin writes a final row with '---' in the
        # Torque/Curvature/Angle columns (Step/Force/Energy stay numeric). The
        # curvature must come from the last fully-numeric row above it, and the
        # parse must NOT crash (the old code did float('---') -> ValueError).
        (tmp_path / "DIMCAR").write_text(
            "Step Force Torque Energy Curvature Angle\n"
            "   12   0.11550   0.76786  -160.69013   -3.17780   2.88614\n"
            "   13   0.07027   ---      -160.69053   ---        ---\n")
        atoms = Atoms("H")
        info = read_vtst_dimer(_FakeCalc(directory=str(tmp_path)), atoms, str(tmp_path))
        assert info["curvature"] == -3.17780                    # row above the '---'

    def test_overflow_curvature_row_skipped(self, tmp_path):
        # A Fortran '*****' overflow in the Curvature column (numeric Force) must be
        # skipped rather than crash, so the last clean curvature wins.
        (tmp_path / "DIMCAR").write_text(
            "Step Force Torque Energy Curvature Angle\n"
            "    1   0.50   0.20   -10.0   -0.30      5.0\n"
            "    2   106.0  61896  -9347   *********  60.6\n")
        atoms = Atoms("H")
        info = read_vtst_dimer(_FakeCalc(directory=str(tmp_path)), atoms, str(tmp_path))
        assert info["curvature"] == -0.30

    def test_missing_files_returns_empty(self, tmp_path):
        atoms = Atoms("H")
        assert read_vtst_dimer(_FakeCalc(directory=str(tmp_path)), atoms, str(tmp_path)) == {}


class TestExtraOutputsWiring:
    def test_builtin_resolves(self):
        assert load_extra_output_parser("vtst_dimer") is read_vtst_dimer

    def test_unknown_name_raises(self):
        with pytest.raises(ValueError, match="Unknown extra_outputs parser"):
            load_extra_output_parser("nonsense")

    def test_read_results_populates_sm_extra_outputs(self, tmp_path):
        (tmp_path / "NEWMODECAR").write_text("0 0 1\n1 0 0\n")
        cfg = {"Main": {"Calculator": "Vasp"}, "ourVasp": {"extra_outputs": "vtst_dimer"}}
        wrapped = resolve_vasp_calc_class(cfg, _FakeCalc)
        assert wrapped is not _FakeCalc
        inst = wrapped(directory=str(tmp_path))
        inst.atoms = Atoms("H2", positions=[[0, 0, 0], [1, 0, 0]])
        inst.read_results()
        assert inst.read_called is True                # super().read_results() ran
        assert "eigenmode" in inst.sm_extra_outputs    # parser ran after it


class TestExtraOutputsReachSinglePointLMDB:
    """End-to-end: a VASP `extra_outputs` parser's results must reach the
    SinglePoint *lmdb* output, exactly as they do in traj output. Drives the real
    ``geomopt.singlepoint()`` lmdb branch with a stand-in VASP calc (no DFT)."""

    def _run_sp_lmdb(self, tmp_path, monkeypatch, sm_extra):
        pytest.importorskip("fairchem.core.datasets")  # registers the aselmdb backend
        from ase.db import connect
        import saddlemill.geomopt as geomopt

        monkeypatch.chdir(tmp_path)
        os.mkdir("SinglePoint_lmdbs")
        os.mkdir("SinglePoint_status_csvs")

        energy = -42.0
        forces = (np.arange(6, dtype=float).reshape(2, 3) + 1) * 0.1

        # Stand in for resolve_vasp_calc: a calc that returns fixed E/F and carries
        # whatever the extra_outputs parser would have stashed (no VASP needed).
        def fake_resolve(config_dict, calc, i, subunit, section, atoms=None):
            c = SinglePointCalculator(atoms, energy=energy, forces=forces)
            c.sm_extra_outputs = sm_extra
            return c

        monkeypatch.setattr(geomopt, "resolve_vasp_calc", fake_resolve)
        monkeypatch.setattr(geomopt, "finalize_if_vasp_interactive", lambda *a, **k: None)

        a = Atoms("H2", positions=[[0, 0, 0], [0, 0, 0.74]], cell=[10, 10, 10], pbc=True)
        # Source row carries a STALE eigenmode guess (zeros) + unrelated info.
        source_info = {"orig_info": {"foo": 1},
                       "eigenmode": [[0., 0., 0.], [0., 0., 0.]],
                       "src_tag": "INPUT"}
        extras = [{"kvp": {"sid": 5},
                   "row_data": {"info": dict(source_info), "traj_path": "/in.traj"}}]

        cfg = make_config_dict(method="SinglePoint", input_format="lmdb",
                               Calculator="Vasp", vasp_command="x", frames_per_job=1)

        geomopt.singlepoint(0, cfg, a, calc=None, consecutive_errors=[0],
                            executorlib_worker_id=0, extras=extras)

        with connect("SinglePoint_lmdbs/collected_sp_rank_0.aselmdb", type="aselmdb") as db:
            row = db.get(1)
        return row, energy, forces, source_info

    def test_vasp_extras_merged_into_lmdb_info(self, tmp_path, monkeypatch):
        sm_extra = {"eigenmode": np.array([[0., 0., 1.], [1., 0., 0.]]), "curvature": -0.77}
        row, energy, forces, source_info = self._run_sp_lmdb(tmp_path, monkeypatch, sm_extra)

        assert row.energy == energy                       # E/F populated via SPC
        assert np.allclose(row.forces, forces)
        info = row.data["info"]
        # the dimer's fresh eigenmode overwrites the stale input guess
        assert np.allclose(np.array(info["eigenmode"]), sm_extra["eigenmode"])
        assert info["curvature"] == -0.77
        # source info preserved verbatim alongside the new keys
        assert info["src_tag"] == "INPUT"
        assert info["orig_info"] == {"foo": 1}
        assert row.data["traj_path"] == "/in.traj"

    def test_no_extras_keeps_source_info_byte_equivalent(self, tmp_path, monkeypatch):
        # FAIRChem-equivalent path: empty sm_extra -> source data passes through
        # untouched (the build_lmdb_parallel byte-equivalence contract).
        row, energy, forces, source_info = self._run_sp_lmdb(tmp_path, monkeypatch, {})

        assert row.energy == energy
        info = row.data["info"]
        assert set(info.keys()) == set(source_info.keys())   # nothing added
        assert "curvature" not in info
        assert info["src_tag"] == "INPUT"
        assert np.allclose(np.array(info["eigenmode"]), 0.0)  # stale guess untouched


class TestSinglePointVaspDebugZip:
    """SP+VASP must archive its scratch dir into SinglePoint_debug_zips/ (heavies
    dropped on success), like the other VASP methods — not silently delete it.
    Drives the real geomopt.singlepoint() VASP path with a stand-in calc (no DFT)."""

    def _run(self, tmp_path, monkeypatch, zip_enabled):
        import saddlemill.geomopt as geomopt

        monkeypatch.chdir(tmp_path)
        for d in ("SinglePoint_trajes", "SinglePoint_status_csvs", "SinglePoint_debug_zips"):
            os.mkdir(d)

        def fake_resolve(config_dict, calc, i, subunit, section, atoms=None):
            d = f"VASP_{i}"
            os.makedirs(d, exist_ok=True)
            for name in ("OUTCAR", "DIMCAR"):                  # light artifacts to keep
                with open(os.path.join(d, name), "w") as f:
                    f.write(f"dummy {name}\n")
            with open(os.path.join(d, "WAVECAR"), "wb") as f:  # heavy -> must be stripped
                f.write(b"\x00" * 64)
            return SinglePointCalculator(atoms, energy=-1.0, forces=np.zeros((2, 3)))

        monkeypatch.setattr(geomopt, "resolve_vasp_calc", fake_resolve)
        monkeypatch.setattr(geomopt, "finalize_if_vasp_interactive", lambda *a, **k: None)

        a = Atoms("H2", positions=[[0, 0, 0], [0, 0, 0.74]], cell=[10, 10, 10], pbc=True)
        cfg = make_config_dict(method="SinglePoint", input_format="traj",
                               Calculator="Vasp", vasp_command="x", frames_per_job=1,
                               zip=zip_enabled)
        geomopt.singlepoint(0, cfg, a, calc=None, consecutive_errors=[0],
                            executorlib_worker_id=0)

    def test_create_results_dir_makes_debug_zips_only_for_vasp(self, tmp_path, monkeypatch):
        from saddlemill.config import create_results_directories
        v = tmp_path / "vasp"; v.mkdir(); monkeypatch.chdir(v)
        create_results_directories(make_config_dict(method="SinglePoint",
                                                    Calculator="Vasp", vasp_command="x"))
        assert (v / "SinglePoint_debug_zips").is_dir()

        f = tmp_path / "fc"; f.mkdir(); monkeypatch.chdir(f)
        create_results_directories(make_config_dict(method="SinglePoint",
                                                    Calculator="FAIRChemCalculator"))
        assert not (f / "SinglePoint_debug_zips").exists()    # FAIRChem SP: none

    def test_success_archives_dir_and_strips_heavies(self, tmp_path, monkeypatch):
        import zipfile
        self._run(tmp_path, monkeypatch, zip_enabled=True)
        zpath = tmp_path / "SinglePoint_debug_zips" / "structure_rank_0_data.zip"
        assert zpath.exists()
        with zipfile.ZipFile(zpath) as zf:
            names = zf.namelist()
        assert any(n.endswith("VASP_0/OUTCAR") for n in names)
        assert any(n.endswith("VASP_0/DIMCAR") for n in names)
        assert not any("WAVECAR" in n for n in names)         # heavies dropped
        assert not (tmp_path / "VASP_0").exists()             # scratch dir cleaned

    def test_zip_false_just_deletes(self, tmp_path, monkeypatch):
        self._run(tmp_path, monkeypatch, zip_enabled=False)
        assert not (tmp_path / "VASP_0").exists()             # dir gone
        assert not list((tmp_path / "SinglePoint_debug_zips").glob("*.zip"))  # no zip written


class TestSinglePointVaspConvergenceStatus:
    """singlepoint() must report the REAL VASP convergence verdict, not always
    'converged': calc.converged OR max per-atom |F| <= |EDIFFG|, with a final-step
    SCF miss -> 'error: scf_not_converged'. Drives the real traj path (no DFT)."""

    def _run(self, tmp_path, monkeypatch, *, converged, forces, ediffg=-0.05, scf_ok=True):
        from ase.io import read as aseread
        import saddlemill.geomopt as geomopt
        monkeypatch.chdir(tmp_path)
        os.mkdir("SinglePoint_trajes")
        os.mkdir("SinglePoint_status_csvs")

        def fake_resolve(config_dict, calc, i, subunit, section, atoms=None):
            c = SinglePointCalculator(atoms, energy=-1.0, forces=np.asarray(forces))
            c.converged = converged                 # what ASE read_convergence would set
            c.sm_extra_outputs = {}
            return c
        monkeypatch.setattr(geomopt, "resolve_vasp_calc", fake_resolve)
        monkeypatch.setattr(geomopt, "finalize_if_vasp_interactive", lambda *a, **k: None)
        monkeypatch.setattr(geomopt, "vasp_final_scf_converged", lambda d: scf_ok)

        a = Atoms("H2", positions=[[0, 0, 0], [0, 0, 0.74]], cell=[10, 10, 10], pbc=True)
        cfg = make_config_dict(method="SinglePoint", Calculator="Vasp",
                               vasp_command="x", frames_per_job=1, zip=False)
        cfg["Vasp"] = {"ediffg": ediffg}
        geomopt.singlepoint(0, cfg, a, calc=None, consecutive_errors=[0],
                            executorlib_worker_id=0)

        status = open("SinglePoint_status_csvs/status_rank_0.csv").read().strip()
        traj = "SinglePoint_trajes/collected_sp_rank_0.traj"
        frame = aseread(traj, index=-1) if (os.path.exists(traj) and os.path.getsize(traj)) else None
        return status, frame

    def test_calc_converged_true(self, tmp_path, monkeypatch):
        big = [[0., 0., 0.2], [0., 0., 0.]]                 # > |EDIFFG|, but VASP says converged
        status, frame = self._run(tmp_path, monkeypatch, converged=True, forces=big)
        assert status.endswith('"converged"')
        assert frame.info["status"] == "converged" and frame.info["converged"] == 1

    def test_nsw_exhausted_not_converged(self, tmp_path, monkeypatch):
        big = [[0., 0., 0.2], [0., 0., 0.]]                 # 0.2 > 0.05 and calc.converged False
        status, frame = self._run(tmp_path, monkeypatch, converged=False, forces=big)
        assert status.endswith('"not_converged"')
        assert frame.info["status"] == "not_converged" and frame.info["converged"] == 0

    def test_force_criterion_overrides_false(self, tmp_path, monkeypatch):
        small = [[0., 0., 0.01], [0., 0., 0.]]              # <= 0.05 -> converged via |F|
        status, frame = self._run(tmp_path, monkeypatch, converged=False, forces=small)
        assert status.endswith('"converged"')
        assert frame.info["converged"] == 1

    def test_scf_gate_errors_and_writes_no_frame(self, tmp_path, monkeypatch):
        small = [[0., 0., 0.01], [0., 0., 0.]]
        status, frame = self._run(tmp_path, monkeypatch, converged=True, forces=small, scf_ok=False)
        assert "error: scf_not_converged" in status
        assert frame is None                                # bad SCF -> nothing written
