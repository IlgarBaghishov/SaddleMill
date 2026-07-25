"""Regression guard on the ASE installation linked to SaddleMill (not on
SaddleMill itself).

VASP's VTST dimer requires the ``DROTMAX`` INCAR tag to be an *integer*; a
decimal form such as ``10.000000`` is silently rejected and VASP falls back to
its default (``RotMax 4``). ASE **<= 3.28** filed ``drotmax`` under
``float_keys``, so its ``write_incar`` formatted it with ``FLOAT_FORMAT``
(``'5.6f'``) and emitted ``DROTMAX = 10.000000`` even when handed an ``int``;
that release line needed a manual patch. ASE **3.29.0** fixed this upstream, and
the ``pyproject.toml`` pin (``ase>=3.29.0``) now enforces it.

This test is kept as a guard on the *linked* ASE rather than on SaddleMill: it
asserts the property directly, so it still fails loudly if an environment is
resolved or downgraded to a pre-3.29 ASE, where a config value of
``drotmax = 10`` (which SaddleMill's ConfigManager infers as a Python ``int``)
would silently be written as ``DROTMAX = 10.000000``.

This test only generates and parses INCAR text — it never runs VASP and needs
no POTCARs.
"""
from ase.build import bulk
from ase.calculators.vasp import Vasp


def test_ase_writes_integer_drotmax(tmp_path):
    """An integer ``drotmax`` (as from config ``drotmax = 10``) must be written
    to the INCAR as ``DROTMAX = 10`` — never ``10.000000`` or ``10.0``."""
    atoms = bulk("Cu", "fcc", a=3.6)

    calc = Vasp(directory=str(tmp_path), drotmax=10)  # int, as ConfigManager yields
    # write_incar normally relies on initialize() having set these two attributes;
    # set them directly so the INCAR can be written without POTCARs or running VASP.
    calc.spinpol = None
    calc.sort = list(range(len(atoms)))
    calc.write_incar(atoms, directory=str(tmp_path))

    incar = (tmp_path / "INCAR").read_text()
    drotmax_lines = [ln.strip() for ln in incar.splitlines()
                     if ln.strip().startswith("DROTMAX")]

    assert drotmax_lines == ["DROTMAX = 10"], (
        f"expected exactly 'DROTMAX = 10' (integer); got {drotmax_lines!r}. "
        "The linked ASE is likely still classifying 'drotmax' as a float key."
    )
