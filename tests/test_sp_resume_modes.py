"""Wall-kill states the SinglePoint resume gate must get right.

Scope is deliberately narrow. A survey of every surviving campaign VASP_*
workdir (1,195 of them, across frontera x3 accounts, vista, ls6 and banff)
found that resume only ever has to deal with an external SIGTERM/SIGKILL from
Slurm -- 713 walltime timeouts and 482 scancels. Runs that VASP itself killed
(segfault, MPI abort, ieee_invalid) are renamed ERROR_VASP_* and are invisible
to banking, so they cannot reach this code at all; runs that never completed an
ionic step have nothing worth resuming. Neither is handled here by design.

What IS defended: a kill landing mid-write. VASP writes CENTCAR/NEWMODECAR
between ionic steps, so the window is small -- 0 of the 1,190 resumable
workdirs were caught in it -- but a torn geometry file keeps a complete header
above a short body, and banking that would seed the resumed dimer with a
truncated structure. _poscar_natoms therefore reports the rows actually
present rather than the header's claim, so the geometry/mode agreement check
rejects it.
"""
import copy
import pytest

from saddlemill.config import ConfigManager
from saddlemill.tools import bank_singlepoint_vasp_restarts
from saddlemill.geomopt import _sp_resume_seed_writers

N = 3
GOOD_POSCAR = """K2Sr
1.0
  5.0000000000 0.0000000000 0.0000000000
  0.0000000000 5.0000000000 0.0000000000
  0.0000000000 0.0000000000 5.0000000000
  K  Sr
  2  1
Direct
  0.1000000000 0.2000000000 0.3000000000
  0.4000000000 0.5000000000 0.6000000000
  0.7000000000 0.8000000000 0.9000000000
"""
GOOD_MODE = "  0.10 0.20 0.30\n  0.40 0.50 0.60\n  0.70 0.80 0.90\n"

# kill landing mid-write: header still claims 3 atoms
TORN_BODY = (GOOD_POSCAR[:GOOD_POSCAR.index("Direct") + len("Direct\n")]
             + "  0.1000000000 0.2000000000 0.3000000000\n")
TORN_ROW = GOOD_POSCAR[:GOOD_POSCAR.rindex("  0.7000000000")] + "  0.70000"

MODES = [
    ("clean_wall_kill_mid_dimer", {"CENTCAR": GOOD_POSCAR, "NEWMODECAR": GOOD_MODE}, True),
    ("centcar_absent_contcar_used", {"CONTCAR": GOOD_POSCAR, "NEWMODECAR": GOOD_MODE}, True),
    ("newmodecar_absent_modecar_used", {"CENTCAR": GOOD_POSCAR, "MODECAR": GOOD_MODE}, True),
    # never completed an ionic step: CONTCAR exists but is still zero bytes
    ("no_ionic_step_zero_byte_contcar",
     {"CONTCAR": "", "MODECAR": GOOD_MODE}, False),
    # torn writes
    ("torn_centcar_missing_rows", {"CENTCAR": TORN_BODY, "NEWMODECAR": GOOD_MODE}, False),
    ("torn_centcar_half_row", {"CENTCAR": TORN_ROW, "NEWMODECAR": GOOD_MODE}, False),
    ("torn_newmodecar_missing_rows",
     {"CENTCAR": GOOD_POSCAR, "NEWMODECAR": "  0.1 0.2 0.3\n  0.4 0.5 0.6\n"}, False),
    ("torn_newmodecar_ragged_row",
     {"CENTCAR": GOOD_POSCAR, "NEWMODECAR": GOOD_MODE + "  0.1 0.2\n"}, False),
]


def _cfg(continue_flag=True):
    c = copy.deepcopy(ConfigManager.DEFAULTS)
    c["Main"]["method"] = "SinglePoint"
    c["Main"]["Calculator"] = "Vasp"
    c["Main"]["continue_from_result"] = continue_flag
    return c


class _Frame:
    def __init__(self, n=N):
        self._n = n

    def __len__(self):
        return self._n


def _workdir(tmp_path, monkeypatch, files):
    monkeypatch.chdir(tmp_path)
    d = tmp_path / "VASP_0"
    d.mkdir()
    for name, content in files.items():
        (d / name).write_text(content)
    (d / "OSZICAR").write_text(" 1 F= -.1E+01 E0= -.1E+01\n")
    return d


@pytest.mark.parametrize("name,files,expect_bank", MODES, ids=[m[0] for m in MODES])
def test_wall_kill_banking_policy(tmp_path, monkeypatch, name, files, expect_bank):
    _workdir(tmp_path, monkeypatch, files)
    banked = bank_singlepoint_vasp_restarts([0], _cfg())
    assert (0 in banked) is expect_bank
    if expect_bank:
        writers, _ = _sp_resume_seed_writers(banked[0], _Frame(), _cfg(), 0, 0)
        assert writers is not None, f"{name}: banked but the gate refused to seed"


def test_seeded_files_are_byte_identical_to_bank(tmp_path, monkeypatch):
    """POSCAR and MODECAR are the complete resume set - nothing else is copied."""
    _workdir(tmp_path, monkeypatch, {"CENTCAR": GOOD_POSCAR, "NEWMODECAR": GOOD_MODE})
    banked = bank_singlepoint_vasp_restarts([0], _cfg())
    writers, _ = _sp_resume_seed_writers(banked[0], _Frame(), _cfg(), 0, 0)
    out = tmp_path / "run"
    out.mkdir()
    (out / "POSCAR").write_text("PLACEHOLDER")
    (out / "MODECAR").write_text("PLACEHOLDER")
    writers[0](None, None, str(out))
    bank = banked[0]["resume_dir"]
    assert (out / "POSCAR").read_text() == open(f"{bank}/POSCAR").read()
    assert (out / "MODECAR").read_text() == open(f"{bank}/MODECAR").read()


def test_both_files_torn_to_the_same_count_still_refused_by_the_second_gate(
        tmp_path, monkeypatch):
    """Backstop: if a kill truncates geometry AND mode to the same short count
    they agree with each other, so banking accepts them -- but the atom-count
    re-check against the live input frame must still refuse to seed."""
    _workdir(tmp_path, monkeypatch,
             {"CENTCAR": TORN_BODY, "NEWMODECAR": "  0.1 0.2 0.3\n"})
    banked = bank_singlepoint_vasp_restarts([0], _cfg())
    if 0 in banked:
        writers, _ = _sp_resume_seed_writers(banked[0], _Frame(N), _cfg(), 0, 0)
        assert writers is None


def test_continue_from_result_false_never_seeds(tmp_path, monkeypatch):
    _workdir(tmp_path, monkeypatch, {"CENTCAR": GOOD_POSCAR, "NEWMODECAR": GOOD_MODE})
    banked = bank_singlepoint_vasp_restarts([0], _cfg())
    writers, _ = _sp_resume_seed_writers(banked[0], _Frame(), _cfg(continue_flag=False), 0, 0)
    assert writers is None
