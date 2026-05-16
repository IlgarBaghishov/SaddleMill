import pytest
import os
import copy
import numpy as np
from pathlib import Path
from ase.io import read, Trajectory
from ase.build import bulk, fcc111, add_adsorbate
from ase.constraints import FixAtoms
from ase.calculators.emt import EMT
from ase.calculators.singlepoint import SinglePointCalculator

FIXTURES_DIR = Path(__file__).parent / "fixtures"


# --------------- Skip helpers ---------------

def has_cuda():
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


def has_flux():
    try:
        from flux import Flux
        return True
    except ImportError:
        return False


def pytest_collection_modifyitems(config, items):
    for item in items:
        if "gpu" in item.keywords and not has_cuda():
            item.add_marker(pytest.mark.skip(reason="No CUDA GPU available"))
        if "flux" in item.keywords and not has_flux():
            item.add_marker(pytest.mark.skip(reason="Flux scheduler not available"))


# --------------- Config helpers (functions, not fixtures) ---------------

def make_config_dict(method="Minimization", **overrides):
    """Build a config dict from ConfigManager.DEFAULTS with test-friendly defaults.

    Overrides are routed to the correct config section based on key name.
    """
    from saddlemill.config import ConfigManager
    config = copy.deepcopy(ConfigManager.DEFAULTS)
    config["Main"]["method"] = method
    config["Main"]["executorlib"] = False
    config["Main"]["zip"] = False
    config["Main"]["max_consecutive_errors"] = 0
    config["Main"]["steps"] = 50
    config["Main"]["fmax"] = 0.05

    # Ensure optimizer sections exist with reasonable defaults
    config.setdefault("BaseNEB", {
        "k": 5, "climb": True,
        "method": "improvedtangent", "allow_shared_calculator": True,
    })
    config.setdefault("DimerControl", {})
    config.setdefault("MDMin", {"dt": 0.05, "maxstep": 0.1})
    config.setdefault("LBFGS", {"maxstep": 0.1})
    config.setdefault("BFGS", {"maxstep": 0.1})
    config.setdefault("FIRE", {})
    config.setdefault("FAIRChemCalculator", {
        "device": "cuda", "name_or_path": "uma-s-1p2", "task_name": "oc20",
    })

    # Route overrides to the appropriate section. ourDimer / ourMinimization /
    # ourDoubleMinimization / ourSinglePoint all share `vasp_command` and
    # `vasp_ncore` keys — routing by method-specific section based on the
    # current method.
    _main_keys = set(config.get("Main", {}).keys())
    method = config["Main"]["method"]
    method_section_map = {
        "NEB": "ourNEB",
        "Dimer": "ourDimer",
        "Minimization": "ourMinimization",
        "DoubleMinimization": "ourDoubleMinimization",
        "SinglePoint": "ourSinglePoint",
    }
    primary_method_section = method_section_map.get(method)
    _vasp_shared_keys = {"vasp_command", "vasp_ncore"}

    _our_neb_keys = set(config.get("ourNEB", {}).keys()) - _vasp_shared_keys
    _our_dimer_keys = set(config.get("ourDimer", {}).keys()) - _vasp_shared_keys
    _our_min_keys = set(config.get("ourMinimization", {}).keys()) - _vasp_shared_keys
    _our_dmin_keys = set(config.get("ourDoubleMinimization", {}).keys()) - _vasp_shared_keys
    _our_sp_keys = set(config.get("ourSinglePoint", {}).keys()) - _vasp_shared_keys

    for key, val in overrides.items():
        if key in _vasp_shared_keys and primary_method_section in config:
            config[primary_method_section][key] = val
        elif key in _main_keys:
            config["Main"][key] = val
        elif key in _our_neb_keys:
            config["ourNEB"][key] = val
        elif key in _our_dimer_keys:
            config["ourDimer"][key] = val
        elif key in _our_min_keys:
            config["ourMinimization"][key] = val
        elif key in _our_dmin_keys:
            config["ourDoubleMinimization"][key] = val
        elif key in _our_sp_keys:
            config["ourSinglePoint"][key] = val
        elif key.startswith("BaseNEB_"):
            config["BaseNEB"][key[len("BaseNEB_"):]] = val
        elif key.startswith("DimerControl_"):
            config["DimerControl"][key[len("DimerControl_"):]] = val
        else:
            # Default: put in Main
            config["Main"][key] = val

    return config


# --------------- Fixture atoms (function-scoped, CPU) ---------------

@pytest.fixture
def bulk_crystal():
    """3-atom FCC crystal from fixture file."""
    return read(str(FIXTURES_DIR / "bulk_crystal.traj"))


@pytest.fixture
def minimization_input():
    """68-atom slab+adsorbate from fixture file."""
    return read(str(FIXTURES_DIR / "minimization_input.traj"))


@pytest.fixture
def oc_adsorbate_slab():
    """131-atom OC slab+adsorbate from fixture file."""
    return read(str(FIXTURES_DIR / "oc_adsorbate_slab.traj"))


@pytest.fixture
def neb_images():
    """10-frame NEB band from fixture file."""
    return read(str(FIXTURES_DIR / "oc_neb_pair.traj"), index=":")


# --------------- EMT fixtures for unit tests (no GPU) ---------------

@pytest.fixture
def emt_cu_bulk():
    """Cu FCC 2x2x2 (32 atoms) with EMT calculator. Good for bulk dimer reaction types."""
    atoms = bulk("Cu", "fcc", a=3.6, cubic=True) * (2, 2, 2)
    atoms.calc = EMT()
    return atoms


@pytest.fixture
def emt_cu_slab_with_adsorbate():
    """Cu(111) 3x3x3 slab + C,O adsorbate, tags set (0=fixed, 1=surface, 2=adsorbate).

    Good for OC-mode structure_edit tests.
    """
    slab = fcc111("Cu", size=(3, 3, 3), vacuum=10.0)

    # Set tags: bottom 2 layers = 0, top layer = 1
    z_positions = slab.positions[:, 2]
    z_unique = np.sort(np.unique(np.round(z_positions, 2)))
    tags = np.zeros(len(slab), dtype=int)
    for idx, atom in enumerate(slab):
        z_rounded = round(atom.position[2], 2)
        if z_rounded >= z_unique[-1]:
            tags[idx] = 1
    slab.set_tags(tags)

    # Add adsorbate atoms
    add_adsorbate(slab, "C", height=1.8, position="ontop")
    add_adsorbate(slab, "O", height=3.0, position="ontop")
    new_tags = slab.get_tags().copy()
    new_tags[-2] = 2  # C
    new_tags[-1] = 2  # O
    slab.set_tags(new_tags)

    # Fix substrate
    substrate_idx = np.where(new_tags == 0)[0]
    slab.set_constraint(FixAtoms(indices=substrate_idx))

    slab.calc = EMT()
    return slab


# --------------- GPU fixtures (session-scoped) ---------------

@pytest.fixture(scope="session")
def fairchem_calc():
    """Session-scoped FAIRChem calculator. Loads model once, reused across all GPU tests."""
    if not has_cuda():
        pytest.skip("No CUDA GPU available")
    from fairchem.core import FAIRChemCalculator
    calc = FAIRChemCalculator.from_model_checkpoint(
        device="cuda",
        name_or_path="uma-s-1p2",
        task_name="oc20",
    )
    return calc


@pytest.fixture(scope="session")
def converged_ts_atoms():
    """Pre-generated converged TS from fixtures/converged_ts.traj.

    Contains eigenmode, converged=1, src_index=0, status='converged' in
    .info, plus SinglePointCalculator with energy/forces. Generated once
    via Dimer with FAIRChem on oc_adsorbate_slab.
    """
    atoms = read(str(FIXTURES_DIR / "converged_ts.traj"))
    atoms.info.setdefault("status", "converged")
    return atoms
