"""Sella adapter for SaddleMill's existing Dimer attempt runner.

Sella is a distinct saddle-search algorithm, but SaddleMill dispatches it from
``dimeropt.py`` so it can reuse the established attempt generation, resume
identity, status CSV, output trajectory, VASP lifecycle, and error handling.
The legacy ASE and Kappa dimer paths do not import this module unless
``[ourDimer] engine = sella`` is selected.
"""

from __future__ import annotations

from importlib import metadata
from typing import Any, Mapping

import numpy as np
from ase.mep import MinModeAtoms

SUPPORTED_SELLA_VERSION = "2.5.0"


def validate_sella_environment(required_version: str = SUPPORTED_SELLA_VERSION):
    """Return ``sella.Sella`` after an exact, tested-version check."""
    try:
        version = metadata.version("sella")
    except metadata.PackageNotFoundError as exc:
        raise ImportError(
            "[ourDimer] engine=sella requires sella=="
            f"{required_version} in the authoritative SaddleMill environment."
        ) from exc

    if version != required_version:
        raise RuntimeError(
            "Unsupported Sella version: expected "
            f"{required_version}, found {version}."
        )

    from sella import Sella

    return Sella, version


def apply_attempt_displacement(
    atoms,
    displacement_dict: Mapping[str, Any] | None,
    dimer_control_kwargs: Mapping[str, Any] | None = None,
):
    """Apply the exact ASE ``MinModeAtoms.displace`` semantics in-place.

    ``structure_edit.get_attempts`` returns a displacement dictionary designed
    for ``MinModeAtoms.displace``. Reusing that function avoids introducing a
    second interpretation of masks, displacement radii, selected atoms, or
    Gaussian vectors. Constructing the wrapper and displacing does not evaluate
    the calculator.
    """
    from saddlemill.dimertools.kappa_dimer import IsolatedDimerControl

    control = IsolatedDimerControl(
        logfile=None,
        eigenmode_logfile=None,
        **dict(dimer_control_kwargs or {}),
    )
    wrapped = MinModeAtoms(atoms, control=control)
    if displacement_dict:
        wrapped.displace(log=False, **dict(displacement_dict))
    else:
        wrapped.displace(
            displacement_vector=np.random.randn(len(atoms), 3) * 1.0e-10,
            method="vector",
            log=False,
        )
    return atoms


def _cartesian_mode_to_pes_coordinates(pes, mode_flat: np.ndarray) -> np.ndarray:
    """Map a real-atom Cartesian direction into the active PES coordinates."""
    if getattr(pes, "int", None) is None:
        return mode_flat

    # InternalPES may include dummy atoms. Give those zero Cartesian motion,
    # then map Cartesian displacement to redundant internal displacement.
    jacobian = np.asarray(pes.int.jacobian(), dtype=float)
    if jacobian.ndim != 2:
        raise RuntimeError("Sella internal-coordinate Jacobian is not 2-D.")
    if jacobian.shape[1] < mode_flat.size:
        raise RuntimeError(
            "Sella internal-coordinate Jacobian has fewer Cartesian columns "
            "than the supplied real-atom eigenmode."
        )
    padded = np.zeros(jacobian.shape[1], dtype=float)
    padded[:mode_flat.size] = mode_flat
    return jacobian @ padded


def _pes_mode_to_cartesian(pes, pes_mode: np.ndarray) -> np.ndarray:
    """Map an active PES-coordinate direction back to real-atom Cartesian form."""
    if getattr(pes, "int", None) is None:
        return np.asarray(pes_mode, dtype=float)

    jacobian = np.asarray(pes.int.jacobian(), dtype=float)
    # Minimum-norm Cartesian displacement satisfying B dx ~= dq.
    cart_all = np.linalg.pinv(jacobian, rcond=1.0e-10) @ np.asarray(
        pes_mode, dtype=float
    )
    return cart_all[: 3 * len(pes.atoms)]


def _project_input_mode(optimizer, eigenmode) -> bool:
    """Project an input Cartesian mode into Sella's constrained free subspace."""
    if eigenmode is None:
        return False

    mode = np.asarray(eigenmode, dtype=float)
    expected = (len(optimizer.pes.atoms), 3)
    if mode.shape != expected:
        raise ValueError(
            f"Sella input eigenmode must have shape {expected}, got {mode.shape}."
        )
    if not np.all(np.isfinite(mode)):
        raise ValueError("Sella input eigenmode contains non-finite values.")

    pes_mode = _cartesian_mode_to_pes_coordinates(
        optimizer.pes, mode.reshape(-1)
    )
    ufree = np.asarray(optimizer.pes.get_Ufree(), dtype=float)
    projected = ufree.T @ pes_mode
    norm = float(np.linalg.norm(projected))
    if norm < 1.0e-14:
        raise ValueError(
            "Sella input eigenmode has zero norm after applying constraints."
        )
    optimizer.pes.v0 = projected / norm
    return True


def setup_sella(
    atoms,
    calc,
    *,
    eigenmode=None,
    displacement_dict=None,
    dimer_control_kwargs=None,
    logfile=None,
    trajectory=None,
    sella_options=None,
):
    """Apply one attempt displacement and construct a first-order Sella run."""
    Sella, version = validate_sella_environment()

    # Match the legacy dimer setup order: attach the calculator before
    # constructing the displacement wrapper. The displacement itself performs
    # no force evaluation, but this avoids relying on undocumented constructor
    # behavior in ASE's MinModeAtoms.
    atoms.calc = calc
    apply_attempt_displacement(
        atoms,
        displacement_dict,
        dimer_control_kwargs=dimer_control_kwargs,
    )

    options = dict(sella_options or {})
    # SaddleMill's Sella engine is explicitly first-order. A caller cannot
    # silently change that scientific target through a forwarded option.
    options.pop("order", None)
    optimizer = Sella(
        atoms,
        order=1,
        logfile=logfile,
        trajectory=trajectory,
        **options,
    )
    used_input_mode = _project_input_mode(optimizer, eigenmode)
    optimizer.sm_sella_version = version
    optimizer.sm_used_input_mode = used_input_mode
    return atoms, optimizer


def extract_lowest_mode(optimizer):
    """Return Sella's lowest model-Hessian mode and free-space spectrum.

    This is Sella's final *approximate constrained Hessian*, not an independent
    full finite-difference Hessian. The distinction is recorded in output
    metadata and documentation.
    """
    # Synchronize the PES cache to the final accepted geometry.
    optimizer.pes.get_g()
    ufree = np.asarray(optimizer.pes.get_Ufree(), dtype=float)
    if ufree.ndim != 2 or ufree.shape[1] == 0:
        raise RuntimeError("Sella has no unconstrained degrees of freedom.")

    hproj = np.asarray(
        optimizer.pes.get_HL_projected(ufree).asarray(), dtype=float
    )
    if hproj.shape != (ufree.shape[1], ufree.shape[1]):
        raise RuntimeError(
            "Unexpected Sella projected-Hessian shape: "
            f"{hproj.shape}, expected {(ufree.shape[1], ufree.shape[1])}."
        )
    if not np.all(np.isfinite(hproj)):
        raise RuntimeError("Sella final projected Hessian is non-finite.")

    hproj = 0.5 * (hproj + hproj.T)
    eigenvalues, eigenvectors = np.linalg.eigh(hproj)
    if eigenvalues.size == 0:
        raise RuntimeError("Sella final model Hessian contains no eigenvalues.")

    pes_mode = ufree @ eigenvectors[:, 0]
    cart_mode = _pes_mode_to_cartesian(optimizer.pes, pes_mode)
    mode_norm = float(np.linalg.norm(cart_mode))
    if not np.isfinite(mode_norm) or mode_norm < 1.0e-14:
        raise RuntimeError("Sella final lowest Cartesian mode is non-finite or zero.")

    mode = (cart_mode / mode_norm).reshape((-1, 3))
    return mode, float(eigenvalues[0]), eigenvalues


def classify_sella_convergence(
    stationary_converged: bool,
    eigenvalues,
    *,
    negative_eigenvalue_tolerance: float = 1.0e-6,
    require_first_order_model: bool = True,
):
    """Classify a Sella result using its approximate constrained Hessian.

    Returns ``(converged, status, negative_mode_count)``. This helper is pure
    and independently unit-tested so a stationary point with the wrong model
    order cannot be mislabeled as converged.
    """
    tolerance = float(negative_eigenvalue_tolerance)
    if tolerance < 0.0:
        raise ValueError("negative_eigenvalue_tolerance must be >= 0")
    values = np.asarray(eigenvalues, dtype=float).reshape(-1)
    if values.size and not np.all(np.isfinite(values)):
        raise ValueError("Sella model-Hessian eigenvalues are non-finite.")
    negative_modes = int(np.sum(values < -tolerance))
    order_ok = negative_modes == 1 if require_first_order_model else True
    converged = bool(stationary_converged and order_ok)
    if converged:
        status = "converged"
    elif stationary_converged and not order_ok:
        status = "not_converged_wrong_order"
    else:
        status = "not_converged"
    return converged, status, negative_modes


def sella_force_calls(optimizer) -> int:
    """Return Sella PES energy/gradient evaluation count."""
    return int(getattr(optimizer.pes, "neval", 0))


def sella_options_from_config(config_dict) -> dict[str, Any]:
    """Convert ``[ourSella]`` into Sella 2.5.0 constructor options."""
    cfg = config_dict.get("ourSella", {}) or {}
    options: dict[str, Any] = {
        "internal": bool(cfg.get("internal", False)),
        "eig": bool(cfg.get("eig", True)),
        "method": str(cfg.get("method", "prfo")),
        "delta0": float(cfg.get("delta0", 0.1)),
        "eta": float(cfg.get("eta", 1.0e-4)),
        "gamma": float(cfg.get("gamma", 0.1)),
        "threepoint": bool(cfg.get("threepoint", False)),
        "constraints_tol": float(cfg.get("constraints_tol", 1.0e-5)),
        "nsteps_per_diag": int(cfg.get("nsteps_per_diag", 3)),
        "allow_fragments": bool(cfg.get("allow_fragments", False)),
    }

    project_translations = cfg.get("project_translations", None)
    if project_translations not in (None, "", "None", "none"):
        if not isinstance(project_translations, bool):
            raise ValueError(
                "[ourSella] project_translations must be True, False, or blank"
            )
        options["proj_trans"] = project_translations
    project_rotations = cfg.get("project_rotations", None)
    if project_rotations not in (None, "", "None", "none"):
        if not isinstance(project_rotations, bool):
            raise ValueError(
                "[ourSella] project_rotations must be True, False, or blank"
            )
        options["proj_rot"] = project_rotations

    restricted_step = cfg.get("restricted_step", None)
    if restricted_step not in (None, "", "None", "none"):
        options["rs"] = str(restricted_step)

    diag_every_n = cfg.get("diag_every_n", None)
    if diag_every_n not in (None, "", "None", "none"):
        options["diag_every_n"] = int(diag_every_n)

    for key in ("sigma_inc", "sigma_dec", "rho_inc", "rho_dec"):
        value = cfg.get(key, None)
        if value not in (None, "", "None", "none"):
            options[key] = float(value)

    return options
