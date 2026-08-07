"""Dimer rotation helpers and ASE-backed translation optimizers.

L-BFGS rotation remains a specialized implementation because ASE's ordinary
Cartesian optimizer cannot directly optimize the normalized dimer mode while
retaining ASE's trial-angle/Fourier rotation.

Pure L-BFGS translation now delegates direction generation, secant-history
updates, initial scaling, damping, and max-step handling to ``ase.optimize.LBFGS``
on the ``MinModeAtoms`` wrapper.  ``MinModeAtoms.get_forces()`` supplies the
projected minimum-mode-following force automatically.

Three safeguards are layered on top of ASE, because ASE's ``LBFGSMethod`` and
its convergence test are written for ordinary minimization and assume a
conservative field:

1. Curvature damping.  ``LBFGSMethod.update`` stores every secant pair as
   ``rho = 1/(s.y)`` with no sign or magnitude guard.  The min-mode-following
   force is not the gradient of any scalar function, so ``s.y <= 0`` occurs
   routinely and inserts a negative ``rho`` that actively corrupts the search
   direction; ``s.y ~ 0`` produces an unbounded ``rho``.  Every pair is passed
   through Powell damping first, which guarantees positive curvature while
   keeping ASE's bookkeeping (``iteration``/``s``/``y``/``rho`` stay aligned,
   so ``compute_step`` indexes the newest pairs as intended).

2. Effective-state history resets.  ``MinModeAtoms.get_projected_forces``
   switches functional form on the sign of the curvature -- it returns
   ``-parallel_vector(f, mode)`` when curvature > 0 and ``f - 2*parallel(f,
   mode)`` when curvature < 0 -- and ``KappaMinModeAtoms`` additionally
   switches gamma scaling on ``translation_regime``.  Those are different
   vector fields, so history that straddles a switch is invalid.  The reset
   key covers both.  Previously only ``translation_regime`` was consulted, and
   ``ConfigurableRotationMinModeAtoms`` never sets that attribute, so under
   ``engine=ase`` the history was never reset at all.

3. Real-force convergence.  ASE converges on the projected force, which the
   kappa gamma scaling and ASE's own curvature branch can drive to zero while
   the real force is large.  Convergence now requires both.
"""

from __future__ import annotations

from collections import deque
from math import atan, cos, pi, sin, tan
from typing import Callable, Optional
import warnings

import numpy as np

from ase.optimize import LBFGS

from ase.mep.dimer import (
    DimerEigenmodeSearch,
    DimerControl,
    MinModeAtoms,
    MinModeTranslate,
    normalize,
    perpendicular_vector,
    rotate_vectors,
)

norm = np.linalg.norm

# Minimum curvature (eV/A^2) accepted along a secant pair before Powell
# damping intervenes. Small enough to be inert on well-behaved steps.
DEFAULT_CURVATURE_FLOOR = 1.0e-3


def _ase_lbfgs_container(optimizer):
    """Return the object that owns ASE L-BFGS history attributes.

    ASE <= 3.28 stores ``iteration/s/y/rho/H0`` directly on ``LBFGS``.
    ASE >= 3.29 stores them on ``LBFGS.state``.  SaddleMill supports both
    layouts here while still delegating every actual L-BFGS step/update to ASE.
    """
    return getattr(optimizer, "state", optimizer)


def _ase_lbfgs_api_name(optimizer):
    return "state_object" if hasattr(optimizer, "state") else "legacy_direct"


def _ase_lbfgs_get(optimizer, name, default=None):
    return getattr(_ase_lbfgs_container(optimizer), name, default)


def _ase_lbfgs_set(optimizer, name, value):
    setattr(_ase_lbfgs_container(optimizer), name, value)


def _ase_lbfgs_increment_iteration(optimizer):
    _ase_lbfgs_set(
        optimizer,
        "iteration",
        int(_ase_lbfgs_get(optimizer, "iteration", 0)) + 1,
    )


def _ase_lbfgs_history_size(optimizer):
    return int(len(_ase_lbfgs_get(optimizer, "s", [])))


def _ase_lbfgs_pairs_total(optimizer):
    # The first ASE L-BFGS iteration has no secant pair.
    return int(max(0, int(_ase_lbfgs_get(optimizer, "iteration", 0)) - 1))


def _ase_lbfgs_reset_history(optimizer, memory):
    """Reset native ASE history without replacing the optimizer wrapper.

    The state-object API needs a new LBFGSMethod instance; the legacy API uses
    direct lists/scalars.  ``H0`` is preserved in both cases.
    """
    if hasattr(optimizer, "state"):
        old_state = optimizer.state
        optimizer.state = type(old_state)(
            memory=int(memory),
            initial_inverse_hessian=old_state.H0,
        )
    else:
        optimizer.iteration = 0
        optimizer.s = []
        optimizer.y = []
        optimizer.rho = []
        # Legacy ASE stores H0 and memory directly on the optimizer.
        optimizer.memory = int(memory)

    optimizer.r0 = None
    optimizer.f0 = None
    optimizer.e0 = None
    optimizer.task = "START"


def _damp_secant_forces(pos, forces, r0, f0, curvature_floor):
    """Powell-damp the secant pair ASE is about to store.

    ASE forms ``s = pos - r0`` and ``y = (-forces) - (-f0) = f0 - forces``.
    When ``s.y`` falls below ``curvature_floor * |s|^2`` the pair is replaced
    by ``y' = y + theta*s`` with ``theta`` chosen so ``s.y' == floor``, which
    is achieved by handing ASE ``forces - theta*s`` instead of ``forces``.

    Returns ``(forces_for_update, was_damped, raw_sy)``.
    """
    s = np.asarray(pos, dtype=float).reshape(-1) - np.asarray(
        r0, dtype=float
    ).reshape(-1)
    y = np.asarray(f0, dtype=float).reshape(-1) - np.asarray(
        forces, dtype=float
    ).reshape(-1)
    ss = float(np.dot(s, s))
    sy = float(np.dot(s, y))
    if not np.isfinite(ss) or not np.isfinite(sy) or ss <= 0.0:
        return forces, False, sy
    floor = float(curvature_floor) * ss
    if sy >= floor:
        return forces, False, sy
    theta = (floor - sy) / ss
    damped = np.asarray(forces, dtype=float).reshape(-1) - theta * s
    return damped.reshape(np.shape(forces)), True, sy


class _CurvatureGuardedLBFGS(LBFGS):
    """ASE ``LBFGS`` with Powell damping applied before each history update.

    Only ``update`` is overridden; direction generation, scaling, max-step
    handling, and the two-loop recursion remain ASE's.
    """

    def __init__(self, *args, curvature_floor=DEFAULT_CURVATURE_FLOOR, **kwargs):
        self.curvature_floor = float(curvature_floor)
        self.sm_pairs_damped = 0
        self.sm_pairs_seen = 0
        self.sm_worst_sy = float("inf")
        super().__init__(*args, **kwargs)

    def update(self, pos, forces, r0, f0):
        container = _ase_lbfgs_container(self)
        if int(getattr(container, "iteration", 0)) > 0 and r0 is not None:
            forces, damped, raw_sy = _damp_secant_forces(
                pos, forces, r0, f0, self.curvature_floor
            )
            self.sm_pairs_seen += 1
            if np.isfinite(raw_sy):
                self.sm_worst_sy = min(self.sm_worst_sy, raw_sy)
            if damped:
                self.sm_pairs_damped += 1
        super().update(pos, forces, r0, f0)

    def sm_guard_metrics(self):
        return {
            "pairs_seen": int(self.sm_pairs_seen),
            "pairs_damped": int(self.sm_pairs_damped),
            "worst_sy": (
                float(self.sm_worst_sy)
                if np.isfinite(self.sm_worst_sy)
                else ""
            ),
        }


def _translation_state_key(dimeratoms):
    """Identity of the effective force field currently being optimized.

    Both the kappa gamma regime and ASE's curvature-sign branch in
    ``get_projected_forces`` change which function the optimizer is following.
    History accumulated under one is meaningless under the other.
    """
    regime = str(getattr(dimeratoms, "translation_regime", "standard"))
    try:
        curvature = float(dimeratoms.get_curvature())
    except Exception:
        curvature = -1.0
    branch = "convex" if curvature > 0.0 else "concave"
    return f"{regime}:{branch}"


class LimitedMemoryInverseHessian:
    """Small, dependency-free L-BFGS inverse-Hessian approximation.

    ``force`` is interpreted as minus the gradient.  Therefore ``apply``
    returns ``H^{-1-like} * force``, i.e. an optimization step direction.
    Only secant pairs satisfying positive curvature are retained, keeping the
    approximation positive definite.
    """

    def __init__(
        self,
        memory: int = 10,
        initial_hessian: float = 1.0,
        dynamic_h0: bool = False,
        curvature_epsilon: float = 1.0e-12,
    ):
        if int(memory) < 1:
            raise ValueError("L-BFGS memory must be >= 1")
        if float(initial_hessian) <= 0.0:
            raise ValueError("initial_hessian must be > 0")
        if float(curvature_epsilon) < 0.0:
            raise ValueError("curvature_epsilon must be >= 0")

        self.memory = int(memory)
        self.initial_hessian = float(initial_hessian)
        self.dynamic_h0 = bool(dynamic_h0)
        self.curvature_epsilon = float(curvature_epsilon)
        self.s_history: deque[np.ndarray] = deque(maxlen=self.memory)
        self.y_history: deque[np.ndarray] = deque(maxlen=self.memory)
        self.accepted_pairs_total = 0
        self.rejected_pairs_total = 0
        self.reset_count = 0
        self.last_reset_reason = "initial"

    @property
    def size(self) -> int:
        return len(self.s_history)

    def reset(self, reason: str = "manual") -> None:
        self.s_history.clear()
        self.y_history.clear()
        self.reset_count += 1
        self.last_reset_reason = str(reason)

    def add_pair(self, s, y) -> bool:
        s = np.asarray(s, dtype=float).reshape(-1)
        y = np.asarray(y, dtype=float).reshape(-1)
        if s.shape != y.shape or not np.all(np.isfinite(s)) or not np.all(np.isfinite(y)):
            self.rejected_pairs_total += 1
            return False

        sy = float(np.dot(s, y))
        scale = float(norm(s) * norm(y))
        threshold = self.curvature_epsilon * max(1.0, scale)
        if sy <= threshold:
            self.rejected_pairs_total += 1
            return False

        self.s_history.append(s.copy())
        self.y_history.append(y.copy())
        self.accepted_pairs_total += 1
        return True

    def _h0_scale(self, pairs) -> float:
        # H0 is the inverse of the configured initial Hessian.
        scale = 1.0 / self.initial_hessian
        if self.dynamic_h0 and pairs:
            s, y, sy = pairs[-1]
            yy = float(np.dot(y, y))
            if yy > 0.0 and sy > 0.0:
                scale = sy / yy
        return scale

    def apply(
        self,
        force,
        projector: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    ) -> np.ndarray:
        shape = np.asarray(force).shape
        q = np.asarray(force, dtype=float).reshape(-1)
        if projector is not None:
            q = np.asarray(projector(q.reshape(shape)), dtype=float).reshape(-1)

        pairs = []
        for s_raw, y_raw in zip(self.s_history, self.y_history):
            s = s_raw.reshape(shape)
            y = y_raw.reshape(shape)
            if projector is not None:
                s = projector(s)
                y = projector(y)
            s = np.asarray(s, dtype=float).reshape(-1)
            y = np.asarray(y, dtype=float).reshape(-1)
            sy = float(np.dot(s, y))
            if sy > self.curvature_epsilon * max(1.0, norm(s) * norm(y)):
                pairs.append((s, y, sy))

        alphas = []
        for s, y, sy in reversed(pairs):
            rho = 1.0 / sy
            alpha = rho * float(np.dot(s, q))
            alphas.append(alpha)
            q = q - alpha * y

        r = self._h0_scale(pairs) * q
        for (s, y, sy), alpha in zip(pairs, reversed(alphas)):
            rho = 1.0 / sy
            beta = rho * float(np.dot(y, r))
            r = r + s * (alpha - beta)

        result = r.reshape(shape)
        if projector is not None:
            result = projector(result)
        return np.asarray(result, dtype=float)


class LBFGSRotationMixin:
    """Replace ASE's rotational steepest direction with L-BFGS.

    The physical rotational-force norms still control ``f_rot_min``,
    ``f_rot_max`` and ``max_num_rot`` exactly as in ASE.  The trial angle,
    Fourier interpolation, and optional endpoint-force extrapolation are also
    retained unchanged.
    """

    def __init__(self, *args, lbfgs_options=None, **kwargs):
        self.lbfgs_options = dict(lbfgs_options or {})
        super().__init__(*args, **kwargs)

    def _project_rotation_vector(self, vector, mode):
        projected = perpendicular_vector(np.asarray(vector, dtype=float), mode)
        if self.basis is not None:
            basis_items = self.basis
            if (
                isinstance(basis_items, np.ndarray)
                and basis_items.shape == projected.shape
            ):
                basis_items = [basis_items]
            for base in basis_items:
                projected = perpendicular_vector(projected, base)
        return projected

    def converge_to_eigenmode(self):
        self.set_up_for_eigenmode_search()
        stoprot = False

        f_rot_min = self.control.get_parameter("f_rot_min")
        f_rot_max = self.control.get_parameter("f_rot_max")
        trial_angle = self.control.get_parameter("trial_angle")
        max_num_rot = self.control.get_parameter("max_num_rot")
        extrapolate = self.control.get_parameter("extrapolate_forces")

        history = LimitedMemoryInverseHessian(**self.lbfgs_options)
        previous_mode = None
        previous_force = None
        direction_fallbacks = 0

        while not stoprot:
            if self.forces1E is None:
                self.update_virtual_forces()
            else:
                self.update_virtual_forces(extrapolated_forces=True)
            self.forces1A = self.forces1
            self.update_curvature()
            f_rot_A = self.get_rotational_force()

            # Preserve ASE's physical-force stopping criteria.
            if norm(f_rot_A) <= f_rot_min:
                self.log(f_rot_A, None)
                stoprot = True
            else:
                n_A = np.asarray(self.eigenmode, dtype=float).copy()

                # Work in a sign-continuous representation because n and -n
                # denote the same dimer axis.
                sign = 1.0
                history_mode = n_A.copy()
                history_force = np.asarray(f_rot_A, dtype=float).copy()
                if previous_mode is not None and np.vdot(
                    previous_mode.ravel(), history_mode.ravel()
                ).real < 0.0:
                    sign = -1.0
                    history_mode *= -1.0
                    history_force *= -1.0

                projector = lambda v: self._project_rotation_vector(v, history_mode)
                if previous_mode is not None:
                    s = projector(history_mode - previous_mode)
                    # y = grad_new - grad_old = force_old - force_new.
                    y = projector(previous_force - history_force)
                    history.add_pair(s, y)

                lbfgs_direction = history.apply(history_force, projector=projector)
                rot_direction = sign * lbfgs_direction
                rot_direction = self._project_rotation_vector(rot_direction, n_A)

                # A positive-definite inverse-Hessian approximation should give
                # a direction with positive projection on the force.  Fall back
                # safely if finite precision or transported history violates it.
                if (
                    not np.all(np.isfinite(rot_direction))
                    or norm(rot_direction) < 1.0e-14
                    or np.vdot(rot_direction, f_rot_A).real <= 0.0
                ):
                    history.reset("non_descent_rotation_direction")
                    direction_fallbacks += 1
                    rot_direction = f_rot_A.copy()

                rot_unit_A = normalize(rot_direction)
                c0 = self.get_curvature()
                c0d = np.vdot((self.forces2 - self.forces1), rot_unit_A) / self.dR

                # ASE/Heyden trial-angle evaluation and Fourier interpolation.
                n_B, rot_unit_B = rotate_vectors(n_A, rot_unit_A, trial_angle)
                self.eigenmode = n_B
                self.update_virtual_forces()
                self.forces1B = self.forces1
                c1d = np.vdot((self.forces2 - self.forces1), rot_unit_B) / self.dR

                a1 = c0d * cos(2 * trial_angle) - c1d / (2 * sin(2 * trial_angle))
                b1 = 0.5 * c0d
                a0 = 2 * (c0 - a1)
                rotangle = atan(b1 / a1) / 2.0
                cmin = a0 / 2.0 + a1 * cos(2 * rotangle) + b1 * sin(2 * rotangle)
                if c0 < cmin:
                    rotangle += pi / 2.0

                n_min, _ = rotate_vectors(n_A, rot_unit_A, rotangle)
                self.update_eigenmode(n_min)
                self.update_curvature(cmin)
                self.log(f_rot_A, rotangle)

                if extrapolate:
                    self.forces1E = (
                        sin(trial_angle - rotangle) / sin(trial_angle) * self.forces1A
                        + sin(rotangle) / sin(trial_angle) * self.forces1B
                        + (
                            1
                            - cos(rotangle)
                            - sin(rotangle) * tan(trial_angle / 2.0)
                        )
                        * self.forces0
                    )
                else:
                    self.forces1E = None

                previous_mode = history_mode.copy()
                previous_force = history_force.copy()

            if not stoprot:
                if self.control.get_counter("rotcount") >= max_num_rot:
                    stoprot = True
                elif norm(f_rot_A) <= f_rot_max:
                    stoprot = True

        self.lbfgs_diagnostics = {
            "optimizer": "lbfgs",
            "rotations": int(self.control.get_counter("rotcount")),
            "history_size": int(history.size),
            "pairs_accepted": int(history.accepted_pairs_total),
            "pairs_rejected": int(history.rejected_pairs_total),
            "history_resets": int(history.reset_count),
            "direction_fallbacks": int(direction_fallbacks),
        }


class LBFGSDimerEigenmodeSearch(LBFGSRotationMixin, DimerEigenmodeSearch):
    pass


class ConfigurableRotationMinModeAtoms(MinModeAtoms):
    """ASE MinModeAtoms with selectable ASE or L-BFGS rotation direction."""

    def __init__(
        self,
        atoms,
        control=None,
        rotation_optimizer="ase",
        rotation_lbfgs_options=None,
        **kwargs,
    ):
        self.rotation_optimizer = str(rotation_optimizer).lower()
        self.rotation_lbfgs_options = dict(rotation_lbfgs_options or {})
        self.last_rotation_diagnostics = {}
        # Declared so the translation-history reset key is meaningful under the
        # ASE engine too; the curvature branch is what actually varies here.
        self.translation_regime = "standard"
        if self.rotation_optimizer not in {"ase", "lbfgs"}:
            raise ValueError(
                "rotation_optimizer must be 'ase' or 'lbfgs'; got "
                f"{rotation_optimizer!r}"
            )
        super().__init__(atoms, control=control, **kwargs)

    def find_eigenmodes(self, order=1):
        if self.rotation_optimizer == "ase":
            super().find_eigenmodes(order=order)
            self.last_rotation_diagnostics = {
                "phase_a": {
                    "optimizer": "ase",
                    "rotations": int(self.control.get_counter("rotcount")),
                    "history_size": 0,
                    "pairs_accepted": 0,
                    "pairs_rejected": 0,
                    "history_resets": 0,
                    "direction_fallbacks": 0,
                }
            }
            return

        if self.control.get_parameter("eigenmode_method").lower() != "dimer":
            raise NotImplementedError("Only the Dimer eigenmode method is implemented")

        phase_diagnostics = []
        for k in range(order):
            if k > 0:
                self.ensure_eigenmode_orthogonality(k + 1)
            search = LBFGSDimerEigenmodeSearch(
                self,
                self.control,
                eigenmode=self.eigenmodes[k],
                basis=self.eigenmodes[:k],
                lbfgs_options=self.rotation_lbfgs_options,
            )
            search.converge_to_eigenmode()
            search.set_up_for_optimization_step()
            self.eigenmodes[k] = search.get_eigenmode()
            self.curvatures[k] = search.get_curvature()
            phase_diagnostics.append(dict(search.lbfgs_diagnostics))

        self.last_rotation_diagnostics = {
            "phase_a": phase_diagnostics[0] if phase_diagnostics else {}
        }



def _force_calls(dimeratoms) -> int:
    try:
        return int(dimeratoms.control.get_counter("forcecalls"))
    except Exception:
        return -1


def _projected_fmax(force) -> float:
    force = np.asarray(force, dtype=float)
    if force.size == 0:
        return 0.0
    if force.ndim == 1:
        force = force.reshape(-1, 3)
    return float(np.sqrt((force * force).sum(axis=1).max()))


def _real_fmax(dimeratoms) -> float:
    """True atomic fmax, independent of any projection or gamma scaling.

    ``MinModeTranslate`` converges on the projected force.  Under the kappa
    gamma scaling, and under ASE's curvature-sign branch, the projected force
    can vanish while the real force is large, which registers as a converged
    run that is not a stationary point.  Recorded here so the condition is at
    least visible in the optimizer diagnostics.
    """
    try:
        real = np.asarray(dimeratoms.get_forces(real=True), dtype=float)
    except Exception:
        return float("nan")
    if real.size == 0:
        return 0.0
    return float(np.sqrt((real * real).sum(axis=1).max()))


def _cosine_alignment(direction, force):
    direction = np.asarray(direction, dtype=float).reshape(-1)
    force = np.asarray(force, dtype=float).reshape(-1)
    denom = float(norm(direction) * norm(force))
    if denom <= 0.0:
        return ""
    return float(np.dot(direction, force) / denom)


def _flatten_rotation_diagnostics(dimeratoms):
    raw = getattr(dimeratoms, "last_rotation_diagnostics", {}) or {}
    phase_a = raw.get("phase_a", {}) or {}
    phase_b = raw.get("phase_b", {}) or {}
    return {
        "rotation_optimizer": phase_a.get("optimizer", "ase"),
        "rotation_steps": phase_a.get(
            "rotations", dimeratoms.control.get_counter("rotcount")
        ),
        "rotation_lbfgs_history_size": phase_a.get("history_size", 0),
        "rotation_lbfgs_pairs_accepted": phase_a.get("pairs_accepted", 0),
        "rotation_lbfgs_pairs_rejected": phase_a.get("pairs_rejected", 0),
        "rotation_lbfgs_resets": phase_a.get("history_resets", 0),
        "rotation_lbfgs_fallbacks": phase_a.get("direction_fallbacks", 0),
        "kappa_rotation_optimizer": phase_b.get("optimizer", ""),
        "kappa_rotation_steps": phase_b.get("rotations", ""),
        "kappa_rotation_lbfgs_pairs_accepted": phase_b.get("pairs_accepted", ""),
        "kappa_rotation_lbfgs_pairs_rejected": phase_b.get("pairs_rejected", ""),
    }


class _RealForceConvergenceMixin:
    """Require both projected and real atomic forces to satisfy fmax."""

    def gradient_converged(self, gradient):
        # First apply ASE/Dimer's normal projected-force and curvature test.
        if not super().gradient_converged(gradient):
            return False

        fmax = getattr(self, "fmax", None)
        if fmax is None:
            return True

        try:
            real = np.asarray(
                self.dimeratoms.get_forces(real=True),
                dtype=float,
            )
        except Exception as exc:
            raise RuntimeError(
                "Unable to evaluate real force for Dimer convergence"
            ) from exc

        if real.size == 0:
            return True

        real_fmax = float(
            np.sqrt((real * real).sum(axis=1).max())
        )

        if real_fmax < float(fmax):
            return True

        self._sm_projected_only_convergences = (
            getattr(self, "_sm_projected_only_convergences", 0) + 1
        )
        return False

class _TranslationDiagnosticsMixin:
    def _initialize_step_diagnostics(self):
        self.last_step_diagnostics = None
        self._diagnostic_serial = 0
        self._previous_force_calls_after_step = 0

    def _start_step_diagnostics(self, force, algorithm, hybrid_state="", switch_event=""):
        entry_calls = _force_calls(self.dimeratoms)
        center_rotation_calls = (
            entry_calls - self._previous_force_calls_after_step
            if entry_calls >= 0
            else ""
        )
        data = {
            "diagnostic_serial": self._diagnostic_serial + 1,
            "accepted_translation_step": int(self.nsteps) + 1,
            "translation_algorithm": algorithm,
            "hybrid_state": hybrid_state,
            "hybrid_switch_event": switch_event,
            "projected_fmax": _projected_fmax(force),
            "real_fmax": _real_fmax(self.dimeratoms),
            "curvature": float(self.dimeratoms.get_curvature()),
            "translation_regime": getattr(
                self.dimeratoms, "translation_regime", "standard"
            ),
            "translation_state_key": _translation_state_key(self.dimeratoms),
            "force_calls_step_entry": entry_calls,
            "force_calls_center_and_rotation": center_rotation_calls,
        }
        data.update(_flatten_rotation_diagnostics(self.dimeratoms))
        return data

    def _finish_step_diagnostics(self, data, step_norm, lbfgs_metrics=None):
        after_calls = _force_calls(self.dimeratoms)
        entry_calls = data["force_calls_step_entry"]
        metrics = dict(lbfgs_metrics or {})
        data.update(
            {
                "step_norm": float(step_norm),
                "force_calls_translation_trial": (
                    after_calls - entry_calls
                    if after_calls >= 0 and entry_calls >= 0
                    else ""
                ),
                "force_calls_cumulative_after_step": after_calls,
                "translation_lbfgs_history_size": metrics.get("history_size", 0),
                "translation_lbfgs_pairs_accepted_total": metrics.get(
                    "pairs_accepted_total", 0
                ),
                "translation_lbfgs_pairs_rejected_total": metrics.get(
                    "pairs_rejected_total", 0
                ),
                "translation_lbfgs_pairs_damped_total": metrics.get(
                    "pairs_damped_total", 0
                ),
                "translation_lbfgs_worst_sy": metrics.get("worst_sy", ""),
                "translation_lbfgs_resets": metrics.get("reset_count", 0),
                "translation_lbfgs_last_reset_reason": metrics.get(
                    "last_reset_reason", ""
                ),
            }
        )
        self._diagnostic_serial += 1
        self.last_step_diagnostics = data
        self._previous_force_calls_after_step = max(after_calls, 0)


class DiagnosticMinModeTranslate(
    _RealForceConvergenceMixin, _TranslationDiagnosticsMixin, MinModeTranslate
):
    """Stock ASE ``MinModeTranslate`` with diagnostics only."""

    def __init__(self, dimeratoms, logfile="-", trajectory=None):
        super().__init__(dimeratoms, logfile=logfile, trajectory=trajectory)
        self._initialize_step_diagnostics()

    def step(self, f=None):
        if f is None:
            f = self.dimeratoms.get_forces()
        r_before = self.dimeratoms.get_positions().copy()
        algorithm = "ase_cg" if self.cg_on else "ase_steepest"
        data = self._start_step_diagnostics(f, algorithm)
        MinModeTranslate.step(self, f)
        step_norm = norm(self.dimeratoms.get_positions() - r_before)
        self._finish_step_diagnostics(data, step_norm)


class _ASELBFGSState:
    """Thin state/diagnostic adapter around an actual ASE ``LBFGS`` object.

    No L-BFGS recursion is implemented here.  ``step()`` delegates to ASE.
    ``rebuild()`` mirrors ASE's own replay loop in generalized coordinates so
    warm starts also work for filters such as ``FrechetCellFilter``.
    """

    def __init__(self, target, *, maxstep, memory, damping, alpha,
                 curvature_floor=DEFAULT_CURVATURE_FLOOR):
        self.target = target
        self.maxstep = float(maxstep)
        self.memory = int(memory)
        self.damping = float(damping)
        self.alpha = float(alpha)
        self.curvature_floor = float(curvature_floor)
        self.optimizer = self._new_optimizer()
        self.reset_count = 0
        self.last_reset_reason = "initial"
        self.pairs_damped_carry = 0
        self.pairs_seen_carry = 0
        self.worst_sy_carry = float("inf")

    def _new_optimizer(self):
        return _CurvatureGuardedLBFGS(
            self.target,
            restart=None,
            logfile=None,
            trajectory=None,
            maxstep=self.maxstep,
            memory=self.memory,
            damping=self.damping,
            alpha=self.alpha,
            use_line_search=False,
            curvature_floor=self.curvature_floor,
        )

    def _absorb_counters(self):
        """Carry guard counters across optimizer replacement."""
        g = self.optimizer.sm_guard_metrics()
        self.pairs_damped_carry += int(g["pairs_damped"])
        self.pairs_seen_carry += int(g["pairs_seen"])
        if g["worst_sy"] != "":
            self.worst_sy_carry = min(self.worst_sy_carry, float(g["worst_sy"]))

    def reset(self, reason="manual"):
        self._absorb_counters()
        self.optimizer = self._new_optimizer()
        self.reset_count += 1
        self.last_reset_reason = str(reason)

    @property
    def history_size(self):
        return _ase_lbfgs_history_size(self.optimizer)

    @property
    def pairs_accepted_total(self):
        # iteration includes the initial no-pair direction; accepted secant
        # updates are represented directly by the stored s/y lists.
        return _ase_lbfgs_pairs_total(self.optimizer)

    def metrics(self):
        g = self.optimizer.sm_guard_metrics()
        damped = self.pairs_damped_carry + int(g["pairs_damped"])
        worst = self.worst_sy_carry
        if g["worst_sy"] != "":
            worst = min(worst, float(g["worst_sy"]))
        return {
            "history_size": self.history_size,
            "pairs_accepted_total": self.pairs_accepted_total,
            # Damped rather than dropped: ASE's compute_step indexes s/y/rho by
            # iteration, so removing a pair would desynchronize its bookkeeping.
            "pairs_rejected_total": damped,
            "pairs_damped_total": damped,
            "worst_sy": float(worst) if np.isfinite(worst) else "",
            "reset_count": self.reset_count,
            "last_reset_reason": self.last_reset_reason,
            "ase_lbfgs_api": _ase_lbfgs_api_name(self.optimizer),
        }

    def rebuild(self, snapshots, reason="warm_start_rebuild"):
        """Rebuild ASE's native state from ``(x, projected_force)`` snapshots.

        The final snapshot is intentionally not replayed.  ASE's next real
        ``step()`` closes that last secant pair, exactly as its private
        ``_replay_trajectory`` implementation does for an ordinary trajectory.
        """
        self.reset(reason)
        opt = self.optimizer
        r0 = None
        f0 = None
        for position, force in list(snapshots)[:-1]:
            position = np.asarray(position, dtype=float).reshape(-1)
            force = np.asarray(force, dtype=float).reshape(-1)
            opt.update(position, force, r0, f0)
            r0 = position.copy()
            f0 = force.copy()
            _ase_lbfgs_increment_iteration(opt)
        opt.r0 = r0
        opt.f0 = f0

    def step(self, force):
        force = np.asarray(force, dtype=float)
        position_before = np.asarray(
            self.optimizer.optimizable.get_x(), dtype=float
        ).reshape(-1).copy()
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=r"Please do not pass forces to step\(\)\..*",
                category=UserWarning,
                module=r"ase\.optimize\.optimize",
            )
            self.optimizer.step(forces=force)
        position_after = np.asarray(
            self.optimizer.optimizable.get_x(), dtype=float
        ).reshape(-1)
        displacement = position_after - position_before
        step_metric = float(norm(displacement))
        raw_direction = np.asarray(self.optimizer.p, dtype=float).reshape(-1)
        raw_metric = float(
            self.optimizer.optimizable.gradient_norm(raw_direction)
        )
        # ASE clips the raw direction before multiplying by damping.
        clipped = bool(raw_metric >= self.optimizer.maxstep)
        alignment = _cosine_alignment(raw_direction, force)
        return step_metric, clipped, alignment


class _DimerLBFGSLogMixin:
    """Retain the dimer optimizer log columns while using ASE optimizers."""

    def _initialize_dimer_log(self, write_header=True):
        self._last_step_size = None
        if write_header and self.logfile is not None:
            self.logfile.write(
                "MinModeTranslate: STEP      TIME          ENERGY    "
                "MAX-FORCE     STEPSIZE    CURVATURE  ROT-STEPS\n"
            )

    def log(self, gradient):
        import time

        force = -np.asarray(gradient, dtype=float).reshape(-1, 3)
        fmax = _projected_fmax(force)
        energy = self.dimeratoms.get_potential_energy()
        curvature = self.dimeratoms.get_curvature()
        rotsteps = self.dimeratoms.control.get_counter("rotcount")
        if self.logfile is None:
            return
        now = time.localtime()
        if self._last_step_size is None:
            step_field = "    --------"
        else:
            step_field = f"{float(self._last_step_size):12.6f}"
        line = (
            f"MinModeTranslate: {self.nsteps:4d}  "
            f"{now[3]:02d}:{now[4]:02d}:{now[5]:02d} "
            f"{energy:15.6f} {fmax:12.4f} {step_field} "
            f"{curvature:12.6f} {rotsteps:10d}\n"
        )
        self.logfile.write(line)


class LBFGSMinModeTranslate(
    _RealForceConvergenceMixin,
    _TranslationDiagnosticsMixin,
    _DimerLBFGSLogMixin,
    _CurvatureGuardedLBFGS,
):
    """ASE ``LBFGS`` applied directly to ``MinModeAtoms`` projected forces."""

    def __init__(self, dimeratoms, logfile="-", trajectory=None, lbfgs_options=None):
        options = dict(lbfgs_options or {})
        self.dimeratoms = dimeratoms
        self.control = dimeratoms.get_control()
        self.reset_on_regime_change = bool(
            options.pop("reset_on_regime_change", True)
        )
        # Legacy custom-recursion knobs are accepted but intentionally ignored.
        options.pop("dynamic_h0", None)
        options.pop("curvature_epsilon", None)
        memory = int(options.pop("memory", 10))
        alpha = float(options.pop("initial_hessian", 70.0))
        damping = float(options.pop("damping", 1.0))
        curvature_floor = float(
            options.pop("curvature_floor", DEFAULT_CURVATURE_FLOOR)
        )
        if options:
            raise TypeError(f"Unknown ASE dimer L-BFGS options: {sorted(options)}")
        self._sm_memory = memory
        self._sm_alpha = alpha
        self._sm_damping = damping
        self._sm_state_key = None
        self._sm_reset_count = 0
        self._sm_last_reset_reason = "initial"
        _CurvatureGuardedLBFGS.__init__(
            self,
            dimeratoms,
            restart=None,
            logfile=logfile,
            trajectory=trajectory,
            maxstep=float(self.control.get_parameter("maximum_translation")),
            memory=memory,
            damping=damping,
            alpha=alpha,
            use_line_search=False,
            curvature_floor=curvature_floor,
        )
        self._initialize_step_diagnostics()
        self._initialize_dimer_log()

    def _reset_native_history(self, reason):
        _ase_lbfgs_reset_history(self, self._sm_memory)
        self._sm_reset_count += 1
        self._sm_last_reset_reason = str(reason)

    def _native_metrics(self):
        guard = self.sm_guard_metrics()
        return {
            "history_size": _ase_lbfgs_history_size(self),
            "pairs_accepted_total": _ase_lbfgs_pairs_total(self),
            "pairs_rejected_total": int(guard["pairs_damped"]),
            "pairs_damped_total": int(guard["pairs_damped"]),
            "worst_sy": guard["worst_sy"],
            "reset_count": self._sm_reset_count,
            "last_reset_reason": self._sm_last_reset_reason,
        }

    def step(self, forces=None):
        if forces is None:
            forces = self.dimeratoms.get_forces()
        forces = np.asarray(forces, dtype=float)
        # Covers both the kappa gamma regime and ASE's curvature-sign branch.
        state_key = _translation_state_key(self.dimeratoms)
        if (
            self._sm_state_key is not None
            and state_key != self._sm_state_key
            and self.reset_on_regime_change
        ):
            self._reset_native_history(
                f"translation_state:{self._sm_state_key}->{state_key}"
            )
        self._sm_state_key = state_key

        data = self._start_step_diagnostics(forces, "ase_lbfgs")
        before = np.asarray(self.optimizable.get_x(), dtype=float).reshape(-1).copy()
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=r"Please do not pass forces to step\(\)\..*",
                category=UserWarning,
                module=r"ase\.optimize\.optimize",
            )
            _CurvatureGuardedLBFGS.step(self, forces=forces)
        after = np.asarray(self.optimizable.get_x(), dtype=float).reshape(-1)
        displacement = after - before
        step_metric = float(norm(displacement))
        raw_direction = np.asarray(self.p, dtype=float).reshape(-1)
        raw_metric = float(self.optimizable.gradient_norm(raw_direction))
        data.update(
            {
                "direction_alignment": _cosine_alignment(raw_direction, forces),
                "step_clipped": int(raw_metric >= self.maxstep),
            }
        )
        self._last_step_size = step_metric
        self._finish_step_diagnostics(
            data, step_metric, lbfgs_metrics=self._native_metrics()
        )
