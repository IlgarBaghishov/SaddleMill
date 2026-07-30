"""Dimer rotation helpers and ASE-backed translation optimizers.

L-BFGS rotation remains a specialized implementation because ASE's ordinary
Cartesian optimizer cannot directly optimize the normalized dimer mode while
retaining ASE's trial-angle/Fourier rotation.

Pure L-BFGS translation now delegates direction generation, secant-history
updates, initial scaling, damping, and max-step handling to ``ase.optimize.LBFGS``
on the ``MinModeAtoms`` wrapper.  ``MinModeAtoms.get_forces()`` supplies the
projected minimum-mode-following force automatically.

The FIRE/L-BFGS hybrid delegates FIRE steps to ``ase.optimize.FIRE`` and
L-BFGS steps to ``ase.optimize.LBFGS``.  Warm starts rebuild ASE's own L-BFGS
state from accepted generalized-coordinate/force snapshots; SaddleMill does
not implement a second L-BFGS two-loop recursion for translation.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from math import atan, cos, pi, sin, tan
from typing import Callable, Optional
import warnings

import numpy as np

from ase.optimize import FIRE, LBFGS

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
            "curvature": float(self.dimeratoms.get_curvature()),
            "translation_regime": getattr(
                self.dimeratoms, "translation_regime", "standard"
            ),
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
                "translation_lbfgs_resets": metrics.get("reset_count", 0),
                "translation_lbfgs_last_reset_reason": metrics.get(
                    "last_reset_reason", ""
                ),
            }
        )
        self._diagnostic_serial += 1
        self.last_step_diagnostics = data
        self._previous_force_calls_after_step = max(after_calls, 0)


class DiagnosticMinModeTranslate(_TranslationDiagnosticsMixin, MinModeTranslate):
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

    def __init__(self, target, *, maxstep, memory, damping, alpha):
        self.target = target
        self.maxstep = float(maxstep)
        self.memory = int(memory)
        self.damping = float(damping)
        self.alpha = float(alpha)
        self.optimizer = self._new_optimizer()
        self.reset_count = 0
        self.last_reset_reason = "initial"
        self.pairs_rejected_total = 0  # ASE 3.29 LBFGS stores every replay/update pair.

    def _new_optimizer(self):
        return LBFGS(
            self.target,
            restart=None,
            logfile=None,
            trajectory=None,
            maxstep=self.maxstep,
            memory=self.memory,
            damping=self.damping,
            alpha=self.alpha,
            use_line_search=False,
        )

    def reset(self, reason="manual"):
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
        return {
            "history_size": self.history_size,
            "pairs_accepted_total": self.pairs_accepted_total,
            "pairs_rejected_total": self.pairs_rejected_total,
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
    _TranslationDiagnosticsMixin, _DimerLBFGSLogMixin, LBFGS
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
        if options:
            raise TypeError(f"Unknown ASE dimer L-BFGS options: {sorted(options)}")
        self._sm_memory = memory
        self._sm_alpha = alpha
        self._sm_damping = damping
        self._sm_regime = None
        self._sm_reset_count = 0
        self._sm_last_reset_reason = "initial"
        LBFGS.__init__(
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
        )
        self._initialize_step_diagnostics()
        self._initialize_dimer_log()

    def _reset_native_history(self, reason):
        _ase_lbfgs_reset_history(self, self._sm_memory)
        self._sm_reset_count += 1
        self._sm_last_reset_reason = str(reason)

    def _native_metrics(self):
        return {
            "history_size": _ase_lbfgs_history_size(self),
            "pairs_accepted_total": _ase_lbfgs_pairs_total(self),
            "pairs_rejected_total": 0,
            "reset_count": self._sm_reset_count,
            "last_reset_reason": self._sm_last_reset_reason,
        }

    def step(self, forces=None):
        if forces is None:
            forces = self.dimeratoms.get_forces()
        forces = np.asarray(forces, dtype=float)
        regime = str(getattr(self.dimeratoms, "translation_regime", "standard"))
        if (
            self._sm_regime is not None
            and regime != self._sm_regime
            and self.reset_on_regime_change
        ):
            self._reset_native_history(
                f"translation_regime:{self._sm_regime}->{regime}"
            )
        self._sm_regime = regime

        data = self._start_step_diagnostics(forces, "ase_lbfgs")
        before = np.asarray(self.optimizable.get_x(), dtype=float).reshape(-1).copy()
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=r"Please do not pass forces to step\(\)\..*",
                category=UserWarning,
                module=r"ase\.optimize\.optimize",
            )
            LBFGS.step(self, forces=forces)
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


@dataclass
class HybridDecision:
    state: str
    switch_event: str = ""
    history_pairs_at_switch: int = 0


class HybridDimerStateController:
    """FIRE -> ASE-LBFGS controller with fmax/curvature hysteresis."""

    def __init__(
        self,
        enabled=False,
        enter_fmax=0.30,
        exit_fmax=0.50,
        enter_curvature=-0.05,
        exit_curvature=0.00,
        enter_stable_steps=3,
        exit_stable_steps=2,
        minimum_history_pairs=3,
        warm_start_history=True,
    ):
        self.enabled = bool(enabled)
        self.enter_fmax = float(enter_fmax)
        self.exit_fmax = float(exit_fmax)
        self.enter_curvature = float(enter_curvature)
        self.exit_curvature = float(exit_curvature)
        self.enter_stable_steps = int(enter_stable_steps)
        self.exit_stable_steps = int(exit_stable_steps)
        self.minimum_history_pairs = int(minimum_history_pairs)
        self.warm_start_history = bool(warm_start_history)
        if self.enter_stable_steps < 1 or self.exit_stable_steps < 1:
            raise ValueError("Hybrid stable-step counts must be >= 1")
        if self.minimum_history_pairs < 0:
            raise ValueError("minimum_history_pairs must be >= 0")
        if self.exit_fmax < self.enter_fmax:
            raise ValueError("hybrid exit_fmax must be >= enter_fmax")
        if self.exit_curvature < self.enter_curvature:
            raise ValueError("hybrid exit_curvature must be >= enter_curvature")
        self.state = "fire"
        self._enter_count = 0
        self._exit_count = 0

    def force_fire(self, event):
        self.state = "fire"
        self._enter_count = 0
        self._exit_count = 0
        return HybridDecision("fire", str(event), 0)

    def update(self, fmax, curvature, history_pairs) -> HybridDecision:
        if not self.enabled:
            return HybridDecision("fire", "", 0)
        history_ready = (
            not self.warm_start_history
            or int(history_pairs) >= self.minimum_history_pairs
        )
        if self.state == "fire":
            enter = (
                float(fmax) <= self.enter_fmax
                and float(curvature) <= self.enter_curvature
                and history_ready
            )
            self._enter_count = self._enter_count + 1 if enter else 0
            self._exit_count = 0
            if self._enter_count >= self.enter_stable_steps:
                self.state = "lbfgs"
                self._enter_count = 0
                return HybridDecision(
                    "lbfgs", "fire_to_lbfgs", int(history_pairs)
                )
        else:
            exit_now = (
                float(fmax) >= self.exit_fmax
                or float(curvature) >= self.exit_curvature
            )
            self._exit_count = self._exit_count + 1 if exit_now else 0
            self._enter_count = 0
            if self._exit_count >= self.exit_stable_steps:
                self.state = "fire"
                self._exit_count = 0
                return HybridDecision(
                    "fire", "lbfgs_to_fire_threshold", int(history_pairs)
                )
        return HybridDecision(self.state, "", 0)


class HybridMinModeTranslate(
    _TranslationDiagnosticsMixin, _DimerLBFGSLogMixin, MinModeTranslate
):
    """ASE FIRE warm-up followed by ASE L-BFGS on ``MinModeAtoms``."""

    def __init__(
        self,
        dimeratoms,
        logfile="-",
        trajectory=None,
        lbfgs_options=None,
        hybrid_options=None,
    ):
        MinModeTranslate.__init__(
            self, dimeratoms, logfile=logfile, trajectory=trajectory
        )
        self._initialize_step_diagnostics()
        self._initialize_dimer_log(write_header=False)

        lbfgs = dict(lbfgs_options or {})
        self.reset_on_regime_change = bool(
            lbfgs.pop("reset_on_regime_change", True)
        )
        lbfgs.pop("dynamic_h0", None)
        lbfgs.pop("curvature_epsilon", None)
        self.lbfgs_memory = int(lbfgs.pop("memory", 10))
        self.lbfgs_alpha = float(lbfgs.pop("initial_hessian", 70.0))
        self.lbfgs_damping = float(lbfgs.pop("damping", 1.0))
        if lbfgs:
            raise TypeError(f"Unknown ASE dimer L-BFGS options: {sorted(lbfgs)}")

        options = dict(hybrid_options or {})
        fire_keys = {
            "fire_dt": "dt",
            "fire_dtmax": "dtmax",
            "fire_Nmin": "Nmin",
            "fire_finc": "finc",
            "fire_fdec": "fdec",
            "fire_astart": "astart",
            "fire_fa": "fa",
        }
        self.fire_options = {
            target: options.pop(source)
            for source, target in fire_keys.items()
            if source in options
        }
        self.reset_history_on_exit = bool(
            options.pop("reset_history_on_exit", True)
        )
        self.controller = HybridDimerStateController(**options)
        self.warm_start_history = self.controller.warm_start_history
        self.snapshots = deque(maxlen=self.lbfgs_memory + 1)
        self.fire_optimizer = self._new_fire_optimizer()
        self.lbfgs_state = _ASELBFGSState(
            dimeratoms,
            maxstep=self.max_step,
            memory=self.lbfgs_memory,
            damping=self.lbfgs_damping,
            alpha=self.lbfgs_alpha,
        )
        self._last_regime = None

    def _new_fire_optimizer(self):
        return FIRE(
            self.dimeratoms,
            restart=None,
            logfile=None,
            trajectory=None,
            maxstep=self.max_step,
            downhill_check=False,
            **self.fire_options,
        )

    def _append_snapshot(self, position, force):
        item = (
            np.asarray(position, dtype=float).reshape(-1).copy(),
            np.asarray(force, dtype=float).reshape(-1).copy(),
        )
        if self.snapshots and np.array_equal(self.snapshots[-1][0], item[0]):
            self.snapshots[-1] = item
        else:
            self.snapshots.append(item)

    def step(self, f=None):
        if f is None:
            f = self.dimeratoms.get_forces()
        f = np.asarray(f, dtype=float)
        position = np.asarray(self.optimizable.get_x(), dtype=float).reshape(-1)
        regime = str(getattr(self.dimeratoms, "translation_regime", "standard"))
        regime_changed = self._last_regime is not None and regime != self._last_regime
        self._last_regime = regime
        if regime_changed and self.reset_on_regime_change:
            self.snapshots.clear()
            self.lbfgs_state.reset(f"translation_regime_change:{regime}")
            self.fire_optimizer = self._new_fire_optimizer()
            if self.controller.state == "lbfgs":
                self.controller.force_fire("lbfgs_to_fire_regime_change")

        self._append_snapshot(position, f)
        available_pairs = max(0, len(self.snapshots) - 1)
        decision = self.controller.update(
            _projected_fmax(f),
            float(self.dimeratoms.get_curvature()),
            available_pairs,
        )

        if decision.switch_event == "fire_to_lbfgs":
            if self.warm_start_history:
                self.lbfgs_state.rebuild(
                    self.snapshots, reason="warm_start_fire_to_lbfgs"
                )
            else:
                self.lbfgs_state.reset("cold_start_fire_to_lbfgs")
        elif decision.switch_event == "lbfgs_to_fire_threshold":
            self.fire_optimizer = self._new_fire_optimizer()
            if self.reset_history_on_exit:
                self.lbfgs_state.reset("lbfgs_to_fire_threshold")
                self.snapshots.clear()
                self._append_snapshot(position, f)

        data = self._start_step_diagnostics(
            f,
            decision.state,
            hybrid_state=decision.state,
            switch_event=decision.switch_event,
        )
        if decision.state == "fire":
            before = position.copy()
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message=r"Please do not pass forces to step\(\)\..*",
                    category=UserWarning,
                    module=r"ase\.optimize\.optimize",
                )
                self.fire_optimizer.step(f=f)
            after = np.asarray(self.optimizable.get_x(), dtype=float).reshape(-1)
            displacement = after - before
            step_metric = float(norm(displacement))
            data.update(
                {
                    "direction_alignment": "",
                    "step_clipped": int(norm(displacement) >= self.max_step),
                }
            )
            metrics = self.lbfgs_state.metrics()
        else:
            step_metric, clipped, alignment = self.lbfgs_state.step(f)
            data.update(
                {
                    "direction_alignment": alignment,
                    "step_clipped": int(clipped),
                }
            )
            metrics = self.lbfgs_state.metrics()

        data.update(
            {
                "hybrid_history_pairs_at_switch": int(
                    decision.history_pairs_at_switch
                ),
                "hybrid_warm_start_history": int(self.warm_start_history),
            }
        )
        self._last_step_size = step_metric
        self._finish_step_diagnostics(data, step_metric, metrics)
