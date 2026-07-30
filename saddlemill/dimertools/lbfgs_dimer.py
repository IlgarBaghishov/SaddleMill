"""Cartesian L-BFGS solvers for ASE dimer rotation and translation.

The rotation implementation follows the algorithmic division used by
Kastner and Sherwood (JCP 128, 014106, 2008): L-BFGS chooses the rotational
search direction, while ASE's existing trial-angle/Fourier interpolation
chooses the rotation angle.  A new rotation-search object is constructed at
every translated geometry, so its L-BFGS history is reset after translation.

The translation implementation keeps a separate L-BFGS history across
accepted translation steps.  It uses no line search or disposable trial
translation force evaluation.  The resulting step is globally capped by
DimerControl.maximum_translation, preserving the existing ASE dimer step-size
meaning.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from math import atan, cos, pi, sin, tan
from typing import Callable, Optional

import numpy as np

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
    if len(force) == 0:
        return 0.0
    return float(np.sqrt((force * force).sum(axis=1).max()))


def _global_step_clip(step, maximum_translation):
    step = np.asarray(step, dtype=float)
    step_norm = float(norm(step))
    if not np.isfinite(step_norm):
        raise RuntimeError("Non-finite L-BFGS translation step")
    if step_norm > maximum_translation > 0.0:
        step = step * (maximum_translation / step_norm)
        step_norm = float(maximum_translation)
    return step, step_norm


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

    def _finish_step_diagnostics(self, data, step_norm, history=None):
        after_calls = _force_calls(self.dimeratoms)
        entry_calls = data["force_calls_step_entry"]
        data.update(
            {
                "step_norm": float(step_norm),
                "force_calls_translation_trial": (
                    after_calls - entry_calls
                    if after_calls >= 0 and entry_calls >= 0
                    else ""
                ),
                "force_calls_cumulative_after_step": after_calls,
                "translation_lbfgs_history_size": history.size if history else 0,
                "translation_lbfgs_pairs_accepted_total": (
                    history.accepted_pairs_total if history else 0
                ),
                "translation_lbfgs_pairs_rejected_total": (
                    history.rejected_pairs_total if history else 0
                ),
                "translation_lbfgs_resets": history.reset_count if history else 0,
                "translation_lbfgs_last_reset_reason": (
                    history.last_reset_reason if history else ""
                ),
            }
        )
        self._diagnostic_serial += 1
        self.last_step_diagnostics = data
        self._previous_force_calls_after_step = max(after_calls, 0)


class DiagnosticMinModeTranslate(_TranslationDiagnosticsMixin, MinModeTranslate):
    """Stock ASE MinModeTranslate with force-call diagnostics only."""

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
        self._finish_step_diagnostics(data, step_norm, history=None)


class LBFGSTranslationModel:
    """Persistent secant history shared by pure and hybrid translators.

    ``observe`` may be called while FIRE is taking the actual steps.  The
    resulting pairs are the same physical ``s`` and ``y`` observations that
    L-BFGS would have collected itself, so the live history can be used
    immediately when the hybrid enters its L-BFGS state.
    """

    def __init__(
        self,
        memory=10,
        initial_hessian=1.0,
        dynamic_h0=False,
        curvature_epsilon=1.0e-12,
        damping=1.0,
        reset_on_regime_change=True,
    ):
        if float(damping) <= 0.0:
            raise ValueError("translation damping must be > 0")
        self.history = LimitedMemoryInverseHessian(
            memory=memory,
            initial_hessian=initial_hessian,
            dynamic_h0=dynamic_h0,
            curvature_epsilon=curvature_epsilon,
        )
        self.damping = float(damping)
        self.reset_on_regime_change = bool(reset_on_regime_change)
        self.previous_position = None
        self.previous_force = None
        self.regime = None
        self.last_regime_changed = False
        self.last_direction_alignment = ""
        self.last_step_clipped = False

    def reset(self, reason):
        self.history.reset(reason)
        self.previous_position = None
        self.previous_force = None
        self.last_direction_alignment = ""
        self.last_step_clipped = False

    def _set_regime(self, regime):
        regime = str(regime)
        changed = self.regime is not None and regime != self.regime
        self.last_regime_changed = bool(changed)
        if changed and self.reset_on_regime_change:
            old = self.regime
            self.reset(f"translation_regime:{old}->{regime}")
        self.regime = regime
        return changed

    def observe(self, positions, force, regime="standard"):
        """Add the current accepted state to the secant history.

        The optimizer that generated the preceding displacement is irrelevant:
        FIRE-, CG-, and L-BFGS-generated accepted steps all provide a valid
        candidate secant observation when the force definition is unchanged.
        """
        self._set_regime(regime)
        rflat = np.asarray(positions, dtype=float).reshape(-1)
        fflat = np.asarray(force, dtype=float).reshape(-1)
        added = False
        if self.previous_position is not None:
            s = rflat - self.previous_position
            if norm(s) > 1.0e-15:
                # y = gradient_new - gradient_old = force_old - force_new.
                y = self.previous_force - fflat
                added = self.history.add_pair(s, y)
        self.previous_position = rflat.copy()
        self.previous_force = fflat.copy()
        return added

    def direction(self, force):
        f = np.asarray(force, dtype=float)
        direction = self.history.apply(f.reshape(-1)).reshape(f.shape)
        denom = float(norm(direction) * norm(f))
        self.last_direction_alignment = (
            float(np.vdot(direction, f).real / denom) if denom > 0.0 else -1.0
        )
        valid = (
            np.all(np.isfinite(direction))
            and norm(direction) >= 1.0e-14
            and np.vdot(direction, f).real > 0.0
        )
        return direction, bool(valid)

    def compute_step(
        self,
        positions,
        force,
        maximum_translation,
        regime="standard",
        observe=True,
        fallback_to_force=True,
    ):
        r = np.asarray(positions, dtype=float)
        f = np.asarray(force, dtype=float)
        if observe:
            self.observe(r, f, regime=regime)
        else:
            self._set_regime(regime)

        direction, valid = self.direction(f)
        if not valid:
            if not fallback_to_force:
                return None, 0.0, False
            self.reset("non_descent_translation_direction")
            self._set_regime(regime)
            self.observe(r, f, regime=regime)
            direction = (1.0 / self.history.initial_hessian) * f

        unscaled = self.damping * direction
        unclipped_norm = float(norm(unscaled))
        step, step_norm = _global_step_clip(unscaled, maximum_translation)
        self.last_step_clipped = bool(step_norm + 1.0e-15 < unclipped_norm)
        return step, step_norm, valid


class LBFGSMinModeTranslate(_TranslationDiagnosticsMixin, MinModeTranslate):
    """No-line-search L-BFGS translation for a MinModeAtoms object."""

    def __init__(
        self,
        dimeratoms,
        logfile="-",
        trajectory=None,
        lbfgs_options=None,
    ):
        super().__init__(dimeratoms, logfile=logfile, trajectory=trajectory)
        self._initialize_step_diagnostics()
        self.lbfgs_model = LBFGSTranslationModel(**dict(lbfgs_options or {}))

    def step(self, f=None):
        if f is None:
            f = self.dimeratoms.get_forces()
        r = self.dimeratoms.get_positions().copy()
        data = self._start_step_diagnostics(f, "lbfgs")
        step, step_norm, _ = self.lbfgs_model.compute_step(
            r,
            f,
            maximum_translation=self.max_step,
            regime=getattr(self.dimeratoms, "translation_regime", "standard"),
        )
        data.update({
            "direction_alignment": self.lbfgs_model.last_direction_alignment,
            "step_clipped": int(self.lbfgs_model.last_step_clipped),
        })
        self.log(f, step_norm)
        self.dimeratoms.set_positions(r + step)
        self.f0 = np.asarray(f).flat.copy()
        self.r0 = r.flat.copy()
        self._finish_step_diagnostics(
            data, step_norm, history=self.lbfgs_model.history
        )


class DimerFIREState:
    """FIRE dynamical state acting on the projected dimer force."""

    def __init__(
        self,
        dt=0.1,
        dtmax=1.0,
        Nmin=5,
        finc=1.1,
        fdec=0.5,
        astart=0.1,
        fa=0.99,
    ):
        self.dt_initial = float(dt)
        self.dt = float(dt)
        self.dtmax = float(dtmax)
        self.Nmin = int(Nmin)
        self.finc = float(finc)
        self.fdec = float(fdec)
        self.astart = float(astart)
        self.a = float(astart)
        self.fa = float(fa)
        self.Nsteps = 0
        self.velocity = None

    def reset(self):
        self.dt = self.dt_initial
        self.a = self.astart
        self.Nsteps = 0
        self.velocity = None

    def compute_step(self, force, maximum_translation):
        force = np.asarray(force, dtype=float)
        flat = force.reshape(-1)
        if self.velocity is None:
            self.velocity = np.zeros_like(flat)
        else:
            vf = float(np.dot(flat, self.velocity))
            f2 = float(np.dot(flat, flat))
            if vf > 0.0 and f2 > 0.0:
                v2 = float(np.dot(self.velocity, self.velocity))
                self.velocity = (
                    (1.0 - self.a) * self.velocity
                    + self.a * flat / np.sqrt(f2) * np.sqrt(v2)
                )
                if self.Nsteps > self.Nmin:
                    self.dt = min(self.dt * self.finc, self.dtmax)
                    self.a *= self.fa
                self.Nsteps += 1
            else:
                self.velocity[:] = 0.0
                self.a = self.astart
                self.dt *= self.fdec
                self.Nsteps = 0
        self.velocity += self.dt * flat
        raw_step = (self.dt * self.velocity).reshape(force.shape)
        raw_norm = float(norm(raw_step))
        step, step_norm = _global_step_clip(raw_step, maximum_translation)
        return step, step_norm, bool(step_norm + 1.0e-15 < raw_norm)


@dataclass
class HybridDecision:
    state: str
    switch_event: str = ""
    history_pairs_at_switch: int = 0


class HybridDimerStateController:
    """FIRE -> L-BFGS controller with fmax/curvature hysteresis."""

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
            raise ValueError(
                "hybrid exit_curvature must be >= enter_curvature"
            )
        self.state = "fire"
        self._enter_count = 0
        self._exit_count = 0

    def force_fire(self, event):
        self.state = "fire"
        self._enter_count = 0
        self._exit_count = 0
        return HybridDecision("fire", str(event), 0)

    def update(self, fmax, curvature, history_size) -> HybridDecision:
        if not self.enabled:
            return HybridDecision("fire", "", 0)

        fmax = float(fmax)
        curvature = float(curvature)
        history_ready = (
            not self.warm_start_history
            or int(history_size) >= self.minimum_history_pairs
        )
        if self.state == "fire":
            enter = (
                fmax <= self.enter_fmax
                and curvature <= self.enter_curvature
                and history_ready
            )
            self._enter_count = self._enter_count + 1 if enter else 0
            self._exit_count = 0
            if self._enter_count >= self.enter_stable_steps:
                self.state = "lbfgs"
                self._enter_count = 0
                return HybridDecision(
                    "lbfgs", "fire_to_lbfgs", int(history_size)
                )
        else:
            exit_now = (
                fmax >= self.exit_fmax
                or curvature >= self.exit_curvature
            )
            self._exit_count = self._exit_count + 1 if exit_now else 0
            self._enter_count = 0
            if self._exit_count >= self.exit_stable_steps:
                self.state = "fire"
                self._exit_count = 0
                return HybridDecision(
                    "fire", "lbfgs_to_fire_threshold", int(history_size)
                )
        return HybridDecision(self.state, "", 0)


class HybridMinModeTranslate(_TranslationDiagnosticsMixin, MinModeTranslate):
    """Warm-started FIRE/L-BFGS dimer translation.

    FIRE takes the early accepted translations.  When
    ``warm_start_history=True``, those same accepted positions and projected
    forces populate the live L-BFGS history.  Switching therefore does not
    require passing a restart object or replaying a trajectory.
    """

    def __init__(
        self,
        dimeratoms,
        logfile="-",
        trajectory=None,
        lbfgs_options=None,
        hybrid_options=None,
    ):
        super().__init__(dimeratoms, logfile=logfile, trajectory=trajectory)
        self._initialize_step_diagnostics()
        self.lbfgs_model = LBFGSTranslationModel(**dict(lbfgs_options or {}))
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
        fire_options = {
            target: options.pop(source)
            for source, target in fire_keys.items()
            if source in options
        }
        self.reset_history_on_exit = bool(
            options.pop("reset_history_on_exit", True)
        )
        self.fire_state = DimerFIREState(**fire_options)
        self.hybrid_controller = HybridDimerStateController(**options)
        self.warm_start_history = self.hybrid_controller.warm_start_history

    def step(self, f=None):
        if f is None:
            f = self.dimeratoms.get_forces()
        f = np.asarray(f, dtype=float)
        r = self.dimeratoms.get_positions().copy()
        regime = getattr(self.dimeratoms, "translation_regime", "standard")
        fmax = _projected_fmax(f)
        curvature = float(self.dimeratoms.get_curvature())
        state_before = self.hybrid_controller.state

        # The current state closes the secant pair generated by the previous
        # accepted step.  During FIRE this is the shadow warm-start update.
        observed = False
        if state_before == "lbfgs" or self.warm_start_history:
            self.lbfgs_model.observe(r, f, regime=regime)
            observed = True

        if state_before == "lbfgs" and self.lbfgs_model.last_regime_changed:
            decision = self.hybrid_controller.force_fire(
                "lbfgs_to_fire_regime_change"
            )
            self.fire_state.reset()
        else:
            decision = self.hybrid_controller.update(
                fmax, curvature, self.lbfgs_model.history.size
            )
        if decision.switch_event == "fire_to_lbfgs" and not self.warm_start_history:
            self.lbfgs_model.reset("cold_start_fire_to_lbfgs")
            self.lbfgs_model.observe(r, f, regime=regime)
            observed = True
        elif decision.switch_event == "lbfgs_to_fire_threshold":
            self.fire_state.reset()
            if self.reset_history_on_exit:
                self.lbfgs_model.reset("lbfgs_to_fire_threshold")
                if self.warm_start_history:
                    self.lbfgs_model.observe(r, f, regime=regime)
                    observed = True

        if decision.state == "fire":
            data = self._start_step_diagnostics(
                f,
                "fire",
                hybrid_state="fire",
                switch_event=decision.switch_event,
            )
            step, step_norm, clipped = self.fire_state.compute_step(
                f, self.max_step
            )
            data.update({
                "direction_alignment": "",
                "step_clipped": int(clipped),
                "hybrid_history_pairs_at_switch": int(
                    decision.history_pairs_at_switch
                ),
                "hybrid_warm_start_history": int(self.warm_start_history),
            })
        else:
            data = self._start_step_diagnostics(
                f,
                "lbfgs",
                hybrid_state="lbfgs",
                switch_event=decision.switch_event,
            )
            step, step_norm, valid = self.lbfgs_model.compute_step(
                r,
                f,
                maximum_translation=self.max_step,
                regime=regime,
                observe=not observed,
                fallback_to_force=False,
            )
            if not valid or step is None:
                invalid_alignment = self.lbfgs_model.last_direction_alignment
                self.lbfgs_model.reset("non_descent_translation_direction")
                decision = self.hybrid_controller.force_fire(
                    "lbfgs_to_fire_non_descent"
                )
                self.fire_state.reset()
                if self.warm_start_history:
                    self.lbfgs_model.observe(r, f, regime=regime)
                step, step_norm, clipped = self.fire_state.compute_step(
                    f, self.max_step
                )
                data["translation_algorithm"] = "fire"
                data["hybrid_state"] = "fire"
                data["hybrid_switch_event"] = decision.switch_event
                data.update({
                    "direction_alignment": invalid_alignment,
                    "step_clipped": int(clipped),
                })
            else:
                data.update({
                    "direction_alignment": self.lbfgs_model.last_direction_alignment,
                    "step_clipped": int(self.lbfgs_model.last_step_clipped),
                })
            data.update({
                "hybrid_history_pairs_at_switch": int(
                    decision.history_pairs_at_switch
                ),
                "hybrid_warm_start_history": int(self.warm_start_history),
            })

        self.log(f, step_norm)
        self.dimeratoms.set_positions(r + step)
        self.f0 = f.flat.copy()
        self.r0 = r.flat.copy()
        self._finish_step_diagnostics(
            data, step_norm, history=self.lbfgs_model.history
        )
