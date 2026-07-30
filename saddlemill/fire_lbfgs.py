"""Warm-started ASE FIRE to ASE L-BFGS optimizer.

Both algorithms are delegated to ASE.  SaddleMill only owns the hysteretic
switch controller, accepted-state snapshot buffer, and diagnostics.  A warm
FIRE -> L-BFGS switch rebuilds ASE's native L-BFGS state in generalized
coordinates, so ordinary Atoms and filters such as FrechetCellFilter use the
same path.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import warnings

import numpy as np
from ase.optimize import FIRE
from ase.optimize.optimize import Optimizer

from saddlemill.dimertools.lbfgs_dimer import _ASELBFGSState


@dataclass
class HybridDecision:
    state: str
    switch_event: str = ""
    history_pairs_at_switch: int = 0


class ForceThresholdController:
    """Two-state FIRE/L-BFGS controller with fmax hysteresis."""

    def __init__(
        self,
        enter_fmax=0.20,
        exit_fmax=0.35,
        enter_stable_steps=3,
        exit_stable_steps=2,
        minimum_history_pairs=3,
        warm_start_history=True,
    ):
        self.enter_fmax = float(enter_fmax)
        self.exit_fmax = float(exit_fmax)
        self.enter_stable_steps = int(enter_stable_steps)
        self.exit_stable_steps = int(exit_stable_steps)
        self.minimum_history_pairs = int(minimum_history_pairs)
        self.warm_start_history = bool(warm_start_history)
        if self.enter_fmax <= 0.0 or self.exit_fmax <= 0.0:
            raise ValueError("FIRELBFGS force thresholds must be > 0")
        if self.exit_fmax < self.enter_fmax:
            raise ValueError("FIRELBFGS exit_fmax must be >= enter_fmax")
        if self.enter_stable_steps < 1 or self.exit_stable_steps < 1:
            raise ValueError("FIRELBFGS stable-step counts must be >= 1")
        if self.minimum_history_pairs < 0:
            raise ValueError("FIRELBFGS minimum_history_pairs must be >= 0")
        self.state = "fire"
        self._enter_count = 0
        self._exit_count = 0

    def force_fire(self, event):
        self.state = "fire"
        self._enter_count = 0
        self._exit_count = 0
        return HybridDecision("fire", str(event), 0)

    def update(self, fmax, history_pairs):
        history_pairs = int(history_pairs)
        if self.state == "fire":
            history_ready = (
                not self.warm_start_history
                or history_pairs >= self.minimum_history_pairs
            )
            enter = float(fmax) <= self.enter_fmax and history_ready
            self._enter_count = self._enter_count + 1 if enter else 0
            self._exit_count = 0
            if self._enter_count >= self.enter_stable_steps:
                self.state = "lbfgs"
                self._enter_count = 0
                return HybridDecision(
                    "lbfgs", "fire_to_lbfgs", history_pairs
                )
        else:
            exit_now = float(fmax) >= self.exit_fmax
            self._exit_count = self._exit_count + 1 if exit_now else 0
            self._enter_count = 0
            if self._exit_count >= self.exit_stable_steps:
                self.state = "fire"
                self._exit_count = 0
                return HybridDecision(
                    "fire", "lbfgs_to_fire_threshold", history_pairs
                )
        return HybridDecision(self.state, "", 0)


class FIRELBFGS(Optimizer):
    """ASE FIRE warm-up followed by no-line-search ASE L-BFGS."""

    def __init__(
        self,
        atoms,
        restart=None,
        logfile="-",
        trajectory=None,
        maxstep=None,
        fire_dt=0.1,
        fire_dtmax=1.0,
        fire_Nmin=5,
        fire_finc=1.1,
        fire_fdec=0.5,
        fire_astart=0.1,
        fire_fa=0.99,
        lbfgs_memory=10,
        lbfgs_initial_hessian=70.0,
        lbfgs_dynamic_h0=False,
        lbfgs_curvature_epsilon=1.0e-12,
        lbfgs_damping=1.0,
        enter_fmax=0.20,
        exit_fmax=0.35,
        enter_stable_steps=3,
        exit_stable_steps=2,
        minimum_history_pairs=3,
        warm_start_history=True,
        reset_history_on_exit=True,
        **kwargs,
    ):
        if restart is not None:
            raise NotImplementedError(
                "FIRELBFGS restart files are not implemented; use "
                "SaddleMill continuation from the saved structure instead."
            )
        if float(lbfgs_damping) <= 0.0:
            raise ValueError("FIRELBFGS lbfgs_damping must be > 0")
        # Retained for config compatibility. ASE's LBFGS uses a fixed H0=1/alpha
        # and stores all update pairs; these custom-recursion options are inert.
        self.lbfgs_dynamic_h0 = bool(lbfgs_dynamic_h0)
        self.lbfgs_curvature_epsilon = float(lbfgs_curvature_epsilon)

        Optimizer.__init__(
            self,
            atoms,
            restart=None,
            logfile=logfile,
            trajectory=trajectory,
            **kwargs,
        )
        self.maxstep = (
            float(maxstep) if maxstep is not None else float(self.defaults["maxstep"])
        )
        self.fire_options = {
            "dt": float(fire_dt),
            "dtmax": float(fire_dtmax),
            "Nmin": int(fire_Nmin),
            "finc": float(fire_finc),
            "fdec": float(fire_fdec),
            "astart": float(fire_astart),
            "fa": float(fire_fa),
        }
        self.lbfgs_memory = int(lbfgs_memory)
        self.lbfgs_alpha = float(lbfgs_initial_hessian)
        self.lbfgs_damping = float(lbfgs_damping)
        self.warm_start_history = bool(warm_start_history)
        self.reset_history_on_exit = bool(reset_history_on_exit)
        self.controller = ForceThresholdController(
            enter_fmax=enter_fmax,
            exit_fmax=exit_fmax,
            enter_stable_steps=enter_stable_steps,
            exit_stable_steps=exit_stable_steps,
            minimum_history_pairs=minimum_history_pairs,
            warm_start_history=warm_start_history,
        )
        self.fire_optimizer = self._new_fire_optimizer()
        self.lbfgs_state = _ASELBFGSState(
            atoms,
            maxstep=self.maxstep,
            memory=self.lbfgs_memory,
            damping=self.lbfgs_damping,
            alpha=self.lbfgs_alpha,
        )
        self.snapshots = deque(maxlen=self.lbfgs_memory + 1)
        self.last_step_diagnostics = None
        self._diagnostic_serial = 0
        self.switch_count = 0

    def initialize(self):
        # Optimizer.__init__ calls this before the internal optimizers exist.
        pass

    def read(self):
        raise NotImplementedError("FIRELBFGS restart files are not implemented")

    def _new_fire_optimizer(self):
        return FIRE(
            self.atoms,
            restart=None,
            logfile=None,
            trajectory=None,
            maxstep=self.maxstep,
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

    def step(self, forces=None):
        force = -self._get_gradient(forces)
        force = np.asarray(force, dtype=float).reshape(-1)
        position = np.asarray(self.optimizable.get_x(), dtype=float).reshape(-1)
        fmax = float(self.optimizable.gradient_norm(force))
        self._append_snapshot(position, force)
        available_pairs = max(0, len(self.snapshots) - 1)

        decision = self.controller.update(fmax, available_pairs)
        if decision.switch_event:
            self.switch_count += 1

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
                self._append_snapshot(position, force)

        active = decision.state
        alignment = ""
        before = position.copy()
        if active == "lbfgs":
            step_norm, clipped, alignment = self.lbfgs_state.step(force)
        else:
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message=r"Please do not pass forces to step\(\)\..*",
                    category=UserWarning,
                    module=r"ase\.optimize\.optimize",
                )
                self.fire_optimizer.step(f=force)
            after = np.asarray(self.optimizable.get_x(), dtype=float).reshape(-1)
            displacement = after - before
            step_norm = float(np.linalg.norm(displacement))
            # ASE FIRE clips the global generalized-coordinate norm; use its
            # actual displacement and maxstep for the diagnostic flag.
            clipped = bool(np.linalg.norm(displacement) >= self.maxstep)

        metrics = self.lbfgs_state.metrics()
        self._diagnostic_serial += 1
        self.last_step_diagnostics = {
            "diagnostic_serial": self._diagnostic_serial,
            "optimizer_step": int(self.nsteps) + 1,
            "active_optimizer": active,
            "switch_event": decision.switch_event,
            "fmax": fmax,
            "step_norm": float(step_norm),
            "step_clipped": int(bool(clipped)),
            "direction_alignment": alignment,
            "warm_start_history": int(self.warm_start_history),
            "history_pairs_at_switch": int(
                decision.history_pairs_at_switch
            ),
            "lbfgs_history_size": int(metrics["history_size"]),
            "lbfgs_pairs_accepted_total": int(
                metrics["pairs_accepted_total"]
            ),
            "lbfgs_pairs_rejected_total": int(
                metrics["pairs_rejected_total"]
            ),
            "lbfgs_history_resets": int(metrics["reset_count"]),
            "lbfgs_last_reset_reason": metrics["last_reset_reason"],
            "fire_dt": float(self.fire_optimizer.dt),
        }

    def hybrid_summary(self):
        metrics = self.lbfgs_state.metrics()
        return {
            "final_active_optimizer": self.controller.state,
            "switch_count": int(self.switch_count),
            "lbfgs_history_size": int(metrics["history_size"]),
            "lbfgs_pairs_accepted_total": int(
                metrics["pairs_accepted_total"]
            ),
            "lbfgs_pairs_rejected_total": int(
                metrics["pairs_rejected_total"]
            ),
            "lbfgs_history_resets": int(metrics["reset_count"]),
            "lbfgs_last_reset_reason": metrics["last_reset_reason"],
            "warm_start_history": int(self.warm_start_history),
        }
