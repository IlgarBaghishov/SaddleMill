"""Kappa dimer with independently selectable ASE or L-BFGS rotations."""

import numpy as np
from ase.mep.dimer import (
    DimerEigenmodeSearch,
    MinModeAtoms,
    perpendicular_vector,
    parallel_vector,
    DimerControl,
)

from saddlemill.dimertools.lbfgs_dimer import (
    LBFGSRotationMixin,
    LBFGSDimerEigenmodeSearch,
)

norm = np.linalg.norm


class IsolatedDimerControl(DimerControl):
    """DimerControl with an immutable baseline copied per instance.

    ASE stores ``parameters`` as a mutable class dictionary and applies
    constructor overrides through ``self.parameters``.  Copying the baseline
    before ``DimerControl.__init__`` prevents one control object's overrides
    from leaking into later controls in the same worker process.
    """

    _baseline_parameters = dict(DimerControl.parameters)

    def __init__(self, *args, **kwargs):
        self.parameters = dict(self._baseline_parameters)
        super().__init__(*args, **kwargs)


class KappaEigenmodeSearch(DimerEigenmodeSearch):
    """Phase-B rotation constrained to the isopotential hyperplane."""

    def get_rotational_force(self):
        rot_force = super().get_rotational_force()
        true_forces = self.dimeratoms.forces0
        fnorm = norm(true_forces)
        if fnorm < 1.0e-8:
            return rot_force
        f_hat = true_forces / fnorm
        return perpendicular_vector(rot_force, f_hat)

    def log(self, f_rot_A, angle):
        if self.logfile is not None:
            if angle:
                line = "DIM:ROT: %7d %9d %9.4f %9.4f %9.4f\n" % (
                    self.control.get_counter("optcount"),
                    self.control.get_counter("rotcount"),
                    self.get_curvature(),
                    np.degrees(angle),
                    norm(f_rot_A),
                )
            else:
                line = "DIM:ROT: %7d %9d %9.4f %9s %9.4f\n" % (
                    self.control.get_counter("optcount"),
                    self.control.get_counter("rotcount"),
                    self.get_curvature(),
                    "---------",
                    norm(f_rot_A),
                )
            self.logfile.write(line)

    def update_eigenmode(self, eigenmode):
        fnorm = norm(self.dimeratoms.forces0)
        if fnorm > 1.0e-8:
            f_hat = self.dimeratoms.forces0 / fnorm
            eigenmode = perpendicular_vector(eigenmode, f_hat)
            eigenmode_norm = norm(eigenmode)
            if eigenmode_norm > 1.0e-14:
                eigenmode = eigenmode / eigenmode_norm
        self.eigenmode = eigenmode
        self.update_virtual_positions()
        self.control.increment_counter("rotcount")


class LBFGSKappaEigenmodeSearch(LBFGSRotationMixin, KappaEigenmodeSearch):
    """L-BFGS direction optimizer for the constrained Phase-B rotation."""


class KappaMinModeAtoms(MinModeAtoms):
    """Phase-A/Phase-B kappa dimer with configurable rotation solver."""

    def __init__(
        self,
        atoms,
        beta=2.0,
        recover_fmax=0.3,
        kappa_control=None,
        rotation_optimizer="ase",
        rotation_lbfgs_options=None,
        **kwargs,
    ):
        super().__init__(atoms, **kwargs)
        self.beta = float(beta)
        self.kappa = 0.0
        self.recover_fmax = float(recover_fmax)
        self.rotation_optimizer = str(rotation_optimizer).lower()
        self.rotation_lbfgs_options = dict(rotation_lbfgs_options or {})
        if self.rotation_optimizer not in {"ase", "lbfgs"}:
            raise ValueError(
                "rotation_optimizer must be 'ase' or 'lbfgs'; got "
                f"{rotation_optimizer!r}"
            )

        if kappa_control is not None:
            self.kappa_control = kappa_control
        else:
            self.kappa_control = IsolatedDimerControl(
                dimer_separation=self.control.get_parameter("dimer_separation"),
                f_rot_min=0.01,
                f_rot_max=2.0,
                max_num_rot=4,
                logfile=self.control.logfile,
                eigenmode_logfile=self.control.logfile,
            )
        self.kappa_mode = None
        self.kappa_active = True
        self.translation_regime = "kappa"
        self.last_rotation_diagnostics = {}
        # Gamma scaling actually applied at the last CENTER evaluation. Exposed
        # so the translator can report how strongly the projected force is
        # being attenuated relative to the real force.
        self.last_gamma_1 = 1.0
        self.last_gamma_2 = 1.0

    @staticmethod
    def _ase_rotation_diagnostics(control):
        return {
            "optimizer": "ase",
            "rotations": int(control.get_counter("rotcount")),
            "history_size": 0,
            "pairs_accepted": 0,
            "pairs_rejected": 0,
            "history_resets": 0,
            "direction_fallbacks": 0,
        }

    def _run_search(self, search_class, control, eigenmode, **kwargs):
        if self.rotation_optimizer == "lbfgs":
            kwargs["lbfgs_options"] = self.rotation_lbfgs_options
        search = search_class(
            self,
            control,
            eigenmode=eigenmode,
            **kwargs,
        )
        search.converge_to_eigenmode()
        diagnostics = (
            dict(search.lbfgs_diagnostics)
            if self.rotation_optimizer == "lbfgs"
            else self._ase_rotation_diagnostics(control)
        )
        return search, diagnostics

    def find_eigenmodes(self, order=1):
        if order > 1:
            raise NotImplementedError("Kappa dimer only supports first-order saddles")

        phase_a_class = (
            LBFGSDimerEigenmodeSearch
            if self.rotation_optimizer == "lbfgs"
            else DimerEigenmodeSearch
        )
        search_A, phase_a_diag = self._run_search(
            phase_a_class,
            self.control,
            self.eigenmodes[0],
        )
        search_A.set_up_for_optimization_step()
        eigenmode = search_A.get_eigenmode()
        curvature_A = search_A.get_curvature()
        self.eigenmodes[0] = eigenmode
        self.curvatures[0] = curvature_A

        # Phase B is unused in the stock-dimer positive-curvature branch.
        if curvature_A > 0.0:
            self.kappa = 0.0
            self.kappa_active = False
            self.translation_regime = "standard"
            self.last_rotation_diagnostics = {"phase_a": phase_a_diag}
            return

        # recover_fmax is a center-geometry switch back to normal dimer.
        # Decide it once here so rotation and translation use the same regime
        # throughout this optimization step, including trial positions.
        self.kappa_active = self.real_fmax() >= self.recover_fmax
        if not self.kappa_active:
            self.kappa = 0.0
            self.translation_regime = "standard"
            self.last_rotation_diagnostics = {"phase_a": phase_a_diag}
            return

        self.translation_regime = "kappa"
        true_forces = self.forces0
        force_norm = norm(true_forces)
        f_hat = true_forces / force_norm if force_norm > 1.0e-8 else None

        def fresh_guess():
            if f_hat is None:
                return eigenmode.copy()
            guess = perpendicular_vector(eigenmode, f_hat)
            if norm(guess) > 1.0e-8:
                return guess / norm(guess)
            dummy = np.random.randn(*eigenmode.shape)
            guess = perpendicular_vector(dummy, f_hat)
            return guess / norm(guess)

        if self.kappa_mode is not None and f_hat is not None:
            guess = perpendicular_vector(self.kappa_mode, f_hat)
            guess = guess / norm(guess) if norm(guess) > 1.0e-8 else fresh_guess()
        else:
            guess = fresh_guess()

        phase_b_class = (
            LBFGSKappaEigenmodeSearch
            if self.rotation_optimizer == "lbfgs"
            else KappaEigenmodeSearch
        )
        search_B, phase_b_diag = self._run_search(
            phase_b_class,
            self.kappa_control,
            guess,
        )
        self.kappa_mode = search_B.get_eigenmode().copy()
        curvature_kappa = search_B.get_curvature()

        true_forces = self.forces0
        force_norm = norm(true_forces)
        self.kappa = -(curvature_kappa / force_norm) if force_norm > 1.0e-8 else 0.0
        self.last_rotation_diagnostics = {
            "phase_a": phase_a_diag,
            "phase_b": phase_b_diag,
        }

    def real_fmax(self):
        """True atomic fmax at the current center, with no gamma attenuation.

        ``get_projected_forces`` scales the real force by gamma_1/gamma_2, and
        the optimizer's convergence test reads that scaled force. When
        gamma_2 -> 0 the perpendicular component is discarded entirely, so a
        geometry whose force is largely perpendicular to the mode can register
        as converged while the real force is still large. Callers use this to
        gate convergence on the physical quantity.
        """
        forces = np.asarray(self.forces0, dtype=float)
        if forces.size == 0:
            return 0.0
        return float(np.sqrt((forces * forces).sum(axis=1).max()))

    def get_projected_forces(self, pos=None):
        if pos is not None:
            forces = self.get_forces(real=True, pos=pos).copy()
        else:
            forces = self.forces0.copy()

        eigenmode = self.eigenmodes[0]
        f_parallel = parallel_vector(forces, eigenmode)
        f_perp = forces - f_parallel

        # Stock dimer behavior above the inflection point: drag directly uphill
        # along the current mode. Phase B and kappa do not enter this force.
        if self.curvatures[0] > 0.0:
            if pos is None:
                self.translation_regime = "standard"
                self.last_gamma_1 = 1.0
                self.last_gamma_2 = 0.0
            return -f_parallel

        # kappa_active was set from the center geometry in find_eigenmodes().
        # Do not switch regimes on optimizer trial positions.
        if not self.kappa_active:
            gamma_1 = 1.0
            gamma_2 = 1.0
            regime = "standard"
        else:
            bk = np.clip(self.beta * self.kappa, -500.0, 500.0)
            exp_term = np.exp(bk)
            gamma_1 = (2.0 / (1.0 + exp_term)) - 1.0
            gamma_2 = 1.0 - (1.0 / (1.0 + exp_term))
            regime = "kappa"

        # Trial-position evaluations must not mutate the accepted center state.
        if pos is None:
            self.translation_regime = regime
            self.last_gamma_1 = float(gamma_1)
            self.last_gamma_2 = float(gamma_2)

        return -(gamma_1 * f_parallel) + (gamma_2 * f_perp)

    def eigenmode_log(self):
        if self.mlogfile is not None:
            line = "MINMODE:MODE: Optimization Step: %i\n" % (
                self.control.get_counter("optcount")
            )
            line += "MINMODE:KAPPA: %15.8f\n" % self.kappa
            for mode_number, mode in enumerate(self.eigenmodes):
                line += "MINMODE:MODE: Order: %i\n" % mode_number
                for atom_index in range(len(mode)):
                    line += "MINMODE:MODE: %7i %15.8f %15.8f %15.8f\n" % (
                        atom_index,
                        mode[atom_index][0],
                        mode[atom_index][1],
                        mode[atom_index][2],
                    )
            self.mlogfile.write(line)
            self.mlogfile.flush()
