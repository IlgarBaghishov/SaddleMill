import numpy as np
from ase.mep.dimer import DimerEigenmodeSearch, MinModeAtoms, perpendicular_vector, parallel_vector, DimerControl
from saddlemill.dimertools.lbfgs_dimer import (
    LBFGSRotationMixin,
    LBFGSDimerEigenmodeSearch,
)

norm = np.linalg.norm


class IsolatedDimerControl(DimerControl):
    """DimerControl that owns its own parameter dict.

    ASE's DimerControl stores `parameters` as a class-level attribute and never
    copies it per instance, so EVERY DimerControl in the process shares one dict:
    constructing a second control overwrites the rotation parameters
    (max_num_rot, f_rot_max, ...) of every control built earlier. The kappa-dimer
    needs two live controls at once (Phase-A `control` and Phase-B `kappa_control`)
    holding different values, so the shared dict silently collapses them to
    whichever was built last. Snapshotting the defaults onto the instance before
    super().__init__() fills them in keeps each control independent. Stored values
    are scalars, so a shallow copy is sufficient.
    """
    def __init__(self, *args, **kwargs):
        self.parameters = dict(type(self).parameters)
        super().__init__(*args, **kwargs)

class KappaEigenmodeSearch(DimerEigenmodeSearch):
    """
    Phase B Rotation: Constrains the dimer rotation to the isopotential hyperplane.
    It inherits the entire converge_to_eigenmode() loop from ASE, but overrides
    the rotational force calculation.
    """
    def get_rotational_force(self):
        rot_force = super().get_rotational_force()
        
        true_forces = self.dimeratoms.forces0
        
        fnorm = norm(true_forces)
        if fnorm < 1e-8:
            return rot_force
        
        f_hat = true_forces / fnorm 
        constrained_rot_force = perpendicular_vector(rot_force, f_hat)
        return constrained_rot_force
    def log(self, f_rot_A, angle):
        """Log each rotational step."""
        # NYI Log for the trial angle
        if self.logfile is not None:
            if angle:
                l = 'DIM:ROT: %7d %9d %9.4f %9.4f %9.4f\n' % \
                    (self.control.get_counter('optcount'),
                     self.control.get_counter('rotcount'),
                     self.get_curvature(), np.degrees(angle), norm(f_rot_A))
            else:
                l = 'DIM:ROT: %7d %9d %9.4f %9s %9.4f\n' % \
                    (self.control.get_counter('optcount'),
                     self.control.get_counter('rotcount'),
                     self.get_curvature(), '---------', norm(f_rot_A))
            self.logfile.write(l)
    def update_eigenmode(self, eigenmode):
        """Update the eigenmode in the MinModeAtoms object."""
        fnorm = norm(self.dimeratoms.forces0)
        if fnorm > 1e-8:
            f_hat = self.dimeratoms.forces0 / fnorm
            eigenmode = perpendicular_vector(eigenmode, f_hat)
            eigenmode = eigenmode / norm(eigenmode)
        self.eigenmode = eigenmode 
        self.update_virtual_positions()
        self.control.increment_counter('rotcount')


class LBFGSKappaEigenmodeSearch(LBFGSRotationMixin, KappaEigenmodeSearch):
    """L-BFGS direction optimizer for constrained Phase-B rotation."""



class KappaMinModeAtoms(MinModeAtoms):
    """
    Extended MinModeAtoms to handle the Phase A/B double rotation and 
    the kappa-weighted translation forces.
    """
    def __init__(self, atoms, beta=2.0, recover_fmax = 0.3, kappa_control=None,
                 rotation_optimizer="ase", rotation_lbfgs_options=None, **kwargs):
        super().__init__(atoms, **kwargs)
        
        # Tuning parameters for the translation step
        self.beta = beta           # Steepness of the switching function
        self.kappa = 0.0           # Initialize isopotential curvature
        self.recover_fmax = recover_fmax # max atom force norm (like EDIFFG). to determine if to switch back to normal dimer method.
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
            # tighter rotation than Phase A: converge kappa each step.
            # IsolatedDimerControl so building this does NOT clobber self.control's
            # rotation parameters (see IsolatedDimerControl docstring).
            self.kappa_control = IsolatedDimerControl(
               dimer_separation=self.control.get_parameter('dimer_separation'),
                f_rot_min=0.01, f_rot_max=2.0,   # don't bail after one rotation
                max_num_rot=4,
                logfile=self.control.logfile, eigenmode_logfile=self.control.logfile)  
        self.kappa_mode = None
        self.kappa_active = True
        self.translation_regime = "kappa"

    def _run_search(self, search_class, control, eigenmode):
        kwargs = {}
        if self.rotation_optimizer == "lbfgs":
            kwargs["lbfgs_options"] = self.rotation_lbfgs_options
        search = search_class(self, control, eigenmode=eigenmode, **kwargs)
        search.converge_to_eigenmode()
        return search

    def real_fmax(self):
        forces = np.asarray(self.forces0, dtype=float)
        if forces.size == 0:
            return 0.0
        return float(np.sqrt((forces * forces).sum(axis=1).max()))
 
    def find_eigenmodes(self, order=1):
        """
        Launches eigenmode search and kappa search
        Overrides ASE's standard eigenmode search to run Phase A and Phase B.
        """
        if order > 1:
            raise NotImplementedError("Kappa-dimer only supports 1st order saddles.")

        # ---------------------------------------------------------
        # PHASE A: Standard unconstrained rotation to find eigenmode and curvature_A
        # ---------------------------------------------------------
        phase_a_class = (
            LBFGSDimerEigenmodeSearch
            if self.rotation_optimizer == "lbfgs"
            else DimerEigenmodeSearch
        )
        search_A = self._run_search(phase_a_class, self.control, self.eigenmodes[0])
        search_A.set_up_for_optimization_step()
        
        eigenmode = search_A.get_eigenmode()
        curvature_A = search_A.get_curvature()
        
        # Store true minimum mode and curvature
        self.eigenmodes[0] = eigenmode
        self.curvatures[0] = curvature_A

        # Positive curvature uses ASE Dimer's drag-up branch. Kappa Phase B
        # and gamma weighting are mathematically unused there.
        if curvature_A > 0.0:
            self.kappa = 0.0
            self.kappa_active = False
            self.translation_regime = "standard"
            return

        # recover_fmax is a center-geometry switch back to normal Dimer.
        # Decide it once here so trial positions cannot flip the regime.
        self.kappa_active = self.real_fmax() >= self.recover_fmax
        if not self.kappa_active:
            self.kappa = 0.0
            self.translation_regime = "standard"
            return
        self.translation_regime = "kappa"

        # ---------------------------------------------------------
        # PHASE B: Constrained rotation to find kappa_mode and kappa
        # ---------------------------------------------------------
        true_forces = self.forces0
        force_norm = norm(true_forces) 
        f_hat = true_forces / force_norm if force_norm > 1e-8 else None


        def fresh_guess(): # If its the first start, it will give the eigenmode projected onto the isopotential hyperplane as the initial guess.  
            if f_hat is None:
                return eigenmode.copy()
            g = perpendicular_vector(eigenmode, f_hat)
            if norm(g) > 1e-8:
                return g / norm(g)
            dummy = np.random.randn(*eigenmode.shape)
            g = perpendicular_vector(dummy, f_hat)
            return g / norm(g)


        if self.kappa_mode is not None and f_hat is not None:
            guess = perpendicular_vector(self.kappa_mode, f_hat)
            guess = guess / norm(guess) if norm(guess) > 1e-8 else fresh_guess()
        else:
            guess = fresh_guess()

        phase_b_class = (
            LBFGSKappaEigenmodeSearch
            if self.rotation_optimizer == "lbfgs"
            else KappaEigenmodeSearch
        )
        search_B = self._run_search(phase_b_class, self.kappa_control, guess)
        self.kappa_mode = search_B.get_eigenmode().copy() 
       
        curvature_kappa = search_B.get_curvature()
        true_forces = self.forces0
        force_norm = norm(true_forces)
        
        self.kappa = -(curvature_kappa / force_norm) if force_norm >1e-8 else 0.0

    def get_projected_forces(self, pos=None):
        """
        Overrides the translation force calculation to apply the kappa penalty
        and switching functions.
        """
        # Get true forces at the current center
        if pos is not None:
            forces = self.get_forces(real=True, pos=pos).copy()
        else:
            forces = self.forces0.copy()

        eigenmode = self.eigenmodes[0]

        # 1. Calculate standard parallel and perpendicular force components
        f_parallel = parallel_vector(forces, eigenmode)
        f_perp = forces - f_parallel

        # Positive curvature is stock Dimer drag-up. Do not apply Kappa
        # gamma weighting before the inflection point.
        if self.curvatures[0] > 0.0:
            if pos is None:
                self.translation_regime = "standard"
            return -f_parallel

        # 2. Calculate switching functions based on the accepted CENTER regime.
        # Trial positions evaluated by an optimizer do not independently switch
        # Kappa on/off.
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

        if pos is None:
            self.translation_regime = regime

        # 3. Construct the final modified translation force
        # A standard dimer is simply: f_translated = f_perp - f_parallel
        # The kappa dimer dynamically blends components and adds the lateral penalty
        f_translated = -(gamma_1 * f_parallel) + (gamma_2 * f_perp)

        return f_translated

    def eigenmode_log(self):
        """Log the eigenmodes (eigenmode estimates)"""
        if self.mlogfile is not None:
            l = 'MINMODE:MODE: Optimization Step: %i\n' % \
                (self.control.get_counter('optcount'))
            l += 'MINMODE:KAPPA: %15.8f\n' % self.kappa
            for m_num, mode in enumerate(self.eigenmodes):
                l += 'MINMODE:MODE: Order: %i\n' % m_num
                for k in range(len(mode)):
                    l += 'MINMODE:MODE: %7i %15.8f %15.8f %15.8f\n' % (
                        k, mode[k][0], mode[k][1], mode[k][2])
            self.mlogfile.write(l)
            self.mlogfile.flush()
