import numpy as np
from ase.mep.dimer import DimerEigenmodeSearch, MinModeAtoms, perpendicular_vector, parallel_vector, DimerControl

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


class CGRotationMixin:
    """Polak-Ribiere conjugate-gradient rotation (tsase KSSDimer rotationOpt='cg').

    ASE's DimerEigenmodeSearch rotates in the plane spanned by the eigenmode and
    the INSTANTANEOUS rotational force -- steepest descent, no memory. tsase's
    rotation_plane() instead mixes the previous search direction in with a PR
    coefficient, with a hard restart (gamma=0) when successive rotational forces
    stay too parallel:

        a = |<F, F_old>|; b = <F_old, F_old>
        gamma = <F, F - F_old> / b   if a <= 0.5 b and b != 0, else 0
        T     = F + gamma * T_old,   then re-orthogonalized to the eigenmode

    That memory is what lets the rotation converge in fewer force calls on the
    noisy soft-mode landscapes where the plain search jitters. CG state lives on
    the search object, which is constructed fresh each translation step, so the
    memory resets per step -- same as tsase's iteration==0 reset.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._cg_f_old = None
        self._cg_dir = None

    def _cg_mix(self, f_rot):
        f_old, d_old = self._cg_f_old, self._cg_dir
        if f_old is None or d_old is None or f_old.shape != f_rot.shape:
            mixed = f_rot.copy()
        else:
            a = abs(np.vdot(f_rot, f_old))
            b = np.vdot(f_old, f_old)
            if a <= 0.5 * b and b > 1e-30:
                gamma = np.vdot(f_rot, f_rot - f_old) / b
            else:
                gamma = 0.0
            mixed = f_rot + gamma * d_old
        # rotation direction must stay perpendicular to the current dimer axis
        mixed = perpendicular_vector(mixed, self.eigenmode)
        self._cg_f_old = f_rot.copy()
        self._cg_dir = mixed.copy()
        return mixed


class CGDimerEigenmodeSearch(CGRotationMixin, DimerEigenmodeSearch):
    """Phase-A rotation with CG mixing; otherwise identical to ASE's search."""

    def get_rotational_force(self):
        return self._cg_mix(super().get_rotational_force())


class KappaEigenmodeSearch(CGRotationMixin, DimerEigenmodeSearch):
    """
    Phase B Rotation: Constrains the dimer rotation to the isopotential hyperplane
    and applies CG mixing. Inherits the converge_to_eigenmode() loop from ASE,
    overriding only the rotational force calculation.

    Order of operations per rotation: raw rotational force -> project onto the
    plane perpendicular to the true force (the isopotential constraint) -> CG mix
    with the previous direction -> re-impose the constraint (the mixed-in history
    can leak a component along f_hat).
    """

    def get_rotational_force(self):
        rot_force = super().get_rotational_force()

        true_forces = self.dimeratoms.forces0
        fnorm = norm(true_forces)
        if fnorm < 1e-8:
            return self._cg_mix(rot_force)

        f_hat = true_forces / fnorm
        constrained = perpendicular_vector(rot_force, f_hat)
        mixed = self._cg_mix(constrained)
        return perpendicular_vector(mixed, f_hat)

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


class KappaMinModeAtoms(MinModeAtoms):
    """
    Extended MinModeAtoms to handle the Phase A/B double rotation and
    the kappa-weighted translation forces.

    Phase B (kappa rotation + gammas) is SKIPPED entirely whenever its output
    would be unused, which is two independent cases re-evaluated every step:

      1. curvature_A > 0 (above an inflection, not a saddle-like region):
         translation drags straight up along the mode, exactly like the stock
         ASE dimer and the k-dimer paper's "drag up directly" branch. Neither
         kappa nor the gammas enter that force, so the constrained rotation
         would be wasted force calls.
      2. fmax < recover_fmax (the releaseF idea): normal-dimer regime,
         gamma_1 = gamma_2 = 1, kappa unused.

    Kappa on/off state (`self.kappa_active`):
      - hysteresis=False (default): active iff fmax >= recover_fmax, re-evaluated
        every step.
      - hysteresis=True: two thresholds. Kappa turns OFF when fmax drops below
        recover_fmax and only turns back ON when fmax rises above reactivate_fmax
        (> recover_fmax). Prevents rapid flip-flopping when fmax hovers at the
        boundary. Fallback cost: after an off->on flip, self.kappa_mode is stale;
        it is re-projected onto the current isopotential plane before Phase B.

    cg_rotation=True applies Polak-Ribiere CG mixing to BOTH rotations
    (Phase A via CGDimerEigenmodeSearch, Phase B via KappaEigenmodeSearch).
    Set cg_rotation=False to recover the previous steepest-descent behavior.
    """
    def __init__(self, atoms, beta=2.0, recover_fmax=0.3, kappa_control=None,
                 hysteresis=False, reactivate_fmax=0.45, cg_rotation=True,
                 **kwargs):
        super().__init__(atoms, **kwargs)

        # Tuning parameters for the translation step
        self.beta = beta           # Steepness of the switching function
        self.kappa = 0.0           # Initialize isopotential curvature
        self.recover_fmax = recover_fmax # max atom force norm (like EDIFFG). to determine if to switch back to normal dimer method.
        self.hysteresis = bool(hysteresis)
        self.reactivate_fmax = float(reactivate_fmax)
        self.cg_rotation = bool(cg_rotation)
        if self.hysteresis and self.reactivate_fmax < self.recover_fmax:
            raise ValueError(
                f"kappa_reactivate_fmax ({self.reactivate_fmax}) must be >= "
                f"kappa_recover_fmax ({self.recover_fmax}) for hysteresis to make sense.")
        self.kappa_active = True   # start in kappa mode (fmax is high at the start)
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

    def _update_kappa_state(self, fmax_atom):
        """Update self.kappa_active from the current per-atom fmax.

        No hysteresis: active <=> fmax >= recover_fmax (memoryless, old condition).
        Hysteresis:    active -> inactive when fmax < recover_fmax;
                       inactive -> active when fmax > reactivate_fmax;
                       otherwise hold the previous state (the dead band).
        """
        if self.hysteresis:
            if self.kappa_active and fmax_atom < self.recover_fmax:
                self.kappa_active = False
            elif not self.kappa_active and fmax_atom > self.reactivate_fmax:
                self.kappa_active = True
        else:
            self.kappa_active = fmax_atom >= self.recover_fmax

    def find_eigenmodes(self, order=1):
        """
        Launches eigenmode search and kappa search.
        Overrides ASE's standard eigenmode search to run Phase A and Phase B.
        Phase B (kappa rotation) is skipped whenever its output would be unused:
        positive Phase-A curvature (translation is a pure drag-up; see
        get_projected_forces) or kappa inactive (normal-dimer regime).
        """
        if order > 1:
            raise NotImplementedError("Kappa-dimer only supports 1st order saddles.")

        # ---------------------------------------------------------
        # PHASE A: Standard unconstrained rotation to find eigenmode and curvature_A
        # ---------------------------------------------------------
        SearchA = CGDimerEigenmodeSearch if self.cg_rotation else DimerEigenmodeSearch
        search_A = SearchA(self, self.control, eigenmode=self.eigenmodes[0])
        search_A.converge_to_eigenmode()
        search_A.set_up_for_optimization_step()

        eigenmode = search_A.get_eigenmode()
        curvature_A = search_A.get_curvature()

        # Store true minimum mode and curvature
        self.eigenmodes[0] = eigenmode
        self.curvatures[0] = curvature_A

        # ---------------------------------------------------------
        # Skip Phase B when its output cannot be used (cheapest checks first)
        # ---------------------------------------------------------
        if curvature_A > 0.0:
            # Above an inflection: get_projected_forces drags straight up along
            # the mode ("drag up directly"); kappa and the gammas never enter.
            self.kappa = 0.0
            return

        fmax_atom = np.sqrt((self.forces0 ** 2).sum(axis=1).max())
        self._update_kappa_state(fmax_atom)
        if not self.kappa_active:
            # Normal-dimer regime: skip the constrained rotation entirely.
            # kappa_mode is kept as-is so a hysteresis reactivation has a warm
            # (if stale) starting guess; it gets re-projected onto the new
            # isopotential plane below when Phase B next runs.
            self.kappa = 0.0
            return

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

        search_B = KappaEigenmodeSearch(self, self.kappa_control, eigenmode=guess)
        search_B.converge_to_eigenmode()
        self.kappa_mode = search_B.get_eigenmode().copy()

        curvature_kappa = search_B.get_curvature()
        true_forces = self.forces0
        force_norm = norm(true_forces)

        self.kappa = -(curvature_kappa / force_norm) if force_norm >1e-8 else 0.0

    def get_projected_forces(self, pos=None):
        """
        Overrides the translation force calculation to apply the kappa penalty
        and switching functions.

        Positive curvature: pure inversion along the mode (f = -f_parallel),
        matching stock ASE MinModeAtoms and the k-dimer paper's "drag up
        directly" branch. The previous version blended the gammas here too,
        which deviated from the reference behavior AND paid for a Phase-B
        rotation whose output never entered the force.

        When kappa is inactive (self.kappa_active False, set in find_eigenmodes
        for this step) this reduces exactly to the standard dimer translation
        force (gamma_1 = gamma_2 = 1).
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

        # 2. Positive curvature: drag up directly (stock-dimer branch).
        if self.curvatures[0] > 0.0:
            return -f_parallel

        # 3. Switching functions. The on/off decision lives in _update_kappa_state
        # (called from find_eigenmodes each step) so rotation and translation
        # always agree within a step -- no re-thresholding here.
        if not self.kappa_active:
            gamma_1 = 1.0
            gamma_2 = 1.0
        else:
            bk = np.clip(self.beta * self.kappa, -500.0, 500.0)
            exp_term = np.exp(bk)
            gamma_1 = (2.0 / (1.0 + exp_term)) - 1.0
            gamma_2 = 1.0 - (1.0 / (1.0 + exp_term))

        # 4. Construct the final modified translation force
        # A standard dimer is simply: f_translated = f_perp - f_parallel
        # The kappa dimer dynamically blends components and adds the lateral penalty
        f_translated = -(gamma_1 * f_parallel) + (gamma_2 * f_perp)

        return f_translated

    def eigenmode_log(self):
        if self.mlogfile is None or not hasattr(self, 'forces0'):
            return
        fmax_atom = np.sqrt((self.forces0 ** 2).sum(axis=1).max())
        dragup = self.curvatures[0] > 0.0
        if dragup or not self.kappa_active:
            g1 = g2 = 1.0
        else:
            bk = np.clip(self.beta * self.kappa, -500.0, 500.0)
            et = np.exp(bk); g1 = (2.0/(1.0+et)) - 1.0; g2 = 1.0 - (1.0/(1.0+et))
        self.mlogfile.write(
            'MINMODE:KAPPA: step %i kappa %15.8f fmax %10.6f g1 %8.5f g2 %8.5f normal %d hyst %d dragup %d\n'
            % (self.control.get_counter('optcount'), self.kappa, fmax_atom, g1, g2,
               int(not self.kappa_active), int(self.hysteresis), int(dragup)))
        self.mlogfile.flush()
