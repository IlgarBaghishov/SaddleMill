from __future__ import annotations

import logging
import numpy as np
from ase.optimize.precon import Precon, PreconImages

from fairchem.core.common.utils import setup_imports, setup_logging
from fairchem.core.datasets.atomic_data import atomicdata_list_to_batch

from ase.mep.neb import DyNEB, NEBState
from ase.mep.neb import FullSpringMethod, ASENEBMethod, ImprovedTangentMethod, SplineMethod, StringMethod, NEBMethod


class swDNEB(NEBMethod):
    """
    Tangent estimates and spring force are according to Eqs. 12-15 in paper IV.
    """

    def get_tangent(self, state, spring1, spring2, i):
        energies = state.energies
        if energies[i + 1] > energies[i] > energies[i - 1]:
            tangent = spring2.t.copy()
        elif energies[i + 1] < energies[i] < energies[i - 1]:
            tangent = spring1.t.copy()
        else:
            deltavmax = max(abs(energies[i + 1] - energies[i]),
                            abs(energies[i - 1] - energies[i]))
            deltavmin = min(abs(energies[i + 1] - energies[i]),
                            abs(energies[i - 1] - energies[i]))
            if energies[i + 1] > energies[i - 1]:
                tangent = spring2.t * deltavmax + spring1.t * deltavmin
            else:
                tangent = spring2.t * deltavmin + spring1.t * deltavmax
        # Normalize the tangent vector
        norm = np.linalg.norm(tangent)
        tangent /= norm if norm > 0 else 1
        return tangent

    def add_image_force(self, state, tangential_force, tangent, imgforce,
                        spring1, spring2, i):
        imgforce -= tangential_force * tangent
        perp_pot_force = imgforce
        perp_pot_force_norm = np.linalg.norm(perp_pot_force)
        perp_pot_force /= perp_pot_force_norm if perp_pot_force_norm > 0 else 1

        # # Improved parallel spring force (formula 12 of paper I)
        # imgforce += (spring2.nt * spring2.k - spring1.nt * spring1.k) * tangent

        spring_force = spring2.t * spring2.k - spring1.t * spring1.k
        
        # Or use this one from aseneb
        imgforce += np.vdot(spring_force, tangent) * tangent

        perp_spring_force = spring_force - np.vdot(spring_force, tangent) * tangent
        perp_spring_force_norm = np.linalg.norm(perp_spring_force) or 1
        sw = 2/np.pi * np.arctan(perp_pot_force_norm**2 / perp_spring_force_norm**2)
        imgforce += sw * (perp_spring_force - np.vdot(perp_spring_force, perp_pot_force) * perp_pot_force)


class OCPNEB(DyNEB):
    def __init__(
        self,
        images,
        k=5,
        fmax=0.05,
        climb=False,
        parallel=False,
        remove_rotation_and_translation=False,
        world=None,
        dynamic_relaxation=False,
        scale_fmax=0.0,
        method="improvedtangent",
        allow_shared_calculator=True,
        precon=None,
        batch_size=8,
        dneb=False,
    ):
        """
        Subclass of NEB that allows for scaled and dynamic optimizations of
        images. This method, which only works in series, does not perform
        force calls on images that are below the convergence criterion.
        The convergence criteria can be scaled with a displacement metric
        to focus the optimization on the saddle point region.

        'Scaled and Dynamic Optimizations of Nudged Elastic Bands',
        P. Lindgren, G. Kastlunger and A. A. Peterson,
        J. Chem. Theory Comput. 15, 11, 5787-5793 (2019).

        dynamic_relaxation: bool
            True skips images with forces below the convergence criterion.
            This is updated after each force call; if a previously converged
            image goes out of tolerance (due to spring adjustments between
            the image and its neighbors), it will be optimized again.
            False reverts to the default NEB implementation.

        fmax: float
            Must be identical to the fmax of the optimizer.

        scale_fmax: float
            Scale convergence criteria along band based on the distance between
            an image and the image with the highest potential energy. This
            keyword determines how rapidly the convergence criteria are scaled.
        """
        super().__init__(
            images,
            k=k,
            climb=climb,
            fmax=fmax,
            dynamic_relaxation=dynamic_relaxation,
            parallel=parallel,
            remove_rotation_and_translation=remove_rotation_and_translation,
            world=world,
            method=method,
            allow_shared_calculator=allow_shared_calculator,
            precon=precon,
            scale_fmax=scale_fmax,
        )
        if dneb: self.neb_method = swDNEB(self)

        self.batch_size = batch_size
        setup_imports()
        setup_logging()

        tmp_calc = self.images[1].calc
        self.predictor = tmp_calc.predictor
        self.a2g = tmp_calc.a2g

        if self.method != 'aseneb':
            self.reactant_energy = self.images[0].get_potential_energy()
            self.product_energy = self.images[-1].get_potential_energy()

        self.intermediate_energies = []
        self.intermediate_forces = []
        self.cached = False


    def get_forces(self):
        images = self.images[1:-1]
        if self.cached:
            energies = self.intermediate_energies
            forces = self.intermediate_forces
        else:
            energies_calcd = []
            forces_calcd = []
            for i in range(0, len(images), self.batch_size):
                batch_images = images[i : i + self.batch_size]
                data_list = [self.a2g(img) for img in batch_images]
                batch = atomicdata_list_to_batch(data_list)

                predictions = self.predictor.predict(batch)
                energies_calcd.extend(predictions["energy"].detach().cpu().flatten().tolist())
                forces_calcd.extend(predictions["forces"].detach().cpu().numpy())

            forces = np.array(forces_calcd)

            energies = np.empty(self.nimages)
            energies[1:-1] = energies_calcd

            if self.method != 'aseneb':
                energies[0] = self.reactant_energy
                energies[-1] = self.product_energy

            # Handle constraints:
            if self.images[0].constraints and np.equal(self.images[0].get_tags(), np.zeros(len(self.images[0]),int)).all():  # if had constraints and all atom tags are 0
                fixed_atoms = self.images[0].constraints[0].get_indices()
            elif not np.equal(self.images[0].get_tags(), np.zeros(len(self.images[0]),int)).all():
                fixed_atoms = np.array([idx for idx, tag in enumerate(self.images[0].get_tags()) if tag == 0])
            else:
                fixed_atoms = np.array([],dtype=int) 

            for i in range(self.nimages - 2):
                for fixed_atom in fixed_atoms:
                    forces[fixed_atom + len(images[0]) * i] = [0, 0, 0]

            forces = np.reshape(forces, (len(images), self.natoms, 3))
            forces = self.get_precon_forces(forces, energies, self.images)

            self.intermediate_forces = forces
            self.intermediate_energies = energies
            self.cached = True

        if not self.dynamic_relaxation:
            return forces
        """Get NEB forces and scale the convergence criteria to focus
           optimization on saddle point region. The keyword scale_fmax
           determines the rate of convergence scaling."""
        n = self.natoms
        for i in range(self.nimages - 2):
            n1 = n * i
            n2 = n1 + n
            force = np.sqrt((forces[n1:n2] ** 2.0).sum(axis=1)).max()
            n_imax = (self.imax - 1) * n  # Image with highest energy.

            positions = self.get_positions()
            pos_imax = positions[n_imax : n_imax + n]

            """Scale convergence criteria based on distance between an
               image and the image with the highest potential energy."""
            rel_pos = np.sqrt(((positions[n1:n2] - pos_imax) ** 2).sum())
            if force < self.fmax * (1 + rel_pos * self.scale_fmax):
                if i == self.imax - 1:
                    # Keep forces at saddle point for the log file.
                    pass
                else:
                    # Set forces to zero before they are sent to optimizer.
                    forces[n1:n2, :] = 0

        return forces

    def set_positions(self, positions):

        self.cached = False
        if not self.dynamic_relaxation:
            return super().set_positions(positions)

        n1 = 0
        # old_positions = self.images[0].get_positions()
        # tags_hier = [self.images[i].get_tags() for i in range(self.nimages)]
        # tags = [x for l in tags_hier for x in l]
        for i, image in enumerate(self.images[1:-1]):
            if self.parallel:
                msg = (
                    "Dynamic relaxation does not work efficiently "
                    "when parallelizing over images. Try AutoNEB "
                    "routine for freezing images in parallel."
                )
                raise ValueError(msg)
            else:
                forces_dyn = self._fmax_all(self.images)
                if forces_dyn[i] < self.fmax:
                    n1 += self.natoms
                else:
                    n2 = n1 + self.natoms
                    # new_positions = [old_positions[idx-n1] if tags[idx] == 0 else positions[idx] for idx in range(n1,n2)]
                    # image.set_positions(new_positions)
                    image.set_positions(positions[n1:n2])
                    n1 = n2
        return None

    def get_precon_forces(self, forces, energies, images):
        if self.precon is None or isinstance(self.precon, (str, Precon, list)):
            self.precon = PreconImages(self.precon, images)

        # apply preconditioners to transform forces
        # for the default IdentityPrecon this does not change their values
        precon_forces = self.precon.apply(forces, index=slice(1, -1))

        # Save for later use in iterimages:
        self.energies = energies
        self.real_forces = np.zeros((self.nimages, self.natoms, 3))
        self.real_forces[1:-1] = forces

        state = NEBState(self, images, energies)

        # Can we get rid of self.energies, self.imax, self.emax etc.?
        self.imax = state.imax
        self.emax = state.emax

        spring1 = state.spring(0)

        self.residuals = []
        for i in range(1, self.nimages - 1):
            spring2 = state.spring(i)
            tangent = self.neb_method.get_tangent(state, spring1, spring2, i)

            # Get overlap between full PES-derived force and tangent
            tangential_force = np.vdot(forces[i - 1], tangent)

            # from now on we use the preconditioned forces (equal for precon=ID)
            imgforce = precon_forces[i - 1]

            if i == self.imax and self.climb:
                """The climbing image, imax, is not affected by the spring
                forces. This image feels the full PES-derived force,
                but the tangential component is inverted:
                see Eq. 5 in paper II."""
                if self.method == "aseneb":
                    tangent_mag = np.vdot(tangent, tangent)  # For normalizing
                    imgforce -= 2 * tangential_force / tangent_mag * tangent
                else:
                    imgforce -= 2 * tangential_force * tangent
            else:
                self.neb_method.add_image_force(
                    state, tangential_force, tangent, imgforce, spring1, spring2, i
                )
                # compute the residual - with ID precon, this is just max force
                residual = self.precon.get_residual(i, imgforce)
                self.residuals.append(residual)

            spring1 = spring2

        return precon_forces.reshape((-1, 3))
