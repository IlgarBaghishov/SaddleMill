from __future__ import annotations

import logging
import numpy as np
from ase.optimize.precon import Precon, PreconImages

from ase.mep.neb import BaseNEB, NEBState
from ase.mep.neb import NEBMethod


class swDNEB(NEBMethod):

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
        perp_pot_force = imgforce.copy()
        perp_pot_force_norm = np.linalg.norm(perp_pot_force)
        perp_pot_force /= perp_pot_force_norm if perp_pot_force_norm > 0 else 1

        # Improved parallel spring force (formula 12 of paper on Improved tangent)
        imgforce += (spring2.nt * spring2.k - spring1.nt * spring1.k) * tangent

        spring_force = spring2.t * spring2.k - spring1.t * spring1.k

        # # Or use this spring formula from aseneb
        # imgforce += np.vdot(spring_force, tangent) * tangent

        perp_spring_force = spring_force - np.vdot(spring_force, tangent) * tangent
        perp_spring_force_norm = np.linalg.norm(perp_spring_force) or 1
        ratio = perp_pot_force_norm / perp_spring_force_norm
        sw = 2/np.pi * np.arctan(ratio**2)
        imgforce += sw * (perp_spring_force - np.vdot(perp_spring_force, perp_pot_force) * perp_pot_force)


class OCPNEB(BaseNEB):
    def __init__(
        self,
        images,
        k=5,
        climb=False,
        parallel=False,
        remove_rotation_and_translation=False,
        world=None,
        method="improvedtangent",
        allow_shared_calculator=True,
        precon=None,
        batch_size=8,
        dneb=False,
        vasp=False,
        intermediate_minima=False,
        intermediate_minima_min_depth=0.01,
    ):
        super().__init__(
            images,
            k=k,
            climb=climb,
            parallel=parallel,
            remove_rotation_and_translation=remove_rotation_and_translation,
            world=world,
            method=method,
            allow_shared_calculator=allow_shared_calculator,
            precon=precon,
        )
        if dneb: self.neb_method = swDNEB(self)
        self.vasp = vasp
        self.intermediate_minima = intermediate_minima
        self.intermediate_minima_min_depth = abs(intermediate_minima_min_depth)

        if not self.vasp:
            from fairchem.core.common.utils import setup_imports, setup_logging
            from fairchem.core.datasets.atomic_data import atomicdata_list_to_batch
            self.atomicdata_list_to_batch = atomicdata_list_to_batch
            
            self.batch_size = batch_size
            setup_imports()
            setup_logging()

            tmp_calc = self.images[1].calc
            self.predictor = tmp_calc.predictor
            self.a2g = tmp_calc.a2g

            self.reactant_energy = self.images[0].get_potential_energy()
            self.reactant_forces = self.images[0].get_forces()
            self.product_energy = self.images[-1].get_potential_energy()
            self.product_forces = self.images[-1].get_forces()

            self.intermediate_energies = []
            self.intermediate_forces = []
            self.cached = False


    def get_forces(self):
        if self.vasp and not self.intermediate_minima:
            return super().get_forces()
        elif self.vasp:
            # VASP + intermediate_minima: per-image evaluation + custom NEB forces
            images = self.images[1:-1]
            forces = np.array([img.get_forces() for img in images])
            energies = np.empty(self.nimages)
            energies[0] = self.images[0].get_potential_energy()
            energies[-1] = self.images[-1].get_potential_energy()
            for idx, img in enumerate(images):
                energies[idx + 1] = img.get_potential_energy()
            self.reactant_forces = self.images[0].get_forces()
            self.product_forces = self.images[-1].get_forces()
            forces = forces.reshape((len(images), self.natoms, 3))
            return self.get_precon_forces(forces, energies, self.images)
        else:
            images = self.images[1:-1]
            if self.cached:
                return self.intermediate_forces
            else:
                energies_calcd = []
                forces_calcd = []
                for i in range(0, len(images), self.batch_size):
                    batch_images = images[i : i + self.batch_size]
                    data_list = [self.a2g(img) for img in batch_images]
                    batch = self.atomicdata_list_to_batch(data_list)

                    predictions = self.predictor.predict(batch)
                    energies_calcd.extend(predictions["energy"].detach().cpu().flatten().tolist())
                    forces_calcd.extend(predictions["forces"].detach().cpu().numpy())

                forces = np.array(forces_calcd)

                energies = np.empty(self.nimages)
                energies[1:-1] = energies_calcd

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

                return forces

    def set_positions(self, positions):
        if not self.vasp:
            self.cached = False
        return super().set_positions(positions)

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
        self.real_forces[0] = self.reactant_forces
        self.real_forces[-1] = self.product_forces

        state = NEBState(self, images, energies)

        # Determine climbing and intermediate minima image sets
        imin_set = set()
        if self.intermediate_minima:
            for i in range(2, self.nimages - 2):  # exclude endpoint-adjacent images to ensure each segment has room for a CI
                if (energies[i] < energies[i - 1] - self.intermediate_minima_min_depth and
                        energies[i] < energies[i + 1] - self.intermediate_minima_min_depth):
                    imin_set.add(i)
        climbing_set = set()
        if self.climb:
            if imin_set:
                # Per-segment climbing: highest energy interior image per segment
                boundaries = sorted([0] + list(imin_set) + [self.nimages - 1])
                for s in range(len(boundaries) - 1):
                    inner = [idx for idx in range(boundaries[s] + 1, boundaries[s + 1])
                             if idx not in imin_set]
                    if inner:
                        climbing_set.add(max(inner, key=lambda idx: energies[idx]))
            else:
                climbing_set.add(state.imax)

        # Set imax to the global highest energy among climbing images (for result collection)
        if climbing_set:
            self.imax = max(climbing_set, key=lambda idx: energies[idx])
            self.emax = energies[self.imax]
        else:
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

            if i in imin_set:
                pass  # Full PES force, no spring force, no tangential modification
            elif i in climbing_set:
                if self.method == "aseneb":
                    tangent_mag = np.vdot(tangent, tangent)
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
