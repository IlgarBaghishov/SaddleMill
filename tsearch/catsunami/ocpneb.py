from __future__ import annotations

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


def _find_segment_ci(seg_start, seg_end, climbing_set, energies):
    """Find the climbing-image index for a segment [seg_start, seg_end].

    Returns the CI index, or None if the segment has no interior images.
    """
    for ci_idx in climbing_set:
        if seg_start < ci_idx < seg_end:
            return ci_idx
    interior = [j for j in range(seg_start + 1, seg_end)]
    if interior:
        return max(interior, key=lambda idx: energies[idx])
    return None


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
        intermediate_minima_min_depth=0.05,
        intermediate_minima_check_interval=100,
        initial_imin_set=None,
        frozen_images=None,
        freeze_fmax=None,
        freeze_endpoint_fmax=None,
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
        self.intermediate_minima_check_interval = intermediate_minima_check_interval
        self._force_call_count = 0
        self._check_imin_first_call = True if initial_imin_set is None else False
        self._imin_set = set(initial_imin_set) if initial_imin_set else set()
        self._imin_seeded = initial_imin_set is not None
        self._climbing_set = set()
        self.image_fmax = np.zeros(self.nimages)
        self._frozen_set = set(frozen_images) if frozen_images else set()
        self._freeze_fmax = freeze_fmax
        self._freeze_endpoint_fmax = freeze_endpoint_fmax

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

            self.intermediate_forces = []
            self.cached = False
            self._frozen_energies = {}
            self._frozen_pbc_forces = {}


    def get_forces(self):
        if self.vasp and not self.intermediate_minima and not self._imin_set and not self._frozen_set:
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
                # Convert absolute frozen indices to relative indices in images[1:-1]
                frozen_rel = set()
                for abs_idx in self._frozen_set:
                    if 1 <= abs_idx <= self.nimages - 2:
                        frozen_rel.add(abs_idx - 1)
                need_eval_rel = sorted(
                    idx for idx in range(len(images))
                    if idx not in frozen_rel or (idx + 1) not in self._frozen_energies
                )
                eval_images = [images[idx] for idx in need_eval_rel]

                energies_calcd = []
                forces_calcd = []
                for i in range(0, len(eval_images), self.batch_size):
                    batch_images = eval_images[i : i + self.batch_size]
                    data_list = [self.a2g(img) for img in batch_images]
                    batch = self.atomicdata_list_to_batch(data_list)

                    predictions = self.predictor.predict(batch)
                    energies_calcd.extend(predictions["energy"].detach().cpu().flatten().tolist())
                    forces_calcd.extend(predictions["forces"].detach().cpu().numpy())

                eval_forces = (np.array(forces_calcd).reshape(len(eval_images), self.natoms, 3)
                               if eval_images else np.empty((0, self.natoms, 3)))
                eval_map = {rel: i for i, rel in enumerate(need_eval_rel)}

                forces = np.zeros((len(images), self.natoms, 3))
                energies = np.empty(self.nimages)
                energies[0] = self.reactant_energy
                energies[-1] = self.product_energy

                for rel_idx in range(len(images)):
                    abs_idx = rel_idx + 1
                    if rel_idx in eval_map:
                        ei = eval_map[rel_idx]
                        energies[abs_idx] = energies_calcd[ei]
                        forces[rel_idx] = eval_forces[ei]
                        if abs_idx in self._frozen_set:
                            self._frozen_energies[abs_idx] = energies_calcd[ei]
                            self._frozen_pbc_forces[abs_idx] = eval_forces[ei].copy()
                    else:
                        energies[abs_idx] = self._frozen_energies[abs_idx]
                        forces[rel_idx] = self._frozen_pbc_forces[abs_idx]

                # Handle constraints
                if self.images[0].constraints and np.equal(self.images[0].get_tags(), np.zeros(len(self.images[0]),int)).all():
                    fixed_atoms = self.images[0].constraints[0].get_indices()
                elif not np.equal(self.images[0].get_tags(), np.zeros(len(self.images[0]),int)).all():
                    fixed_atoms = np.array([idx for idx, tag in enumerate(self.images[0].get_tags()) if tag == 0])
                else:
                    fixed_atoms = np.array([],dtype=int)

                for i in range(len(images)):
                    for fixed_atom in fixed_atoms:
                        forces[i, fixed_atom] = [0, 0, 0]

                forces = self.get_precon_forces(forces, energies, self.images)

                self.intermediate_forces = forces
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
        self._force_call_count += 1

        # Evaluate intermediate minima periodically (frozen between checks)
        should_check_imin = (self.intermediate_minima and
                             (self._check_imin_first_call or
                              self._force_call_count % self.intermediate_minima_check_interval == 0))
        if should_check_imin:
            self._check_imin_first_call = False
            frozen_and_neighbors = set()
            for f in self._frozen_set:
                frozen_and_neighbors.update([f - 1, f, f + 1])
            imin_set = set()
            for i in range(2, self.nimages - 2):  # exclude endpoint-adjacent images to ensure each segment has room for a CI
                if (i not in frozen_and_neighbors and
                        energies[i] < energies[i - 1] - self.intermediate_minima_min_depth and
                        energies[i] < energies[i + 1] - self.intermediate_minima_min_depth):
                    imin_set.add(i)
            self._imin_set = imin_set
        elif not self.intermediate_minima and not self._imin_seeded:
            self._imin_set = set()
        # else: reuse frozen self._imin_set from last check (or seeded set)

        imin_set = self._imin_set

        # Recompute climbing set. Only imin create segment boundaries (never frozen images).
        # Frozen CIs are locked: once a CI freezes, it stays as CI for its segment permanently.
        frozen_cis = self._climbing_set & self._frozen_set
        climbing_set = set()
        if self.climb:
            if imin_set:
                boundaries = sorted(set([0, self.nimages - 1]) | imin_set)
                for s in range(len(boundaries) - 1):
                    seg_start, seg_end = boundaries[s], boundaries[s + 1]
                    seg_frozen_cis = {idx for idx in frozen_cis if seg_start < idx < seg_end}
                    if seg_frozen_cis:
                        climbing_set.update(seg_frozen_cis)
                    else:
                        inner = [idx for idx in range(seg_start + 1, seg_end)
                                 if idx not in imin_set and idx not in self._frozen_set]
                        if inner:
                            climbing_set.add(max(inner, key=lambda idx: energies[idx]))
            else:
                if frozen_cis:
                    climbing_set.update(frozen_cis)
                else:
                    candidates = [idx for idx in range(1, self.nimages - 1)
                                  if idx not in self._frozen_set]
                    if candidates:
                        climbing_set.add(max(candidates, key=lambda idx: energies[idx]))
        self._climbing_set = climbing_set

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

            if i in self._frozen_set:
                imgforce[:] = 0.0
            elif i in imin_set:
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

        # Compute per-image effective fmax (max force magnitude across atoms)
        for img_i in range(1, self.nimages - 1):
            effective_force = precon_forces[img_i - 1]  # (natoms, 3), already modified in-place
            self.image_fmax[img_i] = float(np.sqrt((effective_force**2).sum(axis=1)).max())
            images[img_i].info['effective_fmax'] = self.image_fmax[img_i]
        # Endpoints: PES force only
        self.image_fmax[0] = float(np.sqrt((self.real_forces[0]**2).sum(axis=1)).max())
        self.image_fmax[-1] = float(np.sqrt((self.real_forces[-1]**2).sum(axis=1)).max())
        images[0].info['effective_fmax'] = self.image_fmax[0]
        images[-1].info['effective_fmax'] = self.image_fmax[-1]
        # Store band metadata for debug traj (on all images so any frame can be used for extraction)
        for img_i in range(self.nimages):
            images[img_i].info['nimages'] = self.nimages
        images[0].info['imin_set'] = sorted(self._imin_set)
        images[0].info['climbing_set'] = sorted(self._climbing_set)

        if self._freeze_fmax is not None:
            prev_frozen = set(self._frozen_set)
            self._auto_freeze()
            # Cache energy/forces for newly frozen images so they skip batching next step
            for abs_idx in self._frozen_set - prev_frozen:
                if abs_idx not in self._frozen_energies:
                    self._frozen_energies[abs_idx] = energies[abs_idx]
                    self._frozen_pbc_forces[abs_idx] = self.real_forces[abs_idx].copy()

        return precon_forces.reshape((-1, 3))

    def _auto_freeze(self):
        """Add newly converged sub-bands/imin/CIs to frozen set.

        Never freezes individual regular images. Runs at every step when
        freeze_fmax is set.
        """
        fmax = self._freeze_fmax
        endpoint_fmax = self._freeze_endpoint_fmax or fmax
        imin_set = self._imin_set
        climbing_set = self._climbing_set
        boundaries = sorted([0] + list(imin_set) + [self.nimages - 1])

        for s in range(len(boundaries) - 1):
            seg_start, seg_end = boundaries[s], boundaries[s + 1]

            all_converged = all(
                self.image_fmax[j] < (endpoint_fmax if (j in imin_set or j == 0 or j == self.nimages - 1) else fmax)
                for j in range(seg_start, seg_end + 1)
            )
            if all_converged:
                for j in range(seg_start, seg_end + 1):
                    if 0 < j < self.nimages - 1:
                        self._frozen_set.add(j)
            else:
                for j in [seg_start, seg_end]:
                    if (j in imin_set and 0 < j < self.nimages - 1
                            and j not in self._frozen_set
                            and self.image_fmax[j] < endpoint_fmax):
                        self._frozen_set.add(j)
                seg_ci = _find_segment_ci(seg_start, seg_end, climbing_set, self.energies)
                if (seg_ci is not None and seg_ci not in self._frozen_set
                        and self.image_fmax[seg_ci] < fmax):
                    self._frozen_set.add(seg_ci)
