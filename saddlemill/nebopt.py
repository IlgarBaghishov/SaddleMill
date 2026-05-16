import os
import sys
import shutil
import traceback
import warnings
import zipfile
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from ase.data import covalent_radii
from ase.calculators.singlepoint import SinglePointCalculator
from ase.io import Trajectory
from ase.mep.neb import NEB, NEBTools, NEBState
from saddlemill.catsunami.ocpneb import OCPNEB, _find_segment_ci
from saddlemill.dimeropt import _setup_dimer
from saddlemill.tools import (backup_flux_logs, get_task_name, remove_vasp_heavies,
                              finalize_if_vasp_interactive)


def _expand_band(neb, fmax_threshold, max_num_frames, num_frames, calc):
    """Insert midpoint images into unconverged segments (doubles each segment).

    For each segment whose CI has effective fmax >= fmax_threshold, inserts one
    new image between every consecutive pair (n -> 2n-1 images in that segment).
    Skips a segment if doubling it would exceed num_frames per segment or
    push the total band past max_num_frames.

    Returns new images list, or None if no images could be added.
    """
    imin_set = neb._imin_set
    climbing_set = neb._climbing_set
    boundaries = sorted([0] + list(imin_set) + [neb.nimages - 1])

    expand_gaps = set()
    current_total = len(neb.images)

    for s in range(len(boundaries) - 1):
        seg_start = boundaries[s]
        seg_end = boundaries[s + 1]
        seg_size = seg_end - seg_start + 1

        seg_ci = _find_segment_ci(seg_start, seg_end, climbing_set, neb.energies)
        if seg_ci is None:
            continue

        if neb.image_fmax[seg_ci] < fmax_threshold:
            continue

        # Per-segment cap: doubling would give 2*seg_size - 1
        if 2 * seg_size - 1 > num_frames:
            continue

        images_to_add = seg_end - seg_start  # number of gaps in segment
        if current_total + images_to_add > max_num_frames:
            continue

        for gap_left in range(seg_start, seg_end):
            expand_gaps.add(gap_left)
        current_total += images_to_add

    if not expand_gaps:
        return None

    # Build new image list with IDPP-interpolated midpoints
    # Same logic as initial band setup: linear + MIC, check overlaps, fall back to IDPP
    radii = np.array([covalent_radii[z] for z in neb.images[0].numbers])
    radii_sum = radii[:, None] + radii[None, :]

    new_images = [neb.images[0]]
    for img_idx in range(1, len(neb.images)):
        if (img_idx - 1) in expand_gaps:
            prev = neb.images[img_idx - 1]
            curr = neb.images[img_idx]
            mini_images = [prev.copy(), prev.copy(), curr.copy()]
            mini_neb = NEB(mini_images)
            mini_neb.interpolate(method='linear', mic=True)
            # Check for atom overlap; fall back to IDPP if needed
            dists = mini_images[1].get_all_distances(mic=True)
            np.fill_diagonal(dists, np.inf)
            if np.any(dists < 0.6 * radii_sum):
                try:
                    mini_neb.interpolate(method='idpp', mic=True)
                except Exception:
                    warnings.warn(
                        f"IDPP interpolation failed for midpoint between images "
                        f"{img_idx - 1} and {img_idx}, and the linear midpoint "
                        f"has overlapping atoms. Keeping it anyway."
                    )
            midpoint = mini_images[1]
            midpoint.calc = calc
            new_images.append(midpoint)
        new_images.append(neb.images[img_idx])

    return new_images, expand_gaps


def _remap_indices(old_indices, expand_gaps):
    """Map old band indices to new indices after _expand_band insertion."""
    return {old_idx + sum(1 for g in expand_gaps if g < old_idx)
            for old_idx in old_indices}


def _detect_imin(neb, min_depth):
    """One-shot intermediate minima detection. Merges with existing imin_set.

    Scans interior images (2 .. nimages-3) for local energy minima deeper
    than min_depth below both neighbors. Excludes images within ±1 of
    existing imin (no adjacent imin). Returns set of newly detected indices.
    """
    energies = neb.energies
    # Exclude existing imin and their neighbors (imin can't be adjacent)
    imin_exclusion = set()
    for im in neb._imin_set:
        imin_exclusion.update([im - 1, im, im + 1])

    new_imin = set()
    for i in range(2, neb.nimages - 2):
        if (i not in imin_exclusion and
                energies[i] < energies[i - 1] - min_depth and
                energies[i] < energies[i + 1] - min_depth):
            new_imin.add(i)

    neb._imin_set = neb._imin_set | new_imin
    return new_imin


def _relax_imin(neb, imin_indices, calc, optimizer_class, opt_kwargs,
                target_fmax, freeze_fmax, max_steps, file_tag, temp_files, rank):
    """Relax intermediate minima images and freeze those below freeze_fmax.

    Returns set of newly frozen image indices.
    """
    newly_frozen = set()
    for imin_idx in sorted(imin_indices):
        neb.images[imin_idx].calc = calc
        log_f = f'imin_relax_{file_tag}_img{imin_idx}.log'
        traj_f = f'imin_relax_{file_tag}_img{imin_idx}.traj'
        temp_files.extend([log_f, traj_f])
        try:
            opt = optimizer_class(neb.images[imin_idx], logfile=log_f, trajectory=traj_f, **opt_kwargs)
            opt.run(target_fmax, max_steps)
            imin_fmax = float(np.sqrt((neb.images[imin_idx].get_forces()**2).sum(axis=1)).max())
            if imin_fmax < freeze_fmax:
                neb._frozen_set.add(imin_idx)
                newly_frozen.add(imin_idx)
                print(f"Rank {rank}, structure {file_tag}: relaxed & froze imin {imin_idx} (fmax={imin_fmax:.4f})", flush=True)
            else:
                print(f"Rank {rank}, structure {file_tag}: relaxed imin {imin_idx} (fmax={imin_fmax:.4f}, not frozen)", flush=True)
        except Exception as e:
            print(f"Rank {rank}, structure {file_tag}: imin relax error {imin_idx}: {e}", flush=True)
    return newly_frozen


def nebopt(i, config_dict, images, calc, Optimizer, consecutive_errors=None, executorlib_worker_id=None, **kwargs):

    rank = executorlib_worker_id

    max_consecutive_errors = config_dict["Main"]["max_consecutive_errors"]
    if consecutive_errors is not None and consecutive_errors[0] >= max_consecutive_errors > 0:
        print(f"Rank {rank}: {consecutive_errors[0]} consecutive structures errored. Killing worker for restart.", flush=True)
        backup_flux_logs(rank)
        sys.exit(1)

    continuation_data = kwargs.get('continuation_data')  # dict from extract_previous_results or None
    continue_from_result = config_dict["Main"]["continue_from_result"]

    default_relax = config_dict["ourNEB"]["relax_endpoints"]
    default_interp = config_dict["ourNEB"]["interpolate_method"]
    default_num_frames = config_dict["ourNEB"]["num_frames"]

    # --- Continuation: reconstruct band from previous result or use original input ---
    initial_imin_set = None
    frozen_images = None
    frozen_fmax = None

    if continuation_data is not None and continue_from_result:
        # Reconstruct full band from previous result (de-dup imin at boundaries)
        seen_image_idx = set()
        all_imgs = []
        initial_imin_set = set()
        frozen_images_set = set()
        frozen_fmax_dict = {}
        for sid in sorted(continuation_data.keys()):
            for img in continuation_data[sid]:
                iidx = img.info.get('orig_info', {}).get('image_idx')
                if iidx not in seen_image_idx:
                    seen_image_idx.add(iidx)
                    all_imgs.append(img)
                    orig = img.info.get('orig_info', {})
                    new_idx = len(all_imgs) - 1
                    if orig.get('image_type') == 'intermediate_minimum':
                        initial_imin_set.add(new_idx)
                    if orig.get('image_converged', False):
                        frozen_images_set.add(new_idx)
                        frozen_fmax_dict[new_idx] = orig.get('effective_fmax', 0.0)
        images = all_imgs
        # Endpoints should never be in frozen set (they're fixed by SPC, not by freeze logic)
        last_idx = len(all_imgs) - 1
        frozen_images_set.discard(0)
        frozen_images_set.discard(last_idx)
        frozen_fmax_dict.pop(0, None)
        frozen_fmax_dict.pop(last_idx, None)
        initial_imin_set = initial_imin_set or None
        frozen_images = frozen_images_set or None
        frozen_fmax = frozen_fmax_dict or None
    # else: continuation_data=None or continue_from_result=False → use original images

    zip_name = f"{config_dict['Main']['method']}_debug_zips/structure_rank_{rank}_data.zip"
    status_file = f"{config_dict['Main']['method']}_status_csvs/status_rank_{rank}.csv"
    my_output_file = f"{config_dict['Main']['method']}_trajes/collected_ts_rank_{rank}.traj"
    task_name = get_task_name(config_dict)

    def log_status(status_msg, sub_band_id=0):
        with open(status_file, 'a') as f:
            f.write(f'{i},{rank},{sub_band_id},"{status_msg}"\n')

    perform_aseidpp = False
    num_images = len(images)
    temp_log = f'neb_{i}.log'
    temp_traj = f'neb_{i}.traj'
    temp_plot = f'diffusion_barrier_{i}.png'
    temp_react_relax_log = f'reactant_relaxation_{i}.log'
    temp_prod_relax_log = f'product_relaxation_{i}.log'
    temp_react_relax = f'reactant_relaxation_{i}.traj'
    temp_prod_relax = f'product_relaxation_{i}.traj'
    temp_files = [temp_log, temp_traj, temp_plot, temp_react_relax_log, temp_prod_relax_log, temp_react_relax, temp_prod_relax]

    def _cleanup_temp_files(error=False):
        if config_dict["Main"]["Calculator"] in ("Vasp", "VaspInteractive"):
            for image_idx in range(num_images):
                remove_vasp_heavies(f"VASP_{i}_{image_idx}")
        existing_files = [f for f in temp_files if os.path.exists(f)]
        if existing_files and config_dict['Main']['zip']:
            prefix = "ERROR_" if error else ""
            with zipfile.ZipFile(zip_name, 'a', zipfile.ZIP_DEFLATED) as zf:
                for f_name in existing_files:
                    if os.path.isdir(f_name):
                        for root, dirs, files in os.walk(f_name):
                            for file in files:
                                filepath = os.path.join(root, file)
                                zf.write(filepath, arcname=f"{prefix}{filepath}")
                    else:
                        zf.write(f_name, arcname=f"{prefix}{f_name}")
            for f_name in existing_files:
                if os.path.isdir(f_name):
                    shutil.rmtree(f_name)
                else:
                    os.remove(f_name)

    try:
        need_interpolation = default_interp and num_images <= 2

        def _build_neb_vasp_calc(image_idx, role):
            """role: 'endpoints' or 'intermediates' — picks command/ncore keys."""
            kwargs = {
                "directory": f"VASP_{i}_{image_idx}",
                "command": config_dict["ourNEB"][f"vasp_command_{role}"],
                **config_dict["Vasp"],
            }
            ncore = config_dict["ourNEB"][f"vasp_ncore_{role}"]
            if ncore is not None:
                kwargs["ncore"] = int(ncore)
            return calc(**kwargs)

        reactant = images[0]
        if config_dict["Main"]["Calculator"] in ("Vasp", "VaspInteractive"):
            reactant.calc = _build_neb_vasp_calc(0, "endpoints")
        else:
            reactant.calc = calc

        product = images[-1]
        if config_dict["Main"]["Calculator"] in ("Vasp", "VaspInteractive"):
            product.calc = _build_neb_vasp_calc(num_images - 1, "endpoints")
        else:
            product.calc = calc

        if default_relax:
            if not need_interpolation: print("Are you sure you want to relax end points while keeping the intermediate images from your traj?", flush=True)
            if config_dict["ourNEB"]["endpoint_relax_Optimizer"] is None:
                endpoint_relax_optimizer_name = config_dict["Main"]["Optimizer"]
            else:
                endpoint_relax_optimizer_name = config_dict["ourNEB"]["endpoint_relax_Optimizer"]

            opt = Optimizer[0](reactant, logfile=temp_react_relax_log, trajectory=temp_react_relax, **config_dict[endpoint_relax_optimizer_name])
            opt.run(config_dict["ourNEB"]["endpoint_relax_fmax"], config_dict["ourNEB"]["endpoint_relax_steps"])

            opt = Optimizer[0](product, logfile=temp_prod_relax_log, trajectory=temp_prod_relax, **config_dict[endpoint_relax_optimizer_name])
            opt.run(config_dict["ourNEB"]["endpoint_relax_fmax"], config_dict["ourNEB"]["endpoint_relax_steps"])

        energy, forces = reactant.get_potential_energy(), reactant.get_forces()
        if config_dict["Main"]["Calculator"] == "VaspInteractive": reactant.calc.finalize()
        reactant.calc = SinglePointCalculator(reactant, energy=energy, forces=forces)

        energy, forces = product.get_potential_energy(), product.get_forces()
        if config_dict["Main"]["Calculator"] == "VaspInteractive": product.calc.finalize()
        product.calc = SinglePointCalculator(product, energy=energy, forces=forces)

        if need_interpolation:
            if default_interp == "ocp_idpp":
                from saddlemill.catsunami.autoframe import interpolate
                images = interpolate(reactant, product, default_num_frames)

            elif default_interp[:4] == "ase_":
                images = [reactant]
                images += [reactant.copy() for ii in range(default_num_frames-2)]
                images += [product]

                neb0 = NEB(images, **config_dict["BaseNEB"])

                if default_interp[4:] == "idpp":
                    perform_aseidpp = True
                else:
                    neb0.interpolate(method="linear", mic=True)

                    # Array of covalent radii for the system
                    radii = np.array([covalent_radii[z] for z in reactant.numbers])
                    radii_sum = radii[:, None] + radii[None, :]

                    for atoms in neb0.images[1:-1]:
                        dists = atoms.get_all_distances(mic=True)
                        np.fill_diagonal(dists, np.inf)

                        if np.any(dists < 0.6 * radii_sum):
                            perform_aseidpp = True
                            break

                if perform_aseidpp:
                    neb0.interpolate(method="idpp", mic=True)

            num_images = len(images)

        if config_dict["Main"]["Calculator"] in ("Vasp", "VaspInteractive"):
            temp_files.extend([f"VASP_{i}_{image_idx}" for image_idx in range(num_images)])

        for image_idx in range(1, num_images-1):
            if config_dict["Main"]["Calculator"] in ("Vasp", "VaspInteractive"):
                images[image_idx].calc = _build_neb_vasp_calc(image_idx, "intermediates")
            else:
                images[image_idx].calc = calc

        is_vasp = config_dict["Main"]["Calculator"] in ("Vasp", "VaspInteractive")
        neb_kwargs = dict(config_dict["BaseNEB"])
        if is_vasp:
            neb_kwargs.setdefault("parallel", True)
            neb_kwargs["allow_shared_calculator"] = False

        total_steps = config_dict["Main"]["steps"]
        fmax = config_dict["Main"]["fmax"]
        max_num_frames = config_dict["ourNEB"]["max_num_frames"]
        if max_num_frames is None:
            max_num_frames = default_num_frames
        can_add_images = not is_vasp and max_num_frames > num_images
        optimizer_kwargs = config_dict[config_dict["Main"]["Optimizer"]]
        endpoint_fmax = config_dict["ourNEB"]["endpoint_relax_fmax"] if default_relax else fmax

        # One-shot event steps (0 = disabled)
        imin_check_step = config_dict["ourNEB"]["intermediate_minima_check_step"]
        add_images_step = config_dict["ourNEB"]["add_images_step"]
        min_depth = config_dict["ourNEB"]["intermediate_minima_min_depth"]

        # Imin relaxation config (shared between one-shot and post-NEB)
        imin_relax_opt_name = config_dict["ourNEB"]["endpoint_relax_Optimizer"] or config_dict["Main"]["Optimizer"]
        imin_relax_kwargs = config_dict[imin_relax_opt_name]
        imin_relax_steps = config_dict["ourNEB"]["endpoint_relax_steps"]

        # --- Create OCPNEB ---
        neb = OCPNEB(
            images,
            batch_size=config_dict["ourNEB"]["batch_size"],
            dneb=config_dict["ourNEB"]["DNEB"],
            vasp=is_vasp,
            initial_imin_set=initial_imin_set,
            frozen_images=frozen_images,
            frozen_fmax=frozen_fmax,
            freeze_fmax=fmax if not is_vasp else None,
            freeze_endpoint_fmax=endpoint_fmax if not is_vasp else None,
            **neb_kwargs,
        )

        opt = Optimizer[1](neb, logfile=temp_log, trajectory=temp_traj, **optimizer_kwargs)

        # --- Phase-based optimization with one-shot events ---
        # Build sorted list of events: [(step, event_type), ...]
        events = []
        if add_images_step > 0 and can_add_images:
            events.append((add_images_step, 'add_images'))
        if imin_check_step > 0:
            events.append((imin_check_step, 'imin'))
        events.sort()

        converged = False
        steps_done = 0

        for event_step, event_type in events:
            run_for = event_step - steps_done
            if run_for > 0 and not converged and steps_done < total_steps:
                nsteps_before = opt.nsteps
                converged = opt.run(fmax=fmax, steps=min(run_for, total_steps - steps_done))
                steps_done += (opt.nsteps - nsteps_before)
            if converged or steps_done >= total_steps:
                break

            if event_type == 'add_images' and not converged and len(neb.images) < max_num_frames:
                expand_result = _expand_band(neb, fmax, max_num_frames, default_num_frames, calc)
                if expand_result is not None:
                    new_images, expand_gaps = expand_result
                    new_frozen = _remap_indices(neb._frozen_set, expand_gaps)
                    new_imin = _remap_indices(neb._imin_set, expand_gaps)
                    new_frozen_cis = _remap_indices(
                        neb._climbing_set & neb._frozen_set, expand_gaps)
                    new_frozen_fmax = {_remap_indices({k}, expand_gaps).pop(): v
                                       for k, v in neb._frozen_fmax_cache.items()}
                    neb = OCPNEB(
                        new_images,
                        batch_size=config_dict["ourNEB"]["batch_size"],
                        dneb=config_dict["ourNEB"]["DNEB"],
                        vasp=is_vasp,
                        initial_imin_set=new_imin,
                        frozen_images=new_frozen,
                        frozen_fmax=new_frozen_fmax,
                        freeze_fmax=fmax if not is_vasp else None,
                        freeze_endpoint_fmax=endpoint_fmax if not is_vasp else None,
                        **neb_kwargs,
                    )
                    neb._climbing_set = new_frozen_cis
                    opt = Optimizer[1](neb, logfile=temp_log, trajectory=temp_traj,
                                      append_trajectory=True, **optimizer_kwargs)
                    print(f"Rank {rank}, structure {i}: added images, band now has {len(neb.images)} images", flush=True)

            elif event_type == 'imin' and not converged:
                new_imin = _detect_imin(neb, min_depth)
                if new_imin:
                    print(f"Rank {rank}, structure {i}: detected imin at {sorted(new_imin)}", flush=True)
                    to_relax = neb._imin_set - neb._frozen_set
                    if to_relax:
                        _relax_imin(neb, to_relax, calc, Optimizer[0], imin_relax_kwargs,
                                    endpoint_fmax, fmax, imin_relax_steps,
                                    str(i), temp_files, rank)
                    neb.cached = False

        # Run remaining steps
        remaining = total_steps - steps_done
        if remaining > 0 and not converged:
            converged = opt.run(fmax=fmax, steps=remaining)

        # --- Post-NEB: dimer CI refinement + imin relaxation + refine ---
        refine_band_steps = config_dict["ourNEB"]["refine_band_steps"]

        if not converged and not is_vasp:
            # Dimer on unconverged CIs — update positions but don't freeze yet
            dimer_converged_cis = set()
            if config_dict["ourNEB"]["dimer_refine_ci"]:
                dimer_refine_steps = config_dict["ourNEB"]["dimer_refine_steps"]
                p_imin = neb._imin_set
                p_climb = neb._climbing_set
                p_bounds = sorted([0] + list(p_imin) + [neb.nimages - 1])

                for seg_s in range(len(p_bounds) - 1):
                    seg_start, seg_end = p_bounds[seg_s], p_bounds[seg_s + 1]
                    seg_ci = _find_segment_ci(seg_start, seg_end, p_climb, neb.energies)
                    if seg_ci is None or neb.image_fmax[seg_ci] < fmax:
                        continue

                    state_d = NEBState(neb, neb.images, neb.energies)
                    tangent = neb.neb_method.get_tangent(
                        state_d, state_d.spring(seg_ci - 1), state_d.spring(seg_ci), seg_ci)

                    ci_atoms = neb.images[seg_ci].copy()
                    ci_atoms.constraints = neb.images[seg_ci].constraints
                    d_log = f'dimer_ci_{i}_img{seg_ci}.log'
                    d_ctrl = f'dimer_ci_control_{i}_img{seg_ci}.log'
                    d_traj = f'dimer_ci_{i}_img{seg_ci}.traj'
                    temp_files.extend([d_log, d_ctrl, d_traj])

                    try:
                        d_atoms, dim_rlx = _setup_dimer(
                            ci_atoms, calc, eigenmode=tangent,
                            dimer_control_kwargs=config_dict.get("DimerControl", {}),
                            control_logfile=d_ctrl, logfile=d_log, trajectory=d_traj)
                        if dim_rlx.run(fmax=fmax, steps=dimer_refine_steps):
                            neb.images[seg_ci].positions[:] = ci_atoms.positions
                            dimer_converged_cis.add(seg_ci)
                            print(f"Rank {rank}, structure {i}: dimer converged CI {seg_ci}", flush=True)
                        else:
                            print(f"Rank {rank}, structure {i}: dimer failed CI {seg_ci}", flush=True)
                    except Exception as e:
                        print(f"Rank {rank}, structure {i}: dimer error CI {seg_ci}: {e}", flush=True)

            # Imin relaxation on unconverged imin
            if neb._imin_set:
                to_relax = neb._imin_set - neb._frozen_set
                if to_relax:
                    _relax_imin(neb, to_relax, calc, Optimizer[0], imin_relax_kwargs,
                                endpoint_fmax, fmax, imin_relax_steps,
                                str(i), temp_files, rank)
                    for img in neb.images[1:-1]:
                        img.calc = calc

            # Recompute forces after dimer/imin position changes, then freeze converged CIs
            if dimer_converged_cis or (neb._imin_set - neb._frozen_set):
                neb.cached = False
                neb.get_forces()  # Proper NEB fmax for repositioned images
            for seg_ci in dimer_converged_cis:
                if neb.image_fmax[seg_ci] < fmax:
                    neb._frozen_set.add(seg_ci)
                    neb._frozen_fmax_cache[seg_ci] = neb.image_fmax[seg_ci]

            # Refine: continue the SAME NEB with more steps (new optimizer only)
            if refine_band_steps > 0:
                t_log = f'neb_refine_{i}.log'
                t_traj = f'neb_refine_{i}.traj'
                temp_files.extend([t_log, t_traj])

                neb.cached = False
                opt_refine = Optimizer[1](neb, logfile=t_log, trajectory=t_traj, **optimizer_kwargs)
                converged = opt_refine.run(fmax=fmax, steps=refine_band_steps)
                print(f"Rank {rank}, structure {i}: refine NEB done (converged={converged})", flush=True)

        if config_dict["Main"]["Calculator"] == "VaspInteractive":
            for img in neb.images[1:-1]:
                img.calc.finalize()

        # Ensure forces/energies are up to date for result extraction
        neb.cached = False
        neb.get_forces()

        # --- Result extraction: per-subband ---
        imin_set = neb._imin_set
        climbing_set = neb._climbing_set
        boundaries = sorted([0] + list(imin_set) + [neb.nimages - 1])
        state = NEBState(neb, neb.images, neb.energies)

        # Set SPC on all images so plot_band() reads cached energies
        # instead of calling (potentially stale) image calculators
        for j in range(neb.nimages):
            neb.images[j].calc = SinglePointCalculator(
                neb.images[j], energy=float(neb.energies[j]),
                forces=neb.real_forces[j])

        nebtools = NEBTools(neb.images)
        fig = nebtools.plot_band()
        fig.savefig(temp_plot)
        plt.close(fig)

        interp_method_out = default_interp if need_interpolation else False
        if isinstance(interp_method_out, str) and interp_method_out.startswith("ase_") and perform_aseidpp:
            interp_method_out = "ase_idpp"

        with Trajectory(my_output_file, 'a') as writer:
            for seg_idx in range(len(boundaries) - 1):
                seg_start = boundaries[seg_idx]
                seg_end = boundaries[seg_idx + 1]

                seg_ci = _find_segment_ci(seg_start, seg_end, climbing_set, neb.energies)

                ci_below_fmax = neb.image_fmax[seg_ci] < fmax if seg_ci is not None else False
                all_below_fmax = all(neb.image_fmax[j] < fmax for j in range(seg_start, seg_end + 1))

                if all_below_fmax:
                    seg_status = "converged"
                elif ci_below_fmax:
                    seg_status = "converged_CI"
                else:
                    seg_status = "not_converged"

                # Compute CI tangent (eigenmode) for the segment
                tangent = None
                seg_barrier = None
                seg_dE = float(neb.energies[seg_end] - neb.energies[seg_start])
                if seg_ci is not None:
                    spring1 = state.spring(seg_ci - 1)
                    spring2 = state.spring(seg_ci)
                    tangent = neb.neb_method.get_tangent(state, spring1, spring2, seg_ci)
                    seg_barrier = float(neb.energies[seg_ci] - neb.energies[seg_start])

                # Write all images in this segment (imin endpoints duplicated across segments)
                for j in range(seg_start, seg_end + 1):
                    img = neb.images[j].copy()
                    img_energy = float(neb.energies[j])
                    img_forces = neb.real_forces[j]

                    if j == 0 or j == neb.nimages - 1:
                        image_type = "endpoint"
                    elif j in imin_set:
                        image_type = "intermediate_minimum"
                    elif j == seg_ci:
                        image_type = "climbing"
                    else:
                        image_type = "regular"

                    img.info['src_index'] = i
                    img.info['image_idx'] = j
                    img.info['subband_idx'] = seg_idx
                    img.info['image_type'] = image_type
                    img.info['effective_fmax'] = float(neb.image_fmax[j])
                    img_threshold = endpoint_fmax if image_type in ("endpoint", "intermediate_minimum") else fmax
                    img.info['image_converged'] = bool(neb.image_fmax[j] < img_threshold)
                    img.info['band_converged'] = bool(all_below_fmax)
                    img.info['band_converged_CI'] = bool(ci_below_fmax)
                    img.info['status'] = seg_status
                    img.info['task_name'] = task_name
                    img.info['nimages'] = len(neb.images)
                    img.info['interpolation_method'] = interp_method_out

                    if j == seg_ci:
                        img.info['eigenmode'] = tangent
                        img.info['barrier'] = seg_barrier
                        img.info['dE'] = seg_dE

                    img.wrap()
                    img.calc = SinglePointCalculator(img, energy=img_energy, forces=img_forces)
                    writer.write(img)

                log_status(seg_status, sub_band_id=seg_idx)

        if consecutive_errors is not None:
            consecutive_errors[0] = 0

        _cleanup_temp_files()

    except Exception as e:
        print(f"Rank {rank} FAILED on structure {i}: {e}", flush=True)
        print(f"\nTraceback details:\n{traceback.format_exc()}", flush=True)
        if consecutive_errors is not None:
            consecutive_errors[0] += 1
        if config_dict["Main"]["Calculator"] == "VaspInteractive":
            from vasp_interactive import VaspInteractive
            for image in images:
                if isinstance(image.calc, VaspInteractive):
                    image.calc.finalize()
        _cleanup_temp_files(error=True)
        log_status(f"error: {str(e)}")
