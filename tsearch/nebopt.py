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
from tsearch.catsunami.ocpneb import OCPNEB
from tsearch.tools import backup_flux_logs


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

    return new_images


def nebopt(i, config_dict, images, calc, Optimizer, consecutive_errors=None, executorlib_worker_id=None, **kwargs):

    rank = executorlib_worker_id

    max_consecutive_errors = config_dict["Main"]["max_consecutive_errors"]
    if consecutive_errors is not None and consecutive_errors[0] >= max_consecutive_errors > 0:
        print(f"Rank {rank}: {consecutive_errors[0]} consecutive structures errored. Killing worker for restart.", flush=True)
        backup_flux_logs(rank)
        sys.exit(1)

    continuation_data = kwargs.get('continuation_data')  # {subband_idx: [Atoms]} or None
    entries_to_run = kwargs.get('entries_to_run')        # set of subband_ids or None
    continue_from_result = config_dict["Main"]["continue_from_result"]

    default_relax = config_dict["ourNEB"]["relax_endpoints"]
    default_interp = config_dict["ourNEB"]["interpolate_method"]
    default_num_frames = config_dict["ourNEB"]["num_frames"]

    # Build list of runs: [(subband_idx_override, run_images, num_frames, relax, interp)]
    # Fresh run: one run with original images and default params.
    # All sub-bands + continue=False: same as fresh (use original input).
    # All sub-bands + continue=True: full band continuation from extracted images.
    # Partial sub-bands: one run per sub-band with targeted images.
    runs = []
    if entries_to_run is not None and continuation_data is not None:
        all_subbands = set(continuation_data.keys())
        is_partial = not (entries_to_run >= all_subbands)

        if is_partial:
            for sb_idx in sorted(entries_to_run):
                if sb_idx not in continuation_data:
                    continue
                sb_imgs = continuation_data[sb_idx]
                if continue_from_result:
                    runs.append((sb_idx, sb_imgs, len(sb_imgs), False, False))
                else:
                    runs.append((sb_idx, [sb_imgs[0], sb_imgs[-1]], len(sb_imgs), False, default_interp))
        elif continue_from_result:
            # All sub-bands, continue: reconstruct full band (de-dup imin at boundaries)
            seen_image_idx = set()
            all_imgs = []
            for sid in sorted(continuation_data.keys()):
                for img in continuation_data[sid]:
                    iidx = img.info.get('image_idx')
                    if iidx not in seen_image_idx:
                        seen_image_idx.add(iidx)
                        all_imgs.append(img)
            runs.append((None, all_imgs, len(all_imgs), False, False))
        # else: all sub-bands + continue=False → fall through to fresh

    if not runs:
        runs.append((None, images, default_num_frames, default_relax, default_interp))

    zip_name = f"{config_dict['Main']['method']}_debug_zips/structure_rank_{rank}_data.zip"
    status_file = f"{config_dict['Main']['method']}_status_csvs/status_rank_{rank}.csv"
    my_output_file = f"{config_dict['Main']['method']}_trajes/collected_ts_rank_{rank}.traj"

    def log_status(status_msg, sub_band_id=0):
        with open(status_file, 'a') as f:
            f.write(f"{i},{rank},{sub_band_id},{status_msg}\n")

    for subband_idx_override, images, num_frames, relax_endpoints, interpolate_method in runs:
        perform_aseidpp = False
        suffix = f"_sub{subband_idx_override}" if subband_idx_override is not None else ""
        temp_log = f'neb_{i}{suffix}.log'
        temp_traj = f'neb_{i}{suffix}.traj'
        temp_plot = f'diffusion_barrier_{i}{suffix}.png'
        temp_react_relax_log = f'reactant_relaxation_{i}{suffix}.log'
        temp_prod_relax_log = f'product_relaxation_{i}{suffix}.log'
        temp_react_relax = f'reactant_relaxation_{i}{suffix}.traj'
        temp_prod_relax = f'product_relaxation_{i}{suffix}.traj'
        temp_files = [temp_log, temp_traj, temp_plot, temp_react_relax_log, temp_prod_relax_log, temp_react_relax, temp_prod_relax]
        if config_dict["Main"]["Calculator"] in ("Vasp", "VaspInteractive"):
            temp_files.extend([f"VASP_{i}{suffix}_{image_idx}" for image_idx in range(num_frames)])

        def _cleanup_temp_files():
            if config_dict["Main"]["Calculator"] in ("Vasp", "VaspInteractive"):
                for image_idx in range(num_frames):
                    for vasp_heavy_files in [f'VASP_{i}{suffix}_{image_idx}/WAVECAR',f'VASP_{i}{suffix}_{image_idx}/CHG',f'VASP_{i}{suffix}_{image_idx}/CHGCAR']:
                        if os.path.exists(vasp_heavy_files): os.remove(vasp_heavy_files)
            existing_files = [f for f in temp_files if os.path.exists(f)]
            if existing_files and config_dict['Main']['zip']:
                with zipfile.ZipFile(zip_name, 'a', zipfile.ZIP_DEFLATED) as zf:
                    for f_name in existing_files:
                        if os.path.isdir(f_name):
                            for root, dirs, files in os.walk(f_name):
                                for file in files:
                                    zf.write(os.path.join(root, file))
                        else:
                            zf.write(f_name, arcname=f_name)
                for f_name in existing_files:
                    if os.path.isdir(f_name):
                        shutil.rmtree(f_name)
                    else:
                        os.remove(f_name)

        try:
            reactant = images[0]
            if config_dict["Main"]["Calculator"] in ("Vasp", "VaspInteractive"):
                reactant.calc = calc(
                    directory=f"VASP_{i}_{0}",
                    command=config_dict["ourNEB"]["vasp_command_endpoints"],
                    ncore=int(config_dict["ourNEB"]["vasp_ncore_endpoints"]),
                    **config_dict["Vasp"],
                    )
            else:
                reactant.calc = calc
    
            product = images[-1]
            if config_dict["Main"]["Calculator"] in ("Vasp", "VaspInteractive"):
                product.calc = calc(
                    directory=f"VASP_{i}_{num_frames-1}",
                    command=config_dict["ourNEB"]["vasp_command_endpoints"],
                    ncore=int(config_dict["ourNEB"]["vasp_ncore_endpoints"]),
                    **config_dict["Vasp"],
                    )
            else:
                product.calc = calc
    
            if relax_endpoints:
                if not interpolate_method: print("Are you sure you want to relax end points while keeping the intermediate images from your traj?", flush=True)
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
    
            if interpolate_method:
                if interpolate_method == "ocp_idpp":
                    # `interpolate` function Meta implemented is very similar to idpp but not sensative to periodic boundary crossings. 
                    # Alternatively you can adopt whatever interpolation scheme you prefer. The `interpolate` function lacks some of the extra protections implemented 
                    # in the `interpolate_and_correct_frames` which is used in the CatTSunami enumeration workflow. Care should be taken to ensure the results are reasonable.
                    # 
                    # IMPORTANT NOTES: 
                    # 1. Make sure the indices in the initial and final frame map to the same atoms
                    # 2. Ensure you have the proper constraints on subsurface atoms
                    # 
                    """
                    The approach uses ase, so you must provide ase.Atoms objects
                    with the appropriate constraints (i.e. fixed subsurface atoms).
                    """
                    from tsearch.catsunami.autoframe import interpolate
                    images = interpolate(reactant, product, num_frames)
    
                elif interpolate_method[:4] == "ase_":
                    images = [reactant]
                    images += [reactant.copy() for i in range(num_frames-2)]
                    images += [product]
    
                    neb0 = NEB(images, **config_dict["BaseNEB"])
    
                    if interpolate_method[4:] == "idpp":
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
    
            for image_idx in range(1,num_frames-1):
                if config_dict["Main"]["Calculator"] in ("Vasp", "VaspInteractive"):
                    images[image_idx].calc = calc(
                        directory=f"VASP_{i}_{image_idx}",
                        command=config_dict["ourNEB"]["vasp_command_intermediates"],
                        ncore=int(config_dict["ourNEB"]["vasp_ncore_intermediates"]),
                        **config_dict["Vasp"],
                        )
                else:
                    images[image_idx].calc = calc
    
            is_vasp = config_dict["Main"]["Calculator"] in ("Vasp", "VaspInteractive")
            neb_kwargs = dict(config_dict["BaseNEB"])
            if is_vasp:
                neb_kwargs.setdefault("parallel", True)
                neb_kwargs["allow_shared_calculator"] = False
    
            use_intermediate_minima = config_dict["ourNEB"]["intermediate_minima"] if subband_idx_override is None else False
            total_steps = config_dict["Main"]["steps"]
            fmax = config_dict["Main"]["fmax"]
            max_num_frames = config_dict["ourNEB"]["max_num_frames"]
            if max_num_frames is None:
                max_num_frames = num_frames
            can_add_images = not is_vasp and max_num_frames > num_frames
            add_images_check_interval = config_dict["ourNEB"]["add_images_check_interval"]
            optimizer_kwargs = config_dict[config_dict["Main"]["Optimizer"]]
    
            neb = OCPNEB(
                images,
                batch_size=config_dict["ourNEB"]["batch_size"],
                dneb=config_dict["ourNEB"]["DNEB"],
                vasp=is_vasp,
                intermediate_minima=use_intermediate_minima,
                intermediate_minima_min_depth=config_dict["ourNEB"]["intermediate_minima_min_depth"],
                intermediate_minima_check_interval=config_dict["ourNEB"]["intermediate_minima_check_interval"],
                **neb_kwargs,
            )
    
            opt = Optimizer[1](neb, logfile=temp_log, trajectory=temp_traj, **optimizer_kwargs)
    
            # Optimization loop with optional image addition
            if can_add_images:
                remaining_steps = total_steps
                converged = False
                while remaining_steps > 0 and not converged:
                    run_for = min(add_images_check_interval, remaining_steps)
                    nsteps_before = opt.nsteps
                    converged = opt.run(fmax=fmax, steps=run_for)
                    remaining_steps -= (opt.nsteps - nsteps_before)
                    if converged or remaining_steps <= 0:
                        break
                    if len(neb.images) < max_num_frames:
                        new_images = _expand_band(neb, fmax, max_num_frames, num_frames, calc)
                        if new_images is not None:
                            neb = OCPNEB(
                                new_images,
                                batch_size=config_dict["ourNEB"]["batch_size"],
                                dneb=config_dict["ourNEB"]["DNEB"],
                                vasp=is_vasp,
                                intermediate_minima=use_intermediate_minima,
                                intermediate_minima_min_depth=config_dict["ourNEB"]["intermediate_minima_min_depth"],
                                intermediate_minima_check_interval=config_dict["ourNEB"]["intermediate_minima_check_interval"],
                                **neb_kwargs,
                            )
                            opt = Optimizer[1](neb, logfile=temp_log, trajectory=temp_traj,
                                              append_trajectory=True, **optimizer_kwargs)
                            print(f"Rank {rank}, structure {i}: added images, band now has {len(neb.images)} images", flush=True)
            else:
                converged = opt.run(fmax=fmax, steps=total_steps)
    
            if config_dict["Main"]["Calculator"] == "VaspInteractive":
                for img in neb.images[1:-1]:
                    img.calc.finalize()
    
            # --- Result extraction: per-subband ---
            imin_set = neb._imin_set
            climbing_set = neb._climbing_set
            boundaries = sorted([0] + list(imin_set) + [neb.nimages - 1])
            state = NEBState(neb, neb.images, neb.energies)
    
            nebtools = NEBTools(neb.images)
            fig = nebtools.plot_band()
            fig.savefig(temp_plot)
            plt.close(fig)
    
            interp_method_out = interpolate_method
            if isinstance(interpolate_method, str) and interpolate_method.startswith("ase_") and perform_aseidpp:
                interp_method_out = "ase_idpp"
    
            with Trajectory(my_output_file, 'a') as writer:
                for seg_idx in range(len(boundaries) - 1):
                    seg_start = boundaries[seg_idx]
                    seg_end = boundaries[seg_idx + 1]
    
                    seg_ci = _find_segment_ci(seg_start, seg_end, climbing_set, neb.energies)
    
                    ci_below_fmax = neb.image_fmax[seg_ci] < fmax if seg_ci is not None else False
                    all_below_fmax = all(neb.image_fmax[j] < fmax for j in range(seg_start, seg_end + 1))
    
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
                        img.calc = SinglePointCalculator(img, energy=neb.energies[j], forces=neb.real_forces[j])
    
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
                        img.info['subband_idx'] = subband_idx_override if subband_idx_override is not None else seg_idx
                        img.info['image_type'] = image_type
                        img.info['effective_fmax'] = float(neb.image_fmax[j])
                        img.info['image_converged'] = bool(neb.image_fmax[j] < fmax)
                        img.info['band_converged'] = bool(all_below_fmax)
                        img.info['band_converged_CI'] = bool(ci_below_fmax)
                        img.info['nimages'] = len(neb.images)
                        img.info['interpolation_method'] = interp_method_out
    
                        if j == seg_ci:
                            img.info['eigenmode'] = tangent
                            img.info['barrier'] = seg_barrier
                            img.info['dE'] = seg_dE
    
                        img.wrap()
                        writer.write(img)
    
                    # Per-subband status
                    out_seg_idx = subband_idx_override if subband_idx_override is not None else seg_idx
                    if all_below_fmax:
                        log_status("converged", sub_band_id=out_seg_idx)
                    elif ci_below_fmax:
                        log_status("converged_CI", sub_band_id=out_seg_idx)
                    else:
                        log_status("not_converged", sub_band_id=out_seg_idx)
    
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
            _cleanup_temp_files()
            log_status("error")

