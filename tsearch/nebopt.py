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


def nebopt(i, config_dict, images, calc, Optimizer, consecutive_errors=None, executorlib_worker_id=None):

    rank = executorlib_worker_id

    max_consecutive_errors = config_dict["Main"]["max_consecutive_errors"]
    if consecutive_errors is not None and consecutive_errors[0] >= max_consecutive_errors > 0:
        print(f"Rank {rank}: {consecutive_errors[0]} consecutive structures errored. Killing worker for restart.", flush=True)
        backup_flux_logs(rank)
        sys.exit(1)

    relax_endpoints = config_dict["ourNEB"]["relax_endpoints"]
    interpolate_method = config_dict["ourNEB"]["interpolate_method"]  # this is idpp implementation from Meta OCP, other choises are "ase_idpp" and "ase_linear" or False if you already have a frame set
    perform_aseidpp = False
    num_frames = config_dict["ourNEB"]["num_frames"]

    # Continuation: band extracted from a previous run — skip interpolation and endpoint relaxation
    if (isinstance(images, list) and len(images) > 0 and
            images[0].info.get("orig_info", images[0].info).get("_continuation", False)):
        relax_endpoints = False
        interpolate_method = False
    zip_name = f"{config_dict['Main']['method']}_debug_zips/structure_rank_{rank}_data.zip"
    status_file = f"{config_dict['Main']['method']}_status_csvs/status_rank_{rank}.csv"
    my_output_file = f"{config_dict['Main']['method']}_trajes/collected_ts_rank_{rank}.traj"
    temp_log = f'neb_{i}.log'
    temp_traj = f'neb_{i}.traj'
    temp_plot = f'diffusion_barrier_{i}.png'
    temp_react_relax_log = f'reactant_relaxation_{i}.log'
    temp_prod_relax_log = f'product_relaxation_{i}.log'
    temp_react_relax = f'reactant_relaxation_{i}.traj'
    temp_prod_relax = f'product_relaxation_{i}.traj'
    temp_files = [temp_log, temp_traj, temp_plot, temp_react_relax_log, temp_prod_relax_log, temp_react_relax, temp_prod_relax]
    if config_dict["Main"]["Calculator"] in ("Vasp", "VaspInteractive"):
        temp_files.extend([f"VASP_{i}_{image_idx}" for image_idx in range(num_frames)])

    def log_status(status_msg, sub_band_id=0):
        with open(status_file, 'a') as f:
            f.write(f"{i},{rank},{sub_band_id},{status_msg}\n")

    def _cleanup_temp_files():
        if config_dict["Main"]["Calculator"] in ("Vasp", "VaspInteractive"):
            for image_idx in range(num_frames):
                for vasp_heavy_files in [f'VASP_{i}_{image_idx}/WAVECAR',f'VASP_{i}_{image_idx}/CHG',f'VASP_{i}_{image_idx}/CHGCAR']:
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

        use_intermediate_minima = config_dict["ourNEB"]["intermediate_minima"]
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
                if seg_ci is None:
                    continue

                ci_image = neb.images[seg_ci].copy()
                energy = neb.energies[seg_ci]
                forces = neb.real_forces[seg_ci]

                spring1 = state.spring(seg_ci - 1)
                spring2 = state.spring(seg_ci)
                tangent = neb.neb_method.get_tangent(state, spring1, spring2, seg_ci)

                seg_barrier = float(neb.energies[seg_ci] - neb.energies[seg_start])
                seg_dE = float(neb.energies[seg_end] - neb.energies[seg_start])
                seg_positions = np.array([neb.images[j].positions for j in range(seg_start, seg_end + 1)])
                seg_energies = [float(neb.energies[j]) for j in range(seg_start, seg_end + 1)]
                seg_fmax = [float(neb.image_fmax[j]) for j in range(seg_start, seg_end + 1)]

                ci_below_fmax = neb.image_fmax[seg_ci] < fmax
                all_below_fmax = all(neb.image_fmax[j] < fmax for j in range(seg_start, seg_end + 1))

                ci_image.info['eigenmode'] = tangent
                ci_image.calc = SinglePointCalculator(ci_image, energy=energy, forces=forces)
                ci_image.info['converged'] = 1 if all_below_fmax else 0
                ci_image.info['ci_converged'] = 1 if ci_below_fmax else 0
                ci_image.info['src_index'] = i
                ci_image.info['segment_id'] = seg_idx
                ci_image.info['barrier'] = seg_barrier
                ci_image.info['dE'] = seg_dE
                ci_image.info['NEB_images'] = seg_positions
                ci_image.info['image_energies'] = seg_energies
                ci_image.info['image_fmax'] = seg_fmax
                ci_image.info['nimages'] = len(neb.images)
                ci_image.info['interpolation_method'] = interp_method_out
                ci_image.wrap()
                writer.write(ci_image)

                # Per-subband status
                if all_below_fmax:
                    log_status("converged", sub_band_id=seg_idx)
                elif ci_below_fmax:
                    log_status("converged_only_CI", sub_band_id=seg_idx)
                else:
                    log_status("not_converged", sub_band_id=seg_idx)

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

