#!/usr/bin/env python3
"""
Comprehensive NEB Analysis Script for Not-Converged Jobs.

Extracts and analyzes neb.log / neb.traj (+ refinement files) from debug zips.
Generates detailed plots and a text report for each not-converged job.

Usage:
    python -m tsearch.analyze_neb [directory]

If directory is not specified, uses the current working directory.
"""

import os
import re
import sys
import glob
import zipfile
import tempfile
import shutil
import csv
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.colors import LogNorm, Normalize
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import matplotlib.ticker as ticker
from ase.io import Trajectory

# ── Configuration ──────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEBUG_ZIP_DIR = os.path.join(BASE_DIR, "NEB_debug_zips")
STATUS_CSV_DIR = os.path.join(BASE_DIR, "NEB_status_csvs")
OUTPUT_DIR = os.path.join(BASE_DIR, "neb_analysis")

FMAX_THRESHOLD = 0.05   # from config.ini
ENDPOINT_FMAX = 0.02    # endpoint_relax_fmax
FROZEN_THRESHOLD = 1e-6  # effective_fmax below this → frozen

# Which job statuses to analyze. Options: "not_converged", "converged", "error"
# Set to None or empty to analyze ALL jobs.
ANALYZE_STATUSES = {"not_converged", "converged"}


# ── Log Parsing ────────────────────────────────────────────────────────────

def parse_optimizer_log(log_text):
    """Parse FIRE/LBFGS/MDMin log into sections.

    Returns list of dicts, each with keys:
        steps, times, energies, fmaxes  (lists)
    """
    lines = log_text.strip().split("\n")
    sections = []
    current = {"steps": [], "times": [], "energies": [], "fmaxes": []}

    for line in lines:
        if "Step" in line and "Time" in line and "Energy" in line:
            if current["steps"]:
                sections.append(current)
                current = {"steps": [], "times": [], "energies": [], "fmaxes": []}
            continue
        m = re.match(r"\w+:\s+(\d+)\s+(\S+)\s+(-?\d+\.\d+)\s+(\d+\.\d+)", line)
        if m:
            current["steps"].append(int(m.group(1)))
            current["times"].append(m.group(2))
            current["energies"].append(float(m.group(3)))
            current["fmaxes"].append(float(m.group(4)))

    if current["steps"]:
        sections.append(current)
    return sections


def parse_dimer_log(log_text):
    """Parse MinModeTranslate dimer log.

    Returns dict with keys: steps, times, energies, fmaxes, curvatures, rot_steps
    Each step appears twice (before/after stepsize), take every other line.
    """
    lines = log_text.strip().split("\n")
    data = {"steps": [], "times": [], "energies": [], "fmaxes": [],
            "curvatures": [], "rot_steps": []}
    seen_steps = set()

    for line in lines:
        if "STEP" in line and "TIME" in line:
            continue
        m = re.match(
            r"MinModeTranslate:\s+(\d+)\s+(\S+)\s+(-?\d+\.\d+)\s+(\d+\.\d+)"
            r"\s+\S+\s+(-?\d+\.\d+)\s+(\d+)",
            line,
        )
        if m:
            step = int(m.group(1))
            key = (step, "after")
            if step in seen_steps:
                # second occurrence (after stepsize) — overwrite
                data["steps"][-1] = step
                data["times"][-1] = m.group(2)
                data["energies"][-1] = float(m.group(3))
                data["fmaxes"][-1] = float(m.group(4))
                data["curvatures"][-1] = float(m.group(5))
                data["rot_steps"][-1] = int(m.group(6))
            else:
                data["steps"].append(step)
                data["times"].append(m.group(2))
                data["energies"].append(float(m.group(3)))
                data["fmaxes"].append(float(m.group(4)))
                data["curvatures"].append(float(m.group(5)))
                data["rot_steps"].append(int(m.group(6)))
                seen_steps.add(step)
    return data


# ── Path Distance Helpers ─────────────────────────────────────────────────

def compute_path_distances(step_data):
    """Compute cumulative path distance (Å) between consecutive NEB images.

    Uses minimum-image convention for periodic systems.  Returns an array of
    length nimages where element 0 is 0.0 and element i is the cumulative
    distance from image 0 to image i.
    """
    positions_list = step_data["positions"]
    cell = step_data["cell"]
    pbc = step_data["pbc"]
    nimages = len(positions_list)
    if nimages < 2:
        return np.array([0.0])

    from ase import Atoms
    cumulative = np.zeros(nimages)
    for i in range(1, nimages):
        diff = positions_list[i] - positions_list[i - 1]
        # Apply minimum image convention for periodic directions
        if np.any(pbc):
            fractional = diff @ np.linalg.inv(cell)
            for d in range(3):
                if pbc[d]:
                    fractional[:, d] -= np.round(fractional[:, d])
            diff = fractional @ cell
        # RMS displacement across all atoms
        cumulative[i] = cumulative[i - 1] + np.sqrt((diff ** 2).sum())
    return cumulative


# ── Trajectory Parsing ─────────────────────────────────────────────────────

def parse_neb_trajectory(traj_path, progress_label=""):
    """Parse NEB trajectory into per-step band snapshots.

    Returns:
        steps: list of dicts with keys:
            nimages, imin_set, climbing_set,
            energies (list[float]), effective_fmax (list[float])
        events: list of dicts with keys:
            step_idx, type, detail
    """
    traj = Trajectory(traj_path, "r")
    total = len(traj)
    if progress_label:
        print(f"  Parsing {progress_label}: {total} frames ...", end="", flush=True)

    steps = []
    events = []
    frame_idx = 0
    prev_nimages = None
    prev_imin = None
    prev_climbing = None

    while frame_idx < total:
        a0 = traj[frame_idx]
        nimages = a0.info.get("nimages", 10)

        # Ensure we don't read past end of trajectory
        end_idx = min(frame_idx + nimages, total)
        actual_count = end_idx - frame_idx

        step_data = {
            "nimages": nimages,
            "imin_set": list(a0.info.get("imin_set", [])),
            "climbing_set": list(a0.info.get("climbing_set", [])),
            "frozen_set": list(a0.info.get("frozen_set", [])),
            "energies": [],
            "effective_fmax": [],
            "positions": [],
            "cell": a0.cell.array.copy(),
            "pbc": a0.pbc.copy(),
        }

        for j in range(actual_count):
            a = traj[frame_idx + j]
            try:
                e = a.get_potential_energy()
            except Exception:
                e = float("nan")
            step_data["energies"].append(e)
            step_data["effective_fmax"].append(
                a.info.get("effective_fmax", float("nan"))
            )
            step_data["positions"].append(a.positions.copy())

        step_idx = len(steps)

        # Detect events
        if prev_nimages is not None and nimages != prev_nimages:
            events.append({
                "step_idx": step_idx,
                "type": "image_addition",
                "detail": f"{prev_nimages} → {nimages} images",
            })

        if prev_imin is not None and step_data["imin_set"] != prev_imin:
            events.append({
                "step_idx": step_idx,
                "type": "imin_change",
                "detail": f"{prev_imin} → {step_data['imin_set']}",
            })

        if prev_climbing is not None and step_data["climbing_set"] != prev_climbing:
            events.append({
                "step_idx": step_idx,
                "type": "climbing_change",
                "detail": f"{prev_climbing} → {step_data['climbing_set']}",
            })

        # Detect newly frozen images
        if steps:
            prev_frozen = set(steps[-1].get("frozen_set", []))
            cur_frozen = set(step_data.get("frozen_set", []))
            for img_i in cur_frozen - prev_frozen:
                if img_i < len(step_data["effective_fmax"]):
                    events.append({
                        "step_idx": step_idx,
                        "type": "freeze",
                        "detail": f"Image {img_i} frozen (efmax={step_data['effective_fmax'][img_i]:.6f})",
                    })

        prev_nimages = nimages
        prev_imin = step_data["imin_set"]
        prev_climbing = step_data["climbing_set"]
        steps.append(step_data)
        frame_idx = end_idx

    traj.close()
    if progress_label:
        print(f" {len(steps)} steps parsed.")
    return steps, events


# ── File Extraction ────────────────────────────────────────────────────────

def extract_job_files(rank, job_id, temp_dir):
    """Extract all files for a job from its rank's debug zip.

    Returns dict mapping file type to path.
    """
    zip_path = os.path.join(DEBUG_ZIP_DIR, f"structure_rank_{rank}_data.zip")
    z = zipfile.ZipFile(zip_path, "r")

    # Find all files for this job
    prefixes = [
        f"neb_{job_id}.", f"neb_refine_{job_id}.",
        f"dimer_ci_{job_id}_", f"dimer_ci_control_{job_id}_",
        f"imin_relax_{job_id}_",
        f"reactant_relaxation_{job_id}.", f"product_relaxation_{job_id}.",
        f"diffusion_barrier_{job_id}.",
    ]
    files = {}
    for info in z.infolist():
        for pfx in prefixes:
            if info.filename.startswith(pfx):
                out_path = os.path.join(temp_dir, info.filename)
                with z.open(info) as src, open(out_path, "wb") as dst:
                    shutil.copyfileobj(src, dst)
                files[info.filename] = out_path
                break
    z.close()
    return files


# ── CSV Parsing ────────────────────────────────────────────────────────────

def load_status_csv():
    """Load all CSV status files and identify not-converged jobs."""
    jobs = {}  # (rank, job_id) -> {subband_id: status}
    for csv_path in sorted(glob.glob(os.path.join(STATUS_CSV_DIR, "status_rank_*.csv"))):
        with open(csv_path) as f:
            for row in csv.reader(f):
                if len(row) < 4:
                    continue
                job_id = int(row[0])
                rank = int(row[1])
                subband_id = int(row[2])
                status = row[3].strip().strip('"')
                key = (rank, job_id)
                if key not in jobs:
                    jobs[key] = {}
                jobs[key][subband_id] = status

    selected = {}
    for key, subbands in jobs.items():
        if ANALYZE_STATUSES:
            matched = [sb for sb, st in subbands.items()
                       if any(cat in st for cat in ANALYZE_STATUSES)]
        else:
            matched = list(subbands.keys())
        if matched:
            selected[key] = {
                "selected": sorted(matched),
                "all_subbands": subbands,
            }
    return selected


# ── Plotting: Figure 1 — NEB Overview ──────────────────────────────────────

def plot_overview(job_id, log_sections, refine_sections, traj_steps, traj_events,
                  refine_steps, refine_events, output_path):
    """4-panel overview figure."""
    fig = plt.figure(figsize=(20, 16))
    fig.suptitle(f"Job {job_id} — NEB Evolution Overview", fontsize=16, fontweight="bold")
    gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)

    # ── Panel A: Band fmax vs global step ──
    ax_a = fig.add_subplot(gs[0, 0])

    # Build global step axis from log sections
    global_steps = []
    global_fmax = []
    global_energy = []
    section_boundaries = []
    cumulative = 0
    for sec in log_sections:
        for s, fm, en in zip(sec["steps"], sec["fmaxes"], sec["energies"]):
            global_steps.append(cumulative + s)
            global_fmax.append(fm)
            global_energy.append(en)
        cumulative += sec["steps"][-1] + 1 if sec["steps"] else cumulative
        section_boundaries.append(cumulative)

    main_end = cumulative

    # Add refine sections
    for sec in refine_sections:
        for s, fm, en in zip(sec["steps"], sec["fmaxes"], sec["energies"]):
            global_steps.append(main_end + s)
            global_fmax.append(fm)
            global_energy.append(en)

    ax_a.semilogy(global_steps, global_fmax, "k-", linewidth=0.5, alpha=0.8)
    ax_a.axhline(FMAX_THRESHOLD, color="r", linestyle="--", linewidth=1, label=f"fmax={FMAX_THRESHOLD}")
    ax_a.axvline(main_end, color="purple", linestyle="-.", linewidth=1.5, alpha=0.7, label="Refine start")

    # Mark image addition events
    for ev in traj_events:
        if ev["type"] == "image_addition":
            # Map step_idx to global step (approximate)
            ax_a.axvline(ev["step_idx"], color="blue", linestyle=":", linewidth=1, alpha=0.5)
            ax_a.annotate(ev["detail"], (ev["step_idx"], max(global_fmax) * 0.8),
                          fontsize=7, rotation=90, color="blue", va="top")

    for sb in section_boundaries[:-1]:
        ax_a.axvline(sb, color="gray", linestyle=":", linewidth=0.5, alpha=0.3)

    ax_a.set_xlabel("Global Step")
    ax_a.set_ylabel("Band fmax (eV/Å)", color="black")
    ax_a.set_title("(A) Band fmax & emax vs Step")
    ax_a.legend(fontsize=8, loc="upper right")
    ax_a.grid(True, alpha=0.3)

    # Overlay emax (max image energy) on twin y-axis
    ax_a2 = ax_a.twinx()
    ax_a2.plot(global_steps, global_energy, color="tab:blue", linewidth=0.5, alpha=0.6)
    ax_a2.set_ylabel("emax (eV)", color="tab:blue")
    ax_a2.tick_params(axis="y", labelcolor="tab:blue")

    # ── Panel B: Per-image fmax heatmap (main + refine) ──
    ax_b = fig.add_subplot(gs[0, 1])

    all_heatmap_steps = list(traj_steps) + list(refine_steps)
    main_nsteps = len(traj_steps)
    total_nsteps = len(all_heatmap_steps)
    max_nimages = max(s["nimages"] for s in all_heatmap_steps) if all_heatmap_steps else 1

    # Sample if too many steps (>2000) for readability
    sample_rate = max(1, total_nsteps // 2000)
    sampled_indices = list(range(0, total_nsteps, sample_rate))
    n_sampled = len(sampled_indices)

    heatmap = np.full((max_nimages, n_sampled), np.nan)
    for col, si in enumerate(sampled_indices):
        s = all_heatmap_steps[si]
        frozen = set(s.get("frozen_set", []))
        for img_i, fm in enumerate(s["effective_fmax"]):
            if img_i not in frozen:
                heatmap[img_i, col] = fm
            else:
                heatmap[img_i, col] = FROZEN_THRESHOLD  # mark frozen as minimum

    # Use log scale
    vmin = max(FROZEN_THRESHOLD, np.nanmin(heatmap[heatmap > FROZEN_THRESHOLD])) if np.any(heatmap > FROZEN_THRESHOLD) else 1e-4
    vmax = np.nanmax(heatmap) if np.any(~np.isnan(heatmap)) else 10.0

    im = ax_b.imshow(
        heatmap, aspect="auto", origin="lower",
        norm=LogNorm(vmin=max(vmin, 1e-4), vmax=max(vmax, 1.0)),
        cmap="hot_r",
        extent=[0, total_nsteps, -0.5, max_nimages - 0.5],
    )
    plt.colorbar(im, ax=ax_b, label="effective fmax (eV/Å)")

    # Mark refine start
    if refine_steps:
        ax_b.axvline(main_nsteps, color="purple", linestyle="-.", linewidth=1.5, alpha=0.7)

    # Overlay imin and CI markers (sampled), color by convergence
    marker_sample = max(1, n_sampled // 200)
    for col_i, si in enumerate(sampled_indices):
        if col_i % marker_sample != 0:
            continue
        s = all_heatmap_steps[si]
        frozen_set = set(s.get("frozen_set", []))
        for im_idx in s.get("imin_set", []):
            if im_idx < len(s["effective_fmax"]):
                frozen = im_idx in frozen_set
                color = "darkblue" if frozen else "lightskyblue"
                ax_b.plot(si, im_idx, "s", color=color, markersize=2, alpha=0.8)
        for ci_idx in s.get("climbing_set", []):
            if ci_idx < len(s["effective_fmax"]):
                frozen = ci_idx in frozen_set
                color = "black" if frozen else "gray"
                ax_b.plot(si, ci_idx, "^", color=color, markersize=2, alpha=0.8)

    ax_b.set_xlabel("Step")
    ax_b.set_ylabel("Image Index")
    ax_b.set_title("(B) Per-Image fmax Heatmap (main + refine)")

    legend_elements = [
        Patch(facecolor="darkred", label="High fmax"),
        Patch(facecolor="lightyellow", label="Low fmax"),
        Line2D([0], [0], marker="s", color="lightskyblue", linestyle="None", markersize=5, label="Imin"),
        Line2D([0], [0], marker="s", color="darkblue", linestyle="None", markersize=5, label="Imin (frozen)"),
        Line2D([0], [0], marker="^", color="gray", linestyle="None", markersize=5, label="CI"),
        Line2D([0], [0], marker="^", color="black", linestyle="None", markersize=5, label="CI (frozen)"),
    ]
    ax_b.legend(handles=legend_elements, fontsize=7, loc="upper left")

    # ── Panel C: Band energy profiles at key stages ──
    ax_c = fig.add_subplot(gs[1, 0])

    # Select key steps: first, each image addition, midpoint, last main, last refine
    key_step_indices = [0]
    for ev in traj_events:
        if ev["type"] == "image_addition":
            key_step_indices.append(ev["step_idx"])
    if len(traj_steps) > 2:
        key_step_indices.append(len(traj_steps) // 2)
    key_step_indices.append(len(traj_steps) - 1)

    # Add refine steps if available
    refine_key = []
    if refine_steps:
        refine_key = [len(refine_steps) - 1]

    colors_main = plt.cm.viridis(np.linspace(0, 0.8, len(key_step_indices)))
    for ci, si in enumerate(key_step_indices):
        s = traj_steps[si]
        energies = np.array(s["energies"])
        ref_e = energies[0]
        x = compute_path_distances(s)
        label = f"Step {si}"
        if si == 0:
            label = "Initial"
        elif si == len(traj_steps) - 1:
            label = f"Final main (step {si})"
        ax_c.plot(x, energies - ref_e, "o-", color=colors_main[ci],
                  markersize=3, linewidth=1, label=label)

        # Mark imin and CI
        for im_idx in s.get("imin_set", []):
            if im_idx < len(energies):
                ax_c.plot(x[im_idx], energies[im_idx] - ref_e, "bs", markersize=6, zorder=5)
        for ci_idx in s.get("climbing_set", []):
            if ci_idx < len(energies):
                ax_c.plot(x[ci_idx], energies[ci_idx] - ref_e, "r^", markersize=6, zorder=5)

    # Plot final refine step
    if refine_steps and refine_key:
        s = refine_steps[refine_key[0]]
        energies = np.array(s["energies"])
        ref_e = energies[0]
        x = compute_path_distances(s)
        ax_c.plot(x, energies - ref_e, "o-", color="magenta",
                  markersize=3, linewidth=1.5, label=f"Final refine (step {refine_key[0]})")

    ax_c.set_xlabel("Path distance (Å)")
    ax_c.set_ylabel("Energy − E₀ (eV)")
    ax_c.set_title("(C) Band Energy Profiles at Key Stages")
    ax_c.legend(fontsize=7, loc="best")
    ax_c.grid(True, alpha=0.3)

    # ── Panel D: Counts over time ──
    ax_d = fig.add_subplot(gs[1, 1])

    nsteps = len(traj_steps)
    step_x = np.arange(nsteps)
    nimages_arr = np.array([s["nimages"] for s in traj_steps])
    imin_count = np.array([len(s.get("imin_set", [])) for s in traj_steps])
    ci_count = np.array([len(s.get("climbing_set", [])) for s in traj_steps])
    frozen_count = np.array([
        len(s.get("frozen_set", []))
        for s in traj_steps
    ])

    ax_d.plot(step_x, nimages_arr, "k-", linewidth=1.5, label="Total images")
    ax_d.plot(step_x, imin_count, "b-", linewidth=1, label="Imin count")
    ax_d.plot(step_x, ci_count, "r-", linewidth=1, label="CI count")
    ax_d.plot(step_x, frozen_count, "g-", linewidth=1, label="Frozen count")

    ax_d.set_xlabel("Step")
    ax_d.set_ylabel("Count")
    ax_d.set_title("(D) Image Classification Over Time")
    ax_d.legend(fontsize=8)
    ax_d.grid(True, alpha=0.3)

    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


# ── Plotting: Figure 2 — Per-Image Fmax Trajectories ──────────────────────

def plot_per_image_fmax(job_id, traj_steps, traj_events, output_path):
    """Grid of subplots, one per image, showing fmax evolution."""
    max_nimages = max(s["nimages"] for s in traj_steps)
    nsteps = len(traj_steps)

    ncols = min(5, max_nimages)
    nrows = (max_nimages + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows),
                             sharex=True, squeeze=False)
    fig.suptitle(f"Job {job_id} — Per-Image Fmax Evolution (main NEB)", fontsize=14, fontweight="bold")

    # Build per-image fmax arrays
    for img_i in range(max_nimages):
        row, col = divmod(img_i, ncols)
        ax = axes[row][col]

        step_vals = []
        fmax_vals = []
        colors = []

        for si, s in enumerate(traj_steps):
            if img_i < len(s["effective_fmax"]):
                fm = s["effective_fmax"][img_i]
                step_vals.append(si)
                fmax_vals.append(fm)

                # Determine type
                if img_i in set(s.get("frozen_set", [])):
                    colors.append("green")
                elif img_i in s.get("imin_set", []):
                    colors.append("blue")
                elif img_i in s.get("climbing_set", []):
                    colors.append("red")
                elif img_i == 0 or img_i == s["nimages"] - 1:
                    colors.append("gray")
                else:
                    colors.append("black")

        if not step_vals:
            ax.set_visible(False)
            continue

        # Plot as line with colored segments
        step_arr = np.array(step_vals)
        fmax_arr = np.array(fmax_vals)

        # Background shading by type (simplified: just plot colored dots)
        ax.semilogy(step_arr, np.maximum(fmax_arr, 1e-6), "k-", linewidth=0.3, alpha=0.5)

        # Color scatter by type
        for s_v, f_v, c in zip(step_vals[::max(1, len(step_vals)//500)],
                                fmax_vals[::max(1, len(fmax_vals)//500)],
                                colors[::max(1, len(colors)//500)]):
            ax.plot(s_v, max(f_v, 1e-6), ".", color=c, markersize=1)

        ax.axhline(FMAX_THRESHOLD, color="r", linestyle="--", linewidth=0.5, alpha=0.5)
        ax.axhline(ENDPOINT_FMAX, color="orange", linestyle=":", linewidth=0.5, alpha=0.5)
        ax.set_title(f"Image {img_i}", fontsize=9)
        ax.set_ylim(1e-4, None)
        ax.grid(True, alpha=0.2)

    # Hide unused axes
    for idx in range(max_nimages, nrows * ncols):
        row, col = divmod(idx, ncols)
        axes[row][col].set_visible(False)

    # Common labels
    for ax in axes[-1]:
        if ax.get_visible():
            ax.set_xlabel("Step", fontsize=8)
    for ax in axes[:, 0]:
        ax.set_ylabel("fmax", fontsize=8)

    legend_elements = [
        Line2D([0], [0], marker=".", color="gray", linestyle="None", label="Endpoint"),
        Line2D([0], [0], marker=".", color="black", linestyle="None", label="Regular"),
        Line2D([0], [0], marker=".", color="blue", linestyle="None", label="Imin"),
        Line2D([0], [0], marker=".", color="red", linestyle="None", label="CI"),
        Line2D([0], [0], marker=".", color="green", linestyle="None", label="Frozen"),
    ]
    fig.legend(handles=legend_elements, loc="upper right", fontsize=9)

    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


# ── Plotting: Figure 3 — Post-NEB Refinement ──────────────────────────────

def plot_refinement(job_id, files, temp_dir, output_path):
    """Plot dimer CI refinement, imin relaxation, and refine NEB results.

    Sections are visually separated: dimer CI (top), imin relaxation (middle),
    refine NEB (bottom, full width). In dimer panels, curvature=0 is aligned
    with the fmax threshold so both lines below one reference = converged.
    """
    dimer_files = sorted([
        (fn, fp) for fn, fp in files.items()
        if fn.startswith(f"dimer_ci_{job_id}_img") and fn.endswith(".log")
        and "control" not in fn
    ])
    imin_files = sorted([
        (fn, fp) for fn, fp in files.items()
        if fn.startswith(f"imin_relax_{job_id}_img") and fn.endswith(".log")
    ])
    refine_log_fn = f"neb_refine_{job_id}.log"
    has_refine = refine_log_fn in files

    n_dimer = len(dimer_files)
    n_imin = len(imin_files)

    if n_dimer + n_imin == 0 and not has_refine:
        return

    ncols = min(3, max(n_dimer, n_imin, 1))

    dimer_rows = (n_dimer + ncols - 1) // ncols if n_dimer > 0 else 0
    imin_rows = (n_imin + ncols - 1) // ncols if n_imin > 0 else 0
    refine_rows = 1 if has_refine else 0

    total_rows = dimer_rows + imin_rows + refine_rows
    if total_rows == 0:
        return

    fig = plt.figure(figsize=(5 * ncols, 4 * total_rows + 1))
    fig.suptitle(f"Job {job_id} — Post-NEB Refinement", fontsize=14, fontweight="bold")

    gs = GridSpec(total_rows, ncols, figure=fig, hspace=0.7, wspace=0.45)

    row_offset = 0

    # ── Dimer CI section ──
    for i, (fn, fp) in enumerate(dimer_files):
        r, c = divmod(i, ncols)
        ax = fig.add_subplot(gs[row_offset + r, c])

        img_num = re.search(r"img(\d+)", fn).group(1)
        with open(fp) as f:
            data = parse_dimer_log(f.read())

        if data["steps"]:
            # Left axis: fmax (log scale, red)
            ax.semilogy(data["steps"], data["fmaxes"], "r-", linewidth=1, label="fmax")

            # Right axis: curvature (linear, blue)
            ax2 = ax.twinx()
            ax2.plot(data["steps"], data["curvatures"], "b-", linewidth=0.8, alpha=0.7,
                     label="curvature")
            ax2.set_ylabel("Curvature (eV/Å²)", fontsize=8, color="b")
            ax2.tick_params(axis="y", labelcolor="b")

            # Align curvature=0 with FMAX_THRESHOLD on the log axis so that
            # both red and blue lines below one reference line = converged.
            # Lock fmax limits to prevent render-time re-snapping.
            ymin_log, ymax_log = ax.get_ylim()
            ymin_log = min(ymin_log, FMAX_THRESHOLD * 0.5)
            ymax_log = max(ymax_log, FMAX_THRESHOLD * 2)
            ax.set_ylim(ymin_log, ymax_log)

            log_range = np.log10(ymax_log) - np.log10(ymin_log)
            frac = (np.log10(FMAX_THRESHOLD) - np.log10(ymin_log)) / log_range

            curv_vals = data["curvatures"]
            need_below = max(abs(min(min(curv_vals), -0.05)), 0.05)
            need_above = max(max(curv_vals), 0.05)
            scale = max(need_below / frac, need_above / (1 - frac)) * 1.1
            ax2.set_ylim(-frac * scale, (1 - frac) * scale)

            # Single convergence reference line (fmax threshold + curvature=0)
            ax.axhline(FMAX_THRESHOLD, color="gray", linestyle="--", linewidth=1.2,
                       alpha=0.6, label=f"fmax={FMAX_THRESHOLD} / curv=0")

            final_fm = data["fmaxes"][-1]
            final_curv = data["curvatures"][-1]
            conv_fm = final_fm < FMAX_THRESHOLD
            conv_curv = final_curv < 0
            converged = "YES" if conv_fm and conv_curv else "NO"
            ax.set_title(
                f"Dimer CI img{img_num}\n"
                f"fmax={final_fm:.4f}, curv={final_curv:.3f} [{converged}]",
                fontsize=8)
        else:
            ax.set_title(f"Dimer CI img{img_num}\n(no data)", fontsize=8)

        ax.set_xlabel("Step", fontsize=8)
        ax.set_ylabel("fmax (eV/Å)", fontsize=8, color="r")
        ax.tick_params(axis="y", labelcolor="r")
        ax.grid(True, alpha=0.2)

    row_offset += dimer_rows

    # ── Imin relaxation section ──
    for i, (fn, fp) in enumerate(imin_files):
        r, c = divmod(i, ncols)
        ax = fig.add_subplot(gs[row_offset + r, c])

        img_num = re.search(r"img(\d+)", fn).group(1)
        with open(fp) as f:
            sections = parse_optimizer_log(f.read())

        if sections and sections[0]["steps"]:
            sec = sections[0]
            ax.semilogy(sec["steps"], sec["fmaxes"], "b-", linewidth=1)
            ax.axhline(ENDPOINT_FMAX, color="orange", linestyle="--", linewidth=0.5,
                        alpha=0.5, label=f"endpoint_fmax={ENDPOINT_FMAX}")
            final_fm = sec["fmaxes"][-1]
            converged = "YES" if final_fm < ENDPOINT_FMAX else "NO"
            ax.set_title(f"Imin relax img{img_num}\nfinal fmax={final_fm:.4f} [{converged}]",
                         fontsize=8)
        else:
            ax.set_title(f"Imin relax img{img_num}\n(no data)", fontsize=8)

        ax.set_xlabel("Step", fontsize=8)
        ax.set_ylabel("fmax (eV/Å)", fontsize=8)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.2)

    row_offset += imin_rows

    # ── Refine NEB section (full width, visually separated) ──
    if has_refine:
        ax = fig.add_subplot(gs[row_offset, :])

        with open(files[refine_log_fn]) as f:
            sections = parse_optimizer_log(f.read())

        all_steps = []
        all_fmax = []
        offset = 0
        for sec in sections:
            for s, fm in zip(sec["steps"], sec["fmaxes"]):
                all_steps.append(offset + s)
                all_fmax.append(fm)
            if sec["steps"]:
                offset += sec["steps"][-1] + 1

        if all_steps:
            ax.semilogy(all_steps, all_fmax, "m-", linewidth=1)
            ax.axhline(FMAX_THRESHOLD, color="r", linestyle="--", linewidth=0.5, alpha=0.5)
            ax.set_title(f"Refine NEB\nfinal fmax={all_fmax[-1]:.4f}, {len(all_steps)} steps",
                         fontsize=8)

        ax.set_xlabel("Step", fontsize=8)
        ax.set_ylabel("fmax (eV/Å)", fontsize=8)
        ax.grid(True, alpha=0.2)

    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


# ── Plotting: Figure 4 — Energy Landscape Evolution ───────────────────────

def plot_energy_evolution(job_id, traj_steps, refine_steps, output_path):
    """Band energy profiles at regular intervals, overlaid."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(f"Job {job_id} — Energy Landscape Evolution", fontsize=14, fontweight="bold")

    # Main NEB
    nsteps = len(traj_steps)
    n_profiles = min(30, nsteps)
    indices = np.linspace(0, nsteps - 1, n_profiles, dtype=int)

    cmap = plt.cm.viridis
    for ci, si in enumerate(indices):
        alpha = 0.2 + 0.8 * (ci / max(1, len(indices) - 1))
        s = traj_steps[si]
        energies = np.array(s["energies"])
        ref_e = energies[0]
        x = compute_path_distances(s)
        color = cmap(ci / max(1, len(indices) - 1))
        ax1.plot(x, energies - ref_e, "-", color=color, alpha=alpha, linewidth=0.8)

    # Mark final state more prominently
    s = traj_steps[-1]
    energies = np.array(s["energies"])
    ref_e = energies[0]
    x = compute_path_distances(s)
    ax1.plot(x, energies - ref_e, "ko-", linewidth=2, markersize=4, label="Final", zorder=10)
    for im_idx in s.get("imin_set", []):
        if im_idx < len(energies):
            ax1.plot(x[im_idx], energies[im_idx] - ref_e, "bs", markersize=8, zorder=11)
    for ci_idx in s.get("climbing_set", []):
        if ci_idx < len(energies):
            ax1.plot(x[ci_idx], energies[ci_idx] - ref_e, "r^", markersize=8, zorder=11)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=Normalize(vmin=0, vmax=nsteps))
    plt.colorbar(sm, ax=ax1, label="Step")
    ax1.set_xlabel("Path distance (Å)")
    ax1.set_ylabel("Energy − E₀ (eV)")
    ax1.set_title("Main NEB")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    # Refine NEB
    if refine_steps:
        nsteps_r = len(refine_steps)
        n_profiles_r = min(30, nsteps_r)
        indices_r = np.linspace(0, nsteps_r - 1, n_profiles_r, dtype=int)

        for ci, si in enumerate(indices_r):
            alpha = 0.2 + 0.8 * (ci / max(1, len(indices_r) - 1))
            s = refine_steps[si]
            energies = np.array(s["energies"])
            ref_e = energies[0]
            x = compute_path_distances(s)
            color = cmap(ci / max(1, len(indices_r) - 1))
            ax2.plot(x, energies - ref_e, "-", color=color, alpha=alpha, linewidth=0.8)

        s = refine_steps[-1]
        energies = np.array(s["energies"])
        ref_e = energies[0]
        x = compute_path_distances(s)
        ax2.plot(x, energies - ref_e, "ko-", linewidth=2, markersize=4, label="Final", zorder=10)

        sm2 = plt.cm.ScalarMappable(cmap=cmap, norm=Normalize(vmin=0, vmax=nsteps_r))
        plt.colorbar(sm2, ax=ax2, label="Step")
        ax2.set_title("Refine NEB")
    else:
        ax2.text(0.5, 0.5, "No refine data", transform=ax2.transAxes, ha="center")
        ax2.set_title("Refine NEB (N/A)")

    ax2.set_xlabel("Path distance (Å)")
    ax2.set_ylabel("Energy − E₀ (eV)")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


# ── Plotting: Figure 5 — Fmax per image with type timeline ────────────────

def plot_fmax_and_type_timeline(job_id, traj_steps, traj_events, output_path):
    """Combined plot: top = stacked area of image types, bottom = per-image fmax lines."""
    nsteps = len(traj_steps)
    max_nimages = max(s["nimages"] for s in traj_steps)

    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(16, 10), sharex=True,
                                          gridspec_kw={"height_ratios": [1, 2]})
    fig.suptitle(f"Job {job_id} — Image Types & Fmax Timeline", fontsize=14, fontweight="bold")

    # ── Top: stacked counts ──
    step_x = np.arange(nsteps)
    endpoint_c = np.zeros(nsteps)
    regular_c = np.zeros(nsteps)
    imin_c = np.zeros(nsteps)
    ci_c = np.zeros(nsteps)
    frozen_c = np.zeros(nsteps)

    for si, s in enumerate(traj_steps):
        nim = s["nimages"]
        imin_set = set(s.get("imin_set", []))
        ci_set = set(s.get("climbing_set", []))
        frozen_set = set(s.get("frozen_set", []))
        for img_i in range(nim):
            fm = s["effective_fmax"][img_i] if img_i < len(s["effective_fmax"]) else 0
            if img_i in frozen_set and img_i > 0 and img_i < nim - 1:
                frozen_c[si] += 1
            elif img_i == 0 or img_i == nim - 1:
                endpoint_c[si] += 1
            elif img_i in imin_set:
                imin_c[si] += 1
            elif img_i in ci_set:
                ci_c[si] += 1
            else:
                regular_c[si] += 1

    ax_top.stackplot(step_x, endpoint_c, regular_c, imin_c, ci_c, frozen_c,
                     labels=["Endpoint", "Regular", "Imin", "CI", "Frozen"],
                     colors=["gray", "lightblue", "blue", "red", "green"],
                     alpha=0.7)
    ax_top.set_ylabel("Image Count")
    ax_top.set_title("Image Type Distribution")
    ax_top.legend(fontsize=8, loc="upper left")
    ax_top.grid(True, alpha=0.2)

    # ── Bottom: per-image fmax lines ──
    # Use a color per image, with type indicated by linestyle
    cmap = plt.cm.tab20
    sample_rate = max(1, nsteps // 1500)

    for img_i in range(max_nimages):
        steps_i = []
        fmax_i = []
        for si in range(0, nsteps, sample_rate):
            s = traj_steps[si]
            if img_i < len(s["effective_fmax"]):
                fm = s["effective_fmax"][img_i]
                if img_i not in set(s.get("frozen_set", [])):  # skip frozen for clarity
                    steps_i.append(si)
                    fmax_i.append(fm)

        if steps_i:
            color = cmap(img_i % 20 / 20)
            is_endpoint = img_i == 0  # only first endpoint shown
            lw = 0.5 if img_i > 0 else 1.0
            ax_bot.semilogy(steps_i, fmax_i, "-", color=color, linewidth=lw,
                            alpha=0.6, label=f"img {img_i}" if img_i < 20 else None)

    ax_bot.axhline(FMAX_THRESHOLD, color="r", linestyle="--", linewidth=1, label=f"fmax={FMAX_THRESHOLD}")
    ax_bot.axhline(ENDPOINT_FMAX, color="orange", linestyle=":", linewidth=0.8, label=f"endpoint_fmax={ENDPOINT_FMAX}")

    # Mark events
    for ev in traj_events:
        if ev["type"] == "image_addition":
            ax_bot.axvline(ev["step_idx"], color="blue", linestyle=":", linewidth=1, alpha=0.5)
        elif ev["type"] == "freeze":
            ax_bot.axvline(ev["step_idx"], color="green", linestyle=":", linewidth=0.5, alpha=0.3)

    ax_bot.set_xlabel("Step")
    ax_bot.set_ylabel("effective fmax (eV/Å)")
    ax_bot.set_title("Per-Image Fmax (frozen excluded)")
    if max_nimages <= 20:
        ax_bot.legend(fontsize=6, ncol=4, loc="upper right")
    ax_bot.grid(True, alpha=0.2)

    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


# ── Text Report ────────────────────────────────────────────────────────────

def generate_text_report(job_id, rank, subbands_info, files, log_sections,
                         refine_sections, traj_steps, traj_events,
                         refine_steps, refine_events):
    """Generate detailed text report of NEB evolution."""
    lines = []
    lines.append("=" * 80)
    lines.append(f"  JOB {job_id} (Rank {rank}) — Detailed NEB Analysis")
    lines.append("=" * 80)
    lines.append("")

    # Sub-band status
    lines.append("Sub-band Status:")
    for sb_id, status in sorted(subbands_info.items()):
        marker = "  ✗" if "not_converged" in status else "  ✓"
        lines.append(f"  {marker} Sub-band {sb_id}: {status}")
    lines.append("")

    # ── Endpoint relaxation ──
    lines.append("─" * 60)
    lines.append("PHASE 0: Endpoint Relaxation")
    lines.append("─" * 60)
    for ep_type in ["reactant", "product"]:
        log_fn = f"{ep_type}_relaxation_{job_id}.log"
        if log_fn in files:
            with open(files[log_fn]) as f:
                sections = parse_optimizer_log(f.read())
            if sections and sections[0]["fmaxes"]:
                sec = sections[0]
                lines.append(f"  {ep_type.capitalize()}:")
                lines.append(f"    Steps: {len(sec['steps'])}")
                lines.append(f"    Initial fmax: {sec['fmaxes'][0]:.6f}")
                lines.append(f"    Final fmax:   {sec['fmaxes'][-1]:.6f}")
                lines.append(f"    Converged:    {'YES' if sec['fmaxes'][-1] < ENDPOINT_FMAX else 'NO'}")
    lines.append("")

    # ── Main NEB ──
    lines.append("─" * 60)
    lines.append("PHASE 1: Main NEB Optimization")
    lines.append("─" * 60)

    total_steps = sum(len(sec["steps"]) for sec in log_sections)
    lines.append(f"  Total optimizer steps: {total_steps}")
    lines.append(f"  Log sections (restarts): {len(log_sections)}")

    for si, sec in enumerate(log_sections):
        lines.append(f"  Section {si}: {len(sec['steps'])} steps, "
                      f"fmax {sec['fmaxes'][0]:.4f} → {sec['fmaxes'][-1]:.4f}, "
                      f"energy {sec['energies'][0]:.4f} → {sec['energies'][-1]:.4f}")
    lines.append("")

    # Trajectory analysis
    if traj_steps:
        s0 = traj_steps[0]
        sf = traj_steps[-1]
        lines.append(f"  Initial band: {s0['nimages']} images")
        lines.append(f"  Final band:   {sf['nimages']} images")
        lines.append(f"  Total traj steps: {len(traj_steps)}")
        lines.append("")

        # Initial state
        lines.append(f"  Step 0:")
        lines.append(f"    nimages={s0['nimages']}, imin_set={s0['imin_set']}, climbing_set={s0['climbing_set']}")
        lines.append(f"    Per-image fmax: {[f'{fm:.4f}' for fm in s0['effective_fmax']]}")
        lines.append(f"    Per-image energies: {[f'{e:.3f}' for e in s0['energies']]}")
        lines.append("")

        # Events
        lines.append("  Events:")
        if not traj_events:
            lines.append("    (none)")
        for ev in traj_events:
            lines.append(f"    Step {ev['step_idx']:5d}: [{ev['type']:18s}] {ev['detail']}")
        lines.append("")

        # Final state of main NEB
        lines.append(f"  Final main NEB state (step {len(traj_steps)-1}):")
        lines.append(f"    nimages={sf['nimages']}, imin_set={sf['imin_set']}, climbing_set={sf['climbing_set']}")
        lines.append(f"    Per-image fmax: {[f'{fm:.4f}' for fm in sf['effective_fmax']]}")

        frozen_imgs = sf.get("frozen_set", [])
        lines.append(f"    Frozen images: {frozen_imgs} ({len(frozen_imgs)} total)")

        # Per-image convergence
        lines.append("    Per-image convergence:")
        frozen_set = set(frozen_imgs)
        for img_i, fm in enumerate(sf["effective_fmax"]):
            img_type = "endpoint"
            if img_i in sf.get("imin_set", []):
                img_type = "imin"
            elif img_i in sf.get("climbing_set", []):
                img_type = "CI"
            elif img_i == 0 or img_i == sf["nimages"] - 1:
                img_type = "endpoint"
            elif img_i in frozen_set:
                img_type = "frozen"
            else:
                img_type = "regular"

            threshold = ENDPOINT_FMAX if img_type in ("endpoint", "imin") else FMAX_THRESHOLD
            status = "CONVERGED" if fm < threshold else f"NOT CONVERGED (need < {threshold})"
            if img_i in frozen_set:
                status = "FROZEN"
            lines.append(f"      Image {img_i:2d} [{img_type:8s}]: fmax={fm:.6f}  {status}")

        # Sub-band analysis
        lines.append("")
        lines.append("    Sub-band convergence analysis:")
        imin_set = sf.get("imin_set", [])
        climbing_set = sf.get("climbing_set", [])
        boundaries = sorted([0] + imin_set + [sf["nimages"] - 1])

        for seg_i in range(len(boundaries) - 1):
            seg_start = boundaries[seg_i]
            seg_end = boundaries[seg_i + 1]
            seg_images = list(range(seg_start, seg_end + 1))

            # Find CI for this segment
            seg_ci = None
            for ci in climbing_set:
                if seg_start < ci < seg_end:
                    seg_ci = ci
                    break

            seg_fmaxes = []
            for j in seg_images:
                if j < len(sf["effective_fmax"]):
                    seg_fmaxes.append((j, sf["effective_fmax"][j]))

            all_below = all(
                fm < (ENDPOINT_FMAX if (j in imin_set or j == 0 or j == sf["nimages"]-1) else FMAX_THRESHOLD)
                for j, fm in seg_fmaxes
            )
            ci_below = seg_ci is not None and sf["effective_fmax"][seg_ci] < FMAX_THRESHOLD if seg_ci and seg_ci < len(sf["effective_fmax"]) else False

            lines.append(f"      Sub-band {seg_i}: images {seg_start}..{seg_end} "
                          f"(CI={seg_ci}, converged={'YES' if all_below else 'NO'}, CI_converged={'YES' if ci_below else 'NO'})")
            for j, fm in seg_fmaxes:
                lines.append(f"        img {j}: fmax={fm:.6f}")
    lines.append("")

    # ── Post-NEB: Dimer CI ──
    lines.append("─" * 60)
    lines.append("PHASE 2: Post-NEB Dimer CI Refinement")
    lines.append("─" * 60)

    dimer_files = sorted([
        fn for fn in files
        if fn.startswith(f"dimer_ci_{job_id}_img") and fn.endswith(".log")
        and "control" not in fn
    ])

    if not dimer_files:
        lines.append("  (no dimer CI refinement)")
    for fn in dimer_files:
        img_num = re.search(r"img(\d+)", fn).group(1)
        with open(files[fn]) as f:
            data = parse_dimer_log(f.read())
        if data["steps"]:
            lines.append(f"  CI image {img_num}:")
            lines.append(f"    Steps: {len(data['steps'])}")
            lines.append(f"    Initial: fmax={data['fmaxes'][0]:.6f}, curvature={data['curvatures'][0]:.4f}")
            lines.append(f"    Final:   fmax={data['fmaxes'][-1]:.6f}, curvature={data['curvatures'][-1]:.4f}")
            converged = data["fmaxes"][-1] < FMAX_THRESHOLD
            lines.append(f"    Converged: {'YES' if converged else 'NO'}")
            # Track curvature sign
            neg_curv = sum(1 for c in data["curvatures"] if c < 0)
            lines.append(f"    Negative curvature steps: {neg_curv}/{len(data['curvatures'])}")
    lines.append("")

    # ── Post-NEB: Imin Relaxation ──
    lines.append("─" * 60)
    lines.append("PHASE 3: Post-NEB Imin Relaxation")
    lines.append("─" * 60)

    imin_files = sorted([
        fn for fn in files
        if fn.startswith(f"imin_relax_{job_id}_img") and fn.endswith(".log")
    ])

    if not imin_files:
        lines.append("  (no imin relaxation)")
    for fn in imin_files:
        img_num = re.search(r"img(\d+)", fn).group(1)
        with open(files[fn]) as f:
            sections = parse_optimizer_log(f.read())
        if sections and sections[0]["fmaxes"]:
            sec = sections[0]
            lines.append(f"  Imin image {img_num}:")
            lines.append(f"    Steps: {len(sec['steps'])}")
            lines.append(f"    Initial fmax: {sec['fmaxes'][0]:.6f}")
            lines.append(f"    Final fmax:   {sec['fmaxes'][-1]:.6f}")
            lines.append(f"    Converged:    {'YES' if sec['fmaxes'][-1] < ENDPOINT_FMAX else 'NO'}")
    lines.append("")

    # ── Post-NEB: Refine NEB ──
    lines.append("─" * 60)
    lines.append("PHASE 4: Refine NEB (continuation, climb=False)")
    lines.append("─" * 60)

    if refine_sections:
        total_refine = sum(len(sec["steps"]) for sec in refine_sections)
        lines.append(f"  Total refine steps: {total_refine}")
        for si, sec in enumerate(refine_sections):
            lines.append(f"  Section {si}: {len(sec['steps'])} steps, "
                          f"fmax {sec['fmaxes'][0]:.4f} → {sec['fmaxes'][-1]:.4f}")

        if refine_steps:
            sf = refine_steps[-1]
            lines.append(f"  Final refine state:")
            lines.append(f"    nimages={sf['nimages']}, imin_set={sf['imin_set']}")
            lines.append(f"    Per-image fmax: {[f'{fm:.4f}' for fm in sf['effective_fmax']]}")
            frozen_imgs = sf.get("frozen_set", [])
            lines.append(f"    Frozen images: {frozen_imgs}")

            # Sub-band convergence after refine
            imin_set = sf.get("imin_set", [])
            boundaries = sorted([0] + imin_set + [sf["nimages"] - 1])
            frozen_set = set(frozen_imgs)
            lines.append("    Sub-band convergence after refine:")
            for seg_i in range(len(boundaries) - 1):
                seg_start = boundaries[seg_i]
                seg_end = boundaries[seg_i + 1]
                max_fm = 0
                for j in range(seg_start, seg_end + 1):
                    if j < len(sf["effective_fmax"]):
                        fm = sf["effective_fmax"][j]
                        if j not in frozen_set:
                            max_fm = max(max_fm, fm)
                lines.append(f"      Sub-band {seg_i} ({seg_start}..{seg_end}): max_fmax={max_fm:.6f} "
                              f"({'CONVERGED' if max_fm < FMAX_THRESHOLD else 'NOT CONVERGED'})")
    else:
        lines.append("  (no refine NEB data)")
    lines.append("")

    # ── Summary ──
    lines.append("─" * 60)
    lines.append("SUMMARY")
    lines.append("─" * 60)

    # Why didn't it converge?
    if traj_steps:
        sf = traj_steps[-1]
        frozen_set = set(sf.get("frozen_set", []))
        max_fmax = max((fm for img_i, fm in enumerate(sf["effective_fmax"]) if img_i not in frozen_set), default=0)

        stuck_images = []
        for img_i, fm in enumerate(sf["effective_fmax"]):
            if fm >= FMAX_THRESHOLD and img_i not in frozen_set:
                img_type = "CI" if img_i in sf.get("climbing_set", []) else \
                           "imin" if img_i in sf.get("imin_set", []) else \
                           "endpoint" if (img_i == 0 or img_i == sf["nimages"]-1) else "regular"
                stuck_images.append((img_i, fm, img_type))

        lines.append(f"  Band max fmax at end of main NEB: {max_fmax:.6f}")
        lines.append(f"  Stuck images ({len(stuck_images)}):")
        for img_i, fm, itype in stuck_images:
            lines.append(f"    Image {img_i} ({itype}): fmax={fm:.6f}")

        # Check for oscillation (fmax not decreasing in last 100 steps)
        if len(traj_steps) > 100:
            late_fmax = [max(s["effective_fmax"]) for s in traj_steps[-100:]]
            if max(late_fmax) - min(late_fmax) < 0.01:
                lines.append("  → Band fmax PLATEAUED in last 100 steps (oscillation)")
            elif late_fmax[-1] > late_fmax[0]:
                lines.append("  → Band fmax INCREASING in last 100 steps")
    lines.append("")
    lines.append("=" * 80)
    lines.append("")

    return "\n".join(lines)


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    global BASE_DIR, DEBUG_ZIP_DIR, STATUS_CSV_DIR, OUTPUT_DIR
    BASE_DIR = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else os.getcwd())
    DEBUG_ZIP_DIR = os.path.join(BASE_DIR, "NEB_debug_zips")
    STATUS_CSV_DIR = os.path.join(BASE_DIR, "NEB_status_csvs")
    OUTPUT_DIR = os.path.join(BASE_DIR, "neb_analysis")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load CSV status
    nc_jobs = load_status_csv()
    print(f"Found {len(nc_jobs)} jobs with selected sub-bands ({ANALYZE_STATUSES or 'all'}):")
    for (rank, job_id), info in sorted(nc_jobs.items()):
        print(f"  Rank {rank}, Job {job_id}: sub-bands {info['selected']}")
    print()

    # Create temp dir for extracted files
    temp_base = tempfile.mkdtemp(prefix="neb_analysis_")
    print(f"Temp directory: {temp_base}")
    print()

    full_report = []

    for (rank, job_id), info in sorted(nc_jobs.items()):
        print(f"{'='*60}")
        print(f"Analyzing Job {job_id} (Rank {rank})...")
        print(f"{'='*60}")

        temp_dir = os.path.join(temp_base, f"rank{rank}_job{job_id}")
        os.makedirs(temp_dir, exist_ok=True)

        # Extract files
        print("  Extracting files from debug zip...")
        files = extract_job_files(rank, job_id, temp_dir)
        print(f"  Extracted {len(files)} files")

        # Parse main NEB log
        log_sections = []
        neb_log_fn = f"neb_{job_id}.log"
        if neb_log_fn in files:
            with open(files[neb_log_fn]) as f:
                log_sections = parse_optimizer_log(f.read())
            print(f"  Main NEB log: {len(log_sections)} sections, "
                  f"{sum(len(s['steps']) for s in log_sections)} total steps")

        # Parse refine log
        refine_sections = []
        refine_log_fn = f"neb_refine_{job_id}.log"
        if refine_log_fn in files:
            with open(files[refine_log_fn]) as f:
                refine_sections = parse_optimizer_log(f.read())
            print(f"  Refine NEB log: {len(refine_sections)} sections, "
                  f"{sum(len(s['steps']) for s in refine_sections)} total steps")

        # Parse main NEB trajectory
        traj_steps, traj_events = [], []
        neb_traj_fn = f"neb_{job_id}.traj"
        if neb_traj_fn in files:
            traj_steps, traj_events = parse_neb_trajectory(
                files[neb_traj_fn], progress_label=f"job {job_id} main NEB traj"
            )

        # Parse refine trajectory
        refine_steps, refine_events = [], []
        refine_traj_fn = f"neb_refine_{job_id}.traj"
        if refine_traj_fn in files:
            refine_steps, refine_events = parse_neb_trajectory(
                files[refine_traj_fn], progress_label=f"job {job_id} refine NEB traj"
            )

        # Generate plots
        print("  Generating plots...")

        if traj_steps:
            plot_overview(
                job_id, log_sections, refine_sections,
                traj_steps, traj_events, refine_steps, refine_events,
                os.path.join(OUTPUT_DIR, f"job_{job_id}_overview.png"),
            )

            plot_per_image_fmax(
                job_id, traj_steps, traj_events,
                os.path.join(OUTPUT_DIR, f"job_{job_id}_per_image_fmax.png"),
            )

            plot_fmax_and_type_timeline(
                job_id, traj_steps, traj_events,
                os.path.join(OUTPUT_DIR, f"job_{job_id}_fmax_timeline.png"),
            )

            plot_energy_evolution(
                job_id, traj_steps, refine_steps,
                os.path.join(OUTPUT_DIR, f"job_{job_id}_energy_evolution.png"),
            )

        plot_refinement(
            job_id, files, temp_dir,
            os.path.join(OUTPUT_DIR, f"job_{job_id}_refinement.png"),
        )

        # Generate text report
        report = generate_text_report(
            job_id, rank, info["all_subbands"], files,
            log_sections, refine_sections,
            traj_steps, traj_events, refine_steps, refine_events,
        )
        full_report.append(report)
        print(report)

    # Write combined report
    report_path = os.path.join(OUTPUT_DIR, "analysis_report.txt")
    with open(report_path, "w") as f:
        f.write("\n".join(full_report))
    print(f"\nFull report saved to: {report_path}")

    # Cleanup temp
    shutil.rmtree(temp_base, ignore_errors=True)
    print(f"Cleaned up temp directory: {temp_base}")
    print(f"\nAll plots saved to: {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
