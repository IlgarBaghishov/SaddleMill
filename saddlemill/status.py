"""Status summary for SaddleMill jobs.

Usage:
    python -m saddlemill.status [directory]

If directory is not specified, uses the current working directory.
"""

import os
import sys
import json
from itertools import groupby

from .config import ConfigManager, _categorize_status, read_status_csv_rows


def _read_config(directory):
    """Read config using ConfigManager."""
    config_path = os.path.join(directory, "config.ini")
    if not os.path.exists(config_path):
        print(f"Error: No config.ini found in {directory}")
        sys.exit(1)
    return ConfigManager(config_path)


def _get_total_jobs(directory, config):
    """Get job count from traj_files_ordered.json, post input_statuses filter.

    When ``input_statuses`` is narrower than ``all``, frames whose stored
    status does not match are silently skipped at submission time and will
    never produce CSV rows under the current filter. This function mirrors
    that filter so ``Total``/``Remaining`` reflect only eligible jobs.
    """
    json_path = os.path.join(directory, "traj_files_ordered.json")
    if not os.path.exists(json_path):
        return None
    with open(json_path) as f:
        data = json.load(f)

    raw = config["Main"]["input_statuses"]
    if raw in ("all", None):
        return len(data)

    # Apply the same filter as __main__.py to count eligible input frames.
    from ase.io import Trajectory
    from .tools import load_and_sanitize, passes_input_filter

    count = 0
    for traj_name, group in groupby(data, key=lambda x: x[0]):
        traj_path = traj_name if os.path.isabs(traj_name) \
            else os.path.join(directory, traj_name)
        if not os.path.exists(traj_path):
            continue
        with Trajectory(traj_path, 'r') as traj:
            for _, i, j in group:
                images = load_and_sanitize(traj, i, j)
                if passes_input_filter(images, config):
                    count += 1
    return count


def _compute_expected_entries_per_job(method_name, config):
    """Compute expected entries per job from config, if possible."""
    if method_name in ("Dimer", "Sella"):
        section = "ourSella" if method_name == "Sella" else "ourDimer"
        reaction_types = config.get_value(section, "reaction_types")
        if reaction_types:
            if isinstance(reaction_types, list):
                num_types = len(reaction_types)
            else:
                num_types = len(reaction_types.split())
        else:
            return None
        
        raw_num = config.get_value(section, "num_attempts_per_type", 1)

        if isinstance(raw_num, list):
            counts = [int(x) for x in raw_num]
        elif isinstance(raw_num, str):
            counts = [int(x) for x in raw_num.split()]
        else:
            counts = [int(raw_num)]
            
        if len(counts) == 1:
            return num_types * counts[0]
        else:
            return sum(counts)

    elif method_name == "DoubleMinimization":
        return 2
    elif method_name == "Minimization":
        return 1
    elif method_name == "NEB":
        return None  # depends on imin detection
    return None


def _get_dimer_reaction_type(attempt_id, reaction_types_list, counts):
    """Map attempt_id to reaction type using config ordering."""
    if len(counts) == 1:
        type_idx = attempt_id // counts[0]
    else:
        # Find which bucket the attempt_id falls into
        current_sum = 0
        type_idx = 0
        for count in counts:
            current_sum += count
            if attempt_id < current_sum:
                break
            type_idx += 1

    if type_idx < len(reaction_types_list):
        return reaction_types_list[type_idx]
    return "unknown"

def _print_dimer_reaction_type_table(rows, config, section="ourDimer"):
    """Print per-reaction-type breakdown table for Dimer/Sella jobs.

    Both methods share the reaction-type machinery and the status-CSV layout,
    so the same table serves either; *section* selects which our*-section the
    reaction types are read from.
    """
    reaction_types = config.get_value(section, "reaction_types")
    if not reaction_types:
        return
    if isinstance(reaction_types, list):
        reaction_types_list = reaction_types
    else:
        reaction_types_list = reaction_types.split()

    raw_num = config.get_value(section, "num_attempts_per_type", 1)
    if isinstance(raw_num, list):
        counts = [int(x) for x in raw_num]
    elif isinstance(raw_num, str):
        counts = [int(x) for x in raw_num.split()]
    else:
        counts = [int(raw_num)]

    # Accumulate counts per reaction type
    type_stats = {}  # {rtype: {converged, not_converged, errored, total}}
    for row in rows:
        attempt_id = int(row[2])
        status = row[-1].strip()
        category = _categorize_status(status)
        rtype = _get_dimer_reaction_type(attempt_id, reaction_types_list, counts)
        if rtype not in type_stats:
            type_stats[rtype] = {"converged": 0, "not_converged": 0, "errored": 0, "total": 0}
        type_stats[rtype][category] += 1
        type_stats[rtype]["total"] += 1

    if not type_stats:
        return

    # Print table
    print(f"  Per reaction type:")
    # Header
    max_name = max(len(rt) for rt in type_stats)
    max_name = max(max_name, len("reaction_type"))
    hdr = f"    {'reaction_type':<{max_name}s}  {'total':>6s}  {'conv':>6s}  {'not_conv':>8s}  {'error':>6s}  {'conv%':>6s}"
    print(hdr)
    print(f"    {'-'*(len(hdr)-4)}")

    # Rows in config order
    for rtype in reaction_types_list:
        s = type_stats.get(rtype)
        if s is None:
            continue
        finished = s["converged"] + s["not_converged"]
        conv_pct = 100 * s["converged"] / finished if finished > 0 else 0
        print(f"    {rtype:<{max_name}s}  {s['total']:>6d}  {s['converged']:>6d}  {s['not_converged']:>8d}  {s['errored']:>6d}  {conv_pct:>5.1f}%")

    # Show 'unknown' if any
    if "unknown" in type_stats:
        s = type_stats["unknown"]
        finished = s["converged"] + s["not_converged"]
        conv_pct = 100 * s["converged"] / finished if finished > 0 else 0
        print(f"    {'unknown':<{max_name}s}  {s['total']:>6d}  {s['converged']:>6d}  {s['not_converged']:>8d}  {s['errored']:>6d}  {conv_pct:>5.1f}%")
    print()


def _print_neb_job_summary(rows, jobs_started):
    """Print per-job convergence and sub-band count distribution for NEB."""
    # Group by job_id: collect (sub_band_id, status) per job
    job_data = {}  # {job_id: [(sub_band_id, status_string), ...]}
    for row in rows:
        job_id = int(row[0])
        sub_band_id = int(row[2])
        status = row[-1].strip()
        job_data.setdefault(job_id, []).append((sub_band_id, status))

    # Per-job convergence
    job_converged = 0
    job_converged_ci = 0
    job_not_converged = 0
    job_errored = 0
    # Sub-band count distribution
    subband_counts = {}  # {num_subbands: count}

    for job_id, entries in job_data.items():
        categories = [_categorize_status(s) for _, s in entries]
        statuses = [s for _, s in entries]
        num_subbands = len(set(sb for sb, _ in entries))
        subband_counts[num_subbands] = subband_counts.get(num_subbands, 0) + 1

        if "errored" in categories:
            job_errored += 1
        elif "not_converged" in categories:
            job_not_converged += 1
        elif all(s == "converged" for s in statuses):
            job_converged += 1
        else:
            # All are converged or converged_CI, at least one converged_CI
            job_converged_ci += 1

    jobs_finished = job_converged + job_converged_ci + job_not_converged
    print(f"  Per-job convergence (all sub-bands must pass):")
    if jobs_finished > 0:
        print(f"    Converged:        {job_converged:>6}  ({100*job_converged/jobs_finished:5.1f}% of finished)")
        print(f"    Converged (CI):   {job_converged_ci:>6}  ({100*job_converged_ci/jobs_finished:5.1f}% of finished)")
        print(f"    Not converged:    {job_not_converged:>6}  ({100*job_not_converged/jobs_finished:5.1f}% of finished)")
    else:
        print(f"    Converged:        {job_converged:>6}")
        print(f"    Converged (CI):   {job_converged_ci:>6}")
        print(f"    Not converged:    {job_not_converged:>6}")
    print(f"    Errored:          {job_errored:>6}  ({100*job_errored/jobs_started:5.1f}% of total)")
    print()

    print(f"  Sub-band distribution:")
    # Bucket 6+ together
    bucketed = {}
    for n, count in subband_counts.items():
        key = n if n < 6 else 6
        bucketed[key] = bucketed.get(key, 0) + count
    for n in sorted(bucketed):
        label = "6+" if n == 6 else f"{n}"
        print(f"    {label} sub-band{'s' if n != 1 else ' '}: {bucketed[n]:>6} jobs")
    print()


def main():
    directory = sys.argv[1] if len(sys.argv) > 1 else os.getcwd()
    directory = os.path.abspath(directory)

    if not os.path.isdir(directory):
        print(f"Error: {directory} is not a directory")
        sys.exit(1)

    config = _read_config(directory)
    method = config.get_value("Main", "method")
    total_jobs = _get_total_jobs(directory, config)
    rows = read_status_csv_rows(method, directory)
    entries_per_job = _compute_expected_entries_per_job(method, config)

    # Group entries by job_id, categorize each
    job_entries = {}
    cats = {"converged": 0, "not_converged": 0, "errored": 0}
    # Also track detailed statuses
    detailed = {}
    for row in rows:
        job_id = int(row[0])
        status = row[-1].strip()
        category = _categorize_status(status)
        job_entries.setdefault(job_id, []).append(category)
        cats[category] += 1
        detailed[status] = detailed.get(status, 0) + 1

    jobs_started = len(job_entries)
    total_entries = len(rows)

    # Print summary
    print(f"\n{'='*55}")
    print(f"  SaddleMill Status Summary - {method}")
    print(f"{'='*55}")
    print(f"  Directory: {directory}")
    print()

    # --- Job-level ---
    print(f"  Jobs (structures):")
    if total_jobs is not None and total_jobs > 0:
        jobs_remaining = max(0, total_jobs - jobs_started)
        print(f"    Total:       {total_jobs:>6}")
        print(f"    Started:     {jobs_started:>6}  ({100*jobs_started/total_jobs:5.1f}%)")
        print(f"    Remaining:   {jobs_remaining:>6}  ({100*jobs_remaining/total_jobs:5.1f}%)")
    elif total_jobs == 0:
        print(f"    Total:       {total_jobs:>6}  (no input frames match input_statuses)")
        print(f"    Started:     {jobs_started:>6}")
    else:
        print(f"    Started:     {jobs_started:>6}")
        print(f"    (no traj_files_ordered.json - cannot determine remaining)")
    print()

    # --- Entry-level ---
    print(f"  Entries (per-attempt/sub-band/side):")
    if entries_per_job and total_jobs:
        expected = entries_per_job * total_jobs
        print(f"    Expected:      {expected:>6}  ({entries_per_job} per job x {total_jobs} jobs)")
    print(f"    Completed:     {total_entries:>6}")
    if total_entries > 0:
        finished = cats['converged'] + cats['not_converged']
        print()
        if finished > 0:
            print(f"    Converged:     {cats['converged']:>6}  ({100*cats['converged']/finished:5.1f}% of finished)")
            print(f"    Not converged: {cats['not_converged']:>6}  ({100*cats['not_converged']/finished:5.1f}% of finished)")
        else:
            print(f"    Converged:     {cats['converged']:>6}")
            print(f"    Not converged: {cats['not_converged']:>6}")
        print(f"    Errored:       {cats['errored']:>6}  ({100*cats['errored']/total_entries:5.1f}% of total)")
        print()

        # Per-reaction-type table (Dimer / Sella)
        if method in ("Dimer", "Sella"):
            _print_dimer_reaction_type_table(
                rows, config, "ourSella" if method == "Sella" else "ourDimer")

        # NEB per-job convergence and sub-band distribution
        if method == "NEB":
            _print_neb_job_summary(rows, jobs_started)

        # Detailed status breakdown
        print(f"  Detailed statuses:")
        for status, count in sorted(detailed.items(), key=lambda x: -x[1]):
            print(f"    {status:<40s} {count:>5}  ({100*count/total_entries:5.1f}%)")

    print(f"\n{'='*55}\n")


if __name__ == "__main__":
    main()
