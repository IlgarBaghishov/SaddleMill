"""Status summary for tsearch jobs.

Usage:
    python -m tsearch.status [directory]

If directory is not specified, uses the current working directory.
"""

import os
import sys
import glob
import json
import csv
import configparser


def _categorize_status(status_str):
    """Categorize a CSV status string into converged/not_converged/errored."""
    if status_str.startswith("converged"):
        return "converged"
    elif status_str.startswith("error"):
        return "errored"
    return "not_converged"


def _read_config(directory):
    """Read method from config.ini."""
    config_path = os.path.join(directory, "config.ini")
    if not os.path.exists(config_path):
        print(f"Error: No config.ini found in {directory}")
        sys.exit(1)
    config = configparser.ConfigParser()
    config.read(config_path)
    return config


def _get_total_jobs(directory):
    """Get total job count from traj_files_ordered.json."""
    json_path = os.path.join(directory, "traj_files_ordered.json")
    if not os.path.exists(json_path):
        return None
    with open(json_path) as f:
        data = json.load(f)
    return len(data)


def _read_status_csvs(method_name, directory):
    """Read all status CSVs. Returns list of rows (each row is a list of strings)."""
    csv_dir = os.path.join(directory, f"{method_name}_status_csvs")
    rows = []
    for csv_path in sorted(glob.glob(os.path.join(csv_dir, "status_rank_*.csv"))):
        with open(csv_path) as f:
            for row in csv.reader(f):
                if row:
                    rows.append(row)
    return rows


def _compute_expected_entries_per_job(method_name, config):
    """Compute expected entries per job from config, if possible."""
    if method_name == "Dimer":
        reaction_types = config.get("ourDimer", "reaction_types", fallback=None)
        if reaction_types:
            num_types = len(reaction_types.split())
        else:
            return None
        num_per_type = config.getint("ourDimer", "num_attempts_per_type", fallback=1)
        return num_types * num_per_type
    elif method_name == "DoubleMinimization":
        return 2
    elif method_name == "Minimization":
        return 1
    elif method_name == "NEB":
        return None  # depends on imin detection
    return None


def _get_dimer_reaction_type(attempt_id, reaction_types_list, num_per_type):
    """Map attempt_id to reaction type using config ordering."""
    type_idx = attempt_id // num_per_type
    if type_idx < len(reaction_types_list):
        return reaction_types_list[type_idx]
    return "unknown"


def _print_dimer_reaction_type_table(rows, config):
    """Print per-reaction-type breakdown table for Dimer jobs."""
    reaction_types = config.get("ourDimer", "reaction_types", fallback=None)
    if not reaction_types:
        return
    reaction_types_list = reaction_types.split()
    num_per_type = config.getint("ourDimer", "num_attempts_per_type", fallback=1)

    # Accumulate counts per reaction type
    type_stats = {}  # {rtype: {converged, not_converged, errored, total}}
    for row in rows:
        attempt_id = int(row[2])
        status = row[-1].strip()
        category = _categorize_status(status)
        rtype = _get_dimer_reaction_type(attempt_id, reaction_types_list, num_per_type)
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
        conv_pct = 100 * s["converged"] / s["total"] if s["total"] > 0 else 0
        print(f"    {rtype:<{max_name}s}  {s['total']:>6d}  {s['converged']:>6d}  {s['not_converged']:>8d}  {s['errored']:>6d}  {conv_pct:>5.1f}%")

    # Show 'unknown' if any
    if "unknown" in type_stats:
        s = type_stats["unknown"]
        conv_pct = 100 * s["converged"] / s["total"] if s["total"] > 0 else 0
        print(f"    {'unknown':<{max_name}s}  {s['total']:>6d}  {s['converged']:>6d}  {s['not_converged']:>8d}  {s['errored']:>6d}  {conv_pct:>5.1f}%")
    print()


def main():
    directory = sys.argv[1] if len(sys.argv) > 1 else os.getcwd()
    directory = os.path.abspath(directory)

    if not os.path.isdir(directory):
        print(f"Error: {directory} is not a directory")
        sys.exit(1)

    config = _read_config(directory)
    method = config.get("Main", "method")
    total_jobs = _get_total_jobs(directory)
    rows = _read_status_csvs(method, directory)
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
    print(f"  tsearch Status Summary - {method}")
    print(f"{'='*55}")
    print(f"  Directory: {directory}")
    print()

    # --- Job-level ---
    print(f"  Jobs (structures):")
    if total_jobs is not None:
        jobs_remaining = total_jobs - jobs_started
        print(f"    Total:       {total_jobs:>6}")
        print(f"    Started:     {jobs_started:>6}  ({100*jobs_started/total_jobs:5.1f}%)")
        print(f"    Remaining:   {jobs_remaining:>6}  ({100*jobs_remaining/total_jobs:5.1f}%)")
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
        print()
        print(f"    Converged:     {cats['converged']:>6}  ({100*cats['converged']/total_entries:5.1f}%)")
        print(f"    Not converged: {cats['not_converged']:>6}  ({100*cats['not_converged']/total_entries:5.1f}%)")
        print(f"    Errored:       {cats['errored']:>6}  ({100*cats['errored']/total_entries:5.1f}%)")
        print()

        # Per-reaction-type table (Dimer only)
        if method == "Dimer":
            _print_dimer_reaction_type_table(rows, config)

        # Detailed status breakdown
        print(f"  Detailed statuses:")
        for status, count in sorted(detailed.items(), key=lambda x: -x[1]):
            print(f"    {status:<40s} {count:>5}  ({100*count/total_entries:5.1f}%)")

    print(f"\n{'='*55}\n")


if __name__ == "__main__":
    main()
