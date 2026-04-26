"""Consolidate the first frame of every converged Dimer attempt into per-rank trajs.

Run from a project root that contains Dimer_debug_zips/ and Dimer_status_csvs/.

Inputs
------
- Dimer_debug_zips/structure_rank_<R>_data.zip : per-rank zip of
  dimer_<src_index>_<attempt_id>_<atom_index>.traj files (and .log files,
  occasionally ERROR_dimer_*.traj for crashed attempts — both ignored).
- Dimer_status_csvs/status_rank_<R>.csv : per-rank rows
  `src_index, mpi_rank, attempt_id, atom_index, status` (no header).

Logic
-----
Iterate CSV rows. For rows whose status passes the filter
(`startswith("converged") and != "converged_to_desorption"`), construct the
expected filename `dimer_<src>_<attempt>_<atom>.traj` and read it from the
matching zip. The (src, attempt, atom) check is implicit in the filename
lookup. If a converged row's traj is absent, log it and continue; missing
trajs are summarized at the end.

Output
------
<output-dir>/dimer_init_displaced_rank_<R>.traj — one ASE trajectory per
input rank, each containing the **first frame** of every kept attempt's
traj (i.e., the initial displaced structure fed into Dimer). All per-rank
files in the same directory together form the consolidated set.

Every output frame's `atoms.info` carries:
    status     : str  — "converged" or "converged_after_extension"
    src_index  : int  — Dimer source-structure id
    attempt_id : int  — Dimer attempt id within that source

Parallelism
-----------
Ranks are processed in parallel with `multiprocessing.Pool`. Worker count
defaults to `len(os.sched_getaffinity(0))` and is configurable via
`-j/--workers`. Each worker writes its own per-rank output traj and uses
its own tempfile for in-zip extraction; no shared state.

Downstream workflow (context for future scripts)
------------------------------------------------
The output of this script is intended to be fed into a tsearch `Minimization`
run. Separately, a tsearch `DoubleMinimization` run is performed starting from
the converged Dimer TS outputs (which were produced from these same initial
displaced structures). A future comparison script will join the
minimization-of-init-displaced result against the DoubleMinimization endpoints
by `(src_index, attempt_id)` and check whether the minimized init-displaced
structure matches either of the two double-min endpoints (connectivity
comparison via tsearch.tools.check_reaction). Both sides of that join carry
`src_index` and `attempt_id`: this script writes them by construction, and
DoubleMinimization output preserves them through the tsearch pipeline.
"""
import argparse
import csv
import glob
import multiprocessing as mp
import os
import re
import sys
import tempfile
import zipfile
from pathlib import Path

from ase.io import Trajectory


CSV_DIR = "Dimer_status_csvs"
ZIP_DIR = "Dimer_debug_zips"
DEFAULT_OUT_DIR = "dimer_init_displaced"
OUT_NAME_FMT = "dimer_init_displaced_rank_{rank}.traj"

RANK_RE = re.compile(r"^status_rank_(\d+)\.csv$")


def discover_ranks():
    ranks = []
    for p in glob.glob(os.path.join(CSV_DIR, "status_rank_*.csv")):
        m = RANK_RE.match(os.path.basename(p))
        if m:
            ranks.append(int(m.group(1)))
    return sorted(ranks)


def keep(status):
    return status.startswith("converged") and status != "converged_to_desorption"


def process_rank(rank, out_path):
    """Process one rank → write per-rank output traj.

    Returns (kept, n_kept_rows, missing, log_lines). `log_lines` is a list of
    per-row warning strings the caller should print so progress stays live but
    output from concurrent workers does not interleave.
    """
    csv_path = os.path.join(CSV_DIR, f"status_rank_{rank}.csv")
    zip_path = os.path.join(ZIP_DIR, f"structure_rank_{rank}_data.zip")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(csv_path)
    if not os.path.exists(zip_path):
        raise FileNotFoundError(zip_path)

    log_lines = []
    missing = []
    kept = 0
    n_kept_rows = 0

    fd, tmp_path = tempfile.mkstemp(suffix=".traj")
    os.close(fd)
    out_traj = Trajectory(str(out_path), "w")
    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            names_in_zip = set(zf.namelist())
            with open(csv_path, newline="") as f:
                for i, row in enumerate(csv.reader(f), start=1):
                    if not row:
                        continue
                    if len(row) != 5:
                        raise RuntimeError(
                            f"{csv_path}:{i}: expected 5 columns, got {row}"
                        )
                    src, mpi, attempt, atom, status = row
                    src, mpi, attempt, atom = int(src), int(mpi), int(attempt), int(atom)
                    if mpi != rank:
                        raise RuntimeError(
                            f"{csv_path}:{i}: mpi_rank {mpi} != expected rank {rank}"
                        )
                    if not keep(status):
                        continue
                    n_kept_rows += 1
                    traj_name = f"dimer_{src}_{attempt}_{atom}.traj"
                    if traj_name not in names_in_zip:
                        log_lines.append(
                            f"  MISSING rank {rank} {csv_path}:{i} -> {traj_name} "
                            f"(status={status})"
                        )
                        missing.append((rank, traj_name, status))
                        continue
                    with open(tmp_path, "wb") as f_tmp:
                        f_tmp.write(zf.read(traj_name))
                    with Trajectory(tmp_path, "r") as t:
                        if len(t) == 0:
                            log_lines.append(
                                f"  EMPTY rank {rank} {traj_name} (0 frames) "
                                f"(status={status})"
                            )
                            missing.append((rank, traj_name, status + " [empty traj]"))
                            continue
                        atoms = t[0]
                    atoms.info["status"] = status
                    atoms.info["src_index"] = src
                    atoms.info["attempt_id"] = attempt
                    out_traj.write(atoms)
                    kept += 1
    finally:
        out_traj.close()
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
    return kept, n_kept_rows, missing, log_lines


def _worker(args):
    rank, out_path = args
    try:
        kept, n_kept_rows, missing, log_lines = process_rank(rank, out_path)
    except Exception as e:
        # Surface a clear error tied to the offending rank.
        return ("error", rank, f"{type(e).__name__}: {e}")
    return ("ok", rank, kept, n_kept_rows, missing, log_lines)


def main():
    default_workers = len(os.sched_getaffinity(0))
    ap = argparse.ArgumentParser()
    ap.add_argument("--ranks", type=int, nargs="+", default=None,
                    help="Rank ids to process (default: all discovered)")
    ap.add_argument("--output-dir", default=DEFAULT_OUT_DIR)
    ap.add_argument("-j", "--workers", type=int, default=default_workers,
                    help=f"Worker processes (default: {default_workers}, "
                         f"the number of CPUs available to this process)")
    args = ap.parse_args()

    if args.workers < 1:
        ap.error("--workers must be >= 1")

    ranks = args.ranks if args.ranks is not None else discover_ranks()
    if not ranks:
        raise RuntimeError("no ranks found")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    workers = min(args.workers, len(ranks))
    total = len(ranks)
    print(f"processing {total} rank(s) with {workers} worker(s)", flush=True)

    tasks = [(r, str(out_dir / OUT_NAME_FMT.format(rank=r))) for r in ranks]

    total_kept = 0
    total_kept_rows = 0
    missing = []
    done = 0

    ctx = mp.get_context("fork")
    with ctx.Pool(processes=workers) as pool:
        try:
            for result in pool.imap_unordered(_worker, tasks):
                if result[0] == "error":
                    _, rank, msg = result
                    pool.terminate()
                    pool.join()
                    raise RuntimeError(f"rank {rank} failed: {msg}")
                _, rank, kept, n_kept_rows, rank_missing, log_lines = result
                for line in log_lines:
                    print(line, flush=True)
                done += 1
                total_kept += kept
                total_kept_rows += n_kept_rows
                missing.extend(rank_missing)
                print(
                    f"rank {rank}: kept {kept} / {n_kept_rows} converged rows "
                    f"[done {done}/{total}, kept total {total_kept}/{total_kept_rows}]",
                    flush=True,
                )
        except KeyboardInterrupt:
            pool.terminate()
            pool.join()
            raise

    print(
        f"\ndone. wrote {total_kept} / {total_kept_rows} structures across "
        f"{total} rank file(s) -> {out_dir}/"
    )
    if missing:
        print(f"\n{len(missing)} converged row(s) had no usable traj:")
        for rank, name, status in missing:
            print(f"  rank {rank}: {name} (status={status})")
    else:
        print("\nno missing trajs.")


if __name__ == "__main__":
    main()
