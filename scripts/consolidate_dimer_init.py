"""Consolidate the first frame of every converged Dimer attempt into one traj.

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
<output-dir>/dimer_init_displaced.traj — one ASE trajectory containing the
**first frame** of each kept attempt's traj (i.e., the initial displaced
structure fed into Dimer).

Every output frame's `atoms.info` carries:
    status     : str  — "converged" or "converged_after_extension"
    src_index  : int  — Dimer source-structure id
    attempt_id : int  — Dimer attempt id within that source

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
import os
import re
import tempfile
import zipfile
from pathlib import Path

from ase.io import Trajectory


CSV_DIR = "Dimer_status_csvs"
ZIP_DIR = "Dimer_debug_zips"
DEFAULT_OUT_DIR = "dimer_init_displaced"
OUT_NAME = "dimer_init_displaced.traj"

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


def process_rank(rank, out_traj, tmp_path, missing):
    csv_path = os.path.join(CSV_DIR, f"status_rank_{rank}.csv")
    zip_path = os.path.join(ZIP_DIR, f"structure_rank_{rank}_data.zip")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(csv_path)
    if not os.path.exists(zip_path):
        raise FileNotFoundError(zip_path)

    kept = 0
    n_kept_rows = 0
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
                    print(
                        f"  MISSING rank {rank} {csv_path}:{i} -> {traj_name} "
                        f"(status={status})",
                        flush=True,
                    )
                    missing.append((rank, traj_name, status))
                    continue
                with open(tmp_path, "wb") as f_tmp:
                    f_tmp.write(zf.read(traj_name))
                with Trajectory(tmp_path, "r") as t:
                    if len(t) == 0:
                        print(
                            f"  EMPTY rank {rank} {traj_name} (0 frames) "
                            f"(status={status})",
                            flush=True,
                        )
                        missing.append((rank, traj_name, status + " [empty traj]"))
                        continue
                    atoms = t[0]
                atoms.info["status"] = status
                atoms.info["src_index"] = src
                atoms.info["attempt_id"] = attempt
                out_traj.write(atoms)
                kept += 1
    print(f"rank {rank}: kept {kept} / {n_kept_rows} converged rows", flush=True)
    return kept, n_kept_rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ranks", type=int, nargs="+", default=None,
                    help="Rank ids to process (default: all discovered)")
    ap.add_argument("--output-dir", default=DEFAULT_OUT_DIR)
    args = ap.parse_args()

    ranks = args.ranks if args.ranks is not None else discover_ranks()
    if not ranks:
        raise RuntimeError("no ranks found")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / OUT_NAME

    fd, tmp_path = tempfile.mkstemp(suffix=".traj")
    os.close(fd)

    total_kept = 0
    total_kept_rows = 0
    missing = []
    out_traj = Trajectory(str(out_path), "w")
    try:
        for rank in ranks:
            k, n = process_rank(rank, out_traj, tmp_path, missing)
            total_kept += k
            total_kept_rows += n
    finally:
        out_traj.close()
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

    print(f"\ndone. wrote {total_kept} / {total_kept_rows} structures -> {out_path}")
    if missing:
        print(f"\n{len(missing)} converged row(s) had no usable traj:")
        for rank, name, status in missing:
            print(f"  rank {rank}: {name} (status={status})")
    else:
        print("\nno missing trajs.")


if __name__ == "__main__":
    main()
