"""Consolidate the first frame of every converged Dimer attempt into one traj.

Run from a project root that contains Dimer_debug_zips/ and Dimer_status_csvs/.

Inputs
------
- Dimer_debug_zips/structure_rank_<R>_data.zip : per-rank zip of
  dimer_<src_index>_<attempt_id>_<atom_index>.traj files (and .log files).
- Dimer_status_csvs/status_rank_<R>.csv : per-rank rows
  `src_index, mpi_rank, attempt_id, atom_index, status` (no header).

Filter
------
A row is kept iff `status.startswith("converged") and status != "converged_to_desorption"`.
With the canonical Dimer status set (see /home1/07700/sjung3/software/tsearch/CLAUDE.md
"Status strings by source"), this admits only `converged` and `converged_after_extension`
and drops `not_converged*`, `converged_to_desorption`, and `error: ...`.

Output
------
<output-dir>/dimer_init_displaced.traj — one ASE trajectory containing the **first frame**
of each kept attempt's traj (i.e., the initial displaced structure fed into Dimer).

Every output frame's `atoms.info` carries:
    status     : str  — "converged" or "converged_after_extension"
    src_index  : int  — Dimer source-structure id
    attempt_id : int  — Dimer attempt id within that source
mpi_rank and atom_index are not stored (recoverable from CSVs if needed).

Safety checks (raise on any mismatch)
------------------------------------
1. CSV `mpi_rank` column equals the rank being processed.
2. Number of `.traj` entries in the zip equals number of CSV rows.
3. For every i, the i-th `.traj` filename's parsed (src,attempt,atom) tuple equals
   the i-th CSV row's tuple — verifying ordered one-to-one correspondence.

Downstream workflow (context for future scripts)
------------------------------------------------
The output of this script is intended to be fed into a tsearch `Minimization` run.
Separately, a tsearch `DoubleMinimization` run is performed starting from the converged
Dimer TS outputs (which were produced from these same initial displaced structures).
A future comparison script will join the minimization-of-init-displaced result against
the DoubleMinimization endpoints by `(src_index, attempt_id)` and check whether the
minimized init-displaced structure matches either of the two double-min endpoints
(connectivity comparison via tsearch.tools.check_reaction). Both sides of that join
carry `src_index` and `attempt_id`: this script writes them by construction, and
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


def parse_traj_name(name):
    base = os.path.basename(name)
    if not base.endswith(".traj"):
        raise ValueError(f"not a .traj: {name}")
    parts = base[:-len(".traj")].split("_")
    if len(parts) != 4 or parts[0] != "dimer":
        raise ValueError(f"unexpected traj filename: {name}")
    return int(parts[1]), int(parts[2]), int(parts[3])


def read_csv_rows(rank, csv_path):
    rows = []
    with open(csv_path, newline="") as f:
        for i, row in enumerate(csv.reader(f), start=1):
            if not row:
                continue
            if len(row) != 5:
                raise RuntimeError(f"{csv_path}:{i}: expected 5 columns, got {row}")
            src, mpi, attempt, atom, status = row
            src_i, mpi_i, attempt_i, atom_i = int(src), int(mpi), int(attempt), int(atom)
            if mpi_i != rank:
                raise RuntimeError(
                    f"{csv_path}:{i}: mpi_rank {mpi_i} != expected rank {rank}"
                )
            rows.append((src_i, attempt_i, atom_i, status))
    return rows


def keep(status):
    return status.startswith("converged") and status != "converged_to_desorption"


def process_rank(rank, out_traj, tmp_path):
    csv_path = os.path.join(CSV_DIR, f"status_rank_{rank}.csv")
    zip_path = os.path.join(ZIP_DIR, f"structure_rank_{rank}_data.zip")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(csv_path)
    if not os.path.exists(zip_path):
        raise FileNotFoundError(zip_path)

    rows = read_csv_rows(rank, csv_path)

    kept = 0
    with zipfile.ZipFile(zip_path, "r") as zf:
        traj_names = [n for n in zf.namelist() if n.endswith(".traj")]
        if len(traj_names) != len(rows):
            raise RuntimeError(
                f"rank {rank}: {len(traj_names)} .traj entries in zip vs "
                f"{len(rows)} rows in csv"
            )
        for i, (name, row) in enumerate(zip(traj_names, rows)):
            src, attempt, atom, status = row
            t_src, t_attempt, t_atom = parse_traj_name(name)
            if (t_src, t_attempt, t_atom) != (src, attempt, atom):
                raise RuntimeError(
                    f"rank {rank} idx {i}: traj {name} -> "
                    f"({t_src},{t_attempt},{t_atom}) != csv ({src},{attempt},{atom})"
                )
            if not keep(status):
                continue
            with open(tmp_path, "wb") as f:
                f.write(zf.read(name))
            with Trajectory(tmp_path, "r") as t:
                if len(t) == 0:
                    raise RuntimeError(f"rank {rank}: {name} has 0 frames")
                atoms = t[0]
            atoms.info["status"] = status
            atoms.info["src_index"] = src
            atoms.info["attempt_id"] = attempt
            out_traj.write(atoms)
            kept += 1
    print(f"rank {rank}: kept {kept} / {len(rows)}", flush=True)
    return kept, len(rows)


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
    total_rows = 0
    out_traj = Trajectory(str(out_path), "w")
    try:
        for rank in ranks:
            k, n = process_rank(rank, out_traj, tmp_path)
            total_kept += k
            total_rows += n
    finally:
        out_traj.close()
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

    print(f"done. wrote {total_kept} / {total_rows} structures -> {out_path}")


if __name__ == "__main__":
    main()
