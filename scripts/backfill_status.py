#!/usr/bin/env python
"""Backfill ``atoms.info['status']`` onto output traj frames from CSV rows.

Run from a directory containing ``{Dimer,NEB}_status_csvs/`` and
``{Dimer,NEB}_trajes/``. Rewrites each output traj in place; the original
is moved to ``*.bak`` next to it. Re-running is idempotent — files that
already have a ``.bak`` sibling are skipped.

Usage:
    python backfill_status.py             # patch both Dimer and NEB if present
    python backfill_status.py Dimer       # just Dimer
    python backfill_status.py NEB         # just NEB
"""
import csv
import glob
import os
import shutil
import sys
from ase.io import Trajectory

# method -> (frame info key for sub-unit, csv column index of sub-unit id)
METHOD_KEYS = {
    "Dimer": ("attempt_id", 2),   # csv: job_id, rank, attempt_id, selected_idx, status
    "NEB":   ("subband_idx", 2),  # csv: job_id, rank, sub_band_id, status
}


def load_status_map(method):
    """Return {(src_index, subunit_id): status} from all rank CSVs."""
    _, subunit_col = METHOD_KEYS[method]
    out = {}
    for path in sorted(glob.glob(f"{method}_status_csvs/status_rank_*.csv")):
        with open(path) as fh:
            for row in csv.reader(fh):
                if not row:
                    continue
                src = int(row[0])
                sub = int(row[subunit_col])
                out[(src, sub)] = row[-1].strip()
    return out


def patch_method(method):
    if not (os.path.isdir(f"{method}_status_csvs") and os.path.isdir(f"{method}_trajes")):
        return
    info_key, _ = METHOD_KEYS[method]
    status_map = load_status_map(method)
    print(f"[{method}] loaded {len(status_map)} status entries")

    for traj_path in sorted(glob.glob(f"{method}_trajes/collected_*.traj")):
        bak = traj_path + ".bak"
        if os.path.exists(bak):
            print(f"  skip {traj_path} (.bak already exists)")
            continue
        shutil.copy2(traj_path, bak)
        with Trajectory(bak, "r") as src:
            frames = [src[i] for i in range(len(src))]
        n_set = 0
        for img in frames:
            key = (img.info.get("src_index"), img.info.get(info_key))
            if key in status_map:
                img.info["status"] = status_map[key]
                n_set += 1
        os.remove(traj_path)
        with Trajectory(traj_path, "w") as out:
            for img in frames:
                out.write(img)
        print(f"  {traj_path}: tagged {n_set}/{len(frames)} frames")


if __name__ == "__main__":
    methods = sys.argv[1:] or ["Dimer", "NEB"]
    for m in methods:
        if m not in METHOD_KEYS:
            print(f"unknown method: {m}", file=sys.stderr)
            continue
        patch_method(m)
