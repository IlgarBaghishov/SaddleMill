"""
Parallel multi-shard build for the no-E/F LMDB pipeline.

Spawns N worker processes; each takes a subset of trajes and writes its
own private `.aselmdb` shard plus a per-worker CSV fragment. Auto-resumes
from the master `global_index_map.csv` so ms_id stays globally unique
across clusters. After all workers finish, the per-worker CSV fragments
are appended to the master CSV in ms_id order.

Per-frame contract identical to build_lmdb_no_ef.py:
  KVPs (searchable):  task_name, ms_id, src_index, side, status
  row.data["traj_path"]:  absolute source .traj path
  row.data["info"]:       full original atoms.info plus task_name + ms_id

Filter: only triples whose ALL three frames have status starting with
"converged" are written. A frame missing 'status' is a hard error.

Designed for the multi-cluster workflow: each cluster runs this with its
own --shard-dir (e.g. shards/<cluster_name>/) so shards from different
clusters don't collide when consolidated for the final merge.
"""
import argparse
import csv
import multiprocessing as mp
import os
import shutil
import sys
import threading
import time
from pathlib import Path

from ase.db import connect
from ase.io import Trajectory


# Shared progress counter and per-run config; set by Pool initializer in
# worker processes (so values propagate cleanly under both fork and spawn
# multiprocessing start methods).
_progress_counter = None
_commit_every_rows = None
_filter_status = True
PROGRESS_BATCH = 100  # frames between counter increments per worker

# Default rows between LMDB commit-and-reopen. Limits the size of any
# single LMDB write transaction, which otherwise causes write throughput
# to decay as the b-tree grows. ~50k rows ≈ 300 MB on disk per chunk
# for our schema.
DEFAULT_COMMIT_EVERY_ROWS = 50_000


class _FastIdsList(list):
    """list subclass with O(1) `__contains__` via a parallel set.

    Patched onto the aselmdb backend's `db.ids` to fix a quadratic-time
    scan in `LMDBDatabase._write`: every write does
        if idx not in self.ids: self.ids.append(idx)
    and `self.ids` is a Python list with O(N) `in`. With N rows totalling
    in the millions, the total cost of this check alone is O(N^2).
    """

    def __init__(self, iterable=()):
        super().__init__(iterable)
        self._set = set(self)

    def append(self, x):
        super().append(x)
        self._set.add(x)

    def __contains__(self, x):
        return x in self._set

    def remove(self, x):
        super().remove(x)
        self._set.discard(x)


def _patch_db_ids(db):
    """Replace db.ids with the O(1)-contains version."""
    db.ids = _FastIdsList(db.ids)


def _init_worker(counter, commit_every_rows, filter_status):
    global _progress_counter, _commit_every_rows, _filter_status
    _progress_counter = counter
    _commit_every_rows = commit_every_rows
    _filter_status = filter_status


def _init_count_worker(filter_status):
    global _filter_status
    _filter_status = filter_status


def count_filtered_triples(fp: str) -> int:
    """Number of triples in fp that pass the (optional) converged filter.

    If `_filter_status` is False, returns total triples without inspecting
    `info['status']` (use this for datasets pre-filtered upstream)."""
    with Trajectory(fp) as t:
        m = len(t)
        if m % 3 != 0:
            raise ValueError(f"{fp}: {m} frames not divisible by 3")
        if not _filter_status:
            return m // 3
        n = 0
        for k in range(0, m, 3):
            r, s, p = t[k], t[k + 1], t[k + 2]
            for a in (r, s, p):
                if "status" not in a.info:
                    raise ValueError(
                        f"{fp} group@{k} side={a.info.get('side')}: "
                        f"missing 'status' in atoms.info "
                        f"(use --no-status-filter if pre-filtered)"
                    )
            if all(
                str(a.info["status"]).startswith("converged")
                for a in (r, s, p)
            ):
                n += 1
    return n


def lpt_partition(files_with_counts, n_bins):
    """Greedy LPT: assign heaviest first to least-loaded bin."""
    items = sorted(files_with_counts, key=lambda x: -x[1])
    bins = [[] for _ in range(n_bins)]
    loads = [0] * n_bins
    for fp, c in items:
        i = loads.index(min(loads))
        bins[i].append(fp)
        loads[i] += c
    return bins, loads


def worker(args):
    files, shard_path, csv_path, start_msid, task_name = args
    next_id = start_msid
    written = 0
    skipped = 0
    since_last_report = 0
    rows_since_commit = 0
    t0 = time.perf_counter()

    def _flush_progress():
        nonlocal since_last_report
        if _progress_counter is not None and since_last_report:
            with _progress_counter.get_lock():
                _progress_counter.value += since_last_report
            since_last_report = 0

    cf = open(csv_path, "w", newline="")
    cw = csv.writer(cf)

    # Manually manage the LMDB context so we can commit-and-reopen
    # periodically. ASE's __enter__/__exit__ commits the active txn and
    # closes the env; the next __enter__ opens a fresh txn into the same
    # file (nextid is read from disk).
    db_ctx = connect(shard_path, type="aselmdb")
    db = db_ctx.__enter__()
    _patch_db_ids(db)

    def _cycle_db():
        """Commit current txn, free its pages, reopen a fresh one."""
        nonlocal db, db_ctx, rows_since_commit
        db_ctx.__exit__(None, None, None)
        db_ctx = connect(shard_path, type="aselmdb")
        db = db_ctx.__enter__()
        _patch_db_ids(db)
        rows_since_commit = 0

    try:
        for fp in files:
            abs_path = str(Path(fp).resolve())
            with Trajectory(fp) as t:
                m = len(t)
                if m % 3 != 0:
                    _flush_progress()
                    cf.close(); db_ctx.__exit__(None, None, None)
                    return shard_path, written, skipped, time.perf_counter() - t0, \
                           f"{fp}: {m} frames not divisible by 3"
                for k in range(0, m, 3):
                    r, s, p = t[k], t[k + 1], t[k + 2]

                    # Optional triplet-info validation: only assert fields
                    # that all three frames actually carry.
                    if all("src_index" in a.info for a in (r, s, p)):
                        if not (r.info["src_index"] == s.info["src_index"]
                                == p.info["src_index"]):
                            _flush_progress()
                            cf.close(); db_ctx.__exit__(None, None, None)
                            return shard_path, written, skipped, time.perf_counter() - t0, \
                                   f"{fp} group@{k}: src_index mismatch"
                    if all("side" in a.info for a in (r, s, p)):
                        if (r.info["side"] != -1 or s.info["side"] != 0
                                or p.info["side"] != 1):
                            _flush_progress()
                            cf.close(); db_ctx.__exit__(None, None, None)
                            return shard_path, written, skipped, time.perf_counter() - t0, \
                                   f"{fp} group@{k}: side != (-1,0,+1)"

                    # Optional status filter
                    if _filter_status:
                        for a in (r, s, p):
                            if "status" not in a.info:
                                _flush_progress()
                                cf.close(); db_ctx.__exit__(None, None, None)
                                return shard_path, written, skipped, time.perf_counter() - t0, \
                                       (f"{fp} group@{k} side={a.info.get('side')}: "
                                        f"missing 'status'")
                        if not all(
                            str(a.info["status"]).startswith("converged")
                            for a in (r, s, p)
                        ):
                            skipped += 1
                            continue

                    # Cycle on triple boundaries so a commit can never
                    # interrupt a partial R-S-P group.
                    if rows_since_commit >= _commit_every_rows:
                        _cycle_db()
                    for atoms in (r, s, p):
                        atoms.info["task_name"] = task_name
                        atoms.info["ms_id"] = next_id
                        kvp = {"task_name": task_name, "ms_id": next_id}
                        if "src_index" in atoms.info:
                            kvp["src_index"] = int(atoms.info["src_index"])
                        if "side" in atoms.info:
                            kvp["side"] = int(atoms.info["side"])
                        if "status" in atoms.info:
                            kvp["status"] = str(atoms.info["status"])
                        db.write(
                            atoms,
                            **kvp,
                            data={
                                "info": dict(atoms.info),
                                "traj_path": abs_path,
                            },
                        )
                        cw.writerow([next_id, abs_path])
                        next_id += 1
                        written += 1
                        rows_since_commit += 1
                        since_last_report += 1
                        if since_last_report >= PROGRESS_BATCH:
                            _flush_progress()
        _flush_progress()
    finally:
        cf.close()
        db_ctx.__exit__(None, None, None)

    return shard_path, written, skipped, time.perf_counter() - t0, None


def detect_resume_offset(map_path: Path) -> int:
    if not map_path.exists() or map_path.stat().st_size == 0:
        return 0
    with open(map_path, newline="") as f:
        rows = list(csv.reader(f))
    if not rows or rows[0] != ["global_index", "traj_path"]:
        sys.exit(
            f"ERROR: {map_path} header invalid; expected "
            f"['global_index','traj_path']"
        )
    data_rows = rows[1:]
    if not data_rows:
        return 0
    indices = [int(r[0]) for r in data_rows]
    if indices != list(range(len(indices))):
        sys.exit(f"ERROR: {map_path} global_index column not dense from 0")
    return len(indices)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("task_name")
    ap.add_argument("--trajes-dir", default="DoubleMinimization_trajes")
    ap.add_argument("--shard-dir", default="aselmdb_no-EF",
                    help="output dir for this cluster's shards")
    ap.add_argument("--map-path", default="aselmdb_no-EF/global_index_map.csv",
                    help="master CSV; auto-resumes from max(global_index)+1")
    ap.add_argument("--shard-prefix", required=True,
                    help="filename prefix for output shards "
                         "(e.g. 'htp_batch1' -> htp_batch1_000.aselmdb)")
    # Default chosen for HPC parallel I/O: 32 is a sweet spot on typical
    # Lustre-backed clusters where each worker writes to its own LMDB file.
    # Going past 32 usually hits diminishing returns from filesystem
    # contention; ncpu can be 200+ on big nodes, which would oversaturate.
    # Capped at cpu_count() for small machines.
    ap.add_argument("-j", "--workers", type=int,
                    default=min(32, os.cpu_count() or 1))
    ap.add_argument("--glob", default="*.traj")
    ap.add_argument("--start-index", type=int, default=None,
                    help="override auto-resume (must match a fresh setup)")
    ap.add_argument("--commit-every-rows", type=int,
                    default=DEFAULT_COMMIT_EVERY_ROWS,
                    help="rows between LMDB commit-and-reopen "
                         f"(default {DEFAULT_COMMIT_EVERY_ROWS:,})")
    ap.add_argument("--no-status-filter", action="store_true",
                    help="skip the converged-status filter and the "
                         "missing-'status' check (use for datasets "
                         "that were already filtered upstream)")
    args = ap.parse_args()
    filter_status = not args.no_status_filter

    trajes_dir = Path(args.trajes_dir).resolve()
    shard_dir = Path(args.shard_dir).resolve()
    map_path = Path(args.map_path).resolve()

    if not trajes_dir.is_dir():
        sys.exit(f"ERROR: {trajes_dir} not a directory")
    shard_dir.mkdir(parents=True, exist_ok=True)
    existing = list(shard_dir.glob(f"{args.shard_prefix}_*.aselmdb"))
    if existing:
        sys.exit(
            f"ERROR: {shard_dir} already contains "
            f"{args.shard_prefix}_*.aselmdb"
        )
    map_path.parent.mkdir(parents=True, exist_ok=True)

    if args.start_index is not None:
        if map_path.exists() and map_path.stat().st_size > 0:
            sys.exit(
                f"ERROR: --start-index given but {map_path} already has rows"
            )
        next_id_global = args.start_index
    else:
        next_id_global = detect_resume_offset(map_path)

    files = sorted(str(p) for p in trajes_dir.glob(args.glob))
    if not files:
        sys.exit(f"ERROR: no files matching {args.glob} in {trajes_dir}")

    print(f"trajes-dir:       {trajes_dir}  ({len(files)} files)")
    print(f"shard-dir:        {shard_dir}")
    print(f"map-path:         {map_path}")
    print(f"workers:          {args.workers}")
    print(f"task_name:        {args.task_name}")
    print(f"resume from ms_id: {next_id_global:,}")
    print(flush=True)

    print(f"[step] counting triples per file "
          f"(filter_status={filter_status}, parallel)...", flush=True)
    t0 = time.perf_counter()
    pool_size = min(args.workers, len(files))
    with mp.Pool(pool_size, initializer=_init_count_worker,
                 initargs=(filter_status,)) as pool:
        try:
            counts = pool.map(count_filtered_triples, files)
        except Exception as e:
            sys.exit(f"ERROR during count: {e}")
    total_triples = sum(counts)
    total_frames = total_triples * 3
    print(f"  total: {total_triples:,} triples ({total_frames:,} frames) "
          f"in {time.perf_counter()-t0:.1f}s", flush=True)

    bins, loads = lpt_partition(list(zip(files, counts)), args.workers)
    bins_loads = [(b, ld) for b, ld in zip(bins, loads) if b]
    print(f"[step] worker assignments ({len(bins_loads)} workers active):")
    for i, (b, ld) in enumerate(bins_loads):
        print(f"  worker {i:2d}: {len(b):3d} files, {ld*3:,} frames",
              flush=True)

    cumulative = next_id_global
    starts = []
    for b, ld in bins_loads:
        starts.append(cumulative)
        cumulative += ld * 3
    final_next_id = cumulative
    print(f"[step] this cluster will write ms_id "
          f"[{next_id_global:,}, {final_next_id - 1:,}]")

    tasks = []
    for i, ((b, ld), st) in enumerate(zip(bins_loads, starts)):
        base = f"{args.shard_prefix}_{i:03d}"
        tasks.append((
            b,
            str(shard_dir / f"{base}.aselmdb"),
            str(shard_dir / f"{base}.csv"),
            st,
            args.task_name,
        ))

    print(f"[step] launching {len(tasks)} parallel workers...", flush=True)
    expected_frames = sum(ld for _, ld in bins_loads) * 3
    counter = mp.Value("Q", 0)
    stop_progress = threading.Event()

    def _progress_loop():
        last_n = 0
        last_t = time.perf_counter()
        while not stop_progress.is_set():
            stop_progress.wait(2.0)
            if stop_progress.is_set():
                break
            now = time.perf_counter()
            cur = counter.value
            dn = cur - last_n
            dt = now - last_t
            rate = dn / dt if dt > 0 else 0.0
            pct = 100.0 * cur / expected_frames if expected_frames else 100.0
            eta = (expected_frames - cur) / rate / 60 if rate > 0 else float("inf")
            print(f"  [progress] {cur:,} / {expected_frames:,} "
                  f"({pct:.1f}%) | {rate:,.0f} frames/sec "
                  f"| ETA {eta:.1f} min", flush=True)
            last_n, last_t = cur, now

    pt = threading.Thread(target=_progress_loop, daemon=True)
    pt.start()

    t_start = time.perf_counter()
    with mp.Pool(len(tasks), initializer=_init_worker,
                 initargs=(counter, args.commit_every_rows,
                           filter_status)) as pool:
        results = pool.map(worker, tasks)
    wall = time.perf_counter() - t_start

    stop_progress.set()
    pt.join(timeout=3.0)
    print(f"  [progress] {counter.value:,} / {expected_frames:,} (final)",
          flush=True)

    errors = [r for r in results if r[-1] is not None]
    if errors:
        for sh, w, sk, dt, msg in errors:
            print(f"  WORKER FAILED ({Path(sh).name}): {msg}", flush=True)
        sys.exit("ERROR: at least one worker failed; output is incomplete")

    total_written = sum(r[1] for r in results)
    total_skipped = sum(r[2] for r in results)
    print(f"phase wall: {wall:.1f}s ({wall/60:.2f} min)")
    print(f"frames written:  {total_written:,}")
    print(f"triples skipped: {total_skipped:,} (={3*total_skipped:,} frames)")
    print(f"throughput: {total_written/wall:,.0f} frames/sec")

    print(flush=True)
    print("[step] appending per-worker CSV fragments to master CSV...",
          flush=True)
    write_header = not map_path.exists() or map_path.stat().st_size == 0
    with open(map_path, "a", newline="") as master:
        mw = csv.writer(master)
        if write_header:
            mw.writerow(["global_index", "traj_path"])
        for i, (sh, w, sk, dt, _err) in enumerate(results):
            frag = Path(sh).with_suffix(".csv")
            with open(frag, newline="") as f:
                fr = csv.reader(f)
                for row in fr:
                    mw.writerow(row)
            frag.unlink()
    print(f"  master CSV now has rows up to ms_id "
          f"{final_next_id - 1:,}")

    print()
    print("per-worker:")
    for sh, w, sk, dt, _ in results:
        size_mb = os.path.getsize(sh) / 1e6
        print(f"  {Path(sh).name}: {w:7,} frames, {sk:5,} triples skipped, "
              f"{dt:6.1f}s, {size_mb:7.1f} MB")


if __name__ == "__main__":
    main()
