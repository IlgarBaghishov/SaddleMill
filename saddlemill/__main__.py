import os
import concurrent.futures
from ase.io import Trajectory
from itertools import groupby
from contextlib import nullcontext
from saddlemill.init_function import init_function
from saddlemill.tools import (save_ordered_traj_names, read_ordered_traj_names,
                            clean_up_files, load_and_sanitize, passes_input_filter,
                            extract_previous_results)
from saddlemill.config import (load_config, load_method, get_trajes_and_indices,
                            create_results_directories, get_remaining_trajes,
                            get_flux_resources, archive_and_clean_csvs,
                            archive_and_clean_outputs, build_redo_info)


def check_and_print_status(futures, total):
    done, futures = concurrent.futures.wait(futures, timeout=0.1)
    if len(done)!=0:
        print(f"{len(futures)} REMAINING  ---  {total-len(futures)} FINISHED  ---  {total} TOTAL")
    return futures


def main():

    config_dict = load_config("config.ini")
    print(config_dict,"\n")

    method = load_method(config_dict)
    trajes_and_idxs = get_trajes_and_indices(config_dict)
    can_resume = os.path.exists('traj_files_ordered.json')
    previous_results = {}
    redo_info = {}
    if can_resume:
        trajes_and_idxs_old = read_ordered_traj_names()
        if trajes_and_idxs != trajes_and_idxs_old:
            raise ValueError("Provided dirpath creates a different trajes_and_idxs. I can't resume.")
        job_IDs, trajes_and_idxs = get_remaining_trajes(trajes_and_idxs, config_dict)

        # Build per-job redo info (which subunits to redo)
        redo_info = build_redo_info(job_IDs, config_dict)

        # Extract previous output BEFORE archiving (output files still intact).
        # Always extract when redoing subunits: NEB needs sub-band endpoints,
        # DoubleMinimization needs the kept side for reaction check,
        # Dimer needs previous structures only when continue_from_result=True.
        if redo_info:
            print(f"Extracting previous results for {len(redo_info)} jobs...", flush=True)
            previous_results = extract_previous_results(list(redo_info.keys()), config_dict, redo_info)
            print(f"  Extracted {len(previous_results)} of {len(redo_info)} results.", flush=True)

        from saddlemill.config import _normalize_run_jobs
        categories_to_clean = _normalize_run_jobs(config_dict["Main"]["run_jobs"])
        cleaned = archive_and_clean_csvs(config_dict, job_IDs, categories_to_clean)
        archive_and_clean_outputs(config_dict, cleaned)
        clean_up_files(config_dict)
    else:
        job_IDs = list(range(len(trajes_and_idxs)))
        save_ordered_traj_names(trajes_and_idxs)
        create_results_directories(config_dict)

    if config_dict["Main"]["executorlib"]:
        from executorlib import FluxJobExecutor

        max_workers, cores, gpus_per_core, threads_per_core = get_flux_resources(config_dict)

        executor = FluxJobExecutor(
            flux_log_files = True,
            max_workers = max_workers,
            block_allocation = True,
            init_function = init_function,
            restart_limit = config_dict["Main"]["restart_limit"],
            resource_dict = {
                "cores": cores,
                "gpus_per_core": gpus_per_core,
                "threads_per_core": threads_per_core,
                "num_nodes": 1,
                "error_log_file": "error",
                "cwd": os.getcwd(),
            }
        )
        # 'exe' will be the executor instance
        get_submitter = lambda exe: exe.submit
    else:
        # Serial Mode: Use empty context and a dummy submitter that runs immediately
        init_data = init_function()
        executor = nullcontext()
        get_submitter = lambda _: lambda fn, *args, **kwargs: fn(*args, **init_data, **kwargs)


    # 2. Unified Execution Loop
    input_format = config_dict["Main"].get("input_format", "traj")
    if input_format == "lmdb":
        # SinglePoint + LMDB: read rows via ase.db (fairchem.core.datasets
        # registers the aselmdb backend).
        import fairchem.core.datasets  # noqa: F401
        from ase.db import connect as _lmdb_connect

        def open_src(path):
            return _lmdb_connect(path, type='aselmdb', readonly=True)

        def load_item(src, i, j):
            rows = [src.get(rid) for rid in range(i, j)]
            atoms_list = []
            extras = []
            for r in rows:
                atoms = r.toatoms()
                # row.toatoms() doesn't populate atoms.info from row.data,
                # so we lift it across explicitly. This keeps existing
                # orig_info / status conventions readable by passes_input_filter.
                atoms.info = dict(r.data.get("info", {}))
                atoms_list.append(atoms)
                extras.append({"kvp": dict(r.key_value_pairs),
                               "row_data": dict(r.data)})
            images = atoms_list if len(atoms_list) > 1 else atoms_list[0]
            return images, {"extras": extras}

        def close_src(_):
            pass
    else:
        def open_src(path):
            return Trajectory(path, 'r')

        def load_item(src, i, j):
            return load_and_sanitize(src, i, j), {}

        def close_src(src):
            src.close()

    with executor as exe:
        submitter = get_submitter(exe)
        futures = []
        idx = 0
        submitted = 0

        for src_path, group in groupby(trajes_and_idxs, key=lambda x: x[0]):
            src = open_src(src_path)
            try:
                for _, i, j in group:
                    job_id = job_IDs[idx]
                    images, extra = load_item(src, i, j)
                    if not passes_input_filter(images, config_dict):
                        idx += 1
                        continue
                    try:
                        f = submitter(method, job_id, config_dict, images,
                                      continuation_data=previous_results.get(job_id),
                                      entries_to_run=redo_info.get(job_id),
                                      **extra)
                        if config_dict["Main"]["executorlib"]: futures.append(f)
                        submitted += 1
                    except Exception as e:
                        print(f"CRITICAL ERROR on job {idx} ({src_path}): {e}")
                        # In serial mode, we catch it and move on.
                        # In parallel mode, 'submitter' usually doesn't raise immediately, so this is safe.
                    idx += 1
            finally:
                close_src(src)

        print(f"Scanned {idx} input frame(s); submitted {submitted} "
              f"(skipped {idx - submitted} by input_statuses).", flush=True)

        if config_dict["Main"]["executorlib"]:
            while len(futures):
                futures = check_and_print_status(futures, submitted)


if __name__ == "__main__":
    main()
