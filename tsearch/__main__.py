import os
import concurrent.futures
from ase.io import Trajectory
from itertools import groupby
from contextlib import nullcontext
from tsearch.init_function import init_function
from tsearch.tools import save_ordered_traj_names, read_ordered_traj_names, clean_up_files
from tsearch.config import load_config, load_method, get_trajes_and_indices, create_results_directories, get_remaining_trajes, get_flux_resources


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
    method_name = config_dict["Main"]["method"]
    status_dir = f"{method_name}_status_csvs"
    import glob
    can_resume = (config_dict["Main"]["resume"]
                  and os.path.exists('traj_files_ordered.json')
                  and glob.glob(os.path.join(status_dir, "*.csv")))
    if can_resume:
        trajes_and_idxs_old = read_ordered_traj_names()
        if trajes_and_idxs != trajes_and_idxs_old:
            raise ValueError("Provided dirpath creates a different trajes_and_idxs. I can't resume.")
        job_IDs, trajes_and_idxs = get_remaining_trajes(trajes_and_idxs, config_dict)
        clean_up_files()
    else:
        job_IDs = list(range(len(trajes_and_idxs)))
        save_ordered_traj_names(trajes_and_idxs)
        create_results_directories(config_dict, exist_ok=config_dict["Main"]["resume"])

    if config_dict["Main"]["executorlib"]:
        from executorlib import FluxJobExecutor

        max_workers, cores, gpus_per_core, threads_per_core = get_flux_resources(config_dict)

        executor = FluxJobExecutor(
            flux_log_files = True,
            max_workers = max_workers,
            block_allocation = True,
            init_function = init_function,
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
    with executor as exe:
        submitter = get_submitter(exe)
        futures = []
        idx = 0

        for traj_name, group in groupby(trajes_and_idxs, key=lambda x: x[0]):
            traj = Trajectory(traj_name, 'r')
            for _, i, j in group:
                images = list(traj[i:j]) if j!=i+1 else traj[i]
                try:
                    f = submitter(method, job_IDs[idx], config_dict, images)
                    if config_dict["Main"]["executorlib"]: futures.append(f)
                except Exception as e:
                    print(f"CRITICAL ERROR on job {idx} ({traj_name}): {e}")
                    # In serial mode, we catch it and move on.
                    # In parallel mode, 'submitter' usually doesn't raise immediately, so this is safe.
                idx += 1
            traj.close()

        if config_dict["Main"]["executorlib"]:
            while len(futures):
                futures = check_and_print_status(futures, idx)


if __name__ == "__main__":
    main()
