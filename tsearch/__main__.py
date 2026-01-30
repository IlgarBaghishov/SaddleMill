import concurrent.futures
import time
from ase.io import Trajectory
from itertools import groupby
from contextlib import nullcontext
from tsearch.tools import save_ordered_traj_names, read_ordered_traj_names, clean_up_files
from tsearch.config import load_config, load_method, get_trajes_and_indices, create_results_directories, get_remaining_trajes


def check_and_print_status(futures, total):
    done, futures = concurrent.futures.wait(futures, timeout=0.1)
    if len(done)!=0:
        print(f"{len(futures)} REMAINING  ---  {total-len(futures)} FINISHED  ---  {total} TOTAL")
    return futures


def main():

    config_dict = load_config("config.ini")
    print(config_dict,"\n")

    method = load_method(config_dict)
    st = time.time()
    trajes_and_idxs = get_trajes_and_indices(config_dict)
    print("it took", time.time()-st)
    if config_dict["Main"]["resume"]:
        trajes_and_idxs_old = read_ordered_traj_names()
        if trajes_and_idxs != trajes_and_idxs_old:
            raise ValueError("Provided dirpath creates a different trajes_and_idxs. I can't resume.")
        print(trajes_and_idxs[:10])
        job_IDs, trajes_and_idxs = get_remaining_trajes(trajes_and_idxs, config_dict)
        print(trajes_and_idxs[:10])
        clean_up_files()
    else:
        job_IDs = list(range(len(trajes_and_idxs)))
        create_results_directories(config_dict)
        save_ordered_traj_names(trajes_and_idxs)

    if config_dict["Main"]["executorlib"]:
        from executorlib import FluxJobExecutor
        from flux import Flux, resource

        handle = Flux()
        rs = resource.status.ResourceStatusRPC(handle).get()
        rl = resource.list.resource_list(handle).get()
        all_ncores = rl.all.ncores
        all_ngpus = rl.all.ngpus
        print(all_ncores,all_ngpus)

        jobs_per_gpu = config_dict["Main"]["jobs_per_gpu"]
        gpus_per_core = 1 if jobs_per_gpu == 1 else 0
        cpus_per_job = all_ncores // (all_ngpus*jobs_per_gpu) - 1

        executor = FluxJobExecutor(
            flux_log_files=True,
            max_workers=all_ngpus * jobs_per_gpu,
            block_allocation=True,
            resource_dict={
                "cores": 1, 
                "gpus_per_core": gpus_per_core, 
                "threads_per_core": cpus_per_job, 
                "num_nodes": 1, 
                "error_log_file": "error"
            }
        )
        # 'exe' will be the executor instance
        get_submitter = lambda exe: exe.submit  
    else:
        # Serial Mode: Use empty context and a dummy submitter that runs immediately
        executor = nullcontext()
        # 'exe' will be None; we ignore it and run the function directly
        get_submitter = lambda _: lambda fn, *args: fn(*args)

    # 2. Unified Execution Loop
    with executor as exe:
        submitter = get_submitter(exe)
        futures = []
        idx = 0

        for traj_name, group in groupby(trajes_and_idxs, key=lambda x: x[0]):
            traj = Trajectory(traj_name, 'r')
            for _, i, j in group:
                images = list(traj[i:j])
                f = submitter(method, job_IDs[idx], config_dict, images)
                if config_dict["Main"]["executorlib"]: futures.append(f)
                idx += 1
            traj.close()

        if config_dict["Main"]["executorlib"]:
            while len(futures):
                futures = check_and_print_status(futures, idx)


if __name__ == "__main__":
    main()
