from executorlib import FluxJobExecutor
from flux import Flux, resource
import concurrent.futures
from contextlib import nullcontext
from tsearch.tools import parse_inputfile, load_method, \
    get_all_traj_names, save_ordered_traj_names
import os, pathlib
from ase.io import Trajectory


def check_and_print_status(futures, total):
    done, futures = concurrent.futures.wait(futures, timeout=0.1)
    if len(done)!=0:
        print(f"{len(futures)} REMAINING  ---  {total-len(futures)} FINISHED  ---  {total} TOTAL")
    return futures


def main():

    config_dict = parse_inputfile("config.ini")

    method = load_method(config_dict)
    all_traj_files = get_all_traj_names(config_dict)
    save_ordered_traj_names(all_traj_files)

    # create results directories: status_csvs, trajes, debug_zips
    method_name = config_dict["Main"]["method"]
    pathlib.Path(f"{method_name}_status_csvs").mkdir(exist_ok=False)
    pathlib.Path(f"{method_name}_trajes").mkdir(exist_ok=False)
    pathlib.Path(f"{method_name}_debug_zips").mkdir(exist_ok=False)

    if config_dict["Main"]["executorlib"]:
        # Parallel Mode: Using executorlib
        handle = Flux()
        rs = resource.status.ResourceStatusRPC(handle).get()
        rl = resource.list.resource_list(handle).get()
        all_ncores = rl.all.ncores
        all_ngpus = rl.all.ngpus
        print(all_ncores,all_ngpus)

        jobs_per_gpu = config_dict["Main"]["jobs_per_gpu"]
        gpus_per_core = 1 if jobs_per_gpu == 1 else 0
        cpus_per_job = all_ncores // (all_ngpus*jobs_per_gpu) - 1

        # Use FluxJobExecutor and its submit method
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
        submit_counter = 0

        for traj_name in all_traj_files:
            if method_name == "NEB":
                f = submitter(method, submit_counter, config_dict, traj_name)
                if config_dict["Main"]["executorlib"]: futures.append(f)
                submit_counter += 1
            else:
                with Trajectory(traj_name, 'r') as traj:
                    for atoms in traj:
                        f = submitter(method, submit_counter, config_dict, atoms)
                        if config_dict["Main"]["executorlib"]: futures.append(f)
                        submit_counter += 1

        if config_dict["Main"]["executorlib"]:
            while len(futures):
                futures = check_and_print_status(futures, submit_counter)


if __name__ == "__main__":
    main()
