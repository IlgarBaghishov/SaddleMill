from executorlib import FluxJobExecutor
from flux import Flux, resource
import concurrent.futures
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

    handle = Flux()
    rs = resource.status.ResourceStatusRPC(handle).get()
    rl = resource.list.resource_list(handle).get()
    all_ncores = rl.all.ncores
    all_ngpus = rl.all.ngpus
    print(all_ncores,all_ngpus)

    jobs_per_gpu = config_dict["Main"]["jobs_per_gpu"]
    cpus_per_job = all_ncores // (all_ngpus*jobs_per_gpu)

    method = load_method(config_dict)
    all_traj_files = get_all_traj_names(config_dict)
    save_ordered_traj_names(all_traj_files)

    # create results directories: status_csvs, trajes, debug_zips
    method_name = config_dict["Main"]["method"]
    pathlib.Path(f"{method_name}_status_csvs").mkdir(exist_ok=False)
    pathlib.Path(f"{method_name}_trajes").mkdir(exist_ok=False)
    pathlib.Path(f"{method_name}_debug_zips").mkdir(exist_ok=False)

    # with FluxJobExecutor(flux_log_files=True, max_workers=all_ngpus, block_allocation=True, resource_dict={"cores": 1, "gpus_per_core": 1, "threads_per_core":1, "num_nodes": 1}) as mps_exe:
    with FluxJobExecutor(flux_log_files=True,
                         max_workers=all_ngpus*jobs_per_gpu,
                         block_allocation=True,
                         resource_dict={"cores": 1, "gpus_per_core": 0, "threads_per_core":cpus_per_job, "num_nodes": 1}
                         ) as exe:  # pmi_mode="pmix"
    
        # command = "export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps-pipe-$USER\n"\
        #             "export CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-mps-log-$USER\n"\
        #             "mkdir -p $CUDA_MPS_PIPE_DIRECTORY\n"\
        #             "mkdir -p $CUDA_MPS_LOG_DIRECTORY\n"\
        #             "nvidia-cuda-mps-control -d"
        # # Starting daemons:
        # for i in range(all_ngpus):
        #     mps_exe.submit(os.system,command)
        
        futures = []
        submit_counter = 0
        for traj_name in all_traj_files:
            if method == "NEB":
                futures.append(exe.submit(method, submit_counter, config_dict, traj_name))
                submit_counter += 1
            else:
                with Trajectory(traj_name, 'r') as traj:
                    for atoms in traj:
                        futures.append(exe.submit(method, submit_counter, config_dict, atoms))
                        submit_counter += 1

        while len(futures):
            futures = check_and_print_status(futures, submit_counter)


if __name__ == "__main__":
    main()