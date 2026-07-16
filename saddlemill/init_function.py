import os
import socket
import traceback
import faulthandler
import signal
from saddlemill.config import load_config, load_calculator, load_optimizer

import fcntl

def _claim_local_gpu_slot(worker_id, ngpus, base="/tmp/sm_gpu"):
    os.makedirs(base, exist_ok=True)
    mine = os.path.join(base, f"worker_{worker_id}")
    if os.path.exists(mine):                       # restart of same worker → same slot
        with open(mine) as fh:
            return int(fh.read().strip()) % ngpus
    counter = os.path.join(base, "counter")
    fd = os.open(counter, os.O_RDWR | os.O_CREAT, 0o644)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        if os.path.exists(mine):                   # re-check under lock (raced restart)
            with open(mine) as fh:
                return int(fh.read().strip()) % ngpus
        os.lseek(fd, 0, 0)
        cur = os.read(fd, 32).decode().strip()
        idx = int(cur) if cur else 0
        os.ftruncate(fd, 0); os.lseek(fd, 0, 0)
        os.write(fd, str(idx + 1).encode())
        with open(mine, "w") as fh:
            fh.write(str(idx))
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)
    return idx % ngpus

def init_function(executorlib_worker_id=None):
    print(f"[init-enter] w={executorlib_worker_id}", flush=True)   # FIRST line
    try:
        log_file = open(f"frozen_trace_{executorlib_worker_id}.log", "w")
        faulthandler.register(signal.SIGUSR1, file=log_file)
        config_dict = load_config("config.ini")

        is_gpu_job = (config_dict["Main"]["Calculator"] not in ("Vasp", "VaspInteractive")
                      and config_dict[config_dict["Main"]["Calculator"]].get("device") == "cuda")

        if config_dict["Main"]["executorlib"] == True and config_dict["Main"]["jobs_per_gpu"] != 1:
            if is_gpu_job:
                from flux import Flux, resource
                handle = Flux()
                rset = resource.list.resource_list(handle).get().all
                node_ngpus_list = [[str(rset.copy_ranks(str(i)).nodelist),
                                    rset.copy_ranks(str(i)).ngpus] for i in range(rset.nnodes)]
                ngpus = node_ngpus_list[0][1]      # homogeneous 3×A100 nodes
                physical_gpu = _claim_local_gpu_slot(executorlib_worker_id, ngpus)

                mps_pipe = f"/tmp/mps_{physical_gpu}"

                # Pipe dir selects the physical GPU; client always sees device "0".
                # Set BOTH before any CUDA/torch init, or the driver ignores them.
                control = os.path.join(mps_pipe, "control")
                if not os.path.exists(control):
                    # Do NOT silently fall back to plain GPU 0 — that's the collapse bug.
                    raise RuntimeError(
                        f"Worker {executorlib_worker_id}: MPS control socket missing at "
                        f"{control}; refusing to fall back to GPU 0. Check run_phase MPS startup.")
                os.environ["CUDA_MPS_PIPE_DIRECTORY"] = mps_pipe
                os.environ["CUDA_VISIBLE_DEVICES"] = "0"

                import torch  # only AFTER env is set
                print(f"[assign] w={executorlib_worker_id} "
                      f"cuda_already_init={torch.cuda.is_initialized()} "
                      f"physical_gpu={physical_gpu} pipe={mps_pipe} "
                      f"CVD={os.environ.get('CUDA_VISIBLE_DEVICES')}", flush=True)
                # Print resource info for this worker
        hostname = socket.gethostname()
        cpus = sorted(os.sched_getaffinity(0))
        print(f"Worker {executorlib_worker_id} started on node {hostname}", flush=True)
        print(f"  CPUs: {cpus}", flush=True)
        if is_gpu_job:
            print(f"  CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'not set')}"
                  f"  MPS: {os.environ.get('CUDA_MPS_PIPE_DIRECTORY', 'off')}", flush=True)

        calc = load_calculator(config_dict)
        if config_dict["Main"]["Calculator"] not in ("Vasp", "VaspInteractive"):  # Then initialize, store on device memory and share the calculator object between structures
            calc = calc(**config_dict[config_dict["Main"]["Calculator"]])
        Optimizer = load_optimizer(config_dict)

        return {"calc": calc, "Optimizer": Optimizer, "consecutive_errors": [0]}

    except Exception as e:
        print(f"Worker {executorlib_worker_id} FAILED during init_function: {e}", flush=True)
        print(f"\nTraceback details:\n{traceback.format_exc()}", flush=True)
        raise
