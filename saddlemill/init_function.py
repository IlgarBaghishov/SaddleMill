import os
import socket
import traceback
from saddlemill.config import load_config, load_calculator, load_optimizer

def init_function(executorlib_worker_id=None):
    try:
        config_dict = load_config("config.ini")

        is_gpu_job = (config_dict["Main"]["Calculator"] not in ("Vasp", "VaspInteractive")
                      and config_dict[config_dict["Main"]["Calculator"]].get("device") == "cuda")

        if config_dict["Main"]["executorlib"] == True and config_dict["Main"]["jobs_per_gpu"] != 1:
            if is_gpu_job:
                from flux import Flux, resource
                handle = Flux()
                rset = resource.list.resource_list(handle).get().all
                node_ngpus_list = [[str(rset.copy_ranks(str(i)).nodelist), rset.copy_ranks(str(i)).ngpus] for i in range(rset.nnodes)]
                gpu_ID = executorlib_worker_id

                for i in range(len(node_ngpus_list)):
                    node, ngpus = node_ngpus_list[i]
                    if gpu_ID < config_dict['Main']['jobs_per_gpu']*ngpus:
                        break
                    else:
                        gpu_ID -= config_dict['Main']['jobs_per_gpu']*ngpus
                physical_gpu = gpu_ID % ngpus
                mps_pipe = f"/tmp/mps_{physical_gpu}"
                if os.path.exists(os.path.join(mps_pipe, "control")):
                    os.environ["CUDA_MPS_PIPE_DIRECTORY"] = mps_pipe
                    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
                else:
                    os.environ["CUDA_VISIBLE_DEVICES"] = str(physical_gpu)

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
            calc_kwargs = dict(config_dict[config_dict["Main"]["Calculator"]])
            # SinglePoint + compute_hessian needs the model to expose a hessian
            # output, which is an InferenceSettings flag rather than a plain
            # kwarg. Only ever set for SinglePoint: enabling it makes EVERY force
            # call compute a full Hessian, which would slow a Dimer/Sella search
            # by ~100x for no benefit.
            if (config_dict["Main"]["method"] == "SinglePoint"
                    and config_dict.get("ourSinglePoint", {}).get("compute_hessian")):
                from fairchem.core.units.mlip_unit.api.inference import InferenceSettings
                task = calc_kwargs.get("task_name")
                calc_kwargs["inference_settings"] = InferenceSettings(
                    predict_untrained_hessian={task} if task else set())
            calc = calc(**calc_kwargs)
        Optimizer = load_optimizer(config_dict)

        return {"calc": calc, "Optimizer": Optimizer, "consecutive_errors": [0]}

    except Exception as e:
        print(f"Worker {executorlib_worker_id} FAILED during init_function: {e}", flush=True)
        print(f"\nTraceback details:\n{traceback.format_exc()}", flush=True)
        raise