import os
from tsearch.config import load_config, load_calculator, load_optimizer

def init_function(executorlib_worker_id=None):
    config_dict = load_config("config.ini")
    
    if config_dict["Main"]["executorlib"] == True and config_dict["Main"]["jobs_per_gpu"] != 1:
        if config_dict["Main"]["Calculator"] not in ("Vasp", "VaspInteractive") and config_dict[config_dict["Main"]["Calculator"]]["device"] == "cuda":
            from flux import Flux, resource
            handle = Flux()
            rset = resource.list.resource_list(handle).get().all
            node_ngpus_list = [[str(rset.copy_ranks(str(i)).nodelist), rset.copy_ranks(str(i)).ngpus] for i in range(rset.nnodes)]
            gpu_ID = executorlib_worker_id

            for i in range(len(node_ngpus_list)):
                node, ngpus = node_ngpus_list[i]
                if gpu_ID < config_dict['Main']['jobs_per_gpu']*ngpus:
                    print(f"Worker {executorlib_worker_id} assigned to GPU {gpu_ID} on node {node} with {ngpus} GPUs.")
                    break
                else:
                    gpu_ID -= config_dict['Main']['jobs_per_gpu']*ngpus
            os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_ID%ngpus)

    calc = load_calculator(config_dict)
    if config_dict["Main"]["Calculator"] not in ("Vasp", "VaspInteractive"):  # Then initialize, store on device memory and share the calculator object between structures
        calc = calc(**config_dict[config_dict["Main"]["Calculator"]])
    Optimizer = load_optimizer(config_dict)

    return {"calc": calc, "Optimizer": Optimizer}