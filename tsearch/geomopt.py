import sys, os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import numpy as np
from ase.io import read
from ase.optimize import FIRE
from ase.filters import FrechetCellFilter
from tsearch.tools import parse_inputfile, load_calculator, load_optimizer
from flux import Flux, resource

handle = Flux()
rs = resource.status.ResourceStatusRPC(handle).get()
rl = resource.list.resource_list(handle).get()
all_ncores = rl.all.ncores
all_ngpus = rl.all.ngpus
print("in geomopt", all_ncores, all_ngpus)


config_dict = parse_inputfile("config.ini")
calc = load_calculator(config_dict)
Optimizer = load_optimizer(config_dict)

images = read(config_dict["ourMinimization"]["images_path"],index=":")


def geomopt(i, config_dict, executorlib_worker_id=None):

    rank = executorlib_worker_id
    atoms = images[i]
    atoms.calc = calc

    opt = Optimizer(FrechetCellFilter(atoms), logfile=f'optimization{rank}_{i}.log', **config_dict[config_dict["Main"]["Optimizer"]])
    opt.run(fmax=config_dict["Main"]["fmax"], steps=config_dict["Main"]["steps"])

