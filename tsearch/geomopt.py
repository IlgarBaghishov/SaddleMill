import sys, os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import numpy as np
from ase.io import read
from ase.optimize import FIRE
from ase.filters import FrechetCellFilter
from tsearch.tools import parse_inputfile, load_calculator


config_dict = parse_inputfile("config.ini")
calc = load_calculator(config_dict)

images = read(config_dict["ourMinimization"]["images_path"],index=":")


def geomopt(i, config_dict):
    atoms = images[i]
    atoms.calc = calc

    opt = FIRE(FrechetCellFilter(atoms), logfile=f'optimization{i}.log')
    opt.run(fmax=config_dict["Main"]["fmax"], steps=config_dict["Main"]["steps"])

