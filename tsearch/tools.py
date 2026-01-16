import configparser


def load_calculator(config_dict):
    if config_dict["Main"]["Calculator"] == "FAIRChemCalculator":
        from fairchem.core import FAIRChemCalculator
        calc = FAIRChemCalculator.from_model_checkpoint(
            config_dict["FAIRChemCalculator"]["model_name_or_path"],
            task_name = config_dict["FAIRChemCalculator"]["task_name"],
            device = config_dict["FAIRChemCalculator"]["device"],
            )
    elif config_dict["Main"]["Calculator"] == "VaspInteractive":
        raise NotImplementedError
    elif config_dict["Main"]["Calculator"] == "Vasp":
        raise NotImplementedError
    return calc


def load_method(config_dict):
    method_name = config_dict["Main"]["method"]
    if method_name.lower() == "neb":
        from tsearch.nebopt import nebopt as method
    elif method_name.lower() == "dimer":
        from tsearch.dimeropt import dimeropt as method
    elif method_name.lower() == "minimization":
        from tsearch.geomopt import geomopt as method
    else:
        raise NotImplementedError(
            f"Method '{method_name}' is not implemented. Only NEB, Dimer, and Minimization are supported."
        )
    return method


def load_optimizer(config_dict):
    optimizer_name = config_dict["Main"]["Optimizer"]
    if optimizer_name.lower() == "mdmin":
        from ase.optimize import MDMin as Optimizer
    elif optimizer_name.lower() == "bfgs":
        from ase.optimize import BFGS as Optimizer
    elif optimizer_name.lower() == "lbfgs":
        from ase.optimize import LBFGS as Optimizer
    elif optimizer_name.lower() == "fire":
        from ase.optimize import FIRE as Optimizer
    else:
        raise NotImplementedError(
            f"Method '{optimizer_name}' is not implemented. Only MDMin, BFGS, LBFGS and FIRE are supported."
        )
    return Optimizer



def interpret_string(val):
    val = val.strip()

    if val.lower() == 'true': return True
    if val.lower() == 'false': return False

    try:
        return int(val)
    except ValueError:
        pass

    try:
        return float(val)
    except ValueError:
        pass

    # Only convert to list if the elements look like numbers or booleans
    if ' ' in val:
        parts = val.split()
        interpreted_parts = [interpret_string(p) for p in parts]
        return interpreted_parts

    return val


def configparse(input_path):
    config = configparser.ConfigParser(inline_comment_prefixes='#')
    config.optionxform = str # Preserves case sensitivity
    config.read(input_path)
    return config


def parse_inputfile(input_path):
    config = configparse(input_path)
    
    return {
        section: {
            k: interpret_string(v) for k, v in config.items(section)
        }
        for section in config.sections()
    }