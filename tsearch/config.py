import configparser
import os, sys, glob, copy

class ConfigManager:
    # 1. Define your Safe Defaults here
    DEFAULTS = {
        "Main": {
            "executorlib": True,
            "method": None,  # requires user input
            "dir_path": ".",
            "Optimizer": "MDMin",
            "fmax": 0.05,
            "steps": 1000,
            "Calculator": "FAIRChemCalculator",
            "jobs_per_gpu": 4,
        },
        "FAIRChemCalculator": {
            "device": 'cuda',
            "model_name_or_path": 'uma-s-1p1',
            "task_name": None,
        },
        "ourMinimization": {
        },
        "ourDoubleMinimization": {
        },
        "ourNEB": {
            "relax_endpoints": True,
            "endpoint_relax_fmax": 0.01,
            "endpoint_relax_maxsteps": 500,
            "interpolate_method": "ocp_idpp",
            "num_frames": 10,
            "batch_size": 8
        },
        "ourDimer": {
            "dataset_type": None,
            "num_attempts": 3,
            "delocalization_threshold": 0.8,
            "extension_check_fmax": 0.4,
            "extension_check_curvature": -0.2,
        },
    }

    def __init__(self, config_file="config.ini"):
        self._config = copy.deepcopy(self.DEFAULTS)
        self.user_config_file = config_file
        
        # Load user config if it exists
        if os.path.exists(self.user_config_file):
            self._load_from_file()
        else:
            print(f"Warning: {self.user_config_file} not found. Using default parameters.")

    def _load_from_file(self):
        """Loads the .ini file and updates the config dictionary with type conversion."""
        parser = configparser.ConfigParser(inline_comment_prefixes='#')
        parser.optionxform = str # Preserves case sensitivity
        parser.read(self.user_config_file)

        for section in parser.sections():
            if section not in self._config:
                self._config[section] = {}

            for key, value in parser.items(section):
                # Attempt to convert to int/float/bool, otherwise keep as string
                parsed_value = self._parse_value(value)
                self._config[section][key] = parsed_value

    def _parse_value(self, val):
        """
        Recursively interprets strings into bools, numbers, or lists.
        Matches logic of original `interpret_string`.
        """
        # 1. Strip whitespace
        if isinstance(val, str):
            val = val.strip()

        # 2. Boolean check
        if val.lower() == 'true': return True
        if val.lower() == 'false': return False

        # 3. Integer check
        try:
            return int(val)
        except ValueError:
            pass

        # 4. Float check
        try:
            return float(val)
        except ValueError:
            pass

        # 5. List check (Space-delimited)
        # Only split if it looks like a list of values (contains space)
        if ' ' in val:
            parts = val.split()
            # Recursively interpret each part
            return [self._parse_value(p) for p in parts]

        # 6. Fallback: Return original string
        return val

    def __getitem__(self, key):
        """Allow dict-like access: config['Main']"""
        return self._config.get(key, {})

    def get(self, key, fallback=None):
        """
        Standard dict-like get. 
        Usage: config.get("Main", {}) 
        """
        return self._config.get(key, fallback)

    def get_value(self, section, key, fallback=None):
        """
        Specific helper to get a value deep inside a section.
        Usage: config.get_value("Main", "fmax", 0.05)
        """
        return self._config.get(section, {}).get(key, fallback)

    @property
    def as_dict(self):
        """Return the raw dictionary."""
        return self._config

    def __str__(self):
            """Enables pretty printing via print(config)"""
            import json
            # default=str handles objects that aren't natively JSON serializable
            return json.dumps(self._config, indent=4, default=str)

# --- Helper function to mimic your old parse_inputfile ---
def load_config(path="config.ini"):
    return ConfigManager(path)



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
    if method_name is None:
            raise ValueError("Configuration error: 'Main' -> 'method' is not set. Please specify a method (e.g., 'minimization') in config.ini")
    if method_name.lower() == "neb":
        from tsearch.nebopt import nebopt as method
    elif method_name.lower() == "dimer":
        from tsearch.dimeropt import dimeropt as method
    elif method_name.lower() == "minimization":
        from tsearch.geomopt import geomopt as method
    elif method_name.lower() == "doubleminimization":
        from tsearch.geomopt import doublegeomopt as method
    else:
        raise NotImplementedError(
            f"Method '{method_name}' is not implemented. Only NEB, Dimer, Minimization, and DoubleMinimization are supported."
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


def get_all_traj_names(config_dict):
    main_cfg = config_dict.get("Main", {})
    dir_path = main_cfg.get("dir_path", ".")
    input_pattern = os.path.join(dir_path, "*.traj")
    all_traj_files = sorted(glob.glob(input_pattern))
    return all_traj_files
