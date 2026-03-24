import configparser
import os, glob, copy, pathlib, zipfile
import pandas as pd
from ase.io import Trajectory

VALID_RUN_CATEGORIES = frozenset({"converged", "not_converged", "error", "not_started"})

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
            "device": 'cuda',
            "jobs_per_node": 1,  # this only used if device = 'cpu', otherwise jobs_per_gpu is used
            "jobs_per_gpu": 1,
            "run_jobs": "not_started",
            "zip": True,
            "max_consecutive_errors": 5,
            "restart_limit": 3,
        },
        "ourMinimization": {
            "relax_cell": False,
        },
        "ourDoubleMinimization": {
            "relax_cell": False,
        },
        "ourNEB": {
            "only_endpoints_in_input_traj": False,
            "images_location_in_input_traj": ":",  # can also be 0 or -1, meaning begining or end of file. This defines where are the initial endpoints or the band in the input traj
            "relax_endpoints": True,
            "endpoint_relax_Optimizer": None,
            "endpoint_relax_fmax": 0.01,
            "endpoint_relax_steps": 500,
            "interpolate_method": "ase_linear",
            "num_frames": 10,
            "batch_size": 4,
            "DNEB": False,
            "intermediate_minima": False,
            "intermediate_minima_min_depth": 0.01,
        },
        "ourDimer": {
            "dataset_type": None,
            "num_attempts": 3,
            "reaction_types": None, # Could be these if dataset_type is "bulk": vacancy hop_reuse hop_insert kickout_reuse kickout_insert ring initial_guess
            "num_attempts_per_type": 1,
            "ring_sizes": "3 4",
            "supercell": True,
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

        # Warn about unrecognized keys in sections we control
        for section, defaults in self.DEFAULTS.items():
            if section in self._config:
                unknown = set(self._config[section]) - set(defaults)
                for key in sorted(unknown):
                    print(f"Warning: Unrecognized key '{key}' in [{section}]. "
                          f"Valid keys: {sorted(defaults)}")

    def _parse_value(self, val):
        """
        Recursively interprets strings into bools, numbers, or lists.
        Matches logic of original `interpret_string`.
        """
        if isinstance(val, str):
            val = val.strip()

            if len(val) >= 2 and val[0] in ("'", '"') and val[-1] == val[0]:
                return val[1:-1]

        if str(val).lower() == 'true': return True
        if str(val).lower() == 'false': return False

        try:
            return int(val)
        except ValueError:
            pass

        try:
            return float(val)
        except ValueError:
            pass

        if isinstance(val, str) and ' ' in val:
            parts = val.split()
            return [self._parse_value(p) for p in parts]

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
    calculator_name = config_dict["Main"]["Calculator"]
    if calculator_name == "FAIRChemCalculator":
        from fairchem.core import FAIRChemCalculator
        calc = FAIRChemCalculator.from_model_checkpoint
    elif calculator_name == "VaspInteractive":
        from vasp_interactive import VaspInteractive as calc
    elif calculator_name == "Vasp":
        from ase.calculators.vasp import Vasp as calc
    else:
        raise ValueError(f"Unknown calculator: {calculator_name}")

    # To-do: implement this for Omat24 level DFT:
    # from fairchem.data.omat.vasp.sets import OMat24StaticSet
    # input_set = OMat24StaticSet(structure)
    # input_set.write_input(dir_name)
    # This should overwrite writeinputs function like this
    # class VaspNoWrite(Vasp):
    #     def write_input(self, atoms, properties=None, system_changes=None):
    #         pass
    return calc


def load_method(config_dict):
    method_name = config_dict["Main"]["method"]
    if method_name is None:
        raise ValueError("Configuration error: 'Main' -> 'method' is not set. Please specify a method (e.g., 'minimization') in config.ini")
    if method_name == "NEB":
        from tsearch.nebopt import nebopt as method
    elif method_name == "Dimer":
        from tsearch.dimeropt import dimeropt as method
    elif method_name == "Minimization":
        from tsearch.geomopt import geomopt as method
    elif method_name == "DoubleMinimization":
        from tsearch.geomopt import doublegeomopt as method
    else:
        raise NotImplementedError(
            f"Method '{method_name}' is not implemented. Only NEB, Dimer, Minimization, and DoubleMinimization are supported."
        )
    return method


def _load_optimizer(optimizer_name):
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


def load_optimizer(config_dict):
    Optimizer = _load_optimizer(config_dict["Main"]["Optimizer"])
    if config_dict["Main"]["method"] == "NEB":
        if config_dict["ourNEB"]["endpoint_relax_Optimizer"] is None:
            return Optimizer, Optimizer
        else:
            endpoint_relax_Optimizer = _load_optimizer(config_dict["ourNEB"]["endpoint_relax_Optimizer"])
            return endpoint_relax_Optimizer, Optimizer
    return Optimizer


def get_trajes_and_indices(config_dict):
    
    main_cfg = config_dict.get("Main", {})
    dir_path = main_cfg.get("dir_path", ".")
    input_pattern = os.path.join(dir_path, "*.traj")
    all_traj_files = sorted(glob.glob(input_pattern))
    
    if config_dict["ourNEB"]["images_location_in_input_traj"] in (":", -1):
        traj_lens = []
        for traj_name in all_traj_files:
            with Trajectory(traj_name, 'r') as traj:
                traj_lens.append(len(traj))

    if config_dict["Main"]["method"] == "NEB":
        if config_dict["ourNEB"]["only_endpoints_in_input_traj"]:
            nimages = 2
        else:
            nimages = config_dict["ourNEB"]["num_frames"]
    else:
        nimages = 1

    trajes_and_idxs = []
    for i, traj_name in enumerate(all_traj_files):
        if config_dict["ourNEB"]["images_location_in_input_traj"] == 0:
            trajes_and_idxs.append([traj_name, 0, nimages])
        elif config_dict["ourNEB"]["images_location_in_input_traj"] == -1:
            trajes_and_idxs.append([traj_name, traj_lens[i]-nimages, traj_lens[i]])
        elif config_dict["ourNEB"]["images_location_in_input_traj"] == ":":
            traj_len = traj_lens[i]
            if traj_len%nimages != 0: raise ValueError(f"Can't divide a traj file with {traj_len} atoms objects into batches of {nimages} atoms objects")
            for j in range(traj_len//nimages):
                trajes_and_idxs.append([traj_name, j*nimages, (j+1)*nimages])
    
    return trajes_and_idxs


def create_results_directories(config_dict, exist_ok=False):
    method_name = config_dict["Main"]["method"]
    dirs = [f"{method_name}_status_csvs", f"{method_name}_trajes", f"{method_name}_debug_zips"]
    if exist_ok:
        for d in dirs:
            if os.path.isdir(d) and os.listdir(d):
                raise RuntimeError(
                    f"Directory '{d}' already contains files from a previous run. "
                    f"Cannot start fresh. Delete output directories and "
                    f"traj_files_ordered.json first.")
    for d in dirs:
        pathlib.Path(d).mkdir(exist_ok=exist_ok)


def get_previous_job_status_df(config_dict):
    file_list = glob.glob(os.path.join(f"{config_dict['Main']['method']}_status_csvs/", "*.csv"))
    if not file_list:
        return pd.DataFrame()
    dfs = []
    for f in file_list:
        try:
            dfs.append(pd.read_csv(f, header=None))
        except pd.errors.EmptyDataError:
            continue
    if not dfs:
        return pd.DataFrame()
    return pd.concat(dfs, ignore_index=True)


def _normalize_run_jobs(run_jobs_value):
    """Convert parsed run_jobs config value into a set of job categories."""
    if isinstance(run_jobs_value, str):
        if run_jobs_value == "all":
            return set(VALID_RUN_CATEGORIES)
        cats = {run_jobs_value}
    elif isinstance(run_jobs_value, list):
        cats = {str(c) for c in run_jobs_value}
    else:
        raise ValueError(f"Invalid run_jobs value: {run_jobs_value!r}")
    invalid = cats - VALID_RUN_CATEGORIES
    if invalid:
        raise ValueError(
            f"Invalid run_jobs categories: {invalid}. "
            f"Valid: {sorted(VALID_RUN_CATEGORIES)} or 'all'")
    return cats


def _categorize_job(statuses):
    """Aggregate a list of status strings into a single job category."""
    if any(s.startswith("converged") for s in statuses):
        return "converged"
    if all(s.startswith("error") for s in statuses):
        return "error"
    return "not_converged"


def get_remaining_trajes(trajes_and_idxs, config_dict):
    categories_to_run = _normalize_run_jobs(config_dict["Main"]["run_jobs"])
    status_df = get_previous_job_status_df(config_dict)

    job_categories = {}
    if not status_df.empty:
        for job_id, group in status_df.groupby(status_df.iloc[:, 0]):
            job_categories[job_id] = _categorize_job(
                group.iloc[:, -1].astype(str).tolist()
            )

    remaining = []
    for idx, item in enumerate(trajes_and_idxs):
        if idx in job_categories:
            if job_categories[idx] in categories_to_run:
                remaining.append([idx, item])
        else:
            if "not_started" in categories_to_run:
                remaining.append([idx, item])

    if not remaining:
        return [], []
    job_IDs, trajes_and_idxs_out = zip(*remaining)
    return list(job_IDs), list(trajes_and_idxs_out)


def archive_and_clean_csvs(config_dict, job_ids):
    """Archive old CSVs and remove entries for jobs being re-run.

    Only triggers if any of the job_ids actually have existing CSV entries.
    Preserves all historical data in numbered zip archives while ensuring
    the active CSVs only contain current results for correct future filtering.
    """
    if not job_ids:
        return
    status_dir = f"{config_dict['Main']['method']}_status_csvs"
    csv_files = glob.glob(os.path.join(status_dir, "*.csv"))
    if not csv_files:
        return

    # Read all CSVs and check if any job_ids have existing entries
    job_ids_set = set(job_ids)
    csv_data = {}  # filepath -> DataFrame
    has_entries_to_clean = False
    for f in csv_files:
        try:
            df = pd.read_csv(f, header=None)
        except pd.errors.EmptyDataError:
            continue
        csv_data[f] = df
        if df.iloc[:, 0].isin(job_ids_set).any():
            has_entries_to_clean = True

    if not has_entries_to_clean:
        return  # All job_ids are not_started (no CSV entries) → pure append, skip archiving

    # 1. Archive: zip all current CSVs as previous_{N}.zip
    n = 0
    while os.path.exists(os.path.join(status_dir, f"previous_{n}.zip")):
        n += 1
    archive_path = os.path.join(status_dir, f"previous_{n}.zip")
    with zipfile.ZipFile(archive_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        for f in csv_files:
            zf.write(f, os.path.basename(f))

    # 2. Clean: remove entries for redone jobs, keep the rest
    for f, df in csv_data.items():
        filtered = df[~df.iloc[:, 0].isin(job_ids_set)]
        if filtered.empty:
            os.remove(f)
        else:
            filtered.to_csv(f, header=False, index=False)


def get_flux_resources(config_dict):
    from flux import Flux, resource

    handle = Flux()
    rset = resource.list.resource_list(handle).get().all
    all_ncores = rset.ncores
    all_ngpus = rset.ngpus
    nnodes = rset.nnodes
    print(f"Number of nodes: {nnodes}, total number of CPU cores: {all_ncores}, total number of GPUs: {all_ngpus}")

    jobs_per_gpu = config_dict["Main"]["jobs_per_gpu"]
    jobs_per_node = config_dict["Main"]["jobs_per_node"]

    if config_dict["Main"]["device"] == 'cuda':
        max_workers = all_ngpus * jobs_per_gpu
        gpus_per_core = 1 if jobs_per_gpu == 1 else 0
        cores = 1
        threads_per_core = all_ncores // max_workers # - 1
    elif config_dict["Main"]["device"] == 'cpu':
        max_workers = nnodes * jobs_per_node
        gpus_per_core = 0
        if config_dict["Main"]["Calculator"] in ("#Vasp", "#VaspInteractive"):
            cores = (all_ncores-1) // max_workers
            threads_per_core = 1
        else:
            cores = 1
            # threads_per_core = (all_ncores-1) // nnodes
            threads_per_core = all_ncores // nnodes
    else:
        raise ValueError("Only devices cuda and cpu available. Please set one of the two in Main section of config.ini")
    return max_workers, cores, gpus_per_core, threads_per_core
