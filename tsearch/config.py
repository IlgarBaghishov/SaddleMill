import configparser
import csv
import os, re, glob, copy, pathlib, zipfile
from ase.io import Trajectory

VALID_RUN_CATEGORIES = frozenset({"converged", "not_converged", "errored", "remaining"})
_RUN_CATEGORY_ALIASES = {"not_started": "remaining", "error": "errored"}

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
            "run_jobs": "remaining",
            "input_statuses": "all",
            "continue_from_result": True,
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
            "max_num_frames": None,
            "batch_size": 4,
            "DNEB": False,
            "intermediate_minima_check_step": 0,  # 0 = disabled; >0 = one-shot imin detection at this optimizer step
            "intermediate_minima_min_depth": 0.05,
            "add_images_step": 0,  # 0 = disabled; >0 = one-shot image addition at this optimizer step
            "dimer_refine_ci": False,
            "dimer_refine_steps": 300,
            "refine_band_steps": 0,
            "vasp_command_endpoints": None,
            "vasp_ncore_endpoints": None,
            "vasp_command_intermediates": None,
            "vasp_ncore_intermediates": None,
        },
        "ourDimer": {
            "dataset_type": None,
            "reaction_types": None, # Bulk: vacancy hop_reuse hop_insert kickout_reuse kickout_insert ring initial_guess
                                   # OC: adsorbate_atom adsorbate_atom_neighbors adsorbate diffusion rotation adsorbate_surface surface custom initial_guess
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

        # Migrate renamed config keys (backward compat)
        self._migrate_renamed_keys()

        # Warn about unrecognized keys in sections we control
        for section, defaults in self.DEFAULTS.items():
            if section in self._config:
                unknown = set(self._config[section]) - set(defaults)
                for key in sorted(unknown):
                    print(f"Warning: Unrecognized key '{key}' in [{section}]. "
                          f"Valid keys: {sorted(defaults)}")

    # Renamed keys: (section, old_key, new_key)
    _RENAMED_KEYS = [
        ("ourNEB", "intermediate_minima_check_interval", "intermediate_minima_check_step"),
        ("ourNEB", "add_images_check_interval", "add_images_step"),
    ]

    def _migrate_renamed_keys(self):
        """Silently migrate old config key names to new names."""
        neb = self._config.get("ourNEB", {})

        # Handle renamed keys
        for section, old_key, new_key in self._RENAMED_KEYS:
            sec = self._config.get(section, {})
            if old_key in sec:
                sec[new_key] = sec.pop(old_key)
                print(f"Note: [{section}] '{old_key}' renamed to '{new_key}'.")

        # Handle removed 'intermediate_minima' bool: if True, ensure check_step > 0
        if "intermediate_minima" in neb:
            enabled = neb.pop("intermediate_minima")
            if enabled and neb.get("intermediate_minima_check_step", 0) == 0:
                # User had intermediate_minima=True but no check_step set;
                # use the old default (100) as the step
                neb["intermediate_minima_check_step"] = 100
                print("Note: [ourNEB] 'intermediate_minima=True' converted to "
                      "'intermediate_minima_check_step=100'.")

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
    input_pattern = os.path.join(dir_path, "**", "*.traj")
    all_traj_files = sorted(glob.glob(input_pattern, recursive=True))
    
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


def create_results_directories(config_dict):
    method_name = config_dict["Main"]["method"]
    dirs = [f"{method_name}_status_csvs", f"{method_name}_trajes", f"{method_name}_debug_zips"]
    for d in dirs:
        pathlib.Path(d).mkdir(exist_ok=False)


def read_status_csv_rows(method_name, directory="."):
    """Read all status CSV rows. Returns list of lists of strings (one per row)."""
    csv_dir = os.path.join(directory, f"{method_name}_status_csvs")
    rows = []
    for csv_path in sorted(glob.glob(os.path.join(csv_dir, "status_rank_*.csv"))):
        with open(csv_path) as fh:
            for row in csv.reader(fh):
                if row:
                    rows.append(row)
    return rows


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
    cats = {_RUN_CATEGORY_ALIASES.get(c, c) for c in cats}
    invalid = cats - VALID_RUN_CATEGORIES
    if invalid:
        raise ValueError(
            f"Invalid run_jobs categories: {invalid}. "
            f"Valid: {sorted(VALID_RUN_CATEGORIES)} or 'all'")
    return cats


def _categorize_status(status):
    """Categorize a single status string into a run_jobs category."""
    if status.startswith("converged"):
        return "converged"
    if status.startswith("error"):
        return "errored"
    if status.startswith("not_converged"):
        return "not_converged"
    return "errored"


def _categorize_statuses(statuses, method_name=None):
    """Return the set of categories for a job based on its status lines.

    For NEB (band-level): 'converged' only if ALL sub-bands are converged/converged_CI.
    For other methods: ANY matching status adds its category (original behavior).
    """
    cats_per_line = [_categorize_status(s) for s in statuses]
    if method_name == "NEB":
        # Band-level: converged only if ALL sub-bands converged
        result = set()
        if all(c == "converged" for c in cats_per_line):
            result.add("converged")
        if any(c == "not_converged" for c in cats_per_line):
            result.add("not_converged")
        if any(c == "errored" for c in cats_per_line):
            result.add("errored")
        return result
    return set(cats_per_line)


def get_remaining_trajes(trajes_and_idxs, config_dict):
    categories_to_run = _normalize_run_jobs(config_dict["Main"]["run_jobs"])
    method_name = config_dict["Main"]["method"]
    rows = read_status_csv_rows(method_name)

    job_categories = {}
    for row in rows:
        job_id = int(row[0])
        status = row[-1].strip()
        job_categories.setdefault(job_id, []).append(status)
    job_categories = {jid: _categorize_statuses(statuses, method_name)
                      for jid, statuses in job_categories.items()}

    remaining = []
    for idx, item in enumerate(trajes_and_idxs):
        if idx in job_categories:
            if job_categories[idx] & categories_to_run:
                remaining.append([idx, item])
        else:
            if "remaining" in categories_to_run:
                remaining.append([idx, item])

    if not remaining:
        return [], []
    job_IDs, trajes_and_idxs_out = zip(*remaining)
    return list(job_IDs), list(trajes_and_idxs_out)


def build_redo_info(job_ids, config_dict):
    """Determine which subunits to redo for each job based on CSV status.

    Returns {job_id: set of subunit_ids} where subunit_id is:
      - Dimer: attempt_id (int)
      - NEB: set of ALL sub_band_ids (NEB always re-runs full band)
      - DoubleMinimization: side_id (int, -1 or 1)
      - Minimization: None
    Only jobs with at least one matching status line are included.
    """
    categories_to_run = _normalize_run_jobs(config_dict["Main"]["run_jobs"])
    method_name = config_dict["Main"]["method"]
    subunit_col, _ = _get_subunit_config(method_name)
    rows = read_status_csv_rows(method_name)

    job_ids_set = set(job_ids)

    if method_name == "NEB":
        # NEB always re-runs full band: collect ALL sub-band ids for selected jobs
        job_all_subbands = {}  # {job_id: set of all sub_band_ids}
        job_statuses = {}      # {job_id: [status_strings]}
        for row in rows:
            jid = int(row[0])
            if jid not in job_ids_set:
                continue
            subunit_id = int(row[subunit_col])
            job_all_subbands.setdefault(jid, set()).add(subunit_id)
            job_statuses.setdefault(jid, []).append(row[-1].strip())

        redo_info = {}
        for jid in job_all_subbands:
            cats = _categorize_statuses(job_statuses[jid], method_name)
            if cats & categories_to_run:
                redo_info[jid] = job_all_subbands[jid]  # ALL sub-bands
        return redo_info

    # Other methods: per-subunit selection
    redo_info = {}
    for row in rows:
        jid = int(row[0])
        if jid not in job_ids_set:
            continue
        status = row[-1].strip()
        if _categorize_status(status) in categories_to_run:
            subunit_id = int(row[subunit_col]) if subunit_col is not None else None
            redo_info.setdefault(jid, set()).add(subunit_id)

    return redo_info


def _get_subunit_config(method_name):
    """Return (csv_column_index, info_key) for the sub-unit identifier per method.

    The csv column holds the sub-unit id (attempt_id, sub_band_id, side_id).
    The info_key is the corresponding key in output traj frame .info.
    Returns (None, None) for methods without sub-units (Minimization).
    """
    if method_name == "Dimer":
        return 2, "attempt_id"
    elif method_name == "NEB":
        return 2, "subband_idx"
    elif method_name == "DoubleMinimization":
        return 2, "side"
    return None, None


def archive_and_clean_csvs(config_dict, job_ids, categories_to_clean):
    """Archive old CSVs and remove only entries matching the requested categories.

    Per-line cleaning: for each CSV row belonging to a selected job_id,
    categorize its status. Only remove if the category is in categories_to_clean.
    Returns {job_id: set of sub-unit ids} that were cleaned, for use by
    archive_and_clean_outputs.
    """
    if not job_ids:
        return {}
    method_name = config_dict['Main']['method']
    status_dir = f"{method_name}_status_csvs"
    csv_files = glob.glob(os.path.join(status_dir, "*.csv"))
    if not csv_files:
        return {}

    subunit_col, _ = _get_subunit_config(method_name)
    job_ids_set = set(job_ids)
    csv_data = {}  # {filepath: list of rows}
    has_entries_to_clean = False
    for f in csv_files:
        with open(f) as fh:
            rows = [row for row in csv.reader(fh) if row]
        if not rows:
            continue
        csv_data[f] = rows
        if any(int(row[0]) in job_ids_set for row in rows):
            has_entries_to_clean = True

    if not has_entries_to_clean:
        return {}

    # 1. Archive: zip all current CSVs as previous_{N}.zip
    n = 0
    while os.path.exists(os.path.join(status_dir, f"previous_{n}.zip")):
        n += 1
    archive_path = os.path.join(status_dir, f"previous_{n}.zip")
    with zipfile.ZipFile(archive_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        for f in csv_files:
            zf.write(f, os.path.basename(f))

    # 2. Clean: remove rows matching categories_to_clean.
    # NEB: remove ALL rows for selected jobs (always full-band re-run).
    neb_clean_all = method_name == "NEB"
    cleaned = {}  # {job_id: set of subunit_ids}
    for f, rows in csv_data.items():
        kept = []
        for row in rows:
            jid = int(row[0])
            status = row[-1].strip()
            should_clean = (jid in job_ids_set and
                            (neb_clean_all or _categorize_status(status) in categories_to_clean))
            if should_clean:
                subunit_id = int(row[subunit_col]) if subunit_col is not None else None
                cleaned.setdefault(jid, set()).add(subunit_id)
            else:
                kept.append(row)
        if len(kept) < len(rows):
            if not kept:
                os.remove(f)
            else:
                with open(f, 'w', newline='') as fh:
                    csv.writer(fh).writerows(kept)

    return cleaned


def _get_debug_filename_patterns(method_name):
    """Return compiled regex patterns that capture (job_id, subunit_id) from debug filenames.

    Each pattern should have group(1)=job_id. Group(2), if present, is the subunit_id.
    """
    if method_name == "NEB":
        return [
            re.compile(r'^(?:ERROR_)?neb_(\d+)(?:_sub(\d+))?\.'),
            re.compile(r'^(?:ERROR_)?neb_refine_(\d+)(?:_sub(\d+))?\.'),
            re.compile(r'^(?:ERROR_)?(?:reactant|product)_relaxation_(\d+)(?:_sub(\d+))?\.'),
            re.compile(r'^(?:ERROR_)?diffusion_barrier_(\d+)(?:_sub(\d+))?\.'),
            re.compile(r'^(?:ERROR_)?dimer_ci_(?:control_)?(\d+)(?:_sub(\d+))?_img\d+\.'),
            re.compile(r'^(?:ERROR_)?imin_relax_(\d+)(?:_sub(\d+))?_img\d+\.'),
            re.compile(r'^VASP_(\d+)(?:_sub(\d+))?_'),
        ]
    elif method_name == "Dimer":
        return [re.compile(r'^(?:ERROR_)?dimer_(?:control_|opt_)?(\d+)_(\d+)_')]
    elif method_name == "DoubleMinimization":
        return [re.compile(r'^(?:ERROR_)?optimization_(\d+)_(-?\d+)')]
    elif method_name == "Minimization":
        return [re.compile(r'^(?:ERROR_)?optimization_(\d+)')]
    return []


def _extract_debug_ids(filename, patterns):
    """Extract (job_id, subunit_id) from a debug filename.

    Returns (int, int) or (int, None) or (None, None).
    For Dimer: subunit_id is the attempt_id.
    For NEB: subunit_id is the subband_idx (from _sub{N} suffix), or None for full-band files.
    For DoubleMinimization: subunit_id is the file_idx (0→side=-1, 1→side=1).
    """
    for pat in patterns:
        m = pat.match(filename)
        if m:
            job_id = int(m.group(1))
            subunit_id = int(m.group(2)) if m.lastindex >= 2 and m.group(2) is not None else None
            return job_id, subunit_id
    return None, None


def _should_remove_debug(filename, patterns, cleaned, method_name):
    """Check if a debug file should be removed based on cleaned entries."""
    job_id, subunit_id = _extract_debug_ids(filename, patterns)
    if job_id is None or job_id not in cleaned:
        return False
    if method_name == "Minimization":
        return True  # No subunit, remove all for job
    if method_name == "DoubleMinimization":
        if subunit_id is not None:
            return subunit_id in cleaned[job_id]
        return True  # Can't determine side, remove to be safe
    # Dimer and NEB: subunit_id directly matches
    if subunit_id is not None:
        return subunit_id in cleaned[job_id]
    # No subunit in filename (e.g., full-band NEB file) — remove if job matches
    return True


def _should_remove_frame(img, cleaned, info_key, remove_all_sides=False):
    """Check if an output traj frame matches a cleaned entry."""
    jid = img.info.get('src_index')
    if jid not in cleaned:
        return False
    if info_key is None or remove_all_sides:
        return True  # Remove all frames for this job
    return img.info.get(info_key) in cleaned[jid]


def archive_and_clean_outputs(config_dict, cleaned):
    """Archive output trajectories and debug zips, remove entries matching cleaned.

    cleaned: {job_id: set of subunit_ids} from archive_and_clean_csvs.
    Archive is always a full backup. Cleaning removes only matching entries.
    """
    if not cleaned:
        return

    method_name = config_dict["Main"]["method"]
    _, info_key = _get_subunit_config(method_name)
    # DoubleMinimization: always remove all 3 frames (min1+TS+min2) since they
    # share reaction check metadata and are always re-written together.
    remove_all_sides = method_name == "DoubleMinimization"

    # ---- Output Trajectories ----
    traj_dir = f"{method_name}_trajes"
    traj_files = glob.glob(os.path.join(traj_dir, "*.traj"))

    if traj_files:
        has_stale = False
        for traj_path in traj_files:
            try:
                with Trajectory(traj_path, 'r') as traj:
                    for idx in range(len(traj)):
                        if _should_remove_frame(traj[idx], cleaned, info_key, remove_all_sides):
                            has_stale = True
                            break
            except Exception:
                continue
            if has_stale:
                break

        if has_stale:
            n = 0
            while os.path.exists(os.path.join(traj_dir, f"previous_{n}.zip")):
                n += 1
            with zipfile.ZipFile(os.path.join(traj_dir, f"previous_{n}.zip"), 'w', zipfile.ZIP_DEFLATED) as zf:
                for f in traj_files:
                    zf.write(f, os.path.basename(f))

            for traj_path in traj_files:
                try:
                    with Trajectory(traj_path, 'r') as traj:
                        all_frames = [traj[idx] for idx in range(len(traj))]
                except Exception:
                    continue
                kept = [img for img in all_frames if not _should_remove_frame(img, cleaned, info_key, remove_all_sides)]
                os.remove(traj_path)
                if kept:
                    with Trajectory(traj_path, 'w') as writer:
                        for img in kept:
                            writer.write(img)

    # ---- Debug Zips ----
    zip_dir = f"{method_name}_debug_zips"
    zip_files = [f for f in glob.glob(os.path.join(zip_dir, "*.zip"))
                 if not os.path.basename(f).startswith("previous_")]

    if zip_files:
        patterns = _get_debug_filename_patterns(method_name)
        has_stale = False
        for zip_path in zip_files:
            try:
                with zipfile.ZipFile(zip_path, 'r') as zf:
                    for name in zf.namelist():
                        if _should_remove_debug(name, patterns, cleaned, method_name):
                            has_stale = True
                            break
            except zipfile.BadZipFile:
                continue
            if has_stale:
                break

        if has_stale:
            n = 0
            while os.path.exists(os.path.join(zip_dir, f"previous_{n}.zip")):
                n += 1
            with zipfile.ZipFile(os.path.join(zip_dir, f"previous_{n}.zip"), 'w', zipfile.ZIP_DEFLATED) as zf:
                for f in zip_files:
                    zf.write(f, os.path.basename(f))

            for zip_path in zip_files:
                try:
                    with zipfile.ZipFile(zip_path, 'r') as zf_old:
                        all_entries = zf_old.infolist()
                        kept = [info for info in all_entries
                                if not _should_remove_debug(info.filename, patterns, cleaned, method_name)]

                        if not kept:
                            os.remove(zip_path)
                        elif len(kept) < len(all_entries):
                            tmp_path = zip_path + ".tmp"
                            with zipfile.ZipFile(tmp_path, 'w', zipfile.ZIP_DEFLATED) as zf_new:
                                for info in kept:
                                    zf_new.writestr(info, zf_old.read(info.filename))
                            os.replace(tmp_path, zip_path)
                except zipfile.BadZipFile:
                    continue


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
