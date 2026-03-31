import numpy as np
import json, os, glob, shutil, tempfile, zipfile
from ase.neighborlist import neighbor_list, natural_cutoffs
from ase.io import Trajectory


#==============================================================================
### FLUX LOG BACKUP

def backup_flux_logs(worker_id):
    """Append current flux log files into backup files before worker restart.

    Flux overwrites flux_{id}.out/.err on each new job submission, so we
    append their contents to persistent backup files before sys.exit(1).
    """
    for ext in (".out", ".err"):
        src = f"flux_{worker_id}{ext}"
        dst = f"flux_{worker_id}{ext}.bak"
        if os.path.exists(src):
            with open(src, 'r') as f_in, open(dst, 'a') as f_out:
                f_out.write(f_in.read())


#==============================================================================
### ATOMS LOADING

def load_and_sanitize(traj, i, j):
    """Load images from trajectory and stash original .info into orig_info.

    This prevents per-atom array data (e.g. forces, stress) in .info from
    causing size mismatches when atoms are later added or removed (e.g. vacancy
    mechanism in Dimer). Applied uniformly across all methods for consistency.
    """
    if j != i + 1:
        images = list(traj[i:j])
        for img in images:
            img.info = {"orig_info": dict(img.info)}
    else:
        images = traj[i]
        images.info = {"orig_info": dict(images.info)}
    return images


#==============================================================================
### FILE IO

def save_ordered_traj_names(trajes_and_idxs):
    with open('traj_files_ordered.json', 'w') as f:
        json.dump(trajes_and_idxs, f)


def read_ordered_traj_names():
    with open('traj_files_ordered.json', 'r') as f:
        trajes_and_idxs = json.load(f)
    return trajes_and_idxs


def clean_up_files(config_dict):
    """Remove leftover temp files from a previous interrupted run.

    Each method writes its own set of temp files into the working directory.
    On resume, these leftovers must be cleaned up so they don't collide with
    new runs.  For VASP NEB, per-image directories (VASP_{job_id}_{image_idx}/)
    are also removed.
    """
    import glob as _glob
    import shutil

    method_name = config_dict["Main"]["method"]

    patterns = {
        "NEB": [
            "neb_*.log", "neb_*.traj",
            "reactant_relaxation_*.log", "reactant_relaxation_*.traj",
            "product_relaxation_*.log", "product_relaxation_*.traj",
            "diffusion_barrier_*.png",
        ],
        "Dimer": [
            "dimer_control_*.log", "dimer_opt_*.log", "dimer_*.traj",
        ],
        "Minimization": [
            "optimization_*.log", "optimization_*.traj",
        ],
        "DoubleMinimization": [
            "optimization_*.log", "optimization_*.traj",
        ],
    }

    # VASP NEB creates per-image directories named VASP_{job_id}_{image_idx}/
    if method_name == "NEB" and config_dict["Main"]["Calculator"] in ("Vasp", "VaspInteractive"):
        patterns["NEB"].append("VASP_*_*/")

    # Flux log files and their backups (common to all methods)
    patterns.setdefault(method_name, [])
    patterns[method_name].extend(["flux_*.out", "flux_*.err", "flux_*.out.bak", "flux_*.err.bak"])

    for pat in patterns.get(method_name, []):
        for f in _glob.glob(pat):
            if os.path.isdir(f):
                shutil.rmtree(f)
            else:
                os.remove(f)


#==============================================================================
### PREVIOUS RESULT EXTRACTION (for continue-from-result on resume)

def _build_debug_zip_index(method_name):
    """Build a map: internal_filename -> zip_path for all debug zips."""
    index = {}
    zip_dir = f"{method_name}_debug_zips"
    for zip_path in glob.glob(os.path.join(zip_dir, "*.zip")):
        if os.path.basename(zip_path).startswith("previous_"):
            continue
        try:
            with zipfile.ZipFile(zip_path, 'r') as zf:
                for name in zf.namelist():
                    index[name] = zip_path
        except zipfile.BadZipFile:
            continue
    return index


def _build_output_traj_index(method_name):
    """Scan output trajectories and build a map: src_index -> list of Atoms.

    Stores the deserialized Atoms objects directly so that extraction
    functions can return them without re-reading from disk.

    Works for both Dimer (collected_ts_rank_*.traj) and Minimization
    (collected_opt_rank_*.traj) output files.
    """
    index = {}
    traj_dir = f"{method_name}_trajes"
    for traj_path in sorted(glob.glob(os.path.join(traj_dir, "*.traj"))):
        try:
            with Trajectory(traj_path, 'r') as traj:
                for frame_idx in range(len(traj)):
                    img = traj[frame_idx]
                    src_idx = img.info.get('src_index')
                    if src_idx is not None:
                        index.setdefault(src_idx, []).append(img)
        except Exception:
            continue
    return index


def _extract_neb_band(job_id, num_frames, debug_zip_index):
    """Extract the final NEB band (all images) from debug traj files.

    Looks for neb_{job_id}.traj first as a loose file, then in debug zips.
    Detects the actual band size from the last frame's 'nimages' metadata
    (handles variable-size bands from image addition). Falls back to num_frames.
    Returns a list of Atoms objects, or None if not found.
    """
    traj_name = f"neb_{job_id}.traj"

    def _read_final_band(traj):
        n = len(traj)
        actual_frames = traj[-1].info.get('nimages', num_frames)
        if n >= actual_frames:
            return [traj[idx] for idx in range(n - actual_frames, n)]
        return None

    # Try loose file first
    if os.path.exists(traj_name):
        with Trajectory(traj_name, 'r') as traj:
            return _read_final_band(traj)

    # Try debug zips
    zip_path = debug_zip_index.get(traj_name)
    if zip_path is None:
        return None

    with tempfile.TemporaryDirectory() as tmpdir:
        with zipfile.ZipFile(zip_path, 'r') as zf:
            zf.extract(traj_name, tmpdir)
        extracted = os.path.join(tmpdir, traj_name)
        with Trajectory(extracted, 'r') as traj:
            return _read_final_band(traj)


def _extract_dimer_attempts(job_id, output_traj_index):
    """Extract all successful attempt results for a Dimer job_id.

    Returns a list of Atoms objects (one per attempt that produced output),
    or None if no results found. Each Atoms retains its original .info
    metadata (eigenmode, reaction_type, selected_index, etc.).
    """
    entries = output_traj_index.get(job_id, [])
    return entries if entries else None


def _extract_minimization_structure(job_id, output_traj_index):
    """Extract the relaxed structure for a job_id from output trajectories."""
    entries = output_traj_index.get(job_id, [])
    return entries[0] if entries else None


def _sanitize_with_continuation(atoms):
    """Wrap .info with orig_info (like load_and_sanitize) and mark as continuation."""
    info = dict(atoms.info)
    info["_continuation"] = True
    atoms.info = {"orig_info": info}
    return atoms


def extract_previous_results(job_ids, config_dict):
    """Extract previous results for the given job_ids.

    Returns a dict: job_id -> images.
      - NEB: list of Atoms (the band)
      - Dimer: list of Atoms (one per successful attempt)
      - Minimization: single Atoms
    Each extracted result has _continuation=True in orig_info.
    Jobs with no extractable result are omitted (falls back to original input).
    """
    method_name = config_dict["Main"]["method"]
    results = {}

    if method_name == "NEB":
        num_frames = config_dict["ourNEB"]["num_frames"]
        debug_zip_index = _build_debug_zip_index(method_name)
        for job_id in job_ids:
            band = _extract_neb_band(job_id, num_frames, debug_zip_index)
            if band is not None:
                for img in band:
                    _sanitize_with_continuation(img)
                results[job_id] = band

    elif method_name == "Dimer":
        output_traj_index = _build_output_traj_index(method_name)
        # Filter attempts to match run_jobs categories
        run_jobs = config_dict["Main"]["run_jobs"]
        cats = set(run_jobs) if isinstance(run_jobs, list) else (
            {"converged", "not_converged", "error", "not_started"} if run_jobs == "all" else {run_jobs})
        keep_conv = "converged" in cats
        keep_nconv = "not_converged" in cats
        for job_id in job_ids:
            attempts = _extract_dimer_attempts(job_id, output_traj_index)
            if attempts is not None:
                if not (keep_conv and keep_nconv):
                    attempts = [a for a in attempts
                                if (a.info.get('converged') == 1 and keep_conv) or
                                   (a.info.get('converged') != 1 and keep_nconv)]
                if attempts:
                    for atoms in attempts:
                        _sanitize_with_continuation(atoms)
                    results[job_id] = attempts

    elif method_name == "Minimization":
        output_traj_index = _build_output_traj_index(method_name)
        for job_id in job_ids:
            atoms = _extract_minimization_structure(job_id, output_traj_index)
            if atoms is not None:
                _sanitize_with_continuation(atoms)
                results[job_id] = atoms

    return results


#==============================================================================
### BOND-BREAKING/FORMING DETECTION

def get_bond_set(atoms, cutoffs, tag_filter=None):
    """
    Returns a python set of bonds tuple(atom_index_A, atom_index_B).
    
    Args:
        atoms: The ASE atoms object
        cutoffs: Dictionary or list of cutoff radii
        tag_filter: (Optional) Only include bonds where BOTH atoms have this tag.
    """
    # 'i' and 'j' are indices of bonded atoms
    i_list, j_list = neighbor_list('ij', atoms, cutoffs)
    
    bonds = set()
    tags = atoms.get_tags()
    
    for k in range(len(i_list)):
        a, b = i_list[k], j_list[k]
        
        # We only want each bond once (0-1 is same as 1-0)
        # So we sort them: tuple((min, max))
        bond = tuple(sorted((a, b)))
        
        # If a filter is applied (e.g., tag==2), check tags
        if tag_filter is not None:
            if tags[a] == tag_filter and tags[b] == tag_filter:
                bonds.add(bond)
        else:
            bonds.add(bond)
            
    return bonds


def check_reaction(atoms_initial, atoms_final, neighbor_fudge=1.25):
    """
    Compares connectivity of two structures.
    """
    # 1. Get bonds for both
    assert np.array_equal(atoms_initial.numbers, atoms_final.numbers), \
            "Error: Atomic numbers do not match between initial and final states."
    cutoffs = natural_cutoffs(atoms_initial, mult=neighbor_fudge)
    bonds_ini = get_bond_set(atoms_initial, cutoffs)
    bonds_fin = get_bond_set(atoms_final, cutoffs)
    
    # 2. Compare sets
    # Bonds present in Initial but NOT in Final = BROKEN
    broken = bonds_ini - bonds_fin
    
    # Bonds present in Final but NOT in Initial = FORMED
    formed = bonds_fin - bonds_ini
    
    reaction_occurred = len(broken) > 0 or len(formed) > 0
    
    return {
        "occurred": reaction_occurred,
        "broken_bonds": broken,
        "formed_bonds": formed,
        "n_broken": len(broken),
        "n_formed": len(formed)
    }

def check_adsorbate_reaction(atoms_initial, atoms_final, neighbor_fudge=1.25, target_tag=2):
    """
    Checks for reactions ONLY within atoms having specific tag (e.g. tag=2).
    """
    # 1. Get filtered bonds
    assert np.array_equal(atoms_initial.numbers, atoms_final.numbers), \
            "Error: Atomic numbers do not match between initial and final states."
    cutoffs = natural_cutoffs(atoms_initial, mult=neighbor_fudge)
    bonds_ini = get_bond_set(atoms_initial, cutoffs, tag_filter=target_tag)
    bonds_fin = get_bond_set(atoms_final, cutoffs, tag_filter=target_tag)
    
    # 2. Calculate differences
    broken = bonds_ini - bonds_fin
    formed = bonds_fin - bonds_ini
    
    return {
        "occurred": len(broken) > 0 or len(formed) > 0,
        "broken_bonds": broken,
        "formed_bonds": formed,
        "n_broken": len(broken),
        "n_formed": len(formed)
    }

#==============================================================================

