import numpy as np
import json, os, glob, shutil, tempfile, zipfile, fnmatch
from ase.neighborlist import neighbor_list, natural_cutoffs
from ase.io import Trajectory
from saddlemill.config import VALID_RUN_CATEGORIES, _RUN_CATEGORY_ALIASES, _get_subunit_config


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


def passes_input_filter(images, config_dict):
    """Return True if a sanitized input's status matches ``input_statuses``.

    Patterns support ``fnmatch`` wildcards (e.g. ``converged*`` matches
    ``converged``, ``converged_CI``, ``converged_after_extension``, etc.).
    The special value ``"all"`` (the default) bypasses the filter entirely.
    """
    raw = config_dict["Main"]["input_statuses"]
    if raw in ("all", None):
        return True

    main_atoms = images[0] if isinstance(images, list) else images
    orig = main_atoms.info.get('orig_info', {})
    status = orig.get('status')

    patterns = [raw] if isinstance(raw, str) else list(raw)
    return any(fnmatch.fnmatchcase(status or '', p) for p in patterns)


def get_task_name(config_dict):
    """Return [FAIRChemCalculator] task_name if FAIRChem is the calculator, else None."""
    if config_dict["Main"]["Calculator"] == "FAIRChemCalculator":
        return config_dict["FAIRChemCalculator"].get("task_name")
    return None


#==============================================================================
### VASP HELPERS

def vasp_incar_kwargs(config_dict, atoms=None):
    """Return the INCAR/k-point/setup kwargs for a VASP calculator.

    Starts from an optional ``[ourVasp] input_generator`` (built-in name, dotted
    ``module:func``, or ``file.py:func``) evaluated on ``atoms``, then layers the
    explicit ``[Vasp]`` section keys on top so the user's ``[Vasp]`` always wins.
    With no generator (or no ``atoms``), this is just the ``[Vasp]`` section.
    ``[Vasp]`` is a pure pass-through to ASE's Vasp calculator; SaddleMill's own
    knobs live in ``[ourVasp]`` and are never forwarded to the calculator.
    """
    vasp_section = dict(config_dict.get("Vasp", {}))
    gen_spec = config_dict.get("ourVasp", {}).get("input_generator")
    if gen_spec and atoms is not None:
        from saddlemill.vasp_io import load_input_generator
        gen_kwargs = load_input_generator(gen_spec)(atoms)
        return {**gen_kwargs, **vasp_section}  # [Vasp] keys override generator
    return vasp_section


def _with_extra_io(calc_cls, writers, parsers):
    """Subclass *calc_cls* to run extra-input writers and extra-output parsers.

    Writers ``(calc, atoms, directory) -> None`` run after ASE writes its inputs
    (directory exists, ``calc.sort`` set) and before VASP runs — via ``write_input``.
    Parsers ``(calc, atoms, directory) -> dict`` run after VASP finishes (directory
    populated, ``calc.resort`` set) — via ``read_results`` — and their merged dict is
    stashed on ``calc.sm_extra_outputs`` for the method to stamp onto output frames.
    """
    class _CalcWithExtraIO(calc_cls):
        def write_input(self, atoms, *args, **kwargs):
            super().write_input(atoms, *args, **kwargs)
            directory = kwargs.get("directory", getattr(self, "directory", "."))
            for writer in writers:
                writer(self, atoms, directory)

        def read_results(self):
            super().read_results()
            info = {}
            directory = getattr(self, "directory", ".")
            for parser in parsers:
                info.update(parser(self, self.atoms, directory) or {})
            self.sm_extra_outputs = info

    _CalcWithExtraIO.__name__ = f"{calc_cls.__name__}WithExtraIO"
    return _CalcWithExtraIO


def resolve_vasp_calc_class(config_dict, calc, extra_writers=None):
    """Return *calc*, wrapped for ``[ourVasp] extra_input_files`` / ``extra_outputs`` (if set).

    No-op for FAIRChem or when neither key is set (and no ``extra_writers``).
    Shared by ``resolve_vasp_calc`` and ``nebopt._build_neb_vasp_calc`` so the
    hooks are identical across all methods. Each value is one spec or a
    space-separated list (built-in name, ``module:func``, or ``file.py:func``).
    Output parsers leave their merged dict on ``calc.sm_extra_outputs``; the
    method decides whether to stamp it onto frames.

    ``extra_writers`` is an optional list of caller-supplied writers appended
    AFTER the config writers, so they overwrite the config-written inputs (used
    by SinglePoint resume to seed POSCAR/MODECAR from banked mid-run state).
    """
    if config_dict["Main"]["Calculator"] not in ("Vasp", "VaspInteractive"):
        return calc
    our_vasp = config_dict.get("ourVasp", {})
    in_spec = our_vasp.get("extra_input_files")
    out_spec = our_vasp.get("extra_outputs")
    if not in_spec and not out_spec and not extra_writers:
        return calc
    from saddlemill.vasp_io import (load_extra_input_writer,
                                                  load_extra_output_parser)
    _aslist = lambda s: [s] if isinstance(s, str) else list(s)
    writers = [load_extra_input_writer(s) for s in _aslist(in_spec)] if in_spec else []
    if extra_writers:
        writers = writers + list(extra_writers)   # run last -> overwrite config inputs
    parsers = [load_extra_output_parser(s) for s in _aslist(out_spec)] if out_spec else []
    return _with_extra_io(calc, writers, parsers)


def resolve_vasp_calc(config_dict, calc, i, subunit_id, section, atoms=None,
                      extra_writers=None):
    """Return an instantiated calculator for this (job, subunit).

    For FAIRChem, returns the shared instance unchanged. For VASP/VaspInteractive,
    builds a fresh calculator pointing at ``VASP_{i}[_{subunit_id}]/`` with the
    section's ``vasp_command`` / ``vasp_ncore`` and the INCAR kwargs from
    ``vasp_incar_kwargs`` (``[Vasp]`` plus an optional per-structure
    ``[ourVasp] input_generator``). The class is first wrapped by
    ``resolve_vasp_calc_class`` so ``[ourVasp] extra_input_files`` (e.g. a VTST
    MODECAR) are written too.
    ``subunit_id=None`` produces ``VASP_{i}/`` (Minimization, SinglePoint). Pass
    ``atoms`` to enable ``input_generator`` and the extra-file writers.
    ``extra_writers`` is forwarded to ``resolve_vasp_calc_class`` (SinglePoint
    resume seeds POSCAR/MODECAR from banked mid-run state this way).
    """
    if config_dict["Main"]["Calculator"] not in ("Vasp", "VaspInteractive"):
        return calc
    suffix = f"_{subunit_id}" if subunit_id is not None else ""
    kwargs = {"directory": f"VASP_{i}{suffix}",
              "command": config_dict[section]["vasp_command"],
              **vasp_incar_kwargs(config_dict, atoms)}
    ncore = config_dict[section].get("vasp_ncore")
    if ncore is not None:
        kwargs["ncore"] = int(ncore)
    return resolve_vasp_calc_class(config_dict, calc, extra_writers=extra_writers)(**kwargs)


def remove_vasp_heavies(dir_path):
    """Delete WAVECAR / CHG / CHGCAR from *dir_path* if they exist."""
    for name in ("WAVECAR", "CHG", "CHGCAR"):
        p = os.path.join(dir_path, name)
        if os.path.exists(p):
            os.remove(p)


def archive_and_clear_temp_files(temp_files, zip_name, prefix="", enabled=True):
    """Zip existing temp files/directories into *zip_name* and remove them.

    Mirrors the per-method temp-file cleanup that previously lived inline in
    each method. Walks directories (e.g. VASP working dirs) so every file inside
    is archived under its relative path. Set ``enabled=False`` to skip zipping
    and just remove the entries.
    """
    existing = [f for f in temp_files if os.path.exists(f)]
    if not existing:
        return
    if enabled:
        # The debug-zip dir is created only on a FRESH start (setup_directories
        # uses mkdir(exist_ok=False), which resume skips), so a hand-built or
        # migrated shard dir has none and every finished side would die here --
        # turning completed work into an 'errored' row. Ensure it instead.
        os.makedirs(os.path.dirname(zip_name) or ".", exist_ok=True)
        with zipfile.ZipFile(zip_name, 'a', zipfile.ZIP_DEFLATED) as zf:
            for f_name in existing:
                if os.path.isdir(f_name):
                    for root, _dirs, files in os.walk(f_name):
                        for file in files:
                            filepath = os.path.join(root, file)
                            zf.write(filepath, arcname=f"{prefix}{filepath}")
                else:
                    zf.write(f_name, arcname=f"{prefix}{f_name}")
    for f_name in existing:
        if os.path.isdir(f_name):
            shutil.rmtree(f_name)
        else:
            os.remove(f_name)


def finalize_if_vasp_interactive(config_dict, calc_instance):
    """Call ``.finalize()`` on a VaspInteractive instance; no-op otherwise.

    The matching guard is on the active calculator class, not on the instance
    type — that keeps the call site readable next to other VASP-only branches.
    """
    if config_dict["Main"]["Calculator"] == "VaspInteractive":
        try:
            calc_instance.finalize()
        except Exception:
            pass


def vasp_final_scf_converged(directory):
    """Return True iff the LAST electronic (SCF) loop in OUTCAR reached EDIFF.

    VASP 6 labels each SCF exit: ``aborting loop because EDIFF is reached`` when an
    ionic step's electronic loop converges, and ``aborting loop because EDIFF was
    not reached (unconverged)`` (a NELM miss) when it does not. We keep the verdict
    of the LAST such marker, so an intermediate step that blew NELM but later
    recovered does not fail the job — only the final structure's SCF must be sound.
    Returns True when OUTCAR is missing/unreadable or has no marker (can't tell ->
    don't block; a genuinely broken run errors out elsewhere on parsing).
    """
    outcar = os.path.join(directory, "OUTCAR")
    if not os.path.isfile(outcar):
        return True
    result = True
    try:
        with open(outcar) as f:
            for line in f:
                if "aborting loop" in line:
                    result = "because EDIFF is reached" in line
    except OSError:
        return True
    return result


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
            "imin_relax_*.log", "imin_relax_*.traj",
            "dimer_ci_*.log", "dimer_ci_*.traj",
            "neb_refine_*.log", "neb_refine_*.traj",
        ],
        "Dimer": [
            "dimer_control_*.log", "dimer_opt_*.log", "dimer_*.traj",
        ],
        "Sella": [
            "sella_control_*.log", "sella_opt_*.log", "sella_*.traj",
        ],
        "Minimization": [
            "optimization_*.log", "optimization_*.traj",
        ],
        "DoubleMinimization": [
            "optimization_*.log", "optimization_*.traj",
            "dimer_refine_*.log",
        ],
        "SinglePoint": [],  # SP writes no temp files in cwd.
    }

    # Each method creates per-job-unit directories named VASP_{job_id}[_{subunit}]/
    # (NEB → _image_idx, Dimer → _attempt, DM → _-1/_0/_1, Min/SP → no suffix).
    # The single VASP_* glob matches all of these (file-or-directory).
    if config_dict["Main"]["Calculator"] in ("Vasp", "VaspInteractive"):
        patterns.setdefault(method_name, []).append("VASP_*")

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
### SINGLEPOINT VASP RESUME (bank mid-run VTST-dimer state across a wall-kill)
#
# A wall-killed SinglePoint+VASP job (e.g. a VTST dimer: IBRION=3 IOPT=3
# ICHAIN=2, where VASP -- not an ASE optimizer -- owns the ionic loop) never
# reaches its cleanup, so its VASP_{id}/ workdir is left in place holding the
# mid-run dimer state: CENTCAR (dimer-center geometry) and NEWMODECAR (current
# mode). The output-trajectory resume path (extract_previous_results) cannot
# help here -- a wall-killed job wrote no output frame -- so we bank those small
# restart files BEFORE clean_up_files wipes VASP_*, then seed the resumed run's
# POSCAR/MODECAR from them (see make_sp_resume_seed_writer and
# geomopt.singlepoint). The VTST restart convention is POSCAR <- CENTCAR
# (fallback CONTCAR) and MODECAR <- NEWMODECAR (fallback the run's own MODECAR).
# On ANY doubt (missing/empty/unparseable file, or a geometry/mode atom-count
# mismatch) the job is skipped and falls back to a clean from-scratch run --
# e.g. a first-SCF-death dir with a 0-byte CONTCAR and no CENTCAR/NEWMODECAR.

_SP_RESUME_BANK_DIR = "SinglePoint_resume_states"


def _poscar_natoms(path):
    """Atom count from a POSCAR/CONTCAR/CENTCAR, or None on any doubt.

    Locates the integer counts line (VASP5 line 7 after the symbols line, or
    VASP4 line 6) but returns how many coordinate rows the file ACTUALLY
    delivers, not how many the header claims. A kill landing mid-write leaves a
    complete header above a short body; returning the short count makes the
    caller's geometry/mode agreement check reject it. For an intact file the two
    are identical, so nothing else changes. Returns None for a
    missing/empty/unparseable file so callers treat it as 'no usable state' and
    fall back to scratch.
    """
    if not path or not os.path.isfile(path) or os.path.getsize(path) == 0:
        return None
    try:
        with open(path) as f:
            lines = f.read().splitlines()
    except OSError:
        return None
    for idx in (6, 5):  # VASP5 counts line, then VASP4 fallback
        if idx < len(lines):
            parts = lines[idx].split()
            if parts and all(p.lstrip("+").isdigit() for p in parts):
                n = sum(int(p) for p in parts)
                start = next((i + 1 for i in range(idx + 1, min(len(lines), idx + 4))
                              if lines[i].strip()[:1].lower() in ("d", "c", "k")), None)
                if start is None:
                    return None
                rows = 0
                for line in lines[start:start + n]:
                    if len(line.split()) < 3:
                        break  # row truncated mid-write: stop counting here
                    rows += 1
                return rows or None
    return None


def _modecar_natoms(path):
    """Row count of a MODECAR/NEWMODECAR (one 3-float mode vector per atom), or None.

    Returns None for a missing/empty file or any row that is not three floats.
    """
    if not path or not os.path.isfile(path) or os.path.getsize(path) == 0:
        return None
    try:
        rows = 0
        with open(path) as f:
            for line in f:
                parts = line.split()
                if not parts:
                    continue
                if len(parts) != 3:
                    return None
                [float(x) for x in parts]  # parse-check; ValueError -> not usable
                rows += 1
    except (OSError, ValueError):
        return None
    return rows or None


def _count_oszicar_ionic_steps(directory):
    """Best-effort count of completed ionic steps (OSZICAR ' F= ' lines), or None."""
    oszicar = os.path.join(directory, "OSZICAR")
    if not os.path.isfile(oszicar):
        return None
    try:
        with open(oszicar) as f:
            return sum(1 for line in f if " F= " in line)
    except OSError:
        return None


DOUBLEMIN_PARTIALS_DIR = "DoubleMinimization_partials"


def bank_doublemin_partials(job_ids, config_dict):
    """Bank live DoubleMinimization side trajectories before cleanup deletes them.

    ``clean_up_files`` removes ``optimization_*.traj`` from the working
    directory on every resume, so a wall-killed side's progress is gone before
    any worker could read it. This copies each re-run side's frames into
    ``DoubleMinimization_partials/`` first, APPENDING so the bank stays
    cumulative across repeated wall-kills (the step debit in geomopt reads its
    length, which must count every banked frame, not just the last segment).

    Must be called on resume BEFORE clean_up_files, same ordering constraint as
    bank_singlepoint_vasp_restarts. Returns the number of sides banked.
    Unreadable or empty trajectories are skipped, leaving that side to start
    from its displaced geometry.
    """
    import glob as _glob
    from ase.io import Trajectory as _Traj
    from ase.io import read as _read

    wanted = {str(j) for j in job_ids}
    os.makedirs(DOUBLEMIN_PARTIALS_DIR, exist_ok=True)
    banked = 0
    for src_path in sorted(_glob.glob("optimization_*_*.traj")):
        parts = os.path.basename(src_path)[len("optimization_"):-len(".traj")].split("_")
        if len(parts) != 2 or parts[0] not in wanted:
            continue
        try:
            frames = _read(src_path, ":")
        except Exception:
            continue
        if not frames:
            continue
        dst = os.path.join(DOUBLEMIN_PARTIALS_DIR, os.path.basename(src_path))
        try:
            with _Traj(dst, "a") as bank:
                for atoms in frames:
                    bank.write(atoms)
        except Exception:
            continue
        banked += 1
    return banked


def bank_singlepoint_vasp_restarts(job_ids, config_dict):
    """Bank mid-run VTST-dimer restart files from leftover SinglePoint VASP dirs.

    For each ``job_id`` whose ``VASP_{job_id}/`` workdir survived a wall-kill,
    copy its geometry (CENTCAR, else CONTCAR) as POSCAR and its mode (NEWMODECAR,
    else MODECAR) as MODECAR into ``SinglePoint_resume_states/{job_id}/``. Must be
    called on resume BEFORE clean_up_files removes the VASP_* dirs.

    A job is banked only when both a geometry and a mode file are non-empty,
    parseable, and agree on the atom count; anything else is skipped (that job
    falls back to a clean from-scratch run). Returns
    ``{job_id: {resume_dir, banked_steps, geom_src, mode_src, natoms}}`` for the
    banked jobs. The per-job atom count is re-checked against the live input
    frame in geomopt.singlepoint before it is actually used.
    """
    results = {}
    for job_id in job_ids:
        workdir = f"VASP_{job_id}"
        if not os.path.isdir(workdir):
            continue

        geom_src, geom_n = None, None
        for name in ("CENTCAR", "CONTCAR"):
            n = _poscar_natoms(os.path.join(workdir, name))
            if n:
                geom_src, geom_n = name, n
                break

        mode_src, mode_n = None, None
        for name in ("NEWMODECAR", "MODECAR"):
            n = _modecar_natoms(os.path.join(workdir, name))
            if n:
                mode_src, mode_n = name, n
                break

        if geom_src is None or mode_src is None or geom_n != mode_n:
            print(f"  SP resume: job {job_id} has no usable mid-run state "
                  f"(geom={geom_src}, mode={mode_src}); will run from scratch.",
                  flush=True)
            continue

        bank_dir = os.path.join(_SP_RESUME_BANK_DIR, str(job_id))
        if os.path.isdir(bank_dir):
            shutil.rmtree(bank_dir)
        os.makedirs(bank_dir)
        shutil.copyfile(os.path.join(workdir, geom_src),
                        os.path.join(bank_dir, "POSCAR"))
        shutil.copyfile(os.path.join(workdir, mode_src),
                        os.path.join(bank_dir, "MODECAR"))

        banked_steps = _count_oszicar_ionic_steps(workdir)
        results[job_id] = {"resume_dir": os.path.abspath(bank_dir),
                           "banked_steps": banked_steps,
                           "geom_src": geom_src, "mode_src": mode_src,
                           "natoms": geom_n}
        print(f"  SP resume: banked job {job_id} (geom={geom_src}, "
              f"mode={mode_src}, natoms={geom_n}, ionic_steps={banked_steps}).",
              flush=True)
    return results


def make_sp_resume_seed_writer(resume_dir):
    """Return an extra-input writer that seeds POSCAR/MODECAR from a banked state.

    Runs after ASE writes its inputs (and after any modecar writer), overwriting
    the freshly written POSCAR with the banked dimer-center geometry and MODECAR
    with the banked mode -- so the VTST dimer resumes from the wall-killed mid-run
    state. The banked POSCAR is in POSCAR (symbol) order, identical to what ASE
    just wrote for the same atoms, so no reordering is needed.
    """
    def _seed(calc, atoms, directory):
        for name in ("POSCAR", "MODECAR"):
            src = os.path.join(resume_dir, name)
            if os.path.isfile(src) and os.path.getsize(src) > 0:
                shutil.copyfile(src, os.path.join(directory, name))
    return _seed


#==============================================================================
### PREVIOUS RESULT EXTRACTION (for continue-from-result on resume)

def _build_output_traj_index(method_name):
    """Scan output trajectories and build a map: src_index -> list of Atoms.

    Stores the deserialized Atoms objects directly so that extraction
    functions can return them without re-reading from disk.
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


def _sanitize_with_continuation(atoms):
    """Wrap .info with orig_info (like load_and_sanitize) for extracted results."""
    atoms.info = {"orig_info": dict(atoms.info)}
    return atoms


def extract_previous_results(job_ids, config_dict, redo_info):
    """Extract previous results from output trajs for continuation.

    All methods extract from {method}_trajes/ uniformly.

    Returns {job_id: continuation_data} where continuation_data is:
      - Dimer: {attempt_id: Atoms} for attempts that have output
      - NEB: {subband_idx: [Atoms sorted by image_idx]}
      - DoubleMinimization: {side: Atoms} for all sides (-1, 0, 1)
      - Minimization: Atoms

    All extracted Atoms are wrapped with _sanitize_with_continuation.
    Jobs with no extractable result are omitted (falls back to original input).
    """
    method_name = config_dict["Main"]["method"]
    _, info_key = _get_subunit_config(method_name)
    output_traj_index = _build_output_traj_index(method_name)
    results = {}

    for job_id in job_ids:
        if job_id not in redo_info:
            continue
        frames = output_traj_index.get(job_id, [])
        if not frames:
            continue

        if method_name == "Minimization":
            _sanitize_with_continuation(frames[0])
            results[job_id] = frames[0]
        elif method_name == "SinglePoint":
            # No output-trajectory continuation for SP: a finished frame is the
            # final result, and a wall-killed run wrote none. SP's only resume is
            # from the mid-run VASP workdir, handled separately by
            # bank_singlepoint_vasp_restarts (called in __main__ before cleanup).
            continue
        else:
            # Group frames by subunit_id
            grouped = {}
            for f in frames:
                subunit_id = f.info.get(info_key)
                grouped.setdefault(subunit_id, []).append(f)

            if method_name == "NEB":
                # Sort each subband's images by image_idx
                for sid in grouped:
                    grouped[sid].sort(key=lambda a: a.info.get('image_idx', 0))

            if method_name in ("Dimer", "Sella", "DoubleMinimization"):
                # Flatten: each subunit maps to a single Atoms
                grouped = {sid: atoms_list[0] for sid, atoms_list in grouped.items()
                           if atoms_list}

            # Sanitize all frames
            for sid, data in grouped.items():
                if isinstance(data, list):
                    for atoms in data:
                        _sanitize_with_continuation(atoms)
                else:
                    _sanitize_with_continuation(data)

            results[job_id] = grouped

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


#==============================================================================
### SADDLE INDEX VERIFICATION

def _analytic_hessian(atoms, chunk=4):
    """Exact Hessian from a conservative MLIP, or None if unavailable.

    UMA-S-1.2 has ``direct_forces=False`` - its forces are a true autograd
    gradient of the energy - so ``-grad(forces)`` is the genuine energy Hessian.
    fairchem exposes it, but its ``compute_hessian_vmap`` vmaps over the whole
    ``eye(3N)`` and so materialises 3N backward graphs at once: that OOMs above
    roughly 30 atoms on a 40 GB card. Chunking the vmap over groups of *chunk*
    rows is identical arithmetic with peak memory set by the chunk rather than
    the system size, and covers 100+ atoms in ~10-30 GB.

    Returns the (3N, 3N) Hessian, or None when the calculator cannot produce one
    (direct-force model, hessian output not enabled, or out of memory) so the
    caller can fall back to finite differences.
    """
    try:
        import torch
        import fairchem.core.models.uma.outputs as _O
    except Exception:
        return None

    if getattr(_O, "_sm_chunked", None) != chunk:
        def _chunked(forces_flat, pos, create_graph, _c=chunk):
            n = forces_flat.shape[0]
            eye = torch.eye(n, device=forces_flat.device, dtype=forces_flat.dtype)
            out = []
            for i in range(0, n, _c):
                blk = torch.vmap(lambda v: torch.autograd.grad(
                    -1 * forces_flat, pos, grad_outputs=v,
                    retain_graph=True, create_graph=create_graph)[0])(eye[i:i + _c])
                out.append(blk.reshape(blk.shape[0], -1).detach())
                del blk
            return torch.cat(out, 0)
        _O.compute_hessian_vmap = _chunked
        _O._sm_chunked = chunk

    try:
        n3 = 3 * len(atoms)
        H = np.asarray(atoms.calc.get_property("hessian", atoms)).reshape(n3, n3)
        H = 0.5 * (H + H.T)             # symmetrise away numerical asymmetry
    except Exception:
        return None

    # The model computes the Hessian on raw positions and is BLIND to ASE
    # constraints - unlike atoms.get_forces(), which zeroes constrained forces.
    # Left unprojected, a slab's frozen substrate contributes its own (spurious)
    # modes: measured on OC20/OC22 that inflated the index >= 2 rate from ~10%
    # to 60-75%, and left 3 acoustic modes visible where FixAtoms should give 0.
    # Restrict to the free degrees of freedom before returning.
    fixed = set()
    for c in getattr(atoms, "constraints", []) or []:
        idx = getattr(c, "index", None)
        if idx is None and hasattr(c, "get_indices"):
            try: idx = c.get_indices()
            except Exception: idx = None
        if idx is not None:
            fixed.update(int(i) for i in np.atleast_1d(idx))
    if fixed:
        free = [i for i in range(len(atoms)) if i not in fixed]
        dof = np.array([3 * i + k for i in free for k in range(3)], dtype=int)
        H = H[np.ix_(dof, dof)]
    return H


def hessian_outputs(atoms, nev_store=8, tol=1e-2, chunk=4):
    """Exact Hessian summary for a structure, as a dict to stamp onto output.

    Returns ``{}`` when no analytical Hessian is available (direct-force model,
    hessian output not enabled on the calculator, or out of memory) so the caller
    degrades to a plain single point rather than failing the job.

    The ``eigenmode`` key is the point of the whole thing: it is the EXACT lowest
    eigenvector, and both ``dimeropt`` and ``sellaopt`` already read
    ``atoms.info['eigenmode']`` to seed their search. So a SinglePoint+Hessian
    pass feeds the next reconvergence for free, with no extra plumbing - and an
    exact mode is a strictly better seed than the stored approximate one.

    Keys: hessian_index (count below -tol), hessian_nzero, hessian_eigenvalues
    (lowest nev_store), eigenmode (N,3), curvature (lowest eigenvalue).
    """
    H = _analytic_hessian(atoms, chunk=chunk)
    if H is None:
        return {}
    evals, evecs = np.linalg.eigh(H)

    # Map the lowest eigenvector back to full (N, 3) Cartesian. The Hessian is
    # restricted to free DOF, so constrained atoms take zero displacement.
    fixed = set()
    for c in getattr(atoms, "constraints", []) or []:
        idx = getattr(c, "index", None)
        if idx is None and hasattr(c, "get_indices"):
            try: idx = c.get_indices()
            except Exception: idx = None
        if idx is not None:
            fixed.update(int(i) for i in np.atleast_1d(idx))
    free = [i for i in range(len(atoms)) if i not in fixed]
    mode = np.zeros((len(atoms), 3))
    v = evecs[:, 0]
    for k, i in enumerate(free):
        mode[i] = v[3 * k:3 * k + 3]
    n = np.linalg.norm(mode)
    if n > 1e-12:
        mode /= n

    return {
        "hessian_index": int((evals < -tol).sum()),
        "hessian_nzero": int((np.abs(evals) < 1e-3).sum()),
        "hessian_eigenvalues": [float(x) for x in evals[:nev_store]],
        "eigenmode": mode,
        "curvature": float(evals[0]),
    }


def hessian_index(atoms, nev=4, eps=2e-3, tol=1e-2, maxiter=300, analytic=True):
    """Lowest *nev* Hessian eigenvalues by finite-difference Lanczos.

    Returns ``(eigenvalues, n_negative)``. A genuine first-order saddle has
    exactly one eigenvalue below ``-tol``; ``n_negative >= 2`` means the
    structure is a higher-order saddle.

    Never forms the Hessian: ``eigsh`` only needs ``H @ v``, and ``H @ v`` is one
    central difference of forces along ``v`` (2 force calls per matvec). Cost is
    therefore ``~2 * nev * (Lanczos iterations)`` force calls, not ``6N``.

    This is deliberately independent of any optimizer's internal curvature
    model: Sella's quasi-Newton Hessian and the dimer's rotated mode are both
    *approximations* fitted during the search, so using them to certify the
    index would be marking your own homework. Both methods route through this
    same function so their reported indices are directly comparable.
    """
    from scipy.sparse.linalg import LinearOperator, eigsh

    # Exact route first: ~30-70x cheaper than the Lanczos below AND free of the
    # finite-difference step error, which was measured at up to 3.9e-3 eV/A^2 -
    # uncomfortably close to the 1e-2 tolerance this function defaults to.
    # It also yields the full spectrum, so acoustic modes are visible.
    if analytic:
        H = _analytic_hessian(atoms)
        if H is not None:
            ev = np.linalg.eigvalsh(H)
            return ([float(v) for v in ev[:nev]],
                    int(sum(1 for v in ev if v < -tol)))

    n = len(atoms)
    x_center = atoms.get_positions().copy()

    def hv(v):
        v = np.asarray(v).reshape(n, 3)
        norm = np.linalg.norm(v)
        if norm < 1e-12:
            return np.zeros(3 * n)
        u = v / norm
        atoms.set_positions(x_center + eps * u)
        f_plus = atoms.get_forces()
        atoms.set_positions(x_center - eps * u)
        f_minus = atoms.get_forces()
        atoms.set_positions(x_center)
        return (-(f_plus - f_minus) / (2 * eps) * norm).ravel()

    op = LinearOperator((3 * n, 3 * n), matvec=hv, dtype=float)
    try:
        w = eigsh(op, k=nev, which="SA", return_eigenvectors=False,
                  maxiter=maxiter, tol=1e-3)
    except Exception:
        atoms.set_positions(x_center)
        return None, None
    atoms.set_positions(x_center)
    eigs = sorted(float(x) for x in w)
    return eigs, int(sum(1 for x in eigs if x < -tol))
