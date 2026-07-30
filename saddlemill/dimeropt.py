import os
import sys
import csv
import math
import time
import tempfile
from collections import deque
import traceback
import random
import zipfile
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from ase.neighborlist import natural_cutoffs, neighbor_list
from ase.io import Trajectory
from ase.mep import DimerControl, MinModeAtoms, MinModeTranslate
from ase.calculators.singlepoint import SinglePointCalculator
from saddlemill.dimertools.structure_edit import get_attempts
from saddlemill.tools import (backup_flux_logs, get_task_name, resolve_vasp_calc,
                              remove_vasp_heavies, finalize_if_vasp_interactive,
                              archive_and_clear_temp_files)


class StopRun(Exception):
    pass


MODE_DIAGNOSTIC_FIELDS = [
    "trace_id",
    "trace_start_unix_ns",
    "src_index",
    "rank",
    "attempt_id",
    "selected_index",
    "reaction_type",
    "translation_step",
    "force_calls",
    "curvature",
    "participation_ratio",
    "angle_from_previous_deg",
    "angle_from_initial_deg",
    "pre_post_angle_w5_deg",
    "pre_coherence_w5",
    "post_coherence_w5",
    "atom_participation_overlap_w5",
    "pre_post_angle_w10_deg",
    "pre_coherence_w10",
    "post_coherence_w10",
    "atom_participation_overlap_w10",
]

REACTION_DIAGNOSTIC_FIELDS = [
    "src_index",
    "rank",
    "attempt_id",
    "selected_index",
    "configured_reaction_type",
    "initial_reaction_type",
    "final_reaction_type",
    "converged",
    "n_force_calls",
    "status",
    "classification_source",
    "classification_confidence",
]

OPTIMIZER_DIAGNOSTIC_FIELDS = [
    "record_type",
    "trace_id",
    "trace_start_unix_ns",
    "src_index",
    "rank",
    "attempt_id",
    "selected_index",
    "reaction_type",
    "accepted_translation_step",
    "translation_algorithm",
    "hybrid_state",
    "hybrid_switch_event",
    "projected_fmax",
    "curvature",
    "translation_regime",
    "step_norm",
    "step_clipped",
    "direction_alignment",
    "hybrid_history_pairs_at_switch",
    "hybrid_warm_start_history",
    "rotation_optimizer",
    "rotation_steps",
    "rotation_lbfgs_history_size",
    "rotation_lbfgs_pairs_accepted",
    "rotation_lbfgs_pairs_rejected",
    "rotation_lbfgs_resets",
    "rotation_lbfgs_fallbacks",
    "kappa_rotation_optimizer",
    "kappa_rotation_steps",
    "kappa_rotation_lbfgs_pairs_accepted",
    "kappa_rotation_lbfgs_pairs_rejected",
    "translation_lbfgs_history_size",
    "translation_lbfgs_pairs_accepted_total",
    "translation_lbfgs_pairs_rejected_total",
    "translation_lbfgs_resets",
    "translation_lbfgs_last_reset_reason",
    "force_calls_step_entry",
    "force_calls_center_and_rotation",
    "force_calls_translation_trial",
    "force_calls_cumulative_after_step",
    "final_force_calls",
    "force_calls_final_evaluation",
    "final_translation_steps",
    "converged",
    "status",
]


def _append_csv_row(path, fieldnames, row):
    """Append one row and create a header only for a new/empty shard."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    is_new = not os.path.exists(path) or os.path.getsize(path) == 0
    with open(path, "a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if is_new:
            writer.writeheader()
        writer.writerow({key: row.get(key, "") for key in fieldnames})


def _remove_csv_rows(path, match):
    """Atomically remove prior diagnostic rows for one active attempt.

    SaddleMill's normal resume cleanup does not know about these new diagnostic
    CSVs. Removing only the matching diagnostic rows keeps them aligned with
    the active status/output entry when an attempt is rerun.
    """
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return

    with open(path, newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames
        if not fieldnames or not all(key in fieldnames for key in match):
            return
        rows = []
        found = False
        for row in reader:
            is_match = all(
                str(row.get(key, "")) == str(value)
                for key, value in match.items()
            )
            if is_match:
                found = True
            else:
                rows.append(row)

    if not found:
        return

    directory = os.path.dirname(path) or "."
    with tempfile.NamedTemporaryFile(
        mode="w", newline="", dir=directory, delete=False
    ) as handle:
        temp_path = handle.name
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temp_path, path)


def _split_reaction_types(config_dict):
    value = config_dict.get("ourDimer", {}).get("reaction_types", [])
    if isinstance(value, str):
        return value.split()
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value]
    return []


def _configured_attempt_reaction_types(config_dict):
    """Return the configured base reaction type for each attempt index."""
    reaction_types = _split_reaction_types(config_dict)
    if "initial_guess" in reaction_types:
        return ["initial_guess"]

    counts_value = config_dict.get("ourDimer", {}).get(
        "num_attempts_per_type", 1
    )
    if isinstance(counts_value, str):
        parts = counts_value.split()
        counts = [int(item) for item in parts] if len(parts) > 1 else [int(parts[0])]
    elif isinstance(counts_value, (list, tuple)):
        counts = [int(item) for item in counts_value]
    else:
        counts = [int(counts_value)]

    if len(counts) == 1:
        counts *= len(reaction_types)
    if len(counts) != len(reaction_types):
        return []

    configured = []
    for reaction_type, count in zip(reaction_types, counts):
        configured.extend([reaction_type] * count)
    return configured


def _safe_float(value):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return ""
    return value if np.isfinite(value) else ""


def _normalize_mode(mode, free_indices):
    arr = np.asarray(mode, dtype=float)
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError(f"Expected eigenmode shape (N, 3), got {arr.shape}")
    arr = arr[np.asarray(free_indices, dtype=int)]
    norm = float(np.linalg.norm(arr))
    if norm < 1e-14:
        raise ValueError("Eigenmode norm over free atoms is approximately zero")
    return arr / norm


def _mode_angle_deg(mode_a, mode_b):
    """Projective angle: mode and -mode are treated as identical."""
    dot = float(np.vdot(mode_a.ravel(), mode_b.ravel()).real)
    dot = float(np.clip(abs(dot), 0.0, 1.0))
    return math.degrees(math.acos(dot))


def _sign_align_modes(modes):
    """Sequentially sign-align a mode sequence without changing directions."""
    aligned = [np.array(modes[0], copy=True)]
    for mode in modes[1:]:
        candidate = np.array(mode, copy=True)
        if np.vdot(aligned[-1].ravel(), candidate.ravel()).real < 0.0:
            candidate *= -1.0
        aligned.append(candidate)
    return aligned


def _window_mode_statistics(history, window):
    history = list(history)
    if len(history) < 2 * window:
        return ("", "", "", "")

    pre = _sign_align_modes(history[-2 * window:-window])
    post = _sign_align_modes(history[-window:])

    pre_sum = np.sum(pre, axis=0)
    post_sum = np.sum(post, axis=0)
    pre_norm = float(np.linalg.norm(pre_sum))
    post_norm = float(np.linalg.norm(post_sum))
    if pre_norm < 1e-14 or post_norm < 1e-14:
        return ("", "", "", "")

    pre_mean = pre_sum / pre_norm
    post_mean = post_sum / post_norm
    angle = _mode_angle_deg(pre_mean, post_mean)
    pre_coherence = pre_norm / window
    post_coherence = post_norm / window

    pre_weights = np.mean(
        [np.sum(mode * mode, axis=1) for mode in pre], axis=0
    )
    post_weights = np.mean(
        [np.sum(mode * mode, axis=1) for mode in post], axis=0
    )
    pre_weights /= pre_weights.sum()
    post_weights /= post_weights.sum()
    atom_overlap = float(np.sum(np.sqrt(pre_weights * post_weights)))

    return angle, pre_coherence, post_coherence, atom_overlap


class ModeDiagnosticRecorder:
    """Write one sign-invariant eigenmode diagnostic row per translation state.

    ASE calls optimizer observers once at step 0 after the initial force/mode
    evaluation, and then after every translation step. Therefore the first row
    is the rotated mode at the initial displaced geometry, not the unrefined
    random seed direction.
    """

    def __init__(self, path, src_index, rank, attempt_id, selected_index,
                 reaction_type, d_atoms, dim_rlx, free_indices):
        self.path = path
        self.src_index = src_index
        self.rank = rank
        self.attempt_id = attempt_id
        self.selected_index = selected_index
        self.reaction_type = reaction_type
        self.d_atoms = d_atoms
        self.dim_rlx = dim_rlx
        self.free_indices = list(free_indices)
        self.trace_start_unix_ns = time.time_ns()
        self.trace_id = (
            f"{src_index}-{attempt_id}-{rank}-"
            f"{os.getpid()}-{self.trace_start_unix_ns}"
        )
        self.history = deque(maxlen=20)
        self.initial_mode = None
        self.disabled = False
        self.warned = False

    def __call__(self):
        if self.disabled:
            return
        try:
            mode = _normalize_mode(
                self.d_atoms.get_eigenmode(), self.free_indices
            )
            if self.initial_mode is None:
                self.initial_mode = np.array(mode, copy=True)

            previous_angle = (
                _mode_angle_deg(self.history[-1], mode)
                if self.history else 0.0
            )
            initial_angle = _mode_angle_deg(self.initial_mode, mode)
            self.history.append(np.array(mode, copy=True))

            atom_weights = np.sum(mode * mode, axis=1)
            participation_ratio = 1.0 / (
                len(atom_weights) * np.sum(atom_weights * atom_weights)
            )
            w5 = _window_mode_statistics(self.history, 5)
            w10 = _window_mode_statistics(self.history, 10)

            try:
                curvature = self.d_atoms.get_curvature()
            except Exception:
                curvature = ""
            try:
                force_calls = self.d_atoms.control.get_counter("forcecalls")
            except Exception:
                force_calls = ""

            _append_csv_row(
                self.path,
                MODE_DIAGNOSTIC_FIELDS,
                {
                    "trace_id": self.trace_id,
                    "trace_start_unix_ns": self.trace_start_unix_ns,
                    "src_index": self.src_index,
                    "rank": self.rank,
                    "attempt_id": self.attempt_id,
                    "selected_index": self.selected_index,
                    "reaction_type": self.reaction_type,
                    "translation_step": self.dim_rlx.nsteps,
                    "force_calls": force_calls,
                    "curvature": _safe_float(curvature),
                    "participation_ratio": participation_ratio,
                    "angle_from_previous_deg": previous_angle,
                    "angle_from_initial_deg": initial_angle,
                    "pre_post_angle_w5_deg": w5[0],
                    "pre_coherence_w5": w5[1],
                    "post_coherence_w5": w5[2],
                    "atom_participation_overlap_w5": w5[3],
                    "pre_post_angle_w10_deg": w10[0],
                    "pre_coherence_w10": w10[1],
                    "post_coherence_w10": w10[2],
                    "atom_participation_overlap_w10": w10[3],
                },
            )
        except Exception as exc:
            self.disabled = True
            if not self.warned:
                self.warned = True
                print(
                    f"Mode diagnostics disabled for structure {self.src_index}, "
                    f"attempt {self.attempt_id}: {exc}",
                    flush=True,
                )



class OptimizerDiagnosticRecorder:
    """Write one row per accepted translation and one final summary row."""

    def __init__(self, path, mode_recorder, d_atoms, dim_rlx):
        self.path = path
        self.mode_recorder = mode_recorder
        self.d_atoms = d_atoms
        self.dim_rlx = dim_rlx
        self.last_serial = 0
        self.last_cumulative_after_step = 0
        self.summary_written = False

    def _base_row(self):
        recorder = self.mode_recorder
        return {
            "trace_id": recorder.trace_id,
            "trace_start_unix_ns": recorder.trace_start_unix_ns,
            "src_index": recorder.src_index,
            "rank": recorder.rank,
            "attempt_id": recorder.attempt_id,
            "selected_index": recorder.selected_index,
            "reaction_type": recorder.reaction_type,
        }

    def __call__(self):
        diagnostics = getattr(self.dim_rlx, "last_step_diagnostics", None)
        if not diagnostics:
            return
        serial = int(diagnostics.get("diagnostic_serial", 0))
        if serial <= self.last_serial:
            return
        row = self._base_row()
        row["record_type"] = "step"
        row.update(diagnostics)
        _append_csv_row(self.path, OPTIMIZER_DIAGNOSTIC_FIELDS, row)
        self.last_serial = serial
        try:
            self.last_cumulative_after_step = int(
                diagnostics.get("force_calls_cumulative_after_step", 0)
            )
        except (TypeError, ValueError):
            self.last_cumulative_after_step = 0

    def write_summary(self, status, converged):
        if self.summary_written:
            return
        # Flush a last accepted step if the observer was interrupted by StopRun.
        self()
        row = self._base_row()
        final_force_calls = self.d_atoms.control.get_counter("forcecalls")
        row.update({
            "record_type": "summary",
            "final_force_calls": final_force_calls,
            "force_calls_final_evaluation": (
                int(final_force_calls) - self.last_cumulative_after_step
            ),
            "final_translation_steps": self.dim_rlx.nsteps,
            "converged": int(bool(converged)),
            "status": status,
        })
        _append_csv_row(self.path, OPTIMIZER_DIAGNOSTIC_FIELDS, row)
        self.summary_written = True


def _setup_dimer(atoms, calc, eigenmode=None, displacement_dict=None,
                 dimer_control_kwargs=None, control_logfile=None,
                 mode_logfile=None, logfile=None, trajectory=None,
                 engine="ase", kappa_kwargs=None, kappa_control_kwargs=None,
                 rotation_optimizer="ase", translation_optimizer="ase",
                 rotation_lbfgs_options=None,
                 translation_lbfgs_options=None, hybrid_options=None):
    """Create the selected dimer engine, rotation solver, and translator.

    The default ``ase/ase/ase`` combination preserves the stock scientific
    method.  L-BFGS and hybrid behavior is opt-in through [ourDimer].
    """
    from saddlemill.dimertools.lbfgs_dimer import (
        ConfigurableRotationMinModeAtoms,
        DiagnosticMinModeTranslate,
        HybridMinModeTranslate,
        LBFGSMinModeTranslate,
    )

    atoms.calc = calc
    eig_kw = {"eigenmodes": [np.array(eigenmode)]} if eigenmode is not None else {}
    rotation_optimizer = str(rotation_optimizer).lower()
    translation_optimizer = str(translation_optimizer).lower()

    if engine == "kappa":
        from saddlemill.dimertools.kappa_dimer import (
            KappaMinModeAtoms, IsolatedDimerControl
        )

        d_control = IsolatedDimerControl(
            logfile=control_logfile,
            eigenmode_logfile=mode_logfile,
            **(dimer_control_kwargs or {}),
        )
        kw = dict(kappa_kwargs or {})
        kw.update({
            "rotation_optimizer": rotation_optimizer,
            "rotation_lbfgs_options": rotation_lbfgs_options or {},
        })
        if kappa_control_kwargs:
            kw["kappa_control"] = IsolatedDimerControl(
                logfile=control_logfile, **kappa_control_kwargs
            )
        d_atoms = KappaMinModeAtoms(atoms, control=d_control, **eig_kw, **kw)
    elif engine == "ase":
        d_control = DimerControl(
            logfile=control_logfile,
            eigenmode_logfile=mode_logfile,
            **(dimer_control_kwargs or {}),
        )
        d_atoms = ConfigurableRotationMinModeAtoms(
            atoms,
            control=d_control,
            rotation_optimizer=rotation_optimizer,
            rotation_lbfgs_options=rotation_lbfgs_options or {},
            **eig_kw,
        )
    else:
        raise ValueError(
            f"Unknown [ourDimer] engine={engine!r}; expected 'ase' or 'kappa'."
        )

    if displacement_dict:
        d_atoms.displace(**displacement_dict)
    else:
        d_atoms.displace(
            displacement_vector=np.random.randn(len(atoms), 3) * 1e-10,
            method="vector",
        )

    common = dict(trajectory=trajectory, logfile=logfile)
    if translation_optimizer == "ase":
        dim_rlx = DiagnosticMinModeTranslate(d_atoms, **common)
    elif translation_optimizer == "lbfgs":
        dim_rlx = LBFGSMinModeTranslate(
            d_atoms, lbfgs_options=translation_lbfgs_options or {}, **common
        )
    elif translation_optimizer in {"hybrid", "fire_lbfgs"}:
        dim_rlx = HybridMinModeTranslate(
            d_atoms,
            lbfgs_options=translation_lbfgs_options or {},
            hybrid_options=hybrid_options or {},
            **common,
        )
    else:
        raise ValueError(
            "[ourDimer] translation_optimizer must be ase, lbfgs, or fire_lbfgs; "
            f"got {translation_optimizer!r}."
        )
    return d_atoms, dim_rlx

def _refine_eigenmode(atoms, calc, eigenmode, dimer_control_kwargs=None,
                      control_logfile=None):
    """Refine eigenmode via dimer rotation only (no translation).

    Works on a copy of *atoms* — the original is never modified.
    Returns (refined_eigenmode, curvature).
    """
    refine_atoms = atoms.copy()
    refine_atoms.calc = calc
    d_control = DimerControl(logfile=control_logfile,
                             **(dimer_control_kwargs or {}))
    d_atoms = MinModeAtoms(refine_atoms, d_control,
                           eigenmodes=[np.array(eigenmode)])
    d_atoms.displace(displacement_vector=np.random.randn(len(refine_atoms), 3) * 1e-10,
                     method='vector')
    # get_forces() triggers eigenmode rotation (up to max_num_rot iterations).
    # No translation — only the eigenmode direction and curvature are updated.
    d_atoms.get_forces()
    return d_atoms.get_eigenmode(), float(d_atoms.get_curvature())


def dimeropt(i, config_dict, atoms_orig, calc, consecutive_errors=None, executorlib_worker_id=None, **kwargs):

    rank = executorlib_worker_id

    seed_offset = int(os.environ.get("SM_SEED_OFFSET", config_dict["Main"].get("seed_offset", 0)))
    seed = i + seed_offset * 100000

    random.seed(seed)
    np.random.seed(seed)

    method_name = config_dict["Main"]["method"]
    status_dir = f"{method_name}_status_csvs"
    status_file = f"{status_dir}/status_rank_{rank}.csv"
    reaction_file = f"{status_dir}/reaction_rank_{rank}.csv"
    # Preserve the existing compact reaction CSV for compatibility.
    rxn_file = f"{method_name}_rxn_csvs/rxn_rank_{rank}.csv"
    mode_file = f"{method_name}_mode_csvs/mode_rank_{rank}.csv"
    optimizer_file = f"{method_name}_optimizer_csvs/optimizer_rank_{rank}.csv"
    os.makedirs(status_dir, exist_ok=True)
    os.makedirs(f"{method_name}_rxn_csvs", exist_ok=True)
    os.makedirs(f"{method_name}_mode_csvs", exist_ok=True)
    os.makedirs(f"{method_name}_optimizer_csvs", exist_ok=True)
    my_output_file = f"{method_name}_trajes/collected_ts_rank_{rank}.traj"
    zip_name = f"{method_name}_debug_zips/structure_rank_{rank}_data.zip"
    task_name = get_task_name(config_dict)
    is_vasp = config_dict["Main"]["Calculator"] in ("Vasp", "VaspInteractive")

    max_consecutive_errors = config_dict["Main"]["max_consecutive_errors"]
    if consecutive_errors is not None and consecutive_errors[0] >= max_consecutive_errors > 0:
        print(f"Rank {rank}: {consecutive_errors[0]} consecutive structures errored. Killing worker for restart.", flush=True)
        backup_flux_logs(rank)
        sys.exit(1)

    configured_attempt_types = _configured_attempt_reaction_types(config_dict)

    lbfgs_cfg = config_dict.get("ourDimerLBFGS", {}) or {}
    rotation_lbfgs_options = {
        "memory": int(lbfgs_cfg.get("rotation_memory", 10)),
        "initial_hessian": float(
            lbfgs_cfg.get("rotation_initial_hessian", 1.0)
        ),
        "dynamic_h0": bool(lbfgs_cfg.get("rotation_dynamic_h0", False)),
        "curvature_epsilon": float(
            lbfgs_cfg.get("curvature_epsilon", 1.0e-12)
        ),
    }
    translation_lbfgs_options = {
        "memory": int(lbfgs_cfg.get("translation_memory", 10)),
        "initial_hessian": float(
            lbfgs_cfg.get("translation_initial_hessian", 1.0)
        ),
        "dynamic_h0": bool(
            lbfgs_cfg.get("translation_dynamic_h0", False)
        ),
        "curvature_epsilon": float(
            lbfgs_cfg.get("curvature_epsilon", 1.0e-12)
        ),
        "damping": float(lbfgs_cfg.get("translation_damping", 1.0)),
        "reset_on_regime_change": bool(
            lbfgs_cfg.get("reset_translation_on_regime_change", True)
        ),
    }
    hybrid_cfg = config_dict.get("ourDimerHybrid", {}) or {}
    hybrid_options = {
        "enabled": bool(hybrid_cfg.get("enabled", False)),
        "enter_fmax": float(hybrid_cfg.get("enter_fmax", 0.30)),
        "exit_fmax": float(hybrid_cfg.get("exit_fmax", 0.50)),
        "enter_curvature": float(
            hybrid_cfg.get("enter_curvature", -0.05)
        ),
        "exit_curvature": float(hybrid_cfg.get("exit_curvature", 0.00)),
        "enter_stable_steps": int(
            hybrid_cfg.get("enter_stable_steps", 3)
        ),
        "exit_stable_steps": int(hybrid_cfg.get("exit_stable_steps", 2)),
        "minimum_history_pairs": int(
            hybrid_cfg.get("minimum_history_pairs", 3)
        ),
        "warm_start_history": bool(
            hybrid_cfg.get("warm_start_history", True)
        ),
        "reset_history_on_exit": bool(
            hybrid_cfg.get("reset_history_on_exit", True)
        ),
        "fire_dt": float(hybrid_cfg.get("fire_dt", 0.10)),
        "fire_dtmax": float(hybrid_cfg.get("fire_dtmax", 1.0)),
        "fire_Nmin": int(hybrid_cfg.get("fire_Nmin", 5)),
        "fire_finc": float(hybrid_cfg.get("fire_finc", 1.1)),
        "fire_fdec": float(hybrid_cfg.get("fire_fdec", 0.5)),
        "fire_astart": float(hybrid_cfg.get("fire_astart", 0.1)),
        "fire_fa": float(hybrid_cfg.get("fire_fa", 0.99)),
    }

    def configured_reaction_type(attempt):
        if isinstance(attempt, int) and 0 <= attempt < len(configured_attempt_types):
            return configured_attempt_types[attempt]
        return "unknown"

    def log_status(attempt, slctd_indx, status_msg, n_force_calls=0):
        with open(status_file, 'a') as f:
            f.write(f'{i},{rank},{attempt},{slctd_indx},{n_force_calls},"{status_msg}"\n')

    def log_rxn_legacy(attempt, reaction_type, converged, n_force_calls):
        with open(rxn_file, 'a') as f:
            f.write(f'{i},{attempt},{reaction_type},{int(converged)},{n_force_calls}\n')

    def log_reaction(attempt, slctd_indx, configured_type, initial_type,
                     final_type, converged, n_force_calls, status_msg,
                     source="runtime_metadata", confidence="exact"):
        _append_csv_row(
            reaction_file,
            REACTION_DIAGNOSTIC_FIELDS,
            {
                "src_index": i,
                "rank": rank,
                "attempt_id": attempt,
                "selected_index": slctd_indx,
                "configured_reaction_type": configured_type,
                "initial_reaction_type": initial_type,
                "final_reaction_type": final_type,
                "converged": int(bool(converged)),
                "n_force_calls": int(n_force_calls or 0),
                "status": status_msg,
                "classification_source": source,
                "classification_confidence": confidence,
            },
        )

    # --- MAIN LOOP ---
    any_attempt_succeeded = False
    all_attempts_none = False

    continuation_data = kwargs.get('continuation_data')  # {attempt_id: Atoms} or None
    entries_to_run = kwargs.get('entries_to_run')        # set of attempt_ids or None

    with Trajectory(my_output_file, 'a') as writer:

        attempt = "init"
        slctd_indx = -1
        temp_files = []

        generated = get_attempts(atoms_orig, config_dict)
        all_attempts_none = all(a is None for a in generated[0])
        if all_attempts_none:
            print(f"Rank {rank} WARNING on structure {i}: "
                  "All attempts failed to generate.", flush=True)

        attempts_iter = enumerate(zip(*generated))

        for attempt, (atoms, displacement_dict, slctd_indx) in attempts_iter:

            if entries_to_run is not None and attempt not in entries_to_run:
                continue

            configured_type = configured_reaction_type(attempt)
            initial_reaction_type = configured_type
            reaction_source = "configured_attempt_order"

            # Keep the new diagnostic shards aligned with the active result on
            # resume/retry. Existing status/output cleanup remains unchanged.
            diagnostic_key = {"src_index": i, "attempt_id": attempt}
            # Reaction metadata is one active row per attempt. Mode traces are
            # append-only and carry a unique trace_id, avoiding an expensive
            # rewrite of a potentially large mode-history shard on resume.
            _remove_csv_rows(reaction_file, diagnostic_key)

            if atoms is None:
                status_msg = "error: failed to generate attempt"
                log_status(attempt, -1, status_msg)
                log_reaction(
                    attempt, -1, configured_type, configured_type,
                    configured_type, False, 0, status_msg,
                    source="configured_attempt_order",
                    confidence="base_type_only",
                )
                continue

            initial_reaction_type = atoms.info.get(
                "reaction_type",
                atoms.info.get("orig_info", {}).get(
                    "reaction_type", configured_type
                ),
            )
            reaction_source = "generated_attempt_metadata"

            # Use continuation structure if available for this attempt.
            if continuation_data and attempt in continuation_data:
                atoms = continuation_data[attempt]
                displacement_dict = {"displacement_vector": np.random.randn(len(atoms), 3) * 1e-10, "method": "vector"}
                continuation_reaction_type = atoms.info.get(
                    "reaction_type",
                    atoms.info.get("orig_info", {}).get(
                        "reaction_type", initial_reaction_type
                    ),
                )
                # A previous desorption label is an outcome, not the attempt's
                # initialization mechanism. Preserve the freshly generated
                # mechanism in that case.
                if continuation_reaction_type != "desorption":
                    initial_reaction_type = continuation_reaction_type
                reaction_source = "continuation_metadata"

            temp_log = f'dimer_control_{i}_{attempt}_{slctd_indx}.log'
            temp_opt_log = f'dimer_opt_{i}_{attempt}_{slctd_indx}.log'
            temp_traj = f'dimer_{i}_{attempt}_{slctd_indx}.traj'
            temp_mode_log = f'dimer_mode_{i}_{attempt}_{slctd_indx}.log'
            temp_files = [temp_log, temp_opt_log, temp_traj, temp_mode_log]
            attempt_vasp_dir = f"VASP_{i}_{attempt}" if is_vasp else None
            if attempt_vasp_dir is not None:
                temp_files.append(attempt_vasp_dir)
            attempt_calc = None
            d_atoms = None
            dim_rlx = None
            optimizer_recorder = None

            try:
                # Handle constraints:
                if atoms.constraints:
                    free_indices = [atom.index for atom in atoms if atom.index not in atoms.constraints[0].get_indices()]
                else:
                    free_indices = [atom.index for atom in atoms]

                # Use existing eigenmode if available (top level from
                # get_attempts/initial_guess, or orig_info from continuation),
                # otherwise let ASE derive one from the displacement.
                eigenmode = atoms.info.get('eigenmode')
                if eigenmode is None:
                    eigenmode = atoms.info.get('orig_info', {}).get('eigenmode')
                if eigenmode is not None:
                    eigenmode = np.array(eigenmode)

                attempt_calc = resolve_vasp_calc(config_dict, calc, i, attempt, "ourDimer", atoms=atoms)
                d_atoms, dim_rlx = _setup_dimer(
                    atoms, attempt_calc, eigenmode=eigenmode,
                    displacement_dict=displacement_dict,
                    dimer_control_kwargs=config_dict["DimerControl"],
                    control_logfile=temp_log,
                    mode_logfile=temp_mode_log,
                    logfile=temp_opt_log, trajectory=temp_traj,
                    engine=config_dict["ourDimer"]["engine"],
                    kappa_kwargs={
                        "beta": config_dict["ourDimer"]["kappa_beta"],
                        "recover_fmax": config_dict["ourDimer"]["kappa_recover_fmax"],
                    },
                    kappa_control_kwargs=(config_dict.get("Kappa") or None),
                    rotation_optimizer=config_dict["ourDimer"].get(
                        "rotation_optimizer", "ase"
                    ),
                    translation_optimizer=config_dict["ourDimer"].get(
                        "translation_optimizer", "ase"
                    ),
                    rotation_lbfgs_options=rotation_lbfgs_options,
                    translation_lbfgs_options=translation_lbfgs_options,
                    hybrid_options=hybrid_options,
                )

                # Diagnostic only: this observer never raises into the optimizer
                # and does not trigger force evaluations or alter the mode.
                mode_recorder = ModeDiagnosticRecorder(
                    mode_file,
                    src_index=i,
                    rank=rank,
                    attempt_id=attempt,
                    selected_index=slctd_indx,
                    reaction_type=initial_reaction_type,
                    d_atoms=d_atoms,
                    dim_rlx=dim_rlx,
                    free_indices=free_indices,
                )
                dim_rlx.attach(mode_recorder, interval=1)
                optimizer_recorder = OptimizerDiagnosticRecorder(
                    optimizer_file, mode_recorder, d_atoms, dim_rlx
                )
                dim_rlx.attach(optimizer_recorder, interval=1)

                # PR Check — skip early steps to let the dimer rotate
                # the eigenmode (initial displacement can look delocalized,
                # especially for diffusion/rotation types).
                delocalization_start_step = max(1, int(0.1 * config_dict["Main"]["steps"]))

                def check_delocalization():
                    if dim_rlx.nsteps < delocalization_start_step:
                        return
                    mode = d_atoms.get_eigenmode()
                    v2 = (mode**2).sum(axis=1)
                    v2 = v2[free_indices]
                    sum_v2 = np.sum(v2)
                    if sum_v2 < 1e-12: return
                    pr = (sum_v2**2) / (len(v2) * np.sum(v2**2))
                    if pr > config_dict["ourDimer"]["delocalization_threshold"]:
                        raise StopRun(f"Eigenmode Delocalized (PR={pr:.3f})")

                def check_desorption():
                    check_atoms = d_atoms.atoms
                    cutoffs = natural_cutoffs(check_atoms, mult=2.0)
                    i, j = neighbor_list('ij', check_atoms, cutoffs)
                    adjacency = csr_matrix((np.ones(len(i)), (i, j)), shape=(len(check_atoms), len(check_atoms)))
                    n_components, labels = connected_components(adjacency, connection='weak')
                    if n_components > 1:
                        raise StopRun(f"Adsorbate desorbed")

                dim_rlx.attach(check_delocalization, interval=5)
                dim_rlx.attach(check_desorption, interval=5)

                stop_reason = None
                stopped_early = False
                converged = False
                try:
                    converged = dim_rlx.run(fmax=config_dict["Main"]["fmax"], steps=config_dict["Main"]["steps"])
                except StopRun as e:
                    stopped_early = True
                    stop_reason = str(e)
                    converged = False

                if converged:
                    status = "converged"
                elif not converged and not stopped_early:
                    # Extension check
                    fmax_check = np.sqrt((d_atoms.get_forces()**2).sum(axis=1).max()) < config_dict['ourDimer']['extension_check_fmax']
                    curvature_check = d_atoms.get_curvature() < config_dict['ourDimer']['extension_check_curvature']
                    if fmax_check and curvature_check:
                        try:
                            converged = dim_rlx.run(fmax=config_dict["Main"]["fmax"], steps=150)
                        except StopRun as e:
                            stopped_early = True
                            stop_reason = str(e)
                            converged = False

                        if converged:
                            status = "converged_after_extension"
                        else:
                            status = "not_converged_after_extension"
                    else:
                        status = "not_converged"
                else:
                    status = "not_converged_StopRun"

                # Metadata
                eigenmode = d_atoms.get_eigenmode()
                curvature = d_atoms.get_curvature()
                n_force_calls = d_atoms.control.get_counter('forcecalls')
                energy = atoms.get_potential_energy()
                forces = atoms.get_forces()
                finalize_if_vasp_interactive(config_dict, attempt_calc)
                if attempt_vasp_dir is not None:
                    remove_vasp_heavies(attempt_vasp_dir)

                atoms.info['eigenmode'] = eigenmode
                atoms.info['curvature'] = float(curvature)
                atoms.info['n_force_calls'] = int(n_force_calls)
                atoms.info['converged'] = 1 if converged else 0
                atoms.info['src_index'] = i
                atoms.info['attempt_id'] = attempt
                atoms.info['stoprun'] = 1 if stopped_early else 0
                atoms.info['selected_index'] = slctd_indx
                orig = atoms.info.get('orig_info', {})
                atoms.info['reaction_type'] = atoms.info.get('reaction_type', orig.get('reaction_type', 'unknown'))
                if stop_reason and "desorbed" in stop_reason:
                    status = "converged_to_desorption"
                    atoms.info['converged'] = 1
                    atoms.info['reaction_type'] = 'desorption'
                if optimizer_recorder is not None:
                    optimizer_recorder.write_summary(status, atoms.info['converged'])
                atoms.info['status'] = status
                atoms.info['task_name'] = task_name
                atoms.wrap()
                atoms.calc = SinglePointCalculator(atoms, energy=energy, forces=forces)

                writer.write(atoms)

                # Clean up temp files (the zip block below walks directories too,
                # so the per-attempt VASP dir is captured automatically).
                archive_and_clear_temp_files(temp_files, zip_name, prefix="",
                                   enabled=config_dict['Main']['zip'])

                log_status(attempt, slctd_indx, status, n_force_calls)
                log_rxn_legacy(
                    attempt, atoms.info['reaction_type'],
                    atoms.info['converged'], atoms.info['n_force_calls']
                )
                log_reaction(
                    attempt,
                    slctd_indx,
                    configured_type,
                    initial_reaction_type,
                    atoms.info['reaction_type'],
                    atoms.info['converged'],
                    atoms.info['n_force_calls'],
                    status,
                    source=reaction_source,
                    confidence="exact",
                )
                any_attempt_succeeded = True

            except Exception as e:
                print(f"Rank {rank} FAILED on structure {i}, attempt {attempt}: {e}", flush=True)
                print(f"\nTraceback details:\n{traceback.format_exc()}", flush=True)
                if attempt_calc is not None:
                    finalize_if_vasp_interactive(config_dict, attempt_calc)
                if optimizer_recorder is not None and d_atoms is not None:
                    try:
                        optimizer_recorder.write_summary(
                            f"error: {str(e)}", False
                        )
                    except Exception:
                        pass
                archive_and_clear_temp_files(temp_files, zip_name, prefix="ERROR_",
                                   enabled=config_dict['Main']['zip'])
                status_msg = f"error: {str(e)}"
                try:
                    error_force_calls = (
                        d_atoms.control.get_counter('forcecalls')
                        if d_atoms is not None else 0
                    )
                except Exception:
                    error_force_calls = 0
                log_status(attempt, slctd_indx, status_msg, error_force_calls)
                log_reaction(
                    attempt,
                    slctd_indx,
                    configured_type,
                    initial_reaction_type,
                    initial_reaction_type,
                    False,
                    error_force_calls,
                    status_msg,
                    source=reaction_source,
                    confidence="exact" if reaction_source != "configured_attempt_order" else "base_type_only",
                )

    # Track consecutive structure-level errors for worker health
    if consecutive_errors is not None:
        if any_attempt_succeeded:
            consecutive_errors[0] = 0
        elif all_attempts_none:
            pass  # Data issue (e.g., no adsorbate atoms), not a worker error
        else:
            consecutive_errors[0] += 1