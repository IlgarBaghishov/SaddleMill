"""User-pluggable VASP I/O: input generation, extra input-file writers, and output parsers.

A *generator* maps an ASE ``Atoms`` object to a dict of ASE-``Vasp`` keyword
arguments (lowercased INCAR tags plus ``kpts``/``gamma``/``setups``/``magmom``).
SaddleMill merges this dict UNDER the ``[Vasp]`` section — explicit ``[Vasp]``
keys always win — and hands the result to the calculator. Because ASE then
writes every input file itself, atom sorting, the resort that maps forces back,
POTCAR selection, and the ``VaspInteractive`` interactive flags are all handled
correctly. The generator only decides *what settings to use*, never writes files.

Selection lives in ``[ourVasp] input_generator`` and may be:
  - a built-in name: ``omat24_static``, ``omat24_relax``, ``cheap_omat``,
    ``oc20``, ``cheap_oc20``, ``oc22``, ``cheap_oc22``
  - ``package.module:func`` — import an installed module and use ``func``
  - ``/abs/or/rel/file.py:func`` — load a local ``.py`` file and use ``func``

A custom generator is any callable ``generator(atoms) -> dict`` of ASE-Vasp
kwargs, so users can plug in their own input recipes without touching SaddleMill.

**Ionic-driver tags are stripped.** ``IBRION``/``NSW``/``POTIM``/``EDIFFG`` are
removed from generator output because SaddleMill drives geometry through ASE
optimizers (each VASP call is a single-point force evaluation), and
``VaspInteractive`` forbids overriding ``IBRION``/``POTIM``. Set ``ISIF`` in
``[Vasp]`` if you need stress for cell relaxation. Built-in input sets such as
OMat24/OC20 are authored for *standalone* VASP relaxations, so their relaxation
tags are intentionally dropped here.

----

This module also provides **extra-input-file writers** (``[ourVasp]
extra_input_files``): callables ``writer(calc, atoms, directory) -> None`` that
drop additional files into the VASP working directory *after* ASE has written
INCAR/POSCAR/etc. (so ``calc.sort`` and the directory exist) and *before* VASP
runs. The motivating case is ``modecar`` — a VTST MODECAR built from
``atoms.info['eigenmode']``, reordered to POSCAR order via ``calc.sort``. Same
selection grammar as ``input_generator`` (built-in name, ``module:func``,
``file.py:func``), and a space-separated list runs several writers in order.
Unlike ``input_generator`` (which only computes settings), these write files,
so the hook lives in a ``write_input`` subclass — see ``tools._with_extra_input_files``.
"""
import importlib
import importlib.util
import os
import warnings

# INCAR tags selecting VASP's *internal* ionic driver. SaddleMill controls
# geometry via ASE optimizers, so these must never come from a generator.
_DRIVER_KEYS = {"ibrion", "nsw", "potim", "ediffg"}

# Lanthanides whose f-in-core ``_3`` POTCARs the cheap_* generators must keep:
# the f-in-valence potentials that a light/minimal base would pick make the SCF
# non-convergent on partially-filled-4f systems. (La/Ce/Eu/Gd stay standard.)
_LANTH_FCORE = {"Pr", "Nd", "Pm", "Sm", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu"}


def _to_native(v):
    """Convert numpy scalars / arrays to plain Python for clean INCAR writing."""
    import numpy as np
    if isinstance(v, np.generic):
        return v.item()
    if isinstance(v, np.ndarray):
        return [_to_native(x) for x in v.tolist()]
    if isinstance(v, (list, tuple)):
        return [_to_native(x) for x in v]
    return v


def _pmg_set_to_ase_kwargs(input_set):
    """Translate a pymatgen ``VaspInputSet`` into ASE ``Vasp`` kwargs.

    The input set must be built with ``sort_structure=False`` so that any
    per-site ``MAGMOM`` stays aligned to the ASE atom order — ASE's
    ``set_magmom`` then re-sorts it into POSCAR (symbol) order itself.

    DFT+U is special-cased. pymatgen emits ``LDAUU``/``LDAUL``/``LDAUJ`` as
    *positional* lists aligned to its own POSCAR species order
    (``poscar.site_symbols``). ASE re-sorts the atoms and would write those
    lists verbatim, landing U on the wrong element. We instead emit ASE's
    element-keyed ``ldau_luj`` dict, which ASE re-orders to its own POSCAR — so
    U follows the species. (ASE rejects ``ldau_luj`` alongside the raw lists,
    so they are dropped.) ``MAGMOM`` needs no such treatment: it is per-atom and
    ASE's ``set_magmom`` already re-sorts it.
    """
    kwargs = {}
    incar = input_set.incar
    for k, v in incar.items():
        key = k.lower()
        if key in _DRIVER_KEYS:
            continue
        kwargs[key] = _to_native(v)

    # DFT+U: positional lists -> element-keyed ldau_luj (see docstring).
    if "LDAUU" in incar:
        luj = {}
        for sym, ll, uu, jj in zip(input_set.poscar.site_symbols,
                                   incar["LDAUL"], incar["LDAUU"], incar["LDAUJ"]):
            luj.setdefault(sym, {"L": int(ll), "U": float(uu), "J": float(jj)})
        for raw in ("ldauu", "ldaul", "ldauj"):
            kwargs.pop(raw, None)
        kwargs["ldau_luj"] = luj

    # KPOINTS -> explicit k-mesh + gamma flag.
    kp = getattr(input_set, "kpoints", None)
    if kp is not None and getattr(kp, "kpts", None):
        kwargs["kpts"] = [int(x) for x in kp.kpts[0]]
        kwargs["gamma"] = str(kp.style).lower().startswith("gamma")

    # POTCAR symbols -> per-element ASE `setups` suffix ('Fe_pv' -> {'Fe': '_pv'}).
    setups = {}
    for sym in getattr(input_set, "potcar_symbols", []):
        el, _, suf = sym.partition("_")
        if suf:
            setups[el] = "_" + suf
    if setups:
        kwargs["setups"] = setups
    return kwargs


def _omat24(atoms, set_cls_name, **set_kwargs):
    from pymatgen.io.ase import AseAtomsAdaptor
    from fairchem.data.omat.vasp import sets as omat_sets
    set_cls = getattr(omat_sets, set_cls_name)
    struct = AseAtomsAdaptor.get_structure(atoms)
    iset = set_cls(struct, sort_structure=False, **set_kwargs)
    return _pmg_set_to_ase_kwargs(iset)


def omat24_static(atoms):
    """OMat24 single-point (static) accuracy settings."""
    return _omat24(atoms, "OMat24StaticSet")


def omat24_relax(atoms):
    """OMat24 relaxation accuracy settings (ionic-driver tags stripped)."""
    return _omat24(atoms, "OMat24RelaxSet")


def cheap_omat(atoms):
    """OMat24 recipe tuned DOWN for a cheap first-pass saddle search.

    Two changes vs :func:`omat24_static`: the k-point reciprocal density drops
    from 64 to 16, and the POTCARs are lightened to ASE's ``minimal`` base
    (plain potentials for transition metals, mandatory semicore only for
    alkali/alkaline-earth) with soft ``_s`` O/C/N — so ENCUT can fall to ~300.
    Reconverge the resulting saddle with :func:`omat24_static`. Electronic and
    accuracy knobs (ENCUT, EDIFF, PREC, ISMEAR, SIGMA, ALGO) are intentionally
    left to the ``[Vasp]`` section.

    Note: there is no soft ``_s`` POTCAR for F (and a few other hard anions),
    so fluorine-bearing systems still need a higher ENCUT than 300.

    Lanthanides keep the f-in-core ``_3`` POTCARs (as OMat24StaticSet picks); the
    f-in-valence ones make the SCF non-convergent on partially-filled-4f systems.
    """
    kwargs = _omat24(atoms, "OMat24StaticSet",
                     user_kpoints_settings={"reciprocal_density": 16})
    setups = {"base": "minimal", "O": "_s", "C": "_s", "N": "_s"}
    # Preserve the f-in-core ``_3`` lanthanide POTCARs that OMat24StaticSet
    # selects (only for lanthanides actually present) — see _LANTH_FCORE.
    for _el in _LANTH_FCORE.intersection(atoms.get_chemical_symbols()):
        setups[_el] = "_3"
    kwargs["setups"] = setups
    return kwargs


def oc20(atoms):
    """OC20 slab/adslab VASP settings (RPBE, ENCUT 350, surface k-points)."""
    from fairchem.data.oc.utils.vasp_flags import VASP_FLAGS
    from fairchem.data.oc.utils.vasp import calculate_surface_k_points
    kwargs = {k: v for k, v in VASP_FLAGS.items() if k not in _DRIVER_KEYS}
    if "kpts" not in kwargs:
        kwargs["kpts"] = tuple(calculate_surface_k_points(atoms))
    kwargs.setdefault("setups", "minimal")
    return kwargs


def cheap_oc20(atoms):
    """OC20 recipe tuned DOWN for a cheap first-pass saddle search.

    Two changes vs :func:`oc20`, mirroring :func:`cheap_omat`: the surface
    k-point multiplier is halved from 40 to 20 (~4x fewer in-plane k-points),
    and the POTCARs get soft ``_s`` O/C/N on top of the ``minimal`` base OC20
    already uses — so ENCUT can fall to ~300 (set it in ``[Vasp]``; the
    generator keeps OC20's 350). Reconverge the resulting saddle with
    :func:`oc20`. All other electronic knobs are left to the ``[Vasp]`` section.

    Note: there is no soft ``_s`` POTCAR for F (and a few other hard anions),
    so fluorine-bearing systems still need a higher ENCUT than 300.
    """
    import numpy as np
    kwargs = oc20(atoms)
    # Same formula as fairchem's calculate_surface_k_points (inf-norm + round),
    # with the multiplier halved.
    cell = atoms.get_cell()
    a0 = np.linalg.norm(cell[0], ord=np.inf)
    b0 = np.linalg.norm(cell[1], ord=np.inf)
    kwargs["kpts"] = (max(1, int(round(20 / a0))), max(1, int(round(20 / b0))), 1)
    setups = {"base": "minimal", "O": "_s", "C": "_s", "N": "_s"}
    for _el in _LANTH_FCORE.intersection(atoms.get_chemical_symbols()):
        setups[_el] = "_3"
    kwargs["setups"] = setups
    return kwargs


#==============================================================================
### OC22 (exact Meta settings, frozen)
#
# The OC22 dataset (arXiv:2206.08917) was generated with the WhereWulff
# ``MOSurfaceSet`` (a ``MVLSlabSet``/MPRelaxSet subclass) from the
# ``OC22_dataset`` branch of Open-Catalyst-Project/Open-Catalyst-Dataset,
# rendered with 2022-era pymatgen (v2022.4.19). Today's pymatgen has drifted
# from what Meta actually ran (it now injects ENAUG=4000 and EDIFF=1e-5,
# changed the LDAU-applicability rule, and updated POTCAR mappings), so the
# effective settings are frozen here verbatim instead of being re-derived
# through pymatgen. Every value below was cross-checked against the paper's
# "Additional DFT settings" SI section.

# Materials Project GGA+U values (d-shell, J=0) — SI Table 18.
_OC22_U = {"Co": 3.32, "Cr": 3.7, "Fe": 5.3, "Mn": 3.9,
           "Mo": 4.38, "Ni": 6.2, "V": 3.25, "W": 6.2}

# MP element-default initial moments (2022 VASPIncarBase.yaml MAGMOM, sans
# oxidation-state variants); everything else 0.6. Implements the paper's
# "ferromagnetic or nonmagnetic per Horton et al." initialization.
_OC22_MAGMOM = {"Ce": 5, "Co": 0.6, "Cr": 5, "Eu": 10, "Fe": 5,
                "Mn": 5, "Mo": 5, "Ni": 5, "V": 5, "W": 5}

# 2022-era MPRelaxSet POTCAR table (suffixed entries only; unlisted elements
# use the bare-symbol POTCAR) with the OC22 W_pv -> W_sv fix (W_pv does not
# exist in the PBE 5.4 library Meta used).
_OC22_SETUPS = {
    "Ba": "_sv", "Be": "_sv", "Ca": "_sv", "Cr": "_pv", "Cs": "_sv",
    "Cu": "_pv", "Dy": "_3", "Er": "_3", "Fe": "_pv", "Ga": "_d",
    "Ge": "_d", "Hf": "_pv", "Ho": "_3", "In": "_d", "K": "_sv",
    "Li": "_sv", "Lu": "_3", "Mg": "_pv", "Mn": "_pv", "Mo": "_pv",
    "Na": "_pv", "Nb": "_pv", "Nd": "_3", "Ni": "_pv", "Os": "_pv",
    "Pb": "_d", "Pm": "_3", "Pr": "_3", "Rb": "_sv", "Re": "_pv",
    "Rh": "_pv", "Ru": "_pv", "Sc": "_sv", "Sm": "_3", "Sn": "_d",
    "Sr": "_sv", "Ta": "_pv", "Tb": "_3", "Tc": "_pv", "Ti": "_pv",
    "Tl": "_d", "Tm": "_3", "V": "_pv", "W": "_sv", "Y": "_sv",
    "Yb": "_2", "Zr": "_sv",
}


def _oc22_kwargs(atoms, k_product):
    import numpy as np
    symbols = set(atoms.get_chemical_symbols())
    kwargs = {
        "gga": "PE", "pp": "PBE", "xc": "PBE",
        "encut": 500.0,
        "ediff": 1e-4,
        "algo": "Fast",
        "prec": "Accurate",
        "ismear": 0,
        "sigma": 0.05,
        "ispin": 2,
        "isif": 0,
        "isym": 0,
        "symprec": 1e-10,
        "lreal": False,
        "lasph": True,
        "lorbit": 11,
        "nelm": 60,
        "nelmin": 8,
        "ncore": 4,
        "istart": 1,
        "lwave": True,
        "lvtot": True,
    }

    # Gamma-centered ceil(k_product/a) x ceil(k_product/b) x 1 surface mesh
    # ("non-integer values rounded up to the nearest integer" — the paper).
    abc = atoms.cell.lengths()
    kwargs["kpts"] = (int(np.ceil(k_product / abc[0])),
                      int(np.ceil(k_product / abc[1])), 1)
    kwargs["gamma"] = True

    # Hubbard U when the most electronegative element is O or F — only F beats
    # O on the Pauling scale, so that reduces to "O or F present" — and at
    # least one U-corrected metal is in the cell. Element-keyed ldau_luj so U
    # survives ASE's atom re-sort; ASE gives unlisted elements L=-1/U=0, which
    # is physically identical to the U=0 rows in Meta's INCARs.
    u_elts = symbols & set(_OC22_U)
    if symbols & {"O", "F"} and u_elts:
        kwargs["ldau"] = True
        kwargs["ldautype"] = 2
        kwargs["ldauprint"] = 1
        kwargs["ldau_luj"] = {el: {"L": 2, "U": _OC22_U[el], "J": 0.0}
                              for el in sorted(u_elts)}

    # LMAXMIX by d/f-block presence (2022 DictSet rule, applied regardless of U).
    numbers = atoms.get_atomic_numbers()
    if (numbers > 56).any():
        kwargs["lmaxmix"] = 6
    elif (numbers > 20).any():
        kwargs["lmaxmix"] = 4

    # MAGMOM: magmoms already on the atoms win (ASE writes them itself, like
    # pymatgen's site-magmom precedence); else MP element defaults.
    if not atoms.get_initial_magnetic_moments().any():
        kwargs["magmom"] = [_OC22_MAGMOM.get(s, 0.6)
                            for s in atoms.get_chemical_symbols()]

    # Dipole correction for adsorbate+slabs only (Meta's auto_dipole rule);
    # SaddleMill marks adsorbate atoms with tag==2 (the OC convention). DIPOL
    # is the mass-weighted center of mass in fractional coordinates.
    if (atoms.get_tags() == 2).any():
        com = np.average(atoms.get_scaled_positions(),
                         weights=atoms.get_masses(), axis=0)
        kwargs["ldipol"] = True
        kwargs["idipol"] = 3
        kwargs["dipol"] = [float(x) for x in com]

    kwargs["setups"] = {el: _OC22_SETUPS[el]
                        for el in sorted(symbols & set(_OC22_SETUPS))}
    return kwargs


def oc22(atoms):
    """OC22 oxide slab/adslab settings, exactly as Meta generated the dataset.

    PBE (GGA=PE) + Materials Project Hubbard U (element-keyed, only when O/F is
    the most electronegative element present), spin-polarized with MP-default
    initial moments (atoms' own magmoms win if set), ENCUT 500, EDIFF 1e-4,
    Gaussian smearing (ISMEAR 0, SIGMA 0.05), Gamma-centered
    ceil(30/a) x ceil(30/b) x 1 k-mesh, 2022-era MPRelaxSet POTCAR setups with
    the W_sv fix, and dipole correction iff the structure has tag==2 adsorbate
    atoms (Meta applied it to adsorbate+slabs only). Meta used the PBE 5.4
    POTCAR library — set ``pp_version = 54`` in ``[Vasp]`` if your
    ``$VASP_PP_PATH`` uses versioned ``potpaw_PBE.54`` folders.

    Meta's ionic-driver tags (IBRION=2, NSW=300, EDIFFG=-0.05) are stripped as
    usual — SaddleMill drives geometry through ASE optimizers; put ``ediffg``
    in ``[Vasp]`` for a VASP-internal run (e.g. the VTST dimer launcher).
    Note OC22 wrote WAVECAR/LOCPOT (ISTART=1, LWAVE/LVTOT=True) — override in
    ``[Vasp]`` if you don't want the disk traffic.
    """
    return _oc22_kwargs(atoms, k_product=30)


def cheap_oc22(atoms):
    """OC22 recipe tuned DOWN for a cheap first-pass saddle search.

    Two changes vs :func:`oc22`, mirroring :func:`cheap_omat`: the k-point
    product drops from 30 to 15 (~4x fewer in-plane k-points), and the POTCARs
    are lightened to ASE's ``minimal`` base with soft ``_s`` O/C/N — so ENCUT
    can fall to ~300 (set it in ``[Vasp]``; the generator keeps OC22's 500).
    The physics-defining settings (ISPIN=2, Hubbard U, PBE) are kept, so the
    cheap pass stays on the same PES — reconverge the resulting saddle with
    :func:`oc22`. All other electronic knobs are left to the ``[Vasp]``
    section.

    Lanthanides keep OC22's f-in-core POTCARs (``_3``, Yb ``_2``); the
    f-in-valence ones a bare ``minimal`` base would pick make the SCF
    non-convergent on partially-filled-4f systems. There is no soft ``_s``
    POTCAR for F, so fluorine-bearing systems still need a higher ENCUT.
    """
    kwargs = _oc22_kwargs(atoms, k_product=15)
    setups = {"base": "minimal", "O": "_s", "C": "_s", "N": "_s"}
    for el in set(atoms.get_chemical_symbols()):
        suf = _OC22_SETUPS.get(el)
        if suf in ("_2", "_3"):
            setups[el] = suf
    kwargs["setups"] = setups
    return kwargs


_BUILTINS = {
    "omat24_static": omat24_static,
    "omat24_relax": omat24_relax,
    "cheap_omat": cheap_omat,
    "oc20": oc20,
    "cheap_oc20": cheap_oc20,
    "oc22": oc22,
    "cheap_oc22": cheap_oc22,
}


def _import_callable(spec):
    """Resolve ``package.module:func`` or ``/path/to/file.py:func`` to a callable."""
    target, func_name = spec.rsplit(":", 1)
    if target.endswith(".py"):
        path = os.path.abspath(os.path.expanduser(os.path.expandvars(target)))
        if not os.path.isfile(path):
            raise FileNotFoundError(f"input file not found: {path}")
        mod_name = "_sm_vasp_dyn_" + str(abs(hash(path)))
        mod_spec = importlib.util.spec_from_file_location(mod_name, path)
        module = importlib.util.module_from_spec(mod_spec)
        mod_spec.loader.exec_module(module)
    else:
        module = importlib.import_module(target)
    try:
        return getattr(module, func_name)
    except AttributeError:
        raise AttributeError(f"{func_name!r} not found in {target!r}.")


def load_input_generator(spec):
    """Resolve an ``input_generator`` config value to a callable ``atoms -> dict``.

    ``spec`` is a built-in name, ``package.module:func``, or
    ``/path/to/file.py:func``. Raises a clear error if it cannot be resolved.
    """
    if callable(spec):
        return spec
    if spec in _BUILTINS:
        return _BUILTINS[spec]
    if ":" not in spec:
        raise ValueError(
            f"Unknown input_generator {spec!r}. Use a built-in "
            f"({', '.join(sorted(_BUILTINS))}), 'package.module:func', "
            f"or '/path/to/file.py:func'."
        )
    return _import_callable(spec)


#==============================================================================
### EXTRA INPUT FILES (written after ASE writes its inputs, e.g. VTST MODECAR)

def write_modecar(calc, atoms, directory):
    """Write a VTST ``MODECAR`` (initial dimer mode) from ``atoms.info['eigenmode']``.

    The eigenmode (in atoms order, with the usual ``orig_info`` fallback) is
    reshaped to ``(natoms, 3)``, reordered to POSCAR order via ``calc.sort``,
    normalized, and written one ``nx ny nz`` line per atom. No-op (with a
    warning) when no eigenmode is present, so a batch never hard-fails on it.
    """
    import numpy as np
    eig = atoms.info.get("eigenmode")
    if eig is None:
        eig = atoms.info.get("orig_info", {}).get("eigenmode")
    if eig is None:
        warnings.warn(
            "extra_input_files=modecar but atoms.info has no 'eigenmode'; "
            "skipping MODECAR (VTST will use its default initial mode).")
        return
    eig = np.asarray(eig, dtype=float).reshape(len(atoms), 3)
    sort = getattr(calc, "sort", None)
    if sort is not None:
        eig = eig[sort]                       # atoms order -> POSCAR (symbol) order
    norm = np.linalg.norm(eig)
    if norm > 0:
        eig = eig / norm
    with open(os.path.join(directory, "MODECAR"), "w") as f:
        for vx, vy, vz in eig:
            f.write(f"{vx:.16f} {vy:.16f} {vz:.16f}\n")


_EXTRA_FILE_BUILTINS = {
    "modecar": write_modecar,
}


def load_extra_input_writer(spec):
    """Resolve one ``extra_input_files`` value to a callable ``(calc, atoms, dir) -> None``.

    Same grammar as :func:`load_input_generator`: a built-in name (``modecar``),
    ``package.module:func``, or ``/path/to/file.py:func``.
    """
    if callable(spec):
        return spec
    if spec in _EXTRA_FILE_BUILTINS:
        return _EXTRA_FILE_BUILTINS[spec]
    if ":" not in spec:
        raise ValueError(
            f"Unknown extra_input_files writer {spec!r}. Use a built-in "
            f"({', '.join(sorted(_EXTRA_FILE_BUILTINS))}), 'package.module:func', "
            f"or '/path/to/file.py:func'."
        )
    return _import_callable(spec)


#==============================================================================
### EXTRA OUTPUTS (parsed from the VASP dir after the run, merged into .info)

def read_vtst_dimer(calc, atoms, directory):
    """Parse a finished VTST dimer run: ``eigenmode`` (NEWMODECAR) + ``curvature`` (DIMCAR).

    Returns a dict of ``.info`` keys to merge onto the output frame. The mode in
    ``NEWMODECAR`` is in POSCAR (symbol) order; it's mapped back to atoms order via
    ``calc.resort`` — the inverse of what :func:`write_modecar` does on the way in.
    Missing files are skipped, so this is safe to set even on non-dimer runs.
    """
    import numpy as np
    info = {}

    newmodecar = os.path.join(directory, "NEWMODECAR")
    if os.path.isfile(newmodecar):
        try:
            mode = np.loadtxt(newmodecar, dtype=float).reshape(len(atoms), 3)
            resort = getattr(calc, "resort", None)
            if resort is not None:
                mode = mode[resort]              # POSCAR order -> atoms order
            info["eigenmode"] = mode
        except (ValueError, OSError):
            pass

    # DIMCAR columns: Step Force Torque Energy Curvature Angle -> curvature is col 4.
    # On convergence VTST's Dimer_Fin writes a final row with '---' in the
    # Torque/Curvature/Angle columns (Step/Force/Energy stay numeric). Require BOTH
    # Force AND Curvature to parse, so that terminator row -- and any Fortran '*****'
    # overflow row -- is skipped and the curvature is taken from the last fully
    # numeric row above it (the real near-saddle curvature). float(parts[4]) inside
    # the guard means a non-numeric curvature can never reach the unguarded path.
    dimcar = os.path.join(directory, "DIMCAR")
    if os.path.isfile(dimcar):
        last_curv = None
        try:
            with open(dimcar) as f:
                for line in f:
                    parts = line.split()
                    if len(parts) >= 5:
                        try:
                            float(parts[1])              # Force (skips the header)
                            last_curv = float(parts[4])  # Curvature; skips '---'/overflow
                        except ValueError:
                            continue
            if last_curv is not None:
                info["curvature"] = last_curv
        except OSError:
            pass

    return info


_EXTRA_OUTPUT_BUILTINS = {
    "vtst_dimer": read_vtst_dimer,
}


def load_extra_output_parser(spec):
    """Resolve one ``extra_outputs`` value to a callable ``(calc, atoms, dir) -> dict``.

    Same grammar as :func:`load_input_generator`: a built-in name (``vtst_dimer``),
    ``package.module:func``, or ``/path/to/file.py:func``. The returned dict is
    merged into the output frame's ``.info``.
    """
    if callable(spec):
        return spec
    if spec in _EXTRA_OUTPUT_BUILTINS:
        return _EXTRA_OUTPUT_BUILTINS[spec]
    if ":" not in spec:
        raise ValueError(
            f"Unknown extra_outputs parser {spec!r}. Use a built-in "
            f"({', '.join(sorted(_EXTRA_OUTPUT_BUILTINS))}), 'package.module:func', "
            f"or '/path/to/file.py:func'."
        )
    return _import_callable(spec)
