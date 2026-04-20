"""Extract (reactant, saddle, product) triplets from NEB trajectories.

Input
-----
A directory ``NEB_trajes/`` containing one or more ASE trajectory files
produced by a batched NEB job. Each traj file stores several bands
concatenated. Every image must carry::

    atoms.info['src_index']  # band id (contiguous blocks per file)
    atoms.info['nimages']    # number of images in this band

Exactly one image per band must also carry::

    atoms.info['barrier']    # climbing-image barrier height
    atoms.info['eigenmode']  # CI eigenmode

``task_name`` is assumed to be set upstream; this script does not touch it.

Output
------
A sibling directory ``NEB_trajes_extracted/`` (created next to
``NEB_trajes/``). For each input ``<name>.traj`` a file
``<name>_extracted.traj`` is written. Each output has length
``3 * n_bands_in_input`` and is ordered::

    reactant_0, saddle_0, product_0, reactant_1, saddle_1, product_1, ...

where "reactant" is the first image of a band, "product" the last, and
"saddle" the CI image. A band's length may differ from any other band's
length; the script uses ``nimages`` to slice correctly.

Usage
-----
    python extract_neb_triplets.py                           # uses ./NEB_trajes
    python extract_neb_triplets.py <path/to/NEB_trajes>
    python extract_neb_triplets.py <path/to/NEB_trajes> --pattern "*.traj"
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ase.atoms import Atoms
from ase.io import read
from ase.io.trajectory import Trajectory

try:
    from tqdm import tqdm
except ImportError:  # graceful fallback -- plain iterator + print-based write
    class _FakeTqdm:
        def __init__(self, it=None, **kw):
            self._it = it

        def __iter__(self):
            return iter(self._it) if self._it is not None else iter(())

        def set_postfix_str(self, *a, **kw):
            pass

        @staticmethod
        def write(msg, *a, **kw):
            print(msg)

    def tqdm(it=None, **kw):
        return _FakeTqdm(it, **kw)
    tqdm.write = _FakeTqdm.write


def group_bands(frames):
    """Group a flat frame list into consecutive bands by src_index.

    Returns a list of (src_index, [frames_in_band]).

    Raises RuntimeError if:
      - any frame is missing 'src_index' or 'nimages'
      - the same src_index appears in non-consecutive positions (i.e.
        the traj interleaves bands, which is not allowed)
      - a band's actual length doesn't match its 'nimages'
    """
    bands = []
    seen_src = set()
    current_src = None
    current_band = []
    for k, a in enumerate(frames):
        if 'src_index' not in a.info:
            raise RuntimeError(f"frame {k}: missing atoms.info['src_index']")
        if 'nimages' not in a.info:
            raise RuntimeError(f"frame {k}: missing atoms.info['nimages']")
        si = a.info['src_index']
        if si != current_src:
            if current_band:
                bands.append((current_src, current_band))
            if si in seen_src:
                raise RuntimeError(
                    f"src_index {si} appears in non-consecutive frames; "
                    f"bands must be contiguous in the input traj")
            seen_src.add(si)
            current_src = si
            current_band = [a]
        else:
            current_band.append(a)
    if current_band:
        bands.append((current_src, current_band))

    for si, chunk in bands:
        expected = chunk[0].info['nimages']
        if len(chunk) != expected:
            raise RuntimeError(
                f"band src_index={si}: nimages={expected} but chunk "
                f"has {len(chunk)} frames")
    return bands


def extract_triplet(chunk):
    """Return (reactant, saddle, product) from a band's image list."""
    reactant = chunk[0]
    product = chunk[-1]
    ci_candidates = [a for a in chunk if 'barrier' in a.info]
    if len(ci_candidates) != 1:
        raise RuntimeError(
            f"expected exactly one image with 'barrier' per band, "
            f"got {len(ci_candidates)} "
            f"(src_index={chunk[0].info.get('src_index')})")
    saddle = ci_candidates[0]
    if 'eigenmode' not in saddle.info:
        raise RuntimeError(
            f"saddle (src_index={saddle.info.get('src_index')}) "
            f"has 'barrier' but no 'eigenmode'")
    return reactant, saddle, product


def process_file(in_path: Path, out_path: Path):
    frames = read(str(in_path), index=':')
    bands = group_bands(frames)
    with Trajectory(str(out_path), 'w') as tr:
        for si, chunk in bands:
            r, s, p = extract_triplet(chunk)
            tr.write(r)
            tr.write(s)
            tr.write(p)
    return len(bands)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('neb_trajes_dir', nargs='?', default='NEB_trajes',
                    help='Path to the NEB_trajes/ directory '
                         '(default: NEB_trajes).')
    ap.add_argument('--pattern', default='*.traj',
                    help='Glob for input files inside NEB_trajes (default: *.traj).')
    args = ap.parse_args()

    in_dir = Path(args.neb_trajes_dir).resolve()
    if not in_dir.is_dir():
        raise SystemExit(f"not a directory: {in_dir}")
    if in_dir.name != 'NEB_trajes':
        print(f"[warn] input directory is named '{in_dir.name}', "
              f"not 'NEB_trajes' -- continuing anyway.",
              file=sys.stderr)

    out_dir = in_dir.parent / 'NEB_trajes_extracted'
    out_dir.mkdir(parents=True, exist_ok=True)

    inputs = sorted(in_dir.glob(args.pattern))
    if not inputs:
        raise SystemExit(f"no files matched {in_dir}/{args.pattern}")

    pbar = tqdm(inputs, desc='extract', unit='file', leave=True)
    for in_path in pbar:
        if hasattr(pbar, 'set_postfix_str'):
            pbar.set_postfix_str(in_path.name)
        out_name = in_path.stem + '_extracted' + in_path.suffix
        out_path = out_dir / out_name
        n_bands = process_file(in_path, out_path)
        tqdm.write(f"[wrote] {out_path} (n_bands={n_bands}, frames={3*n_bands})")


if __name__ == '__main__':
    main()
