#!/usr/bin/env python
"""Parity figure comparing two TS-guess sources on the same DFT dimer campaign.

Question answered
-----------------
Two guesses for the same transition state are each handed to an identical DFT
dimer search: a cheap *dataset* guess (e.g. an NNP-predicted saddle) and a
*midpoint* guess (linear interpolation between the two endpoints). For every
structure the campaign gives, per arm, how far the guess sat from the DFT
saddle it converged to (max per-atom displacement, MIC) and how many force
calls it took to get there. This script plots those two arms against each
other: which guess starts closer, and which costs less.

Layout: square parity panel (x = dataset arm, y = midpoint arm) with a marginal
histogram per arm, marker shape = convergence category, marker color =
force-call speedup, colorbar on the right.

Input
-----
One CSV, one row per matched structure, with these columns:

    mid_maxdisp  uma_maxdisp   max per-atom displacement guess -> DFT saddle (A)
    mid_steps    uma_steps     optimizer translation steps (reported only)
    mid_conv     uma_conv      1 if that arm reached the saddle, else 0
    mid_fcalls   uma_fcalls    force calls = VASP ionic steps

`mid_*` is the midpoint arm, `uma_*` the dataset arm (the column names are
historical; the arms are labeled "midpoint" and "dataset" on the figure).
Extra columns are ignored. Nothing about the campaign's splits or directory
layout is assumed: run it once per CSV.

Shared force-call budget
------------------------
The two arms are judged on the SAME budget (`--budget`, default 600 = the
midpoint run's NSW). A dataset guess counts as converged only if it reached
the saddle within that many force calls, and force calls are clipped to the
budget for both arms, so the reported speedup is like-for-like.

Axis modes (`--mode`)
---------------------
    logfix    log axes, no 0.01 clamp (lo follows the data), exact zeros
              floored at eps = 0.5*min-positive and drawn as crimson-edged
              stars; floored values included in the marginal histograms
    logbreak  logfix + eps tick labeled "0" + axis-break glyphs
    linear    linear axes, zeros at the origin, uniform 2 A axis cap (points
              beyond 2 A are intentionally outside the window, unannotated);
              star identified in its own legend box. p90 reference lines are
              drawn in this mode only.

Self-checks printed after saving: ZEROCHECK (every zero-displacement row lands
visibly inside the panel), AXCHECK (axis-window accounting), P90CHECK (drawn
p90 lines equal the CSV-computed percentiles), DASHCHECK (no em/en dashes in
rendered text). All must print OK before a figure is used.

Usage
-----
    python scripts/plot_guess_parity.py --csv matched.csv --out parity --mode linear
"""
import argparse, csv, sys
from collections import Counter

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.transforms as mtr
from matplotlib.lines import Line2D
from matplotlib.text import Text


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", required=True,
                    help="Matched CSV, one row per structure (columns above).")
    ap.add_argument("--out", required=True,
                    help="Output path prefix; writes <out>.pdf (and <out>.png with --png).")
    ap.add_argument("--mode", choices=["logfix", "logbreak", "linear"], default="logfix",
                    help="Axis mode (default: logfix).")
    ap.add_argument("--clip", type=float, default=10.0,
                    help="Color-range half-width for the force-call ratio, "
                         "i.e. colors span [1/CLIP, CLIP] (default: 10).")
    ap.add_argument("--budget", type=int, default=600,
                    help="Shared force-call budget, normally the midpoint run's "
                         "NSW (default: 600).")
    ap.add_argument("--png", action="store_true",
                    help="Also write <out>.png alongside the PDF.")
    args = ap.parse_args()
    OUT, MODE, CLIP, BUDGET = args.out, args.mode, args.clip, args.budget

    rows = list(csv.DictReader(open(args.csv)))
    if not rows:
        sys.exit(f"no rows in {args.csv}")
    mx = np.array([float(r["mid_maxdisp"]) for r in rows])
    my = np.array([float(r["uma_maxdisp"]) for r in rows])
    ms = np.array([int(r["mid_steps"]) for r in rows], float)
    us = np.array([int(r["uma_steps"]) for r in rows], float)
    mc = np.array([int(r["mid_conv"]) for r in rows], bool)
    uc = np.array([int(r["uma_conv"]) for r in rows], bool)
    mf = np.array([float(r["mid_fcalls"]) for r in rows])   # force calls = VASP ionic steps
    uf = np.array([float(r["uma_fcalls"]) for r in rows])

    # Judge both guesses on the SAME force-call budget = midpoint's NSW (600). A dataset guess
    # counts as converged only if it reached the saddle within 600 force calls; force calls are
    # clipped at 600 for both runs so the speedup is a like-for-like comparison.
    uc_raw = uc.copy()
    uc = uc & (uf <= BUDGET)            # redefined dataset convergence: saddle found within budget
    mfc = np.minimum(mf, BUDGET)
    ufc = np.minimum(uf, BUDGET)

    # convergence category -> marker
    cat = np.where(mc & uc, "both", np.where(uc & ~mc, "uma_only",
            np.where(mc & ~uc, "mid_only", "neither")))
    MARK = {"both": "o", "uma_only": "^", "mid_only": "v", "neither": "x"}
    LBL = {"both": "both converged", "uma_only": "dataset only",
           "mid_only": "midpoint only", "neither": "neither"}

    # color = force-call speedup of dataset(uma) over midpoint (both <=budget), clipped [1/CLIP,CLIP]
    ratio = np.clip(mfc / ufc, 1/CLIP, CLIP)
    c = np.log10(ratio)
    vlim = np.log10(CLIP)
    cmap = plt.get_cmap("RdBu")      # high (midpoint spent more force calls) -> blue, low -> red
    norm = plt.Normalize(vmin=-vlim, vmax=vlim)

    # --- converged-at-start (maxdisp == 0) class + mode-dependent geometry ---
    zx = my == 0.0                       # x axis = dataset(uma) arm exact zeros
    zy = mx == 0.0                       # y axis = midpoint arm exact zeros
    zrow = zx | zy                       # rows rendered as stars
    allv = np.concatenate([mx, my])
    minpos = allv[allv > 0].min()
    if MODE in ("logfix", "logbreak"):
        eps = 0.5 * minpos               # floor for exact zeros, below the real data
        lo = 0.8 * eps                   # no 0.01 clamp: axis follows the data
        hi = 1.1 * allv.max()
        X = np.where(zx, eps, my)        # plotted coords (zeros floored)
        Y = np.where(zy, eps, mx)
    else:                                # linear
        eps = 0.0
        hi = 2.0                         # uniform cap, all subsets (lead's spec); points
        lo = -0.03 * hi                  # beyond 2 A intentionally clipped, unannotated
        X, Y = my, mx                    # zeros plot naturally at the origin

    # manual layout: square main panel + marginal histograms (top = x/dataset,
    # right = y/midpoint) + colorbar outside the right histogram. Square figure +
    # equal-fraction main box keeps the panel square without set_aspect (which
    # fights shared axes).
    fig = plt.figure(figsize=(7.0, 7.0))
    ax  = fig.add_axes([0.10, 0.09, 0.60, 0.60])                 # main parity panel
    axt = fig.add_axes([0.10, 0.705, 0.60, 0.15], sharex=ax)     # top histogram (x)
    axr = fig.add_axes([0.715, 0.09, 0.15, 0.60], sharey=ax)     # right histogram (y)
    cax = fig.add_axes([0.93, 0.09, 0.025, 0.60])                # colorbar axis
    for k, mk in MARK.items():
        sel = (cat == k) & ~zrow          # star rows drawn separately below
        if not sel.any():
            continue
        if k == "neither":               # ratio == budget/budget == 1 by construction -> plain black
            ax.scatter(X[sel], Y[sel], marker=mk, s=24, c="k",
                       linewidths=0.4, alpha=0.9)
            continue
        face = cmap(norm(c[sel]))         # alpha on the FACE only -> black edges stay crisp
        face[:, 3] = 0.78
        ax.scatter(X[sel], Y[sel], marker=mk, s=24, facecolors=face,
                   edgecolors="k", linewidths=0.3)

    # converged-at-start rows as high-visibility stars -- distinct shape AND
    # crimson edge (never color alone); face keeps the ratio-color convention
    star = None
    if zrow.any():
        sface = cmap(norm(c[zrow]))
        star = ax.scatter(X[zrow], Y[zrow], marker="*", s=170, facecolors=sface,
                          edgecolors="crimson", linewidths=1.2, zorder=5)

    sc = plt.cm.ScalarMappable(norm=norm, cmap=cmap)   # mappable for the colorbar
    sc.set_array([])

    # lo/hi/allv computed above (mode-dependent); parity line from 0 on linear
    p0 = lo if MODE != "linear" else 0.0
    ax.plot([p0, hi], [p0, hi], "k--", lw=1, zorder=0)
    if MODE != "linear":
        ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel(r"max per-atom displacement, dataset $\to$ DFT saddle ($\AA$)")
    ax.set_ylabel(r"max per-atom displacement, midpoint $\to$ DFT saddle ($\AA$)")

    # marginal histograms spanning the shared axis limits; floored coords so the
    # converged-at-start rows land in the first bin instead of vanishing
    if MODE != "linear":
        bins = np.logspace(np.log10(lo), np.log10(hi), 41)
    else:
        bins = np.linspace(0.0, hi, 41)
    hstyle = dict(bins=bins, histtype="bar", color="0.80",
                  edgecolor="0.35", linewidth=0.6)
    axt.hist(X, **hstyle)                              # x of main panel = dataset
    axr.hist(Y, orientation="horizontal", **hstyle)    # y of main panel = midpoint
    axt.tick_params(labelbottom=False)                 # shared edges: no labels
    axr.tick_params(labelleft=False)
    axt.tick_params(axis="y", labelsize=7)
    axr.tick_params(axis="x", labelsize=7)
    for a in (axt, axr):
        a.spines["top"].set_visible(False)
        a.spines["right"].set_visible(False)

    # mean/median markers, reference style: solid violet = median, black dashed = mean
    PUR = "darkviolet"
    xmn, xmd = my.mean(), np.median(my)     # x data = dataset (arithmetic, not log-space)
    ymn, ymd = mx.mean(), np.median(mx)     # y data = midpoint
    axt.axvline(xmd, color=PUR, lw=2.0, zorder=3)
    axt.axvline(xmn, color="k", ls="--", lw=1.6, zorder=3)
    axr.axhline(ymd, color=PUR, lw=2.0, zorder=3)
    axr.axhline(ymn, color="k", ls="--", lw=1.6, zorder=3)
    bt = mtr.blended_transform_factory(axt.transData, axt.transAxes)   # labels above top hist
    axt.text(xmn, 1.06, rf"mean = {xmn:.3f} $\AA$", color="k", ha="center",
             va="bottom", fontsize=8, transform=bt, clip_on=False)
    axt.text(xmd, 1.30, rf"median = {xmd:.3f} $\AA$", color=PUR, ha="center",
             va="bottom", fontsize=8, transform=bt, clip_on=False)
    br = mtr.blended_transform_factory(axr.transAxes, axr.transData)   # rotated, right of right hist
    axr.text(1.10, ymn, rf"mean = {ymn:.3f} $\AA$", color="k", ha="center",
             va="center", fontsize=8, rotation=270, transform=br, clip_on=False)
    axr.text(1.28, ymd, rf"median = {ymd:.3f} $\AA$", color=PUR, ha="center",
             va="center", fontsize=8, rotation=270, transform=br, clip_on=False)

    # p90 reference lines (CONVERGED-ONLY population per arm, zeros included),
    # linear mode only; matches the rebuttal table's convention.
    # Marginal histograms ONLY (lead's spec, no lines across the scatter panel).
    # Arm is encoded by which marginal carries the line: top hist = dataset(uma, x),
    # right hist = midpoint(mid, y). Dash-dot goldenrod so they read as reference
    # lines, never as the black dashed mean or the violet median.
    GLD = "darkgoldenrod"
    DDOT = (0, (3, 1, 1, 1))
    p90_lines, p90_texts = [], []
    if MODE == "linear":
        p90u = float(np.percentile(my[uc_raw], 90))   # raw conv flag: the table's population, NOT the within-budget uc (they diverge once continuation rows carry fcalls > budget)
        p90m = float(np.percentile(mx[mc], 90))
        p90_lines.append(axt.axvline(p90u, color=GLD, ls=DDOT, lw=1.4, zorder=3))
        p90_lines.append(axr.axhline(p90m, color=GLD, ls=DDOT, lw=1.4, zorder=3))
        # x-arm label: third tier above the top hist (mean 1.06, median 1.30, p90 1.54)
        p90_texts.append(axt.text(p90u, 1.54, rf"p90 = {p90u:.3f} $\AA$", color=GLD,
                                  ha="center", va="bottom", fontsize=8, transform=bt,
                                  clip_on=False))
        # y-arm label: INSIDE the right hist, rotated, just above its line in the
        # sparse tail region (the outer tier at 1.46 would collide with the colorbar)
        p90_texts.append(axr.text(0.90, p90m + 0.17, rf"p90 = {p90m:.3f} $\AA$",
                                  color=GLD, ha="center", va="center", fontsize=8,
                                  rotation=270, transform=br))

    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)   # after hists: they can autoscale shared axes

    # annotations: converged-at-start count (log modes only -- the linear legend
    # carries the star count); "0" break (logbreak)
    if MODE != "linear":
        ax.text(0.02, 0.975,
                rf"$\bigstar$ converged at start (maxdisp = 0 $\AA$): "
                rf"dataset {int(zx.sum())}, midpoint {int(zy.sum())}",
                transform=ax.transAxes, ha="left", va="top", fontsize=8, color="crimson")
    if MODE == "logbreak":
        bt0 = mtr.blended_transform_factory(ax.transData, ax.transAxes)
        br0 = mtr.blended_transform_factory(ax.transAxes, ax.transData)
        ax.text(eps, -0.012, "0", transform=bt0, ha="center", va="top", fontsize=9)
        ax.text(-0.012, eps, "0", transform=br0, ha="right", va="center", fontsize=9)
        g = (eps * minpos) ** 0.5        # break glyphs between the zero bin and real data
        ax.plot([g / 1.07, g * 1.07], [0, 0], transform=bt0, ls="", marker=(2, 0, -65),
                ms=8, color="k", mew=1.2, clip_on=False, zorder=6)
        ax.plot([0, 0], [g / 1.07, g * 1.07], transform=br0, ls="", marker=(2, 0, 65),
                ms=8, color="k", mew=1.2, clip_on=False, zorder=6)

    # colorbar with ratio ticks
    cb = fig.colorbar(sc, cax=cax)
    ticks = np.log10([1/CLIP, 1/2, 1, 2, CLIP])
    cb.set_ticks(ticks); cb.set_ticklabels([f"1/{CLIP:g}", "1/2", "1", "2", f"{CLIP:g}"])
    cb.set_label(f"force-call speedup (midpoint / dataset), capped at {BUDGET}")

    # legend with per-category counts (the caption's inset)
    cnt = Counter(cat)
    handles = [Line2D([0], [0], marker=MARK[k], color=("k" if k == "neither" else "0.3"),
                      ls="", mec="k", ms=7, label=f"{LBL[k]} ({cnt.get(k,0)})") for k in MARK]
    above = int((my < mx).sum())
    star_h = None
    if zrow.any():                       # star legend entry. Linear mode: the star gets
        # its OWN small legend box (created after the category legend below), with no
        # wording tying it to the category counts; log modes keep the in-legend entry
        slbl = (rf"converged at start (0 $\AA$)" if MODE == "linear"
                else f"converged at start ({int(zrow.sum())})")
        star_h = Line2D([0], [0], marker="*", color="0.3", ls="", mec="crimson",
                        mew=1.2, ms=11, label=slbl)
        if MODE != "linear":
            handles.append(star_h)
    # raw per-arm convergence rates from the plotted CSV (uc_raw/mc = CSV flags)
    convline = (f"dataset-guess conv: {100*uc_raw.mean():.1f}%   "
                f"midpoint conv: {100*mc.mean():.1f}%")
    leg = ax.legend(handles=handles, loc="lower right", bbox_to_anchor=(1, 0.055),
                    fontsize=8, framealpha=0.9,   # lifted off the bottom zero/eps band
                    title=f"N={len(rows)}   above y=x: {above} ({100*above/len(rows):.0f}%)"
                          f"\n{convline}", title_fontsize=8)
    leg2 = None
    if MODE == "linear" and star_h is not None:   # separate star-identification box,
        ax.add_artist(leg)                        # bottom-right corner (empty in all subsets)
        leg2 = ax.legend(handles=[star_h], loc="lower right", bbox_to_anchor=(1, 0),
                         fontsize=8, framealpha=0.9)

    exts = ["pdf"] + (["png"] if args.png else [])   # the PDF is the deliverable
    for ext in exts:
        fig.savefig(f"{OUT}.{ext}", dpi=150, bbox_inches="tight")

    # sanity check: every zero-disp row must land visibly inside the main panel
    fig.canvas.draw()
    nvis = 0
    if star is not None:
        pts = np.asarray(ax.transData.transform(star.get_offsets()))
        bb = ax.get_window_extent()
        nvis = int(sum(bb.x0 <= px <= bb.x1 and bb.y0 <= py <= bb.y1 for px, py in pts))
    exp = int(zrow.sum())
    nocc = 0
    if star is not None:                 # stars hidden under a legend frame OR a p90 label
        for L in [leg] + ([leg2] if leg2 is not None else []) + p90_texts:
            lb = L.get_window_extent()
            nocc += int(sum(lb.x0 <= px <= lb.x1 and lb.y0 <= py <= lb.y1 for px, py in pts))
    print(f"ZEROCHECK mode={MODE} expected={exp} visible={nvis} legend-occluded={nocc} "
          f"{'OK' if nvis == exp and nocc == 0 else 'FAIL'}")
    print(f"CONV dataset={100*uc_raw.mean():.1f}% ({int(uc_raw.sum())}/{len(rows)})  "
          f"midpoint={100*mc.mean():.1f}% ({int(mc.sum())}/{len(rows)})")
    # star rows vs convergence category (are stars a subset of the counts above?)
    print("STARCAT", dict(Counter(cat[zrow])) if zrow.any() else "{}")
    # axis-window accounting. Linear caps at 2 A by design (outside counts reported,
    # never annotated on-figure); all star rows must sit inside the window
    xlo, xhi = ax.get_xlim(); ylo, yhi = ax.get_ylim()
    in_row = (X >= xlo) & (X <= xhi) & (Y >= ylo) & (Y <= yhi)
    sin_, sexp = int((in_row & zrow).sum()), int(zrow.sum())
    print(f"AXCHECK mode={MODE} xlim=({xlo:.4g},{xhi:.4g}) ylim=({ylo:.4g},{yhi:.4g}) "
          f"rows-in-window={int(in_row.sum())}/{len(rows)} "
          f"outside: dataset={int((my > xhi).sum())} midpoint={int((mx > yhi).sum())} "
          f"stars-in={sin_}/{sexp} {'OK' if sin_ == sexp else 'FAIL'}")
    # drawn p90 line positions must equal the CSV-computed percentiles
    # (converged-only per arm, same convention as the rebuttal table)
    if p90_lines:
        d_u = float(p90_lines[0].get_xdata()[0])
        d_m = float(p90_lines[1].get_ydata()[0])
        r_u = float(np.percentile(np.array([float(r["uma_maxdisp"]) for r in rows
                                            if r["uma_conv"] == "1"]), 90))
        r_m = float(np.percentile(np.array([float(r["mid_maxdisp"]) for r in rows
                                            if r["mid_conv"] == "1"]), 90))
        print(f"P90CHECK drawn uma={d_u:.6f} mid={d_m:.6f} recomputed uma={r_u:.6f} "
              f"mid={r_m:.6f} {'OK' if d_u == r_u and d_m == r_m else 'FAIL'}")

    # no em/en dashes in any rendered text (lead's spec for the linear figures)
    bad = [t.get_text() for t in fig.findobj(Text)
           if "—" in t.get_text() or "–" in t.get_text()]
    print(f"DASHCHECK mode={MODE} texts-with-em/en-dash={len(bad)} "
          f"{'OK' if not bad else 'FAIL ' + repr(bad)}")
    print("LEGEND-MAIN title:", repr(leg.get_title().get_text()))
    print("LEGEND-MAIN entries:", [t.get_text() for t in leg.get_texts()])
    if leg2 is not None:
        print("LEGEND-STAR entries:", [t.get_text() for t in leg2.get_texts()])

    # console summary
    print(f"N matched = {len(rows)}")
    print("categories:", dict(cnt))
    print(f"median maxdisp  midpoint={np.median(mx):.3f}  dataset={np.median(my):.3f} A")
    print(f"points with dataset closer to DFT saddle (x<y): {above}/{len(rows)} "
          f"({100*above/len(rows):.1f}%)")
    print(f"median translation steps  midpoint={np.median(ms):.0f}  dataset={np.median(us):.0f}")
    print(f"median force calls (<={BUDGET})  midpoint={np.median(mfc):.0f}  dataset={np.median(ufc):.0f}")
    print(f"median force-call ratio (mid/uma, <={BUDGET}) = {np.median(mfc/ufc):.2f}")
    print(f"dataset converged: raw={int(uc_raw.sum())} -> within-{BUDGET} budget={int(uc.sum())} "
          f"(reclassified {int((uc_raw & ~uc).sum())} that needed >{BUDGET} fc)")
    print("saved", " / ".join(f"{OUT}.{e}" for e in exts))


if __name__ == "__main__":
    main()
