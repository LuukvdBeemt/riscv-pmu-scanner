#!/usr/bin/env python3
"""
figs_anomaly.py -- the single-panel figures for the anomaly-triage section of
the evaluation.

Each figure is emitted on its own so the LaTeX side can place them
independently or compose them into a multi-panel float with subfigure.

    fig_mem_mechanism.png   replaces  tab:mem-mechanism
    fig_triage_novelty.png  complements  tab:triage-tiers

fig_storefp_map() is defined below but is NOT called by main().  It was cut
from the thesis: its content is field value against field value, which is
inherently tabular, and tab:store-fp-reserved carries it with less ink.  It
is kept because it works as a talk slide, where a spoken walkthrough suits a
coloured grid better than a caption does.  Call it explicitly if wanted.

Numbers come from the same code path as the tables: signature_classify.load()
for the deviation frame and the documented/active masks, triage.triage() for
the tier assignment.  Nothing here recomputes a statistic independently, so a
figure and its table cannot drift apart.

Usage:
    python3 figs_anomaly.py --scan consensus_scan.csv \
            --screen out/binutils_screen.csv --outdir out
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

import signature_classify as SC
import triage as TR


# ------------------------------------------------------------------ style --
# Same try/except shape the other scripts use: prefer the shared thesis style
# module, fall back to a self-contained approximation so the script still runs
# outside the thesis tree.
def _style():
    try:
        from kernel import apply_figure_style
        apply_figure_style(sizes=(8, 7.5, 7))
    except Exception:
        mpl.rcParams.update({
            "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8.5,
            "legend.fontsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7,
            "figure.dpi": 200, "savefig.dpi": 200,
        })


def _frame(ax, keep=("left", "bottom")):
    try:
        from kernel import set_frame
        set_frame(ax, keep)
        return
    except Exception:
        pass
    for s, sp in ax.spines.items():
        sp.set_visible(s in keep)


def _save(fig, path):
    try:
        from kernel import savefig
        savefig(fig, path)
    except Exception:
        fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print("wrote", path)
    return path


# Fallback palette, used only when kernel.py is absent.  The four values must
# stay mutually distinguishable in print: tier C and tier D are the pair most
# at risk of collapsing into "two greys", so tier C carries a hue.
_TIER_FALLBACK = {"A": "#c2452d", "B": "#1f6fb4", "C": "#e0a030", "D": "#b8bec6"}


def _tier_colors():
    try:
        from kernel import TRIAGE_COLORS as K
    except Exception:
        K = {}
    pick = lambda k: K.get(k, _TIER_FALLBACK[k])
    return {TR.TIER_A: pick("A"), TR.TIER_B: pick("B"),
            TR.TIER_C: pick("C"), TR.TIER_D: pick("D")}


def _palette():
    try:
        from kernel import COLORS
        return dict(COLORS)
    except Exception:
        return {}


TIER_SHORT = {TR.TIER_A: "A", TR.TIER_B: "B", TR.TIER_C: "C", TR.TIER_D: "D"}
TIER_LABEL = {TR.TIER_A: "A: breaks an invariant",
              TR.TIER_B: "B: no twin, out of range",
              TR.TIER_C: "C: no twin, in range",
              TR.TIER_D: "D: exact documented twin"}


# ------------------------------------------------------------------- data --
def build(scan_csv, screen_csv):
    """Load the scan and attach the fields the three figures need."""
    df, ev, Dev, _ = SC.load(scan_csv, screen_csv, "C")
    d = df[["enc", "encoding", "opclass", "mnemonic"]].copy()
    documented = df.documented.values.astype(bool)
    active = df.active.values.astype(bool)
    T, inv = TR.triage(Dev, d, documented, active, ev, subtype=None, min_ref=8)

    enc = df.enc.values
    df = df.assign(width=(enc >> 12) & 7, mop2=(enc >> 26) & 3,
                   bit28=(enc >> 28) & 1, nf=(enc >> 29) & 7,
                   vm=(enc >> 25) & 1)
    df["tier"] = T.tier.reindex(df.index)
    return df, ev, Dev, documented, active, T, inv


def memory_subtype(row):
    """Direction x addressing mode for documented memory encodings.

    Derived from the mnemonic for direction and from the addressing-mode field
    for the mode, so the two are not read off the same source.
    """
    m = row.mnemonic
    if not isinstance(m, str):
        return None
    if row.opclass == "LOAD":
        return "scalar-load"
    if row.opclass == "STORE":
        return "scalar-store"
    if row.opclass not in ("LOAD-FP", "STORE-FP"):
        return None
    if m.startswith("fl"):
        return "scalar-load"
    if m.startswith("fs"):
        return "scalar-store"
    if not m.startswith("th.v"):
        return None
    direction = "vload" if m.startswith("th.vl") else "vstore"
    mode = {0: "unit", 4: "unit", 2: "strided", 6: "strided",
            3: "indexed", 7: "indexed"}.get((row.enc >> 26) & 7)
    if mode is None:
        return None
    seg = "-seg" if "seg" in m else ""
    return f"{direction}-{mode}{seg}"


# -------------------------------------------------------- fig 1: mechanism --
ORDER = ["scalar-load", "scalar-store", "vload-unit", "vstore-unit",
         "vload-strided", "vstore-strided", "vload-indexed", "vstore-indexed"]


def fig_mem_mechanism(df, Dev, documented, active, outdir):
    """0x0016 and 0x0016 - 0x0014, per documented memory sub-type.

    Deliberately unnamed.  The candidate encoding is squashed and never
    commits, and instret reads zero for all 28,672 encodings, so no counter
    here can be read as a retirement or commit count; calling the open marker
    "retired" or the filled one "issued" would assert a mechanism this scan
    cannot see.  The figure shows two counter values and their difference.

    Positions, not bar lengths, carry the argument, so the log axis is honest.
    Within the scalar pair and within the unit-stride pair the filled markers
    sit at the same x for load and store, so the whole of the store's higher
    0x0016 reading is 0x0014.  The strided and indexed pairs at the bottom
    visibly fail to line up, which is the hedge the text otherwise states.
    """
    _style()
    sub = df.apply(memory_subtype, axis=1)
    m = documented & active & sub.notna().values
    G = pd.DataFrame(dict(sub=sub[m], retired=Dev.loc[m, "0x0016"],
                          replay=Dev.loc[m, "0x0014"]))
    g = G.groupby("sub").agg(n=("retired", "size"), retired=("retired", "median"),
                             replay=("replay", "median")).reindex(ORDER)
    g["diff"] = g.retired - g.replay

    pal = _palette()
    c_issued = pal.get("blue", "#2f6fb0")
    c_replay = pal.get("red", "#c0392b")

    fig, ax = plt.subplots(figsize=(3.4, 2.9))
    ax.set_xscale("log")
    y = np.arange(len(g))

    # Guide lines joining the two pairs whose issued counts coincide.
    for lo in (0, 2):
        v = float(g["diff"].iloc[lo])
        ax.plot([v, v], [lo - .32, lo + 1.32], color="#444444", lw=.8,
                ls=(0, (2, 2)), zorder=2)
        ax.text(v * .92, lo + .5, f"{int(v)}", fontsize=6.6, color="#444444",
                ha="right", va="center", zorder=4)

    for i, s_ in enumerate(ORDER):
        iss, ret = float(g["diff"][s_]), float(g.retired[s_])
        if ret > iss:
            ax.plot([iss, ret], [i, i], color=c_replay, lw=2.2, zorder=3,
                    solid_capstyle="butt")
        ax.plot(ret, i, "o", ms=5, mfc="white", mec=c_issued, mew=1.1, zorder=4)
        ax.plot(iss, i, "o", ms=5, color=c_issued, zorder=5)

    ax.set_yticks(y)
    ax.set_yticklabels([f"{s_}  (n={int(g.n[s_])})" for s_ in ORDER])
    ax.set_ylim(len(g) - .4, -.6)
    ax.set_xlim(2, 400)
    ax.set_xticks([3, 10, 30, 100, 300])
    ax.set_xticklabels(["3", "10", "30", "100", "300"])
    ax.set_xlabel("counter value, median per sub-type")
    ax.axhline(3.5, color="#cccccc", lw=.7, zorder=1)
    ax.grid(axis="x", color="#eeeeee", lw=.6, zorder=0)
    # See the note in fig_triage_novelty: log minor ticks read as stray dashes
    # at kernel.py's tick width, and the labelled decades carry the axis.
    ax.tick_params(axis="x", which="minor", length=0)

    handles = [plt.Line2D([], [], marker="o", ls="", ms=5, color=c_issued,
                          label=r"$\mathtt{0x0016}-\mathtt{0x0014}$"),
               plt.Line2D([], [], marker="o", ls="", ms=5, mfc="white",
                          mec=c_issued, mew=1.1, label=r"$\mathtt{0x0016}$"),
               plt.Line2D([], [], color=c_replay, lw=2.2,
                          label=r"$\mathtt{0x0014}$")]
    ax.legend(handles=handles, frameon=False, loc="upper right", fontsize=6.4,
              handletextpad=.4, borderpad=.2)
    _frame(ax)
    return _save(fig, f"{outdir}/fig_mem_mechanism.png")


# ---------------------------------------------------------- fig 2: novelty --
def fig_triage_novelty(T, outdir):
    """Novelty against total counter deviation, one point per undocumented-
    active encoding, coloured by tier.

    This is the only view in the section that shows the two axes of the
    argument at once: tier B is far from the documented cloud AND almost
    silent, which no table can express.  Tier D collapses onto novelty 0 by
    construction and is drawn small so it reads as the backdrop it is.
    """
    _style()
    TC = _tier_colors()
    fig, ax = plt.subplots(figsize=(3.4, 2.9))
    ax.set_yscale("log")
    ax.set_ylim(.6, 1500)
    ax.set_xlim(-.15, 3.35)
    # kernel.py widens tick marks; on a log axis that makes the minor ticks
    # read as a row of dashes down the spine.  Decades are enough here.
    ax.tick_params(axis="y", which="minor", length=0)

    for t in TR.TIER_ORDER:
        s = T[T.tier == t]
        if s.empty:
            continue
        small = t == TR.TIER_D
        ax.scatter(s.novelty, s.total_dev.clip(lower=.7),
                   s=8 if small else 20, color=TC[t], lw=0,
                   alpha=.4 if small else .8,
                   zorder=2 if small else 3, label=TIER_LABEL[t])

    b = T[T.tier == TR.TIER_B]
    ax.scatter(b.novelty, b.total_dev.clip(lower=.7), s=120, marker="*",
               color=TC[TR.TIER_B], edgecolor="w", lw=.5, zorder=5)
    ax.annotate("4 encodings: far out\nand near-silent",
                xy=(float(b.novelty.iloc[0]), float(b.total_dev.iloc[0]) * 1.5),
                xytext=(1.28, 22), fontsize=6.8, color=TC[TR.TIER_B],
                arrowprops=dict(arrowstyle="->", color=TC[TR.TIER_B], lw=.9))

    ax.axvline(1.0, color="#999999", lw=.8, ls=":", zorder=1)
    ax.text(1.06, 700, "novelty 1.0: as far out as documented\nsignatures are from each other",
            fontsize=6, color="#888888", va="top")
    ax.set_xlabel("novelty  (NN distance / documented spacing)")
    ax.set_ylabel("total counter deviation, 26 events")
    ax.legend(frameon=False, fontsize=6.2, ncol=2, loc="upper center",
              bbox_to_anchor=(.5, -.20), handletextpad=.3,
              borderpad=.2, columnspacing=1.0)
    _frame(ax)
    return _save(fig, f"{outdir}/fig_triage_novelty.png")


# ------------------------------------------------------- fig 3: STORE-FP map --
ROWS = [("unit-stride", 0, 0), ("unit-stride", 0, 1),
        ("strided", 2, 0), ("strided", 2, 1),
        ("indexed", 3, 0), ("indexed", 3, 1)]


def fig_storefp_map(df, Dev, documented, active, outdir):
    """The bit-28 STORE-FP subspace: width against addressing mode and segment
    count, cell coloured by status, annotated with the median 0x0016 reading.

    NOT used in the thesis; see the module docstring.  Emitted only with
    --storefp-map.

    This carries what the table cannot, because the table lists only the
    undocumented-active rows.  Here the documented and inert cells are drawn
    too, so the unit-stride nf=0 row reads left to right as: quiet, three
    ordinary scalar stores, three dead widths, quiet again.  One cell pair is
    unlike everything around it.
    """
    _style()
    TC = _tier_colors()
    pal = _palette()
    C_DOC = pal.get("green", "#3b7d4f")
    C_INERT = "#e8eaed"

    m = (df.opclass == "STORE-FP").values & (df.bit28 == 1).values
    F = pd.DataFrame(dict(width=df.width[m], mop2=df.mop2[m], nf=df.nf[m],
                          doc=documented[m], act=active[m],
                          tier=df.tier[m], e16=Dev.loc[m, "0x0016"]))

    fig, ax = plt.subplots(figsize=(3.4, 2.7))
    for r, (name, mop, seg) in enumerate(ROWS):
        for w in range(8):
            sel = F[(F.mop2 == mop) & (F.width == w) &
                    ((F.nf > 0) if seg else (F.nf == 0))]
            if sel.empty:
                ax.add_patch(plt.Rectangle((w, r), 1, 1, facecolor="white",
                                           edgecolor="#dddddd", lw=.5))
                continue
            if not sel.act.any():
                fc, txt, tc = C_INERT, "inert", "#8a8f96"
            elif sel.doc.all():
                fc, txt, tc = C_DOC, f"{int(sel.e16.median())}", "white"
            else:
                t = sel.tier.mode().iloc[0]
                fc, tc = TC[t], "white"
                txt = f"{int(sel.e16.median())}"
            ax.add_patch(plt.Rectangle((w, r), 1, 1, facecolor=fc,
                                       edgecolor="white", lw=1.0))
            ax.text(w + .5, r + .5, txt, ha="center", va="center",
                    fontsize=6.2, color=tc,
                    fontweight="bold" if txt == "1" else "normal")

    ax.set_xlim(0, 8)
    ax.set_ylim(-1.1, len(ROWS))
    ax.invert_yaxis()
    ax.set_xticks(np.arange(8) + .5)
    ax.set_xticklabels(range(8))
    ax.set_yticks(np.arange(len(ROWS)) + .5)
    ax.set_yticklabels([f"{n}, nf{'>0' if s else '=0'}" for n, _, s in ROWS])
    ax.set_xlabel("width field (bits 14:12)")
    ax.tick_params(length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)

    ax.annotate(r"$\mathtt{0x0016}=1$", xy=(.55, .1), xytext=(4.0, -.85),
                fontsize=6.8, color=TC[TR.TIER_B], ha="center", va="center",
                arrowprops=dict(arrowstyle="->", color=TC[TR.TIER_B], lw=.9,
                                shrinkA=2, shrinkB=2))
    ax.annotate("", xy=(7.45, .1), xytext=(5.3, -.72),
                arrowprops=dict(arrowstyle="->", color=TC[TR.TIER_B], lw=.9,
                                shrinkA=2, shrinkB=2))

    handles = [Patch(facecolor=C_DOC, label="documented, active"),
               Patch(facecolor=TC[TR.TIER_D], label="undocumented, tier D"),
               Patch(facecolor=C_INERT, label="inert"),
               Patch(facecolor=TC[TR.TIER_A], label="undocumented, tier A"),
               Patch(facecolor="white", edgecolor="none", label=" "),
               Patch(facecolor=TC[TR.TIER_B], label="undocumented, tier B")]
    ax.legend(handles=handles, frameon=False, fontsize=6.2, ncol=2,
              loc="upper center", bbox_to_anchor=(.5, -.22),
              handlelength=1.1, handletextpad=.4, columnspacing=1.0)
    return _save(fig, f"{outdir}/fig_storefp_map.png")


# ------------------------------------------------------------------- main --
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scan", default="out/consensus_scan.csv")
    ap.add_argument("--screen", default="out/binutils_screen.csv")
    ap.add_argument("--outdir", default="out")
    ap.add_argument("--storefp-map", action="store_true",
                    help="also emit fig_storefp_map.png, which is not used in "
                         "the thesis (see module docstring)")
    a = ap.parse_args()

    df, ev, Dev, documented, active, T, inv = build(a.scan, a.screen)
    fig_mem_mechanism(df, Dev, documented, active, a.outdir)
    fig_triage_novelty(T, a.outdir)
    if a.storefp_map:
        fig_storefp_map(df, Dev, documented, active, a.outdir)


if __name__ == "__main__":
    main()