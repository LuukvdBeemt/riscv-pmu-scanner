"""
kernel.py -- shared matplotlib styling for the thesis evaluation figures.

This is the one place to edit font, sizing, and color choices for every
evaluation figure. classify_instructions.py, find_anomalies.py,
detect_ghostwrite.py, and analyze_runs.py already import from this module
with a try/except fallback:

    try:
        from kernel import apply_figure_style, panel_letter
    except Exception:
        ... hardcoded fallback ...

so dropping this file next to them is enough to make all four pick it up;
no changes to those scripts are required. analyze_runs.py additionally
imports `set_frame`, defined below.

Any new figure script should do the same:

    from kernel import apply_figure_style, panel_letter, COLORS, FAMILY_COLORS
    apply_figure_style()
    ...
    panel_letter(ax, "a")
"""
import matplotlib as mpl

# ---------------------------------------------------------------------------
# Fonts
# ---------------------------------------------------------------------------
# Sans-serif at small print sizes reads more cleanly than the thesis's LaTeX
# serif once shrunk into a figure. This does NOT need to match the thesis
# body font; figures are conventionally set in a different, plainer face.
# If you want an exact match to the thesis's LaTeX font instead, the usual
# route is mpl.rcParams["text.usetex"] = True, which requires a working
# local LaTeX install and is slower to render; not done here by default.
FONT_FAMILY = "sans-serif"
FONT_SANS = ["DejaVu Sans", "Helvetica", "Arial"]

# ---------------------------------------------------------------------------
# Color palette
# ---------------------------------------------------------------------------
# One palette, used everywhere, by semantic role rather than literal hex code,
# so a figure asks for COLORS["alert"] and every figure stays in sync if the
# palette changes later. This replaces the ad hoc PALETTE / C / FOCAL-COMP-ALARM
# / CATCOL dicts that were previously duplicated (with slightly different
# values) inside each script.
COLORS = {
    "primary":   "#1f6fb4",  # standard / expected / documented behavior
    "secondary": "#8fb8d8",  # secondary series, lighter tint of primary
    "neutral":   "#b9c6d2",  # comparison / background population
    "inert":     "#c9d2da",  # inert / baseline encodings
    "alert":     "#c2452d",  # anomaly / flagged / true-positive
    "warn":      "#e0a030",  # control-flow / caution, secondary flag class
    "gray":      "#8a8f96",  # de-emphasized / neutral family
}

# Categorical palette for the seven functional families, ordered for
# reasonable contrast at small figure sizes.
FAMILY_COLORS = {
    "FP-compute":  "#1f6fb4",
    "INT-compute": "#8a8f96",
    "memory":      "#2ca25f",
    "control":     "#e0a030",
    "system":      "#6a51a3",
    "vector":      "#00a2b3",
    "custom":      "#c2452d",
}

# Anomaly-triage tiers (§Anomaly triage).  Keyed by tier letter, which is what
# the evaluation actually uses; the older three-way verdict names are kept as
# aliases so nothing that imported them breaks.
#
# The four values must stay mutually distinguishable in print.  Tier C and
# tier D previously both resolved to a grey, which made the 68 tier-C points
# in fig_triage_novelty hard to separate from the tier-D backdrop, so tier C
# now carries its own hue.
TRIAGE_COLORS = {
    "A": COLORS["alert"],    # breaks a class invariant
    "B": COLORS["primary"],  # out of documented range, no twin
    "C": COLORS["warn"],     # novel signature, within range
    "D": "#b8bec6",          # exact documented twin: backdrop
}

# Backwards-compatible aliases for the earlier three-way naming.
TRIAGE_COLORS.update({
    "TRUE_POSITIVE":   TRIAGE_COLORS["A"],
    "BENIGN_TRUE_POS": TRIAGE_COLORS["B"],
    "FALSE_POSITIVE":  TRIAGE_COLORS["C"],
})

# ---------------------------------------------------------------------------
# Sizing
# ---------------------------------------------------------------------------
# \onecolgrid / \twocolgrid widths, in inches, for figures meant to span
# exactly one or two columns of the thesis body text.
#
# STILL UNVERIFIED against vusec.sty.  To get the real number, put
# \showthe\onecolgrid in the document and read the log; the value is in
# points, so divide by 72.27 for inches.  Until then figures are drawn at
# ONECOL_WIDTH_IN and scaled by \includegraphics[width=\onecolgrid], so if
# this constant is wrong every figure is silently rescaled and its font sizes
# come out wrong in print.  Worth fixing before the figures are final.
ONECOL_WIDTH_IN = 3.4
TWOCOL_WIDTH_IN = 7.0
FIG_DPI = 300


def apply_figure_style(sizes=(8, 7, 6)):
    """
    Apply consistent rcParams for all thesis evaluation figures.

    sizes: (base, label, annotation)
        base       -- default font size; tick labels
        label      -- axis labels, legend text, +1 for axes titles
        annotation -- small in-plot text (value labels, footnotes)
    """
    base, label, annotation = sizes
    mpl.rcParams.update({
        "font.family": FONT_FAMILY,
        "font.sans-serif": FONT_SANS,
        "font.size": base,
        "axes.labelsize": label,
        "axes.titlesize": label + 1,
        "legend.fontsize": annotation,
        "xtick.labelsize": base,
        "ytick.labelsize": base,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.8,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "savefig.dpi": FIG_DPI,
        "savefig.bbox": "tight",
    })


def panel_letter(ax, letter, dx=-0.13, dy=1.06, fontsize=11, **kwargs):
    """Place a bold panel label (a, b, c, ...) above-left of an axes."""
    kwargs.setdefault("fontweight", "bold")
    kwargs.setdefault("va", "bottom")
    kwargs.setdefault("ha", "left")
    return ax.text(dx, dy, letter, transform=ax.transAxes, fontsize=fontsize, **kwargs)


def set_frame(ax, spines=("top", "right")):
    """Hide the given spines on ax. Default matches apply_figure_style's rcParams,
    for axes created before apply_figure_style() ran, or for one-off overrides."""
    for s in spines:
        ax.spines[s].set_visible(False)


def savefig(fig, path, dpi=FIG_DPI):
    """Consistent save: fixed dpi, tight bbox, closes the figure after."""
    import matplotlib.pyplot as plt
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)