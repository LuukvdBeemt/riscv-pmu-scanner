#!/usr/bin/env python3
"""
screen_report.py -- analysis and figures for the binutils screen.

Consumes out/binutils_screen.csv (written by binutils_screen.py) plus the raw
scan, and produces every table and figure. Nothing here is interactive: run
binutils_screen.py then this, and the full result set regenerates.

    python3 binutils_screen.py --scan out/consensus_scan.csv
    python3 screen_report.py   --scan out/consensus_scan.csv

OUTPUTS
  out/confusion_matrix.csv                 2x2 recognition x activity
  out/inert_documented_taxonomy.csv        why documented encodings are inert
  out/extension_activity_fingerprint.csv   per-extension PMU activity rate
    out/undocumented_active_arch_faults.csv  arch_fault counts for the 635
  out/undocumented_signature_classes.csv   distinct signatures in the headline
  out/confusion_variants.csv               the 2x2 under 3 documented-axis defs
  out/inert_opcode_profile_<v>.csv         documented-inert per opcode + mechanism
  out/fig_inert_opcode_profile_<v>.png     the documented-but-inert cell
  out/opcode_signature_profile_<V>.csv     per-opcode, under variant V
  out/fig_confusion.png                    2x2, default (set B) axis
  out/fig_confusion_variants.png           the three matrices side by side
  out/fig_opcode_profile_<V>.png           where the activity sits in opcode
                                           space (standalone; default set C)
  out/fig_signature_landscape.png          signature classes
  out/fig_inert_taxonomy.png               inert breakdown
"""
import argparse
import json
import os
import re

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

PROBES = {0x10000027: "GhostWrite-family th.vsb.v"}


def event_names(path="event_names.json"):
    try:
        return json.load(open(path))
    except OSError:
        return {}


def fmt_dev(nz, names, k=None):
    it = list(nz.items())[:k] if k else nz.items()
    return ", ".join(f"{names.get(c, c)}{v:+.0f}" for c, v in it)

def _panel(ax, L):
    ax.annotate(L, xy=(0, 1), xycoords="axes fraction", xytext=(-34, 12),
                textcoords="offset points", fontsize=11, fontweight="bold")


GREY = "#8c8c8c"
BLUE = "#4c78a8"
RED = "#b5341f"


def event_columns(scan):
    ev = [c for c in scan.columns if re.fullmatch(r"ev_0x[0-9a-fA-F]{4}", str(c))]
    if not ev:
        ev = [c for c in scan.columns
              if str(c).startswith("0x") and "_spread" not in str(c)]
    ev = [c for c in ev if not re.search(r"cycle|instret", str(c), re.I)]
    if not ev:
        raise RuntimeError("no event columns resolved from scan")
    return ev


def load(screen_csv, scan_csv):
    df = pd.read_csv(screen_csv)
    df["enc"] = df.encoding.apply(lambda s: int(s, 16)).astype("int64")
    if "arch_fault" not in df.columns:
        base, ext = os.path.splitext(screen_csv)
        for sidecar in (f"{base}_arch{ext}", os.path.join(os.path.dirname(screen_csv),
                                                           "binutils_screen_arch.csv")):
            if sidecar != screen_csv and os.path.exists(sidecar):
                arch = pd.read_csv(sidecar, usecols=["encoding", "arch_fault"])
                df = df.merge(arch, on="encoding", how="left")
                break
    scan = pd.read_csv(scan_csv)
    scan["enc"] = scan.encoding.apply(
        lambda v: int(v, 16) if isinstance(v, str) else int(v)).astype("int64")
    ev = event_columns(scan)
    sig = scan.set_index("enc")[ev]
    return df, sig, ev


def confusion_matrix(df, outdir):
    """2x2 recognition x activity. One axis only: can the device ISA name it?
    Lax decodes stay undocumented -- no manual gives those words a meaning."""
    ct = pd.crosstab(df.recognised.map({True: "documented",
                                        False: "undocumented"}),
                     df.active.map({True: "active", False: "inert"}))
    ct = ct.reindex(index=["documented", "undocumented"],
                    columns=["active", "inert"])
    odds, p = stats.fisher_exact(ct.values)
    ct.to_csv(os.path.join(outdir, "confusion_matrix.csv"))
    return ct, odds, p


def unadvertised_extensions(df, outdir):
    """Encodings only a decoder wider than the device ISA can name, grouped by
    the extension that names them. Activity near 0% = correctly excluded.
    Activity near 100% = the chip implements it without advertising it."""
    b = df[df.broad_mnemonic.notna()].copy()
    b["ext"] = b.broad_ext.fillna("rv64gc")
    t = b.groupby("ext").agg(
        n=("enc", "size"), n_active=("active", "sum"),
        n_lax=("lax_decode_of", "count"),
        n_undoc_active=("verdict",
                        lambda s: (s == "undocumented_active").sum())).reset_index()
    t["pct_active"] = (100 * t.n_active / t.n).round(1)
    t["call"] = np.where(t.pct_active >= 50, "IMPLEMENTED, not advertised",
                         np.where(t.n_active == 0, "absent (correctly excluded)",
                                  "partial - inspect"))
    t = t.sort_values(["n_active", "n"], ascending=False)
    t.to_csv(os.path.join(outdir, "unadvertised_extensions.csv"), index=False)
    return t


def arch_fault_breakdown(df, outdir):
    """Architectural fault counts for undocumented-active encodings.

    Uses the enriched arch_fault column emitted by arch_fault_scanner.c when it
    is available. If an older screen CSV does not have the column, skip the
    report rather than failing the whole run.
    """
    if "arch_fault" not in df.columns:
        return None
    t = df[df.verdict == "undocumented_active"].copy()
    if t.empty:
        return pd.DataFrame(columns=["arch_fault", "n", "pct"])
    order = ["none", "sigill", "sigsegv", "sigbus", "sigtrap", "sigfpe",
             "sigalrm"]
    t = t.groupby("arch_fault").size().rename("n").reset_index()
    t["_order"] = t.arch_fault.map(lambda x: order.index(x) if x in order
                                    else len(order))
    t = t.sort_values(["_order", "n", "arch_fault"],
                      ascending=[True, False, True]).drop(columns=["_order"])
    t["pct"] = (100 * t.n / int((df.verdict == "undocumented_active").sum())).round(1)
    t.to_csv(os.path.join(outdir, "undocumented_active_arch_faults.csv"), index=False)
    return t


def inert_taxonomy(df, outdir):
    """Why is a DOCUMENTED encoding inert? Three mechanisms, per the analysis:
    touches no counted event; the core rejects it though binutils accepts it
    (reserved rounding modes); or vector state the scan never establishes."""
    d = df[df.recognised & ~df.active].copy()
    rm = d.enc.apply(lambda v: (v >> 12) & 0x7)
    is_fp = d.mnemonic.str.match(
        r"^f(add|sub|mul|div|sqrt|madd|msub|nmadd|nmsub|cvt)", na=False)
    cat = np.select(
        [is_fp & rm.isin([5, 6]),
         d.mnemonic.str.startswith("auipc", na=False),
         d.mnemonic.str.startswith("th.vf", na=False),
         d.mnemonic.str.startswith("th.v", na=False)],
        ["reserved rounding mode (rm=5,6): binutils over-accepts",
         "auipc: executes, touches no counted event",
         "vector floating-point (th.vf*)",
         "vector, other"],
        default="other")
    t = pd.Series(cat).value_counts().rename_axis("category").reset_index(name="n")
    t["pct"] = (100 * t.n / len(d)).round(1)
    t.to_csv(os.path.join(outdir, "inert_documented_taxonomy.csv"), index=False)
    return t, len(d)


def rounding_mode_check(df, outdir):
    """The rm=5,6 finding as a standalone table: reserved rounding modes should
    be inert on every FP encoding. If they are not, the core accepts them."""
    fp = df[df.mnemonic.str.match(
        r"^f(add|sub|mul|div|sqrt|madd|msub|nmadd|nmsub|cvt)", na=False)].copy()
    fp["rm"] = fp.enc.apply(lambda v: (v >> 12) & 0x7)
    t = fp.groupby("rm").active.agg(["size", "sum"])
    t["pct_active"] = (100 * t["sum"] / t["size"]).round(1)
    t["reserved"] = t.index.isin([5, 6])
    t.to_csv(os.path.join(outdir, "fp_rounding_mode_activity.csv"))
    return t


def extension_fingerprint(df, outdir):
    """Per-extension activity rate. An extension the core does not implement
    should be 0% active -- this is what exposed the too-wide march."""
    rows = []
    for ext, g in df[df.recognised].groupby(df.attributed_to.fillna("rv64gc")):
        rows.append(dict(ext=ext, n=len(g), n_active=int(g.active.sum()),
                         pct_active=round(100 * g.active.mean(), 1)))
    t = pd.DataFrame(rows).sort_values("pct_active")
    t.to_csv(os.path.join(outdir, "extension_activity_fingerprint.csv"),
             index=False)
    return t


def signature_classes(df, sig, ev, outdir, names):
    """Group the headline set by exact PMU signature. A class no documented
    instruction produces is the stronger finding; a match is weak evidence."""
    hid = df[df.verdict == "undocumented_active"]
    doc_active = df[df.recognised & df.active]
    doc_sigs = {}
    for e in doc_active.enc:
        if e in sig.index:
            doc_sigs.setdefault(tuple(sig.loc[e]), e)
    mn = df.set_index("enc").mnemonic

    groups = {}
    for e in hid.enc:
        if e in sig.index:
            groups.setdefault(tuple(sig.loc[e]), []).append(e)

    inert = sig[~df.set_index("enc").reindex(sig.index).active.fillna(True)]
    baseline = (inert if len(inert) else sig).median()

    rows, cls = [], {}
    for i, (s, encs) in enumerate(sorted(groups.items(),
                                         key=lambda kv: -len(kv[1]))):
        cls.update(dict.fromkeys(encs, i))
        owner = doc_sigs.get(s)
        dev = pd.Series(s, index=ev) - baseline
        nz = dev[dev != 0].sort_values(key=abs, ascending=False)
        rows.append(dict(
            sig_class=i, n=len(encs),
            opcodes=",".join(f"0x{o:02x}" for o in sorted({e & 0x7f for e in encs})),
            n_dev=len(nz),
            deviations=fmt_dev(nz, names),
            novel_signature=owner is None,
            matches_documented=(mn.get(owner) if owner is not None else None),
            example=f"0x{encs[0]:08x}",
            members=" ".join(f"0x{e:08x}" for e in sorted(encs))))
    t = pd.DataFrame(rows)
    t.to_csv(os.path.join(outdir, "undocumented_signature_classes.csv"),
             index=False)
    return t, cls, len(doc_sigs), len(doc_active)


def undocumented_mask(df, variant):
    """The documented/undocumented split under one of the confusion-matrix
    variants. Returns a boolean Series aligned to df, True = undocumented.

      B  Zfh counted as documented (phase 2, the default screen verdict)
      C  B, minus every rm=5/6 FP encoding binutils over-accepts -- those are
         decoder bugs, not documented instructions, so they belong on the
         undocumented side. All are inert, so the ACTIVE column is unchanged.
      A  Zfh not counted as documented (phase 1, the as-shipped device view)

    Only the split moves; `active` is measured and identical across variants.
    """
    need = {"A": ["recognised_zfh_excl"], "B": ["recognised_zfh_incl"],
            "C": ["recognised_zfh_incl", "reserved_rm"]}
    if variant not in need:
        raise ValueError(f"variant must be one of {sorted(need)}, got {variant!r}")
    missing = [c for c in need[variant] if c not in df.columns]
    if missing:
        raise RuntimeError(
            f"screen CSV lacks {missing}: re-run binutils_screen.py to emit "
            "the confusion-matrix variant columns")
    if variant == "A":
        return ~df.recognised_zfh_excl
    if variant == "B":
        return ~df.recognised_zfh_incl
    return ~(df.recognised_zfh_incl & ~df.reserved_rm)


def opcode_profile(df, cls, outdir, variant="C", fname=None):
    """Per-opcode: how much of the undocumented space is active, and how many
    distinct behaviours that activity collapses into.

    `variant` selects the documented axis (see undocumented_mask). It changes
    the DENOMINATOR -- how many encodings each opcode contributes to the
    undocumented space -- and therefore pct_active. It cannot change n_active:
    every encoding a variant moves is inert, so the headline set is the same
    635 under A, B and C. That invariance is asserted, not assumed.
    """
    undoc = undocumented_mask(df, variant)
    rows = []
    for op, g in df[undoc].groupby("opcode"):
        h = g[g.verdict == "undocumented_active"]
        ids = {cls[e] for e in h.enc if e in cls}
        rows.append(dict(
            opcode=f"0x{op:02x}", n_undocumented=len(g), n_active=len(h),
            pct_active=round(100 * len(h) / len(g), 1),
            n_signatures=len(ids),
            enc_per_signature=round(len(h) / len(ids), 1) if ids else None))
    t = pd.DataFrame(rows).sort_values("n_active", ascending=False)
    n_head = int((df.verdict == "undocumented_active").sum())
    assert int(t.n_active.sum()) == n_head, (
        f"variant {variant} lost headline encodings: "
        f"{int(t.n_active.sum())} != {n_head}")
    t.attrs["variant"] = variant
    t.to_csv(os.path.join(
        outdir, fname or f"opcode_signature_profile_{variant}.csv"), index=False)
    return t


def fig_opcode_profile(op, outdir, variant="C"):
    """Panel (a) of the signature landscape, standalone: where in the opcode
    map the undocumented-active encodings actually are.

    Lollipops rather than bars -- the count is a single observation per opcode,
    and the dot lets the zero-active opcodes be summarised in one line instead
    of drawn as 19 empty rows.
    """
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    live = op[op.n_active > 0].sort_values("n_active")
    dead = op[op.n_active == 0]
    y = np.arange(len(live))
    ax.hlines(y, 0, live.n_active, color=BLUE, lw=1.4, zorder=2)
    ax.scatter(live.n_active, y, s=44, color=BLUE, zorder=3)
    ax.set_yticks(y, [f"0x{r.opcode[2:]}" for r in live.itertuples()],
                  fontsize=8)
    for i, r in enumerate(live.itertuples()):
        ax.annotate(f"{r.n_active} of {r.n_undocumented:,} ({r.pct_active:.0f}%)"
                    f"  ·  {r.n_signatures} signature"
                    f"{'s' if r.n_signatures != 1 else ''}",
                    (r.n_active, i), xytext=(9, 0), textcoords="offset points",
                    va="center", fontsize=7)
    ax.set_xlim(0, live.n_active.max() * 2.05)
    ax.set_ylim(-1.4, len(live) - .4)
    ax.set_xlabel("undocumented encodings that are PMU-active", fontsize=8.5)
    ax.set_ylabel("opcode", fontsize=8.5)
    ax.set_title(f"Undocumented activity is confined to {len(live)} of "
                 f"{len(op)} opcodes", loc="left", fontsize=9.5)
    ax.tick_params(labelsize=8)
    ax.annotate(f"{len(dead)} further opcodes carry "
                f"{int(dead.n_undocumented.sum()):,} undocumented\nencodings, "
                f"none PMU-active", xy=(.985, .04), xycoords="axes fraction",
                ha="right", va="bottom", fontsize=7, color=GREY)
    ax.annotate(f"documented axis: set {variant}"
                + (" (Zfh included; reserved rm=5/6 counted as undocumented)"
                   if variant == "C" else ""),
                xy=(0, -.155), xycoords="axes fraction", fontsize=6.8,
                color=GREY)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    fig.tight_layout()
    p = os.path.join(outdir, f"fig_opcode_profile_{variant}.png")
    fig.savefig(p, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return p


INERT_MECHANISM = [
    # (predicate over the inert-documented frame, label). Order matters: the
    # first match wins, so the specific cases precede "unclassified".
    (lambda d: d.mnemonic_base.eq("auipc"),
     "auipc: executes, no counted event"),
    (lambda d: d.mnemonic_base.str.startswith("th.vf", na=False),
     "vector FP (th.vf*): no vtype set"),
    (lambda d: d.mnemonic_base.str.startswith("th.v", na=False),
     "vector, other: no vtype set"),
    (lambda d: d.branch_never_taken,
     "branch never taken (rs1==rs2)"),
]


def inert_opcode_profile(df, outdir, variant="C", fname=None):
    """Per-opcode: how much of the DOCUMENTED space is inert, and by what
    mechanism -- the mirror of opcode_profile for the other off-diagonal cell.

    Unlike the undocumented-active profile, this table IS variant-sensitive in
    both columns, and that is the point of parameterising it. Under C the
    rm=5/6 encodings leave the documented side entirely, so 0x53 and the four
    FMADD majors lose their inert population; under B they are the single
    largest inert mechanism. n_inert is therefore not invariant and must not
    be asserted as such.

    Mechanism is attributed per encoding, not per opcode: an opcode can mix
    (0x57 carries both vector-FP and other-vector inertness).
    """
    doc = ~undocumented_mask(df, variant)
    d = df[doc].copy()
    if "mnemonic_base" not in d.columns:
        d["mnemonic_base"] = d.mnemonic.str.split().str[0]
    # A conditional branch with rs1==rs2 under blt/bltu/bne can never be taken,
    # so it retires without moving a branch-taken counter. This is a real
    # property of the encoding, not a measurement artefact -- see the note in
    # fig_inert_opcode_profile.
    e = d.enc.to_numpy()
    d["branch_never_taken"] = (
        (d.opcode == 0x63).to_numpy()
        & np.isin(np.right_shift(e, 12) & 0x7, [1, 4, 6])
        & (((np.right_shift(e, 15) & 0x1f) == (np.right_shift(e, 20) & 0x1f))))
    inert = d[~d.active]
    mech = pd.Series("unclassified", index=inert.index)
    for pred, label in reversed(INERT_MECHANISM):
        mech[pred(inert)] = label

    rows = []
    for op, g in d.groupby("opcode"):
        gi = g[~g.active]
        m = mech.reindex(gi.index)
        rows.append(dict(
            opcode=f"0x{op:02x}", n_documented=len(g), n_inert=len(gi),
            pct_inert=round(100 * len(gi) / len(g), 1),
            n_mnemonics=int(gi.mnemonic_base.nunique()),
            mechanism="; ".join(f"{k} ({v})" for k, v in
                                m.value_counts().items()) or "-"))
    t = pd.DataFrame(rows).sort_values("n_inert", ascending=False)
    n_inert = int((~d.active).sum())
    assert t.n_inert.sum() == n_inert, "per-opcode inert must sum to the cell"
    t.to_csv(os.path.join(
        outdir, fname or f"inert_opcode_profile_{variant}.csv"), index=False)
    return t, mech.value_counts()


def fig_inert_opcode_profile(op, mech, outdir, variant="C"):
    """The documented-but-inert cell, laid out like the undocumented-active
    profile so the two off-diagonal cells can be read side by side.

    Same lollipop form and same reading order (magnitude, descending) as
    fig_opcode_profile deliberately: these two figures answer the mirrored
    halves of one question, and a reader should not have to relearn the
    encoding between them. Colour is the one difference -- inertness is the
    'expected-but-absent' failure, so it takes RED, matching the taxonomy
    figure rather than the activity figure.
    """
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    live = op[op.n_inert > 0].sort_values("n_inert")
    dead = op[op.n_inert == 0]
    y = np.arange(len(live))
    ax.hlines(y, 0, live.n_inert, color=RED, lw=1.4, zorder=2)
    ax.scatter(live.n_inert, y, s=44, color=RED, zorder=3)
    ax.set_yticks(y, [f"0x{r.opcode[2:]}" for r in live.itertuples()],
                  fontsize=8)
    for i, r in enumerate(live.itertuples()):
        ax.annotate(f"{r.n_inert:,} of {r.n_documented:,} ({r.pct_inert:.0f}%)"
                    f"  ·  {r.n_mnemonics} mnemonic"
                    f"{'s' if r.n_mnemonics != 1 else ''}",
                    (r.n_inert, i), xytext=(9, 0), textcoords="offset points",
                    va="center", fontsize=7)
    # 1.62 not 2.05 (the undocumented profile's value): the annotations here
    # are shorter, and the longest lollipop is the widest, so less trailing
    # room is needed to keep its label inside the axes.
    ax.set_xlim(0, live.n_inert.max() * 1.62)
    ax.set_ylim(-1.4, len(live) - .4)
    ax.set_xlabel("documented encodings that are PMU-inert", fontsize=8.5)
    ax.set_ylabel("opcode", fontsize=8.5)
    ax.set_title(f"Documented inertness is confined to {len(live)} of "
                 f"{len(op)} opcodes", loc="left", fontsize=9.5)
    ax.tick_params(labelsize=8)
    ax.annotate(f"{len(dead)} further opcodes carry "
                f"{int(dead.n_documented.sum()):,} documented\nencodings, "
                f"all PMU-active", xy=(.985, .04), xycoords="axes fraction",
                ha="right", va="bottom", fontsize=7, color=GREY)
    ax.annotate(f"documented axis: set {variant}"
                + (" (Zfh included; reserved rm=5/6 counted as undocumented)"
                   if variant == "C" else ""),
                xy=(0, -.155), xycoords="axes fraction", fontsize=6.8,
                color=GREY)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    fig.tight_layout()
    p = os.path.join(outdir, f"fig_inert_opcode_profile_{variant}.png")
    fig.savefig(p, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return p


def probe_encodings(df, cls, sg, outdir, probes):
    """Where named encodings of interest land."""
    rows = []
    for enc, label in probes.items():
        r = df[df.enc == enc]
        if not len(r):
            rows.append(dict(label=label, encoding=f"0x{enc:08x}",
                             verdict="NOT IN SCAN"))
            continue
        r = r.iloc[0]
        c = cls.get(enc)
        s = sg[sg.sig_class == c].iloc[0] if c is not None else None
        rows.append(dict(
            label=label, encoding=f"0x{enc:08x}", opcode=f"0x{r.opcode:02x}",
            verdict=r.verdict, sig_class=c,
            class_size=(int(s.n) if s is not None else None),
            novel_signature=(bool(s.novel_signature) if s is not None else None),
            deviations=(s.deviations if s is not None else None),
            class_members=(s.members if s is not None else None)))
    t = pd.DataFrame(rows)
    t.to_csv(os.path.join(outdir, "probe_encodings.csv"), index=False)
    return t


def headline_table(df, cls, sg, outdir):
    """The raw list, enriched: every headline encoding with its signature
    class and what that class does. binutils_hidden.csv has verdicts only."""
    hid = df[df.verdict == "undocumented_active"].copy()
    hid["sig_class"] = hid.enc.map(cls)
    m = sg.set_index("sig_class")
    for c in ("n_dev", "deviations", "novel_signature"):
        hid[c] = hid.sig_class.map(m[c])
    hid["class_size"] = hid.sig_class.map(m.n)

    cols = ["encoding", "opcode", "sig_class", "class_size", "novel_signature",
            "n_dev", "deviations", "broad_mnemonic", "broad_ext", "attributed_to"]
    out = hid[cols].sort_values(["sig_class", "encoding"])
    out.to_csv(os.path.join(outdir, "undocumented_active_detail.csv"),
               index=False)
    return out


def fig_confusion(ct, odds, outdir):
    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    M = ct.values
    face = [[BLUE, "#b0b7c3"], [RED, "#dfe3e8"]]
    for i in range(2):
        for j in range(2):
            ax.add_patch(plt.Rectangle((j - .5, i - .5), 1, 1,
                                       facecolor=face[i][j], edgecolor="white",
                                       lw=2))
            tc = "white" if face[i][j] in (BLUE, RED) else "#222"
            ax.text(j, i - .08, f"{M[i, j]:,}", ha="center", va="center",
                    fontsize=13, fontweight="bold", color=tc)
            ax.text(j, i + .20, f"{100 * M[i, j] / M.sum():.1f}%", ha="center",
                    va="center", fontsize=8, color=tc)
    ax.set_xticks([0, 1], ["active", "inert"])
    ax.set_yticks([0, 1], ["documented", "undocumented"])
    ax.set_xlabel("PMU response"); ax.set_ylabel("toolchain recognition")
    ax.set_title(f"Recognition predicts activity (OR = {odds:.0f})", loc="left")
    ax.set_xlim(-.5, 1.5); ax.set_ylim(1.5, -.5)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.tick_params(length=0)
    fig.tight_layout()
    p = os.path.join(outdir, "fig_confusion.png")
    fig.savefig(p, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return p


def confusion_variants_report(df, outdir):
    """Read back the three-variant matrix binutils_screen.py wrote, or rebuild
    it from the screen CSV if that file predates the variant columns."""
    p = os.path.join(outdir, "confusion_variants.csv")
    if os.path.exists(p):
        return pd.read_csv(p)
    need = {"recognised_zfh_incl", "recognised_zfh_excl", "reserved_rm",
            "reserved_rm_zfh"}
    if not need.issubset(df.columns):
        raise RuntimeError(
            "confusion_variants.csv absent and the screen CSV lacks "
            f"{sorted(need - set(df.columns))}: re-run binutils_screen.py")
    def cells(rec):
        r, a = np.asarray(rec, bool), np.asarray(df.active, bool)
        return dict(documented_active=int((r & a).sum()),
                    documented_inert=int((r & ~a).sum()),
                    undocumented_active=int((~r & a).sum()),
                    undocumented_inert=int((~r & ~a).sum()))
    return pd.DataFrame([
        dict(variant="A", basis="Zfh excluded", **cells(df.recognised_zfh_excl)),
        dict(variant="B", basis="Zfh included", **cells(df.recognised_zfh_incl)),
        dict(variant="C", basis="Zfh included, rm=5/6 reclassified",
             **cells(df.recognised_zfh_incl & ~df.reserved_rm)),
        dict(variant="C'", basis="Zfh included, Zfh-added rm=5/6 reclassified",
             **cells(df.recognised_zfh_incl & ~df.reserved_rm_zfh))])


def fig_confusion_variants(cv, outdir):
    """The same 28,672 encodings under three definitions of 'documented'.
    Cells are shaded by share of the grid so the eye reads the split, not the
    absolute count; the count itself is printed, because that is the number
    anyone will quote."""
    v = cv[cv.variant.isin(["A", "B", "C"])].reset_index(drop=True)
    titles = {"A": "A  Zfh excluded\n(as-shipped device ISA)",
              "B": "B  Zfh included\n(confirmed capability)",
              "C": "C  Zfh included, reserved rm=5/6\nmoved to undocumented+inert"}
    fig, axes = plt.subplots(1, 3, figsize=(11.4, 3.5))
    total = int(v.iloc[0][["documented_active", "documented_inert",
                           "undocumented_active", "undocumented_inert"]].sum())
    for ax, r in zip(axes, v.itertuples()):
        M = np.array([[r.documented_active, r.documented_inert],
                      [r.undocumented_active, r.undocumented_inert]])
        face = [[BLUE, "#b0b7c3"], [RED, "#dfe3e8"]]
        for i in range(2):
            for j in range(2):
                ax.add_patch(plt.Rectangle((j - .5, i - .5), 1, 1,
                                           facecolor=face[i][j],
                                           edgecolor="white", lw=2))
                tc = "white" if face[i][j] in (BLUE, RED) else "#222"
                ax.text(j, i - .08, f"{M[i, j]:,}", ha="center", va="center",
                        fontsize=12, fontweight="bold", color=tc)
                ax.text(j, i + .20, f"{100 * M[i, j] / M.sum():.1f}%",
                        ha="center", va="center", fontsize=7.5, color=tc)
        ax.set_xticks([0, 1], ["active", "inert"], fontsize=8)
        ax.set_yticks([0, 1], ["documented", "undocumented"], fontsize=8)
        ax.set_title(titles[r.variant], loc="left", fontsize=8.5)
        ax.set_xlim(-.5, 1.5); ax.set_ylim(1.5, -.5)
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.tick_params(length=0)
    axes[0].set_ylabel("toolchain recognition", fontsize=8)
    for ax in axes:
        ax.set_xlabel("PMU response", fontsize=8)
    fig.suptitle(f"Only the documented axis moves: {total:,} encodings, "
                 "identical activity measurement in all three", fontsize=9.5,
                 x=.01, y=.99, ha="left", va="top")
    fig.tight_layout(rect=(0, 0, 1, .94))
    p = os.path.join(outdir, "fig_confusion_variants.png")
    fig.savefig(p, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return p


def fig_signature_landscape(sg, pr, outdir):
    """The signature panel alone. The opcode profile that used to sit beside it
    as panel (a) is now fig_opcode_profile(), a standalone figure driven by an
    explicit documented-axis variant -- keeping a copy here would be a second
    rendering of the same numbers, free to drift from the first."""
    fig, a2 = plt.subplots(figsize=(7.6, 5.4))

    s_ = sg.sort_values("n").reset_index(drop=True)
    yy = np.arange(len(s_))
    col = [RED if n else GREY for n in s_.novel_signature]
    a2.hlines(yy, .85, s_.n, color=col, lw=1.2, zorder=2)
    a2.scatter(s_.n, yy, s=22, color=col, zorder=3)
    a2.set_xscale("log"); a2.set_xlim(.8, 700); a2.set_ylim(-1.2, len(s_) + .6)
    lab = []
    for r in s_.itertuples():
        d = r.deviations.split(", ")
        short = ", ".join(x.replace("IDU_RF_", "").replace("INST_", "")
                          for x in d[:2])
        lab.append(f"{r.sig_class:>2d}  {short}" + (" …" if len(d) > 2 else ""))
    a2.set_yticks(yy, lab, fontsize=5.6, fontfamily="DejaVu Sans Mono")
    a2.set_xlabel("encodings sharing the signature")
    # Count from the table, never a literal: the class count is data-dependent
    # (it moves with the march and the activity basis) and a stale "33" in the
    # title of a figure that now draws 31 rows is a silent error.
    a2.set_title(f"All {len(sg)} signatures in the {int(sg.n.sum())}-encoding "
                 "headline set, by the counters they move", loc="left",
                 fontsize=9.5)

    g = pr[pr.sig_class.notna()]
    if len(g):
        gi = int(s_.index[s_.sig_class == int(g.sig_class.iloc[0])][0])
        gn = float(g.class_size.iloc[0])
        a2.scatter([gn], [gi], s=150, facecolor="none", edgecolor=RED, lw=1.3,
                   zorder=4)
        a2.annotate(g.label.iloc[0], xy=(gn * 1.25, gi), xytext=(26, gi),
                    fontsize=7, color=RED, va="center",
                    arrowprops=dict(arrowstyle="-", lw=.9, color=RED))
        a2.get_yticklabels()[gi].set_color(RED)
    for t, c in ((f"novel: no documented instruction produces it "
                  f"({int(sg.novel_signature.sum())} classes, "
                  f"{int(sg[sg.novel_signature].n.sum())} enc.)", RED),
                 (f"matches a documented signature "
                  f"({int((~sg.novel_signature).sum())} classes, "
                  f"{int(sg[~sg.novel_signature].n.sum())} enc.)", GREY)):
        a2.plot([], [], color=c, lw=2.4, label=t)
    a2.legend(loc="lower right", frameon=False, fontsize=6.5)
    # No panel letter: this is a standalone figure now, not panel (b) of a pair.

    fig.tight_layout()
    p = os.path.join(outdir, "fig_signature_landscape.png")
    fig.savefig(p, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return p


def fig_inert_taxonomy(tax, n_inert, ext, outdir):
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(11.6, 4.4))
    t = tax.sort_values("n").reset_index(drop=True)
    lab = [c.replace(": ", ":\n") for c in t.category]
    col = [RED if "over-accepts" in c else (BLUE if "auipc" in c else GREY)
           for c in t.category]
    y = np.arange(len(t))
    axA.barh(y, t.n, color=col, height=.66)
    axA.set_yticks(y); axA.set_yticklabels(lab, fontsize=7.2)
    axA.set_xlabel("encodings")
    axA.set_xlim(0, t.n.max() * 1.25)
    axA.set_title(f"Why {n_inert:,} documented encodings are inert", loc="left")
    for i, (n_, p_) in enumerate(zip(t.n, t.pct)):
        axA.annotate(f"{n_:,} ({p_}%)", (n_, i), xytext=(5, 0),
                     textcoords="offset points", va="center", fontsize=7.5)

    e = ext.sort_values("pct_active").reset_index(drop=True)
    y2 = np.arange(len(e))
    axB.hlines(y2, 0, e.pct_active, color=BLUE, lw=1.0, alpha=.7)
    axB.scatter(e.pct_active, y2, s=np.clip(e.n.values * .35, 14, 110),
                color=BLUE, zorder=3)
    axB.set_yticks(y2); axB.set_yticklabels(e.ext, fontsize=6.8)
    axB.set_xlim(-4, 112); axB.set_xlabel("encodings PMU-active (%)")
    axB.set_title("Activity rate per attributed extension", loc="left")
    axB.axvline(0, color="#999", lw=.6, zorder=0)
    for ax, L in ((axA, "a"), (axB, "b")):
        ax.annotate(L, xy=(0, 1), xycoords="axes fraction", xytext=(-34, 12),
                    textcoords="offset points", fontsize=11, fontweight="bold")
    fig.tight_layout()
    p = os.path.join(outdir, "fig_inert_taxonomy.png")
    fig.savefig(p, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scan", required=True)
    ap.add_argument("--screen", default=None,
                    help="default: <outdir>/binutils_screen.csv")
    ap.add_argument("--outdir", default="out")
    ap.add_argument("--profile-variant", choices=["A", "B", "C"], default="C",
                    help="documented axis for the opcode profile figure/table. "
                         "C (default): Zfh included and reserved rm=5/6 counted "
                         "as undocumented. B: Zfh included only. A: Zfh excluded "
                         "(as-shipped device view).")
    a = ap.parse_args()
    screen = a.screen or os.path.join(a.outdir, "binutils_screen.csv")

    df, sig, ev = load(screen, a.scan)
    print(f"loaded {len(df)} encodings, {len(ev)} event columns")

    ct, odds, p = confusion_matrix(df, a.outdir)
    print("\n=== 2x2 recognition x activity ===")
    print(ct.to_string())
    print(f"  odds ratio {odds:.1f}   Fisher p {p:.3g}")
    print(f"  undocumented active rate: "
          f"{100 * ct.loc['undocumented', 'active'] / ct.loc['undocumented'].sum():.2f}%")

    cv = confusion_variants_report(df, a.outdir)
    print("\n=== confusion matrix under three documented-axis definitions ===")
    print(cv[["variant", "documented_active", "documented_inert",
              "undocumented_active", "undocumented_inert", "basis"]]
          .to_string(index=False))

    tax, n_inert = inert_taxonomy(df, a.outdir)
    print(f"\n=== why {n_inert} documented encodings are inert ===")
    print(tax.to_string(index=False))

    iop, imech = inert_opcode_profile(df, a.outdir, variant=a.profile_variant)
    print(f"\n=== documented space per opcode, inert share (documented axis: "
          f"set {a.profile_variant}) ===")
    print(iop.to_string(index=False))
    if a.profile_variant != "B":
        iopB, _ = inert_opcode_profile(df, a.outdir, variant="B")
        lost = (iop.set_index("opcode").n_inert
                - iopB.set_index("opcode").n_inert).fillna(0).astype(int)
        lost = lost[lost != 0]
        if len(lost):
            print(f"  vs set B, the inert count changes for: "
                  + ", ".join(f"{k} {v:+d}" for k, v in lost.items())
                  + f" (total {int(lost.sum()):+d}: the rm=5/6 encodings "
                    "leave the documented side entirely)")

    rm = rounding_mode_check(df, a.outdir)
    print("\n=== FP activity by rounding mode (5,6 are reserved) ===")
    print(rm.to_string())

    ext = extension_fingerprint(df, a.outdir)
    print("\n=== per-extension activity ===")
    print(ext.to_string(index=False))

    faults = arch_fault_breakdown(df, a.outdir)
    if faults is not None:
        print(f"\n=== architectural faults over the {int((df.verdict == 'undocumented_active').sum())} undocumented-active encodings ===")
        print(faults.to_string(index=False))

    names = event_names()
    sg, cls, n_doc_sig, n_doc = signature_classes(df, sig, ev, a.outdir, names)
    print(f"\n=== signatures in the headline set ===")
    print(f"{int(sg.n.sum())} encodings -> {len(sg)} distinct signatures; "
          f"{int(sg.novel_signature.sum())} classes "
          f"({int(sg[sg.novel_signature].n.sum())} encodings) are novel")
    print(f"(context: {n_doc} documented-active encodings span only "
          f"{n_doc_sig} signatures, so a MATCH is weak evidence)")
    show = ["sig_class", "n", "opcodes", "n_dev", "deviations",
            "novel_signature", "matches_documented", "example"]
    print(sg[show].to_string(index=False, max_colwidth=44))

    ux = unadvertised_extensions(df, a.outdir)
    print("\n=== extensions the device does NOT advertise ===")
    print(ux[ux.n_active > 0].to_string(index=False))
    z = ux[ux.pct_active >= 50]
    if len(z):
        print(f"  -> {', '.join(z.ext)} behave as IMPLEMENTED; "
              f"{int(ux[ux.n_active == 0].n.sum())} encodings across "
              f"{int((ux.n_active == 0).sum())} other extensions are fully inert")

    op = opcode_profile(df, cls, a.outdir, variant=a.profile_variant)
    print(f"\n=== undocumented space per opcode (documented axis: set "
          f"{a.profile_variant}) ===")
    print(op.to_string(index=False))
    if a.profile_variant != "B":
        opB = opcode_profile(df, cls, a.outdir, variant="B")
        moved = (op.set_index("opcode").n_undocumented
                 - opB.set_index("opcode").n_undocumented).fillna(0)
        moved = moved[moved != 0].astype(int)
        if len(moved):
            print(f"  vs set B, the undocumented denominator grows for: "
                  + ", ".join(f"{k} +{v}" for k, v in moved.items())
                  + f" (total +{int(moved.sum())}, all inert; "
                    "n_active unchanged)")

    pr = probe_encodings(df, cls, sg, a.outdir, PROBES)
    print("\n=== named probes ===")
    for _, r in pr.iterrows():
        print(f"  {r.label} {r.encoding}: {r.verdict}, class {r.sig_class} "
              f"(n={r.class_size}, novel={r.novel_signature})")
        print(f"    deviations : {r.deviations}")
        print(f"    class      : {r.class_members}")

    det = headline_table(df, cls, sg, a.outdir)

    for f in (fig_confusion(ct, odds, a.outdir),
              fig_confusion_variants(cv, a.outdir),
              fig_inert_taxonomy(tax, n_inert, ext, a.outdir),
              fig_opcode_profile(op, a.outdir, variant=a.profile_variant),
              fig_inert_opcode_profile(iop, imech, a.outdir,
                                       variant=a.profile_variant),
              fig_signature_landscape(sg, pr, a.outdir)):
        print(f"wrote {f}")


if __name__ == "__main__":
    main()
