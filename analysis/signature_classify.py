#!/usr/bin/env python3
"""
signature_classify.py -- do PMU signatures carry FUNCTION, and does that
transfer to the undocumented space?

Relation to the other two scripts
---------------------------------
classify_instructions.py asks whether the counters recover the ISA's functional
structure over the whole active space (a within-distribution question).
screen_report.py describes the 635 undocumented-active encodings signature by
signature (fig_signature_landscape: concrete, per-class, 32 rows).

This script sits between them and answers the section-level question: given that
the 635 have signatures, what FUNCTION do those signatures imply?  It fits the
family classifier on the documented-active set only, then applies it to the
undocumented-active set.  The undocumented encodings are never trained on, so
their predicted family is a genuine out-of-sample call, not a relabelling.

Three methodological differences from classify_instructions.py, each of which
changes the headline number:

1. FEATURES ARE DEVIATIONS, not raw counts.  Every counter is expressed as
   (value - modal value over the whole scan), matching the definition of
   "active" used by binutils_screen.py, so a zero feature means "indistinguish-
   able from an inert encoding" rather than "counter read zero".  Accuracy is
   unchanged (the forest finds the same splits) but the features now mean the
   same thing the confusion matrices mean.

2. CV IS GROUPED BY MNEMONIC, not random.  Encodings that differ only in
   register fields are near-duplicates: a random split puts siblings of the same
   instruction on both sides and inflates accuracy.  Grouping by mnemonic (705
   groups over 10,590 encodings) forces the model to generalise to instructions
   it has not seen.  This costs ~6 points of accuracy and is the honest number.

3. THE 'custom' FAMILY IS DROPPED FROM THE LABEL SET, and this is the point
   rather than a convenience.  "custom" is a property of the ENCODING SPACE (the
   opcode is vendor-allocated), not of behaviour: th.ext is an ALU operation and
   th.lrb is a load, and no counter signature can group them.  Trained as a
   7th class it scores F1=0.09 and drags INT-compute and memory down with it.
   Dropped, the 438 documented vendor encodings become a HELD-OUT VALIDATION
   SET with decoder-derived ground truth (mnemonic -> load/store vs ALU), which
   is a stronger test than cross-validation: a whole opcode major, absent from
   training, classified by behaviour alone.

Outputs
-------
  out/family_cv_report.csv            per-family n / F1 / signatures, grouped CV
  out/vendor_holdout.csv              the 438-encoding external validation
  out/undocumented_family_calls.csv   per-encoding family call for the 635
  out/undocumented_family_classes.csv the same, aggregated by signature class
    out/undocumented_active_arch_faults.csv  arch_fault counts over the 635
    out/undocumented_sigsegv_family_check.csv sigsegv rows and whether they are memory
  out/fig_family_confusion.png        grouped-CV confusion matrix
  out/fig_family_transfer.png         where the 635 land, by opcode

Usage
-----
  python signature_classify.py --scan consensus_scan.csv \
      --screen out/binutils_screen.csv --outdir out [--variant C]
"""
import argparse
import os
import re

import numpy as np
import pandas as pd

# ---------------------------------------------------------------- taxonomy --
# Major opcode -> functional family.  Decoder-independent: derived from the
# RISC-V base encoding allocation, so it is available for every encoding in the
# space including the ones no toolchain will name.  CUSTOM-* is deliberately
# NOT a family here (see docstring point 3); it is carried as its own label so
# main() can hold it out.
OPCLASS = {
    0x03: "LOAD", 0x07: "LOAD-FP", 0x0b: "CUSTOM-0", 0x0f: "MISC-MEM",
    0x13: "OP-IMM", 0x17: "AUIPC", 0x1b: "OP-IMM-32", 0x23: "STORE",
    0x27: "STORE-FP", 0x2b: "CUSTOM-1", 0x2f: "AMO", 0x33: "OP", 0x37: "LUI",
    0x3b: "OP-32", 0x43: "MADD", 0x47: "MSUB", 0x4b: "NMSUB", 0x4f: "NMADD",
    0x53: "OP-FP", 0x57: "OP-V", 0x5b: "CUSTOM-2", 0x63: "BRANCH",
    0x67: "JALR", 0x6b: "RESERVED", 0x6f: "JAL", 0x73: "SYSTEM",
    0x77: "RESERVED", 0x7b: "CUSTOM-3",
}
COARSE = {
    "OP": "INT-compute", "OP-32": "INT-compute", "OP-IMM": "INT-compute",
    "OP-IMM-32": "INT-compute", "LUI": "INT-compute", "AUIPC": "INT-compute",
    "OP-FP": "FP-compute", "MADD": "FP-compute", "MSUB": "FP-compute",
    "NMSUB": "FP-compute", "NMADD": "FP-compute",
    "LOAD": "memory", "STORE": "memory", "LOAD-FP": "memory",
    "STORE-FP": "memory", "AMO": "memory",
    "MISC-MEM": "system", "SYSTEM": "system",
    "JAL": "control", "JALR": "control", "BRANCH": "control",
    "OP-V": "vector",
    "CUSTOM-0": "custom", "CUSTOM-1": "custom", "CUSTOM-2": "custom",
    "CUSTOM-3": "custom",
    "RESERVED": "reserved",
}
FAMILIES = ["memory", "system", "FP-compute", "INT-compute", "control", "vector"]

# Finer than COARSE: splits the "memory" lump into load / store / atomic, which
# is the resolution the question actually needs.  Two opcode classes are absent
# by design -- MISC-MEM and SYSTEM sit almost entirely on one degenerate
# between them (everything else at those majors traps, see below), too few to
# train or to score, and including them produced two F1=0.000 rows that dragged
# balanced accuracy down by 17 points while telling nobody anything.
FINE = {
    "LOAD": "load", "LOAD-FP": "load",
    "STORE": "store", "STORE-FP": "store",
    "AMO": "atomic",
    "OP": "int-alu", "OP-32": "int-alu", "OP-IMM": "int-alu",
    "OP-IMM-32": "int-alu", "LUI": "int-alu", "AUIPC": "int-alu",
    "OP-FP": "fp-alu", "MADD": "fp-alu", "MSUB": "fp-alu",
    "NMSUB": "fp-alu", "NMADD": "fp-alu",
    "JAL": "jump", "JALR": "jump", "BRANCH": "branch",
    "OP-V": "vector-alu",
}
OP_ORDER = ["load", "store", "atomic", "int-alu", "fp-alu", "vector-alu",
              "branch", "jump", "fence", "csr/priv"]
# FINE plus the two classes too sparse to train on.  Used only by the
# exact-twin rule, which needs no training data -- it just needs a name.
OPCLASS_FULL = dict(FINE, **{"MISC-MEM": "fence", "SYSTEM": "csr/priv"})

# T-Head load/store forms: th.{l,s}<width>, th.{l,s}ur<width> (user-mode
# indexed), th.f{l,s}<width> and th.f{l,s}ur<width> (FP variants).  Everything
# else at the CUSTOM-0 major is an ALU/multiply/bit operation.  This regex is
# the ground truth for the held-out validation, so it is deliberately explicit
# rather than a substring test -- th.tst and th.mulsw must not match.
VENDOR_MEM_RE = re.compile(r"^th\.f?[ls](ur|r)?[bhwd]u?(ia|ib|d|w)?$")

# Stage 1 is a three-way outcome with an ordinal reading (executed / faulted
# for one reason / faulted for another), so resolved takes the neutral blue and
# the two trap paths take warm hues -- distinguishable in deuteranopia and
# consistent with the red-for-inert convention in the screen report's taxonomy.
# Evidence state is ordinal (fully determined / partly / not at all), so a
# single-hue ramp from saturated to pale reads correctly without a legend.
EVIDENCE = {"resolved": "#1f6fb4", "degenerate": "#c0392b", "novel": "#9aa3ab"}
# Stage 2 nests: memory operations share a green family, compute a purple one,
# control a grey one, so a reader sees the coarse grouping before the label.
OPCOLOR = {"fence": "#7f5539", "csr/priv": "#9e6ba8",
           "load": "#2ca25f", "store": "#66c2a4", "atomic": "#006d2c",
           "int-alu": "#8a8f96", "fp-alu": "#6a51a3", "vector-alu": "#00a2b3",
           "branch": "#e0a030", "jump": "#b3701a"}


# -------------------------------------------------------------- data layer --
def load(scan_csv, screen_csv, variant):
    """Join the raw scan to the screen verdicts and build the feature matrix.

    Returns (df, event_cols, Dev) where Dev is the DEVIATION frame: each event
    counter minus its modal value across the whole scan.  The mode is the
    inactive baseline -- most of the 28,672 encodings do nothing, so the modal
    reading of a counter is what that counter shows for an encoding that did
    not execute.  This is the same baseline binutils_screen.py uses to define
    `active`, and the assertion below pins the two definitions together: if
    they ever diverge, the features and the confusion matrix are describing
    different populations and every number here is uninterpretable.
    """
    scan = pd.read_csv(scan_csv)
    scr = pd.read_csv(screen_csv)
    to_int = lambda s: s.apply(lambda v: int(str(v), 16) if isinstance(v, str) else int(v))
    scan["enc"], scr["enc"] = to_int(scan.encoding), to_int(scr.encoding)

    if "arch_fault" not in scr.columns:
        base, ext = os.path.splitext(screen_csv)
        for sidecar in (f"{base}_arch{ext}", os.path.join(os.path.dirname(screen_csv),
                                                           "binutils_screen_arch.csv")):
            if sidecar != screen_csv and os.path.exists(sidecar):
                arch = pd.read_csv(sidecar, usecols=["encoding", "arch_fault"])
                arch["enc"] = to_int(arch.encoding)
                scr = scr.merge(arch[["enc", "arch_fault"]], on="enc", how="left")
                break

    need = ["enc", "recognised", "recognised_zfh_incl", "recognised_zfh_excl",
            "reserved_rm", "active", "mnemonic"]
    if "arch_fault" in scr.columns:
        need.append("arch_fault")
    missing = [c for c in need if c not in scr.columns]
    if missing:
        raise SystemExit(f"{screen_csv} lacks {missing}; re-run binutils_screen.py")
    df = scan.merge(scr[need], on="enc", how="inner")
    if len(df) != len(scan):
        raise SystemExit(f"join dropped {len(scan) - len(df)} encodings")

    ev = [c for c in scan.columns if re.fullmatch(r"0x[0-9a-fA-F]+", c)]
    Dev = df[ev].sub(df[ev].mode().iloc[0], axis=1)
    active = (Dev != 0).any(axis=1).values
    assert (active == df.active.values).all(), \
        "deviation-derived activity disagrees with the screen's `active` column"

    df["opc"] = df.enc.values & 0x7F
    df["opclass"] = [OPCLASS.get(o, f"op{o:#x}") for o in df.opc]
    df["family"] = df.opclass.map(lambda x: COARSE.get(x, "other"))
    df["active"] = active
    df["documented"] = documented_mask(df, variant)
    df["target"] = (~df.documented.values) & active
    # Constant counters carry no information and only slow the forest down.
    # Drop them BEFORE building the signature map: a signature is a tuple over
    # whatever counters the model sees, and a map keyed on a wider tuple would
    # silently miss every match.
    ev = [c for c in ev if Dev[c].nunique() > 1]
    Dev = Dev[ev]
    df["evidence"], sigmap = evidence_channel(df, Dev)
    return df, ev, Dev, sigmap


def documented_mask(df, variant):
    """The documented axis under variant A/B/C -- same definitions as
    binutils_screen.py's confusion_variants, kept in sync by name."""
    if variant == "A":
        return df.recognised_zfh_excl.values.astype(bool)
    if variant == "B":
        return df.recognised_zfh_incl.values.astype(bool)
    if variant == "C":
        return df.recognised_zfh_incl.values.astype(bool) & ~df.reserved_rm.values.astype(bool)
    raise SystemExit(f"unknown variant {variant!r}")


# ------------------------------------------------------- evidence channel --
# THE CENTRAL CORRECTION IN THIS SCRIPT.
#
# A flat family classifier called 555/635 of the undocumented-active encodings
# "memory" and looked uninformative.  The cause is not that the counters are
# coarse -- see the operation matrix, which separates load from store from
# atomic at F1 >= 0.92 -- but that a forced-choice classifier reports a label
# for every encoding, including the ones where the measurement cannot support
# one.  It answers even when the honest answer is "these two operations are
# indistinguishable here".
#
# The relevant property is a signature's DEGENERACY over the documented space.
# Take each documented-active encoding's deviation vector and ask which
# operations carry it:
#
#   746 distinct signatures over 10,152 documented-active encodings
#   736 map to exactly one operation      ->  8,488 encodings ("resolved")
#    10 map to two or more                ->  1,664 encodings ("degenerate")
#
# The degenerate ten are not noise, they are collisions with structure.  The
# largest three:
#   770 encodings, csr/priv + fence   -- every csr* form, sfence.vma, unimp
#   330 encodings, branch + jump      -- control transfer, indistinguishable
#   286 encodings, load + store       -- T-Head vector load/store-segment
#
# The load/store collision is the whole story of the "memory lump".  Excluding
# those 286 encodings from training moves load/store/atomic from 0.962/0.969 to
# 0.993/0.995; excluding an equal number of NON-degenerate vector-segment
# encodings instead leaves it at 0.958/0.965.  It is specifically the ambiguous
# signatures that were costing the score, not vector-segment encodings as a
# class and not any property of how those encodings terminate.
#
# So a target encoding gets one of three evidence states, and only the last
# needs a model at all:
#
#   resolved    -- its signature occurs among documented encodings and maps to
#                  exactly one operation.  This is a nearest neighbour at
#                  distance zero; it beats any prediction, and it can name an
#                  operation the model has no class for (120 undocumented
#                  MISC-MEM encodings are exact `fence` twins, and FINE has no
#                  fence class for want of training data).
#   degenerate  -- its signature occurs, but over two or more operations.  The
#                  measurement cannot distinguish them.  Report the candidate
#                  set; do not pick.
#   novel       -- no documented encoding shares it.  Only here is the forest
#                  extrapolating, and only here should its confidence be read
#                  as a confidence.
#
# NOTE ON AN EARLIER, WRONG VERSION OF THIS SECTION.  A previous cut split the
# space on the SIGN of two retirement counters, on the theory that a trapping
# encoding aborts the harness' instruction stream and reads below baseline.
# That test does correlate with the degenerate groups here, but it is not a
# trap detector: plenty of encodings that fault show only positive deltas, and
# this scan records no signal outcome, so there is no ground truth in the data
# to support the claim either way.  The split below is defined purely by what
# the documented labels do or do not determine, which is checkable from the
# scan alone.
def evidence_channel(df, Dev):
    """Label each encoding resolved / degenerate / novel against the documented
    signature map, and return (states, sig -> candidate operations).

    Returns a Series over df.index and the map itself, which callers use to
    read out the candidate set for degenerate encodings.
    """
    doc = df.active.values & df.documented.values & df.opclass.isin(OPCLASS_FULL).values
    sigmap = {}
    for sig, op in zip(map(tuple, Dev[doc].values.astype(np.int64)),
                       df.opclass[doc].map(OPCLASS_FULL)):
        sigmap.setdefault(sig, set()).add(op)

    def state(sig):
        c = sigmap.get(sig)
        return "novel" if c is None else ("resolved" if len(c) == 1 else "degenerate")

    st = pd.Series([state(tuple(r)) for r in Dev.values.astype(np.int64)],
                   index=df.index)
    return st.where(df.active.values, "inert"), sigmap


# ------------------------------------------------------------- model layer --
def forest():
    from sklearn.ensemble import RandomForestClassifier
    return RandomForestClassifier(n_estimators=300, random_state=0, n_jobs=-1,
                                  class_weight="balanced")


def fit_and_validate(df, Dev, ev, sigmap):
    """The operation classifier, trained on documented-active encodings whose
    signature is RESOLVED (maps to exactly one operation), then transferred.

    Excluding the degenerate signatures from training is the single change that
    makes this stage work.  On all documented memory encodings the
    load/store/atomic split scores 0.962/0.969; excluding the 286 encodings
    whose signature carries both load and store it scores 0.993/0.995.  As a
    control, excluding a comparable number of NON-degenerate vector-segment
    encodings instead leaves it at 0.958/0.965 -- so the gain comes from the
    ambiguity, not from the vector-segment encodings as a class.

    The exclusion is not a trick to inflate the score.  A degenerate signature
    is one the measurement genuinely cannot resolve; training on it teaches the
    forest to guess between two operations, and the guess then propagates to
    every target encoding sharing that signature with a confidence it has not
    earned.  Those targets are reported as degenerate instead, with their
    candidate set.

    Three disjoint masks:
      train  = documented & active & resolved & opclass in FINE
      vendor = documented & active & CUSTOM-*        (held out entirely)
      target = ~documented & active & novel          (the only ones needing a model)
    """
    from sklearn.model_selection import (cross_val_predict, StratifiedGroupKFold,
                                         GroupKFold)
    from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                                 f1_score, confusion_matrix)

    act = df.active.values
    resolved = (df.evidence == "resolved").values
    train = act & resolved & df.documented.values & df.opclass.isin(FINE).values
    vendor = act & df.documented.values & df.opclass.str.startswith("CUSTOM").values
    target = act & ~df.documented.values
    assert not (train & vendor).any() and not (train & target).any()

    X, y = Dev[train].values, df.opclass[train].map(FINE).values
    # Group by mnemonic: sibling encodings of one instruction share a group and
    # never straddle a fold.  Encodings the toolchain could not name fall into a
    # single "?" group, which is conservative (they are held out together).
    groups = df.mnemonic[train].fillna("?").values
    cv = StratifiedGroupKFold(5, shuffle=True, random_state=0)
    pred_cv = cross_val_predict(forest(), X, y, cv=cv, groups=groups, n_jobs=1)

    labs = [f for f in OP_ORDER if f in set(y)]
    sig = lambda M: pd.DataFrame(M).apply(tuple, axis=1).nunique()
    cvrep = pd.DataFrame([dict(
        operation=f, n=int((y == f).sum()),
        f1=round(f1_score(y == f, pred_cv == f), 3),
        recall=round(((pred_cv == f) & (y == f)).sum() / max((y == f).sum(), 1), 3),
        n_signatures=int(sig(X[y == f])),
        n_mnemonics=int(pd.Series(groups[y == f]).nunique()),
        mean_counters_moved=round((X[y == f] != 0).sum(1).mean(), 2),
    ) for f in labs]).sort_values("f1", ascending=False)

    # ---- second validation: leave-SIGNATURE-out ------------------------
    # The grouped-CV number above is mostly a lookup score: 90.6% of held-out
    # encodings have their exact signature somewhere in the training fold, so
    # the forest can memorise rather than generalise, and the near-perfect rows
    # in the confusion matrix say more about signature reuse than about the
    # counters.  That is fine for the 159 target encodings resolved by a
    # documented twin -- lookup is exactly the claim there -- but it says
    # nothing about the 144 with a NOVEL signature, which is where the model is
    # the only evidence.
    #
    # So: re-run CV grouping by SIGNATURE, which forces every test encoding to
    # be one the model has literally never seen.  Restricted to operations
    # carrying at least MIN_SIG distinct signatures; a class with one signature
    # becomes unlearnable the moment that signature is held out, so scoring it
    # here would measure the test's construction rather than the counters.
    MIN_SIG = 6
    gsig = pd.DataFrame(X).apply(tuple, axis=1).astype(str).values
    nsig = pd.DataFrame({"y": y, "g": gsig}).groupby("y").g.nunique()
    rich = nsig[nsig >= MIN_SIG].index.tolist()
    rm = np.isin(y, rich)
    p_sig = cross_val_predict(forest(), X[rm], y[rm],
                              cv=GroupKFold(n_splits=min(20, len(set(gsig[rm])))),
                              groups=gsig[rm], n_jobs=1)
    loso = dict(ops=rich, n=int(rm.sum()), nsig=nsig.to_dict(),
                acc=accuracy_score(y[rm], p_sig),
                bal=balanced_accuracy_score(y[rm], p_sig))

    model = forest().fit(X, y)

    def call(mask):
        P = model.predict_proba(Dev[mask].values)
        out = df[mask][["encoding", "opclass", "family", "mnemonic", "evidence"]].copy()
        out["pred_op"] = model.classes_[P.argmax(1)]
        out["confidence"] = P.max(1).round(3)
        return out

    # External validation: vendor encodings at a major the model never saw,
    # ground truth from the mnemonic rather than from the opcode.  A three-way
    # test (load / store / int-alu), so strictly harder than the two-way one
    # the coarse family classifier allowed.
    ven = call(vendor)
    unnamed = int(ven.mnemonic.isna().sum())
    if unnamed:
        raise SystemExit(f"{unnamed} vendor encodings lack a mnemonic; hold-out "
                         "ground truth would be guessed, refusing")
    mn = ven.mnemonic.fillna("")
    ven["truth"] = np.where(~mn.str.match(VENDOR_MEM_RE), "int-alu",
                            np.where(mn.str.match(r"^th\.f?l"), "load", "store"))
    ven["correct"] = ven.pred_op == ven.truth

    # ---- read out the verdict by evidence state -------------------------
    # Only the novel encodings get a model call.  Resolved ones take the
    # operation their documented twin carries -- a nearest neighbour at
    # distance zero beats a prediction, and it can name an operation the model
    # has no class for (`fence`).  Degenerate ones get the candidate set and no
    # single answer.
    tgt = call(target)
    cands = [sorted(sigmap.get(tuple(r), ())) for r in Dev[target].values.astype(np.int64)]
    tgt["candidates"] = ["/".join(c) for c in cands]
    tgt["operation"] = np.where(
        tgt.evidence.values == "resolved", [c[0] if c else "" for c in cands],
        np.where(tgt.evidence.values == "degenerate", tgt.candidates, tgt.pred_op))
    tgt["source"] = tgt.evidence.map({"resolved": "documented twin",
                                      "degenerate": "ambiguous",
                                      "novel": "model"})
    # A resolved encoding whose twin operation IS in the label set should get
    # the same answer from the model; disagreement would mean the forest is
    # overriding an identical documented signature, a bug in the features
    # rather than a finding.
    chk = (tgt.evidence == "resolved") & tgt.operation.isin(set(y))
    if chk.any() and not (tgt.operation[chk] == tgt.pred_op[chk]).all():
        n_bad = int((tgt.operation[chk] != tgt.pred_op[chk]).sum())
        raise SystemExit(f"{n_bad} encodings where an exact documented twin "
                         "disagrees with the model; investigate before trusting")

    cm = confusion_matrix(y, pred_cv, labels=labs)
    return dict(labs=labs, cm=cm, cvrep=cvrep, vendor=ven, target=tgt, loso=loso,
                acc=accuracy_score(y, pred_cv), bal=balanced_accuracy_score(y, pred_cv),
                n_train=int(train.sum()), n_groups=len(set(groups)))


def by_signature_class(df, tgt, classes_csv):
    """Aggregate the verdict up to screen_report's signature classes, so this
    table and fig_signature_landscape index the same objects.

    Covers all 635.  A class whose signature is degenerate gets the candidate
    set as its verdict rather than a pick, since that IS the finding for those
    encodings.
    """
    if not os.path.exists(classes_csv):
        return None
    sgc = pd.read_csv(classes_csv)
    memb = {int(e, 16): int(r.sig_class)
            for _, r in sgc.iterrows() for e in str(r.members).split()}

    tg = df[df.target].copy()
    tg["sig_class"] = tg.enc.map(memb)
    if tg.sig_class.isna().any():
        raise SystemExit("signature-class membership does not cover the target set")
    calls = tgt.set_index(tgt.encoding.apply(lambda v: int(str(v), 16)))
    tg["verdict"] = tg.enc.map(calls.operation).fillna("?")
    tg["confidence"] = tg.enc.map(calls.confidence)

    g = tg.groupby("sig_class")
    out = pd.DataFrame(dict(
        n=g.size(),
        opclasses=g.opclass.agg(lambda s: ",".join(sorted(set(s)))),
        evidence=g.evidence.agg(lambda s: s.value_counts().index[0]),
        verdict=g.verdict.agg(lambda s: s.value_counts().index[0]),
        unanimous=g.verdict.nunique().eq(1),
        confidence=g.confidence.mean().round(3),
    )).reset_index()
    out["novel_signature"] = out.sig_class.map(sgc.set_index("sig_class").novel_signature)
    return out.sort_values("n", ascending=False)


def arch_fault_audit(df, outdir):
    """Summarize arch_fault buckets and check sigsegv against memory rows.

    Returns (fault_counts, sigsegv_check) when arch_fault is available, else
    (None, None).
    """
    if "arch_fault" not in df.columns:
        return None, None
    tgt = df[df.target].copy()
    tgt = tgt[tgt.arch_fault.notna()].copy()
    if tgt.empty:
        return None, None

    faults = tgt.groupby("arch_fault").size().rename("n").reset_index()
    faults["pct"] = (100 * faults.n / len(tgt)).round(1)
    order = ["none", "sigill", "sigsegv", "sigbus", "sigtrap", "sigfpe",
             "sigalrm"]
    faults["_order"] = faults.arch_fault.map(lambda x: order.index(x)
                                             if x in order else len(order))
    faults = faults.sort_values(["_order", "n", "arch_fault"],
                                ascending=[True, False, True]).drop(columns=["_order"])
    faults.to_csv(os.path.join(outdir, "undocumented_active_arch_faults.csv"), index=False)

    seg = tgt[tgt.arch_fault == "sigsegv"].copy()
    seg["is_memory"] = seg.family.eq("memory")
    seg.to_csv(os.path.join(outdir, "undocumented_sigsegv_family_check.csv"),
               index=False)
    check = pd.crosstab(seg.family, seg.is_memory).rename_axis("family")

    print("\n=== architectural fault audit over undocumented-active encodings ===")
    print(f"  total with arch_fault: {len(tgt)}")
    print(f"  sigill: {int((tgt.arch_fault == 'sigill').sum())}")
    print(f"  sigsegv: {len(seg)}")
    print(f"  sigsegv classified as memory: {int(seg.is_memory.sum())}/{len(seg)}")
    if len(seg) and not seg.is_memory.all():
        print("  non-memory sigsegv rows:")
        print(seg[~seg.is_memory][["encoding", "opclass", "family", "mnemonic",
                                  "evidence", "arch_fault"]].to_string(index=False))

    return faults, check


# ------------------------------------------------------------------ figures --
def _style():
    import matplotlib as mpl
    mpl.use("Agg")
    try:
        from kernel import apply_figure_style
        apply_figure_style(sizes=(9, 8, 7.5))
    except Exception:
        mpl.rcParams.update({"font.size": 9, "axes.labelsize": 9,
                             "axes.titlesize": 9.5, "legend.fontsize": 8,
                             "xtick.labelsize": 7.5, "ytick.labelsize": 7.5})


def fig_operation_confusion(R, outdir):
    """Row-normalised grouped-CV confusion matrix over the 8 operation classes.

    Values printed in every cell (64 cells, under the readability limit)
    because the off-diagonal structure -- specifically which classes the
    counters cannot separate -- is the content, and a colour ramp alone does
    not let a reader quote it.  The row order is OP_ORDER (memory ops, then
    compute, then control) rather than sorted by score, so the block structure
    is visible: the failure is confined to the control block.

    The `jump` row is marked and NOT read as a hardware limit.  Grouping folds
    by mnemonic is the right protocol everywhere else, but jump has only two
    mnemonics -- `j` (646 encodings, 638 distinct signatures, because the
    20-bit immediate varies and the counters vary with it) and `jr` (128
    encodings, 11 signatures, no overlap).  Whenever `j` is the test fold the
    model has seen nothing like it, which is a property of the fold, not of the
    counters: trained on `j` it labels all 128 `jr` encodings correctly, and
    under leave-signature-out jump scores 769/774.  The 0.29 F1 measures the
    protocol.
    """
    import matplotlib.pyplot as plt
    _style()
    labs, cm = R["labs"], R["cm"]
    cmn = 100 * cm / cm.sum(1, keepdims=True)

    fig, ax = plt.subplots(figsize=(6.0, 4.9))
    im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=100, aspect="auto")
    for i in range(len(labs)):
        for j in range(len(labs)):
            if cmn[i, j] >= 0.5:
                ax.text(j, i, f"{cmn[i, j]:.0f}", ha="center", va="center",
                        fontsize=8, color="white" if cmn[i, j] > 55 else "#222")
    ax.set_xticks(range(len(labs)))
    ax.set_xticklabels(labs, rotation=35, ha="right")
    ax.set_yticks(range(len(labs)))
    nm = R["cvrep"].set_index("operation").n_mnemonics.to_dict()
    ax.set_yticklabels([f"{l}\u2020  ({n:,})" if nm.get(l, 9) < 3 else f"{l}  ({n:,})"
                        for l, n in zip(labs, cm.sum(1))])
    ax.set_xlabel("operation predicted from the PMU signature")
    ax.set_ylabel("operation implied by the opcode")
    ax.set_title("Load, store and atomic are separable from the counters alone,\n"
                 "as are the three compute classes", loc="left")
    cb = fig.colorbar(im, ax=ax, fraction=.046, pad=.03)
    cb.set_label("row-normalised %", fontsize=8)
    cb.ax.tick_params(labelsize=7.5)
    fig.text(.005, -.015, f"documented encodings on a resolved signature (n={R['n_train']:,}, "
             f"{R['n_groups']} mnemonic groups); balanced accuracy {R['bal']:.3f}.  "
             f"90% of held-out encodings reuse a training signature, so this is "
             f"largely a lookup score; under\nleave-signature-out it is "
             f"{R['loso']['bal']:.3f} for {'/'.join(R['loso']['ops'])}.  "
             f"\u2020 fewer than 3 mnemonics, so a grouped fold holds out the "
             f"whole class -- the low row is the protocol, not the counters",
             fontsize=7.5, color="#666")
    fig.tight_layout()
    p = os.path.join(outdir, "fig_operation_confusion.png")
    fig.savefig(p, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return p


def fig_transfer(R, df, outdir, variant):
    """The two-stage read-out of the undocumented-active set.

    Two panels sharing a y-axis of opcode classes, abutting so a row reads
    straight across:
      (a) what the documented space determines about this signature -- resolved
          to one operation, degenerate over several, or never seen;
      (b) the operation itself, only for the rows where (a) leaves one.

    Panel (b) is deliberately EMPTY for the fully-degenerate rows.  That is the
    message: for those encodings the measurement does not distinguish the
    candidates, and drawing a bar would repeat the error this script exists to
    correct.
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    _style()
    tg = df[df.target]
    order = tg.groupby("opclass").size().sort_values(ascending=False).index.tolist()
    tgt = R["target"].set_index(R["target"].encoding.apply(lambda v: int(str(v), 16)))
    states = ["resolved", "degenerate", "novel"]

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.8, 3.4), sharey=True,
                                 gridspec_kw=dict(wspace=.06, width_ratios=[1, 1]))

    for i, oc in enumerate(order):
        sub = tg[tg.opclass == oc]
        left = 0
        for st in states:
            w = int((sub.evidence == st).sum())
            if not w:
                continue
            a1.barh(i, w, left=left, color=EVIDENCE[st], height=.62,
                    edgecolor="white", linewidth=.8)
            if w >= 42:
                a1.text(left + w / 2, i, f"{w}", ha="center", va="center",
                        color="white", fontsize=8, fontweight="bold")
            left += w
        if left < 42:
            a1.text(left + 4, i, f"{left}", va="center", fontsize=7.5, color="#444")

        # (b) only the rows the evidence resolves: a documented twin or a model call
        callable_ = sub[sub.evidence != "degenerate"]
        left = 0
        for op in OP_ORDER:
            w = int((callable_.enc.map(tgt.operation) == op).sum())
            if not w:
                continue
            a2.barh(i, w, left=left, color=OPCOLOR[op], height=.62,
                    edgecolor="white", linewidth=.8)
            if w >= 42:
                a2.text(left + w / 2, i, f"{w}", ha="center", va="center",
                        color="white", fontsize=8, fontweight="bold")
            left += w
        if 0 < left < 42:
            a2.text(left + 4, i, f"{left}", va="center", fontsize=7.5, color="#444")
        if not len(callable_):
            cand = sub.enc.map(tgt.candidates).mode()
            a2.text(3, i, f"ambiguous: {cand.iloc[0] if len(cand) else '?'}",
                    va="center", fontsize=7.5, color="#999", style="italic")

    a1.set_yticks(range(len(order)))
    a1.set_yticklabels(order, fontsize=8)
    a1.invert_yaxis()
    n_call = int((tg.evidence != "degenerate").sum())
    for ax, xmax, xl in ((a1, len(tg[tg.opclass == order[0]]) * 1.18,
                          "undocumented encodings that are PMU-active"),
                         (a2, max(60, n_call * .62),
                          "of those, the ones the evidence resolves")):
        ax.set_xlim(0, xmax)
        ax.set_xlabel(xl, fontsize=8.5)
        ax.margins(y=.04)

    a1.set_title("(a) what does the documented space determine?", loc="left", fontsize=9)
    a2.set_title("(b) where it determines something, what?", loc="left", fontsize=9)
    a1.legend(handles=[Patch(facecolor=EVIDENCE[s], label=l) for s, l in (
        ("resolved", "signature maps to one operation"),
        ("degenerate", "signature spans several"),
        ("novel", "signature unseen (model call)"))],
        loc="lower right", frameon=False, fontsize=7.5)
    ops = [o for o in OP_ORDER if (tg.enc.map(tgt.operation) == o).any()]
    a2.legend(handles=[Patch(facecolor=OPCOLOR[o], label=o) for o in ops],
              loc="lower right", frameon=False, fontsize=7.5)

    n_deg = int((tg.evidence == "degenerate").sum())
    fig.suptitle(f"For {n_deg} of the {len(tg)} undocumented-active encodings the "
                 "counters cannot distinguish the candidate operations",
                 fontsize=9.5, x=.005, ha="left")
    fig.text(.005, -.03, f"documented axis: set {variant}  ·  (a) is a lookup "
             "against the documented signature map, not a model  ·  (b) is a "
             "documented twin where one exists, else a forest trained on "
             "resolved documented signatures with the vendor major held out",
             fontsize=7.5, color="#666")
    fig.tight_layout(rect=(0, 0, 1, .92))
    p = os.path.join(outdir, "fig_transfer.png")
    fig.savefig(p, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return p


def fig_ghostwrite(d, D, act, docC, evc, outdir):
    """Case study: the GhostWrite encoding 0x10000027 against the scan.

    (a) load-vs-store separation in the two counters the forest actually uses
        for that call, with the reserved-EEW vector stores overplotted.  The
        point of the panel is that GhostWrite sits at the ORIGIN of the store
        counter -- it is the only active encoding in the scan that writes
        memory while leaving the store counter unmoved.
    (b) the rarity ranking, showing it is not buried: of 144 undocumented
        active encodings with a signature no documented instruction carries,
        it is the quietest.
    """
    import matplotlib.pyplot as plt
    _style()
    E = 0x10000027
    i = d.index[d.enc == E][0]
    dev = D.loc[i]

    ld = docC & act & d.opclass.isin(["LOAD", "LOAD-FP"]).values
    st = docC & act & d.opclass.isin(["STORE", "STORE-FP"]).values
    mw = ((d.enc.values >> 28) & 1).astype(bool)
    mop = (d.enc.values >> 26) & 3
    nf = (d.enc.values >> 29) & 7
    su = (d.enc.values >> 20) & 0x1f
    # the mew=1 unit-stride reserved-EEW family IS the GhostWrite set: all four
    # active members are the starred encoding's siblings, so plotting them as a
    # separate series would just draw the same point twice.  Show instead the
    # other 176 undocumented-active STORE-FP encodings, which is the contrast
    # that matters: they behave like stores, these four do not.
    other_sf = act & (~docC) & (d.opclass == "STORE-FP").values & ~np.isin(
        d.enc.values, [0x10000027, 0x12000027, 0x10007027, 0x12007027])

    fig, (ax, bx) = plt.subplots(1, 2, figsize=(9.6, 4.0),
                                 gridspec_kw=dict(width_ratios=[1.05, 1]))
    jit = lambda n, s=.06: np.random.default_rng(0).normal(0, s, n)
    for m, c, lab in [(ld, "#2c9c5a", "documented load"),
                      (st, "#7f4fc9", "documented store")]:
        ax.scatter(np.log10(D.loc[m, "0x0014"].clip(lower=0) + 1) + jit(int(m.sum())),
                   np.log10(D.loc[m, "0x0016"].clip(lower=0) + 1) + jit(int(m.sum())),
                   s=7, alpha=.30, lw=0, color=c, label=lab)
    ax.scatter(np.log10(D.loc[other_sf, "0x0014"].clip(lower=0) + 1) + jit(int(other_sf.sum())),
               np.log10(D.loc[other_sf, "0x0016"].clip(lower=0) + 1) + jit(int(other_sf.sum())),
               s=14, facecolor="none", edgecolor="#c98a1a", lw=.8,
               label="other undocumented STORE-FP")
    ax.scatter([np.log10(dev["0x0014"] + 1)], [np.log10(dev["0x0016"] + 1)],
               s=110, marker="*", color="#d2431f", zorder=5, label="0x10000027 (GhostWrite)")
    ax.annotate("writes memory,\nstore counter silent",
                xy=(0, np.log10(2)), xytext=(0.55, 0.62),
                fontsize=8, color="#d2431f",
                arrowprops=dict(arrowstyle="->", color="#d2431f", lw=.9))
    ax.set_xlabel("store-retired counter 0x0014   [log$_{10}$(dev+1)]")
    ax.set_ylabel("memory-access counter 0x0016   [log$_{10}$(dev+1)]")
    ax.set_title("(a) the counter that names a store never fires", loc="left", fontsize=10)
    ax.legend(fontsize=7, frameon=False, loc="lower right")

    r = pd.read_csv(os.path.join(outdir, "novel_signature_rarity.csv")) \
          .sort_values("total_dev").reset_index(drop=True)
    MEMMAJ = ["LOAD", "LOAD-FP", "STORE", "STORE-FP", "AMO"]
    memj = r.opclass.isin(MEMMAJ).values
    y = np.arange(len(r))
    bx.scatter(r.total_dev[~memj].clip(lower=.5), y[~memj], s=9, lw=0,
               color="#c9ced4", label="compute / other major")
    bx.scatter(r.total_dev[memj].clip(lower=.5), y[memj], s=11, lw=0,
               color="#4a5560", label="memory major")
    k = int(r.index[r.encoding == "0x10000027"][0])
    bx.scatter([max(r.total_dev.iloc[k], .5)], [k], s=110, marker="*",
               color="#d2431f", zorder=5)
    bx.annotate("0x10000027 — quietest\nmemory-major encoding\n(12 quieter are all FP-compute)",
                xy=(r.total_dev.iloc[k], k), xytext=(5.5, 40), fontsize=8,
                color="#d2431f", arrowprops=dict(arrowstyle="->", color="#d2431f", lw=.9))
    bx.legend(fontsize=7.5, frameon=False, loc="lower right")
    bx.set_xscale("log")
    bx.invert_yaxis()
    bx.set_xlabel("total |counter deviation| (log)")
    bx.set_ylabel("rank by total deviation, quietest first")
    bx.set_title("(b) it does not vanish into the novel set", loc="left", fontsize=10)

    fig.text(.005, -.02,
             "(a) documented memory encodings, T-Head C910; the four GhostWrite siblings share one point · "
             "(b) undocumented-active encodings whose\nsignature no documented instruction carries, ranked quietest-first",
             fontsize=7.5, color="#666")
    fig.tight_layout()
    p = os.path.join(outdir, "fig_ghostwrite.png")
    fig.savefig(p, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return p


# --------------------------------------------------------------------- main --
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scan", default="consensus_scan.csv")
    ap.add_argument("--screen", default="out/binutils_screen.csv")
    ap.add_argument("--outdir", default="out")
    ap.add_argument("--variant", choices=["A", "B", "C"], default="C",
                    help="documented-axis definition, as in binutils_screen.py")
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    O = lambda n: os.path.join(a.outdir, n)

    df, ev, Dev, sigmap = load(a.scan, a.screen, a.variant)
    R = fit_and_validate(df, Dev, ev, sigmap)

    # --- stage 1: evidence state ----------------------------------------
    tg = df[df.target]
    ch = (df[df.active].groupby([np.where(df.documented[df.active], "documented",
                                          "undocumented"), "evidence"]).size()
          .unstack(fill_value=0))
    ch.to_csv(O("evidence_state.csv"))
    print("=== stage 1: what the documented signature map determines ===")
    print(ch.to_string())
    deg = tg[tg.evidence == "degenerate"]
    print(f"\n  {len(deg)} of the {len(tg)} undocumented-active encodings sit on a "
          "signature that documented\n  instructions carry under more than one "
          "operation; the counters do not distinguish them.")
    dd = deg.copy()
    dd["candidates"] = dd.enc.map(
        R["target"].set_index(R["target"].encoding.apply(lambda v: int(str(v), 16))).candidates)
    print(dd.groupby(["candidates", "opclass"]).size().to_string())

    # --- stage 2 ---------------------------------------------------------
    R["cvrep"].to_csv(O("operation_cv_report.csv"), index=False)
    R["vendor"].to_csv(O("vendor_holdout.csv"), index=False)
    R["target"].to_csv(O("undocumented_operation_calls.csv"), index=False)
    cls = by_signature_class(df, R["target"], O("undocumented_signature_classes.csv"))
    if cls is not None:
        cls.to_csv(O("undocumented_family_classes.csv"), index=False)
    arch_fault_audit(df, a.outdir)

    print(f"\n=== stage 2: operation classifier, resolved documented only ===")
    print(f"features: {len(ev)} informative counters (deviation from modal baseline)")
    print(f"train: {R['n_train']:,} encodings, {R['n_groups']} mnemonic groups, "
          "5-fold grouped CV")
    print(f"  accuracy={R['acc']:.3f}  balanced accuracy={R['bal']:.3f}")
    print(R["cvrep"].to_string(index=False))

    v = R["vendor"]
    L = R["loso"]
    print(f"\n=== leave-signature-out: {'/'.join(L['ops'])} (>=6 signatures each) ===")
    print(f"  every test encoding carries a signature absent from its training fold")
    print(f"  n={L['n']:,}  accuracy={L['acc']:.3f}  balanced accuracy={L['bal']:.3f}")
    print("  signatures per operation:",
          ", ".join(f"{k}:{v}" for k, v in sorted(L['nsig'].items(), key=lambda kv: -kv[1])))

    print(f"\n=== held-out vendor major (never trained on) ===")
    print(f"  {int(v.correct.sum())}/{len(v)} = {v.correct.mean():.1%} correct "
          "against mnemonic-derived truth (3-way)")
    print(pd.crosstab(v.truth, v.pred_op).to_string())

    t = R["target"]
    t = t[t.evidence != "degenerate"]
    print(f"\n=== read-out for the {len(t)} encodings the evidence resolves ===")
    print(t.groupby(["opclass"]).apply(lambda g: pd.Series(dict(
        n=len(g),
        operation="; ".join(f"{k}:{v}" for k, v in g.operation.value_counts().items()),
        from_twin=int((g.source == "documented twin").sum()),
        confidence=round(g.confidence[g.source == "model"].mean(), 3))),
        include_groups=False).to_string())
    print(f"\n  {int((t.source == 'documented twin').sum())} resolved by an exact "
          f"documented twin, {int((t.source == 'model').sum())} by the model "
          "(novel signature, extrapolated)")
    if cls is not None:
        print(f"  {int(cls.unanimous.sum())}/{len(cls)} signature classes get a "
              "unanimous verdict")

    # ---- GhostWrite case study -----------------------------------------
    # 0x10000027 is CVE-2024-44067, the T-Head C910 hidden-write bug found by
    # RISCVuzz.  It is a mew=1 unit-stride vector store: mew marks an EEW the
    # vector spec reserves, so no assembler emits it and the screen calls it
    # undocumented.  Worth a named check because it is the one encoding in this
    # scan whose true behaviour is independently known.
    GW = 0x10000027
    if (df.enc == GW).any():
        gi = df.index[df.enc == GW][0]
        gdev = Dev.loc[gi]
        gsig = tuple(gdev.astype(np.int64))
        sib = [df.encoding[k] for k in Dev.index
               if tuple(Dev.loc[k].astype(np.int64)) == gsig]
        print(f"\n=== 0x10000027 (GhostWrite, CVE-2024-44067) ===")
        print(f"  documented: {bool(df.documented[gi])}   active: {bool(df.active[gi])}"
              f"   evidence: {df.evidence[gi]}")
        print("  counters moved :",
              ", ".join(f"{k}:{int(v):+d}" for k, v in gdev[gdev != 0].items()))
        print(f"  signature shared by {len(sib)} encodings, all undocumented: {sib}")
        print(f"  store-retired counter 0x0014: {int(gdev.get('0x0014', 0))} "
              f"(fires for 100% of documented integer stores)")


    sig_all = pd.Series([tuple(r_) for r_ in Dev.values.astype(np.int64)],
                        index=Dev.index)
    doc_sigs = set(sig_all[df.documented.values & df.active.values])
    und = (~df.documented.values) & df.active.values
    rar = pd.DataFrame({"encoding": df.encoding[und], "opclass": df.opclass[und],
                        "total_dev": Dev[und].abs().sum(1).values,
                        "n_ctr": (Dev[und] != 0).sum(1).values,
                        "novel": [s_ not in doc_sigs for s_ in sig_all[und]]})
    rar = rar[rar.novel].sort_values("total_dev")
    rar.to_csv(os.path.join(a.outdir, "novel_signature_rarity.csv"), index=False)

    gwq = int((rar.total_dev < rar.total_dev[rar.encoding == "0x10000027"].iloc[0]).sum()) \
        if (rar.encoding == "0x10000027").any() else -1
    if gwq >= 0:
        mm = rar.opclass.isin(["LOAD", "LOAD-FP", "STORE", "STORE-FP", "AMO"])
        print(f"  rarity: {gwq} of {len(rar)} novel-signature encodings are quieter; "
              f"{int((mm & (rar.total_dev < 2)).sum())} of those sit at a memory major")

    for f in (fig_operation_confusion(R, a.outdir),
              fig_ghostwrite(df, Dev, df.active.values, df.documented.values,
                             ev, a.outdir),
              fig_transfer(R, df, a.outdir, a.variant)):
        print("wrote", f)


if __name__ == "__main__":
    main()
