#!/usr/bin/env python3
"""
triage.py -- a generic, model-free triage for undocumented PMU-active encodings.

The problem this solves.  `signature_classify.py` asks "what operation is this?"
and answers with a forest.  That is the wrong first question for a bug hunt: the
forest always answers, it answers with a label from a fixed set, and its answer
is least trustworthy exactly where the encoding is most interesting -- because
"interesting" means "unlike anything documented", which is also the definition of
"outside the training distribution".  GhostWrite (0x10000027) is the worked
example: the forest calls it a load, confidently, because the counter that
defines a store is the one it defeats.

So this module asks a different question, one that needs no model and no label
set: *does this encoding behave like the documented encodings that share its
major opcode?*  Everything here is a comparison against a reference set drawn
from the scan itself, so every verdict is checkable by hand.

Three independent views, each answering a question the others cannot:

  1. INVARIANTS (mine_invariants / check_violations)
     For each major opcode, which counters fire for *every* documented-active
     encoding there, and which fire for *none*?  Those are behavioural contracts
     the hardware appears to assert for the class.  An undocumented encoding at
     the same major that is silent on a required counter, or fires a forbidden
     one, is doing something no documented member of its class does.  This is
     the sharpest signal in the module and it is a pure set operation.

  2. NOVELTY (novelty_score)
     Nearest-neighbour distance in log-counter space to the closest *distinct*
     documented signature at the same major, scaled by how far apart the
     documented signatures at that major already are.  Answers "how far outside
     the documented cloud is this, in units of the cloud's own spread?" --
     a magnitude question the invariant view cannot express.

  3. TWINS (twin_table)
     Exact signature matches against documented-active encodings.  A twin is a
     nearest neighbour at distance zero, so it names behaviour with no model at
     all.  A twin whose documented members disagree about the operation is
     equally informative in the negative direction: the PMU cannot separate
     those operations, so no classifier built on these counters ever will.

The three combine into four tiers (assign_tiers).  The tiers are ordered by how
much attention an encoding deserves, not by confidence in any label:

  A  breaks a class invariant           -- behaves unlike its documented class
  B  out of documented range, no twin   -- novel magnitude, nothing to compare to
  C  novel signature, within range      -- unseen but unremarkable
  D  exact documented twin              -- behaviour is already known

On the T-Head C920 scan this puts the four GhostWrite encodings alone in tier B,
and they are the only undocumented-active encodings in the whole scan that are
simultaneously (a) without any documented twin and (b) outside the documented
range of their own major.  That is not a tuned result -- no threshold in this
module was chosen with GhostWrite in view -- and it is the argument that the
method finds real bugs without being told what it is looking for.

A caveat that matters for reading the output.  The scan varies only the opcode,
funct3 and funct7 fields; rd, rs1 and rs2 are held at a fixed value throughout.
So "which fields differ inside a signature class" (field_variation) can only
ever answer with those three, and the question "is this the same instruction
with different register arguments?" is not answerable from this scan at all --
it would need a second scan that sweeps the register fields.
"""

from __future__ import annotations

import re
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors


# Fields the scan actually varies.  Register fields are listed so that
# field_variation() can report -- correctly -- that they never differ.
FIELDS = [("opcode", 0, 7), ("rd/vd", 7, 12), ("funct3", 12, 15),
          ("rs1", 15, 20), ("rs2/vs2", 20, 25), ("funct7", 25, 32)]

# The vector load/store layout, for encodings at the LOAD-FP / STORE-FP majors.
#
# Bit 28 is split out from `mop` here rather than folded into it.  Under RVV
# 1.0 that bit is `mew` and `mop` is the two bits 27:26; under XTheadVector,
# which is what this core implements, the addressing-mode field is the three
# bits 28:26 and bit 28 is its high bit, allocated at LOAD-FP (th.vlbu.v vs
# th.vlb.v) and, for unit-stride and strided stores, unallocated at STORE-FP.
# The split naming is kept because it is how the field is reported throughout,
# but "mew" is the RVV-1.0 name and does not describe this core.  Read the
# pair (mew, mop) as the single 3-bit mode field when interpreting output.
VFIELDS = [("opcode", 0, 7), ("vd", 7, 12), ("width", 12, 15), ("rs1", 15, 20),
           ("lsumop", 20, 25), ("vm", 25, 26), ("mop", 26, 28),
           ("mew", 28, 29), ("nf", 29, 32)]

TIER_A = "A: breaks a class invariant"
TIER_B = "B: out of documented range"
TIER_C = "C: novel, in range"
TIER_D = "D: matches a documented signature"
TIER_ORDER = [TIER_A, TIER_B, TIER_C, TIER_D]


# ------------------------------------------------------------------ encoding --

def field_variation(encodings, fields=FIELDS):
    """Which instruction fields differ across a set of encodings?

    Returns (field_names, xor_mask).  The XOR of every encoding against the
    first gives the bits that are not constant across the set; a field is
    reported when any of its bits appears there.
    """
    a = np.asarray(list(encodings), dtype=np.int64)
    varying = int(np.bitwise_or.reduce(a ^ a[0]))
    names = [nm for nm, lo, hi in fields
             if varying & (((1 << (hi - lo)) - 1) << lo)]
    return names, varying


def signature_series(D, ev=None):
    """Row-wise counter signature as a hashable tuple, for exact matching."""
    cols = ev if ev is not None else list(D.columns)
    return pd.Series([tuple(r) for r in D[cols].values.astype(np.int64)],
                     index=D.index)


def signature_class_fields(d, sig, active=None, fields=FIELDS):
    """For every signature shared by more than one encoding, report which
    instruction fields vary inside it.

    This is the "are these the same instruction with different arguments?"
    question, as far as the scan can answer it.  Since the scan holds the
    register fields fixed, a class can only differ in opcode/funct3/funct7 --
    so a class that differs in funct3 alone is genuinely one operation whose
    width or rounding-mode field does not reach the counters, whereas a class
    spanning several opcodes is a signature collision between distinct
    instructions.
    """
    m = np.ones(len(d), bool) if active is None else np.asarray(active)
    g = d[m].groupby(sig[m].values)
    rows = []
    for s, grp in g:
        if len(grp) < 2:
            continue
        names, xor = field_variation(grp.enc)
        rows.append(dict(n=len(grp), fields=",".join(names) or "none",
                         n_bits=int(bin(xor).count("1")),
                         opclasses=";".join(sorted(set(grp.opclass))),
                         n_opclass=grp.opclass.nunique()))
    return pd.DataFrame(rows).sort_values("n", ascending=False)


# ---------------------------------------------------------------- invariants --

def mine_invariants(D, d, documented, active, ev, min_ref=8):
    """Per-major behavioural contracts, mined from documented-active encodings.

    `always` = counters that fire for every documented-active encoding at that
    major; `never` = counters that fire for none of them.  Majors with fewer
    than `min_ref` documented references are skipped -- a contract mined from
    three encodings is not a contract.
    """
    inv = {}
    for maj in sorted(set(d.opclass[documented & active])):
        ref = documented & active & (d.opclass == maj).values
        if ref.sum() < min_ref:
            continue
        nz = (D.loc[ref, ev] != 0)
        cols = np.array(ev)
        inv[maj] = dict(always=set(cols[nz.all().values]),
                        never=set(cols[(~nz).all().values]),
                        n=int(ref.sum()))
    return inv


def check_violations(D, d, documented, active, ev, invariants):
    """Flag undocumented-active encodings that break their major's contract.

    structural -- silent on a counter that always fires for the class, or
                  firing one that never does
    magnitude  -- fires only permitted counters, but at a value outside the
                  documented range for that counter at that major
    """
    out = []
    for maj, v in invariants.items():
        tgt = (~documented) & active & (d.opclass == maj).values
        if not tgt.any():
            continue
        ref = documented & active & (d.opclass == maj).values
        lo, hi = D.loc[ref, ev].min(), D.loc[ref, ev].max()
        for k in d.index[tgt]:
            r = D.loc[k, ev]
            miss = sorted(c for c in v["always"] if r[c] == 0)
            extra = sorted(c for c in v["never"] if r[c] != 0)
            oob = sorted(c for c in ev if r[c] != 0 and not (lo[c] <= r[c] <= hi[c]))
            out.append(dict(i=k, encoding=d.encoding[k], opclass=maj,
                            silent_on_required=";".join(miss),
                            fires_forbidden=";".join(extra),
                            out_of_range=";".join(oob),
                            n_struct=len(miss) + len(extra), n_oob=len(oob)))
    return pd.DataFrame(out).set_index("i")


# ------------------------------------------------------------------- novelty --

def novelty_score(D, d, documented, active, ev, min_ref=8):
    """Distance to the nearest documented signature at the same major, in units
    of the documented spread at that major.

    Counters are log-compressed before the distance: raw counts span three
    orders of magnitude across majors, and without compression the metric would
    only ever notice the loudest counter.  The reference set is *distinct*
    documented signatures -- duplicated signatures would otherwise drive the
    spacing estimate to zero and make every novelty score infinite.  Majors with
    too few documented siblings fall back to the whole documented-active set,
    which is recorded in `ref_set` so the fallback is never silent.
    """
    L = np.log1p(np.abs(D[ev].values)) * np.sign(D[ev].values)
    und = (~documented) & active
    out = []
    for maj in sorted(set(d.opclass[und])):
        tgt = und & (d.opclass == maj).values
        ref = documented & active & (d.opclass == maj).values
        src = "same major"
        if ref.sum() < min_ref:
            ref, src = documented & active, "all documented"
        R = np.unique(L[ref], axis=0)
        if len(R) < 2:
            continue
        dist = NearestNeighbors(n_neighbors=1).fit(R).kneighbors(L[tgt])[0].ravel()
        spacing = np.percentile(
            NearestNeighbors(n_neighbors=2).fit(R).kneighbors(R)[0][:, 1], 90)
        spacing = max(spacing, 1e-9)
        for k, dv in zip(d.index[tgt], dist):
            out.append(dict(i=k, encoding=d.encoding[k], opclass=maj,
                            nn_dist=dv, novelty=dv / spacing, ref_set=src,
                            n_ref_sig=len(R), doc_spacing=spacing))
    return pd.DataFrame(out).set_index("i")


# --------------------------------------------------------------------- twins --

def twin_table(d, sig, documented, active, subtype=None):
    """Exact documented twins for every encoding, with what they disagree about.

    `twin_opclass` listing more than one class is the useful negative result:
    those operations are indistinguishable to this PMU, so the ceiling on any
    classifier built from these counters is set here, not by the model.
    """
    lut = {}
    for k in d.index[documented & active]:
        lut.setdefault(sig[k], []).append(k)
    rows = []
    for k in d.index:
        ks = lut.get(sig[k], [])
        ops = sorted(set(d.opclass[ks])) if ks else []
        subs = sorted({x for x in subtype[ks].dropna()}) if (ks and subtype is not None) else []
        rows.append(dict(i=k, n_twin=len(ks), twin_opclass=";".join(ops),
                         twin_subtype=";".join(subs), twin_ambiguous=len(ops) > 1))
    return pd.DataFrame(rows).set_index("i")


# -------------------------------------------------------------------- tiers --

def assign_tiers(T):
    """Combine the three views into an attention ordering.

    Order matters: a structural violation outranks a magnitude one, and both
    outrank mere novelty.  Tier D is not a dismissal -- a documented twin is the
    strongest possible statement about behaviour -- it just means the encoding
    needs no further investigation to be understood.
    """
    return np.select(
        [T.n_struct.fillna(0) > 0,
         (T.n_oob.fillna(0) > 0) & ~T.has_doc_twin,
         ~T.has_doc_twin],
        [TIER_A, TIER_B, TIER_C], default=TIER_D)


def triage(D, d, documented, active, ev, subtype=None, min_ref=8):
    """Run all three views and assemble the triage table."""
    sig = signature_series(D, ev)
    inv = mine_invariants(D, d, documented, active, ev, min_ref)
    VI = check_violations(D, d, documented, active, ev, inv)
    NO = novelty_score(D, d, documented, active, ev, min_ref)
    TW = twin_table(d, sig, documented, active, subtype)

    T = NO.join(VI[["silent_on_required", "fires_forbidden", "out_of_range",
                    "n_struct", "n_oob"]])
    T = T.join(TW[["n_twin", "twin_opclass", "twin_subtype", "twin_ambiguous"]])
    T["has_doc_twin"] = T.n_twin > 0
    T["total_dev"] = D.loc[T.index].abs().sum(1).values
    T["n_ctr"] = (D.loc[T.index] != 0).sum(1).values
    T["tier"] = assign_tiers(T)
    return T.sort_values(["tier", "novelty"], ascending=[True, False]), inv


# ------------------------------------------------------------- event naming --

def load_event_names(path="pmu_events.json"):
    """Optional: map raw counter columns (e.g. '0x0016') to vendor event names.

    Not used to produce any result in the thesis.  No vendor event table was
    available for the C920, so every counter is referred to by event code
    throughout, and 16 of the 26 informative counters have no published
    identity at all.  The hook is kept because the invariant and novelty views
    are purely relational and gain nothing from names, so a table can be
    dropped in later without changing a single verdict.

    Returns empty maps when `path` does not exist, which is the normal case.
    """
    import json, os
    if not os.path.exists(path):
        return {}, {}
    j = json.load(open(path))
    if isinstance(j, list):   # raw vendor JSON
        n = {("0x%04x" % int(e["EventCode"], 16)): e["EventName"] for e in j}
        b = {("0x%04x" % int(e["EventCode"], 16)): e.get("BriefDescription", "") for e in j}
        return n, b
    return j.get("name", {}), j.get("desc", {})


def label_counters(cols, names=None):
    """'0x0016' -> '0x0016 SOME_EVENT_NAME'; unnamed counters pass through.

    With no event table present every column comes back as "(unnamed)", which
    is the honest rendering for this core and the one used in the thesis.
    """
    names = names or load_event_names()[0]
    return [f"{c} {names[c]}" if c in names else f"{c} (unnamed)" for c in cols]