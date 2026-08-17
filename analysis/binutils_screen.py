#!/usr/bin/env python3
"""
binutils_screen.py -- decode an instruction-scan encoding set with GNU binutils
and cross it against measured PMU activity.

Finds encodings that are ACTIVE on silicon (PMU counters deviate from the
inactive baseline) but that the toolchain does NOT recognise: encodings that
execute and do measurable work while belonging to no documented instruction.

The -march string IS the definition of "documented", so it is built from what
the silicon reports (`isa : rv64imafdcv`, mvendorid 0x5b7) plus the T-Head
vendor extensions, which no isa string can advertise. A march WIDER than the
hardware hides findings (unimplemented extensions' encodings get labelled
"documented"); a narrower one only creates visible, triageable false positives.

Rationale, caveats and the validation history are in METHODS.md.

OUTPUTS
-------
  out/binutils_decode.csv   every encoding: mnemonic (or null), recognised flag
  out/binutils_screen.csv   the above joined to PMU activity + the screen verdict
  out/binutils_hidden.csv   the headline set: active AND unrecognised

Usage:
    python3 binutils_screen.py --scan consensus_scan.csv
"""

import argparse
import os
import re
import shutil
import subprocess
import sys
import tempfile

import numpy as np

# Where to find the RISC-V binutils. Two layouts exist and they differ in the
# TOOL NAME, not just the directory: a target-private bindir holds plain `as` /
# `objdump`, while a shared bindir (Homebrew's /opt/homebrew/bin) holds
# `riscv64-elf-as`. Resolving only the directory and appending "as" finds the
# HOST assembler, or nothing. So carry the name prefix too.
#
# $RISCV_BINUTILS_PREFIX overrides. It may be either a directory or a full
# tool prefix ending in '-' (e.g. /opt/homebrew/bin/riscv64-elf-).
_TARGETS = ("riscv64-elf", "riscv64-unknown-elf", "riscv64-linux-gnu")


def _resolve_binutils():
    env = os.environ.get("RISCV_BINUTILS_PREFIX")
    if env:
        # Accept a directory (plain or prefixed names inside) or a tool prefix.
        if not env.endswith("-") and os.path.isdir(env):
            # Try TARGET-PREFIXED names first. A shared bindir often holds both
            # `riscv64-elf-as` and the host `as`; preferring the bare name
            # there silently selects the host assembler.
            for t in tuple(f"{t}-" for t in _TARGETS) + ("",):
                if os.path.exists(os.path.join(env, f"{t}as")):
                    return os.path.join(env, t)
            return os.path.join(env, "")
        return env
    for t in _TARGETS:                       # prefixed tool on $PATH
        p = shutil.which(f"{t}-as")
        if p:
            return p[: -len("as")]           # keep dir + 'riscv64-elf-'
    for d in ("/opt/homebrew/Cellar/riscv64-elf-binutils/2.47/riscv64-elf/bin",):
        if os.path.exists(os.path.join(d, "as")):
            return os.path.join(d, "")
    return ""


BINUTILS_PREFIX = _resolve_binutils()

MARCH_BASE = "rv64gc"

# Candidate extensions, probed against the installed `as` at runtime and
# accumulated greedily so that a conflicting one is skipped rather than
# aborting the whole string. NOTE `xtheadvector` conflicts with standard
# `v`/`zve32x` -- that IS the documented XTheadVector/V encoding conflict, so
# anything pulling in `v` (e.g. zvfbfmin) is necessarily excluded here.
# `xtheadzvlsseg` also conflicts; its segment loads are part of xtheadvector.
_CANDIDATE_EXTS = """
xtheadba xtheadbb xtheadbs xtheadcmo xtheadcondmov xtheadfmemidx xtheadfmv
xtheadint xtheadmac xtheadmemidx xtheadmempair xtheadsync xtheadvector
xtheadvdot xtheadzvamo xtheadzvlsseg
zba zbb zbc zbs zbkb zbkc zbkx zknd zkne zknh zksed zksh
zicsr zifencei zicond zfh zfa zfbfmin zicboz zicbom zicbop zawrs zacas zabha
svinval svpbmt svnapot smaia ssaia zihintpause zihintntl zimop zcmop
""".split()


def _march_ok(as_, march, s, o):
    return subprocess.run([as_, f"-march={march}", "-o", o, s],
                          capture_output=True, text=True).returncode == 0


def build_march_max(seed="rv64gc", verbose=False):
    """Greedily accumulate every candidate extension the installed `as` accepts
    on top of `seed`. An extension that conflicts with what is already in the
    string is SKIPPED, not fatal -- which is how the xtheadvector/v conflict is
    handled. Order matters: whatever is in `seed` wins the conflict.

    Returns (march_string, accepted, skipped)."""
    as_ = _tool("as")
    with tempfile.TemporaryDirectory() as td:
        s, o = os.path.join(td, "n.s"), os.path.join(td, "n.o")
        with open(s, "w") as fh:
            fh.write("    .text\n    nop\n")
        march, accepted, skipped = seed, [], []
        for e in _CANDIDATE_EXTS:
            if f"_{e}" in march:
                continue
            if _march_ok(as_, f"{march}_{e}", s, o):
                march, _ = f"{march}_{e}", accepted.append(e)
            else:
                skipped.append(e)
    if verbose:
        print(f"  {seed!r:12s} -> {len(accepted)} exts, skipped {skipped}")
    return march, accepted, skipped


# From /proc/cpuinfo on the SG2042. The trailing `v` is NOT standard V: T-Head
# cores report their 0.7.1-derived vector extension in that slot, so it is
# translated to `xtheadvector`. See METHODS.md for the evidence.
DEVICE_ISA_REPORTED = "rv64imafdcv"

# Extensions the running system does NOT advertise but that three independent
# sources confirm the silicon implements: the PMU screen (100% active at legal
# rounding modes, matching advertised widths), Linux 6.17's SG2042 device tree
# (adds `zfh`), and the XuanTie C910/C920 manual (half-precision, sec 3.2.6).
# Phase 1 (--phase 1) omits these: it reproduces the as-shipped device view in
# which zfh reads as undocumented, which is how the discovery was made. Phase 2
# (--phase 2, default) folds them in as known capabilities and the analysis
# continues on what remains undocumented after zfh is accounted for.
CONFIRMED_EXTS = ["zfh"]

# Vendor extensions are never advertised in the isa string; they come from the
# vendor identity (mvendorid 0x5b7 = T-Head, C920). The authoritative list is
# GCC's own core definition for xt-c920 (gcc/config/riscv/riscv-cores.def) --
# what the compiler targets when it is told the part is a C920. An earlier
# hand-curated list added four extensions GCC's def does NOT claim
# (xtheadfmv, xtheadint, xtheadvdot, xtheadzvamo). That was a too-wide march,
# which hides findings: xtheadzvamo alone was absorbing 72 ACTIVE encodings
# (th.vamo* vector atomics) into the documented set. Aligning to GCC's def
# surfaces them. (GCC's def also lists zfh -- a fourth confirmation of the
# half-precision finding, independent of counters, kernel DT, and the manual.)
DEVICE_VENDOR_EXTS = [
    "xtheadba", "xtheadbb", "xtheadbs", "xtheadcmo", "xtheadcondmov",
    "xtheadfmemidx", "xtheadmac", "xtheadmemidx", "xtheadmempair",
    "xtheadsync", "xtheadvector",
]


def build_march_device(verbose=False, confirmed=()):
    """The march string for THIS silicon, built from what the chip reports
    rather than from what binutils will accept.

    Earlier revisions greedily added every non-conflicting extension binutils
    knows (~47 of them). That was wrong in the direction that matters: it
    declares Zfa, Zacas, Zabha, Zicond, Zk*, Zb* and others that `rv64imafdcv`
    does not claim, so encodings from unimplemented extensions were being
    labelled "documented" and silently removed from the undocumented set.
    A too-wide march cannot create false positives, only false NEGATIVES --
    it hides real findings. The screen must therefore use the narrowest
    defensible march, which is the one the hardware reports.
    """
    # Drop the misleading 'v' (handled as xtheadvector below), then restore
    # Zicsr/Zifencei. Those were split out of base-I in 2019 and are implied by
    # `g`, but a kernel reporting `rv64imafdcv` does not list them separately --
    # so spelling the string out literally silently drops them and every
    # `fence.i` / CSR encoding lands in the headline set as a false positive.
    # This core runs Linux, which requires both.
    base = DEVICE_ISA_REPORTED[:-1] + "_zicsr_zifencei"
    as_ = _tool("as")
    ver = _assert_riscv(as_)
    if verbose:
        print(f"  assembler               : {as_}\n  {'':24s}  {ver}")
    exts, skipped = [], []
    with tempfile.TemporaryDirectory() as td:
        s = os.path.join(td, "p.s"); o = os.path.join(td, "p.o")
        with open(s, "w") as fh:
            fh.write("nop\n")
        for e in list(DEVICE_VENDOR_EXTS) + list(confirmed):
            (exts if _march_ok(as_, f"{base}_{e}", s, o) else skipped).append(e)
        march = base + ("_" + "_".join(exts) if exts else "")
        assert _march_ok(as_, march, s, o), march
    if verbose:
        print(f"  device ISA reported     : {DEVICE_ISA_REPORTED} "
              f"(mvendorid 0x5b7 = T-Head; 'v' => xtheadvector)")
        print(f"  + vendor extensions     : {len(exts)}"
              + (f", unsupported by binutils: {skipped}" if skipped else ""))
        print(f"  -march                  : {march}")
    return march, exts, skipped


def build_march_broad(verbose=False):
    """Annotation-only: the widest march binutils will accept, seeded with `v`
    so the V/XTheadVector conflict resolves toward standard V. Names encodings
    the device march cannot -- standard vector AND every Z* extension the chip
    does not advertise. Feeds broad_mnemonic/broad_ext and NOTHING else."""
    march, ok, skip = build_march_max("rv64gcv", verbose)
    return march, ["v"] + ok, skip


def attribute_extensions(encs, base_recognised, exts, verbose=False):
    """Which extension is responsible for each encoding beyond plain rv64gc.
    Leave-one-out: decode under rv64gc_<ext> alone and subtract the rv64gc set.
    An encoding can be attributed to more than one extension (they overlap);
    all are recorded, joined by '|'."""
    owner = {}
    for e in exts:
        if e == "v":
            march = "rv64gcv"
        else:
            march = f"rv64gc_{e}"
        try:
            d = disassemble(encs, march)
        except SystemExit:
            continue
        for enc, mn in d.items():
            if mn is not None and enc not in base_recognised:
                owner.setdefault(enc, []).append(e)
    if verbose:
        print(f"  attributed {len(owner)} encodings across {len(exts)} extensions")
    return {k: "|".join(v) for k, v in owner.items()}



# ---- reserved floating-point rounding modes ------------------------------
# rm (bits 14:12) values 5 and 6 are RESERVED by the RISC-V spec. A core that
# implements the FP extension correctly raises an illegal-instruction trap on
# them, so the word has no architectural meaning -- yet binutils happily prints
# a mnemonic for it. Every such encoding is therefore a DECODER bug, not a
# documented instruction: the toolchain over-accepts. Variant C of the
# confusion matrix moves them out of the documented column.
#
# The regex is the same one screen_report.inert_taxonomy uses, kept here as the
# single source of truth so the taxonomy and the matrix cannot drift apart.
# It deliberately matches only FP *arithmetic* -- fmv/fsgnj/fclass/fle/feq use
# bits 14:12 as a funct3 selector, not as rm, so 5/6 are legal values there.
FP_ARITH_RE = r"^f(add|sub|mul|div|sqrt|madd|msub|nmadd|nmsub|cvt)"


def reserved_rm_mask(mnemonic, enc):
    """Boolean mask over encodings a decoder NAMES as FP arithmetic but whose
    rm field holds a reserved value (5 or 6).

    Returns a plain numpy array, not a Series: these masks get combined with
    columns from a different frame and pandas index alignment would silently
    produce NaN-filled garbage instead of an error.
    """
    import pandas as pd
    named = pd.Series(list(mnemonic)).str.match(FP_ARITH_RE, na=False).to_numpy()
    rm = np.right_shift(np.asarray(enc, dtype=np.int64), 12) & 0x7
    return named & np.isin(rm, (5, 6))


_CM_FIELDS = ("documented_active", "documented_inert",
              "undocumented_active", "undocumented_inert")


def _confusion(recognised, active):
    """The four cells, from a documented/undocumented mask and an active mask."""
    r = np.asarray(recognised, dtype=bool)
    a = np.asarray(active, dtype=bool)
    return dict(zip(_CM_FIELDS, (int((r & a).sum()), int((r & ~a).sum()),
                                 int((~r & a).sum()), int((~r & ~a).sum()))))


def confusion_variants(df, outdir):
    """The three confusion matrices the screen has to defend, plus the
    reconciliation row, computed from one decode pass.

      A  Zfh EXCLUDED   -- the as-shipped device view (phase 1). Zfh encodings
                           read as undocumented; this is how zfh was found.
      B  Zfh INCLUDED   -- phase 2. Zfh is a confirmed capability (PMU, kernel
                           DT, C910/C920 manual, GCC's xt-c920 def), so its
                           encodings move into the documented column.
      C  B + reserved rounding modes reclassified -- every encoding B calls
                           documented purely because binutils over-accepts
                           rm=5/6 moves to undocumented. They are all inert, so
                           the move is documented_inert -> undocumented_inert
                           and the ACTIVE cells are untouched.
      C' the Zfh-attributable subset of C -- only the rm=5/6 encodings that
                           entering Zfh newly made "documented" (all .h). This
                           is the 272-encoding slice; C is the same rule applied
                           to the whole FP space, which is strictly larger.

    Active/inert is identical across all four -- only the documented axis moves.
    """
    import pandas as pd
    rows = [dict(variant="A", zfh="excluded", rm56="documented",
                 basis="phase 1: as-shipped device march",
                 n_reclassified=0,
                 **_confusion(df.recognised_zfh_excl, df.active)),
            dict(variant="B", zfh="included", rm56="documented",
                 basis="phase 2: Zfh folded in as a confirmed capability",
                 n_reclassified=0,
                 **_confusion(df.recognised_zfh_incl, df.active)),
            dict(variant="C", zfh="included", rm56="undocumented+inert",
                 basis="B, minus every rm=5/6 FP encoding binutils over-accepts",
                 n_reclassified=int(df.reserved_rm.sum()),
                 **_confusion(df.recognised_zfh_incl & ~df.reserved_rm,
                              df.active)),
            dict(variant="C'", zfh="included", rm56="undocumented+inert (Zfh only)",
                 basis="C restricted to the rm=5/6 encodings Zfh itself added",
                 n_reclassified=int(df.reserved_rm_zfh.sum()),
                 **_confusion(df.recognised_zfh_incl & ~df.reserved_rm_zfh,
                              df.active))]
    t = pd.DataFrame(rows)
    assert (t[list(_CM_FIELDS)].sum(axis=1) == len(df)).all(), "cells must total"
    t.to_csv(os.path.join(outdir, "confusion_variants.csv"), index=False)
    return t


def _tool(name):
    # Plain concatenation, NOT os.path.join: BINUTILS_PREFIX may end in a tool
    # name prefix ('.../bin/riscv64-elf-') rather than a directory separator.
    p = BINUTILS_PREFIX + name
    if not os.path.exists(p):
        sys.exit(
            f"RISC-V binutils not found: tried '{p}'.\n"
            f"Install one (brew install riscv64-elf-binutils) or set\n"
            f"  RISCV_BINUTILS_PREFIX=/path/to/bin            (plain 'as')\n"
            f"  RISCV_BINUTILS_PREFIX=/path/to/bin/riscv64-elf-   (prefixed)")
    return p


def _assert_riscv(as_):
    """Guard against silently using the HOST assembler: an x86/arm64 `as` will
    happily run and reject every -march, which looks like a data problem."""
    try:
        out = subprocess.run([as_, "--version"], capture_output=True, text=True,
                             timeout=20).stdout
    except Exception as e:                                   # noqa: BLE001
        sys.exit(f"could not run {as_}: {e}")
    if "riscv" not in out.lower():
        sys.exit(f"{as_} is not a RISC-V assembler:\n  "
                 + (out.strip().splitlines() or ["<no output>"])[0])
    return out.strip().splitlines()[0]


def disassemble(encodings, march, chunk=8192):
    """Assemble each encoding as a raw 4-byte word and disassemble it.

    Returns {encoding: mnemonic or None}.  None means binutils printed the
    word back as a `.insn`/`.word` directive, i.e. it did not recognise it.
    """
    as_, objdump = _tool("as"), _tool("objdump")
    out = {}
    for i in range(0, len(encodings), chunk):
        batch = encodings[i:i + chunk]
        with tempfile.TemporaryDirectory() as td:
            s, o = os.path.join(td, "b.s"), os.path.join(td, "b.o")
            with open(s, "w") as fh:
                fh.write("    .text\n")
                for e in batch:
                    fh.write(f"    .insn 4, 0x{e:08x}\n")
            r = subprocess.run([as_, f"-march={march}", "-o", o, s],
                               capture_output=True, text=True)
            if r.returncode != 0:
                sys.exit(f"as failed:\n{r.stderr[:2000]}")
            r = subprocess.run([objdump, "-d", "--insn-width=4", o],
                               capture_output=True, text=True)
            seen = set()
            for line in r.stdout.splitlines():
                m = re.match(r'\s*[0-9a-f]+:\t([0-9a-f]{8}) \t(.*)$', line)
                if not m:
                    continue
                enc, txt = int(m.group(1), 16), m.group(2).strip()
                if enc in seen:          # identical words repeat; first wins
                    continue
                seen.add(enc)
                unknown = txt.startswith(".insn") or txt.startswith(".word") \
                    or txt.startswith("unknown")
                out[enc] = None if unknown else txt.split("\t")[0].split()[0]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scan", default="consensus_scan.csv")
    ap.add_argument("--outdir", default="out")
    ap.add_argument("--phase", type=int, choices=[1, 2], default=2,
                    help="1: as-shipped device march (zfh reads undocumented, "
                         "the discovery view). 2: fold in CONFIRMED_EXTS as "
                         "known capabilities and analyse the remainder.")
    args = ap.parse_args()
    confirmed = CONFIRMED_EXTS if args.phase == 2 else []

    import pandas as pd

    scan = pd.read_csv(args.scan)
    scan["enc"] = scan["encoding"].apply(lambda s: int(str(s), 16) & 0xFFFFFFFF)
    encs = sorted(scan["enc"].unique())

    print(f"building device ISA (phase {args.phase}):")
    march_dev, dev_exts, dev_skip = build_march_device(verbose=True,
                                                       confirmed=confirmed)
    # The OTHER device view is decoded on every run whatever --phase says: the
    # confusion-matrix variants have to answer "what if Zfh is / is not counted
    # as documented?" from one pass, and a second decode costs a fraction of a
    # second against a screen that otherwise has to be run twice and joined by
    # hand. --phase still decides which view drives `verdict` and the outputs.
    march_alt, _, _ = build_march_device(
        verbose=False, confirmed=([] if args.phase == 2 else CONFIRMED_EXTS))
    march_v, v_exts, _ = build_march_broad(verbose=True)
    dec_dev = disassemble(encs, march_dev)
    dec_alt = disassemble(encs, march_alt)
    dec_incl = dec_dev if args.phase == 2 else dec_alt   # Zfh in the march
    dec_excl = dec_alt if args.phase == 2 else dec_dev   # Zfh out of the march
    dec_v = disassemble(encs, march_v)
    dec_base = disassemble(encs, MARCH_BASE)
    assert len(dec_dev) == len(encs), (len(dec_dev), len(encs))

    df = pd.DataFrame({"enc": encs})
    df["mnemonic"] = df.enc.map(dec_dev)            # the ONLY verdict basis
    df["recognised"] = df.mnemonic.notna()
    df["mnemonic_base"] = df.enc.map(dec_base)
    df["recognised_base"] = df.mnemonic_base.notna()
    # The two device views, side by side, independent of --phase. Only the
    # documented axis differs between them; activity is measured, not decoded.
    df["mnemonic_zfh_incl"] = df.enc.map(dec_incl)
    df["recognised_zfh_incl"] = df.mnemonic_zfh_incl.notna()
    df["recognised_zfh_excl"] = df.enc.map(dec_excl).notna()
    df["zfh_only"] = df.recognised_zfh_incl & ~df.recognised_zfh_excl
    # Reserved rounding modes, named against the Zfh-INCLUDED decode (the wider
    # of the two, so the mask is the full set) and against the Zfh-added slice.
    df["reserved_rm"] = reserved_rm_mask(df.mnemonic_zfh_incl, df.enc)
    df["reserved_rm_zfh"] = df.reserved_rm & df.zfh_only
    df["vendor_only"] = df.recognised & ~df.recognised_base
    # ANNOTATION ONLY: what a decoder wider than the device ISA would call an
    # encoding the device march cannot name. Feeds no verdict and no count.
    df["broad_mnemonic"] = [
        dec_v.get(e) if not r else None
        for e, r in zip(df.enc, df.recognised)]
    # Which unadvertised extension names each of them. A high activity rate
    # here means the chip implements an extension it does not advertise.
    brd = [e for e, m in zip(df.enc, df.broad_mnemonic) if m is not None]
    bo = attribute_extensions(brd, set(), v_exts, verbose=False)
    df["broad_ext"] = df.enc.map(bo).where(df.broad_mnemonic.notna())

    df["opcode"] = df.enc & 0x7f
    df["encoding"] = df.enc.apply(lambda v: f"0x{v:08x}")

    base_rec = {e for e, m in dec_base.items() if m is not None}
    owners = attribute_extensions(encs, base_rec, dev_exts, verbose=True)
    df["attributed_to"] = df.enc.map(owners).fillna(
        df.recognised_base.map({True: "rv64gc", False: None}))

    # Phase 1 (as-shipped view) writes to a sibling dir so it never clobbers the
    # canonical phase-2 outputs the rest of the pipeline reads.
    if args.phase == 1:
        args.outdir = os.path.join(args.outdir, "phase1")
    os.makedirs(args.outdir, exist_ok=True)
    df.drop(columns=["enc"]).to_csv(
        os.path.join(args.outdir, "binutils_decode.csv"), index=False)

    # ---- join measured PMU activity -------------------------------------
    # An encoding is ACTIVE if any event counter deviates from that event's
    # inactive baseline. Baseline = the modal value of the event across the
    # whole scan (the overwhelmingly common "nothing happened" reading).
    # Cycles and instret are EXCLUDED from the activity basis: cycle counts
    # vary run-to-run by design, and instret increments for every retired
    # instruction including the harness. Both are still carried for reporting.
    ev = [c for c in scan.columns if re.fullmatch(r'ev_0x[0-9a-fA-F]{4}', str(c))]
    if not ev:
        ev = [c for c in scan.columns
              if str(c).startswith("0x") and "_spread" not in str(c)]
    ev = [c for c in ev if not re.search(r'cycle|instret', str(c), re.I)]
    if not ev:
        sys.exit("no event columns found in scan; cannot compute activity")

    per_enc = scan.drop_duplicates("enc").set_index("enc")
    baseline = per_enc[ev].mode().iloc[0]
    active = (per_enc[ev].sub(baseline, axis=1).abs() > 0).any(axis=1)

    df["active"] = df.enc.map(active).fillna(False).astype(bool)
    df["n_events_deviating"] = df.enc.map(
        (per_enc[ev].sub(baseline, axis=1).abs() > 0).sum(axis=1)).fillna(0).astype(int)

    # ---- alias filter: reserved/ignored bits, not hidden instructions ----
    # A CPU may ignore reserved bits in an otherwise-legal encoding. binutils
    # is strict and calls such a word unrecognised, but the core just executes
    # the base instruction -- so it looks "active but undocumented" while being
    # nothing of the sort. Detect it: for each unrecognised-but-active
    # encoding, search encodings that ARE recognised and differ only in bits
    # binutils treats as fixed; if any recognised sibling has a byte-identical
    # PMU signature, this is an alias, not a discovery.
    #
    # Concretely: clear one contiguous high field at a time (the usual home of
    # reserved bits) and test the resulting canonical word.
    sigs = per_enc[ev]
    # Guard the two ways this comparison can be silently, vacuously True.
    # With zero event columns `(a == b).all()` is True for EVERY pair, so every
    # active encoding matches its first candidate sibling and the headline set
    # collapses to 0 with no error raised. With a duplicated index `.loc[x]`
    # returns a DataFrame, which changes the comparison's shape. Both are
    # configuration faults, not data findings -- fail loudly.
    if len(ev) == 0:
        raise RuntimeError(
            "no event columns resolved from the scan: every alias comparison "
            "would be vacuously True and the headline set would collapse to 0")
    if not sigs.index.is_unique:
        raise RuntimeError(
            f"duplicate encodings in scan index "
            f"({int(sigs.index.duplicated().sum())} dupes)")

    sig_arr = sigs.to_numpy()
    sig_row = {e: i for i, e in enumerate(sigs.index)}

    MASKS = ((31, 25), (31, 20), (31, 26), (24, 20), (31, 12))

    def lax_decode_of(enc):
        """Return the documented instruction this encoding is a lax decode of.

        A lax decode is one where the core does not decode a field, so the word
        executes as the base instruction. Two conditions, both required:
          1. no wider decoder names it -- if some extension does, it is a
             real instruction, not the core being lax;
          2. clearing the field yields a *recognised* encoding with the same
             signature;
          3. the field is genuinely a don't-care -- every value of it gives one
             signature. Without (3), a match is coincidence: fadd.s and fmul.h
             both read INST_FP+1 yet differ in funct5, which the core decodes.
        """
        i = sig_row.get(enc)
        if i is None or broad_map.get(enc):
            return None                 # a real extension names it: not lax
        for hi, lo in MASKS:
            mask = ((1 << (hi - lo + 1)) - 1) << lo
            c = enc & ~mask
            if c == enc or not recognised_map.get(c, False):
                continue
            j = sig_row.get(c)
            if j is None or not np.array_equal(sig_arr[j], sig_arr[i]):
                continue
            sibs = [k for v in range(1 << (hi - lo + 1))
                    if (k := sig_row.get(c | (v << lo))) is not None]
            if len(np.unique(sig_arr[sibs], axis=0)) == 1:
                return c
        return None

    recognised_map = dict(zip(df.enc, df.recognised))
    broad_map = dict(zip(df.enc, df.broad_mnemonic.notna()))
    df["lax_decode_of"] = [
        (f"0x{a:08x}" if (a := lax_decode_of(e)) is not None else None)
        if (not r and act) else None
        for e, r, act in zip(df.enc, df.recognised, df.active)]

    # ---- the screen verdict ---------------------------------------------
    # Exactly one axis decides the documented/undocumented split: can the
    # DEVICE ISA name it? Active/inert is reported alongside that split so the
    # terminal summary shows the four final categories explicitly.
    df["verdict"] = np.select(
        [df.recognised & df.active,
         df.recognised & ~df.active,
         ~df.recognised & df.active],
        ["documented_active", "documented_inert",
         "undocumented_active"],
        default="undocumented_inert")

    cols = ["encoding", "opcode", "mnemonic", "recognised", "attributed_to",
            "vendor_only", "broad_mnemonic", "broad_ext", "active",
            "n_events_deviating", "lax_decode_of", "verdict",
            "recognised_zfh_incl", "recognised_zfh_excl", "zfh_only",
            "reserved_rm", "reserved_rm_zfh"]
    df[cols].to_csv(os.path.join(args.outdir, "binutils_screen.csv"), index=False)

    hidden = df[df.verdict == "undocumented_active"].sort_values(
        ["opcode", "encoding"])
    hidden[cols].to_csv(os.path.join(args.outdir, "binutils_hidden.csv"),
                        index=False)

    n_sv = int(df.broad_mnemonic.notna().sum())
    n_sv_act = int((df.broad_mnemonic.notna() & df.active).sum())

    print(f"\nencodings decoded         : {len(df)}")
    print(f"  recognised (device ISA) : {int(df.recognised.sum())}")
    print(f"  recognised, rv64gc only : {int(df.recognised_base.sum())}")
    print(f"  vendor-extension only   : {int(df.vendor_only.sum())}")
    print(f"  undefined here          : {int((~df.recognised).sum())}")
    print(f"\nactivity basis: {len(ev)} event columns (cycles/instret excluded)")
    print(f"  active                  : {int(df.active.sum())}")
    print("\nverdicts:")
    for verdict in ["documented_active", "documented_inert",
                    "undocumented_active", "undocumented_inert"]:
        print(f"  {verdict:26s} {int((df.verdict == verdict).sum())}")

    cv = confusion_variants(df, args.outdir)
    print("\nconfusion matrix, three documented-axis definitions")
    print(f"  (activity identical throughout: {int(df.active.sum())} of "
          f"{len(df)} encodings are PMU-active)")
    print(f"  {'':3s} {'doc+active':>11s} {'doc+inert':>10s} "
          f"{'undoc+active':>13s} {'undoc+inert':>12s}   basis")
    for r in cv.itertuples():
        print(f"  {r.variant:3s} {r.documented_active:11,d} "
              f"{r.documented_inert:10,d} {r.undocumented_active:13,d} "
              f"{r.undocumented_inert:12,d}   {r.basis}")
    print(f"  Zfh accounts for {int(df.zfh_only.sum())} encodings "
          f"({int((df.zfh_only & df.active).sum())} active); "
          f"rm=5/6 over-acceptance for {int(df.reserved_rm.sum())} "
          f"({int(df.reserved_rm_zfh.sum())} of them Zfh-added), "
          f"{int((df.reserved_rm & df.active).sum())} active")
    print(f"\n>>> HEADLINE undocumented + active: {len(hidden)}")
    nl = int(df.lax_decode_of.notna().sum())
    print(f"  of which lax decodes of a documented instruction: {nl} "
          f"({len(hidden) - nl} not explained by lax decoding)")
    print(f"\nannotations (no effect on verdicts): {n_sv} encodings the device "
          f"march cannot name\n  are named by a wider decoder, {n_sv_act} of "
          f"those active. Per-extension activity:\n  see "
          f"unadvertised_extensions.csv -- a high rate means an extension the "
          f"chip\n  implements but does not advertise.")
    return df


if __name__ == "__main__":
    main()
