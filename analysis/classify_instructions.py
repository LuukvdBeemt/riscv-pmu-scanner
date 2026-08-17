#!/usr/bin/env python3
"""
classify_instructions.py -- measure how well PMU event counters classify RISC-V
instructions by FUNCTION (not just "executes / traps").

Thesis question
---------------
The Ghostwrite/anomaly work shows PMU counters flag *undocumented* instructions.
This script measures the more general claim: can the 40 PMU event counters alone
recover the ISA's functional structure -- put all loads together, all FP ops
together, all integer ALU ops together -- using each encoding's behavior as the
only feature? That quantifies the method's classification power over the WHOLE
instruction space, which is the headline "how well did our method work" number.

Method
------
1. Ground truth (decoder-INDEPENDENT): every encoding's functional family is
   derived from its major opcode via the hand-written OPCLASS/COARSE tables
   below -- one entry per major opcode, no per-encoding disassembly. This
   matters beyond convenience: the documented/undocumented axis of the
   confusion matrix is itself decided by a decoder (binutils_screen.py), so
   taking labels from a decoder here would make the classifier's input overlap
   that axis. The optional Capstone decode fills a `mnem` column that is
   COSMETIC ONLY -- it appears in labeled_encodings.csv for readability and is
   never used for a label, a fold, or a metric. The script produces identical
   numbers with Capstone absent.
2. Features: the PMU event counters from consensus_scan.csv. `cycles` is dropped
   (per-run baseline-noisy); constant counters are dropped (no information).
   Classification is done on the ACTIVE set (encodings that execute) -- inert
   encodings carry zero behavioral signal by definition.
3. Supervised: 5-fold cross-validated RandomForest predicting the coarse family
   (INT-compute / FP-compute / memory / control / system / vector / custom) and
   the fine opcode-group. Report accuracy, balanced accuracy, per-family F1,
   confusion matrix.
4. Behavioral richness: number of distinct PMU signatures per family -- this
   exposes the integer-compute blind spot (rd=x0 makes ALU ops near-inert).
5. Figures: confusion matrix as a stand-alone figure, and a separate family x
    counter activation heatmap (why it works: each family fires a distinct
    counter set). The per-family separability summary is exported as CSV.

Usage
-----
  python classify_instructions.py consensus_scan.csv --outdir out
"""
import argparse, os, struct
import numpy as np
import pandas as pd

# ---- functional taxonomy (opcode -> fine class -> coarse family) -----------
OPCLASS = {
    0x03: "LOAD",
    0x07: "LOAD-FP",
    0x0b: "CUSTOM-0",
    0x0f: "MISC-MEM",
    0x13: "OP-IMM",
    0x17: "AUIPC",
    0x1b: "OP-IMM-32",
    0x23: "STORE",
    0x27: "STORE-FP",
    0x2b: "CUSTOM-1",
    0x2f: "AMO",
    0x33: "OP",
    0x37: "LUI",
    0x3b: "OP-32",

    0x43: "MADD",
    0x47: "MSUB",
    0x4b: "NMSUB",
    0x4f: "NMADD",

    0x53: "OP-FP",
    0x57: "OP-V",
    0x5b: "CUSTOM-2",

    0x63: "BRANCH",
    0x67: "JALR",
    0x6b: "RESERVED",
    0x6f: "JAL",
    0x73: "SYSTEM",
    0x77: "RESERVED",
    0x7b: "CUSTOM-3",
}
COARSE = {
    "OP": "INT-compute",
    "OP-32": "INT-compute",
    "OP-IMM": "INT-compute",
    "OP-IMM-32": "INT-compute",
    "LUI": "INT-compute",
    "AUIPC": "INT-compute",

    "OP-FP": "FP-compute",
    "MADD": "FP-compute",
    "MSUB": "FP-compute",
    "NMSUB": "FP-compute",
    "NMADD": "FP-compute",

    "LOAD": "memory",
    "STORE": "memory",
    "LOAD-FP": "memory",
    "STORE-FP": "memory",
    "AMO": "memory",

    "MISC-MEM": "system",
    "SYSTEM": "system",

    "JAL": "control",
    "JALR": "control",
    "BRANCH": "control",

    "OP-V": "vector",

    "CUSTOM-0": "custom",
    "CUSTOM-1": "custom",
    "CUSTOM-2": "custom",
    "CUSTOM-3": "custom",

    "RESERVED": "reserved",
}

PALETTE = {"FP-compute": "#1f6fb4", "INT-compute": "#8a8f96", "memory": "#2ca25f",
           "control": "#e0a030", "system": "#6a51a3", "vector": "#00a2b3", "custom": "#c2452d"}


def try_decoder():
    try:
        import capstone as cap
        md = cap.Cs(cap.CS_ARCH_RISCV, cap.CS_MODE_RISCV64)
        def decode(enc_int):
            ins = list(md.disasm(struct.pack("<I", enc_int), 0))
            return ins[0].mnemonic if ins else ""
        return decode
    except Exception:
        return lambda e: ""


def load(path):
    df = pd.read_csv(path)
    ev = [c for c in df.columns if c.startswith("0x") and not c.endswith("_spread")
          and c != "encoding"]
    E = np.array([int(s, 16) for s in df.encoding])
    df["opc"] = E & 0x7F
    df["opclass"] = [OPCLASS.get(o, f"op{o:#x}") for o in df["opc"]]
    df["coarse"] = df["opclass"].map(lambda x: COARSE.get(x, "other"))
    df["active"] = (df[ev].values != 0).sum(1) > 0
    decode = try_decoder()
    df["mnem"] = [decode(int(e)) for e in E]   # cosmetic; never used as a label
    return df, ev


def run(df, ev):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import cross_val_predict, StratifiedKFold
    from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score,
                                 confusion_matrix, classification_report)

    informative = [c for c in ev if df[c].nunique() > 1]
    act = df[df.active].copy()
    Xa = act[informative].values.astype(float)
    yc = act.coarse.values

    rf = RandomForestClassifier(n_estimators=300, random_state=0, n_jobs=1,
                                class_weight="balanced")
    cv = StratifiedKFold(5, shuffle=True, random_state=0)

    # coarse family
    pc = cross_val_predict(rf, Xa, yc, cv=cv, n_jobs=1)
    coarse_acc = accuracy_score(yc, pc)
    coarse_bal = balanced_accuracy_score(yc, pc)

    # fine opcode-group (classes with >=10 members)
    yf = act.opclass.values
    vc = pd.Series(yf).value_counts(); keep = vc[vc >= 10].index
    m = np.isin(yf, keep)
    pf = cross_val_predict(rf, Xa[m], yf[m], cv=cv, n_jobs=1)
    fine_acc = accuracy_score(yf[m], pf)
    fine_bal = balanced_accuracy_score(yf[m], pf)

    # per-family separability + behavioral richness
    rows = []
    for fam in sorted(set(yc)):
        mm = yc == fam
        sub = act[mm]
        nsig = sub[informative].apply(tuple, axis=1).nunique()
        rows.append(dict(family=fam, n=int(mm.sum()), f1=round(f1_score(yc == fam, pc == fam), 3),
                         n_signatures=int(nsig),
                         mean_nonzero=round((sub[informative].values != 0).sum(1).mean(), 2)))
    sep = pd.DataFrame(rows).sort_values("f1", ascending=False)

    labs = sorted(set(yc))
    cm = confusion_matrix(yc, pc, labels=labs)
    cmn = cm / cm.sum(1, keepdims=True)
    report = classification_report(yc, pc, digits=3, zero_division=0)
    return dict(informative=informative, act=act, yc=yc, pc=pc, labs=labs, cmn=cmn,
                sep=sep, coarse_acc=coarse_acc, coarse_bal=coarse_bal,
                fine_acc=fine_acc, fine_bal=fine_bal, n_active=len(act), report=report)


def make_figure_a(R, path):
    import matplotlib as mpl
    mpl.use("Agg")
    import matplotlib.pyplot as plt
    try:
        from kernel import apply_figure_style
        apply_figure_style(sizes=(11, 12, 11))
    except Exception:
        mpl.rcParams.update({"font.size": 11, "axes.labelsize": 12, "axes.titlesize": 12,
                             "legend.fontsize": 11, "xtick.labelsize": 11, "ytick.labelsize": 11})
    fig = plt.figure(figsize=(6.9, 5.1))
    ax = fig.add_subplot(111)
    labs, cmn = R["labs"], R["cmn"]
    im = ax.imshow(100 * cmn, cmap="Blues", vmin=0, vmax=100, aspect="auto")
    ax.set_xticks(range(len(labs))); ax.set_xticklabels(labs, rotation=45, ha="right", fontsize=11)
    ax.set_yticks(range(len(labs))); ax.set_yticklabels(labs, fontsize=11)
    for i in range(len(labs)):
        for j in range(len(labs)):
            v = 100 * cmn[i, j]
            if v >= 1:
                ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=11,
                        color="white" if v > 55 else "#222")
    ax.set_xlabel("predicted family"); ax.set_ylabel("true family (opcode)", labelpad=1)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("row-normalized %", fontsize=11)
    cb.ax.tick_params(labelsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def make_figure_c(R, path):
    import matplotlib as mpl
    mpl.use("Agg")
    import matplotlib.pyplot as plt
    try:
        from kernel import apply_figure_style, panel_letter
        apply_figure_style(sizes=(11, 12, 11))
    except Exception:
        mpl.rcParams.update({"font.size": 11, "axes.labelsize": 12, "axes.titlesize": 12,
                             "legend.fontsize": 11, "xtick.labelsize": 11, "ytick.labelsize": 11})
        panel_letter = lambda ax, l, **k: ax.text(-0.15, 1.05, l, transform=ax.transAxes,
                                                  fontsize=11, fontweight="bold")

    act, informative = R["act"], R["informative"]
    fam_order = ["INT-compute", "FP-compute", "memory", "vector", "system", "control", "custom"]
    fam_order = [f for f in fam_order if f in set(R["yc"])]
    Hm = act.groupby("coarse")[informative].mean().loc[fam_order]
    Hz = (Hm - Hm.mean(0)) / (Hm.std(0) + 1e-9)
    keepc = [c for c in informative if Hm[c].std() > 1e-6]
    Hz = Hz[keepc]

    fig, ax = plt.subplots(figsize=(10.5, 4.4))
    im = ax.imshow(Hz.values, cmap="RdBu_r", vmin=-2, vmax=2, aspect="auto")
    ax.set_yticks(range(len(fam_order))); ax.set_yticklabels(fam_order, fontsize=11)
    ax.set_xticks(range(len(keepc))); ax.set_xticklabels(keepc, rotation=90, fontsize=11)
    ax.set_xlabel("PMU event counter", labelpad=1)
    ax.set_title("Each family fires a distinct counter set (z-scored)")
    cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cb.set_label("z", fontsize=11); cb.ax.tick_params(labelsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("consensus_csv")
    ap.add_argument("--outdir", default="out")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    O = lambda n: os.path.join(args.outdir, n)

    df, ev = load(args.consensus_csv)
    R = run(df, ev)

    # labeled catalog (ground truth for anomaly triage)
    lab = df[["encoding", "opclass", "coarse", "mnem", "active"]].copy()
    lab["opcode_hex"] = [f"0x{o:02x}" for o in df["opc"]]
    lab.to_csv(O("labeled_encodings.csv"), index=False)
    R["sep"].to_csv(O("class_separability.csv"), index=False)
    make_figure_a(R, O("coarse_confusion_matrix.png"))
    make_figure_c(R, O("functional_classification_c.png"))

    print(f"Active set: {R['n_active']} / {len(df)} encodings executed.")
    print(f"COARSE family  : acc={R['coarse_acc']:.3f}  balanced={R['coarse_bal']:.3f}")
    print(f"FINE opcode-grp: acc={R['fine_acc']:.3f}  balanced={R['fine_bal']:.3f}")
    print("\nPer-family separability (F1 | #distinct PMU signatures):")
    print(R["sep"].to_string(index=False))
    print("\nCoarse classification report:")
    print(R["report"])


if __name__ == "__main__":
    main()