#!/usr/bin/env python3
import argparse, os, sys
import numpy as np
import pandas as pd

TIMING_COLS = ["cycles"]
CONTROL_FLOW_OPCODES = {0x6f: "JAL", 0x63: "BRANCH", 0x67: "JALR"}

def load_runs(paths):
    runs, order = {}, None
    for i, p in enumerate(paths, 1):
        df = pd.read_csv(p)
        if "encoding" not in df.columns or df.encoding.duplicated().any():
            sys.exit(f"{p}: bad 'encoding' column")
        df = df.set_index("encoding")
        if order is None:
            order, cols = df.index, df.columns
        else:
            if set(df.index) != set(order) or list(df.columns) != list(cols):
                sys.exit(f"{p}: incompatible columns/encodings with {paths[0]}")
        runs[i] = df.reindex(order)
        print(f"  run{i}: {os.path.basename(p)}  {df.shape[0]} encodings, {df.shape[1]} counters")
    return runs, order, list(cols)

def decode_fields(encodings):
    e = np.array([int(s, 16) for s in encodings], dtype=np.uint32)
    return pd.DataFrame({"encoding": list(encodings), "enc_int": e,
                         "opcode": e & 0x7F, "funct3": (e >> 12) & 0x7,
                         "funct7": (e >> 25) & 0x7F, "funct5": (e >> 27) & 0x1F})

def inert_baseline(df, cols):
    vc = df[cols].apply(tuple, axis=1).value_counts()
    return pd.Series(dict(zip(cols, vc.index[0])), dtype="int64"), int(vc.iloc[0])

def normalize(runs, cols):
    norm, rows = {}, []
    for k, df in runs.items():
        b, n = inert_baseline(df, cols)
        norm[k] = df[cols] - b
        rows.append({"run": k, "baseline_rows_matched": n,
                     "baseline_match_frac": round(n / len(df), 4), **b.to_dict()})
        if n / len(df) < 0.10:
            print(f"  WARNING run{k}: weak inert-baseline match {100*n/len(df):.1f}%")
    return norm, pd.DataFrame(rows).set_index("run")

def stack_of(frames, col):
    return np.array([frames[k][col].values for k in sorted(frames)])

def classify(a):
    eq_all = np.all(a == a[0], axis=0)
    n = a.shape[0]
    maj = np.zeros(a.shape[1], dtype=bool)
    for i in range(n):
        maj |= (a == a[i]).sum(axis=0) > n / 2
    out = np.full(a.shape[1], "split", dtype=object)
    out[maj & ~eq_all] = "majority"
    out[eq_all] = "unanimous"
    return out

def event_agreement(runs, norm, cols):
    rows = []
    for c in cols:
        raw, nrm = stack_of(runs, c), stack_of(norm, c)
        dr = raw.max(0) != raw.min(0)
        dn = nrm.max(0) != nrm.min(0)
        klass = classify(nrm)
        arr0 = np.array([runs[k][c].values for k in sorted(runs)])[:, 0]
        baseline_range = int(np.ptp(arr0))
        rows.append({
            "counter": c,
            "is_timing": c in TIMING_COLS,
            "raw_disagree_n": int(dr.sum()),
            "raw_disagree_pct": round(100 * dr.mean(), 4),
            "norm_disagree_n": int(dn.sum()),
            "norm_disagree_pct": round(100 * dn.mean(), 4),
            "explained_by_baseline_pct": round(100 * (1 - dn.sum() / dr.sum()), 2) if dr.sum() else np.nan,
            "unanimous_n": int((klass == "unanimous").sum()),
            "majority_n": int((klass == "majority").sum()),
            "split_n": int((klass == "split").sum()),
            "norm_spread_max": int((nrm.max(0) - nrm.min(0)).max()),
            "norm_spread_p99": float(np.percentile(nrm.max(0) - nrm.min(0), 99)),
            "baseline_offset_range": baseline_range,
        })
    return pd.DataFrame(rows)

def noise_localization(norm, fields, event_cols):
    dis = pd.DataFrame({c: stack_of(norm, c).max(0) != stack_of(norm, c).min(0) for c in event_cols})
    dis.index = fields.index
    mat = dis.groupby(fields.opcode.values).mean().mul(100).round(3)
    mat.index = [f"0x{i:02x}" for i in mat.index]; mat.index.name = "opcode"
    per_op = pd.DataFrame({
        "n_encodings": fields.groupby("opcode").size(),
        "n_noisy_encodings": dis.any(axis=1).groupby(fields.opcode.values).sum(),
        "n_noisy_events_total": dis.sum(axis=1).groupby(fields.opcode.values).sum(),
    })
    per_op["pct_noisy_encodings"] = (100 * per_op.n_noisy_encodings / per_op.n_encodings).round(3)
    per_op["opcode_hex"] = [f"0x{i:02x}" for i in per_op.index]
    per_op["class"] = [CONTROL_FLOW_OPCODES.get(i, "non-control-flow") for i in per_op.index]
    return per_op.sort_values("pct_noisy_encodings", ascending=False), mat, dis

def build_consensus(norm, fields, cols, event_cols):
    out = fields[["encoding", "opcode", "funct3", "funct7", "funct5"]].copy()
    out["opcode_hex"] = [f"0x{i:02x}" for i in out.opcode]
    per_col_class = {}
    for c in cols:
        a = stack_of(norm, c); klass = classify(a); val = np.median(a, axis=0)
        n = a.shape[0]
        for i in range(n):
            take = (a == a[i]).sum(axis=0) > n / 2
            val = np.where(take, a[i], val)
        out[c] = val.astype("int64"); out[c + "_spread"] = (a.max(0) - a.min(0)).astype("int64")
        per_col_class[c] = klass
    K = pd.DataFrame(per_col_class)
    out["n_events_unanimous"] = (K[event_cols] == "unanimous").sum(axis=1).values
    out["n_events_majority"] = (K[event_cols] == "majority").sum(axis=1).values
    out["n_events_unresolved"] = (K[event_cols] == "split").sum(axis=1).values
    out["confidence"] = np.where(out.n_events_unresolved > 0, "unresolved",
                          np.where(out.n_events_majority > 0, "majority", "unanimous"))
    out["cycles_class"] = K["cycles"].values if "cycles" in K else "n/a"
    out["is_control_flow"] = out.opcode.isin(CONTROL_FLOW_OPCODES).values
    return out, K

def make_figure(agree, per_op, mat, consensus, baselines, path, overlay=None):
    import matplotlib as mpl
    mpl.use("Agg")
    import matplotlib.pyplot as plt
    try:
        from kernel import apply_figure_style, panel_letter, set_frame
        apply_figure_style(sizes=(8, 7, 6))
    except Exception:
        mpl.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False})
        panel_letter = lambda ax, l, **k: ax.text(-0.12, 1.06, l, transform=ax.transAxes,
                                                  fontsize=11, fontweight="bold", va="bottom")
        set_frame = lambda *a, **k: None
    F, C, A = "#1f6fb4", "#b9c6d2", "#c2452d"
    fig = plt.figure(figsize=(11.6, 8.6))
    gs = fig.add_gridspec(2, 2, hspace=0.46, wspace=0.42, left=0.075, right=0.955, top=0.895, bottom=0.085)
    total_encodings = int(len(agree))

    def add_value_label(ax, x, y, text, dx=0.8, dy=0.0, fontsize=5.8):
        ax.text(x + dx, y + dy, text, va="center", ha="left", fontsize=fontsize)

    ax = fig.add_subplot(gs[0, 0])
    sub_source = overlay if overlay is not None else agree
    sub = sub_source[sub_source.raw_disagree_n > 0].sort_values("raw_disagree_pct")
    y = np.arange(len(sub))
    ax.hlines(y, sub.norm_disagree_pct, sub.raw_disagree_pct, color=C, lw=1.6)
    ax.scatter(sub.raw_disagree_pct, y, s=26, color=A)
    ax.scatter(sub.norm_disagree_pct, y, s=26, color=F)
    for yi, (_, row) in enumerate(sub.iterrows()):
        add_value_label(ax, row.raw_disagree_pct, yi, f"{int(row.raw_disagree_n)}/{total_encodings}", dx=0.7)
        add_value_label(ax, row.norm_disagree_pct, yi, f"{int(row.norm_disagree_n)}/{total_encodings}", dx=0.7)
    ax.set_yticks(y); ax.set_yticklabels(sub.counter, fontsize=6); ax.set_xlabel("encodings where runs disagree (%)")
    ax.set_xlim(-3, 104); panel_letter(ax, "a")
    ax = fig.add_subplot(gs[0, 1])
    p = per_op.sort_values("pct_noisy_encodings"); y = np.arange(len(p))
    cols_ = [A if c != "non-control-flow" else F for c in p["class"]]
    ax.hlines(y, 0, p.pct_noisy_encodings, color=C, lw=1.2); ax.scatter(p.pct_noisy_encodings, y, s=24, color=cols_)
    for yi, (_, row) in enumerate(p.iterrows()):
        add_value_label(ax, row.pct_noisy_encodings, yi,
                        f"{int(row.n_noisy_encodings)} enc, {int(row.n_noisy_events_total)} ev",
                        dx=0.8, fontsize=5.5)
    ax.set_yticks(y); ax.set_yticklabels(p.opcode_hex, fontsize=6); ax.set_xlim(-2, 118); panel_letter(ax, "b")
    ax = fig.add_subplot(gs[1, 0])
    order = mat.mean(axis=1).sort_values(ascending=False).index; m = mat.loc[order]
    im = ax.imshow(m.values, aspect="auto", cmap="magma_r", vmin=0, vmax=100)
    ax.set_yticks(range(len(m))); ax.set_yticklabels(m.index, fontsize=5.5)
    ax.set_xticks(range(len(m.columns))); ax.set_xticklabels(m.columns, fontsize=5, rotation=90)
    cb = fig.colorbar(im, ax=ax, pad=0.015, fraction=0.045); cb.set_label("encodings disagreeing (%)", fontsize=6.5)
    panel_letter(ax, "c")
    ax = fig.add_subplot(gs[1, 1])
    control_flow = consensus.opcode.isin([0x6f, 0x63])
    groups = [("other", ~control_flow),
              ("JAL / BRANCH", control_flow)]
    order_k = ["unanimous", "majority", "unresolved"]
    shade = {"unanimous": F, "majority": "#8fb8d8", "unresolved": A}
    for i, (lab, mask) in enumerate(groups):
        vc = consensus.confidence[mask].value_counts(); tot = int(mask.sum()); left = 0.0
        for k in order_k:
            count = int(vc.get(k, 0))
            w = 100 * count / tot
            ax.barh(i, w, left=left, color=shade[k], height=0.34, edgecolor="white", lw=0.6)
            if w > 6 or k == "unresolved":
                ax.text(left + w / 2, i, f"{w:.1f}%\n{count}/{tot}", ha="center", va="center", fontsize=6.2)
            left += w
        ax.text(1.5, i + 0.29, f"{lab}   (n={tot})", va="bottom", ha="left", fontsize=7)
    ax.set_yticks([]); ax.set_ylim(-0.45, 1.95); ax.set_xlim(0, 100); panel_letter(ax, "d")
    nb = baselines.drop(columns=[c for c in ("baseline_rows_matched", "baseline_match_frac") if c in baselines.columns])
    shifted = [c for c in nb.columns if nb[c].nunique() > 1]
    fig.suptitle("Cross-run consistency\nper-run baseline shifted on: " + ", ".join(shifted), fontsize=9.5, y=0.975)
    fig.savefig(path, dpi=300); plt.close(fig); return path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csvs", nargs="+")
    ap.add_argument("--outdir", default="out")
    ap.add_argument("--no-figure", action="store_true")
    args = ap.parse_args(); os.makedirs(args.outdir, exist_ok=True); O = lambda n: os.path.join(args.outdir, n)
    runs, order, cols = load_runs(args.csvs)
    event_cols = [c for c in cols if c not in TIMING_COLS]
    fields = decode_fields(order).set_index(order)
    norm, baselines = normalize(runs, cols); baselines.to_csv(O("baselines.csv"))
    agree = event_agreement(runs, norm, cols); agree.to_csv(O("event_agreement.csv"), index=False)
    per_op, mat, dis = noise_localization(norm, fields, event_cols)
    per_op.to_csv(O("noise_by_opcode.csv")); mat.to_csv(O("noise_opcode_event.csv"))
    consensus, K = build_consensus(norm, fields, cols, event_cols)
    consensus.to_csv(O("consensus_scan.csv"), index=False)
    consensus[consensus.confidence == "unresolved"].to_csv(O("unresolved_encodings.csv"), index=False)
    # prepare overlay (exclude noisy opcodes 0x6f,0x63,0x67)
    bad = {0x6f, 0x63, 0x67}
    mask = ~fields.opcode.isin(list(bad))
    filtered_order = fields.index[mask]
    runs_f = {k: runs[k].reindex(filtered_order) for k in runs}
    norm_f = {k: norm[k].reindex(filtered_order) for k in norm}
    agree_f = event_agreement(runs_f, norm_f, cols)
    agree_f.to_csv(O("panel_a_filtered_agreement.csv"), index=False)
    if not args.no_figure:
        make_figure(agree, per_op, mat, consensus, baselines, O("cross_run_consistency.png"), overlay=agree_f)
    print(f"Done -> {args.outdir}")

if __name__ == "__main__":
    main()
