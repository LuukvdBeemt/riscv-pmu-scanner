#!/usr/bin/env python3
"""Plot per-major-opcode activity relative to the dominant inert signature.

The graph is built from the consensus PMU scan CSV and answers:
  1) what is the single most common exact PMU signature?
  2) which encodings differ from that signature (i.e. are active)?
  3) for each major opcode, how many of its encodings are active?
  4) how do official RISC-V opcode names map onto the bars?

The script saves both the chart and a CSV summary.
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from classify_instructions import OPCLASS


DEFAULT_PRIMARY = "#1f6fb4"
DEFAULT_SECONDARY = "#e0a030"


def find_major_signature(df):
    event_cols = [
        c for c in df.columns if c.startswith("0x") and not c.endswith("_spread")
    ]
    if not event_cols:
        raise ValueError("No PMU event columns found in the consensus scan CSV")

    sig_series = df[event_cols].astype(int).apply(tuple, axis=1)
    counts = sig_series.value_counts()
    dominant_sig = counts.idxmax()
    dominant_count = int(counts.iloc[0])
    dominant_pct = 100.0 * dominant_count / len(df)
    return event_cols, dominant_sig, dominant_pct


def opcode_label(opcode):
    label = OPCLASS.get(int(opcode), None)
    if label is not None:
        return label.upper()
    return "RESERVED"


def make_activity_table(df, dominant_sig):
    event_cols = [
        c for c in df.columns if c.startswith("0x") and not c.endswith("_spread")
    ]
    sig_series = df[event_cols].astype(int).apply(tuple, axis=1)

    work = df.copy()
    work["opcode"] = work["opcode"].astype(int)
    work["is_active"] = sig_series.apply(lambda s: s != dominant_sig).astype(int)

    by_opcode = (
        work.groupby("opcode")
        .agg(n_encodings=("encoding", "size"), n_active=("is_active", "sum"))
        .reset_index()
    )
    by_opcode["active_pct"] = 100.0 * by_opcode["n_active"] / by_opcode["n_encodings"]
    by_opcode["opcode_hex"] = by_opcode["opcode"].map(lambda x: f"0x{x:02x}")
    by_opcode["label"] = by_opcode["opcode"].map(opcode_label)
    by_opcode["interesting"] = (
        by_opcode["label"].str.lower().str.startswith(("custom", "reserved"))
    )
    return by_opcode.sort_values("opcode").reset_index(drop=True)


def save_plot(table, dominant_pct, out_path):
    table = table.sort_values("active_pct", ascending=False).reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(11, 9))
    colors = [DEFAULT_SECONDARY if row["interesting"] else DEFAULT_PRIMARY for _, row in table.iterrows()]
    bars = ax.barh(range(len(table)), table["active_pct"].to_numpy(), color=colors, edgecolor="black", linewidth=0.3)

    ax.set_xlabel("Active encodings (%)", fontsize=22)
    ax.set_xlim(0, 105)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.tick_params(axis="x", labelsize=14)

    labels = []
    for _, row in table.iterrows():
        labels.append(f"{row['label']} ({row['opcode_hex']})")
    ax.set_yticks(range(len(table)))
    ax.set_yticklabels(labels, fontsize=15)
    ax.invert_yaxis()
    ax.tick_params(axis="y", labelsize=15)

    for bar, row in zip(bars, table.itertuples(index=False)):
        pct = float(row.active_pct)
        x_pos = bar.get_width()
        if x_pos >= 10:
            text_x = x_pos / 2
            ha = "center"
            dx = 0
            text_color = "white"
        else:
            text_x = x_pos + 4
            ha = "left"
            dx = 0
            text_color = "black"
        ax.annotate(
            f"{pct:.0f}%",
            (text_x, bar.get_y() + bar.get_height() / 2),
            xytext=(dx, 0),
            textcoords="offset points",
            va="center",
            ha=ha,
            fontsize=14,
            color=text_color,
            weight="bold",
        )

    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    ax.grid(axis="x", linestyle="--", linewidth=0.5, alpha=0.25)
    legend_handles = [
        plt.Line2D([0], [0], color=DEFAULT_PRIMARY, lw=10, label="standard opcode space"),
        plt.Line2D([0], [0], color=DEFAULT_SECONDARY, lw=10, label="CUSTOM / RESERVED"),
    ]
    ax.legend(handles=legend_handles, loc="lower right", frameon=True, fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    import os
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scan", default=None,
                    help="consensus scan CSV (default: <outdir>/consensus_scan.csv)")
    ap.add_argument("--outdir", default="out")
    a = ap.parse_args()

    out_dir = Path(a.outdir)
    out_dir.mkdir(parents=True, exist_ok=True)
    in_path = Path(a.scan) if a.scan else out_dir / "consensus_scan.csv"
    if not in_path.exists():
        raise SystemExit(f"scan not found: {in_path}")

    df = pd.read_csv(in_path)
    event_cols, dominant_sig, dominant_pct = find_major_signature(df)
    table = make_activity_table(df, dominant_sig)

    csv_out = out_dir / "opcode_activity.csv"
    table.to_csv(csv_out, index=False)

    plot_out = out_dir / "opcode_activity.png"
    save_plot(table, dominant_pct, plot_out)

    print(f"dominant signature share: {dominant_pct:.2f}%")
    print(f"active encodings: {int(table['n_active'].sum())} / {int(table['n_encodings'].sum())}")
    print(f"saved CSV: {csv_out}")
    print(f"saved plot: {plot_out}")

    print("\nOpcode activity summary:")
    print(table[["opcode_hex", "label", "n_encodings", "n_active", "active_pct"]].to_string(index=False))


if __name__ == "__main__":
    main()