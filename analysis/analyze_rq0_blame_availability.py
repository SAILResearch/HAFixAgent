#!/usr/bin/env python3
"""
RQ0 Combined Analysis: Generate unified figures and tables for both
Defects4J and BugsInPy blame commit distributions.

Reads from existing CSVs (no re-computation needed):
- results/blame_commit_analysis/defects4j_blame_commit_counts.csv
- results/rq0_bugsinpy/bugsinpy_blame_commit_counts.csv

Outputs:
- results/rq0_combined/rq0_blame_commit_distribution_combined.pdf  (figure*)
- results/rq0_combined/rq0_blame_availability_combined.tex          (table)

Usage:
    python -m analysis.analyze_rq0_blame_availability
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd

# Project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "results" / "rq0_combined"

# Source CSVs
D4J_CSV = PROJECT_ROOT / "results" / "blame_commit_analysis" / "defects4j_blame_commit_counts.csv"
BIP_CSV = PROJECT_ROOT / "results" / "rq0_bugsinpy" / "bugsinpy_blame_commit_counts.csv"

CATEGORIES = ["SL", "SH", "SFMH", "MFMH"]
CATEGORY_FULL = {
    "SL": "Single-Line",
    "SH": "Single-Hunk",
    "SFMH": "Single-File Multi-Hunk",
    "MFMH": "Multi-File Multi-Hunk",
}


def load_data():
    """Load both CSVs and return DataFrames."""
    d4j = pd.read_csv(D4J_CSV)
    bip = pd.read_csv(BIP_CSV)

    # Normalize column names (D4J has Total_Hunks, BugsInPy may not)
    if "Is_Blameless" in d4j.columns:
        d4j["Is_Blameless"] = d4j["Is_Blameless"].map(
            {True: True, False: False, "True": True, "False": False}
        )
    if "Is_Blameless" in bip.columns:
        bip["Is_Blameless"] = bip["Is_Blameless"].map(
            {True: True, False: False, "True": True, "False": False}
        )

    return d4j, bip


def compute_commit_groups(df, categories):
    """Compute commit count groups (0, 1, 2, 3, >=4) per category.

    Returns dict: category -> [count_0, count_1, count_2, count_3, count_4plus]
    """
    data = {}
    for cat in categories:
        cat_df = df[df["Category"] == cat]
        n = len(cat_df)
        if n == 0:
            data[cat] = [0, 0, 0, 0, 0]
            continue
        c0 = len(cat_df[cat_df["Unique_Commit_Count"] == 0])
        c1 = len(cat_df[cat_df["Unique_Commit_Count"] == 1])
        c2 = len(cat_df[cat_df["Unique_Commit_Count"] == 2])
        c3 = len(cat_df[cat_df["Unique_Commit_Count"] == 3])
        c4p = len(cat_df[cat_df["Unique_Commit_Count"] >= 4])
        data[cat] = [c0, c1, c2, c3, c4p]
    return data


def compute_blame_availability(df, categories):
    """Compute blameable/blameless counts per category.

    Returns list of dicts with keys: category, blameable, blameless, total
    """
    rows = []
    for cat in categories:
        cat_df = df[df["Category"] == cat]
        total = len(cat_df)
        blameless = len(cat_df[cat_df["Unique_Commit_Count"] == 0])
        blameable = total - blameless
        rows.append({
            "category": cat,
            "blameable": blameable,
            "blameless": blameless,
            "total": total,
        })
    # Total row
    total_all = sum(r["total"] for r in rows)
    blame_all = sum(r["blameable"] for r in rows)
    bless_all = sum(r["blameless"] for r in rows)
    rows.append({
        "category": "Total",
        "blameable": blame_all,
        "blameless": bless_all,
        "total": total_all,
    })
    return rows


def create_combined_figure(d4j_df, bip_df):
    """Create side-by-side percentage stacked bar chart (figure*).

    Two subfigures: (a) Defects4J, (b) BugsInPy
    Three groups: 0, 1, >=2 commits.
    Percentage-based y-axis for fair cross-dataset comparison.
    Rare >=2 cases in Defects4J get arrow annotations.
    """
    # Publication-quality settings
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "font.size": 9,
        "axes.labelsize": 10,
        "axes.titlesize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 8,
        "legend.title_fontsize": 8,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "text.usetex": False,
    })

    d4j_data = compute_commit_groups(d4j_df, CATEGORIES)
    bip_data = compute_commit_groups(bip_df, CATEGORIES)

    # Convert to percentages
    d4j_pct = {}
    bip_pct = {}
    for cat in CATEGORIES:
        d4j_total = sum(d4j_data[cat])
        bip_total = sum(bip_data[cat])
        d4j_pct[cat] = [v / d4j_total * 100 if d4j_total > 0 else 0 for v in d4j_data[cat]]
        bip_pct[cat] = [v / bip_total * 100 if bip_total > 0 else 0 for v in bip_data[cat]]

    commit_labels = ["0", "1", "2", "3", "\u22654"]
    # Colors: gray=blameless, green=1 commit, blue=2, orange=3, red=≥4
    colors = ["#BDBDBD", "#66BB6A", "#42A5F5", "#FFA726", "#EF5350"]

    fig, axes = plt.subplots(1, 2, figsize=(7, 2.5), sharey=True)

    datasets = [
        ("Defects4J (854 bugs)", d4j_pct, d4j_data),
        ("BugsInPy (501 bugs)", bip_pct, bip_data),
    ]

    for ax_idx, (title, pct_data, raw_data) in enumerate(datasets):
        ax = axes[ax_idx]
        x = np.arange(len(CATEGORIES))
        width = 0.55

        bottom = np.zeros(len(CATEGORIES))
        for i, (label, color) in enumerate(zip(commit_labels, colors)):
            heights = [pct_data[cat][i] for cat in CATEGORIES]
            raw_heights = [raw_data[cat][i] for cat in CATEGORIES]
            bars = ax.bar(
                x, heights, width,
                label=label,
                color=color,
                bottom=bottom,
                edgecolor="white",
                linewidth=0.8,
            )

            # Add count labels inside bars
            for j, (bar, pct_h, raw_h) in enumerate(zip(bars, heights, raw_heights)):
                if raw_h == 0:
                    pass  # Skip zero segments
                elif pct_h >= 5:
                    # Enough space: label inside bar
                    y_pos = bottom[j] + pct_h / 2
                    text_color = "white" if color in ["#42A5F5", "#FFA726", "#EF5350"] else "black"
                    ax.text(
                        bar.get_x() + bar.get_width() / 2.0, y_pos,
                        str(int(raw_h)),
                        ha="center", va="center",
                        fontsize=8,
                        color=text_color,
                    )
                else:
                    # Small but non-zero: arrow annotation pointing to segment
                    y_top = bottom[j] + pct_h
                    ax.annotate(
                        str(int(raw_h)),
                        xy=(bar.get_x() + bar.get_width() / 2.0, y_top),
                        xytext=(bar.get_x() + bar.get_width() / 2.0, y_top + 5),
                        ha="center", va="bottom",
                        fontsize=7.5,
                        color=color if color != "#BDBDBD" else "#757575",
                        arrowprops=dict(arrowstyle="->", color=color if color != "#BDBDBD" else "#757575", lw=0.8),
                    )

            bottom += heights

        ax.set_xlabel("Bug Category", fontsize=9, labelpad=4)
        ax.set_xticks(x)
        ax.set_xticklabels(CATEGORIES, fontsize=8.5)
        ax.grid(axis="y", alpha=0.25, linestyle="--", linewidth=0.5)
        ax.set_axisbelow(True)
        ax.set_ylim(0, 115)  # Leave room for annotations

    axes[0].set_ylabel("Percentage of Bugs (%)", fontsize=9)

    # Centered titles with (a)/(b) label merged in
    subfig_titles = [
        "(a) Defects4J (854 bugs)",
        "(b) BugsInPy (501 bugs)",
    ]
    for ax, stitle in zip(axes, subfig_titles):
        ax.set_title(stitle, fontsize=9, pad=5)

    # Single shared legend below figure, tight spacing
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels,
        title="Unique Blame Commits",
        loc="lower center",
        ncol=5,
        fontsize=8,
        title_fontsize=8,
        bbox_to_anchor=(0.5, -0.07),
        frameon=True,
        edgecolor="#CCCCCC",
    )

    plt.tight_layout(rect=[0, 0.07, 1, 1])

    output_path = OUTPUT_DIR / "rq0_blame_commit_distribution_combined.pdf"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved figure: {output_path}")
    plt.close()


def create_combined_table(d4j_df, bip_df):
    """Generate a combined LaTeX table for blame availability across both datasets."""
    d4j_rows = compute_blame_availability(d4j_df, CATEGORIES)
    bip_rows = compute_blame_availability(bip_df, CATEGORIES)

    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"    \centering")
    lines.append(r"    \caption{Blame availability by bug category, assuming perfect fault localization. Within-category percentages for Blameable/Blameless; Total column shows the dataset share.}")
    lines.append(r"    \label{tab:blame-availability}")
    lines.append(r"    \begin{tabular}{llrrrrrr}")
    lines.append(r"    \toprule")
    lines.append(r"    \multirow{2}{*}{\textbf{Dataset}}")
    lines.append(r"      & \multirow{2}{*}{\textbf{Category}}")
    lines.append(r"      & \multicolumn{2}{c}{\textbf{Blameable}}")
    lines.append(r"      & \multicolumn{2}{c}{\textbf{Blameless}}")
    lines.append(r"      & \multicolumn{2}{c}{\textbf{Total}} \\")
    lines.append(r"    \cmidrule(lr){3-4}\cmidrule(lr){5-6}\cmidrule(lr){7-8}")
    lines.append(r"      & & \# & \% & \# & \% & \# & \% \\")
    lines.append(r"    \midrule")

    def format_rows(dataset_name, rows):
        """Format rows for one dataset."""
        result = []
        for i, r in enumerate(rows):
            cat_label = r["category"]
            total = r["total"]
            blameable = r["blameable"]
            blameless = r["blameless"]
            dataset_total = rows[-1]["total"]  # last row is Total

            blame_pct = blameable / total * 100 if total > 0 else 0
            bless_pct = blameless / total * 100 if total > 0 else 0
            total_pct = total / dataset_total * 100 if dataset_total > 0 else 0

            # Dataset name only on first row, use multirow
            if i == 0:
                ds_col = r"    \multirow{5}{*}{" + dataset_name + "}"
            else:
                ds_col = "    "

            if cat_label == "Total":
                result.append(
                    r"    \cmidrule(lr){2-8}"
                )
                result.append(
                    f"      & \\textbf{{{cat_label}}} & \\textbf{{{blameable}}} & \\textbf{{{blame_pct:.1f}}} "
                    f"& \\textbf{{{blameless}}} & \\textbf{{{bless_pct:.1f}}} "
                    f"& \\textbf{{{total}}} & \\textbf{{100.0}} \\\\"
                )
            else:
                result.append(
                    f"{ds_col} & {cat_label} & {blameable} & {blame_pct:.1f} "
                    f"& {blameless} & {bless_pct:.1f} "
                    f"& {total} & {total_pct:.1f} \\\\"
                )
        return result

    lines.extend(format_rows("Defects4J", d4j_rows))
    lines.append(r"    \midrule")
    lines.extend(format_rows("BugsInPy", bip_rows))
    lines.append(r"    \bottomrule")
    lines.append(r"    \end{tabular}")
    lines.append(r"\end{table}")

    tex_content = "\n".join(lines)
    output_path = OUTPUT_DIR / "rq0_blame_availability_combined.tex"
    with open(output_path, "w") as f:
        f.write(tex_content)
    print(f"Saved table: {output_path}")
    return tex_content


def print_cross_dataset_summary(d4j_df, bip_df):
    """Print key cross-dataset comparison numbers for paper text."""
    print("\n" + "=" * 60)
    print("Cross-Dataset Summary (for paper text)")
    print("=" * 60)

    for name, df in [("Defects4J", d4j_df), ("BugsInPy", bip_df)]:
        total = len(df)
        blameable = len(df[df["Unique_Commit_Count"] > 0])
        single = len(df[df["Unique_Commit_Count"] == 1])
        multi = len(df[df["Unique_Commit_Count"] >= 2])
        blameless = len(df[df["Unique_Commit_Count"] == 0])

        print(f"\n[{name}] ({total} bugs)")
        print(f"  Blameable: {blameable} ({blameable/total*100:.1f}%)")
        print(f"  Blameless: {blameless} ({blameless/total*100:.1f}%)")
        print(f"  Single commit: {single} ({single/total*100:.1f}%)")
        print(f"  Multi commit (>=2): {multi} ({multi/total*100:.1f}%)")
        print(f"  Single / blameable: {single}/{blameable} = {single/blameable*100:.1f}%")

        # Per-category breakdown
        for cat in CATEGORIES:
            cat_df = df[df["Category"] == cat]
            n = len(cat_df)
            c0 = len(cat_df[cat_df["Unique_Commit_Count"] == 0])
            c1 = len(cat_df[cat_df["Unique_Commit_Count"] == 1])
            c2p = len(cat_df[cat_df["Unique_Commit_Count"] >= 2])
            blame = n - c0
            print(f"  {cat}: {n} total, {blame} blameable ({blame/n*100:.1f}%), "
                  f"{c1} single ({c1/n*100:.1f}%), {c2p} multi ({c2p/n*100:.1f}%)")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    d4j_df, bip_df = load_data()
    print(f"  Defects4J: {len(d4j_df)} bugs")
    print(f"  BugsInPy: {len(bip_df)} bugs")

    print("\nGenerating combined figure...")
    create_combined_figure(d4j_df, bip_df)

    print("\nGenerating combined table...")
    create_combined_table(d4j_df, bip_df)

    print_cross_dataset_summary(d4j_df, bip_df)


if __name__ == "__main__":
    main()
