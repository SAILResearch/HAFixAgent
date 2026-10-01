#!/usr/bin/env python3
"""
RQ1 Combined Analysis: Generate tables and figures for the effectiveness RQ.

Reads from raw result files:
- results/defects4j/llm_judge_1line/          (HAFixAgent on D4J)
- results/bugsinpy/llm_judge_1line/           (HAFixAgent on BugsInPy)
- results/defects4j/baselines/repairagent_deepseek_deepseek-v3.2-exp/  (RepairAgent)
- results/defects4j/baselines/birch_deepseek/ (BIRCH-feedback)

Outputs:
- results/rq1_combined/  (tables, figures, verification CSVs)

Usage:
    python -m analysis.analyze_rq1_effectiveness
"""

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# For 4-set Venn diagrams
try:
    import venn
    HAS_VENN = True
except ImportError:
    HAS_VENN = False
    print("WARNING: 'venn' package not installed. Venn diagrams will be skipped.")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "results" / "rq1_combined"

# Category CSV files
D4J_CAT_CSV = PROJECT_ROOT / "dataset" / "defects4j" / "defects4j_blame_feasibility.csv"
BIP_CAT_CSV = PROJECT_ROOT / "dataset" / "bugsinpy" / "bugsinpy_blame_feasibility.csv"

# Result paths
D4J_RESULTS = PROJECT_ROOT / "results" / "defects4j" / "llm_judge_1line"
BIP_RESULTS = PROJECT_ROOT / "results" / "bugsinpy" / "llm_judge_1line"
REPAIRAGENT_DIR = PROJECT_ROOT / "results" / "defects4j" / "baselines" / "repairagent_deepseek_deepseek-v3.2-exp"
BIRCH_CSV = PROJECT_ROOT / "results" / "defects4j" / "baselines" / "birch_deepseek" / "test_results_mode_4_merged.csv"

CATEGORIES = ["SL", "SH", "SFMH", "MFMH"]
CATEGORY_DIR_MAP = {
    "SL": "single_line",
    "SH": "single_hunk",
    "SFMH": "single_file_multi_hunk",
    "MFMH": "multi_file_multi_hunk",
}

# Config number to name mapping
CONFIG_MAP = {
    "1": "non-history",
    "5": "fn_all",
    "7": "fn_pair",
    "8": "fl_diff",
}
CONFIG_ORDER = ["1", "5", "7", "8"]
CONFIG_NAMES = ["non-history", "fn_all", "fn_pair", "fl_diff"]


def load_category_mapping(csv_path: Path) -> dict:
    """Load Bug_ID -> Category mapping from feasibility CSV."""
    mapping = {}
    df = pd.read_csv(csv_path)
    for _, row in df.iterrows():
        mapping[row["Bug_ID"]] = row["Category"]
    return mapping


def count_hafixagent_results(results_dir: Path, dataset_name: str) -> dict:
    """Count passing bugs per config per category from HAFixAgent result JSONs.

    Returns: {config_num: {category: {bug_id: success_bool}}}
    """
    results = {c: defaultdict(dict) for c in CONFIG_ORDER}

    for cat, dirname in CATEGORY_DIR_MAP.items():
        cat_dir = results_dir / dirname
        if not cat_dir.exists():
            print(f"  WARNING: {cat_dir} does not exist")
            continue

        for bug_dir in sorted(cat_dir.iterdir()):
            if not bug_dir.is_dir():
                continue
            bug_name = bug_dir.name

            for config_num in CONFIG_ORDER:
                # Find result file: {bug_name}_{config_num}_result.json
                result_file = bug_dir / f"{bug_name}_{config_num}_result.json"
                if not result_file.exists():
                    continue

                try:
                    with open(result_file) as f:
                        data = json.load(f)
                    success = data.get("repair_result", {}).get("success", False)
                    results[config_num][cat][bug_name] = success
                except (json.JSONDecodeError, KeyError) as e:
                    print(f"  ERROR reading {result_file}: {e}")
                    results[config_num][cat][bug_name] = False

    return results


def count_repairagent_results(ra_dir: Path, d4j_categories: dict) -> dict:
    """Count RepairAgent plausible patches, mapped to RQ0-consistent categories.

    Returns: {category: {bug_id: has_plausible_bool}}
    """
    results = defaultdict(dict)

    for bug_dir in sorted(ra_dir.iterdir()):
        if not bug_dir.is_dir():
            continue
        bug_name = bug_dir.name  # e.g., "Chart_1"

        # Check if this bug is in our 854-bug set
        if bug_name not in d4j_categories:
            continue

        cat = d4j_categories[bug_name]
        pp_dir = bug_dir / "plausible_patches"
        has_plausible = False
        if pp_dir.exists():
            # Check if any plausible patch files exist and are non-empty
            for pp_file in pp_dir.iterdir():
                if pp_file.suffix == ".json":
                    try:
                        with open(pp_file) as f:
                            content = json.load(f)
                        # Non-empty content means plausible patch exists
                        if content:
                            has_plausible = True
                            break
                    except (json.JSONDecodeError, KeyError):
                        pass

        results[cat][bug_name] = has_plausible

    return dict(results)


def count_birch_results(birch_csv: Path, d4j_categories: dict) -> dict:
    """Count BIRCH passing bugs from merged CSV.

    Returns: {category: {bug_id: passed_bool}}
    """
    results = defaultdict(dict)

    # Increase CSV field size limit for large fields
    csv.field_size_limit(sys.maxsize)

    with open(birch_csv) as f:
        reader = csv.DictReader(f)
        for row in reader:
            # bug format: "Chart-14" -> "Chart_14"
            bug_raw = row["bug"]
            bug_name = bug_raw.replace("-", "_")

            if bug_name not in d4j_categories:
                continue

            cat = d4j_categories[bug_name]
            passed = row.get("pass", "").strip().lower() == "yes"

            # A bug counts as passing if ANY attempt passes
            if bug_name in results[cat]:
                results[cat][bug_name] = results[cat][bug_name] or passed
            else:
                results[cat][bug_name] = passed

    return dict(results)


def compute_unique_pass(ha_results: dict, baseline_config="1") -> dict:
    """Compute unique passes for each history config vs baseline.

    Returns: {config_num: {category: count_unique}}
    """
    unique = {}
    baseline = ha_results[baseline_config]

    for config_num in ["5", "7", "8"]:
        unique[config_num] = {}
        config_results = ha_results[config_num]
        for cat in CATEGORIES:
            baseline_pass = {b for b, s in baseline.get(cat, {}).items() if s}
            config_pass = {b for b, s in config_results.get(cat, {}).items() if s}
            unique[config_num][cat] = len(config_pass - baseline_pass)

    return unique


def print_and_save_summary(
    dataset_name: str,
    ha_results: dict,
    total_per_cat: dict,
    unique_pass: dict,
):
    """Print and return formatted summary table."""
    lines = []
    lines.append(f"\n{'='*70}")
    lines.append(f"HAFixAgent on {dataset_name}")
    lines.append(f"{'='*70}")
    lines.append(f"{'Category':<8} {'Total':>6} {'non-hist':>9} {'fn_all':>9} {'fn_pair':>9} {'fl_diff':>9}")
    lines.append("-" * 70)

    grand_totals = {c: 0 for c in CONFIG_ORDER}
    grand_total_bugs = 0

    for cat in CATEGORIES:
        total = total_per_cat.get(cat, 0)
        grand_total_bugs += total
        counts = []
        for config_num in CONFIG_ORDER:
            pass_count = sum(1 for s in ha_results[config_num].get(cat, {}).values() if s)
            grand_totals[config_num] += pass_count
            pct = pass_count / total * 100 if total > 0 else 0
            counts.append(f"{pass_count:>4} ({pct:4.1f}%)")
        lines.append(f"{cat:<8} {total:>6} {counts[0]:>9} {counts[1]:>9} {counts[2]:>9} {counts[3]:>9}")

    lines.append("-" * 70)
    total_line = f"{'Total':<8} {grand_total_bugs:>6}"
    for config_num in CONFIG_ORDER:
        pct = grand_totals[config_num] / grand_total_bugs * 100 if grand_total_bugs > 0 else 0
        total_line += f" {grand_totals[config_num]:>4} ({pct:4.1f}%)"
    lines.append(total_line)

    # Unique pass
    if unique_pass:
        lines.append(f"\n{'Category':<8} {'fn_all unique':>14} {'fn_pair unique':>14} {'fl_diff unique':>14}")
        lines.append("-" * 55)
        for cat in CATEGORIES:
            u5 = unique_pass["5"].get(cat, 0)
            u7 = unique_pass["7"].get(cat, 0)
            u8 = unique_pass["8"].get(cat, 0)
            lines.append(f"{cat:<8} {u5:>14} {u7:>14} {u8:>14}")
        lines.append("-" * 55)
        lines.append(f"{'Total':<8} {sum(unique_pass['5'].values()):>14} {sum(unique_pass['7'].values()):>14} {sum(unique_pass['8'].values()):>14}")

    text = "\n".join(lines)
    print(text)
    return text


def generate_baseline_comparison_table(
    ha_results: dict,
    ra_results: dict,
    birch_results: dict,
    d4j_total_per_cat: dict,
):
    """Generate single merged LaTeX table for same-LLM baseline comparison.

    One table with RepairAgent (all 854 bugs) and BIRCH (371 multi-hunk, marked with *).
    """
    lines = []
    lines.append(r"\begin{table*}[t]")
    lines.append(r"    \centering")
    lines.append(r"    \caption{Same-LLM effectiveness comparison on Defects4J (all methods use DeepSeek-V3.2-Exp). Numbers show plausible patches passing the full test suite. Bold = best per category.}")
    lines.append(r"    \label{tab:rq1-baseline-comparison}")
    lines.append(r"    \begin{threeparttable}")
    lines.append(r"    \begin{tabular}{l r r r r r r r}")
    lines.append(r"    \toprule")
    lines.append(r"    \multirow{2}{*}{\textbf{Category}}")
    lines.append(r"      & \multirow{2}{*}{\textbf{Total}}")
    lines.append(r"      & \multicolumn{4}{c}{\textbf{HAFixAgent}}")
    lines.append(r"      & \textbf{Repair-}")
    lines.append(r"      & \textbf{BIRCH-}")
    lines.append(r"    \\")
    lines.append(r"    \cmidrule(lr){3-6}")
    lines.append(r"      & & \nonhistory & \fnall & \fnpair & \fldiff & \textbf{Agent} & \textbf{fb}\tnote{*} \\")
    lines.append(r"    \midrule")

    # Compute BIRCH counts on its common bugs, and HAFixAgent counts on those same bugs
    birch_common = {}
    for cat in ["SFMH", "MFMH"]:
        birch_common[cat] = set(birch_results.get(cat, {}).keys())

    ha_totals = {c: 0 for c in CONFIG_ORDER}
    ra_total = 0
    ha_birch_totals = {c: 0 for c in CONFIG_ORDER}
    birch_pass_total = 0
    grand_total = 0

    for cat in CATEGORIES:
        total = d4j_total_per_cat[cat]
        grand_total += total

        ha_counts = {}
        for config_num in CONFIG_ORDER:
            ha_counts[config_num] = sum(1 for s in ha_results[config_num].get(cat, {}).values() if s)
            ha_totals[config_num] += ha_counts[config_num]

        ra_count = sum(1 for s in ra_results.get(cat, {}).values() if s)
        ra_total += ra_count

        # BIRCH: only for SFMH/MFMH, on its common bug subset
        if cat in birch_common:
            birch_count = sum(1 for s in birch_results.get(cat, {}).values() if s)
            birch_pass_total += birch_count
            birch_str_raw = birch_count
            # HAFixAgent on BIRCH's common subset
            for config_num in CONFIG_ORDER:
                c = sum(1 for bug, s in ha_results[config_num].get(cat, {}).items()
                        if s and bug in birch_common[cat])
                ha_birch_totals[config_num] += c
        else:
            birch_str_raw = None

        # Best across HAFixAgent + RepairAgent (BIRCH excluded from best since different subset)
        all_vals = list(ha_counts.values()) + [ra_count]
        best = max(all_vals)

        def fmt(v):
            return r"\textbf{" + str(v) + "}" if v == best else str(v)

        birch_str = "-" if birch_str_raw is None else str(birch_str_raw)

        lines.append(
            f"    {cat} & {total} & {fmt(ha_counts['1'])} & {fmt(ha_counts['5'])} "
            f"& {fmt(ha_counts['7'])} & {fmt(ha_counts['8'])} & {fmt(ra_count)} & {birch_str} \\\\"
        )

    lines.append(r"    \midrule")
    all_totals = list(ha_totals.values()) + [ra_total]
    best_total = max(all_totals)

    def fmt_t(v):
        return r"\textbf{" + str(v) + "}" if v == best_total else str(v)

    birch_total_bugs = sum(len(v) for v in birch_common.values())
    lines.append(
        f"    Total & {grand_total} & {fmt_t(ha_totals['1'])} & {fmt_t(ha_totals['5'])} "
        f"& {fmt_t(ha_totals['7'])} & {fmt_t(ha_totals['8'])} & {fmt_t(ra_total)} & {birch_pass_total} \\\\"
    )
    lines.append(r"    \bottomrule")
    lines.append(r"    \end{tabular}")
    lines.append(r"    \begin{tablenotes}")
    lines.append(f"    \\item[*] BIRCH-feedback evaluated on {birch_total_bugs} multi-hunk bugs (SFMH + MFMH) only.")
    lines.append(r"    \end{tablenotes}")
    lines.append(r"    \end{threeparttable}")
    lines.append(r"\end{table*}")

    tex = "\n".join(lines)
    output_path = OUTPUT_DIR / "rq1_baseline_comparison.tex"
    with open(output_path, "w") as f:
        f.write(tex)
    print(f"\nSaved: {output_path}")
    return tex


def compute_union_by_category(ha_results: dict, history_configs=("5", "7", "8")) -> dict:
    """Per-category and total count of bugs fixed by at least one history config.

    Union = bugs fixed by at least one of \\fnall/\\fnpair/\\fldiff (configs 5/7/8).
    This is the same computation that previously produced the summary panel inside
    the combined Venn figure; centralized here so the ablation table and any figure
    share one definition. Returns {category: count, ..., "Total": total}.
    """
    result = {}
    for cat in CATEGORIES:
        hu = set()
        for cn in history_configs:
            hu |= {b for b, s in ha_results[cn].get(cat, {}).items() if s}
        result[cat] = len(hu)
    result["Total"] = sum(result[cat] for cat in CATEGORIES)
    return result


def generate_ablation_table(
    d4j_results: dict,
    bip_results: dict,
    d4j_totals: dict,
    bip_totals: dict,
    d4j_unique: dict,
    bip_unique: dict,
):
    """Generate compact ablation table with configs as columns.

    Format: Dataset | Cat | Total | non-hist | fn_all [uniq] | fn_pair [uniq] | fl_diff [uniq]
    """
    lines = []
    lines.append(r"\begin{table*}[t]")
    lines.append(r"    \centering")
    lines.append(r"    \caption{History ablation on Defects4J (854 Java bugs) and BugsInPy (501 Python bugs). Each cell shows \#Pass (Plausible@1). Subscript = \#Unique bugs fixed by that config but not by \nonhistory. \textbf{Union} = number of bugs fixed by at least one of the three history heuristics (\fnall, \fnpair, \fldiff). Bold = best per category among the four configurations.}")
    lines.append(r"    \label{tab:rq1-ablation}")
    lines.append(r"    \begin{tabular}{l l r r r r r r}")
    lines.append(r"    \toprule")
    lines.append(r"    \textbf{Dataset} & \textbf{Category} & \textbf{Total} & \textbf{\nonhistory} & \textbf{\fnall} & \textbf{\fnpair} & \textbf{\fldiff} & \textbf{Union} \\")
    lines.append(r"    \midrule")

    for ds_name, ha_res, totals, uniq in [
        ("Defects4J", d4j_results, d4j_totals, d4j_unique),
        ("BugsInPy", bip_results, bip_totals, bip_unique),
    ]:
        first_ds_row = True
        union_by_cat = compute_union_by_category(ha_res)
        for cat in CATEGORIES:
            total = totals[cat]
            passes = {}
            for config_num in CONFIG_ORDER:
                passes[config_num] = sum(1 for s in ha_res[config_num].get(cat, {}).values() if s)

            best_pass = max(passes.values())

            cells = []
            for config_num in CONFIG_ORDER:
                p = passes[config_num]
                pct = p / total * 100 if total > 0 else 0
                cell = f"{p} ({pct:.1f}\\%)"
                if p == best_pass:
                    cell = r"\textbf{" + cell + "}"
                # Add unique count as subscript for history configs
                if config_num != "1":
                    u = uniq[config_num].get(cat, 0)
                    cell += f"$_{{+{u}}}$"
                cells.append(cell)

            ds_col = r"\multirow{5}{*}{" + ds_name + "}" if first_ds_row else ""
            lines.append(f"    {ds_col} & {cat} & {total} & {cells[0]} & {cells[1]} & {cells[2]} & {cells[3]} & {union_by_cat[cat]} \\\\")
            first_ds_row = False

        # Total row
        lines.append(r"    \cmidrule(lr){2-8}")
        total_all = sum(totals.values())
        total_passes = {}
        for config_num in CONFIG_ORDER:
            total_passes[config_num] = sum(
                sum(1 for s in ha_res[config_num].get(cat, {}).values() if s)
                for cat in CATEGORIES
            )
        best_total = max(total_passes.values())
        total_uniques = {}
        for cn in ["5", "7", "8"]:
            total_uniques[cn] = sum(uniq[cn].values())

        cells = []
        for config_num in CONFIG_ORDER:
            tp = total_passes[config_num]
            tpct = tp / total_all * 100 if total_all > 0 else 0
            cell = f"{tp} ({tpct:.1f}\\%)"
            if tp == best_total:
                cell = r"\textbf{" + cell + "}"
            if config_num != "1":
                tu = total_uniques.get(config_num, 0)
                cell += f"$_{{+{tu}}}$"
            cells.append(cell)

        lines.append(f"     & \\textbf{{Total}} & {total_all} & {cells[0]} & {cells[1]} & {cells[2]} & {cells[3]} & {union_by_cat['Total']} \\\\")

        if ds_name == "Defects4J":
            lines.append(r"    \midrule")

    lines.append(r"    \bottomrule")
    lines.append(r"    \end{tabular}")
    lines.append(r"\end{table*}")

    tex = "\n".join(lines)
    output_path = OUTPUT_DIR / "rq1_ablation_table.tex"
    with open(output_path, "w") as f:
        f.write(tex)
    print(f"\nSaved: {output_path}")
    return tex


def save_verification_csv(
    dataset_name: str,
    ha_results: dict,
    totals: dict,
    extra_cols: dict = None,
):
    """Save per-bug verification CSV for auditability."""
    rows = []
    for cat in CATEGORIES:
        for config_num in CONFIG_ORDER:
            for bug_id, success in sorted(ha_results[config_num].get(cat, {}).items()):
                row = {
                    "dataset": dataset_name,
                    "bug_id": bug_id,
                    "category": cat,
                    "config": CONFIG_MAP[config_num],
                    "config_num": config_num,
                    "success": success,
                }
                rows.append(row)

    df = pd.DataFrame(rows)
    output_path = OUTPUT_DIR / f"rq1_{dataset_name.lower()}_verification.csv"
    df.to_csv(output_path, index=False)
    print(f"Saved verification CSV: {output_path} ({len(df)} rows)")


def generate_venn_grid(ha_results: dict, dataset_name: str):
    """Generate 2x2 grid of 4-set Venn diagrams for one dataset (kept for standalone use)."""
    if not HAS_VENN:
        print(f"  Skipping Venn diagrams for {dataset_name} (venn package not installed)")
        return

    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    })

    categories = [
        ("SL", "a"), ("SH", "b"), ("SFMH", "c"), ("MFMH", "d"),
    ]

    config_labels = ["non-history", "fn_all", "fn_pair", "fl_diff"]
    venn_colors = [plt.cm.viridis(i / 3) for i in range(4)]

    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    axes = axes.flatten()

    for idx, (cat, label) in enumerate(categories):
        ax = axes[idx]
        labels_dict = {}
        for config_num, config_name in zip(CONFIG_ORDER, config_labels):
            successful = {
                bug for bug, success in ha_results[config_num].get(cat, {}).items()
                if success
            }
            labels_dict[config_name] = successful

        venn.venn(labels_dict, ax=ax, fontsize=11, legend_loc=None)
        ymin, ymax = ax.get_ylim()
        ax.set_ylim(ymin, ymin + (ymax - ymin) * 0.85)
        ax.text(0.5, 0.02, f"({label}) {cat}", transform=ax.transAxes,
                fontsize=14, va="bottom", ha="center")

    from matplotlib.patches import Patch
    legend_patches = [
        Patch(facecolor=venn_colors[i], alpha=0.4, label=lbl)
        for i, lbl in enumerate(config_labels)
    ]
    fig.legend(
        handles=legend_patches, loc="lower center",
        bbox_to_anchor=(0.5, -0.035), ncol=2, fontsize=13, frameon=True,
    )
    plt.subplots_adjust(left=0.05, right=0.95, top=0.7, bottom=0.05,
                        hspace=0, wspace=-0.35)

    output_path = OUTPUT_DIR / f"rq1_{dataset_name.lower()}_venn_grid.pdf"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved Venn grid: {output_path}")


def generate_combined_venn_grid(d4j_results: dict, bip_results: dict):
    """Generate a single 2-row × 4-col Venn grid: Defects4J top, BugsInPy bottom."""
    if not HAS_VENN:
        print("  Skipping combined Venn (venn package not installed)")
        return

    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    })

    categories = ["SL", "SH", "SFMH", "MFMH"]
    config_labels = ["non-history", "fn_all", "fn_pair", "fl_diff"]
    venn_colors = [plt.cm.viridis(i / 3) for i in range(4)]

    # 2 rows × 4 cols — reduced width for tighter column spacing
    fig, axes = plt.subplots(2, 4, figsize=(11, 5.6))

    datasets = [
        ("Defects4J", d4j_results),
        ("BugsInPy", bip_results),
    ]

    for row_idx, (ds_name, ha_res) in enumerate(datasets):
        for col_idx, cat in enumerate(categories):
            ax = axes[row_idx][col_idx]

            labels_dict = {}
            for config_num, config_name in zip(CONFIG_ORDER, config_labels):
                successful = {
                    bug for bug, success in ha_res[config_num].get(cat, {}).items()
                    if success
                }
                labels_dict[config_name] = successful

            venn.venn(labels_dict, ax=ax, fontsize=9, legend_loc=None)

            # Regular weight numbers, readable size
            for text in ax.texts:
                text.set_fontsize(10)

            # Tighten both axes to reduce whitespace
            ymin, ymax = ax.get_ylim()
            ax.set_ylim(ymin * 1.10, ymax * 0.98)
            xmin, xmax = ax.get_xlim()
            ax.set_xlim(xmin * 1.05, xmax * 0.95)

            # Column header (category) inside top of first row only
            if row_idx == 0:
                ax.text(0.5, 0.97, cat, transform=ax.transAxes,
                        fontsize=11, ha="center", va="top")

        # Row label on the left — vertically centered with Venn content
        axes[row_idx][0].annotate(
            ds_name, xy=(-0.22, 0.45),
            xycoords="axes fraction",
            fontsize=11,
            ha="center", va="center", rotation=90,
        )

    # The per-category non-history / history-union counts formerly drawn here as a
    # summary panel are now reported in the Union column of Table tab:rq1-ablation
    # (see compute_union_by_category); this figure shows only the Venn diagrams.

    # Shared legend below the Venn panels
    from matplotlib.patches import Patch
    legend_patches = [
        Patch(facecolor=venn_colors[i], alpha=0.4, label=lbl)
        for i, lbl in enumerate(config_labels)
    ]
    fig.legend(
        handles=legend_patches, loc="lower center",
        bbox_to_anchor=(0.5, -0.02), ncol=4, fontsize=9, frameon=True,
    )

    plt.subplots_adjust(left=0.06, right=0.99, top=0.97, bottom=0.10,
                        hspace=-0.12, wspace=-0.05)

    output_path = OUTPUT_DIR / "rq1_combined_venn_grid.pdf"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved combined Venn grid: {output_path}")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load category mappings
    print("Loading category mappings...")
    d4j_cats = load_category_mapping(D4J_CAT_CSV)
    bip_cats = load_category_mapping(BIP_CAT_CSV)
    print(f"  Defects4J: {len(d4j_cats)} bugs")
    print(f"  BugsInPy: {len(bip_cats)} bugs")

    # Compute totals per category
    d4j_totals = defaultdict(int)
    for cat in d4j_cats.values():
        d4j_totals[cat] += 1
    bip_totals = defaultdict(int)
    for cat in bip_cats.values():
        bip_totals[cat] += 1

    print(f"\n  D4J totals: {dict(d4j_totals)} = {sum(d4j_totals.values())}")
    print(f"  BIP totals: {dict(bip_totals)} = {sum(bip_totals.values())}")

    # --- HAFixAgent results ---
    print("\n--- HAFixAgent Defects4J ---")
    d4j_ha = count_hafixagent_results(D4J_RESULTS, "Defects4J")
    d4j_unique = compute_unique_pass(d4j_ha)
    print_and_save_summary("Defects4J", d4j_ha, dict(d4j_totals), d4j_unique)

    print("\n--- HAFixAgent BugsInPy ---")
    bip_ha = count_hafixagent_results(BIP_RESULTS, "BugsInPy")
    bip_unique = compute_unique_pass(bip_ha)
    print_and_save_summary("BugsInPy", bip_ha, dict(bip_totals), bip_unique)

    # --- Baselines ---
    print("\n--- RepairAgent (DeepSeek) on Defects4J ---")
    ra_results = count_repairagent_results(REPAIRAGENT_DIR, d4j_cats)
    ra_total = 0
    for cat in CATEGORIES:
        count = sum(1 for s in ra_results.get(cat, {}).values() if s)
        total = d4j_totals[cat]
        coverage = len(ra_results.get(cat, {}))
        print(f"  {cat}: {count}/{total} plausible (coverage: {coverage} bugs)")
        ra_total += count
    print(f"  Total: {ra_total}/{sum(d4j_totals.values())}")

    print("\n--- BIRCH-feedback (DeepSeek) on Defects4J ---")
    birch_results = count_birch_results(BIRCH_CSV, d4j_cats)
    birch_total = 0
    for cat in ["SFMH", "MFMH"]:
        count = sum(1 for s in birch_results.get(cat, {}).values() if s)
        coverage = len(birch_results.get(cat, {}))
        print(f"  {cat}: {count}/{coverage} plausible")
        birch_total += count
    print(f"  Total: {birch_total}/{sum(len(v) for v in birch_results.values())}")

    # --- Generate tables ---
    print("\n\n" + "=" * 70)
    print("Generating LaTeX tables")
    print("=" * 70)

    generate_baseline_comparison_table(d4j_ha, ra_results, birch_results, dict(d4j_totals))
    generate_ablation_table(d4j_ha, bip_ha, dict(d4j_totals), dict(bip_totals), d4j_unique, bip_unique)

    # --- Venn diagrams ---
    print("\n--- Generating Venn diagrams ---")
    generate_venn_grid(d4j_ha, "Defects4J")
    generate_venn_grid(bip_ha, "BugsInPy")
    generate_combined_venn_grid(d4j_ha, bip_ha)

    # --- Verification CSVs ---
    print("\n--- Saving verification CSVs ---")
    save_verification_csv("Defects4J", d4j_ha, dict(d4j_totals))
    save_verification_csv("BugsInPy", bip_ha, dict(bip_totals))

    # --- Cross-validation: check D4J numbers match paper ---
    print("\n\n" + "=" * 70)
    print("CROSS-VALIDATION against paper numbers")
    print("=" * 70)
    expected_d4j = {
        "1": {"SL": 137, "SH": 97, "SFMH": 238, "MFMH": 50},
        "5": {"SL": 136, "SH": 96, "SFMH": 251, "MFMH": 47},
        "7": {"SL": 136, "SH": 108, "SFMH": 249, "MFMH": 43},
        "8": {"SL": 140, "SH": 103, "SFMH": 254, "MFMH": 48},
    }
    all_match = True
    for config_num, expected_cats in expected_d4j.items():
        for cat, expected_count in expected_cats.items():
            actual = sum(1 for s in d4j_ha[config_num].get(cat, {}).values() if s)
            status = "OK" if actual == expected_count else "MISMATCH"
            if status == "MISMATCH":
                all_match = False
            print(f"  {CONFIG_MAP[config_num]:>11} {cat}: expected={expected_count}, actual={actual} [{status}]")

    if all_match:
        print("\n  ALL D4J numbers match paper. Verification PASSED.")
    else:
        print("\n  WARNING: Some numbers don't match!")


if __name__ == "__main__":
    main()
