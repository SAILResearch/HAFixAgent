#!/usr/bin/env python3
"""
RQ2b Cost/Efficiency Analysis: Cross-language cost and step comparison.

Extracts cost and step metrics from both Defects4J and BugsInPy results.
Generates compact summary table for the paper.

Usage:
    python -m analysis.analyze_rq3_cost_efficiency
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "results" / "rq2_combined"

D4J_RESULTS = PROJECT_ROOT / "results" / "defects4j" / "llm_judge_1line"
BIP_RESULTS = PROJECT_ROOT / "results" / "bugsinpy" / "llm_judge_1line"

CATEGORIES = ["SL", "SH", "SFMH", "MFMH"]
CATEGORY_DIR_MAP = {
    "SL": "single_line",
    "SH": "single_hunk",
    "SFMH": "single_file_multi_hunk",
    "MFMH": "multi_file_multi_hunk",
}

CONFIG_ORDER = ["1", "5", "7", "8"]
CONFIG_NAMES = ["non-history", "fn_all", "fn_pair", "fl_diff"]


def extract_metrics(results_dir: Path):
    """Extract cost and step metrics from result JSONs.

    Returns: list of dicts with keys: bug_id, category, config, success, cost, steps
    """
    rows = []
    for cat, dirname in CATEGORY_DIR_MAP.items():
        cat_dir = results_dir / dirname
        if not cat_dir.exists():
            continue

        for bug_dir in sorted(cat_dir.iterdir()):
            if not bug_dir.is_dir():
                continue
            bug_name = bug_dir.name

            for config_num in CONFIG_ORDER:
                result_file = bug_dir / f"{bug_name}_{config_num}_result.json"
                if not result_file.exists():
                    continue

                try:
                    with open(result_file) as f:
                        data = json.load(f)
                    rr = data.get("repair_result", {})
                    success = rr.get("success", False)
                    cost = rr.get("model_cost", 0) or 0
                    steps = rr.get("agent_steps", 0) or 0

                    rows.append({
                        "bug_id": bug_name,
                        "category": cat,
                        "config": CONFIG_NAMES[CONFIG_ORDER.index(config_num)],
                        "config_num": config_num,
                        "success": success,
                        "cost": cost,
                        "steps": steps,
                    })
                except (json.JSONDecodeError, KeyError) as e:
                    pass

    return pd.DataFrame(rows)


def summarize_metrics(df: pd.DataFrame, dataset_name: str):
    """Print and return summary table."""
    lines = []
    lines.append(f"\n{'='*80}")
    lines.append(f"Cost/Step Summary: {dataset_name}")
    lines.append(f"{'='*80}")

    for cat in CATEGORIES:
        lines.append(f"\n--- {cat} ---")
        lines.append(f"{'Config':<12} {'N_succ':>6} {'N_fail':>6} | {'Med Cost (S)':>12} {'Med Cost (F)':>12} | {'Med Steps (S)':>14} {'Med Steps (F)':>14}")
        lines.append("-" * 95)

        for config_name in CONFIG_NAMES:
            cat_cfg = df[(df["category"] == cat) & (df["config"] == config_name)]
            succ = cat_cfg[cat_cfg["success"] == True]
            fail = cat_cfg[cat_cfg["success"] == False]

            n_succ = len(succ)
            n_fail = len(fail)
            med_cost_s = succ["cost"].median() if n_succ > 0 else float("nan")
            med_cost_f = fail["cost"].median() if n_fail > 0 else float("nan")
            med_steps_s = succ["steps"].median() if n_succ > 0 else float("nan")
            med_steps_f = fail["steps"].median() if n_fail > 0 else float("nan")

            lines.append(
                f"{config_name:<12} {n_succ:>6} {n_fail:>6} | "
                f"${med_cost_s:>10.4f}  ${med_cost_f:>10.4f} | "
                f"{med_steps_s:>13.0f}  {med_steps_f:>13.0f}"
            )

    text = "\n".join(lines)
    print(text)
    return text


def compute_friedman_tests(df: pd.DataFrame, dataset_name: str):
    """Compute Friedman tests on matched set (bugs fixed by all 4 configs)."""
    results = {}
    print(f"\n  Friedman tests for {dataset_name}:")
    for cat in CATEGORIES:
        cat_df = df[df["category"] == cat]
        bugs_by_config = {}
        for cn in CONFIG_NAMES:
            bugs_by_config[cn] = set(
                cat_df[(cat_df["config"] == cn) & (cat_df["success"] == True)]["bug_id"]
            )
        matched = bugs_by_config["non-history"]
        for cn in CONFIG_NAMES[1:]:
            matched = matched & bugs_by_config[cn]
        n = len(matched)
        results[cat] = {}

        if n < 4:
            for metric in ["cost", "steps"]:
                results[cat][metric] = {"friedman_p": float("nan"), "n": n, "pairwise": {}}
            continue

        bug_order = sorted(matched)
        matched_df = cat_df[cat_df["bug_id"].isin(matched) & (cat_df["success"] == True)]

        for metric in ["cost", "steps"]:
            data_by_config = {}
            for cn in CONFIG_NAMES:
                cfg_df = matched_df[matched_df["config"] == cn].set_index("bug_id")
                data_by_config[cn] = [cfg_df.loc[b, metric] for b in bug_order]

            friedman_stat, friedman_p = stats.friedmanchisquare(
                data_by_config["non-history"], data_by_config["fn_all"],
                data_by_config["fn_pair"], data_by_config["fl_diff"],
            )
            pairwise = {}
            if friedman_p < 0.05:
                for h in ["fn_all", "fn_pair", "fl_diff"]:
                    _, p_val = stats.wilcoxon(data_by_config["non-history"], data_by_config[h])
                    pairwise[h] = p_val
            results[cat][metric] = {"friedman_p": friedman_p, "n": n, "pairwise": pairwise}

        print(f"    {cat} (N={n}): cost p={results[cat]['cost']['friedman_p']:.4f}, "
              f"steps p={results[cat]['steps']['friedman_p']:.4f}")
        # Surface pairwise Wilcoxon post-hoc (non-history vs each history) for any
        # metric whose Friedman omnibus is significant (Bonferroni alpha = 0.05/3).
        for metric in ["cost", "steps"]:
            pw = results[cat][metric]["pairwise"]
            if pw:
                flags = ", ".join(
                    f"{h} p={p:.4f}{'*' if p < 0.05 / 3 else ''}" for h, p in pw.items()
                )
                print(f"        {metric} pairwise vs non-history: {flags}")
    return results


def format_p(p):
    if np.isnan(p):
        return "-"
    if p < 0.001:
        return "$<$0.001"
    return f"{p:.3f}"


def generate_merged_table(d4j_df, bip_df, d4j_tests, bip_tests):
    """Generate merged table with medians + Friedman p-values."""
    lines = []
    lines.append(r"\begin{table*}[t]")
    lines.append(r"    \centering")
    lines.append(r"    \caption{Median cost (USD) and agent steps for successful repairs, with Friedman test p-values for cost on the matched set (N = bugs fixed by all 4 configs). Bold p-values indicate significance ($p < 0.05$); $^{*}$ marks configs with significantly higher cost than \nonhistory (pairwise Wilcoxon, Bonferroni $\alpha = 0.0167$).}")
    lines.append(r"    \label{tab:rq2b-cost-crosslang}")
    lines.append(r"    \setlength{\tabcolsep}{4pt}")
    lines.append(r"    \small")
    lines.append(r"    \begin{tabular}{l l r r r r r r r r r r}")
    lines.append(r"    \toprule")
    lines.append(r"    & & & \multicolumn{4}{c}{\textbf{Median Cost (USD)}} & \multicolumn{4}{c}{\textbf{Median Steps}} \\")
    lines.append(r"    \cmidrule(lr){4-7} \cmidrule(lr){8-11}")
    lines.append(r"    \textbf{Dataset} & \textbf{Cat.} & \textbf{N} & \nonhistory & \fnall & \fnpair & \fldiff & \nonhistory & \fnall & \fnpair & \fldiff & \textbf{Fried. p} \\")
    lines.append(r"    \midrule")

    bonferroni_alpha = 0.05 / 3

    for ds_name, df, tests in [("Defects4J", d4j_df, d4j_tests), ("BugsInPy", bip_df, bip_tests)]:
        first_row = True
        for cat in CATEGORIES:
            cat_df = df[df["category"] == cat]
            n = tests[cat]["cost"]["n"]
            pairwise = tests[cat]["cost"].get("pairwise", {})

            cost_cells = []
            step_cells = []
            for cn in CONFIG_NAMES:
                succ_df = cat_df[(cat_df["config"] == cn) & (cat_df["success"] == True)]
                if len(succ_df) > 0:
                    med_cost = f"{succ_df['cost'].median():.3f}"
                    # Add star if pairwise Wilcoxon is significant for this config
                    if cn != "non-history" and cn in pairwise and pairwise[cn] < bonferroni_alpha:
                        med_cost += "$^{*}$"
                    cost_cells.append(med_cost)
                    step_cells.append(f"{succ_df['steps'].median():.0f}")
                else:
                    cost_cells.append("-")
                    step_cells.append("-")

            fp = tests[cat]["cost"]["friedman_p"]
            fp_str = format_p(fp)
            if not np.isnan(fp) and fp < 0.05:
                fp_str = r"\textbf{" + fp_str + "}"

            ds_col = r"\multirow{4}{*}{" + ds_name + "}" if first_row else ""
            lines.append(
                f"    {ds_col} & {cat} & {n} & "
                + " & ".join(cost_cells) + " & "
                + " & ".join(step_cells) + f" & {fp_str} \\\\"
            )
            first_row = False
        if ds_name == "Defects4J":
            lines.append(r"    \midrule")

    lines.append(r"    \bottomrule")
    lines.append(r"    \end{tabular}")
    lines.append(r"\end{table*}")

    tex = "\n".join(lines)
    output_path = OUTPUT_DIR / "rq2b_cost_table_merged.tex"
    with open(output_path, "w") as f:
        f.write(tex)
    print(f"\nSaved merged table: {output_path}")
    return tex


def generate_cost_table_latex(d4j_df: pd.DataFrame, bip_df: pd.DataFrame):
    """Generate compact LaTeX cost/step table for both datasets.

    Uses full set (all successful repairs per config), NOT matched set.
    This is consistent with the violin/box plot figure median tables.
    """
    lines = []
    lines.append(r"\begin{table*}[t]")
    lines.append(r"    \centering")
    lines.append(r"    \caption{Median cost (USD) and agent steps for successful repairs on Defects4J and BugsInPy. N = number of successful repairs per configuration.}")
    lines.append(r"    \label{tab:rq2b-cost}")
    lines.append(r"    \setlength{\tabcolsep}{5pt}")
    lines.append(r"    \begin{tabular}{l l r r r r r r r r}")
    lines.append(r"    \toprule")
    lines.append(r"    & & \multicolumn{4}{c}{\textbf{Median Cost (USD)}} & \multicolumn{4}{c}{\textbf{Median Steps}} \\")
    lines.append(r"    \cmidrule(lr){3-6} \cmidrule(lr){7-10}")
    lines.append(r"    \textbf{Dataset} & \textbf{Cat.} & \nonhistory & \fnall & \fnpair & \fldiff & \nonhistory & \fnall & \fnpair & \fldiff \\")
    lines.append(r"    \midrule")

    for ds_name, df in [("Defects4J", d4j_df), ("BugsInPy", bip_df)]:
        first_row = True
        for cat in CATEGORIES:
            cat_df = df[df["category"] == cat]

            cost_cells = []
            step_cells = []
            for cn in CONFIG_NAMES:
                succ_df = cat_df[(cat_df["config"] == cn) & (cat_df["success"] == True)]
                n_succ = len(succ_df)
                if n_succ > 0:
                    med_cost = succ_df["cost"].median()
                    med_steps = succ_df["steps"].median()
                    cost_cells.append(f"{med_cost:.3f}")
                    step_cells.append(f"{med_steps:.0f}")
                else:
                    cost_cells.append("-")
                    step_cells.append("-")

            ds_col = r"\multirow{4}{*}{" + ds_name + "}" if first_row else ""
            lines.append(
                f"    {ds_col} & {cat} & "
                + " & ".join(cost_cells)
                + " & " + " & ".join(step_cells) + r" \\"
            )
            first_row = False

        if ds_name == "Defects4J":
            lines.append(r"    \midrule")

    lines.append(r"    \bottomrule")
    lines.append(r"    \end{tabular}")
    lines.append(r"\end{table*}")

    tex = "\n".join(lines)
    output_path = OUTPUT_DIR / "rq2b_cost_table.tex"
    with open(output_path, "w") as f:
        f.write(tex)
    print(f"\nSaved: {output_path}")
    return tex


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Extracting Defects4J metrics...")
    d4j_df = extract_metrics(D4J_RESULTS)
    print(f"  {len(d4j_df)} rows")

    print("Extracting BugsInPy metrics...")
    bip_df = extract_metrics(BIP_RESULTS)
    print(f"  {len(bip_df)} rows")

    summarize_metrics(d4j_df, "Defects4J")
    summarize_metrics(bip_df, "BugsInPy")

    print("\n--- Generating LaTeX table ---")
    generate_cost_table_latex(d4j_df, bip_df)

    print("\n--- Computing Friedman tests ---")
    d4j_tests = compute_friedman_tests(d4j_df, "Defects4J")
    bip_tests = compute_friedman_tests(bip_df, "BugsInPy")

    print("\n--- Generating merged table (medians + Friedman) ---")
    generate_merged_table(d4j_df, bip_df, d4j_tests, bip_tests)

    # Save raw data for verification
    d4j_df.to_csv(OUTPUT_DIR / "rq2b_defects4j_metrics.csv", index=False)
    bip_df.to_csv(OUTPUT_DIR / "rq2b_bugsinpy_metrics.csv", index=False)
    print(f"\nSaved verification CSVs")


if __name__ == "__main__":
    main()
