#!/usr/bin/env python3
"""
RQ2 baseline significance: McNemar exact test of HAFixAgent vs. external baselines.

Answers the reviewer question "are these improvements statistically significant?"
for the RepairAgent and BIRCH-feedback comparisons. All four HAFixAgent
configurations (non-history, fn_all, fn_pair, fl_diff) are tested against each
baseline on the exact bug set the baseline was evaluated on.

Self-contained: reads only from the committed results/ tree (no RepairAgent repo
required). Plausible@1 is the metric for every system, matching the paper.

  - HAFixAgent : results/defects4j/llm_judge_1line/<category>/progress_<flag>_both.json
  - RepairAgent: results/defects4j/baselines/repairagent_deepseek_deepseek-v3.2-exp/<Bug>/
                 (plausible == non-empty plausible_patches/)
  - BIRCH      : results/defects4j/baselines/birch_deepseek/test_results_mode_4_merged.csv
                 (plausible == pass == "Yes"), on its own 371 multi-hunk subset.

Usage: python analysis/analyze_rq1_baseline_significance.py
"""

import os
import csv
import sys
import json
import argparse
from pathlib import Path

from scipy.stats import binomtest

# Repo-root-relative default (analysis/ -> repo root)
REPO_ROOT = Path(__file__).resolve().parents[1]

CATEGORIES = [
    "single_line",
    "single_hunk",
    "single_file_multi_hunk",
    "multi_file_multi_hunk",
]
MULTI_HUNK = ["single_file_multi_hunk", "multi_file_multi_hunk"]

# config name -> history flag (see hafix_agent.blame.core.history_name_to_category)
CONFIG_FLAGS = {
    "non-history": "1",
    "fn_all": "5",
    "fn_pair": "7",
    "fl_diff": "8",
}


def load_hafix(hafix_dir: Path, flag: str, categories):
    """Return (pass_set, universe_set) for one config across the given categories."""
    passed, universe = set(), set()
    for cat in categories:
        f = hafix_dir / cat / f"progress_{flag}_both.json"
        data = json.loads(f.read_text())
        succ = set(data.get("successful_bugs", []))
        failed = set(sum((b for b in data.get("failed_bugs", {}).values()), []))
        passed |= succ
        universe |= succ | failed
    return passed, universe


def load_repairagent(ra_dir: Path):
    """Plausible == non-empty plausible_patches/ directory."""
    passed, evaluated = set(), set()
    for d in os.listdir(ra_dir):
        dp = ra_dir / d
        if not dp.is_dir():
            continue
        evaluated.add(d)
        pp = dp / "plausible_patches"
        if pp.is_dir() and any(pp.iterdir()):
            passed.add(d)
    return passed, evaluated


def load_birch(csv_path: Path):
    """Plausible == pass column == 'Yes'."""
    csv.field_size_limit(sys.maxsize)  # failed_tests field can be very large
    passed, evaluated = set(), set()
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            bug = row["bug"].strip().replace("-", "_").replace(" ", "_")
            evaluated.add(bug)
            if row["pass"].strip().lower() == "yes":
                passed.add(bug)
    return passed, evaluated


def mcnemar_exact(a_pass, b_pass, universe):
    """Exact (binomial) McNemar over discordant pairs restricted to `universe`.

    Returns (b, c, p) where b = A-pass & B-fail, c = A-fail & B-pass.
    """
    b = c = 0
    for bug in universe:
        ap, bp = bug in a_pass, bug in b_pass
        if ap and not bp:
            b += 1
        elif bp and not ap:
            c += 1
    n = b + c
    p = binomtest(min(b, c), n, 0.5, alternative="two-sided").pvalue if n else 1.0
    return b, c, p


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results-dir", type=Path,
                    default=REPO_ROOT / "results" / "defects4j",
                    help="Base Defects4J results directory")
    args = ap.parse_args()

    base = args.results_dir
    hafix_dir = base / "llm_judge_1line"
    ra_dir = base / "baselines" / "repairagent_deepseek_deepseek-v3.2-exp"
    birch_csv = base / "baselines" / "birch_deepseek" / "test_results_mode_4_merged.csv"

    # ---- RepairAgent: all 854 Defects4J bugs ----
    ra_pass, _ = load_repairagent(ra_dir)
    hafix_all = {n: load_hafix(hafix_dir, f, CATEGORIES) for n, f in CONFIG_FLAGS.items()}
    universe_854 = hafix_all["non-history"][1]

    print("=" * 68)
    print("HAFixAgent vs. RepairAgent  (all 854 Defects4J bugs, plausible@1)")
    print("=" * 68)
    print(f"RepairAgent plausible: {len(ra_pass & universe_854)} / {len(universe_854)}")
    for name in CONFIG_FLAGS:
        hp = hafix_all[name][0]
        print(f"  HAFixAgent-{name:11s} plausible: {len(hp & universe_854)}")
    print()
    for name in CONFIG_FLAGS:
        hp = hafix_all[name][0]
        b, c, p = mcnemar_exact(hp, ra_pass, universe_854)
        star = "***" if p < 1e-3 else ("**" if p < 1e-2 else ("*" if p < 5e-2 else "ns"))
        print(f"  {name:11s} vs RepairAgent: HAFix-only={b:3d} RA-only={c:3d}  p={p:.2e} {star}")

    # ---- BIRCH-feedback: its own 371 multi-hunk subset ----
    birch_pass, birch_eval = load_birch(birch_csv)
    hafix_mh = {n: load_hafix(hafix_dir, f, MULTI_HUNK) for n, f in CONFIG_FLAGS.items()}
    universe_371 = birch_eval & hafix_mh["non-history"][1]

    print()
    print("=" * 68)
    print("HAFixAgent vs. BIRCH-feedback  (371 multi-hunk bugs, plausible@1)")
    print("=" * 68)
    print(f"BIRCH-feedback plausible: {len(birch_pass & universe_371)} / {len(universe_371)}")
    for name in CONFIG_FLAGS:
        hp = hafix_mh[name][0]
        print(f"  HAFixAgent-{name:11s} plausible: {len(hp & universe_371)}")
    print()
    for name in CONFIG_FLAGS:
        hp = hafix_mh[name][0] & universe_371
        b, c, p = mcnemar_exact(hp, birch_pass, universe_371)
        star = "***" if p < 1e-3 else ("**" if p < 1e-2 else ("*" if p < 5e-2 else "ns"))
        print(f"  {name:11s} vs BIRCH:       HAFix-only={b:3d} BIRCH-only={c:3d}  p={p:.2e} {star}")


if __name__ == "__main__":
    main()
