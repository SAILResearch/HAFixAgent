"""
BugsInPy RQ0 Analysis: Blame Feasibility and Bug Categorization.

This module performs the RQ0 preliminary study for BugsInPy dataset:
1. Distribution of blameable vs blameless bugs
2. Distribution of unique blame commits per bug
3. Bug categorization by patch complexity (SL/SH/SFMH/MFMH)

Note: BugsInPy patches are FORWARD patches (buggy -> fixed), so '-' lines are blameable.
This is the same format as SWE-Bench, opposite to Defects4J reverse patches.
"""

import csv
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict

from .util import (
    get_all_bugs,
    get_patch_file_path,
    BUGSINPY_BASE_PATH,
)


def is_single_line_bug(patch_file: str) -> bool:
    """Check if a patch represents a single-line bug fix."""
    with open(patch_file, "r", encoding="utf-8", errors="replace") as file:
        lines = file.readlines()

    added_lines = 0
    removed_lines = 0
    diff_started = False
    changes = []

    for index, line in enumerate(lines):
        if line.startswith("@@"):
            diff_started = True
        elif diff_started:
            if line.startswith("+") and not line.startswith("+++"):
                added_lines += 1
                changes.append(index)
            elif line.startswith("-") and not line.startswith("---"):
                removed_lines += 1
                changes.append(index)

    total_changes = added_lines + removed_lines

    # Case 1: Exactly 1 line changed
    if total_changes == 1:
        return True

    # Case 2: 1 addition + 1 deletion (line modification)
    if added_lines == 1 and removed_lines == 1:
        if abs(changes[0] - changes[1]) == 1:
            return True

    return False


def is_single_hunk_bug(patch_file: str) -> bool:
    """Check if a patch is a single-hunk bug (consecutive changes in one hunk)."""
    with open(patch_file, "r", encoding="utf-8", errors="replace") as file:
        lines = file.readlines()

    hunk_started = False
    changes = []

    for index, line in enumerate(lines):
        if line.startswith("@@"):
            if hunk_started and changes:
                return False  # More than one hunk
            hunk_started = True
            changes = []
            continue

        if hunk_started:
            if line.startswith("+") and not line.startswith("+++"):
                changes.append(index)
            elif line.startswith("-") and not line.startswith("---"):
                changes.append(index)

    # Check if changes form a single contiguous block
    if changes and max(changes) - min(changes) + 1 == len(changes):
        return True
    return False


def analyze_patch_files(patch_file: str) -> Dict:
    """
    Patch analysis to categorize bugs by complexity.

    Returns:
        Dict with detailed categorization info
    """
    with open(patch_file, "r", encoding="utf-8", errors="replace") as file:
        lines = file.readlines()

    files_changed = set()
    hunks_per_file = defaultdict(int)
    lines_changed_per_file = defaultdict(int)
    total_added = 0
    total_removed = 0

    current_file = None
    current_hunk = False

    for line in lines:
        if line.startswith("diff --git"):
            parts = line.split()
            if len(parts) >= 4:
                current_file = parts[3].replace("b/", "")
                files_changed.add(current_file)

        elif line.startswith("@@"):
            current_hunk = True
            if current_file:
                hunks_per_file[current_file] += 1

        elif current_hunk and current_file:
            if line.startswith("+") and not line.startswith("+++"):
                total_added += 1
                lines_changed_per_file[current_file] += 1
            elif line.startswith("-") and not line.startswith("---"):
                total_removed += 1
                lines_changed_per_file[current_file] += 1

    return {
        "files_changed": len(files_changed),
        "files_list": list(files_changed),
        "hunks_per_file": dict(hunks_per_file),
        "total_hunks": sum(hunks_per_file.values()),
        "lines_added": total_added,
        "lines_removed": total_removed,
        "total_lines_changed": total_added + total_removed,
        "lines_changed_per_file": dict(lines_changed_per_file),
    }


def analyze_git_blame_feasibility(
    patch_file: str, patch_format: str = "swe_bench"
) -> Dict:
    """
    Analyze if a patch is suitable for git blame extraction.

    Args:
        patch_file: Path to patch file
        patch_format: Dataset format ("swe_bench" for forward patches like BugsInPy)

    Blame logic:
    - BugsInPy/SWE-Bench (forward): '-' lines are blameable
    - Defects4J (reverse): '+' lines are blameable

    Returns:
        Dict with blame feasibility analysis
    """
    with open(patch_file, "r", encoding="utf-8", errors="replace") as file:
        lines = file.readlines()

    pure_additions = 0
    pure_deletions = 0
    modifications = 0
    total_additions = 0
    total_deletions = 0

    consecutive_changes = []
    current_hunk_changes = []
    diff_started = False

    for i, line in enumerate(lines):
        if line.startswith("@@"):
            if current_hunk_changes:
                consecutive_changes.append(current_hunk_changes)
            current_hunk_changes = []
            diff_started = True

        elif diff_started:
            if line.startswith("+") and not line.startswith("+++"):
                total_additions += 1
                current_hunk_changes.append((i, "add"))
            elif line.startswith("-") and not line.startswith("---"):
                total_deletions += 1
                current_hunk_changes.append((i, "del"))

    if current_hunk_changes:
        consecutive_changes.append(current_hunk_changes)

    # Analyze modification patterns
    for hunk_changes in consecutive_changes:
        dels = [i for i, op in hunk_changes if op == "del"]
        adds = [i for i, op in hunk_changes if op == "add"]

        paired_changes = min(len(dels), len(adds))
        modifications += paired_changes
        pure_additions += len(adds) - paired_changes
        pure_deletions += len(dels) - paired_changes

    total_changes = total_additions + total_deletions

    # BugsInPy uses forward patches: '-' lines are blameable
    if patch_format.lower() == "swe_bench":
        blameable_changes = total_deletions
    elif patch_format.lower() == "defects4j":
        blameable_changes = total_additions
    else:
        raise ValueError(f"Unsupported patch format: {patch_format}")

    blame_feasibility = blameable_changes / total_changes if total_changes > 0 else 0.0

    return {
        "total_additions": total_additions,
        "total_deletions": total_deletions,
        "total_changes": total_changes,
        "pure_additions": pure_additions,
        "pure_deletions": pure_deletions,
        "modifications": modifications,
        "blameable_changes": blameable_changes,
        "blame_feasibility_ratio": blame_feasibility,
        "blame_suitable": blame_feasibility > 0.0,
        "change_pattern": "pure_addition"
        if pure_additions > 0 and pure_deletions == 0 and modifications == 0
        else "pure_deletion"
        if pure_deletions > 0 and pure_additions == 0 and modifications == 0
        else "mix",
        "patch_format": patch_format,
    }


def categorize_bugs(
    base_path: Path, all_bugs: Dict[str, List[int]]
) -> Tuple[Dict, Dict]:
    """
    Bug categorization with detailed multi-hunk analysis.

    Categories:
    1. Single-line bugs (1 line changed) - SUBSET of single-hunk
    2. Single-hunk bugs (1 hunk with CONSECUTIVE changes)
    3. Single-file multi-hunk bugs (multiple hunks, same file)
    4. Multi-file multi-hunk bugs (multiple hunks, different files)
    """

    categories = {
        "single_line": defaultdict(list),
        "single_hunk": defaultdict(list),
        "single_file_multi_hunk": defaultdict(list),
        "multi_file_multi_hunk": defaultdict(list),
    }

    detailed_analysis = {}

    for project_name, bug_ids in all_bugs.items():
        for bug_id in bug_ids:
            patch_file_path = get_patch_file_path(project_name, bug_id, base_path)

            if not patch_file_path.exists():
                continue

            analysis = analyze_patch_files(str(patch_file_path))
            detailed_analysis[f"{project_name}_{bug_id}"] = analysis

            if is_single_line_bug(str(patch_file_path)):
                categories["single_line"][project_name].append(bug_id)

            elif is_single_hunk_bug(str(patch_file_path)):
                categories["single_hunk"][project_name].append(bug_id)

            elif analysis["files_changed"] == 1:
                categories["single_file_multi_hunk"][project_name].append(bug_id)

            else:
                categories["multi_file_multi_hunk"][project_name].append(bug_id)

    return categories, detailed_analysis


def analyze_dataset_blame_feasibility(
    base_path: Path, all_bugs: Dict[str, List[int]]
) -> Tuple[Dict, Dict]:
    """Analyze git blame feasibility across the entire BugsInPy dataset."""

    print("BugsInPy Git Blame Feasibility Analysis")
    print("=" * 60)

    feasibility_stats = {
        "total_analyzed": 0,
        "blame_suitable": 0,
        "pure_additions": 0,
        "pure_deletions": 0,
        "mix": 0,
    }

    pattern_counts = defaultdict(int)
    feasibility_by_category = {
        "single_line": {"suitable": 0, "total": 0},
        "single_hunk": {"suitable": 0, "total": 0},
        "single_file_multi_hunk": {"suitable": 0, "total": 0},
        "multi_file_multi_hunk": {"suitable": 0, "total": 0},
    }

    # Get categorization for context
    categories, _ = categorize_bugs(base_path, all_bugs)

    # Create category lookup
    bug_to_category = {}
    for category, projects in categories.items():
        for project, bugs in projects.items():
            for bug in bugs:
                bug_to_category[f"{project}_{bug}"] = category

    # Analyze each bug
    detailed_results = {}

    for project_name, bug_ids in all_bugs.items():
        for bug_id in bug_ids:
            patch_file_path = get_patch_file_path(project_name, bug_id, base_path)

            if not patch_file_path.exists():
                continue

            # Analyze blame feasibility (BugsInPy uses forward patches like SWE-Bench)
            blame_analysis = analyze_git_blame_feasibility(
                str(patch_file_path), patch_format="swe_bench"
            )
            bug_key = f"{project_name}_{bug_id}"
            detailed_results[bug_key] = blame_analysis

            feasibility_stats["total_analyzed"] += 1

            pattern = blame_analysis["change_pattern"]
            pattern_counts[pattern] += 1

            if pattern == "pure_addition":
                feasibility_stats["pure_additions"] += 1
            elif pattern == "pure_deletion":
                feasibility_stats["pure_deletions"] += 1
            else:
                feasibility_stats["mix"] += 1

            if blame_analysis["blame_suitable"]:
                feasibility_stats["blame_suitable"] += 1

                category = bug_to_category.get(bug_key)
                if category and category in feasibility_by_category:
                    feasibility_by_category[category]["suitable"] += 1

            category = bug_to_category.get(bug_key)
            if category and category in feasibility_by_category:
                feasibility_by_category[category]["total"] += 1

    # Print results
    total = feasibility_stats["total_analyzed"]
    suitable = feasibility_stats["blame_suitable"]

    print(f"Total bugs analyzed: {total}")
    print(f"Blame-suitable bugs: {suitable} ({suitable / total:.1%})")
    print(f"")
    print(f"Change Patterns (BugsInPy forward patches):")
    print(
        f"  Pure additions:  {feasibility_stats['pure_additions']:3d} ({feasibility_stats['pure_additions'] / total:.1%}) - 0% blameable (forward patches)"
    )
    print(
        f"  Pure deletions:  {feasibility_stats['pure_deletions']:3d} ({feasibility_stats['pure_deletions'] / total:.1%}) - 100% blameable (forward patches)"
    )
    print(
        f"  Mix:             {feasibility_stats['mix']:3d} ({feasibility_stats['mix'] / total:.1%}) - Partially blameable"
    )

    print(f"\nBlame Feasibility by Bug Category:")
    for category, stats in feasibility_by_category.items():
        if stats["total"] > 0:
            ratio = stats["suitable"] / stats["total"]
            print(
                f"  {category:20s}: {stats['suitable']:3d}/{stats['total']:3d} ({ratio:.1%})"
            )

    # Show examples of unblameable cases
    unblameable_examples = []
    for bug_key, analysis in detailed_results.items():
        if analysis["change_pattern"] == "pure_addition":
            unblameable_examples.append(bug_key)

    if unblameable_examples:
        print(f"\nUnblameable Bug Examples (pure additions - 0% blameable):")
        for i, bug_id in enumerate(unblameable_examples[:5]):
            print(f"  {i + 1}. {bug_id}")
        if len(unblameable_examples) > 5:
            print(f"  ... and {len(unblameable_examples) - 5} more")

    return detailed_results, feasibility_stats


def save_categorization_to_csv(categories: Dict, output_path: str = None):
    """Save bug categorization results to CSV for paper tables."""

    if output_path is None:
        current_dir = Path(__file__).parent
        output_path = current_dir / "bugsinpy_categorization.csv"

    with open(output_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(
            [
                "Project",
                "Single_Line",
                "Single_Hunk",
                "Single_File_Multi_Hunk",
                "Multi_File_Multi_Hunk",
                "Total",
            ]
        )

        all_projects = set()
        for category_bugs in categories.values():
            all_projects.update(category_bugs.keys())

        for project in sorted(all_projects):
            sl = len(categories["single_line"].get(project, []))
            sh = len(categories["single_hunk"].get(project, []))
            sfmh = len(categories["single_file_multi_hunk"].get(project, []))
            mfmh = len(categories["multi_file_multi_hunk"].get(project, []))
            total = sl + sh + sfmh + mfmh

            writer.writerow([project, sl, sh, sfmh, mfmh, total])

        # Write totals row
        total_sl = sum(len(bugs) for bugs in categories["single_line"].values())
        total_sh = sum(len(bugs) for bugs in categories["single_hunk"].values())
        total_sfmh = sum(
            len(bugs) for bugs in categories["single_file_multi_hunk"].values()
        )
        total_mfmh = sum(
            len(bugs) for bugs in categories["multi_file_multi_hunk"].values()
        )
        grand_total = total_sl + total_sh + total_sfmh + total_mfmh

        writer.writerow(
            ["TOTAL", total_sl, total_sh, total_sfmh, total_mfmh, grand_total]
        )

    print(f"Categorization data saved to {output_path}")


def save_blame_feasibility_to_csv(
    detailed_results: Dict,
    feasibility_stats: Dict,
    categories: Dict,
    output_path: str = None,
):
    """Save blame feasibility analysis to CSV for paper tables and HAFixAgent evaluation."""

    if output_path is None:
        current_dir = Path(__file__).parent
        output_path = current_dir / "bugsinpy_blame_feasibility.csv"

    # Create bug category lookup
    bug_to_category = {}
    category_mapping = {
        "single_line": "SL",
        "single_hunk": "SH",
        "single_file_multi_hunk": "SFMH",
        "multi_file_multi_hunk": "MFMH",
    }

    for category, projects in categories.items():
        for project, bugs in projects.items():
            for bug in bugs:
                bug_to_category[f"{project}_{bug}"] = category_mapping[category]

    with open(output_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(
            [
                "Bug_ID",
                "Project",
                "Bug_Number",
                "Category",
                "Change_Pattern",
                "Blame_Suitable",
                "Feasibility_Ratio",
                "Total_Additions",
                "Total_Deletions",
            ]
        )

        for bug_key, analysis in detailed_results.items():
            project, bug_num = bug_key.split("_", 1)
            category = bug_to_category.get(bug_key, "Unknown")

            writer.writerow(
                [
                    bug_key,
                    project,
                    bug_num,
                    category,
                    analysis["change_pattern"],
                    1 if analysis["blame_suitable"] else 0,
                    f"{analysis['blame_feasibility_ratio']:.2f}",
                    analysis["total_additions"],
                    analysis["total_deletions"],
                ]
            )

    # Create cross-tabulation by category and blame status
    summary_path = str(output_path).replace(".csv", "_summary.csv")

    category_stats = {
        "SFMH": {"blameable": 0, "blameless": 0},
        "MFMH": {"blameable": 0, "blameless": 0},
        "SH": {"blameable": 0, "blameless": 0},
        "SL": {"blameable": 0, "blameless": 0},
    }

    for bug_key, analysis in detailed_results.items():
        category = bug_to_category.get(bug_key, "Unknown")
        if category in category_stats:
            blame_status = "blameable" if analysis["blame_suitable"] else "blameless"
            category_stats[category][blame_status] += 1

    with open(summary_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["", "SFMH", "MFMH", "SH", "SL", "total"])

        blameable_counts = [
            category_stats[cat]["blameable"] for cat in ["SFMH", "MFMH", "SH", "SL"]
        ]
        blameable_total = sum(blameable_counts)
        writer.writerow(["blameable"] + blameable_counts + [blameable_total])

        blameless_counts = [
            category_stats[cat]["blameless"] for cat in ["SFMH", "MFMH", "SH", "SL"]
        ]
        blameless_total = sum(blameless_counts)
        writer.writerow(["blameless"] + blameless_counts + [blameless_total])

        total_counts = [
            category_stats[cat]["blameable"] + category_stats[cat]["blameless"]
            for cat in ["SFMH", "MFMH", "SH", "SL"]
        ]
        grand_total = sum(total_counts)
        writer.writerow(["total"] + total_counts + [grand_total])

    print("\nBlame Feasibility by Category:")
    print(f"{'':12}{'SFMH':8}{'MFMH':8}{'SH':8}{'SL':8}{'total':8}")
    print(
        f"{'blameable':12}{blameable_counts[0]:8}{blameable_counts[1]:8}{blameable_counts[2]:8}{blameable_counts[3]:8}{blameable_total:8}"
    )
    print(
        f"{'blameless':12}{blameless_counts[0]:8}{blameless_counts[1]:8}{blameless_counts[2]:8}{blameless_counts[3]:8}{blameless_total:8}"
    )
    print(
        f"{'total':12}{total_counts[0]:8}{total_counts[1]:8}{total_counts[2]:8}{total_counts[3]:8}{grand_total:8}"
    )

    print(f"\nBlame feasibility data saved to {output_path}")
    print(f"Category cross-tabulation saved to {summary_path}")


if __name__ == "__main__":
    print("BugsInPy Bug Categorization and Blame Feasibility Analysis")
    print("=" * 80)

    all_bugs = get_all_bugs()
    total_bugs = sum(len(bugs) for bugs in all_bugs.values())

    print(f"Total bugs discovered: {total_bugs}")
    print(f"Projects: {len(all_bugs)}")

    print(f"\nBugs per Project:")
    for project, bugs in sorted(all_bugs.items(), key=lambda x: -len(x[1])):
        print(f"  {project}: {len(bugs)} bugs")

    # Categorize bugs
    categories, detailed = categorize_bugs(BUGSINPY_BASE_PATH, all_bugs)

    sl_total = sum(len(bugs) for bugs in categories["single_line"].values())
    sh_total = sum(len(bugs) for bugs in categories["single_hunk"].values())
    sfmh_total = sum(
        len(bugs) for bugs in categories["single_file_multi_hunk"].values()
    )
    mfmh_total = sum(len(bugs) for bugs in categories["multi_file_multi_hunk"].values())

    print(f"\nBug Categories:")
    print(f"  Single-line (SL):           {sl_total}")
    print(f"  Single-hunk (SH):           {sh_total}")
    print(f"  Single-file multi-hunk (SFMH): {sfmh_total}")
    print(f"  Multi-file multi-hunk (MFMH):  {mfmh_total}")
    print(f"  Total: {sl_total + sh_total + sfmh_total + mfmh_total}")

    # Analyze git blame feasibility
    print(f"\n")
    blame_results, blame_stats = analyze_dataset_blame_feasibility(
        BUGSINPY_BASE_PATH, all_bugs
    )

    # Save results to CSV files
    print(f"\n")
    print("Saving analysis results to CSV files...")
    save_categorization_to_csv(categories)
    save_blame_feasibility_to_csv(blame_results, blame_stats, categories)

    # Per-project breakdown
    print(f"\nPer-Project Breakdown:")
    for project in sorted(all_bugs.keys()):
        sl = len(categories["single_line"].get(project, []))
        sh = len(categories["single_hunk"].get(project, []))
        sfmh = len(categories["single_file_multi_hunk"].get(project, []))
        mfmh = len(categories["multi_file_multi_hunk"].get(project, []))
        total_proj = sl + sh + sfmh + mfmh

        print(
            f"{project:15s}: {total_proj:4d} total | SL:{sl:4d} SH:{sh:4d} SFMH:{sfmh:4d} MFMH:{mfmh:4d}"
        )
