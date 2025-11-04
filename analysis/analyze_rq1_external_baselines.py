#!/usr/bin/env python3
"""
HAFixAgent External Baseline Comparison (RQ1.1)

This script compares HAFixAgent configurations against external SOTA baselines
(RepairAgent and HUNK4J) on common bug sets, generating comparison tables and CSV outputs.
"""

import json
import csv
from pathlib import Path
from typing import Dict, Set
import argparse

from utils import (
    load_progress_data,
    load_hunk4j_data,
    find_common_bugs_with_hunk4j,
    get_history_flag,
    get_history_name,
    resolve_config_label,
)

BASELINE_FLAG = get_history_flag('baseline')
FN_ALL_FLAG = get_history_flag('fn_all')
FN_PAIR_FLAG = get_history_flag('fn_pair')
FL_DIFF_FLAG = get_history_flag('fl_diff')

BIRCH_BASELINE_LABEL = resolve_config_label('baseline_birch_fb', default='BIRCH-feedback')


def normalize_bug_id(bug_id: str, format: str = 'underscore') -> str:
    """
    Normalize bug ID between different formats.

    Args:
        bug_id: Bug ID to normalize
        format: Target format ('underscore' for Chart_1, 'dash' for Chart-1, 'space' for Chart 1)

    Returns:
        Normalized bug ID
    """
    if format == 'underscore':
        return bug_id.replace('-', '_').replace(' ', '_')
    elif format == 'dash':
        return bug_id.replace('_', '-').replace(' ', '-')
    elif format == 'space':
        return bug_id.replace('-', ' ').replace('_', ' ')
    else:
        raise ValueError(f"Unknown format: {format}")


def parse_repairagent_bugs(batches_dir: Path, deprecated_list: Set[str]) -> Set[str]:
    """
    Parse RepairAgent evaluated bugs from batch files, excluding deprecated cases.

    Args:
        batches_dir: Directory containing batch files (0, 1, 2, 3, 4)
        deprecated_list: Set of deprecated bug IDs (in space format like 'Chart 1')

    Returns:
        Set of bug IDs in our format (e.g., 'Chart_1')
    """
    evaluated_bugs = set()

    # Read all batch files
    batch_files = sorted(batches_dir.glob('[0-9]*'))

    for batch_file in batch_files:
        with open(batch_file, 'r') as f:
            content = f.read().strip()
            # Each line is a bug ID in format "Project BugNum" (e.g., "Chart 1")
            # Lines are separated by double newlines
            bug_ids = [line.strip() for line in content.split('\n\n') if line.strip()]

            for bug_id in bug_ids:
                # Skip deprecated bugs
                if bug_id in deprecated_list:
                    continue

                # Convert to our format (Chart 1 -> Chart_1)
                normalized = normalize_bug_id(bug_id, format='underscore')
                evaluated_bugs.add(normalized)

    return evaluated_bugs


def get_repairagent_deprecated_bugs() -> Set[str]:
    """
    Get the list of deprecated bugs from RepairAgent evaluation.
    These bugs should be excluded from the common set.

    Returns:
        Set of deprecated bug IDs in space format (e.g., 'Chart 1')
    """
    deprecated = [
        "Cli 6", "Closure 63", "Closure 93",
        "Collections 1", "Collections 2", "Collections 3", "Collections 4",
        "Collections 5", "Collections 10", "Collections 15", "Collections 20",
        "Collections 6", "Collections 11", "Collections 16", "Collections 21",
        "Collections 7", "Collections 12", "Collections 17", "Collections 22",
        "Collections 8", "Collections 13", "Collections 18", "Collections 23",
        "Collections 9", "Collections 14", "Collections 19", "Collections 24",
        "Lang 2", "Mockito 21"
    ]
    return set(deprecated)


def load_repairagent_results(repairagent_base_dir: Path) -> Set[str]:
    """
    Load RepairAgent successful repair results.

    Args:
        repairagent_base_dir: Base directory of RepairAgent repo

    Returns:
        Set of successfully repaired bug IDs in our format
    """
    # Load successful repairs from final_list_of_fixed_bugs
    results_file = repairagent_base_dir / "data" / "final_list_of_fixed_bugs"

    if not results_file.exists():
        raise FileNotFoundError(f"RepairAgent results file not found at {results_file}")

    successful = set()
    with open(results_file, 'r') as f:
        for line in f:
            bug_id = line.strip()
            if bug_id:
                # Normalize to our format (Chart 1 -> Chart_1)
                normalized = normalize_bug_id(bug_id, format='underscore')
                successful.add(normalized)

    return successful


def classify_by_category(bug_id: str, hafix_base_dir: Path,
                        selector_type: str, n_lines: int) -> str:
    """
    Classify a bug into one of the four categories: SL, SH, SFMH, MFMH.

    Args:
        bug_id: Bug ID to classify
        hafix_base_dir: Base HAFixAgent results directory
        selector_type: Line selection strategy
        n_lines: Number of lines selected

    Returns:
        Category name: 'SL', 'SH', 'SFMH', or 'MFMH'
    """
    selector_dir = f"{selector_type}_{n_lines}line"

    # Map category dirs to category codes
    category_mapping = {
        'single_line': 'SL',
        'single_hunk': 'SH',
        'single_file_multi_hunk': 'SFMH',
        'multi_file_multi_hunk': 'MFMH'
    }

    # Check which category directory contains this bug
    for cat_dir, cat_code in category_mapping.items():
        category_path = hafix_base_dir / selector_dir / cat_dir

        # Check progress files for this bug
        for pattern in ['progress_*_both.json', 'progress_*_blameable.json', 'progress_*_blameless.json']:
            files = list(category_path.glob(pattern))
            if files:
                with open(files[0], 'r') as f:
                    data = json.load(f)
                    all_bugs = set(data.get('successful_bugs', []))
                    for status, failed_list in data.get('failed_bugs', {}).items():
                        all_bugs.update(failed_list)

                    if bug_id in all_bugs:
                        return cat_code
                break  # Only check first file per pattern

    # Default to unknown
    return 'Unknown'


def find_common_bugs_with_repairagent(hafix_base_dir: Path, repairagent_bugs: Set[str],
                                      selector_type: str, n_lines: int) -> Set[str]:
    """
    Find bugs that are common between HAFixAgent and RepairAgent evaluations.

    Args:
        hafix_base_dir: Base HAFixAgent results directory
        repairagent_bugs: Set of bug IDs evaluated by RepairAgent
        selector_type: Line selection strategy
        n_lines: Number of lines selected

    Returns:
        Set of common bug IDs
    """
    selector_dir = f"{selector_type}_{n_lines}line"

    # Collect all HAFixAgent bugs
    all_hafix_bugs = set()
    for category in ['single_line', 'single_hunk', 'single_file_multi_hunk', 'multi_file_multi_hunk']:
        category_dir = hafix_base_dir / selector_dir / category

        # Try _both.json files first
        both_files = list(category_dir.glob("progress_*_both.json"))
        if both_files:
            with open(both_files[0], 'r') as f:
                data = json.load(f)
                all_hafix_bugs.update(data.get("successful_bugs", []))
                for status, failed_list in data.get("failed_bugs", {}).items():
                    all_hafix_bugs.update(failed_list)
        else:
            # Fall back to blameable/blameless files
            for pattern in ["progress_*_blameable.json", "progress_*_blameless.json"]:
                files = list(category_dir.glob(pattern))
                if files:
                    with open(files[0], 'r') as f:
                        data = json.load(f)
                        all_hafix_bugs.update(data.get("successful_bugs", []))
                        for status, failed_list in data.get("failed_bugs", {}).items():
                            all_hafix_bugs.update(failed_list)

    # Find intersection
    common_bugs = all_hafix_bugs.intersection(repairagent_bugs)

    return common_bugs


def generate_comparison_table(common_bugs: Set[str], hafix_results: Dict[str, Dict],
                             baseline_results: Set[str], baseline_name: str,
                             hafix_base_dir: Path, selector_type: str, n_lines: int) -> Dict:
    """
    Generate comparison table data with category breakdown.

    Args:
        common_bugs: Set of common bug IDs
        hafix_results: Dict mapping config names to their result data
        baseline_results: Set of successful bugs for the external baseline
        baseline_name: Name of the baseline (e.g., 'RepairAgent', 'HUNK4J')
        hafix_base_dir: Base HAFixAgent results directory
        selector_type: Line selection strategy
        n_lines: Number of lines selected

    Returns:
        Dict with table data organized by category
    """
    # Classify all common bugs by category
    bugs_by_category = {'SL': set(), 'SH': set(), 'SFMH': set(), 'MFMH': set()}

    print("\nClassifying bugs by category...")
    for bug_id in common_bugs:
        category = classify_by_category(bug_id, hafix_base_dir, selector_type, n_lines)
        if category in bugs_by_category:
            bugs_by_category[category].add(bug_id)

    print(f"Bug distribution: SL={len(bugs_by_category['SL'])}, SH={len(bugs_by_category['SH'])}, "
          f"SFMH={len(bugs_by_category['SFMH'])}, MFMH={len(bugs_by_category['MFMH'])}")

    # Build table data
    table_data = {}

    for category, bugs in bugs_by_category.items():
        if not bugs:
            continue

        row = {
            'category': category,
            'common_bugs': len(bugs),
            'baseline': 0,
            'fn_all': 0,
            'fn_pair': 0,
            'fl_diff': 0,
            baseline_name: 0
        }

        # Count successful repairs for each configuration
        for config_name, data in hafix_results.items():
            successful = data['successful_bugs'] & bugs
            row[config_name] = len(successful)

        # Count baseline successful repairs
        baseline_successful = baseline_results & bugs
        row[baseline_name] = len(baseline_successful)

        table_data[category] = row

    return table_data


def export_to_csv(table_data: Dict, output_path: Path, baseline_name: str):
    """
    Export comparison table to CSV format for LaTeX.

    Args:
        table_data: Table data organized by category
        output_path: Path to save CSV file
        baseline_name: Name of the baseline column
    """
    # Define column order
    columns = ['Category', 'Common Bugs', 'HAFixAgent-baseline', 'HAFixAgent-fn_all',
               'HAFixAgent-fn_pair', 'HAFixAgent-fl_diff', baseline_name]

    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(columns)

        # Write rows in category order
        for category in ['SL', 'SH', 'SFMH', 'MFMH']:
            if category in table_data:
                row_data = table_data[category]
                writer.writerow([
                    category,
                    row_data['common_bugs'],
                    row_data.get('baseline', 0),
                    row_data.get('fn_all', 0),
                    row_data.get('fn_pair', 0),
                    row_data.get('fl_diff', 0),
                    row_data.get(baseline_name, 0)
                ])

    print(f"CSV exported to: {output_path}")


def print_table(table_data: Dict, baseline_name: str):
    """
    Print formatted comparison table to console.

    Args:
        table_data: Table data organized by category
        baseline_name: Name of the baseline
    """
    print("\n" + "="*100)
    print(f"COMPARISON TABLE: HAFixAgent vs {baseline_name}")
    print("="*100)

    # Header
    print(f"{'Category':<12} {'Common':<8} {'Baseline':<10} {'fn_all':<10} {'fn_pair':<10} {'fl_diff':<10} {baseline_name:<15}")
    print(f"{'':12} {'Bugs':<8} {'(No-Hist)':<10} {'':10} {'':10} {'':10} {'':15}")
    print("-"*100)

    # Rows
    for category in ['SL', 'SH', 'SFMH', 'MFMH']:
        if category in table_data:
            row = table_data[category]
            print(f"{category:<12} {row['common_bugs']:<8} {row.get('baseline', 0):<10} "
                  f"{row.get('fn_all', 0):<10} {row.get('fn_pair', 0):<10} "
                  f"{row.get('fl_diff', 0):<10} {row.get(baseline_name, 0):<15}")

    print("="*100)


def compare_with_repairagent(hafix_base_dir: Path, repairagent_base_dir: Path,
                             selector_type: str, n_lines: int, output_dir: Path):
    """
    Compare HAFixAgent with RepairAgent on common bug set.

    Args:
        hafix_base_dir: Base HAFixAgent results directory
        repairagent_base_dir: Base RepairAgent repository directory
        selector_type: Line selection strategy
        n_lines: Number of lines selected
        output_dir: Output directory for results
    """
    print(f"\n{'='*80}")
    print("COMPARING HAFixAgent vs RepairAgent")
    print(f"{'='*80}")

    # Parse RepairAgent evaluated bugs
    print("\nParsing RepairAgent evaluated bugs...")
    batches_dir = repairagent_base_dir / "repair_agent" / "experimental_setups" / "batches"
    deprecated_bugs = get_repairagent_deprecated_bugs()
    repairagent_bugs = parse_repairagent_bugs(batches_dir, deprecated_bugs)

    print(f"RepairAgent evaluated bugs (after deprecation filter): {len(repairagent_bugs)}")

    # Find common bugs
    print("\nFinding common bugs with HAFixAgent...")
    common_bugs = find_common_bugs_with_repairagent(hafix_base_dir, repairagent_bugs,
                                                     selector_type, n_lines)
    print(f"Common bugs: {len(common_bugs)}")

    if len(common_bugs) == 0:
        print("No common bugs found! Aborting comparison.")
        return

    # Load HAFixAgent results for all 4 configurations
    print("\nLoading HAFixAgent results...")
    hafix_configs = {
        'baseline': BASELINE_FLAG,
        'fn_all': FN_ALL_FLAG,
        'fn_pair': FN_PAIR_FLAG,
        'fl_diff': FL_DIFF_FLAG
    }

    hafix_results = {}
    for config_name, flag in hafix_configs.items():
        # Merge results across all categories
        merged_successful = set()
        for category in ['single_line', 'single_hunk', 'single_file_multi_hunk', 'multi_file_multi_hunk']:
            data = load_progress_data(category, flag, hafix_base_dir, selector_type, n_lines)
            merged_successful.update(data['successful_bugs'])

        # Filter to common bugs
        hafix_results[config_name] = {
            'successful_bugs': merged_successful & common_bugs
        }
        print(f"  {config_name}: {len(hafix_results[config_name]['successful_bugs'])} successful on common bugs")

    # Load RepairAgent results
    print("\nLoading RepairAgent results...")
    repairagent_successful = load_repairagent_results(repairagent_base_dir)
    repairagent_successful_common = repairagent_successful & common_bugs
    print(f"  RepairAgent: {len(repairagent_successful_common)} successful on common bugs")

    # Generate comparison table
    print("\nGenerating comparison table...")
    table_data = generate_comparison_table(common_bugs, hafix_results, repairagent_successful_common,
                                           'RepairAgent', hafix_base_dir, selector_type, n_lines)

    # Print table
    print_table(table_data, 'RepairAgent')

    # Export to CSV
    csv_filename = f"rq1_repairagent_comparison_{len(common_bugs)}_common_bugs.csv"
    csv_path = output_dir / csv_filename
    export_to_csv(table_data, csv_path, 'RepairAgent')


def compare_with_hunk4j(hafix_base_dir: Path, selector_type: str, n_lines: int, output_dir: Path):
    """
    Compare HAFixAgent with HUNK4J on common bug set.

    Args:
        hafix_base_dir: Base HAFixAgent results directory
        selector_type: Line selection strategy
        n_lines: Number of lines selected
        output_dir: Output directory for results
    """
    print(f"\n{'='*80}")
    print("COMPARING HAFixAgent vs HUNK4J (BIRCH-feedback)")
    print(f"{'='*80}")

    # Find common bugs (reuse existing utility)
    print("\nFinding common bugs with HUNK4J...")
    common_bugs = find_common_bugs_with_hunk4j(hafix_base_dir, selector_type, n_lines)
    print(f"Common bugs: {len(common_bugs)}")

    if len(common_bugs) == 0:
        print("No common bugs found! Aborting comparison.")
        return

    # Load HAFixAgent results for all 4 configurations
    print("\nLoading HAFixAgent results...")
    hafix_configs = {
        'baseline': BASELINE_FLAG,
        'fn_all': FN_ALL_FLAG,
        'fn_pair': FN_PAIR_FLAG,
        'fl_diff': FL_DIFF_FLAG
    }

    hafix_results = {}
    for config_name, flag in hafix_configs.items():
        # Merge results across multi-hunk categories only (HUNK4J focuses on multi-hunk)
        merged_successful = set()
        for category in ['single_file_multi_hunk', 'multi_file_multi_hunk']:
            data = load_progress_data(category, flag, hafix_base_dir, selector_type, n_lines)
            merged_successful.update(data['successful_bugs'])

        # Filter to common bugs
        hafix_results[config_name] = {
            'successful_bugs': merged_successful & common_bugs
        }
        print(f"  {config_name}: {len(hafix_results[config_name]['successful_bugs'])} successful on common bugs")

    # Load HUNK4J results
    print("\nLoading HUNK4J results...")
    hunk4j_data = load_hunk4j_data("multi_hunk", selector_type, n_lines, hafix_base_dir)
    hunk4j_successful_common = hunk4j_data['successful_bugs'] & common_bugs
    print(f"  HUNK4J: {len(hunk4j_successful_common)} successful on common bugs")

    # Generate comparison table (only SFMH and MFMH for HUNK4J)
    print("\nGenerating comparison table...")

    # Classify bugs by category
    bugs_by_category = {'SFMH': set(), 'MFMH': set()}

    selector_dir = f"{selector_type}_{n_lines}line"
    for category_dir, category_code in [('single_file_multi_hunk', 'SFMH'), ('multi_file_multi_hunk', 'MFMH')]:
        category_path = hafix_base_dir / selector_dir / category_dir

        # Find bugs in this category
        for pattern in ['progress_*_both.json', 'progress_*_blameable.json', 'progress_*_blameless.json']:
            files = list(category_path.glob(pattern))
            if files:
                with open(files[0], 'r') as f:
                    data = json.load(f)
                    category_bugs = set(data.get('successful_bugs', []))
                    for status, failed_list in data.get('failed_bugs', {}).items():
                        category_bugs.update(failed_list)

                    # Filter to common bugs
                    bugs_by_category[category_code] = category_bugs & common_bugs
                break

    # Build table data
    table_data = {}
    for category, bugs in bugs_by_category.items():
        if not bugs:
            continue

        row = {
            'category': category,
            'common_bugs': len(bugs),
            'baseline': 0,
            'fn_all': 0,
            'fn_pair': 0,
            'fl_diff': 0,
            BIRCH_BASELINE_LABEL: 0
        }

        # Count successful repairs for each configuration
        for config_name, data in hafix_results.items():
            successful = data['successful_bugs'] & bugs
            row[config_name] = len(successful)

        # Count HUNK4J successful repairs
        hunk4j_successful = hunk4j_successful_common & bugs
        row[BIRCH_BASELINE_LABEL] = len(hunk4j_successful)

        table_data[category] = row

    # Print table
    print_table(table_data, BIRCH_BASELINE_LABEL)

    # Export to CSV
    csv_filename = f"rq1_hunk4j_comparison_{len(common_bugs)}_common_bugs.csv"
    csv_path = output_dir / csv_filename
    export_to_csv(table_data, csv_path, BIRCH_BASELINE_LABEL)


def main():
    parser = argparse.ArgumentParser(description='Compare HAFixAgent with external SOTA baselines (RQ1.1)')
    parser.add_argument('--baseline', '-b', required=True, choices=['repairagent', 'hunk4j'],
                       help='External baseline to compare against')
    parser.add_argument('--selector-type', '-s', default='llm_judge',
                       help='Line selection strategy: first, random, llm_judge (default: llm_judge)')
    parser.add_argument('--n-lines', '-n', type=int, default=1,
                       help='Number of lines selected for blame (default: 1)')
    parser.add_argument('--output', '-o', default='results/comparison',
                       help='Output directory for CSV results')

    args = parser.parse_args()

    # Setup paths
    base_dir = Path(__file__).parent.parent / "results" / "defects4j"
    selector_subdir = f"{args.selector_type}_{args.n_lines}line"
    output_dir = Path(args.output) / selector_subdir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Run comparison based on baseline choice
    if args.baseline == 'repairagent':
        repairagent_base_dir = Path(__file__).parent.parent / "vendor" / "RepairAgent"
        compare_with_repairagent(base_dir, repairagent_base_dir, args.selector_type, args.n_lines, output_dir)
    elif args.baseline == 'hunk4j':
        compare_with_hunk4j(base_dir, args.selector_type, args.n_lines, output_dir)


if __name__ == "__main__":
    main()