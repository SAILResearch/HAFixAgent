#!/usr/bin/env python3
"""
Shared utilities for HAFixAgent analysis scripts.

This module provides reusable functions for:
- Loading progress and result data
- Loading HUNK4J baseline data
- Common bug set filtering
- Publication-quality figure settings
"""

import json
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Set, Dict, List, Optional, Tuple
import matplotlib
from hafix_agent.blame.core import history_name_to_category

# Set publication-quality defaults
matplotlib.use('Agg')
plt.ioff()

# Publication-quality matplotlib settings
def set_publication_style():
    """Set consistent publication-quality matplotlib style."""
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams.update({
        'font.size': 12,
        'axes.titlesize': 14,
        'axes.labelsize': 12,
        'xtick.labelsize': 11,
        'ytick.labelsize': 11,
        'legend.fontsize': 11,
        'figure.titlesize': 16,
        'font.family': 'serif',
        'axes.grid': True,
        'grid.alpha': 0.3,
        'figure.dpi': 300
    })

CANONICAL_NAME_OVERRIDES = {
    'adaptive': 'HAFixAgent-Adaptive',
    'baseline': 'HAFixAgent-non-history',
    'fn_all': 'HAFixAgent-fn_all',
    'fn_pair': 'HAFixAgent-fn_pair',
    'fl_diff': 'HAFixAgent-fl_diff',
    'cfn_modified': 'HAFixAgent-cfn_modified',
    'cfn_all': 'HAFixAgent-cfn_all',
    'fn_modified': 'HAFixAgent-fn_modified',
    'fln_all': 'HAFixAgent-fln_all',
}

# Canonical label helpers keep figure/text terminology aligned with paper naming
FLAG_TO_CANONICAL_LABEL = {
    str(category.value): CANONICAL_NAME_OVERRIDES.get(short_name, f'HAFixAgent-{short_name}')
    for short_name, category in history_name_to_category.items()
}

SHORT_NAME_TO_CANONICAL_LABEL = {
    short_name: FLAG_TO_CANONICAL_LABEL[str(category.value)]
    for short_name, category in history_name_to_category.items()
}

LEGACY_LABEL_ALIASES = {
    'baseline_non-history': FLAG_TO_CANONICAL_LABEL['1'],
    'baseline_non_history': FLAG_TO_CANONICAL_LABEL['1'],
}

BIRCH_FEEDBACK_LABEL = 'BIRCH-feedback'


def _normalize_identifier(identifier: str) -> str:
    """Normalize identifiers for dictionary lookups."""
    return identifier.replace('-', '_').lower()


def resolve_config_label(identifier: Optional[str], default: Optional[str] = None) -> str:
    """Return canonical label for any config identifier (flag, short name, or alias)."""
    if identifier is None:
        return default or ''

    value = str(identifier).strip()
    if not value:
        return default or ''

    if value in FLAG_TO_CANONICAL_LABEL:
        return FLAG_TO_CANONICAL_LABEL[value]

    normalized = _normalize_identifier(value)
    if normalized in FLAG_TO_CANONICAL_LABEL:
        return FLAG_TO_CANONICAL_LABEL[normalized]

    if value in SHORT_NAME_TO_CANONICAL_LABEL:
        return SHORT_NAME_TO_CANONICAL_LABEL[value]
    if normalized in SHORT_NAME_TO_CANONICAL_LABEL:
        return SHORT_NAME_TO_CANONICAL_LABEL[normalized]

    if value in LEGACY_LABEL_ALIASES:
        return LEGACY_LABEL_ALIASES[value]
    if normalized in LEGACY_LABEL_ALIASES:
        return LEGACY_LABEL_ALIASES[normalized]

    if normalized == 'hunk4j':
        return 'HUNK4J'

    if normalized == 'baseline_birch_fb':
        return BIRCH_FEEDBACK_LABEL

    if value.startswith('HAFixAgent-'):
        return value

    return default or value


def get_history_flag(short_name: str) -> str:
    """Return the numeric flag for a given history short name."""
    if short_name is None:
        raise ValueError('History configuration name cannot be None')

    key = short_name.strip()
    if not key:
        raise ValueError('History configuration name cannot be empty')

    normalized = key.lower()
    if normalized.startswith('hafixagent-'):
        normalized = normalized[len('hafixagent-'):]

    if normalized in history_name_to_category:
        return str(history_name_to_category[normalized].value)

    raise KeyError(f"Unknown history configuration name: {short_name}")


def canonicalize_config_sequence(labels: List[str]) -> List[str]:
    """Map a list of config identifiers to canonical display labels."""
    return [resolve_config_label(label) for label in labels]



def load_progress_data(bug_category: str, history_flag: str, base_dir: Path,
                       selector_type: str = "llm_judge", n_lines: int = 1) -> Dict:
    """
    Load progress data for a specific history heuristic and bug category.
    Handles both _both.json files and combines _blameable.json + _blameless.json files.

    Args:
        bug_category: Bug category (e.g., 'single_file_multi_hunk', 'multi_file_multi_hunk')
        history_flag: History flag (e.g., '1' for baseline, '5' for fn_all, '0' for adaptive)
        base_dir: Base results directory (e.g., Path('results/defects4j'))
        selector_type: Line selection strategy ('first', 'random', 'llm_judge')
        n_lines: Number of lines selected for blame

    Returns:
        Dict with keys: 'successful_bugs' (set), 'total_bugs' (int), 'success_rate' (float)
    """
    selector_dir = f"{selector_type}_{n_lines}line"
    category_dir = base_dir / selector_dir / bug_category

    # Try to find _both.json first
    both_file = category_dir / f"progress_{history_flag}_both.json"
    if both_file.exists():
        with open(both_file, 'r') as f:
            data = json.load(f)
        return {
            'successful_bugs': set(data.get('successful_bugs', [])),
            'total_bugs': data.get('total_bugs', 0),
            'success_rate': data.get('repair_results', {}).get('success_rate', 0),
            'total_cost': data.get('total_model_cost', 0.0),
            'failed_bugs': set(sum([bugs for bugs in data.get('failed_bugs', {}).values()], []))
        }

    # Otherwise combine _blameable.json and _blameless.json
    blameable_file = category_dir / f"progress_{history_flag}_blameable.json"
    blameless_file = category_dir / f"progress_{history_flag}_blameless.json"

    combined_successful = set()
    combined_failed = set()
    total_bugs = 0
    total_cost = 0.0

    files_found = []
    for file_path, label in [(blameable_file, 'blameable'), (blameless_file, 'blameless')]:
        if file_path.exists():
            files_found.append(label)
            with open(file_path, 'r') as f:
                data = json.load(f)
                combined_successful.update(data.get('successful_bugs', []))
                for bugs in data.get('failed_bugs', {}).values():
                    combined_failed.update(bugs)
                total_bugs += data.get('total_bugs', 0)
                total_cost += data.get('total_model_cost', 0.0)

    if not files_found:
        raise FileNotFoundError(f"No progress files found for {bug_category} with history_flag={history_flag}")

    success_rate = round(len(combined_successful) / total_bugs * 100, 2) if total_bugs > 0 else 0

    return {
        'successful_bugs': combined_successful,
        'failed_bugs': combined_failed,
        'total_bugs': total_bugs,
        'success_rate': success_rate,
        'total_cost': total_cost,
        'combined_from': files_found
    }


def load_hunk4j_data(
    bug_category: str = "multi_hunk",
    selector_type: str = "llm_judge",
    n_lines: int = 1,
    hafix_results_dir: Optional[Path] = None,
) -> Dict:
    """
    Load HUNK4J results data for comparison.

    Args:
        bug_category: Bug category to filter ('single_file_multi_hunk', 'multi_file_multi_hunk', 'multi_hunk')
        selector_type: Line selection strategy used by HAFixAgent results directory
        n_lines: Number of lines selected for blame
        hafix_results_dir: Optional override for HAFixAgent results base directory

    Returns:
        Dict with keys: 'successful_bugs' (set), 'total_bugs' (int), 'success_rate' (float)
    """
    base_dir = Path(__file__).parent.parent / "vendor" / "birch-543D"

    # Load successful bugs (using specific model results)
    passed_bugs_file = base_dir / "birch-augmented-prompting" / "results" / "augmented_prompting-feedback_loop" / "passed_bugs.json"
    with open(passed_bugs_file, 'r') as f:
        passed_data = json.load(f)
        model_key = "mode_4_model_o4-mini-2025-04-16"
        successful_bugs = set(passed_data[model_key]["passed"])

    # Load all evaluated bugs (dataset) to filter by category
    dataset_file = base_dir / "hunk4j" / "dataset" / "method_multihunk.json"
    with open(dataset_file, 'r') as f:
        dataset = json.load(f)
        all_bugs = set(dataset.keys())

    # Normalize bug IDs for comparison (Chart-2 -> Chart_2)
    def normalize_bug_id(bug_id: str) -> str:
        return bug_id.replace('-', '_')

    successful_normalized = {normalize_bug_id(bug) for bug in successful_bugs}
    all_normalized = {normalize_bug_id(bug) for bug in all_bugs}

    # Filter by category if not 'both' or 'multi_hunk'
    if bug_category in ['both', 'multi_hunk']:
        category_bugs = all_normalized
        category_successful = successful_normalized & category_bugs
    else:
        # For specific categories, load HAFixAgent bug lists to determine category mapping
        hafix_base_dir = hafix_results_dir or (Path(__file__).parent.parent / "results" / "defects4j")
        selector_dir = f"{selector_type}_{n_lines}line"
        category_dir = hafix_base_dir / selector_dir / bug_category
        category_bugs = set()

        # Try to find any progress file to get the bug list for this category
        for pattern in ['progress_*_both.json', 'progress_*_blameable.json', 'progress_*_blameless.json']:
            files = list(category_dir.glob(pattern))
            if files:
                with open(files[0], 'r') as f:
                    data = json.load(f)
                    category_bugs.update(data.get('successful_bugs', []))
                    for status, failed_list in data.get('failed_bugs', {}).items():
                        category_bugs.update(failed_list)
                break

        if not category_bugs:
            category_bugs = all_normalized

        category_successful = successful_normalized & category_bugs

    total_bugs = len(category_bugs)
    success_rate = round(len(category_successful) / total_bugs * 100, 2) if total_bugs > 0 else 0

    return {
        'successful_bugs': category_successful,
        'total_bugs': total_bugs,
        'success_rate': success_rate
    }


def find_common_bugs_with_hunk4j(hafix_base_dir: Path, selector_type: str = "llm_judge",
                                  n_lines: int = 1) -> Set[str]:
    """
    Find bugs that are common between HAFixAgent datasets and HUNK4J dataset.
    This ensures fair comparison on the same bug set (371 common bugs).

    Args:
        hafix_base_dir: Base HAFixAgent results directory (e.g., Path('results/defects4j'))
        selector_type: Line selection strategy
        n_lines: Number of lines selected

    Returns:
        Set of common bug IDs (normalized format)
    """
    selector_dir = f"{selector_type}_{n_lines}line"

    # Collect all HAFixAgent bugs
    all_hafix_bugs = set()
    for category in ["single_file_multi_hunk", "multi_file_multi_hunk"]:
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

    # Load HUNK4J dataset
    base_dir = Path(__file__).parent.parent / "vendor" / "birch-543D"
    dataset_file = base_dir / "hunk4j" / "dataset" / "method_multihunk.json"
    with open(dataset_file, 'r') as f:
        dataset = json.load(f)
        all_hunk4j_bugs = set(dataset.keys())

    # Normalize bug IDs for comparison
    def normalize_bug_id(bug_id: str) -> str:
        return bug_id.replace('-', '_')

    hafix_normalized = {normalize_bug_id(bug) for bug in all_hafix_bugs}
    hunk4j_normalized = {normalize_bug_id(bug) for bug in all_hunk4j_bugs}

    # Find intersection
    common_bugs = hafix_normalized.intersection(hunk4j_normalized)

    return common_bugs


def filter_to_common_bugs(data: Dict, common_bugs: Set[str]) -> Dict:
    """
    Filter dataset to only include bugs that are in the common bugs set.

    Args:
        data: Original data dict with 'successful_bugs' set
        common_bugs: Set of common bug IDs

    Returns:
        Filtered data dict
    """
    filtered_successful = data['successful_bugs'] & common_bugs
    filtered_failed = data.get('failed_bugs', set()) & common_bugs

    return {
        'successful_bugs': filtered_successful,
        'failed_bugs': filtered_failed,
        'total_bugs': len(common_bugs),
        'success_rate': round(len(filtered_successful) / len(common_bugs) * 100, 2) if len(common_bugs) > 0 else 0
    }


def get_history_name(history_flag: str) -> str:
    """
    Convert history flag to the canonical display label used in figures/tables.
    """
    value = str(history_flag)

    # Direct resolution handles already-canonical names or aliases
    direct_label = resolve_config_label(value, default='')
    if direct_label:
        return direct_label

    # Fall back to mapping enum values to short names, then canonicalize
    flag_to_name = {str(category.value): name for name, category in history_name_to_category.items()}
    short_name = flag_to_name.get(value, value)

    return resolve_config_label(short_name, default=short_name)


def load_individual_results(results_dir: Path, history_flag: str) -> List[Dict]:
    """
    Load individual bug result files (*_result.json).

    Args:
        results_dir: Directory containing bug subdirectories
        history_flag: History flag to filter results

    Returns:
        List of result dictionaries
    """
    pattern = f"*_{history_flag}_result.json"
    result_files = list(results_dir.rglob(pattern))

    results = []
    for result_file in result_files:
        try:
            with open(result_file) as f:
                data = json.load(f)
                results.append(data)
        except Exception as e:
            print(f"Warning: Error loading {result_file}: {e}")
            continue

    return results


def parse_trajectory_for_heuristics(traj_file: Path) -> Dict:
    """
    Parse trajectory file to extract hafix-context heuristic usage.

    Args:
        traj_file: Path to trajectory JSON file

    Returns:
        Dict with heuristic usage info: {
            'bug_id': str,
            'success': bool,
            'heuristics_used': [{'heuristic': str, 'step': int}, ...],
            'total_steps': int
        }
    """
    with open(traj_file) as f:
        traj = json.load(f)

    bug_id = traj_file.stem.replace('.traj', '')
    success = traj.get('info', {}).get('exit_status') == 'Submitted'
    messages = traj.get('messages', [])

    heuristics_used = []
    total_steps = 0

    for i, msg in enumerate(messages):
        if msg.get('role') == 'assistant':
            total_steps += 1
            content = msg.get('content', '')

            # Check if hafix-context command was used
            if 'hafix-context' in content:
                # Extract heuristic type (fn_all, fl_diff, fn_pair)
                for heuristic in ['fn_all', 'fl_diff', 'fn_pair']:
                    if f'hafix-context {heuristic}' in content:
                        heuristics_used.append({
                            'heuristic': heuristic,
                            'step': total_steps
                        })
                        break

    return {
        'bug_id': bug_id,
        'success': success,
        'heuristics_used': heuristics_used,
        'total_steps': total_steps
    }


def merge_categories(data_list: List[Dict], categories: List[str]) -> Dict:
    """
    Merge data from multiple bug categories.

    Args:
        data_list: List of data dicts from different categories
        categories: List of category names

    Returns:
        Merged data dict
    """
    merged = {
        'successful_bugs': set(),
        'failed_bugs': set(),
        'total_bugs': 0,
        'total_cost': 0.0
    }

    for data in data_list:
        merged['successful_bugs'].update(data.get('successful_bugs', set()))
        merged['failed_bugs'].update(data.get('failed_bugs', set()))
        merged['total_bugs'] += data.get('total_bugs', 0)
        merged['total_cost'] += data.get('total_cost', 0.0)

    merged['success_rate'] = round(
        len(merged['successful_bugs']) / merged['total_bugs'] * 100, 2
    ) if merged['total_bugs'] > 0 else 0

    return merged
