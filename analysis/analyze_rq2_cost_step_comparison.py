"""
Performance Analysis for HAFixAgent Evaluation Results

Supports three analysis modes:
1. Single configuration analysis (default)
2. Multi-configuration comparison (RQ3a: Efficiency - tokens, steps, cost)
3. Adaptive behavior analysis (RQ3b: Heuristic usage patterns)

Usage Examples:

    # Single configuration analysis
    python analysis/analyze_rq3_cost_step_comparison.py --results-dir results/defects4j/llm_judge_1line/multi_file_multi_hunk --history-flag 5
    python analysis/analyze_rq3_cost_step_comparison.py --mode single --bug-category single_file_multi_hunk --history-flag 1

    # Multi-configuration comparison (RQ3a: Efficiency)
    # Compare baseline and 3 heuristics (multi_hunk category)
    python analysis/analyze_rq3_cost_step_comparison.py --mode multi-config \
        --rq1-dir results/defects4j \
        --rq2-dir results/defects4j_adaptive \
        --bug-category multi_hunk \
        --selector-type llm_judge \
        --n-lines 1 \
        --output results/rq3_analysis

    # Multi-configuration with adaptive included (default)
    python analysis/analyze_rq3_cost_step_comparison.py --mode multi-config \
        --rq1-dir results/defects4j \
        --rq2-dir results/defects4j_adaptive \
        --bug-category multi_hunk \
        --include-adaptive \
        --output results/rq3_analysis

    # Multi-configuration WITHOUT adaptive
    python analysis/analyze_rq3_cost_step_comparison.py --mode multi-config \
        --rq1-dir results/defects4j \
        --rq2-dir results/defects4j_adaptive \
        --bug-category multi_hunk \
        --no-adaptive \
        --output results/rq3_analysis

    # Multi-configuration with ALL categories (SFMH+MFMH+SH+SL)
    python analysis/analyze_rq3_cost_step_comparison.py --mode multi-config \
        --rq1-dir results/defects4j \
        --rq2-dir results/defects4j_adaptive \
        --bug-category all \
        --output results/rq3_analysis

    # Adaptive behavior analysis (RQ3b: Heuristic Usage)
    python analysis/analyze_rq3_cost_step_comparison.py --mode adaptive \
        --trajectories-dir results/defects4j_adaptive \
        --bug-category multi_hunk \
        --output results/rq3_analysis

Category Options:
    - 'single_file_multi_hunk': SFMH only
    - 'multi_file_multi_hunk': MFMH only
    - 'single_hunk': SH only
    - 'single_line': SL only
    - 'multi_hunk': SFMH + MFMH (default for RQ1/RQ2)
    - 'all': SFMH + MFMH + SH + SL (all bugs)

Output Files:
    - Single mode: steps_{flag}.pdf, cost_{flag}.pdf, cost_vs_steps_{flag}.pdf
    - Multi-config mode: rq3a_step_distribution_{category}.pdf (violin), rq3a_cost_distribution_{category}.pdf (boxplot), rq3a_metrics_{category}.csv
    - Adaptive mode: rq3b_heuristic_usage.pdf (stacked bar: success/failed), rq3b_heuristic_timing.pdf (boxplot)
"""

import json
import matplotlib.pyplot as plt
from matplotlib import ticker
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path
import warnings
from typing import Optional, List, Dict, Set, Tuple
import typer
from rich.console import Console
from scipy import stats
from utils import get_history_flag, resolve_config_label

warnings.filterwarnings('ignore')

# Set style for publication quality
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 11,
    'figure.titlesize': 16,
    'font.family': 'serif',
    'axes.grid': True,
    'grid.alpha': 0.3
})
sns.set_palette(["#2E8B57", "#DC143C"])  # Professional green/red

ADAPTIVE_HIGHLIGHT_COLOR = '#FFF4CE'

BASELINE_FLAG = get_history_flag('baseline')
FN_ALL_FLAG = get_history_flag('fn_all')
FN_PAIR_FLAG = get_history_flag('fn_pair')
FL_DIFF_FLAG = get_history_flag('fl_diff')
ADAPTIVE_FLAG = get_history_flag('adaptive')


def load_metrics(results_dir: Path, history_flag: Optional[str] = None) -> pd.DataFrame:
    """Load all metrics into a clean DataFrame."""
    if history_flag:
        pattern = f"*_{history_flag}_result.json"
    else:
        pattern = "*_result.json"
    result_files = list(results_dir.rglob(pattern))

    if not result_files:
        raise ValueError(f"No result files found in {results_dir}")

    data = []
    for result_file in result_files:
        try:
            with open(result_file) as f:
                json_data = json.load(f)
                repair_result = json_data.get("repair_result", {})

                row = {
                    'bug_id': f"{json_data.get('project_name', 'Unknown')}-{json_data.get('bug_id', 'Unknown')}",
                    'success': repair_result.get("success", False),
                    'exit_status': repair_result.get("exit_status", "Unknown"),
                    'steps': repair_result.get("agent_steps"),
                    'cost': repair_result.get("model_cost"),
                    'duration_minutes': repair_result.get("timing", {}).get("duration_seconds", 0) / 60,
                    'model_calls': repair_result.get("model_calls"),
                }
                data.append(row)
        except Exception as e:
            print(f"Warning: Error processing {result_file}: {e}")
            continue

    if not data:
        raise ValueError("No valid data found")

    df = pd.DataFrame(data)

    # Clean data
    df = df.dropna(subset=['steps', 'cost', 'duration_minutes'])
    df['status'] = df['success'].map({True: 'Successful', False: 'Failed'})

    return df


def create_individual_figures(df: pd.DataFrame, save_path: Path, history_flag: Optional[str] = None):
    """Create individual figures with clean naming (for single-config mode only)."""
    successful = df[df['success']]
    failed = df[~df['success']]

    # Step Distribution
    fig, ax = plt.subplots(1, 1, figsize=(6, 5))
    parts = ax.violinplot([successful['steps'], failed['steps']],
                         positions=[1, 2], showmeans=True, showmedians=True)
    ax.set_xticks([1, 2])
    ax.set_xticklabels(['Successful', 'Failed'])
    ax.set_ylabel('Agent Steps')
    ax.set_title('Step Distribution')
    ax.grid(True, alpha=0.3)
    ax.text(0.5, 0.85, f'Success: {successful["steps"].mean():.0f} avg\n'
                       f'Failed: {failed["steps"].mean():.0f} avg',
            transform=ax.transAxes, verticalalignment='top', horizontalalignment='center', fontsize=9,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    plt.tight_layout()
    plt.savefig(save_path / f'steps_{history_flag}.pdf', dpi=300, bbox_inches='tight')
    plt.close()

    # Cost Distribution
    fig, ax = plt.subplots(1, 1, figsize=(6, 5))
    cost_data = [successful['cost'], failed['cost']]
    bp = ax.boxplot(cost_data, labels=['Successful', 'Failed'], patch_artist=True)
    bp['boxes'][0].set_facecolor('#2E8B57')
    bp['boxes'][1].set_facecolor('#DC143C')
    ax.set_ylabel('Cost (USD)')
    ax.set_title('Cost Distribution')
    ax.grid(True, alpha=0.3)
    ax.text(0.02, 0.98, f'Success: ${successful["cost"].mean():.2f} avg\n'
                        f'Failed: ${failed["cost"].mean():.2f} avg',
            transform=ax.transAxes, verticalalignment='top', fontsize=9,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    plt.tight_layout()
    plt.savefig(save_path / f'cost_{history_flag}.pdf', dpi=300, bbox_inches='tight')
    plt.close()

    # Cost vs Steps Scatter
    fig, ax = plt.subplots(1, 1, figsize=(7, 5))
    ax.scatter(successful['steps'], successful['cost'],
               alpha=0.6, c='#2E8B57', label='Successful', s=40)
    ax.scatter(failed['steps'], failed['cost'],
               alpha=0.6, c='#DC143C', label='Failed', s=40)
    ax.set_xlabel('Agent Steps')
    ax.set_ylabel('Cost (USD)')
    ax.set_title('Cost vs Steps')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path / f'cost_vs_steps_{history_flag}.pdf', dpi=300, bbox_inches='tight')
    plt.close()


def print_summary_stats(df: pd.DataFrame, category_name: str):
    """Print concise summary statistics."""
    total_bugs = len(df)
    successful = df[df['success']]
    failed = df[~df['success']]

    print(f"\n{'='*60}")
    print(f"📊 {category_name.upper()} ANALYSIS SUMMARY")
    print(f"{'='*60}")
    print(f"Total bugs: {total_bugs}")
    print(f"Success rate: {len(successful)}/{total_bugs} ({len(successful)/total_bugs*100:.1f}%)")

    if len(successful) > 0:
        print(f"\n✅ SUCCESSFUL CASES:")
        print(f"   Steps: {successful['steps'].mean():.1f} avg (range: {successful['steps'].min()}-{successful['steps'].max()})")
        print(f"   Cost: ${successful['cost'].mean():.3f} avg (range: ${successful['cost'].min():.3f}-${successful['cost'].max():.3f})")
        print(f"   Time: {successful['duration_minutes'].mean():.1f}m avg (range: {successful['duration_minutes'].min():.1f}-{successful['duration_minutes'].max():.1f}m)")

    if len(failed) > 0:
        print(f"\n❌ FAILED CASES:")
        print(f"   Steps: {failed['steps'].mean():.1f} avg (range: {failed['steps'].min()}-{failed['steps'].max()})")
        print(f"   Cost: ${failed['cost'].mean():.3f} avg (range: ${failed['cost'].min():.3f}-${failed['cost'].max():.3f})")
        print(f"   Time: {failed['duration_minutes'].mean():.1f}m avg (range: {failed['duration_minutes'].min():.1f}-{failed['duration_minutes'].max():.1f}m)")

        if len(successful) > 0:
            print(f"\n📈 FAILED vs SUCCESSFUL MULTIPLIERS:")
            print(f"   Cost: {failed['cost'].mean()/successful['cost'].mean():.1f}x more expensive")
            print(f"   Time: {failed['duration_minutes'].mean()/successful['duration_minutes'].mean():.1f}x longer")


def load_individual_results(results_dir: Path, history_flag: str) -> List[Dict]:
    """Load individual result JSON files from a results directory."""
    pattern = f"*_{history_flag}_result.json"
    result_files = list(results_dir.rglob(pattern))

    results = []
    for result_file in result_files:
        try:
            with open(result_file) as f:
                results.append(json.load(f))
        except Exception as e:
            print(f"Warning: Error processing {result_file}: {e}")

    return results


def load_multi_config_metrics(configs: List[Dict], rq1_dir: Path, rq2_dir: Path,
                              bug_category: str, selector_type: str, n_lines: int) -> pd.DataFrame:
    """
    Load metrics from multiple configurations for RQ3a comparison.

    Args:
        configs: List of {'name': str, 'flag': str, 'is_adaptive': bool}
        rq1_dir: RQ1 results directory
        rq2_dir: RQ2 results directory (for adaptive)
        bug_category: 'single_file_multi_hunk', 'multi_file_multi_hunk', 'multi_hunk', or 'all'
        selector_type: Selector type (e.g., 'llm_judge')
        n_lines: Number of lines

    Returns:
        DataFrame with columns: bug_id, config, success, steps, cost, input_tokens, output_tokens, total_tokens
    """
    data = []

    for config in configs:
        name = config['name']
        flag = config['flag']
        is_adaptive = config.get('is_adaptive', False)

        # Both adaptive and static heuristics use selector subdirectory
        base_dir = (rq2_dir if is_adaptive else rq1_dir) / f"{selector_type}_{n_lines}line"

        print(f"\nLoading {name} (flag={flag}, adaptive={is_adaptive})...")

        # Determine result directories based on category
        if bug_category == 'all':
            categories = ['single_file_multi_hunk', 'multi_file_multi_hunk', 'single_hunk', 'single_line']
        elif bug_category == 'multi_hunk':
            categories = ['single_file_multi_hunk', 'multi_file_multi_hunk']
        else:
            categories = [bug_category]

        for cat in categories:
            results_dir = base_dir / cat
            if not results_dir.exists():
                print(f"Warning: Directory not found: {results_dir}")
                continue

            result_files = load_individual_results(results_dir, flag)

            for result in result_files:
                repair_result = result.get('repair_result', {})
                token_usage = repair_result.get('token_usage', {})

                # RQ1 used agent_steps (old counting), RQ2 uses model_calls (new counting)
                # RQ1 steps are 1 higher, so subtract 1 for consistency
                steps = repair_result.get('agent_steps')
                if not is_adaptive and steps is not None and cat != 'single_hunk' and cat != 'single_line':
                    steps = steps - 1  # Normalize RQ1 steps to match RQ2 counting

                row = {
                    'bug_id': f"{result.get('project_name', 'Unknown')}-{result.get('bug_id', 'Unknown')}",
                    'config': name,
                    'category': cat,
                    'success': repair_result.get('success', False),
                    'steps': steps,
                    'cost': repair_result.get('model_cost'),
                    'input_tokens': token_usage.get('input_tokens', 0),
                    'output_tokens': token_usage.get('output_tokens', 0),
                    'total_tokens': token_usage.get('total_tokens', 0)
                }
                data.append(row)

    df = pd.DataFrame(data)
    df = df.dropna(subset=['steps', 'cost'])
    return df


def create_multi_config_violin_separate(df: pd.DataFrame, output_dir: Path, category: str, include_adaptive: bool = True):
    """
    Create separate plots for RQ3a efficiency comparison.
    Each configuration has 2 groups (successful and failed).
    Generates 2 separate PNG files: steps (violin), cost (boxplot).

    Args:
        df: DataFrame with metrics
        output_dir: Output directory for plots
        category: Bug category name for filename suffix
        include_adaptive: If False, exclude adaptive configuration from plots
    """
    # Prepare data with success/failure labels
    df['status'] = df['success'].map({True: 'Success', False: 'Failed'})
    df['config_status'] = df['config'] + '-' + df['status']

    # Define professional color palette for research papers
    # Green shades for successful, red/orange shades for failed
    all_config_order = ['baseline', 'fn_all', 'fn_pair', 'fl_diff', 'adaptive']

    # Filter config_order based on include_adaptive and what's in the data
    available_configs = df['config'].unique()
    if not include_adaptive:
        config_order = [c for c in all_config_order if c != 'adaptive' and c in available_configs]
    else:
        config_order = [c for c in all_config_order if c in available_configs]

    colors = {
        'baseline-Success': '#2E7D32',    # Dark green
        'baseline-Failed': '#C62828',     # Dark red
        'fn_all-Success': '#388E3C',      # Green
        'fn_all-Failed': '#D32F2F',       # Red
        'fl_diff-Success': '#43A047',     # Light green
        'fl_diff-Failed': '#E53935',      # Light red
        'fn_pair-Success': '#4CAF50',     # Lighter green
        'fn_pair-Failed': '#F44336',      # Lighter red
        'adaptive-Success': '#66BB6A',    # Lightest green
        'adaptive-Failed': '#EF5350',     # Lightest red
    }

    # Create order for x-axis (alternating success/failed for each config)
    plot_order = []
    for config in config_order:
        plot_order.extend([f'{config}-Success', f'{config}-Failed'])

    # Filter to only include existing combinations
    plot_order = [x for x in plot_order if x in df['config_status'].values]

    # Figure 1: Step distribution (Violin plot)
    fig, ax = plt.subplots(1, 1, figsize=(14, 6))
    sns.violinplot(data=df, x='config_status', y='steps', order=plot_order,
                   palette=colors, inner='box', ax=ax)
    ax.set_xlabel('')  # Remove default xlabel, we'll add it manually below config names
    ax.set_ylabel('Agent Steps', fontsize=24)
    # ax.set_title('Step Count Distribution Across Configurations', fontsize=14, fontweight='bold')
    ax.tick_params(axis='x', rotation=0)
    ax.tick_params(axis='y', labelsize=21)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_xlim(-0.5, len(plot_order) - 0.5)

    # Highlight HAFixAgent-Adaptive with a subtle background band
    adaptive_success = 'adaptive-Success'
    adaptive_failed = 'adaptive-Failed'
    if adaptive_success in plot_order and adaptive_failed in plot_order:
        adaptive_start = plot_order.index(adaptive_success) - 0.5
        adaptive_end = plot_order.index(adaptive_failed) + 0.5
        adaptive_center = (adaptive_start + adaptive_end) / 2
        ax.axvspan(adaptive_start, adaptive_end, color=ADAPTIVE_HIGHLIGHT_COLOR, alpha=0.35, zorder=0)
        ax.axvline(x=adaptive_start, color='gray', linestyle='--', alpha=0.6, linewidth=1)

    # Clear default labels and set config names at center of each pair
    ax.set_xticklabels([])

    # Add config labels between pairs and vertical separators
    y_range = ax.get_ylim()[1] - ax.get_ylim()[0]
    for i, config in enumerate(config_order):
        if f'{config}-Success' in plot_order or f'{config}-Failed' in plot_order:
            # Find position of this config's violins
            success_idx = plot_order.index(f'{config}-Success') if f'{config}-Success' in plot_order else None
            failed_idx = plot_order.index(f'{config}-Failed') if f'{config}-Failed' in plot_order else None

            if success_idx is not None and failed_idx is not None:
                center_pos = (success_idx + failed_idx) / 2
                # Config names on first line (closer to axis)
                display_label = resolve_config_label(config, default=config.replace('_', '-'))
                # Strip "HAFixAgent-" prefix for shorter labels
                if display_label.startswith('HAFixAgent-'):
                    display_label = display_label[len('HAFixAgent-'):]
                ax.text(center_pos, ax.get_ylim()[0] - y_range * 0.045,
                       display_label, ha='center', va='top', fontsize=21)

                # Add vertical separator after each config (except last)
                if i < len(config_order) - 1 and failed_idx < len(plot_order) - 1:
                    ax.axvline(x=failed_idx + 0.5, color='gray', linestyle='--', alpha=0.3, linewidth=1)

    # Add "HAFixAgent Context Configuration" label below config names (second line)
    ax.text(0.5, -0.17, 'HAFixAgent Context Configuration', ha='center', va='top', fontsize=24, transform=ax.transAxes)

    # Add legend at bottom center
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#2E7D32', edgecolor='black', label='Success', alpha=0.9),
        Patch(facecolor='#C62828', edgecolor='black', label='Failed', alpha=0.9)
    ]
    ax.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, -0.24),
              fontsize=22, framealpha=0.9, ncol=2)

    plt.tight_layout()
    output_path = output_dir / f'rq3a_step_distribution_{category}.pdf'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Step distribution saved to: {output_path}")

    # Figure 2: Cost distribution (Boxplot for better comparison)
    fig, ax = plt.subplots(1, 1, figsize=(14, 6))

    # Prepare data for boxplot
    box_data = []
    box_labels = []
    box_colors = []
    for config_status in plot_order:
        data_subset = df[df['config_status'] == config_status]['cost'].values
        if len(data_subset) > 0:
            box_data.append(data_subset)
            box_labels.append(config_status)
            box_colors.append(colors[config_status])

    bp = ax.boxplot(box_data, labels=box_labels, patch_artist=True, widths=0.6)

    # Color the boxes
    for patch, color in zip(bp['boxes'], box_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.9)
        patch.set_edgecolor('black')
        patch.set_linewidth(1.2)

    ax.set_xlabel('')  # Remove default xlabel
    ax.set_ylabel('Cost (USD)', fontsize=24)
    # ax.set_title('Monetary Cost Distribution Across Configurations', fontsize=14, fontweight='bold')
    ax.tick_params(axis='x', rotation=0)
    ax.tick_params(axis='y', labelsize=21)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, 0.3)
    ax.set_yticks(np.linspace(0, 0.3, 4))
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))

    # Clear default labels and set config names at center of each pair
    ax.set_xticklabels([])
    ax.set_xlim(0.5, len(box_data) + 0.5)

    # Highlight Adaptive column pair
    if adaptive_success in plot_order and adaptive_failed in plot_order:
        adaptive_start = plot_order.index(adaptive_success) + 0.5
        adaptive_end = plot_order.index(adaptive_failed) + 1.5
        adaptive_center = (adaptive_start + adaptive_end) / 2
        ax.axvspan(adaptive_start, adaptive_end, color=ADAPTIVE_HIGHLIGHT_COLOR, alpha=0.35, zorder=0)
        ax.axvline(x=adaptive_start, color='gray', linestyle='--', alpha=0.6, linewidth=1)

    # Add config labels between pairs and vertical separators
    y_range = ax.get_ylim()[1] - ax.get_ylim()[0]
    for i, config in enumerate(config_order):
        if f'{config}-Success' in plot_order or f'{config}-Failed' in plot_order:
            # Find position of this config's boxes
            success_idx = plot_order.index(f'{config}-Success') if f'{config}-Success' in plot_order else None
            failed_idx = plot_order.index(f'{config}-Failed') if f'{config}-Failed' in plot_order else None

            if success_idx is not None and failed_idx is not None:
                center_pos = (success_idx + failed_idx) / 2
                # Config names on first line (closer to axis)
                display_label = resolve_config_label(config, default=config.replace('_', '-'))
                # Strip "HAFixAgent-" prefix for shorter labels
                if display_label.startswith('HAFixAgent-'):
                    display_label = display_label[len('HAFixAgent-'):]
                ax.text(center_pos + 1, ax.get_ylim()[0] - y_range * 0.045,
                       display_label, ha='center', va='top', fontsize=21)

                # Add vertical separator after each config (except last)
                if i < len(config_order) - 1 and failed_idx < len(plot_order) - 1:
                    ax.axvline(x=failed_idx + 1.5, color='gray', linestyle='--', alpha=0.3, linewidth=1)

    # Add "HAFixAgent Context Configuration" label below config names (second line)
    ax.text(0.5, -0.17, 'HAFixAgent Context Configuration', ha='center', va='top', fontsize=24, transform=ax.transAxes)

    # Add legend at bottom center
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#2E7D32', edgecolor='black', label='Success'),
        Patch(facecolor='#C62828', edgecolor='black', label='Failed')
    ]
    ax.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, -0.24),
              fontsize=22, framealpha=0.9, ncol=2)

    plt.tight_layout()
    output_path = output_dir / f'rq3a_cost_distribution_{category}.pdf'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Cost distribution saved to: {output_path}")


def print_multi_config_stats(df: pd.DataFrame):
    """Print summary statistics for multi-config efficiency metrics."""
    print("\n" + "="*80)
    print("RQ3A: EFFICIENCY METRICS (Multi-Configuration Comparison)")
    print("="*80)

    for config in df['config'].unique():
        config_df = df[df['config'] == config]
        success_df = config_df[config_df['success']]
        failed_df = config_df[~config_df['success']]

        label = resolve_config_label(config, default=config.replace('_', '-'))
        print(f"\n{label}:")
        print(f"  Total cases: {len(config_df)}")
        if len(config_df) > 0:
            print(f"  Successful: {len(success_df)} ({len(success_df)/len(config_df)*100:.1f}%)")

        if len(success_df) > 0:
            print(f"  Success - Avg steps: {success_df['steps'].mean():.1f}, "
                  f"cost: ${success_df['cost'].mean():.4f}, "
                  f"tokens: {success_df['total_tokens'].mean():.0f}")

        if len(failed_df) > 0:
            print(f"  Failed  - Avg steps: {failed_df['steps'].mean():.1f}, "
                  f"cost: ${failed_df['cost'].mean():.4f}, "
                  f"tokens: {failed_df['total_tokens'].mean():.0f}")


def print_median_summary_tables(df: pd.DataFrame):
    """Print median steps and cost tables split by success/failure for each configuration."""
    if 'category' not in df.columns:
        return

    category_labels = {
        'single_line': 'SL',
        'single_hunk': 'SH',
        'single_file_multi_hunk': 'SFMH',
        'multi_file_multi_hunk': 'MFMH',
        'multi_hunk': 'Multi-Hunk',
        'all': 'ALL',
    }
    category_order = ['single_line', 'single_hunk', 'single_file_multi_hunk', 'multi_file_multi_hunk']

    categories_present = []
    for cat in category_order:
        if cat in df['category'].unique():
            categories_present.append(cat)
    # Include any remaining categories (e.g., aggregated labels) preserving appearance order
    for cat in df['category'].unique():
        if cat not in categories_present:
            categories_present.append(cat)

    configs_order = ['baseline', 'fn_all', 'fn_pair', 'fl_diff', 'adaptive']
    configs_present = [c for c in configs_order if c in df['config'].unique()]

    grouped = df.groupby(['category', 'config', 'success']).agg({
        'steps': 'median',
        'cost': 'median'
    })
    steps_median = grouped['steps'].to_dict()
    cost_median = grouped['cost'].to_dict()

    def format_value(value, is_steps):
        if pd.isna(value):
            return '—'
        if is_steps:
            return f"{int(round(value))}"
        return f"{value:.3f}"

    label_lookup = [category_labels.get(cat, cat) for cat in categories_present]
    default_value_width = len("0.000 / 0.000") + 2
    col_width = max(12,
                    max(len(lbl) for lbl in label_lookup) + 4 if label_lookup else 0,
                    default_value_width)

    print("\nMedian steps (Success / Failed):")
    header = "Configuration".ljust(24) + "".join(lbl.center(col_width) for lbl in label_lookup)
    print(header)
    for config in configs_present:
        label = resolve_config_label(config, default=config)
        row_values = []
        for cat in categories_present:
            success_val = steps_median.get((cat, config, True), float('nan'))
            failed_val = steps_median.get((cat, config, False), float('nan'))
            row_values.append(f"{format_value(success_val, True)} / {format_value(failed_val, True)}")
        print(label.ljust(24) + "".join(val.center(col_width) for val in row_values))

    print("\nMedian cost (USD, Success / Failed):")
    print(header)
    for config in configs_present:
        label = resolve_config_label(config, default=config)
        row_values = []
        for cat in categories_present:
            success_val = cost_median.get((cat, config, True), float('nan'))
            failed_val = cost_median.get((cat, config, False), float('nan'))
            row_values.append(f"{format_value(success_val, False)} / {format_value(failed_val, False)}")
        print(label.ljust(24) + "".join(val.center(col_width) for val in row_values))


def find_bug_intersection(df: pd.DataFrame, configs: List[str]) -> Dict[str, Set[str]]:
    """
    Find bugs that were successfully fixed by ALL configurations within each category.

    Args:
        df: DataFrame with columns: bug_id, config, category, success
        configs: List of configuration names (e.g., ['baseline', 'fn_all', 'fn_pair', 'fl_diff'])

    Returns:
        Dict mapping category -> set of bug_ids that all configs successfully fixed
    """
    intersection_by_category = {}

    # Get unique categories
    categories = df['category'].unique()

    for category in categories:
        category_df = df[df['category'] == category]

        # For each config, get set of successfully fixed bugs
        success_sets = []
        for config in configs:
            config_df = category_df[(category_df['config'] == config) & (category_df['success'] == True)]
            success_sets.append(set(config_df['bug_id'].values))

        # Find intersection across all configs
        if success_sets:
            intersection = set.intersection(*success_sets)
            intersection_by_category[category] = intersection
        else:
            intersection_by_category[category] = set()

    return intersection_by_category


def run_statistical_tests(df: pd.DataFrame, output_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run Friedman test + pairwise Wilcoxon tests comparing history heuristics against baseline.

    Only tests bugs that ALL 4 configurations successfully fixed (intersection).
    Tests are run separately for each bug category (SL, SH, SFMH, MFMH).

    Args:
        df: DataFrame with columns: bug_id, config, category, success, steps, cost
        output_dir: Directory to save CSV results

    Returns:
        Tuple of (summary_df, bug_intersection_df)
    """
    print("\n" + "="*80)
    print("STATISTICAL TESTS: Friedman + Pairwise Wilcoxon (Baseline vs History Heuristics)")
    print("="*80)

    # Configurations to compare (exclude adaptive)
    configs = ['baseline', 'fn_all', 'fn_pair', 'fl_diff']

    # Find intersection of successful bugs for each category
    intersection_by_category = find_bug_intersection(df, configs)

    # Prepare results storage
    summary_results = []
    bug_intersection_records = []

    # Category order for consistent output
    category_order = ['single_line', 'single_hunk', 'single_file_multi_hunk', 'multi_file_multi_hunk']
    categories_to_test = [cat for cat in category_order if cat in intersection_by_category]

    for category in categories_to_test:
        intersection_bugs = intersection_by_category[category]
        n_bugs = len(intersection_bugs)

        print(f"\n{category.upper()}: {n_bugs} bugs in intersection")

        # Record bug IDs for this category
        for bug_id in sorted(intersection_bugs):
            bug_intersection_records.append({'category': category, 'bug_id': bug_id})

        if n_bugs == 0:
            print("  Skipping (no common successful bugs)")
            continue

        # Filter to intersection bugs only
        category_df = df[(df['category'] == category) & (df['bug_id'].isin(intersection_bugs)) & (df['success'] == True)]

        # Test both cost and steps
        for metric in ['cost', 'steps']:
            # Prepare data arrays for each config (same bug order for paired tests)
            bug_order = sorted(intersection_bugs)
            data_by_config = {}

            for config in configs:
                config_df = category_df[category_df['config'] == config].set_index('bug_id')
                # Ensure same bug order for all configs
                data_by_config[config] = [config_df.loc[bug, metric] for bug in bug_order]

            # Compute medians
            medians = {config: np.median(data_by_config[config]) for config in configs}

            # Friedman test (non-parametric repeated measures)
            friedman_stat, friedman_p = stats.friedmanchisquare(
                data_by_config['baseline'],
                data_by_config['fn_all'],
                data_by_config['fn_pair'],
                data_by_config['fl_diff']
            )

            # Pairwise Wilcoxon tests (baseline vs each heuristic)
            pairwise_p = {}
            pairwise_r = {}  # Effect size (rank-biserial correlation)
            bonferroni_alpha = 0.05 / 3  # 3 comparisons

            if friedman_p < 0.05:
                for heuristic in ['fn_all', 'fn_pair', 'fl_diff']:
                    # Wilcoxon signed-rank test (paired)
                    stat, p_value = stats.wilcoxon(data_by_config['baseline'], data_by_config[heuristic])
                    pairwise_p[heuristic] = p_value

                    # Calculate rank-biserial correlation: r = 1 - (2*T) / (n*(n+1))
                    # where T is the test statistic and n is the number of pairs
                    n = len(data_by_config['baseline'])
                    r_effect = 1 - (2 * stat) / (n * (n + 1))
                    pairwise_r[heuristic] = r_effect

                    sig_marker = "***" if p_value < bonferroni_alpha else ""
                    print(f"  {metric.capitalize()} - baseline vs {heuristic}: p={p_value:.4f}, r={r_effect:.3f} {sig_marker}")
            else:
                print(f"  {metric.capitalize()} - Friedman test not significant (p={friedman_p:.4f}), skipping pairwise tests")
                for heuristic in ['fn_all', 'fn_pair', 'fl_diff']:
                    pairwise_p[heuristic] = np.nan
                    pairwise_r[heuristic] = np.nan

            # Store results with proper formatting
            is_cost = (metric == 'cost')

            # Format p-values: keep 4 decimals, but preserve scientific notation for very small values
            def format_pvalue(p):
                if pd.isna(p):
                    return np.nan
                elif p < 0.0001:
                    return p  # Keep original scientific notation
                else:
                    return round(p, 4)

            summary_results.append({
                'category': category,
                'metric': metric,
                'n_bugs': n_bugs,
                'baseline_median': round(medians['baseline'], 3) if is_cost else int(round(medians['baseline'])),
                'fn_all_median': round(medians['fn_all'], 3) if is_cost else int(round(medians['fn_all'])),
                'fn_pair_median': round(medians['fn_pair'], 3) if is_cost else int(round(medians['fn_pair'])),
                'fl_diff_median': round(medians['fl_diff'], 3) if is_cost else int(round(medians['fl_diff'])),
                'friedman_p': format_pvalue(friedman_p),
                'fn_all_vs_baseline_p': format_pvalue(pairwise_p['fn_all']),
                'fn_all_vs_baseline_r': round(pairwise_r['fn_all'], 3) if not pd.isna(pairwise_r['fn_all']) else np.nan,
                'fn_pair_vs_baseline_p': format_pvalue(pairwise_p['fn_pair']),
                'fn_pair_vs_baseline_r': round(pairwise_r['fn_pair'], 3) if not pd.isna(pairwise_r['fn_pair']) else np.nan,
                'fl_diff_vs_baseline_p': format_pvalue(pairwise_p['fl_diff']),
                'fl_diff_vs_baseline_r': round(pairwise_r['fl_diff'], 3) if not pd.isna(pairwise_r['fl_diff']) else np.nan,
                'bonferroni_alpha': 0.0167
            })

    # Create DataFrames
    summary_df = pd.DataFrame(summary_results)
    bug_intersection_df = pd.DataFrame(bug_intersection_records)

    # Save to CSV
    summary_csv = output_dir / 'statistical_tests_summary.csv'
    bug_intersection_csv = output_dir / 'statistical_tests_bug_intersection.csv'

    summary_df.to_csv(summary_csv, index=False)
    bug_intersection_df.to_csv(bug_intersection_csv, index=False)

    print(f"\n✅ Statistical tests (successful bugs only) saved to:")
    print(f"   {summary_csv}")
    print(f"   {bug_intersection_csv}")
    print(f"\nNote: Bonferroni-corrected α = 0.05/3 ≈ 0.0167 for significance")

    # ========================================================================
    # SECOND ANALYSIS: All bugs (successful + failed)
    # ========================================================================
    print("\n" + "="*80)
    print("STATISTICAL TESTS (ALL BUGS): Success + Failed")
    print("="*80)

    summary_results_all = []
    bug_all_records = []

    for category in categories_to_test:
        category_df = df[df['category'] == category]

        # Find bugs present in all 4 configs (regardless of success)
        all_bugs_by_config = {}
        for config in configs:
            config_bugs = set(category_df[category_df['config'] == config]['bug_id'].values)
            all_bugs_by_config[config] = config_bugs

        # Intersection of ALL bugs (not just successful)
        all_bugs_intersection = set.intersection(*all_bugs_by_config.values())
        n_bugs_all = len(all_bugs_intersection)

        print(f"\n{category.upper()}: {n_bugs_all} bugs tested by all configs")

        # Record bug IDs
        for bug_id in sorted(all_bugs_intersection):
            bug_all_records.append({'category': category, 'bug_id': bug_id})

        if n_bugs_all == 0:
            print("  Skipping (no common bugs)")
            continue

        # Filter to bugs in intersection (both success and failure)
        category_df_all = category_df[category_df['bug_id'].isin(all_bugs_intersection)]

        # Test both cost and steps
        for metric in ['cost', 'steps']:
            bug_order = sorted(all_bugs_intersection)
            data_by_config = {}

            for config in configs:
                config_df = category_df_all[category_df_all['config'] == config].set_index('bug_id')
                data_by_config[config] = [config_df.loc[bug, metric] for bug in bug_order]

            # Compute medians
            medians = {config: np.median(data_by_config[config]) for config in configs}

            # Friedman test
            friedman_stat, friedman_p = stats.friedmanchisquare(
                data_by_config['baseline'],
                data_by_config['fn_all'],
                data_by_config['fn_pair'],
                data_by_config['fl_diff']
            )

            # Pairwise Wilcoxon tests
            pairwise_p = {}
            bonferroni_alpha = 0.05 / 3

            if friedman_p < 0.05:
                for heuristic in ['fn_all', 'fn_pair', 'fl_diff']:
                    stat, p_value = stats.wilcoxon(data_by_config['baseline'], data_by_config[heuristic])
                    pairwise_p[heuristic] = p_value
                    sig_marker = "***" if p_value < bonferroni_alpha else ""
                    print(f"  {metric.capitalize()} - baseline vs {heuristic}: p={p_value:.4f} {sig_marker}")
            else:
                print(f"  {metric.capitalize()} - Friedman test not significant (p={friedman_p:.4f}), skipping pairwise tests")
                for heuristic in ['fn_all', 'fn_pair', 'fl_diff']:
                    pairwise_p[heuristic] = np.nan

            # Store results
            is_cost = (metric == 'cost')

            def format_pvalue(p):
                if pd.isna(p):
                    return np.nan
                elif p < 0.0001:
                    return p
                else:
                    return round(p, 4)

            summary_results_all.append({
                'category': category,
                'metric': metric,
                'n_bugs': n_bugs_all,
                'baseline_median': round(medians['baseline'], 3) if is_cost else int(round(medians['baseline'])),
                'fn_all_median': round(medians['fn_all'], 3) if is_cost else int(round(medians['fn_all'])),
                'fn_pair_median': round(medians['fn_pair'], 3) if is_cost else int(round(medians['fn_pair'])),
                'fl_diff_median': round(medians['fl_diff'], 3) if is_cost else int(round(medians['fl_diff'])),
                'friedman_p': format_pvalue(friedman_p),
                'fn_all_vs_baseline_p': format_pvalue(pairwise_p['fn_all']),
                'fn_pair_vs_baseline_p': format_pvalue(pairwise_p['fn_pair']),
                'fl_diff_vs_baseline_p': format_pvalue(pairwise_p['fl_diff']),
                'bonferroni_alpha': 0.0167
            })

    # Create DataFrames for all bugs analysis
    summary_df_all = pd.DataFrame(summary_results_all)
    bug_all_df = pd.DataFrame(bug_all_records)

    # Save to CSV
    summary_csv_all = output_dir / 'statistical_tests_all_bugs_summary.csv'
    bug_all_csv = output_dir / 'statistical_tests_all_bugs_intersection.csv'

    summary_df_all.to_csv(summary_csv_all, index=False)
    bug_all_df.to_csv(bug_all_csv, index=False)

    print(f"\n✅ Statistical tests (all bugs: success + failed) saved to:")
    print(f"   {summary_csv_all}")
    print(f"   {bug_all_csv}")

    return summary_df, bug_intersection_df


def parse_trajectory_for_heuristics(traj_file: Path) -> Dict:
    """
    Parse a trajectory file to extract heuristic usage patterns.

    Returns:
        Dict with keys: bug_id, success, heuristics_used (list of {heuristic, step})
    """
    try:
        with open(traj_file) as f:
            traj_data = json.load(f)

        bug_id = traj_file.stem.replace('.traj', '')

        # Try both 'messages' (new format) and 'history' (old format)
        messages = traj_data.get('messages', traj_data.get('history', []))

        # Extract success status from info or last message
        info = traj_data.get('info', {})
        exit_status = info.get('exit_status', '').lower()
        success = exit_status == 'submitted'

        # Find hafix-context command usages
        heuristics_used = []
        step = 0
        for msg in messages:
            if msg.get('role') == 'assistant':
                step += 1
                content = msg.get('content', '')

                # Convert content to string if it's a list (tool calls format)
                if isinstance(content, list):
                    content_str = ' '.join([str(item) for item in content])
                else:
                    content_str = str(content)

                # Look for hafix-context commands - be specific to avoid false positives
                if 'hafix-context fn_all' in content_str:
                    heuristics_used.append({'heuristic': 'fn_all', 'step': step})
                elif 'hafix-context fl_diff' in content_str:
                    heuristics_used.append({'heuristic': 'fl_diff', 'step': step})
                elif 'hafix-context fn_pair' in content_str:
                    heuristics_used.append({'heuristic': 'fn_pair', 'step': step})

        return {
            'bug_id': bug_id,
            'success': success,
            'heuristics_used': heuristics_used
        }
    except Exception as e:
        print(f"Warning: Error parsing {traj_file}: {e}")
        return {'bug_id': traj_file.stem, 'success': False, 'heuristics_used': []}


def analyze_adaptive_heuristic_usage(trajectories_dir: Path, bug_category: str, output_dir: Path, selector_type: str = 'llm_judge', n_lines: int = 1):
    """
    RQ3b: Analyze heuristic usage patterns in adaptive mode.

    Generates 3 separate PNG files:
    - Heuristic selection frequency
    - Heuristic selection timing
    - Success rate by heuristic
    """
    print("\n" + "="*80)
    print("RQ3B: ADAPTIVE HEURISTIC USAGE ANALYSIS")
    print("="*80)

    # Collect trajectory files
    # Adaptive results may be in selector subdirectory
    selector_subdir = f"{selector_type}_{n_lines}line"

    if bug_category == 'multi_hunk':
        # Try with selector subdirectory first
        traj_dirs_with_selector = [
            trajectories_dir / selector_subdir / 'single_file_multi_hunk' / 'trajectories',
            trajectories_dir / selector_subdir / 'multi_file_multi_hunk' / 'trajectories'
        ]
        # Fallback without selector subdirectory
        traj_dirs_without_selector = [
            trajectories_dir / 'single_file_multi_hunk' / 'trajectories',
            trajectories_dir / 'multi_file_multi_hunk' / 'trajectories'
        ]
        # Check which structure exists
        if any(d.exists() for d in traj_dirs_with_selector):
            traj_dirs = traj_dirs_with_selector
        else:
            traj_dirs = traj_dirs_without_selector
    else:
        # Try with selector subdirectory first
        traj_dir_with_selector = trajectories_dir / selector_subdir / bug_category / 'trajectories'
        traj_dir_without_selector = trajectories_dir / bug_category / 'trajectories'

        if traj_dir_with_selector.exists():
            traj_dirs = [traj_dir_with_selector]
        else:
            traj_dirs = [traj_dir_without_selector]

    all_heuristic_data = []

    for traj_dir in traj_dirs:
        if not traj_dir.exists():
            print(f"Warning: Trajectory directory not found: {traj_dir}")
            continue

        traj_files = list(traj_dir.glob("*.traj.json"))
        print(f"Found {len(traj_files)} trajectory files in {traj_dir.name}")

        for traj_file in traj_files:
            parsed = parse_trajectory_for_heuristics(traj_file)
            all_heuristic_data.append(parsed)

    if not all_heuristic_data:
        print("No trajectory data found!")
        return

    # Analyze heuristic usage
    total_bugs = len(all_heuristic_data)
    bugs_using_heuristics = [b for b in all_heuristic_data if b['heuristics_used']]
    successful_bugs = [b for b in all_heuristic_data if b['success']]

    print(f"\nTotal bugs: {total_bugs}")
    print(f"Bugs that used heuristics: {len(bugs_using_heuristics)} ({len(bugs_using_heuristics)/total_bugs*100:.1f}%)")
    print(f"Successful bugs: {len(successful_bugs)} ({len(successful_bugs)/total_bugs*100:.1f}%)")

    # Count heuristic usage at bug level (each bug counted once per heuristic type)
    bugs_using_heuristic = {'fn_all': set(), 'fl_diff': set(), 'fn_pair': set()}
    bugs_succeeded_with_heuristic = {'fn_all': set(), 'fl_diff': set(), 'fn_pair': set()}
    bugs_failed_with_heuristic = {'fn_all': set(), 'fl_diff': set(), 'fn_pair': set()}
    step_data = {'fn_all': [], 'fl_diff': [], 'fn_pair': []}

    for bug in all_heuristic_data:
        heuristics_this_bug = set()
        for h_usage in bug['heuristics_used']:
            heuristic = h_usage['heuristic']
            heuristics_this_bug.add(heuristic)
            step_data[heuristic].append(h_usage['step'])

        # Track unique bugs per heuristic
        for h in heuristics_this_bug:
            bugs_using_heuristic[h].add(bug['bug_id'])
            if bug['success']:
                bugs_succeeded_with_heuristic[h].add(bug['bug_id'])
            else:
                bugs_failed_with_heuristic[h].add(bug['bug_id'])

    heuristics = list(bugs_using_heuristic.keys())

    # Figure 1: Heuristic usage with success/failure breakdown (stacked bar)
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    success_counts = [len(bugs_succeeded_with_heuristic[h]) for h in heuristics]
    failed_counts = [len(bugs_failed_with_heuristic[h]) for h in heuristics]

    # Stacked bar chart
    bars1 = ax.bar(heuristics, success_counts, color='#2E7D32', alpha=0.8,
                   edgecolor='black', linewidth=1.5, label='Success')
    bars2 = ax.bar(heuristics, failed_counts, bottom=success_counts, color='#C62828',
                   alpha=0.8, edgecolor='black', linewidth=1.5, label='Failed')

    ax.set_xlabel('Heuristic', fontsize=13)
    ax.set_ylabel('Number of Bugs', fontsize=13)
    # ax.set_title('Heuristic Usage Pattern (Success vs Failed)', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    ax.legend(loc='upper right', fontsize=11, framealpha=0.9)

    # Add total counts on top of bars
    for i, h in enumerate(heuristics):
        total = success_counts[i] + failed_counts[i]
        ax.text(i, total + max([s+f for s,f in zip(success_counts, failed_counts)])*0.02,
               str(total), ha='center', va='bottom', fontsize=11)

    plt.tight_layout()
    output_path = output_dir / 'rq3b_heuristic_usage.pdf'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\nHeuristic usage saved to: {output_path}")

    # Figure 2: Selection timing (step number)
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    step_data_list = [step_data[h] for h in heuristics if step_data[h]]
    heuristics_with_data = [h for h in heuristics if step_data[h]]

    colors_timing = ['#4A90E2', '#E67E22', '#9B59B6']

    if step_data_list and heuristics_with_data:
        bp = ax.boxplot(step_data_list, labels=heuristics_with_data, patch_artist=True)
        for patch, color in zip(bp['boxes'], colors_timing):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
            patch.set_edgecolor('black')
            patch.set_linewidth(1.5)
    else:
        ax.text(0.5, 0.5, 'No heuristic usage data available',
                ha='center', va='center', transform=ax.transAxes, fontsize=14)

    ax.set_xlabel('Heuristic', fontsize=13)
    ax.set_ylabel('Step Number', fontsize=13)
    # ax.set_title('Heuristic Selection Timing', fontsize=14)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    output_path = output_dir / 'rq3b_heuristic_timing.pdf'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Heuristic timing saved to: {output_path}")

    # Print detailed statistics
    print(f"\nHeuristic Usage Statistics:")
    for h in heuristics:
        total_bugs = len(bugs_using_heuristic[h])
        if total_bugs > 0:
            success_bugs = len(bugs_succeeded_with_heuristic[h])
            success_rate = success_bugs / total_bugs * 100
            print(f"  {h}:")
            print(f"    Used by {total_bugs} bugs")
            print(f"    Success rate: {success_bugs}/{total_bugs} ({success_rate:.1f}%)")
            if step_data[h]:
                print(f"    Avg selection step: {sum(step_data[h])/len(step_data[h]):.1f}")


def process_results_directory(results_dir: Path, output_dir: Path, category_name: str, history_flag: Optional[str] = None):
    """Process a results directory and generate analysis."""
    try:
        # Load data
        history_desc = f" (history_flag {history_flag})" if history_flag else ""
        print(f"Loading data from {results_dir}{history_desc}...")
        df = load_metrics(results_dir, history_flag)
        print(f"Loaded {len(df)} cases")

        # Print summary
        print_summary_stats(df, category_name)

        # Create individual figures (excluding summary table and duration)
        create_individual_figures(df, output_dir, history_flag)

        print(f"\n✅ Analysis complete! Generated figures:")
        print(f"      {output_dir}/steps_{history_flag}.pdf")
        print(f"      {output_dir}/cost_{history_flag}.pdf")
        print(f"      {output_dir}/cost_vs_steps_{history_flag}.pdf")

    except Exception as e:
        print(f"Error: {e}")
        raise


# Create typer app
app = typer.Typer(rich_markup_mode="rich", add_completion=False)
console = Console()

@app.command(help="Analyze HAFixAgent evaluation results and generate visualizations")
def main(
    # Mode selection
    mode: Optional[str] = typer.Option(None, "--mode", "-m", help="Analysis mode: 'single' (default), 'multi-config' (RQ3a), or 'adaptive' (RQ3b)", rich_help_panel="Mode"),

    # Single-config mode options
    results_dir: Optional[str] = typer.Option(None, "--results-dir", "-r", help="Path to specific results directory (single-config mode)", rich_help_panel="Single Config"),
    project: Optional[str] = typer.Option(None, "--project", "-p", help="Specific project (single-config mode)", rich_help_panel="Single Config"),
    history_flag: Optional[str] = typer.Option(None, "--history-flag", "-h", help="History flag (single-config mode)", rich_help_panel="Single Config"),

    # Multi-config mode options (RQ3a)
    rq1_dir: Optional[str] = typer.Option(None, "--rq1-dir", help="RQ1 results directory (multi-config mode)", rich_help_panel="Multi-Config (RQ3a)"),
    rq2_dir: Optional[str] = typer.Option(None, "--rq2-dir", help="RQ2 results directory (multi-config mode)", rich_help_panel="Multi-Config (RQ3a)"),
    include_adaptive: bool = typer.Option(True, "--include-adaptive/--no-adaptive", help="Include adaptive configuration in comparison (default: True)", rich_help_panel="Multi-Config (RQ3a)"),

    # Adaptive mode options (RQ3b)
    trajectories_dir: Optional[str] = typer.Option(None, "--trajectories-dir", help="Trajectories directory (adaptive mode)", rich_help_panel="Adaptive (RQ3b)"),

    # Common options
    bug_category: Optional[str] = typer.Option(None, "--bug-category", "-c", help="Bug category: 'single_file_multi_hunk', 'multi_file_multi_hunk', 'single_hunk', 'single_line', 'multi_hunk', or 'all'", rich_help_panel="Common"),
    selector_type: str = typer.Option("llm_judge", "--selector-type", "-s", help="Line selection strategy", rich_help_panel="Common"),
    n_lines: int = typer.Option(1, "--n-lines", "-n", help="Number of lines selected", rich_help_panel="Common"),
    output: str = typer.Option("results/rq3_analysis", "--output", "-o", help="Output directory", rich_help_panel="Common"),
) -> None:

    console.print("[bold blue]HAFixAgent Results Analysis[/bold blue]", style="bold")

    # Determine mode
    if mode is None:
        mode = 'single'  # Default mode

    if mode not in ['single', 'multi-config', 'adaptive']:
        console.print(f"[red]Error: Invalid mode '{mode}'. Must be 'single', 'multi-config', or 'adaptive'[/red]")
        raise typer.Exit(1)

    output_dir = Path(output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # MODE 1: Single configuration analysis
    if mode == 'single':
        if results_dir:
            results_path = Path(results_dir)
            if not results_path.exists():
                console.print(f"[red]Error: Results directory {results_path} does not exist[/red]")
                raise typer.Exit(1)
            category_name = results_path.name
        elif bug_category:
            base_results = Path("results/defects4j")
            selector_dir = f"{selector_type}_{n_lines}line"
            if project:
                results_path = base_results / selector_dir / bug_category / project.lower()
                category_name = f"{bug_category} - {project}"
            else:
                results_path = base_results / selector_dir / bug_category
                category_name = bug_category

            if not results_path.exists():
                console.print(f"[red]Error: Results directory {results_path} does not exist[/red]")
                raise typer.Exit(1)
        else:
            console.print("[red]Error: Must specify either --results-dir or --bug-category for single-config mode[/red]")
            raise typer.Exit(1)

        console.print(f"Mode: [green]Single Configuration[/green]")
        console.print(f"Results: [cyan]{results_path}[/cyan]")
        console.print(f"Output: [cyan]{output_dir}[/cyan]")
        process_results_directory(results_path, output_dir, category_name, history_flag)

    # MODE 2: Multi-configuration comparison (RQ3a)
    elif mode == 'multi-config':
        if not rq1_dir or not rq2_dir or not bug_category:
            console.print("[red]Error: Multi-config mode requires --rq1-dir, --rq2-dir, and --bug-category[/red]")
            console.print("[yellow]Example:[/yellow]")
            console.print("  [cyan]python analysis/analyze_running_result.py --mode multi-config --rq1-dir results/defects4j --rq2-dir results/defects4j_adaptive --bug-category multi_hunk[/cyan]")
            raise typer.Exit(1)

        console.print(f"Mode: [green]Multi-Configuration (RQ3a: Efficiency)[/green]")
        console.print(f"RQ1 directory: [cyan]{rq1_dir}[/cyan]")
        console.print(f"RQ2 directory: [cyan]{rq2_dir}[/cyan]")
        console.print(f"Bug Category: [cyan]{bug_category}[/cyan]")
        console.print(f"Output: [cyan]{output_dir}[/cyan]")

        # Define configurations
        configs = [
            {'name': 'baseline', 'flag': BASELINE_FLAG, 'is_adaptive': False},
            {'name': 'fn_all', 'flag': FN_ALL_FLAG, 'is_adaptive': False},
            {'name': 'fn_pair', 'flag': FN_PAIR_FLAG, 'is_adaptive': False},
            {'name': 'fl_diff', 'flag': FL_DIFF_FLAG, 'is_adaptive': False},
            {'name': 'adaptive', 'flag': ADAPTIVE_FLAG, 'is_adaptive': True}
        ]

        # Load multi-config metrics
        df = load_multi_config_metrics(configs, Path(rq1_dir), Path(rq2_dir),
                                      bug_category, selector_type, n_lines)

        # Filter dataframe if adaptive should be excluded
        if not include_adaptive:
            df = df[df['config'] != 'adaptive']
            console.print("[yellow]Note: Adaptive configuration excluded from analysis[/yellow]")

        # Print statistics
        print_multi_config_stats(df)
        print_median_summary_tables(df)

        # Generate violin plots
        create_multi_config_violin_separate(df, output_dir, bug_category, include_adaptive=include_adaptive)

        # Save CSV
        csv_path = output_dir / f'rq3a_metrics_{bug_category}.csv'
        df.to_csv(csv_path, index=False)
        print(f"\n✅ Metrics saved to: {csv_path}")

        # Run statistical tests (only if adaptive is excluded)
        if not include_adaptive:
            run_statistical_tests(df, output_dir)

    # MODE 3: Adaptive behavior analysis (RQ3b)
    elif mode == 'adaptive':
        if not trajectories_dir or not bug_category:
            console.print("[red]Error: Adaptive mode requires --trajectories-dir and --bug-category[/red]")
            console.print("[yellow]Example:[/yellow]")
            console.print("  [cyan]python analysis/analyze_running_result.py --mode adaptive --trajectories-dir results/defects4j_adaptive --bug-category multi_hunk[/cyan]")
            raise typer.Exit(1)

        console.print(f"Mode: [green]Adaptive Behavior (RQ3b: Heuristic Usage)[/green]")
        console.print(f"Trajectories: [cyan]{trajectories_dir}[/cyan]")
        console.print(f"Bug Category: [cyan]{bug_category}[/cyan]")
        console.print(f"Output: [cyan]{output_dir}[/cyan]")

        # Analyze adaptive heuristic usage
        analyze_adaptive_heuristic_usage(Path(trajectories_dir), bug_category, output_dir, selector_type, n_lines)

    print("\n✅ Analysis complete!")


if __name__ == "__main__":
    app()
