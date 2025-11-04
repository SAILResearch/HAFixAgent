"""
Cost-Effectiveness Trade-off Analysis for HAFixAgent

Analyzes the Pareto frontier of cost vs. effectiveness by exploring all possible
combinations of configurations (ensemble approach). This helps understand whether
running multiple configurations in parallel could improve success rates, and at what cost.

For each bug category, generates:
- Scatter plots showing cost vs. success rate for all configuration combinations
- CSV files with detailed metrics for each combination
- Pareto frontier highlighting (optimal cost-effectiveness points)

Usage Examples:

    # Analyze single bug category
    python analysis/analyze_rq3_cost_effectiveness_tradeoff.py --bug-category single_line

    # Analyze all categories
    python analysis/analyze_rq3_cost_effectiveness_tradeoff.py --bug-category all

    # Use simplified visualization for research papers
    python analysis/analyze_rq3_cost_effectiveness_tradeoff.py --bug-category single_file_multi_hunk --viz-style simplified

    # Use individual markers for each combination (15 different markers)
    python analysis/analyze_rq3_cost_effectiveness_tradeoff.py --bug-category all --viz-style individual

Bug Categories:
    - 'single_line': SL only
    - 'single_hunk': SH only
    - 'single_file_multi_hunk': SFMH only
    - 'multi_file_multi_hunk': MFMH only
    - 'multi_hunk': SFMH + MFMH
    - 'all': All bugs (SL + SH + SFMH + MFMH)

Visualization Styles:
    - 'default': Color by number of configs (1/2/3/4), Pareto highlighted
    - 'simplified': Show only single configs + Pareto frontier (best for papers)
    - 'individual': Each combination has unique marker (15 different)

Output Files:
    - rq3_tradeoff_{category}.pdf: Scatter plot of cost vs. success rate
    - rq3_tradeoff_{category}_combined.pdf: Combined plot with all categories
    - rq3_tradeoff_{category}_details.csv: Detailed metrics for each combination
"""

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import ticker
import seaborn as sns
from pathlib import Path
from itertools import combinations
import typer
from rich.console import Console
from typing import List, Dict, Tuple
import numpy as np

# Set style for publication quality
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 16,
    'font.family': 'serif',
    'axes.grid': True,
    'grid.alpha': 0.3
})

# Configuration names (excluding adaptive for RQ1 comparison)
CONFIGS = ['baseline', 'fn_all', 'fn_pair', 'fl_diff']

# Color scheme for different combination sizes
COLORS = {
    1: '#1f77b4',  # Blue for single configs
    2: '#ff7f0e',  # Orange for 2-config combinations
    3: '#2ca02c',  # Green for 3-config combinations
    4: '#d62728',  # Red for all 4 configs
}

# Marker styles for individual combinations (15 total)
MARKERS = ['o', 's', '^', 'v', 'D', 'p', '*', 'h', 'H', 'X', 'P', '<', '>', '8', 'd']

# Bug category mapping for filtering from rq3a_metrics_all.csv
CATEGORY_MAPPING = {
    'single_line': 'single_line',
    'single_hunk': 'single_hunk',
    'single_file_multi_hunk': 'single_file_multi_hunk',
    'multi_file_multi_hunk': 'multi_file_multi_hunk',
    'multi_hunk': ['single_file_multi_hunk', 'multi_file_multi_hunk'],
    'all': ['single_line', 'single_hunk', 'single_file_multi_hunk', 'multi_file_multi_hunk']
}


def get_bug_category_from_bug_id(bug_id: str) -> str:
    """
    Determine bug category from bug_id based on naming convention.

    Args:
        bug_id: Bug identifier (e.g., "Lang-1", "Chart-20")

    Returns:
        Category name: 'single_line', 'single_hunk', 'single_file_multi_hunk', or 'multi_file_multi_hunk'
    """
    # This is a placeholder - we'll infer from separate CSVs if needed
    # For now, we rely on the category-specific CSVs
    return 'unknown'


def load_metrics_csv(input_dir: Path, bug_category: str, use_all_csv: bool = True) -> pd.DataFrame:
    """
    Load metrics from pre-generated CSV file.

    Strategy:
    1. Try to use rq3a_metrics_all.csv and filter by category (most efficient)
    2. Fall back to category-specific CSV if _all.csv doesn't exist or filtering fails
    3. For 'multi_hunk' and 'all', always use _all.csv and filter

    Args:
        input_dir: Directory containing rq3a_metrics_*.csv files
        bug_category: Bug category name
        use_all_csv: If True, prefer using _all.csv (default: True)

    Returns:
        DataFrame with columns: bug_id, config, success, cost, etc.
    """
    # For composite categories, we must use _all.csv or individual category CSVs
    if bug_category in ['multi_hunk', 'all']:
        all_csv = input_dir / 'rq3a_metrics_all.csv'

        if not all_csv.exists():
            raise FileNotFoundError(f"Metrics CSV not found: {all_csv}\n"
                                  f"For composite categories ('{bug_category}'), rq3a_metrics_all.csv is required.")

        df_all = pd.read_csv(all_csv)

        # Get list of category-specific CSVs to determine which bugs belong to which category
        category_to_use = CATEGORY_MAPPING[bug_category]

        if isinstance(category_to_use, list):
            # Load individual category CSVs to get bug lists
            all_bug_ids = set()
            for cat in category_to_use:
                cat_csv = input_dir / f'rq3a_metrics_{cat}.csv'
                if cat_csv.exists():
                    df_cat = pd.read_csv(cat_csv)
                    all_bug_ids.update(df_cat['bug_id'].unique())

            # Filter _all.csv to only include bugs from these categories
            df = df_all[df_all['bug_id'].isin(all_bug_ids)]
        else:
            # Single category - can use category-specific CSV
            cat_csv = input_dir / f'rq3a_metrics_{category_to_use}.csv'
            if cat_csv.exists():
                df = pd.read_csv(cat_csv)
            else:
                # Fall back to filtering _all.csv (less reliable without category mapping)
                raise FileNotFoundError(f"Category-specific CSV not found: {cat_csv}")

    else:
        # Single category - try category-specific CSV first
        csv_file = input_dir / f'rq3a_metrics_{bug_category}.csv'

        if use_all_csv:
            # Try _all.csv if it exists
            all_csv = input_dir / 'rq3a_metrics_all.csv'
            if all_csv.exists() and csv_file.exists():
                # Load category-specific CSV to get bug list
                df_cat = pd.read_csv(csv_file)
                bug_ids = df_cat['bug_id'].unique()

                # Filter _all.csv
                df_all = pd.read_csv(all_csv)
                df = df_all[df_all['bug_id'].isin(bug_ids)]
            elif csv_file.exists():
                df = pd.read_csv(csv_file)
            else:
                raise FileNotFoundError(f"Metrics CSV not found: {csv_file}")
        else:
            if not csv_file.exists():
                raise FileNotFoundError(f"Metrics CSV not found: {csv_file}")
            df = pd.read_csv(csv_file)

    # Filter to only include baseline and 3 heuristics (exclude adaptive)
    df = df[df['config'].isin(CONFIGS)]

    return df


def compute_combination_metrics(df: pd.DataFrame, config_combo: Tuple[str, ...]) -> Dict:
    """
    Compute metrics for a specific configuration combination.

    For each bug:
    - Cost = sum of costs across all configs in the combination
    - Success = True if at least one config succeeded

    Args:
        df: DataFrame with bug results
        config_combo: Tuple of config names (e.g., ('baseline', 'fn_all'))

    Returns:
        Dictionary with:
        - combination: config names joined by '+'
        - num_configs: number of configs in combination
        - total_cost: sum of costs for all bugs
        - num_bugs_fixed: number of bugs where at least one config succeeded
        - success_rate: percentage of bugs fixed
        - avg_cost_per_bug: average cost per bug
    """
    # Get unique bugs
    all_bugs = df['bug_id'].unique()

    # For each bug, compute combined cost and success
    bug_costs = []
    bug_successes = []

    for bug_id in all_bugs:
        bug_data = df[df['bug_id'] == bug_id]

        # Get data for configs in this combination
        combo_data = bug_data[bug_data['config'].isin(config_combo)]

        # Sum costs
        total_cost = combo_data['cost'].sum()
        bug_costs.append(total_cost)

        # Check if any config succeeded
        any_success = combo_data['success'].any()
        bug_successes.append(any_success)

    # Aggregate metrics
    total_bugs = len(all_bugs)
    num_bugs_fixed = sum(bug_successes)
    total_cost = sum(bug_costs)
    success_rate = (num_bugs_fixed / total_bugs * 100) if total_bugs > 0 else 0
    avg_cost_per_bug = total_cost / total_bugs if total_bugs > 0 else 0

    return {
        'combination': '+'.join(config_combo),
        'num_configs': len(config_combo),
        'total_cost': total_cost,
        'num_bugs_fixed': num_bugs_fixed,
        'total_bugs': total_bugs,
        'success_rate': success_rate,
        'avg_cost_per_bug': avg_cost_per_bug,
        'configs': list(config_combo)
    }


def generate_all_combinations(configs: List[str]) -> List[Dict]:
    """
    Generate all possible combinations of configurations.

    Args:
        configs: List of configuration names

    Returns:
        List of combination metrics dictionaries
    """
    all_combinations = []

    # Generate combinations of sizes 1, 2, 3, 4
    for size in range(1, len(configs) + 1):
        for combo in combinations(configs, size):
            all_combinations.append(combo)

    return all_combinations


def compute_pareto_frontier(results: List[Dict], cost_metric: str = 'total') -> List[bool]:
    """
    Identify points on the Pareto frontier (optimal cost-effectiveness).

    A point is on the Pareto frontier if no other point has both:
    - Lower or equal cost AND higher success rate

    Args:
        results: List of combination metrics

    Returns:
        List of booleans indicating which points are on the frontier
    """
    cost_key = 'total_cost' if cost_metric == 'total' else 'avg_cost_per_bug'
    pareto_mask = []

    for i, point_i in enumerate(results):
        is_pareto = True

        for j, point_j in enumerate(results):
            if i == j:
                continue

            # Check if point_j dominates point_i
            # (lower/equal cost AND higher success rate)
            if (point_j[cost_key] <= point_i[cost_key] and
                point_j['success_rate'] > point_i['success_rate']):
                is_pareto = False
                break

        pareto_mask.append(is_pareto)

    return pareto_mask


def create_scatter_plot(results: List[Dict], output_path: Path, category: str,
                        viz_style: str = 'default', highlight_pareto: bool = True,
                        cost_metric: str = 'total'):
    """
    Create scatter plot of cost vs. success rate for a single category.

    Uses consistent visual encoding with single-panel plot:
    - Fixed color for the category
    - Different marker shapes for number of configs (1=circle, 2=square, 3=triangle, 4=diamond)

    Args:
        results: List of combination metrics
        output_path: Path to save the plot
        category: Bug category name
        viz_style: Visualization style ('default', 'simplified', or 'individual')
        highlight_pareto: Whether to highlight Pareto frontier points
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 7))

    # Prepare data
    df = pd.DataFrame(results)

    # Compute Pareto frontier
    pareto_mask = compute_pareto_frontier(results, cost_metric)
    df['is_pareto'] = pareto_mask

    # Category color (consistent with single-panel plot)
    category_colors = {
        'single_line': '#1f77b4',      # Medium blue
        'single_hunk': '#2ca02c',      # Green
        'single_file_multi_hunk': '#000000',  # Black
        'multi_file_multi_hunk': '#d62728',   # Red
    }

    # Marker shapes for number of configs
    config_markers = {
        1: 'o',  # Circle
        2: 's',  # Square
        3: '^',  # Triangle
        4: 'D'   # Diamond
    }

    category_color = category_colors.get(category, '#1f77b4')
    cost_key = 'total_cost' if cost_metric == 'total' else 'avg_cost_per_bug'
    x_label = 'Total Cost (USD)' if cost_metric == 'total' else 'Average Cost per Bug (USD)'

    if viz_style == 'simplified':
        # Simplified: Show only single configs + Pareto frontier
        # Best for research papers - clean and focused
        single_configs = df[df['num_configs'] == 1]
        pareto_points = df[df['is_pareto']]

        # Plot single configs
        config_colors = {'baseline': '#1f77b4', 'fn_all': '#ff7f0e',
                        'fn_pair': '#2ca02c', 'fl_diff': '#d62728'}
        for _, row in single_configs.iterrows():
            config_name = row['combination']
            ax.scatter(row[cost_key], row['success_rate'],
                      color=config_colors.get(config_name, '#gray'),
                      s=150, alpha=0.7, edgecolors='black', linewidth=1,
                      label=config_name.upper(), zorder=5)

        # Plot Pareto frontier points (excluding singles)
            pareto_multi = pareto_points[pareto_points['num_configs'] > 1]
            if len(pareto_multi) > 0:
                ax.scatter(pareto_multi[cost_key], pareto_multi['success_rate'],
                      color='purple', s=250, alpha=0.9,
                      edgecolors='gold', linewidth=3, marker='*',
                      label='Pareto Optimal', zorder=10)

            # Annotate Pareto points
                for _, row in pareto_multi.iterrows():
                    cost_text = f"${row['total_cost']:.0f}" if cost_metric == 'total' else f"${row['avg_cost_per_bug']:.2f}"
                    ax.annotate(f"{row['combination']}\n({row['success_rate']:.1f}%, {cost_text})",
                               xy=(row[cost_key], row['success_rate']),
                           xytext=(15, 10), textcoords='offset points',
                           fontsize=8, alpha=0.9,
                           bbox=dict(boxstyle='round,pad=0.4', facecolor='lightyellow',
                                   alpha=0.8, edgecolor='gold', linewidth=1.5),
                           arrowprops=dict(arrowstyle='->', color='gray', lw=0.8))

    elif viz_style == 'individual':
        # Individual: Each combination has unique marker
        # Shows all 15 combinations distinctly
        for idx, row in df.iterrows():
            marker = MARKERS[idx % len(MARKERS)]
            color = COLORS[row['num_configs']]

            # Highlight Pareto points
            if row['is_pareto']:
                edgecolor = 'gold'
                linewidth = 2.5
                size = 180
                alpha = 1.0
            else:
                edgecolor = 'black'
                linewidth = 1
                size = 120
                alpha = 0.6

            ax.scatter(row[cost_key], row['success_rate'],
                      color=color, s=size, alpha=alpha,
                      marker=marker, edgecolors=edgecolor, linewidth=linewidth,
                      label=row['combination'], zorder=5 if row['is_pareto'] else 3)

    else:  # default - consistent with single-panel style
        # Plot all points with consistent category color and different marker shapes
        for _, row in df.iterrows():
            marker = config_markers[row['num_configs']]
            cost_value = row['total_cost'] if cost_metric == 'total' else row['avg_cost_per_bug']

            # Determine size and edge based on Pareto status
            if highlight_pareto and row['is_pareto']:
                size = 150
                edgecolor = '#FFD700'  # Gold
                linewidth = 2
                alpha = 0.9
                zorder = 10
            else:
                size = 100
                edgecolor = 'black'
                linewidth = 0.8
                alpha = 0.7
                zorder = 5

            ax.scatter(row[cost_key], row['success_rate'],
                      color=category_color, s=size, alpha=alpha,
                      marker=marker, edgecolors=edgecolor, linewidth=linewidth,
                      zorder=zorder)

    ax.set_xlabel(x_label, fontsize=14)
    ax.set_ylabel('Pass Rate (%)', fontsize=14)
    # ax.set_title(f'Cost-Effectiveness Trade-off Analysis\n({category.replace("_", " ").title()})',
    #             fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)

    # Legend - consistent with single-panel style
    if viz_style == 'individual':
        # Put legend outside for individual (too many items)
        ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5),
                 fontsize=8, framealpha=0.9, ncol=1)
    elif viz_style == 'simplified':
        ax.legend(loc='lower right', fontsize=10, framealpha=0.9)
    else:
        # Default: Show marker shapes for config counts (no Pareto legend, consistent with --no-pareto)
        from matplotlib.lines import Line2D

        config_labels = {
            1: '1 configuration',
            2: '2 configurations',
            3: '3 configurations',
            4: '4 configurations'
        }

        legend_elements = [
            Line2D([0], [0], marker=config_markers[i], color='w',
                   markerfacecolor='white', markersize=9, label=config_labels[i],
                   markeredgecolor='black', markeredgewidth=1.2)
            for i in sorted(config_markers.keys())
        ]

        # Add Pareto legend if enabled
        if highlight_pareto:
            legend_elements.append(
                Line2D([0], [0], marker='o', color='w',
                       markerfacecolor='white', markersize=10, label='Pareto Optimal',
                       markeredgecolor='#FFD700', markeredgewidth=2.5)
            )

        ax.legend(handles=legend_elements, loc='lower right', fontsize=10, framealpha=0.95)

    # Set y-axis to 0-100 range
    ax.set_ylim(-5, 105)

    if viz_style == 'individual':
        plt.tight_layout(rect=[0, 0, 0.85, 1])  # Make room for legend
    else:
        plt.tight_layout()

    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Scatter plot ({viz_style} style) saved to: {output_path}")


def save_detailed_csv(results: List[Dict], output_path: Path):
    """
    Save detailed results to CSV.

    Args:
        results: List of combination metrics
        output_path: Path to save the CSV
    """
    df = pd.DataFrame(results)

    # Reorder columns for clarity
    column_order = ['combination', 'num_configs', 'success_rate', 'num_bugs_fixed',
                   'total_bugs', 'total_cost', 'avg_cost_per_bug']
    df = df[column_order]

    # Sort by num_configs, then by success_rate descending
    df = df.sort_values(['num_configs', 'success_rate'], ascending=[True, False])

    df.to_csv(output_path, index=False, float_format='%.4f')
    print(f"Detailed CSV saved to: {output_path}")


def create_combined_plot(all_results: Dict[str, List[Dict]], output_path: Path,
                         single_panel: bool = False, highlight_pareto: bool = True,
                         cost_metric: str = 'total'):
    """
    Create a combined plot showing all bug categories.

    Args:
        all_results: Dictionary mapping category names to results lists
        output_path: Path to save the combined plot
        single_panel: If True, create one panel with all categories overlaid
        highlight_pareto: Whether to highlight Pareto frontier points
    """
    if single_panel:
        # Single panel: All categories in one plot
        create_single_panel_combined(all_results, output_path, highlight_pareto, cost_metric)
    else:
        # Multi-panel: Separate subplot for each category
        create_multi_panel_combined(all_results, output_path, highlight_pareto, cost_metric)


def create_single_panel_combined(all_results: Dict[str, List[Dict]], output_path: Path,
                                  highlight_pareto: bool = True, cost_metric: str = 'total'):
    """
    Create a single-panel plot with all categories overlaid.

    Visual encoding:
    - Color: Category (SL=blue, SH=green, SFMH=orange, MFMH=red)
    - Marker: Number of configs (1=circle, 2=square, 3=triangle, 4=diamond)
    - Size/Edge: Pareto optimal (larger, gold edge)
    """
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))

    # Category colors - optimized for distinction in research papers
    category_colors = {
        'single_line': '#1f77b4',      # Medium blue
        'single_hunk': '#2ca02c',      # Green
        'single_file_multi_hunk': '#000000',  # Black
        'multi_file_multi_hunk': '#d62728',   # Red
        'all': '#7f7f7f'               # Gray (if included)
    }

    # Marker shapes for number of configs
    config_markers = {
        1: 'o',  # Circle
        2: 's',  # Square
        3: '^',  # Triangle
        4: 'D'   # Diamond
    }

    # Filter to only 4 base categories (exclude composite categories like 'all')
    base_categories = ['single_line', 'single_hunk', 'single_file_multi_hunk', 'multi_file_multi_hunk']
    filtered_results = {k: v for k, v in all_results.items() if k in base_categories}

    # Track legend entries (avoid duplicates)
    legend_entries = {}

    cost_key = 'total_cost' if cost_metric == 'total' else 'avg_cost_per_bug'
    x_label = 'Total Cost (USD)' if cost_metric == 'total' else 'Average Cost per Bug (USD)'

    for category, results in filtered_results.items():
        df = pd.DataFrame(results)

        if highlight_pareto:
            pareto_mask = compute_pareto_frontier(results, cost_metric)
            df['is_pareto'] = pareto_mask
        else:
            df['is_pareto'] = False

        color = category_colors.get(category, '#gray')

        # Plot each point
        for _, row in df.iterrows():
            marker = config_markers[row['num_configs']]
            cost_value = row['total_cost'] if cost_metric == 'total' else row['avg_cost_per_bug']

            # Determine size and edge based on Pareto status
            if highlight_pareto and row['is_pareto']:
                size = 150
                edgecolor = '#FFD700'  # Gold
                linewidth = 2
                alpha = 0.9
                zorder = 10
            else:
                size = 100
                edgecolor = 'black'
                linewidth = 0.8
                alpha = 0.7
                zorder = 5

            # Create label for legend (only once per category)
            label = None
            if category not in legend_entries:
                label = category.replace('_', ' ').title()
                legend_entries[category] = True

            ax.scatter(row[cost_key], row['success_rate'],
                      color=color, s=size, alpha=alpha,
                      marker=marker, edgecolors=edgecolor, linewidth=linewidth,
                      label=label, zorder=zorder)

    ax.set_xlabel(x_label, fontsize=14)
    ax.set_ylabel('Pass Rate (%)', fontsize=14)
    # ax.set_title('Cost-Effectiveness Trade-off Across Bug Categories',
    #             fontsize=16, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-5, 105)

    # Create comprehensive legend
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    # Category legend (colors) with shorter labels
    category_labels = {
        'single_line': 'SL',
        'single_hunk': 'SH',
        'single_file_multi_hunk': 'SFMH',
        'multi_file_multi_hunk': 'MFMH'
    }

    category_legend = [
        Patch(facecolor=category_colors[cat], label=category_labels[cat], alpha=0.8)
        for cat in base_categories if cat in filtered_results
    ]

    # Config count legend (markers) - white fill for clarity
    # Use grammatically correct singular/plural: "1 configuration", "2 configurations", etc.
    config_labels = {
        1: '1 configuration',
        2: '2 configurations',
        3: '3 configurations',
        4: '4 configurations'
    }

    config_legend = [
        Line2D([0], [0], marker=config_markers[i], color='w',
               markerfacecolor='white', markersize=9, label=config_labels[i],
               markeredgecolor='black', markeredgewidth=1.2)
        for i in sorted(config_markers.keys())
    ]

    # Pareto legend (if enabled)
    pareto_legend = []
    if highlight_pareto:
        pareto_legend = [
            Line2D([0], [0], marker='o', color='w',
                   markerfacecolor='white', markersize=10, label='Pareto Optimal',
                   markeredgecolor='#FFD700', markeredgewidth=2.5)
        ]

    # Create two separate legend boxes at bottom center (no titles for cleaner look)
    # First legend: Bug Category (left box)
    legend1 = ax.legend(handles=category_legend,
                       loc='upper center', bbox_to_anchor=(0.35, -0.08),
                       fontsize=10, framealpha=0.95, ncol=1)
    ax.add_artist(legend1)  # Add first legend to axes

    # Second legend: Configuration (right box)
    all_config_handles = config_legend + pareto_legend
    ax.legend(handles=all_config_handles,
             loc='upper center', bbox_to_anchor=(0.65, -0.08),
             fontsize=10, framealpha=0.95, ncol=1)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Single-panel combined plot saved to: {output_path}")


def create_multi_panel_combined(all_results: Dict[str, List[Dict]], output_path: Path,
                                 highlight_pareto: bool = True, cost_metric: str = 'total'):
    """
    Create a multi-panel plot (2x2 grid) with separate subplot for each category.

    Uses consistent style with individual category plots:
    - Category-specific colors
    - Different marker shapes for config counts
    - Captions at bottom-center of each subplot
    - Shared legend at bottom-center of figure
    """
    num_categories = len(all_results)

    # Category colors (consistent with single-panel and individual plots)
    category_colors = {
        'single_line': '#1f77b4',      # Blue
        'single_hunk': '#2ca02c',      # Green
        'single_file_multi_hunk': '#000000',  # Black
        'multi_file_multi_hunk': '#d62728',   # Red
    }

    # Marker shapes for number of configs
    config_markers = {
        1: 'o',  # Circle
        2: 's',  # Square
        3: '^',  # Triangle
        4: 'D'   # Diamond
    }

    # Category labels for captions
    category_labels = {
        'single_line': 'Single-Line',
        'single_hunk': 'Single-Hunk',
        'single_file_multi_hunk': 'Single-File-Multi-Hunk',
        'multi_file_multi_hunk': 'Multi-File-Multi-Hunk'
    }

    # Determine subplot layout
    if num_categories <= 2:
        nrows, ncols = 1, num_categories
        figsize = (10 * num_categories, 7)
    else:
        nrows = (num_categories + 1) // 2
        ncols = 2
        figsize = (16, 7 * nrows)

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)

    # Flatten axes for easier iteration
    if num_categories == 1:
        axes = [axes]
    elif num_categories > 1:
        axes = axes.flatten()

    for idx, (category, results) in enumerate(all_results.items()):
        ax = axes[idx]

        df = pd.DataFrame(results)
        pareto_mask = compute_pareto_frontier(results, cost_metric)
        df['is_pareto'] = pareto_mask

        # Get category color
        category_color = category_colors.get(category, '#1f77b4')

        # Plot all points with consistent category color and different marker shapes
        for _, row in df.iterrows():
            marker = config_markers[row['num_configs']]
            cost_value = row['total_cost'] if cost_metric == 'total' else row['avg_cost_per_bug']

            # Determine size and edge based on Pareto status
            if highlight_pareto and row['is_pareto']:
                size = 120
                edgecolor = '#FFD700'  # Gold
                linewidth = 2
                alpha = 0.9
                zorder = 10
            else:
                size = 80
                edgecolor = 'black'
                linewidth = 0.8
                alpha = 0.7
                zorder = 5

            ax.scatter(cost_value, row['success_rate'],
                      color=category_color, s=size, alpha=alpha,
                      marker=marker, edgecolors=edgecolor, linewidth=linewidth,
                      zorder=zorder)

        ax.set_xlabel('Total Cost (USD)' if cost_metric == 'total' else 'Average Cost per Bug (USD)', fontsize=11)
        ax.set_ylabel('Pass Rate (%)', fontsize=11)

        # Add caption at bottom-center instead of title
        # Use subplot letter (a, b, c, d)
        subplot_letter = chr(97 + idx)  # 97 is ASCII for 'a'
        caption = f"({subplot_letter}) {category_labels.get(category, category.replace('_', ' ').title())}"
        ax.text(0.5, -0.15, caption, transform=ax.transAxes,
               ha='center', va='top', fontsize=11)

        ax.grid(True, alpha=0.3)
        ax.set_ylim(-5, 105)

    # Hide unused subplots
    for idx in range(num_categories, len(axes)):
        axes[idx].axis('off')

    # Add shared legend at bottom-center of the figure
    from matplotlib.lines import Line2D

    config_labels = {
        1: '1 configuration',
        2: '2 configurations',
        3: '3 configurations',
        4: '4 configurations'
    }

    legend_elements = [
        Line2D([0], [0], marker=config_markers[i], color='w',
               markerfacecolor='white', markersize=9, label=config_labels[i],
               markeredgecolor='black', markeredgewidth=1.2)
        for i in sorted(config_markers.keys())
    ]

    if highlight_pareto:
        legend_elements.append(
            Line2D([0], [0], marker='o', color='w',
                   markerfacecolor='white', markersize=10, label='Pareto Optimal',
                   markeredgecolor='#FFD700', markeredgewidth=2.5)
        )

    fig.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, -0.02),
              ncol=5, fontsize=11, framealpha=0.95)

    plt.tight_layout(rect=[0, 0.05, 1, 1])  # Leave space at bottom for legend
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Multi-panel combined plot saved to: {output_path}")


def print_summary_stats(results: List[Dict], category: str):
    """Print summary statistics for the analysis."""
    df = pd.DataFrame(results)

    print(f"\n{'='*80}")
    print(f"COST-EFFECTIVENESS TRADE-OFF ANALYSIS: {category.upper()}")
    print(f"{'='*80}")

    # Overall stats
    print(f"\nTotal combinations analyzed: {len(results)}")
    print(f"  - 1 config:  {len(df[df['num_configs'] == 1])} combinations")
    print(f"  - 2 configs: {len(df[df['num_configs'] == 2])} combinations")
    print(f"  - 3 configs: {len(df[df['num_configs'] == 3])} combinations")
    print(f"  - 4 configs: {len(df[df['num_configs'] == 4])} combinations")

    # Best performers
    print(f"\n📊 TOP PERFORMERS:")

    # Highest success rate
    best_success = df.loc[df['success_rate'].idxmax()]
    print(f"  Highest success rate: {best_success['combination']}")
    print(f"    Success: {best_success['success_rate']:.1f}% ({best_success['num_bugs_fixed']}/{best_success['total_bugs']})")
    print(f"    Cost: ${best_success['total_cost']:.2f} (${best_success['avg_cost_per_bug']:.4f}/bug)")

    # Best cost-effectiveness (highest success per dollar)
    df['success_per_dollar'] = df['success_rate'] / df['total_cost']
    best_efficiency = df.loc[df['success_per_dollar'].idxmax()]
    print(f"\n  Best cost-effectiveness: {best_efficiency['combination']}")
    print(f"    Success: {best_efficiency['success_rate']:.1f}% ({best_efficiency['num_bugs_fixed']}/{best_efficiency['total_bugs']})")
    print(f"    Cost: ${best_efficiency['total_cost']:.2f} (${best_efficiency['avg_cost_per_bug']:.4f}/bug)")
    print(f"    Success per $: {best_efficiency['success_per_dollar']:.2f}%/$")

    # Pareto frontier
    pareto_mask = compute_pareto_frontier(results)
    pareto_points = df[pareto_mask]
    print(f"\n🎯 PARETO FRONTIER:")
    print(f"  {len(pareto_points)} optimal configurations:")
    for _, point in pareto_points.sort_values('total_cost').iterrows():
        print(f"    {point['combination']:30s} - {point['success_rate']:5.1f}% @ ${point['total_cost']:7.2f}")


# Create typer app
app = typer.Typer(rich_markup_mode="rich", add_completion=False)
console = Console()


@app.command(help="Analyze cost-effectiveness trade-offs across configuration combinations")
def main(
    bug_category: str = typer.Option(..., "--bug-category", "-c",
                                    help="Bug category: 'single_line', 'single_hunk', 'single_file_multi_hunk', 'multi_file_multi_hunk', 'multi_hunk', or 'all'"),
    input_dir: str = typer.Option("results/rq3_analysis", "--input", "-i",
                                  help="Input directory containing rq3a_metrics_*.csv files"),
    output_dir: str = typer.Option("results/rq3_analysis", "--output", "-o",
                                   help="Output directory for plots and CSV files"),
    viz_style: str = typer.Option("default", "--viz-style", "-v",
                                 help="Visualization style: 'default' (color by num configs), 'simplified' (single configs + Pareto, best for papers), or 'individual' (15 unique markers)"),
    highlight_pareto: bool = typer.Option(True, "--highlight-pareto/--no-pareto",
                                         help="Highlight Pareto frontier points (default: True)"),
    single_panel: bool = typer.Option(False, "--single-panel",
                                     help="Create single-panel combined plot instead of multi-panel (only for 'all' category)"),
    cost_metric: str = typer.Option('total', "--cost-metric", "-m",
                                   help="Cost metric for x-axis: 'total' (sum of costs, default) or 'avg' (average cost per bug)")
) -> None:

    console.print("[bold blue]Cost-Effectiveness Trade-off Analysis[/bold blue]", style="bold")

    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Validate viz_style
    if viz_style not in ['default', 'simplified', 'individual']:
        console.print(f"[red]Error: Invalid viz-style '{viz_style}'. Must be 'default', 'simplified', or 'individual'[/red]")
        raise typer.Exit(1)

    if cost_metric not in ['total', 'avg']:
        console.print(f"[red]Error: Invalid cost metric '{cost_metric}'. Use 'total' or 'avg'.[/red]")
        raise typer.Exit(1)

    # Determine categories to analyze
    if bug_category == 'all':
        categories = ['single_line', 'single_hunk', 'single_file_multi_hunk', 'multi_file_multi_hunk']
    else:
        categories = [bug_category]

    all_results = {}

    for category in categories:
        console.print(f"\n[cyan]Analyzing {category}...[/cyan]")

        # Load data
        try:
            df = load_metrics_csv(input_path, category)
        except FileNotFoundError as e:
            console.print(f"[yellow]Skipping {category}: {e}[/yellow]")
            continue

        console.print(f"  Loaded {len(df)} records for {len(df['bug_id'].unique())} bugs")

        # Generate all combinations
        all_combos = generate_all_combinations(CONFIGS)
        console.print(f"  Analyzing {len(all_combos)} configuration combinations...")

        # Compute metrics for each combination
        results = []
        for combo in all_combos:
            metrics = compute_combination_metrics(df, combo)
            results.append(metrics)

        all_results[category] = results

        # Print summary statistics
        print_summary_stats(results, category)

        # Create scatter plot for individual categories
        suffix = '_avg' if cost_metric == 'avg' else ''
        plot_path = output_path / f'rq3_tradeoff_{category}{suffix}.pdf'
        create_scatter_plot(results, plot_path, category, viz_style, highlight_pareto, cost_metric)

        # Save detailed CSV
        csv_path = output_path / f'rq3_tradeoff_{category}_details.csv'
        save_detailed_csv(results, csv_path)

    # Create combined plot if analyzing multiple categories
    if len(all_results) > 1:
        suffix = '_avg' if cost_metric == 'avg' else ''
        if single_panel:
            combined_path = output_path / f'rq3_tradeoff_combined_single_panel{suffix}.pdf'
        else:
            combined_path = output_path / f'rq3_tradeoff_combined{suffix}.pdf'

        create_combined_plot(all_results, combined_path, single_panel, highlight_pareto, cost_metric)

    console.print("\n[green]✅ Analysis complete![/green]")


if __name__ == "__main__":
    app()
