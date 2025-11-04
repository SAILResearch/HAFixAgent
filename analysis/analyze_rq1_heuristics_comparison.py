#!/usr/bin/env python3
"""
HAFixAgent History Heuristics Comparison

This script compares the bug fixing performance between different history heuristics
(baseline, fn_all, etc.) on specified bug categories, visualizing results with Venn diagrams.
Supports 2-way, 3-way, and 4-way comparisons.
"""

import json
import matplotlib.pyplot as plt
from matplotlib_venn import venn2, venn3
from pathlib import Path
from typing import Dict
import argparse

from utils import (
    filter_to_common_bugs,
    find_common_bugs_with_hunk4j,
    get_history_name,
    get_history_flag,
    load_hunk4j_data,
    load_progress_data,
    resolve_config_label,
)

# For 4-set Venn diagrams
try:
    import venn
    HAS_VENN = True
except ImportError:
    HAS_VENN = False
    print("Warning: 'venn' package not installed. 4-way comparisons will use table format.")
    print("Install with: pip install venn")

# Set matplotlib to non-interactive backend
plt.ioff()
import matplotlib
matplotlib.use('Agg')

BIRCH_BASELINE_LABEL = resolve_config_label('baseline_birch_fb', default='BIRCH-feedback')

BASELINE_FLAG = get_history_flag('baseline')
FN_ALL_FLAG = get_history_flag('fn_all')
FN_PAIR_FLAG = get_history_flag('fn_pair')
FL_DIFF_FLAG = get_history_flag('fl_diff')

def create_3way_common_bugs_comparison(merged_data1: Dict, merged_data2: Dict, hunk4j_data: Dict,
                                     heuristic1: str, heuristic2: str, bug_category: str,
                                     output_dir: Path, results_base_dir: Path,
                                     selector_type: str = "first", n_lines: int = 1) -> Dict:
    """
    Create 3-way comparison on common bugs only.

    Args:
        merged_data1: Aggregated results for the first heuristic
        merged_data2: Aggregated results for the second heuristic
        hunk4j_data: HUNK4J baseline data (already loaded)
        heuristic1: History flag for the first heuristic
        heuristic2: History flag for the second heuristic
        bug_category: Category label used for filenames and logging
        output_dir: Directory where figures should be written
        results_base_dir: Root path for HAFixAgent evaluation outputs
        selector_type: Line selection strategy (matches results directory naming)
        n_lines: Number of selected lines (matches results directory naming)
    """
    print(f"\n{'='*60}")
    print("FINDING COMMON BUGS FOR FAIR COMPARISON")
    print(f"{'='*60}")

    # Find common bugs
    common_bugs = find_common_bugs_with_hunk4j(results_base_dir, selector_type, n_lines)

    if len(common_bugs) == 0:
        print("No common bugs found! Skipping common bugs comparison.")
        return None

    print(f"Found {len(common_bugs)} common bugs in shared evaluation set.")
    print(f"\nFiltering data to common bugs only...")

    # Filter all datasets to common bugs
    common_data1 = filter_to_common_bugs(merged_data1, common_bugs)
    common_data2 = filter_to_common_bugs(merged_data2, common_bugs)
    common_hunk4j = filter_to_common_bugs(hunk4j_data, common_bugs)

    print(f"Common bugs filtered results:")
    print(f"  {get_history_name(heuristic1)}: {len(common_data1['successful_bugs'])}/{common_data1['total_bugs']} ({common_data1['success_rate']:.1f}%)")
    print(f"  {get_history_name(heuristic2)}: {len(common_data2['successful_bugs'])}/{common_data2['total_bugs']} ({common_data2['success_rate']:.1f}%)")
    print(f"  HUNK4J: {len(common_hunk4j['successful_bugs'])}/{common_hunk4j['total_bugs']} ({common_hunk4j['success_rate']:.1f}%)")

    # Create comparison on common bugs
    output_filename = f"rq1_heuristic_{heuristic1}_vs_{heuristic2}_vs_hunk4j_{bug_category}_common_bugs.pdf"
    output_path = output_dir / output_filename

    print(f"\nCreating 3-way Venn diagram for common bugs...")
    comparison_stats = create_3way_comparison_venn(
        common_data1, common_data2, common_hunk4j, heuristic1, heuristic2, f"{'multi_hunk bugs' if bug_category == 'multi_hunk' else bug_category} (Common Bugs)", str(output_path)
    )

    # Print analysis for common bugs
    print_3way_comparison_analysis(common_data1, common_data2, common_hunk4j, heuristic1, heuristic2, f"{bug_category} (Common Bugs)", comparison_stats)

    print(f"\nCommon bugs Venn diagram saved to: {output_path}")

    return {
        'common_bugs_count': len(common_bugs),
        'heuristic1': {
            'flag': heuristic1,
            'name': get_history_name(heuristic1),
            'total_bugs': common_data1['total_bugs'],
            'successful': len(common_data1['successful_bugs']),
            'success_rate': common_data1['success_rate']
        },
        'heuristic2': {
            'flag': heuristic2,
            'name': get_history_name(heuristic2),
            'total_bugs': common_data2['total_bugs'],
            'successful': len(common_data2['successful_bugs']),
            'success_rate': common_data2['success_rate']
        },
        'hunk4j': {
            'total_bugs': common_hunk4j['total_bugs'],
            'successful': len(common_hunk4j['successful_bugs']),
            'success_rate': common_hunk4j['success_rate']
        },
        'comparison': comparison_stats,
        'output_file': output_filename
    }

def create_comparison_venn(data1: Dict, data2: Dict, heuristic1: str, heuristic2: str,
                          bug_category: str, output_path: str) -> Dict:
    """Create Venn diagram comparing two history heuristics."""

    successful1 = data1['successful_bugs']
    successful2 = data2['successful_bugs']

    plt.figure(figsize=(12, 8))

    # Create Venn diagram (no title - will be added in Overleaf)
    venn_diagram = venn2(
        [successful1, successful2],
        set_labels=(get_history_name(heuristic1), get_history_name(heuristic2))
    )

    # plt.title(f'Performance Comparison on {bug_category}\n'
    #           f'{get_history_name(heuristic1)} vs {get_history_name(heuristic2)}',
    #           fontsize=14, fontweight='bold')

    # Calculate statistics
    only_1 = len(successful1 - successful2)
    only_2 = len(successful2 - successful1)
    both = len(successful1 & successful2)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    return {
        'heuristic1_only': only_1,
        'heuristic2_only': only_2,
        'both': both,
        'heuristic1_total': len(successful1),
        'heuristic2_total': len(successful2)
    }

def create_3way_comparison_venn(data1: Dict, data2: Dict, hunk4j_data: Dict,
                               heuristic1: str, heuristic2: str, bug_category: str,
                               output_path: str) -> Dict:
    """Create 3-set Venn diagram comparing two heuristics and HUNK4J."""

    successful1 = data1['successful_bugs']
    successful2 = data2['successful_bugs']
    successful_hunk4j = hunk4j_data['successful_bugs']

    # Create 3-way Venn diagram with publication-quality settings
    fig, ax = plt.subplots(figsize=(10, 8))

    venn_diagram = venn3(
        [successful1, successful2, successful_hunk4j],
        set_labels=(get_history_name(heuristic1), get_history_name(heuristic2), BIRCH_BASELINE_LABEL),
        ax=ax
    )

    # Increase font sizes for publication quality
    for text in venn_diagram.set_labels:
        if text:
            text.set_fontsize(18)

    for text in venn_diagram.subset_labels:
        if text:
            text.set_fontsize(16)

    # Calculate statistics for all 7 regions of 3-set Venn diagram
    # Using set operations to determine each region
    only_1 = len(successful1 - successful2 - successful_hunk4j)
    only_2 = len(successful2 - successful1 - successful_hunk4j)
    only_hunk4j = len(successful_hunk4j - successful1 - successful2)
    only_1_and_2 = len((successful1 & successful2) - successful_hunk4j)
    only_1_and_hunk4j = len((successful1 & successful_hunk4j) - successful2)
    only_2_and_hunk4j = len((successful2 & successful_hunk4j) - successful1)
    all_three = len(successful1 & successful2 & successful_hunk4j)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    return {
        'heuristic1_only': only_1,
        'heuristic2_only': only_2,
        'hunk4j_only': only_hunk4j,
        'heuristic1_and_heuristic2_only': only_1_and_2,
        'heuristic1_and_hunk4j_only': only_1_and_hunk4j,
        'heuristic2_and_hunk4j_only': only_2_and_hunk4j,
        'all_three': all_three,
        'heuristic1_total': len(successful1),
        'heuristic2_total': len(successful2),
        'hunk4j_total': len(successful_hunk4j)
    }

def create_2x2_grid_venn(data_by_category: Dict[str, list],
                         h1: str, h2: str, h3: str, h4: str,
                         output_path: str) -> None:
    """
    Create a 2x2 grid of 4-set Venn diagrams for different bug categories.

    Args:
        data_by_category: Dictionary mapping category names to list of 4 data dicts
                         e.g., {'single_line': [data1, data2, data3, data4], ...}
        h1, h2, h3, h4: History flags for the four heuristics
        output_path: Path to save the combined figure
    """
    if not HAS_VENN:
        print("Warning: 'venn' package not installed. Cannot create 2x2 grid Venn diagram.")
        return

    # Define the 4 categories and their display order
    categories = [
        ('single_line', 'SL', 'a'),
        ('single_hunk', 'SH', 'b'),
        ('single_file_multi_hunk', 'SFMH', 'c'),
        ('multi_file_multi_hunk', 'MFMH', 'd')
    ]

    # Create 2x2 subplot figure with minimal spacing
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    axes = axes.flatten()

    # Store legend handles for shared legend
    legend_labels = [
        get_history_name(h1),
        get_history_name(h2),
        get_history_name(h3),
        get_history_name(h4)
    ]

    # Define colors matching the venn package's actual colormap (viridis)
    # The venn package uses viridis colormap for 4-set diagrams
    venn_colors = [plt.cm.viridis(i/3) for i in range(4)]

    # Create each subfigure
    for idx, (cat_key, cat_display, label) in enumerate(categories):
        if cat_key not in data_by_category:
            print(f"Warning: {cat_key} not found in data")
            continue

        ax = axes[idx]
        data_list = data_by_category[cat_key]

        # Extract successful bug sets
        s1 = data_list[0]['successful_bugs']
        s2 = data_list[1]['successful_bugs']
        s3 = data_list[2]['successful_bugs']
        s4 = data_list[3]['successful_bugs']

        # Create labels dictionary with set names
        labels_dict = {
            get_history_name(h1): s1,
            get_history_name(h2): s2,
            get_history_name(h3): s3,
            get_history_name(h4): s4
        }

        # Create venn diagram without legend
        venn.venn(labels_dict, ax=ax, fontsize=11, legend_loc=None)

        # Tighten vertical limits to reduce unused headroom (figures default to -0.05..1.05)
        ymin, ymax = ax.get_ylim()
        ax.set_ylim(ymin, ymin + (ymax - ymin) * 0.85)

        # Add caption at bottom middle (no bold text)
        caption_text = f'({label}) {cat_display}'
        ax.text(0.5, 0.02, caption_text, transform=ax.transAxes,
               fontsize=14, va='bottom', ha='center')

    # Create a single shared legend at the bottom center with matching colors
    from matplotlib.patches import Patch
    legend_patches = [Patch(facecolor=venn_colors[i], alpha=0.4, label=label)
                     for i, label in enumerate(legend_labels)]

    # Add legend below all subplots
    fig.legend(handles=legend_patches, loc='lower center',
              bbox_to_anchor=(0.5, -0.035), ncol=2, fontsize=13, frameon=True)

    # Adjust layout to minimize whitespace (especially top and middle)
    # hspace controls vertical spacing between rows, reduced significantly
    # top increased to use more of the figure area
    plt.subplots_adjust(left=0.05, right=0.95, top=0.7, bottom=0.05,
                       hspace=0, wspace=-0.35)

    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"2x2 grid Venn diagram saved to: {output_path}")

def create_4way_comparison_venn(data1: Dict, data2: Dict, data3: Dict, data4: Dict,
                               h1: str, h2: str, h3: str, h4: str,
                               bug_category: str, output_path: str) -> Dict:
    """Create 4-set Venn diagram comparing four heuristics."""

    s1 = data1['successful_bugs']
    s2 = data2['successful_bugs']
    s3 = data3['successful_bugs']
    s4 = data4['successful_bugs']

    if HAS_VENN:
        # Use venn package for 4-set diagram with publication-quality settings
        fig, ax = plt.subplots(figsize=(10, 8))

        # Create labels dictionary with set names
        labels = {
            get_history_name(h1): s1,
            get_history_name(h2): s2,
            get_history_name(h3): s3,
            get_history_name(h4): s4
        }

        # Create venn diagram with legend
        venn.venn(labels, ax=ax, fontsize=14)

        # Increase font sizes for all text elements
        for text in ax.texts:
            text.set_fontsize(14)

        # Manually reposition the legend to bottom center outside the plot area
        legend = ax.get_legend()
        if legend:
            for text in legend.get_texts():
                text.set_fontsize(14)
            # Position legend at bottom center, outside the axes
            legend.set_bbox_to_anchor((0.5, 0.08))
            legend.set_loc('upper center')
            legend.set_ncols(2)  # Use 2 columns to make it more compact
            legend.set_frame_on(True)  # Ensure frame is visible

        # Reduce whitespace inside the figure by adjusting axis limits
        ylim = ax.get_ylim()
        ax.set_ylim(ylim[0], ylim[1] * 0.85)  # Reduce top whitespace by 15%

        # Adjust layout to minimize whitespace
        plt.subplots_adjust(top=0.95, bottom=0.12, left=0.05, right=0.95)
        # Use bbox_inches='tight' to include the legend that's outside the axes
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        # Fallback: create table visualization
        fig, ax = plt.subplots(figsize=(14, 10))
        ax.axis('tight')
        ax.axis('off')

        # Calculate key statistics
        all_four = s1 & s2 & s3 & s4
        only_1 = s1 - s2 - s3 - s4
        only_2 = s2 - s1 - s3 - s4
        only_3 = s3 - s1 - s2 - s4
        only_4 = s4 - s1 - s2 - s3

        table_data = [
            ['Method', 'Total Fixed', 'Unique Fixes'],
            [get_history_name(h1), len(s1), len(only_1)],
            [get_history_name(h2), len(s2), len(only_2)],
            [get_history_name(h3), len(s3), len(only_3)],
            [get_history_name(h4), len(s4), len(only_4)],
            ['', '', ''],
            ['All Four', len(all_four), '-'],
        ]

        table = ax.table(cellText=table_data, cellLoc='center', loc='center',
                        colWidths=[0.4, 0.3, 0.3])
        table.auto_set_font_size(False)
        table.set_fontsize(12)
        table.scale(1, 2)

        for i in range(len(table_data[0])):
            table[(0, i)].set_facecolor('#4CAF50')
            table[(0, i)].set_text_props(weight='bold', color='white')

        # plt.title(f'4-Way Comparison on {bug_category}\n'
        #          f'{get_history_name(h1)} vs {get_history_name(h2)} vs {get_history_name(h3)} vs {get_history_name(h4)}',
        #          fontsize=14, fontweight='bold', pad=20)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

    # Calculate all statistics for 4-set Venn (15 non-empty regions)
    only_1 = s1 - s2 - s3 - s4
    only_2 = s2 - s1 - s3 - s4
    only_3 = s3 - s1 - s2 - s4
    only_4 = s4 - s1 - s2 - s3

    only_1_2 = (s1 & s2) - s3 - s4
    only_1_3 = (s1 & s3) - s2 - s4
    only_1_4 = (s1 & s4) - s2 - s3
    only_2_3 = (s2 & s3) - s1 - s4
    only_2_4 = (s2 & s4) - s1 - s3
    only_3_4 = (s3 & s4) - s1 - s2

    only_1_2_3 = (s1 & s2 & s3) - s4
    only_1_2_4 = (s1 & s2 & s4) - s3
    only_1_3_4 = (s1 & s3 & s4) - s2
    only_2_3_4 = (s2 & s3 & s4) - s1

    all_four = s1 & s2 & s3 & s4

    return {
        'h1_only': len(only_1),
        'h2_only': len(only_2),
        'h3_only': len(only_3),
        'h4_only': len(only_4),
        'h1_h2_only': len(only_1_2),
        'h1_h3_only': len(only_1_3),
        'h1_h4_only': len(only_1_4),
        'h2_h3_only': len(only_2_3),
        'h2_h4_only': len(only_2_4),
        'h3_h4_only': len(only_3_4),
        'h1_h2_h3_only': len(only_1_2_3),
        'h1_h2_h4_only': len(only_1_2_4),
        'h1_h3_h4_only': len(only_1_3_4),
        'h2_h3_h4_only': len(only_2_3_4),
        'all_four': len(all_four),
        'h1_total': len(s1),
        'h2_total': len(s2),
        'h3_total': len(s3),
        'h4_total': len(s4)
    }

def print_4way_comparison_analysis(data1: Dict, data2: Dict, data3: Dict, data4: Dict,
                                   h1: str, h2: str, h3: str, h4: str,
                                   bug_category: str, comparison_stats: Dict):
    """Print detailed 4-way comparison analysis."""

    n1, n2, n3, n4 = get_history_name(h1), get_history_name(h2), get_history_name(h3), get_history_name(h4)
    category_display = bug_category.replace('_', ' ').title()

    print("\n" + "="*80)
    print(f"4-WAY COMPARISON: {n1} vs {n2} vs {n3} vs {n4}")
    print(f"Bug Category: {category_display}")
    print("="*80)

    print(f"\nOverall Performance:")
    print(f"  {n1}: {data1['total_bugs']} bugs, {len(data1['successful_bugs'])} successful ({data1['success_rate']:.1f}%)")
    print(f"  {n2}: {data2['total_bugs']} bugs, {len(data2['successful_bugs'])} successful ({data2['success_rate']:.1f}%)")
    print(f"  {n3}: {data3['total_bugs']} bugs, {len(data3['successful_bugs'])} successful ({data3['success_rate']:.1f}%)")
    print(f"  {n4}: {data4['total_bugs']} bugs, {len(data4['successful_bugs'])} successful ({data4['success_rate']:.1f}%)")

    print(f"\nUnique Contributions:")
    print(f"  {n1} only: {comparison_stats['h1_only']}")
    print(f"  {n2} only: {comparison_stats['h2_only']}")
    print(f"  {n3} only: {comparison_stats['h3_only']}")
    print(f"  {n4} only: {comparison_stats['h4_only']}")

    print(f"\nFour-way Overlap:")
    print(f"  All four methods: {comparison_stats['all_four']}")

    # Calculate total unique fixes
    s1, s2, s3, s4 = data1['successful_bugs'], data2['successful_bugs'], data3['successful_bugs'], data4['successful_bugs']
    total_unique = len(s1 | s2 | s3 | s4)
    max_bugs = max(data1['total_bugs'], data2['total_bugs'], data3['total_bugs'], data4['total_bugs'])
    combined_rate = total_unique / max_bugs * 100 if max_bugs > 0 else 0

    print(f"\nCombined Impact:")
    print(f"  Total bugs fixed by any approach: {total_unique}")
    print(f"  Combined success rate: {combined_rate:.1f}%")

    # Performance ranking
    performances = [
        (n1, data1['success_rate']),
        (n2, data2['success_rate']),
        (n3, data3['success_rate']),
        (n4, data4['success_rate'])
    ]
    performances.sort(key=lambda x: x[1], reverse=True)

    print(f"\nPerformance Ranking:")
    for i, (method, rate) in enumerate(performances, 1):
        print(f"  {i}. {method}: {rate:.1f}%")

def print_comparison_analysis(data1: Dict, data2: Dict, heuristic1: str, heuristic2: str,
                            bug_category: str, comparison_stats: Dict):
    """Print detailed comparison analysis."""

    name1 = get_history_name(heuristic1)
    name2 = get_history_name(heuristic2)
    category_display = bug_category.replace('_', ' ').title()

    print("\n" + "="*70)
    print(f"HISTORY HEURISTIC COMPARISON: {name1} vs {name2}")
    print(f"Bug Category: {category_display}")
    print("="*70)

    print(f"\nOverall Performance:")
    print(f"  {name1}: {data1['total_bugs']} bugs, {len(data1['successful_bugs'])} successful ({data1['success_rate']:.1f}%)")
    print(f"  {name2}: {data2['total_bugs']} bugs, {len(data2['successful_bugs'])} successful ({data2['success_rate']:.1f}%)")

    # Check if datasets are comparable
    if data1['total_bugs'] != data2['total_bugs']:
        print(f"\n⚠️  WARNING: Different dataset sizes! This may indicate different bug sets were evaluated.")

    print(f"\nSuccess Breakdown:")
    print(f"  {name1} unique fixes: {comparison_stats['heuristic1_only']}")
    print(f"  {name2} unique fixes: {comparison_stats['heuristic2_only']}")
    print(f"  Shared successful fixes: {comparison_stats['both']}")

    # Calculate combined impact
    total_unique_fixes = comparison_stats['heuristic1_total'] + comparison_stats['heuristic2_total'] - comparison_stats['both']
    max_bugs = max(data1['total_bugs'], data2['total_bugs'])
    combined_rate = total_unique_fixes / max_bugs * 100 if max_bugs > 0 else 0

    print(f"\nCombined Impact:")
    print(f"  Total bugs fixed by either approach: {total_unique_fixes}")
    print(f"  Combined success rate: {combined_rate:.1f}%")

    # Performance comparison
    if data1['success_rate'] > data2['success_rate']:
        better = name1
        diff = data1['success_rate'] - data2['success_rate']
    elif data2['success_rate'] > data1['success_rate']:
        better = name2
        diff = data2['success_rate'] - data1['success_rate']
    else:
        better = "Neither (tied)"
        diff = 0

    if diff > 0:
        print(f"\nPerformance Summary:")
        print(f"  {better} performs better by {diff:.1f} percentage points")

def print_3way_comparison_analysis(data1: Dict, data2: Dict, hunk4j_data: Dict,
                                  heuristic1: str, heuristic2: str, bug_category: str,
                                  comparison_stats: Dict):
    """Print detailed 3-way comparison analysis."""

    name1 = get_history_name(heuristic1)
    name2 = get_history_name(heuristic2)
    category_display = bug_category.replace('_', ' ').title()

    print("\n" + "="*80)
    print(f"3-WAY COMPARISON: {name1} vs {name2} vs HUNK4J")
    print(f"Bug Category: {category_display}")
    print("="*80)

    print(f"\nOverall Performance:")
    print(f"  {name1}: {data1['total_bugs']} bugs, {len(data1['successful_bugs'])} successful ({data1['success_rate']:.1f}%)")
    print(f"  {name2}: {data2['total_bugs']} bugs, {len(data2['successful_bugs'])} successful ({data2['success_rate']:.1f}%)")
    print(f"  HUNK4J: {hunk4j_data['total_bugs']} bugs, {len(hunk4j_data['successful_bugs'])} successful ({hunk4j_data['success_rate']:.1f}%)")

    print(f"\nUnique Contributions:")
    print(f"  {name1} only: {comparison_stats['heuristic1_only']}")
    print(f"  {name2} only: {comparison_stats['heuristic2_only']}")
    print(f"  HUNK4J only: {comparison_stats['hunk4j_only']}")

    print(f"\nPairwise Overlaps (excluding other):")
    print(f"  {name1} & {name2} only: {comparison_stats['heuristic1_and_heuristic2_only']}")
    print(f"  {name1} & HUNK4J only: {comparison_stats['heuristic1_and_hunk4j_only']}")
    print(f"  {name2} & HUNK4J only: {comparison_stats['heuristic2_and_hunk4j_only']}")

    print(f"\nThree-way Overlap:")
    print(f"  All three methods: {comparison_stats['all_three']}")

    # Calculate combined impact
    total_unique_fixes = (comparison_stats['heuristic1_total'] +
                         comparison_stats['heuristic2_total'] +
                         comparison_stats['hunk4j_total'] -
                         comparison_stats['heuristic1_and_heuristic2_only'] -
                         comparison_stats['heuristic1_and_hunk4j_only'] -
                         comparison_stats['heuristic2_and_hunk4j_only'] -
                         2 * comparison_stats['all_three'])

    max_bugs = max(data1['total_bugs'], data2['total_bugs'], hunk4j_data['total_bugs'])
    combined_rate = total_unique_fixes / max_bugs * 100 if max_bugs > 0 else 0

    print(f"\nCombined Impact:")
    print(f"  Total bugs fixed by any approach: {total_unique_fixes}")
    print(f"  Combined success rate: {combined_rate:.1f}%")

    # Performance ranking
    performances = [
        (name1, data1['success_rate']),
        (name2, data2['success_rate']),
        ('HUNK4J', hunk4j_data['success_rate'])
    ]
    performances.sort(key=lambda x: x[1], reverse=True)

    print(f"\nPerformance Ranking:")
    for i, (method, rate) in enumerate(performances, 1):
        print(f"  {i}. {method}: {rate:.1f}%")

def create_bar_chart_comparison(data_dict: Dict[str, Dict], bug_category: str,
                                output_path: str) -> None:
    """
    Create grouped bar chart comparing heuristics with SFMH/MFMH breakdown.

    Args:
        data_dict: Dictionary mapping method names to their data
            e.g., {'baseline': {'sfmh_successful': N1, 'mfmh_successful': N2}, ...}
        bug_category: Bug category name for labeling
        output_path: Path to save the figure
    """
    import numpy as np

    # Extract data for plotting
    methods = list(data_dict.keys())
    sfmh_counts = [data_dict[m].get('sfmh_successful', 0) for m in methods]
    mfmh_counts = [data_dict[m].get('mfmh_successful', 0) for m in methods]

    # Set up the bar chart
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(methods))
    width = 0.35

    # Create bars using shared palette
    bars1 = ax.bar(x - width/2, sfmh_counts, width, label='SFMH',
                   color='#4A90E2', alpha=0.9)
    bars2 = ax.bar(x + width/2, mfmh_counts, width, label='MFMH',
                   color='#E67E22', alpha=0.9)

    # Customize the plot
    ax.set_xlabel('Agent Configuration', fontsize=13, labelpad=14)
    ax.set_ylabel('Number of Bugs Fixed', fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels(methods, fontsize=11)
    ax.tick_params(axis='x', pad=8)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), fontsize=11, framealpha=False, ncol=2)
    ax.grid(True, alpha=0.3, axis='y')

    # Add value labels on bars
    def autolabel(bars):
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.annotate(f'{int(height)}',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3),  # 3 points vertical offset
                           textcoords="offset points",
                           ha='center', va='bottom', fontsize=10)

    autolabel(bars1)
    autolabel(bars2)

    # Add a dashed boundary before HUNK4J to indicate external baseline
    if BIRCH_BASELINE_LABEL in methods:
        birch_idx = methods.index(BIRCH_BASELINE_LABEL)
        ax.axvline(birch_idx - 0.5, color='gray', linestyle='--', linewidth=1.2, alpha=0.7)

    # Tight layout with a small bottom margin so legend remains visible
    plt.tight_layout(rect=[0, 0.08, 1, 1])
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Bar chart saved to: {output_path}")

def create_simple_bar_chart_comparison(data_dict: Dict[str, Dict], bug_category: str,
                                       output_path: str) -> None:
    """
    Create simple 2-way grouped bar chart comparing Minimal-agent vs BIRCH-feedback.

    Args:
        data_dict: Dictionary mapping method names to their data
            e.g., {'Minimal-agent': {'sfmh_successful': N1, 'mfmh_successful': N2}, ...}
        bug_category: Bug category name for labeling
        output_path: Path to save the figure
    """
    import numpy as np

    # Extract data for plotting (should only have 2 methods)
    methods = list(data_dict.keys())
    sfmh_counts = [data_dict[m].get('sfmh_successful', 0) for m in methods]
    mfmh_counts = [data_dict[m].get('mfmh_successful', 0) for m in methods]

    # Set up the bar chart with tighter spacing
    fig, ax = plt.subplots(figsize=(6, 6))
    # Use narrower spacing between the two groups
    x = np.array([0, 1.2])
    width = 0.25

    # Create bars using shared palette
    bars1 = ax.bar(x - width/2, sfmh_counts, width, label='SFMH',
                   color='#4A90E2', alpha=0.9)
    bars2 = ax.bar(x + width/2, mfmh_counts, width, label='MFMH',
                   color='#E67E22', alpha=0.9)

    # Customize the plot
    ax.set_ylabel('Number of Bugs Fixed', fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels(methods, fontsize=11)
    ax.tick_params(axis='x', pad=8)
    # Move legend closer to x-axis by adjusting bbox_to_anchor
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.08), fontsize=11, framealpha=False, ncol=2)
    ax.grid(True, alpha=0.3, axis='y')

    # Set x-axis limits to reduce white space around bars
    ax.set_xlim(-0.4, x[-1] + 0.4)

    # Add value labels on bars
    def autolabel(bars):
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.annotate(f'{int(height)}',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3),  # 3 points vertical offset
                           textcoords="offset points",
                           ha='center', va='bottom', fontsize=10)

    autolabel(bars1)
    autolabel(bars2)

    # Tight layout with minimal bottom margin
    plt.tight_layout(rect=[0, 0.02, 1, 1])
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Simple bar chart saved to: {output_path}")

def main():
    parser = argparse.ArgumentParser(description='Compare HAFixAgent history heuristics performance (2-way or 4-way)')
    parser.add_argument('--heuristic1', '-h1', help='First history flag (e.g., 1 for baseline)')
    parser.add_argument('--heuristic2', '-h2', help='Second history flag (e.g., 5 for fn_all)')
    parser.add_argument('--heuristic3', '-h3', help='Third history flag (optional, for 4-way comparison)')
    parser.add_argument('--heuristic4', '-h4', help='Fourth history flag (optional, for 4-way comparison)')
    parser.add_argument('--bug-category', '-c', required=True,
                       choices=['single_line', 'single_hunk', 'single_file_multi_hunk', 'multi_file_multi_hunk', 'multi_hunk', 'all'],
                       help='Bug category to compare')
    parser.add_argument('--selector-type', '-s', default='llm_judge',
                       help='Line selection strategy: first, random, llm_judge (default: llm_judge)')
    parser.add_argument('--n-lines', '-n', type=int, default=1,
                       help='Number of lines selected for blame (default: 1)')
    parser.add_argument('--include-hunk4j', action='store_true',
                       help='Include HUNK4J baseline in comparison (for 2-way: creates 3-set Venn)')
    parser.add_argument('--bar-chart', action='store_true',
                       help='Generate bar chart comparison (with SFMH/MFMH breakdown)')
    parser.add_argument('--minimal-bar-chart', action='store_true',
                       help='Generate minimal 2-way bar chart (Minimal-agent vs BIRCH-feedback only)')
    parser.add_argument('--grid', action='store_true',
                       help='Generate 2x2 grid figure with all 4 bug categories (only for 4-way comparison with -c all)')
    parser.add_argument('--output', '-o', default='results/comparison',
                       help='Output directory for results')

    args = parser.parse_args()

    # Validate arguments based on mode
    bar_chart_only = args.bar_chart and args.include_hunk4j and not args.heuristic1 and not args.heuristic2
    minimal_bar_chart_only = args.minimal_bar_chart and args.include_hunk4j and not args.heuristic1 and not args.heuristic2

    if not bar_chart_only and not minimal_bar_chart_only:
        # Normal mode: require h1 and h2
        if not args.heuristic1 or not args.heuristic2:
            parser.error("--heuristic1/-h1 and --heuristic2/-h2 are required (unless using --bar-chart/--minimal-bar-chart with --include-hunk4j)")

    # If minimal-bar-chart-only mode, generate simple 2-way comparison
    if minimal_bar_chart_only:
        print(f"\n{'='*60}")
        print("MINIMAL BAR CHART MODE")
        print("Generating simple 2-way bar chart: Minimal-agent vs BIRCH-feedback")
        print(f"{'='*60}")

        base_dir = Path(__file__).parent.parent / "results" / "defects4j"
        selector_subdir = f"{args.selector_type}_{args.n_lines}line"
        output_dir = Path(args.output) / selector_subdir
        output_dir.mkdir(parents=True, exist_ok=True)

        if args.bug_category == 'multi_hunk':
            # Load HUNK4J data
            print(f"\nLoading HUNK4J data...")
            hunk4j_data = load_hunk4j_data(
                "multi_hunk",
                selector_type=args.selector_type,
                n_lines=args.n_lines,
                hafix_results_dir=base_dir,
            )

            # Find common bugs using shared utility
            common_bugs = find_common_bugs_with_hunk4j(base_dir, args.selector_type, args.n_lines)

            print(f"\nGenerating minimal bar chart for Minimal-agent vs BIRCH-feedback ({len(common_bugs)} common bugs)...")

            # Load baseline data
            sfmh_baseline = load_progress_data('single_file_multi_hunk', BASELINE_FLAG, base_dir, args.selector_type, args.n_lines)
            mfmh_baseline = load_progress_data('multi_file_multi_hunk', BASELINE_FLAG, base_dir, args.selector_type, args.n_lines)

            # Filter to common bugs
            sfmh_baseline_common = sfmh_baseline['successful_bugs'] & common_bugs
            mfmh_baseline_common = mfmh_baseline['successful_bugs'] & common_bugs

            # Load our bug categorization to split HUNK4J results
            sfmh_bugs_set = set()
            mfmh_bugs_set = set()

            for cat, bug_set in [('single_file_multi_hunk', sfmh_bugs_set), ('multi_file_multi_hunk', mfmh_bugs_set)]:
                cat_data = load_progress_data(cat, BASELINE_FLAG, base_dir, args.selector_type, args.n_lines)
                bug_set.update(cat_data.get('successful_bugs', set()))
                bug_set.update(cat_data.get('failed_bugs', set()))

            hunk4j_successful = hunk4j_data['successful_bugs'] & common_bugs
            hunk4j_sfmh = hunk4j_successful & sfmh_bugs_set
            hunk4j_mfmh = hunk4j_successful & mfmh_bugs_set

            # Create minimal data dict with renamed baseline
            minimal_bar_data_dict = {
                'Minimal-agent': {
                    'sfmh_successful': len(sfmh_baseline_common),
                    'mfmh_successful': len(mfmh_baseline_common)
                },
                BIRCH_BASELINE_LABEL: {
                    'sfmh_successful': len(hunk4j_sfmh),
                    'mfmh_successful': len(hunk4j_mfmh)
                }
            }

            minimal_bar_chart_filename = f"rq1_minimal_barchart_baseline_vs_hunk4j_multi_hunk_{len(common_bugs)}_common_bugs.pdf"
            minimal_bar_chart_path = output_dir / minimal_bar_chart_filename
            create_simple_bar_chart_comparison(minimal_bar_data_dict, f"multi_hunk ({len(common_bugs)} common bugs)", str(minimal_bar_chart_path))

            print(f"\n{'='*50}")
            print("SUMMARY")
            print(f"{'='*50}")
            print(f"Minimal bar chart saved to: {minimal_bar_chart_path}")
        else:
            print("Minimal bar chart only mode only supported for multi_hunk category")

        return

    # If bar-chart-only mode, skip to bar chart generation
    if bar_chart_only:
        print(f"\n{'='*60}")
        print("BAR CHART ONLY MODE")
        print("Generating comprehensive bar chart with HUNK4J comparison")
        print(f"{'='*60}")

        base_dir = Path(__file__).parent.parent / "results" / "defects4j"
        selector_subdir = f"{args.selector_type}_{args.n_lines}line"
        output_dir = Path(args.output) / selector_subdir
        output_dir.mkdir(parents=True, exist_ok=True)

        if args.bug_category == 'multi_hunk':
            # Load data to get common bugs
            print(f"\nLoading baseline data to find common bugs...")
            merged_data1 = {'successful_bugs': set(), 'total_bugs': 0}
            for cat in ['single_file_multi_hunk', 'multi_file_multi_hunk']:
                data1 = load_progress_data(cat, BASELINE_FLAG, base_dir, args.selector_type, args.n_lines)
                merged_data1['successful_bugs'].update(data1['successful_bugs'])
                merged_data1['total_bugs'] += data1['total_bugs']

            merged_data1['success_rate'] = round(len(merged_data1['successful_bugs']) / merged_data1['total_bugs'] * 100, 2) if merged_data1['total_bugs'] > 0 else 0

            # Load HUNK4J
            print(f"\nLoading HUNK4J data...")
            hunk4j_data = load_hunk4j_data(
                "multi_hunk",
                selector_type=args.selector_type,
                n_lines=args.n_lines,
                hafix_results_dir=base_dir,
            )

            # Find common bugs using shared utility
            common_bugs = find_common_bugs_with_hunk4j(base_dir, args.selector_type, args.n_lines)

            print(f"\nGenerating bar chart for all heuristics vs HUNK4J ({len(common_bugs)} common bugs)...")

            bar_data_dict = {}
            heuristics = [
                (BASELINE_FLAG, 'baseline'),
                (FN_ALL_FLAG, 'fn_all'),
                (FN_PAIR_FLAG, 'fn_pair'),
                (FL_DIFF_FLAG, 'fl_diff')
            ]

            for flag, name in heuristics:
                sfmh_data = load_progress_data('single_file_multi_hunk', flag, base_dir, args.selector_type, args.n_lines)
                mfmh_data = load_progress_data('multi_file_multi_hunk', flag, base_dir, args.selector_type, args.n_lines)

                # Filter to common bugs
                sfmh_common = sfmh_data['successful_bugs'] & common_bugs
                mfmh_common = mfmh_data['successful_bugs'] & common_bugs

                bar_data_dict[get_history_name(flag)] = {
                    'sfmh_successful': len(sfmh_common),
                    'mfmh_successful': len(mfmh_common)
                }

            # Add HUNK4J data
            hunk4j_successful = hunk4j_data['successful_bugs'] & common_bugs

            # Load our bug categorization to split HUNK4J results
            sfmh_bugs_set = set()
            mfmh_bugs_set = set()

            for cat, bug_set in [('single_file_multi_hunk', sfmh_bugs_set), ('multi_file_multi_hunk', mfmh_bugs_set)]:
                cat_data = load_progress_data(cat, BASELINE_FLAG, base_dir, args.selector_type, args.n_lines)
                bug_set.update(cat_data.get('successful_bugs', set()))
                bug_set.update(cat_data.get('failed_bugs', set()))

            hunk4j_sfmh = hunk4j_successful & sfmh_bugs_set
            hunk4j_mfmh = hunk4j_successful & mfmh_bugs_set

            bar_data_dict[BIRCH_BASELINE_LABEL] = {
                'sfmh_successful': len(hunk4j_sfmh),
                'mfmh_successful': len(hunk4j_mfmh)
            }

            bar_chart_filename = f"rq1_barchart_all_vs_hunk4j_multi_hunk_{len(common_bugs)}_common_bugs.pdf"
            bar_chart_path = output_dir / bar_chart_filename
            create_bar_chart_comparison(bar_data_dict, f"multi_hunk ({len(common_bugs)} common bugs)", str(bar_chart_path))

            print(f"\n{'='*50}")
            print("SUMMARY")
            print(f"{'='*50}")
            print(f"Bar chart saved to: {bar_chart_path}")
        else:
            print("Bar chart only mode only supported for multi_hunk category")

        return

    # Check if it's a 4-way comparison
    is_4way = args.heuristic3 and args.heuristic4
    if (args.heuristic3 and not args.heuristic4) or (not args.heuristic3 and args.heuristic4):
        parser.error("For 4-way comparison, both --heuristic3 and --heuristic4 are required")

    base_dir = Path(__file__).parent.parent / "results" / "defects4j"
    selector_subdir = f"{args.selector_type}_{args.n_lines}line"
    output_dir = Path(args.output) / selector_subdir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Handle 4-way comparison
    if is_4way:
        print(f"\n{'='*60}")
        print(f"4-WAY COMPARISON MODE")
        print(f"Comparing: {get_history_name(args.heuristic1)}, {get_history_name(args.heuristic2)}, ")
        print(f"          {get_history_name(args.heuristic3)}, {get_history_name(args.heuristic4)}")
        print(f"{'='*60}")

        heuristics = [args.heuristic1, args.heuristic2, args.heuristic3, args.heuristic4]
        all_results = {}

        # Define categories to process and their suffixes
        if args.bug_category == 'all':
            categories_to_process = [
                ('single_line', 'SL'),
                ('single_hunk', 'SH'),
                ('single_file_multi_hunk', 'SFMH'),
                ('multi_file_multi_hunk', 'MFMH'),
                ('all', 'all')  # Combined category for all bugs
            ]
        elif args.bug_category == 'multi_hunk':
            # Only generate one combined diagram for SFMH + MFMH
            categories_to_process = [
                ('multi_hunk', 'multi_hunk')  # Combined category
            ]
        else:
            # Single category mode
            categories_to_process = [(args.bug_category, args.bug_category)]

        # Process each category
        for category_name, suffix in categories_to_process:
            print(f"\n{'='*50}")
            print(f"Processing category: {category_name}")
            print(f"{'='*50}")

            # Load data for all 4 heuristics for this category
            data_list = []

            if category_name == 'multi_hunk':
                # Special case: merge SFMH + MFMH
                data_list = [{'successful_bugs': set(), 'total_bugs': 0} for _ in range(4)]
                for cat in ['single_file_multi_hunk', 'multi_file_multi_hunk']:
                    for i, heur in enumerate(heuristics):
                        print(f"Loading {get_history_name(heur)} from {cat}...")
                        data = load_progress_data(cat, heur, base_dir, args.selector_type, args.n_lines)
                        data_list[i]['successful_bugs'].update(data['successful_bugs'])
                        data_list[i]['total_bugs'] += data['total_bugs']

                # Calculate success rates
                for data in data_list:
                    data['success_rate'] = round(len(data['successful_bugs']) / data['total_bugs'] * 100, 2) if data['total_bugs'] > 0 else 0
            elif category_name == 'all':
                # Special case: merge all four categories
                data_list = [{'successful_bugs': set(), 'total_bugs': 0} for _ in range(4)]
                for cat in ['single_line', 'single_hunk', 'single_file_multi_hunk', 'multi_file_multi_hunk']:
                    for i, heur in enumerate(heuristics):
                        print(f"Loading {get_history_name(heur)} from {cat}...")
                        data = load_progress_data(cat, heur, base_dir, args.selector_type, args.n_lines)
                        data_list[i]['successful_bugs'].update(data['successful_bugs'])
                        data_list[i]['total_bugs'] += data['total_bugs']

                # Calculate success rates
                for data in data_list:
                    data['success_rate'] = round(len(data['successful_bugs']) / data['total_bugs'] * 100, 2) if data['total_bugs'] > 0 else 0
            else:
                # Regular category: load directly
                for heur in heuristics:
                    print(f"Loading {get_history_name(heur)} data...")
                    data_list.append(load_progress_data(category_name, heur, base_dir, args.selector_type, args.n_lines))

            # Create 4-way Venn diagram with category suffix
            output_filename = f"rq1_heuristic_{args.heuristic1}_{args.heuristic2}_{args.heuristic3}_{args.heuristic4}_{suffix}.pdf"
            output_path = output_dir / output_filename

            print(f"\nCreating 4-way Venn diagram for {category_name}...")
            comparison_stats = create_4way_comparison_venn(
                data_list[0], data_list[1], data_list[2], data_list[3],
                args.heuristic1, args.heuristic2, args.heuristic3, args.heuristic4,
                category_name, str(output_path)
            )

            print_4way_comparison_analysis(
                data_list[0], data_list[1], data_list[2], data_list[3],
                args.heuristic1, args.heuristic2, args.heuristic3, args.heuristic4,
                category_name, comparison_stats
            )

            # Save results for this category
            all_results[category_name] = {
                'heuristic1': {'flag': args.heuristic1, 'name': get_history_name(args.heuristic1),
                             'total_bugs': data_list[0]['total_bugs'], 'successful': len(data_list[0]['successful_bugs']),
                             'success_rate': data_list[0]['success_rate']},
                'heuristic2': {'flag': args.heuristic2, 'name': get_history_name(args.heuristic2),
                             'total_bugs': data_list[1]['total_bugs'], 'successful': len(data_list[1]['successful_bugs']),
                             'success_rate': data_list[1]['success_rate']},
                'heuristic3': {'flag': args.heuristic3, 'name': get_history_name(args.heuristic3),
                             'total_bugs': data_list[2]['total_bugs'], 'successful': len(data_list[2]['successful_bugs']),
                             'success_rate': data_list[2]['success_rate']},
                'heuristic4': {'flag': args.heuristic4, 'name': get_history_name(args.heuristic4),
                             'total_bugs': data_list[3]['total_bugs'], 'successful': len(data_list[3]['successful_bugs']),
                             'success_rate': data_list[3]['success_rate']},
                'comparison': comparison_stats,
                'output_file': output_filename
            }

            print(f"Venn diagram saved to: {output_path}")

        # Generate 2x2 grid if requested and we have 'all' category with individual categories loaded
        if args.grid and args.bug_category == 'all':
            print(f"\n{'='*50}")
            print("Generating 2x2 grid Venn diagram...")
            print(f"{'='*50}")

            # Collect data by category (excluding 'all' combined)
            data_by_category = {}
            categories_for_grid = ['single_line', 'single_hunk', 'single_file_multi_hunk', 'multi_file_multi_hunk']

            for cat in categories_for_grid:
                if cat in all_results:
                    # Reconstruct data_list from all_results
                    data_list = []
                    for i, heur in enumerate(heuristics):
                        # Load data for this category and heuristic
                        data = load_progress_data(cat, heur, base_dir, args.selector_type, args.n_lines)
                        data_list.append(data)
                    data_by_category[cat] = data_list

            # Create 2x2 grid figure
            grid_filename = f"rq1_heuristic_{args.heuristic1}_{args.heuristic2}_{args.heuristic3}_{args.heuristic4}_grid.pdf"
            grid_path = output_dir / grid_filename
            create_2x2_grid_venn(data_by_category, args.heuristic1, args.heuristic2, args.heuristic3, args.heuristic4, str(grid_path))

        # Save comprehensive results
        results_filename = f"heuristic_comparison_{args.heuristic1}_{args.heuristic2}_{args.heuristic3}_{args.heuristic4}_{args.bug_category}_results.json"
        results_path = output_dir / results_filename
        with open(results_path, 'w') as f:
            json.dump(all_results, f, indent=2, default=str)

        print(f"\n{'='*60}")
        print("SUMMARY")
        print(f"{'='*60}")
        print(f"Results saved to: {results_path}")
        print(f"Generated {len(all_results)} 4-way Venn diagram(s)")
        if args.grid and args.bug_category == 'all':
            print(f"Generated 1 2x2 grid Venn diagram")

        # Generate bar chart if requested
        if args.bar_chart:
            print(f"\nGenerating bar chart for 4-way comparison...")

            # Need to load SFMH/MFMH breakdown for bar chart
            if args.bug_category == 'multi_hunk':
                bar_data_dict = {}
                for i, heur in enumerate(heuristics):
                    sfmh_data = load_progress_data('single_file_multi_hunk', heur, base_dir, args.selector_type, args.n_lines)
                    mfmh_data = load_progress_data('multi_file_multi_hunk', heur, base_dir, args.selector_type, args.n_lines)

                    bar_data_dict[get_history_name(heur)] = {
                        'sfmh_successful': len(sfmh_data['successful_bugs']),
                        'mfmh_successful': len(mfmh_data['successful_bugs'])
                    }

                bar_chart_filename = f"rq1_barchart_{args.heuristic1}_{args.heuristic2}_{args.heuristic3}_{args.heuristic4}_multi_hunk.pdf"
                bar_chart_path = output_dir / bar_chart_filename
                create_bar_chart_comparison(bar_data_dict, 'multi_hunk', str(bar_chart_path))
            else:
                print("Bar chart only supported for multi_hunk category (requires SFMH/MFMH breakdown)")

        return

    # Handle 2-way comparison (existing logic)
    if args.bug_category == 'multi_hunk':
        # Special case: merge both categories into one analysis
        print(f"\n{'='*50}")
        print("Processing multi_hunk (SFMH + MFMH combined)")
        print(f"{'='*50}")

        # Load and merge data from both categories
        merged_data1 = {'successful_bugs': set(), 'total_bugs': 0}
        merged_data2 = {'successful_bugs': set(), 'total_bugs': 0}

        for cat in ['single_file_multi_hunk', 'multi_file_multi_hunk']:
            print(f"\nLoading {get_history_name(args.heuristic1)} from {cat}...")
            data1 = load_progress_data(cat, args.heuristic1, base_dir, args.selector_type, args.n_lines)
            merged_data1['successful_bugs'].update(data1['successful_bugs'])
            merged_data1['total_bugs'] += data1['total_bugs']

            print(f"Loading {get_history_name(args.heuristic2)} from {cat}...")
            data2 = load_progress_data(cat, args.heuristic2, base_dir, args.selector_type, args.n_lines)
            merged_data2['successful_bugs'].update(data2['successful_bugs'])
            merged_data2['total_bugs'] += data2['total_bugs']

        # Calculate success rates for merged data
        merged_data1['success_rate'] = round(len(merged_data1['successful_bugs']) / merged_data1['total_bugs'] * 100, 2) if merged_data1['total_bugs'] > 0 else 0
        merged_data2['success_rate'] = round(len(merged_data2['successful_bugs']) / merged_data2['total_bugs'] * 100, 2) if merged_data2['total_bugs'] > 0 else 0

        # Load HUNK4J data if requested
        hunk4j_data = None
        if args.include_hunk4j:
            print(f"\nLoading HUNK4J data for merged categories...")
            hunk4j_data = load_hunk4j_data(
                "multi_hunk",
                selector_type=args.selector_type,
                n_lines=args.n_lines,
                hafix_results_dir=base_dir,
            )

        # Create comparison (only common bugs for fair comparison)
        if args.include_hunk4j:
            # Only create common bugs comparison for fair external comparison
            common_bugs_result = create_3way_common_bugs_comparison(
                merged_data1, merged_data2, hunk4j_data, args.heuristic1, args.heuristic2, "multi_hunk",
                output_dir, base_dir, args.selector_type, args.n_lines
            )

            # Generate bar chart if requested
            if args.bar_chart and common_bugs_result:
                print(f"\nGenerating bar chart for comparison with HUNK4J...")

                # Load SFMH/MFMH breakdown for all 4 heuristics + HUNK4J (on common bugs)
                common_bugs = find_common_bugs_with_hunk4j(base_dir, args.selector_type, args.n_lines)

                bar_data_dict = {}
                heuristics = [
                    (BASELINE_FLAG, 'baseline'),
                    (FN_ALL_FLAG, 'fn_all'),
                    (FN_PAIR_FLAG, 'fn_pair'),
                    (FL_DIFF_FLAG, 'fl_diff')
                ]

                for flag, name in heuristics:
                    sfmh_data = load_progress_data('single_file_multi_hunk', flag, base_dir, args.selector_type, args.n_lines)
                    mfmh_data = load_progress_data('multi_file_multi_hunk', flag, base_dir, args.selector_type, args.n_lines)

                    # Filter to common bugs
                    sfmh_common = sfmh_data['successful_bugs'] & common_bugs
                    mfmh_common = mfmh_data['successful_bugs'] & common_bugs

                    bar_data_dict[get_history_name(flag)] = {
                        'sfmh_successful': len(sfmh_common),
                        'mfmh_successful': len(mfmh_common)
                    }

                # Add HUNK4J data (need SFMH/MFMH breakdown)
                # HUNK4J doesn't have SFMH/MFMH breakdown in their data, so we estimate by checking our bug categories
                hunk4j_successful = hunk4j_data['successful_bugs'] & common_bugs

                # Load our bug categorization to split HUNK4J results
                # We need to get ALL bugs (successful + failed) to determine category membership
                sfmh_bugs_set = set()
                mfmh_bugs_set = set()

                for cat, bug_set in [('single_file_multi_hunk', sfmh_bugs_set), ('multi_file_multi_hunk', mfmh_bugs_set)]:
                    cat_data = load_progress_data(cat, BASELINE_FLAG, base_dir, args.selector_type, args.n_lines)
                    bug_set.update(cat_data.get('successful_bugs', set()))
                    bug_set.update(cat_data.get('failed_bugs', set()))

                hunk4j_sfmh = hunk4j_successful & sfmh_bugs_set
                hunk4j_mfmh = hunk4j_successful & mfmh_bugs_set

                bar_data_dict[BIRCH_BASELINE_LABEL] = {
                    'sfmh_successful': len(hunk4j_sfmh),
                    'mfmh_successful': len(hunk4j_mfmh)
                }

                bar_chart_filename = f"rq1_barchart_all_vs_hunk4j_multi_hunk_{len(common_bugs)}_common_bugs.pdf"
                bar_chart_path = output_dir / bar_chart_filename
                create_bar_chart_comparison(bar_data_dict, f"multi_hunk ({len(common_bugs)} common bugs)", str(bar_chart_path))

            # Save common bugs results
            all_results = {}
            if common_bugs_result:
                all_results['multi_hunk_common_bugs'] = common_bugs_result
        else:
            output_filename = f"heuristic_comparison_{args.heuristic1}_vs_{args.heuristic2}_multi_hunk.pdf"
            output_path = output_dir / output_filename

            print(f"\nCreating merged Venn diagram...")
            comparison_stats = create_comparison_venn(
                merged_data1, merged_data2, args.heuristic1, args.heuristic2, "multi_hunk", str(output_path)
            )

            print_comparison_analysis(merged_data1, merged_data2, args.heuristic1, args.heuristic2, "multi_hunk", comparison_stats)

            # Save 2-way results
            all_results = {
                'multi_hunk': {
                    'heuristic1': {
                        'flag': args.heuristic1,
                        'name': get_history_name(args.heuristic1),
                        'total_bugs': merged_data1['total_bugs'],
                        'successful': len(merged_data1['successful_bugs']),
                        'success_rate': merged_data1['success_rate']
                    },
                    'heuristic2': {
                        'flag': args.heuristic2,
                        'name': get_history_name(args.heuristic2),
                        'total_bugs': merged_data2['total_bugs'],
                        'successful': len(merged_data2['successful_bugs']),
                        'success_rate': merged_data2['success_rate']
                    },
                    'comparison': comparison_stats,
                    'output_file': output_filename
                }
            }
            print(f"\nVenn diagram saved to: {output_path}")
    elif args.bug_category == 'all':
        # Special case: merge all four categories into one analysis
        print(f"\n{'='*50}")
        print("Processing all (single_line + single_hunk + SFMH + MFMH combined)")
        print(f"{'='*50}")

        # Load and merge data from all four categories
        merged_data1 = {'successful_bugs': set(), 'total_bugs': 0}
        merged_data2 = {'successful_bugs': set(), 'total_bugs': 0}

        for cat in ['single_line', 'single_hunk', 'single_file_multi_hunk', 'multi_file_multi_hunk']:
            print(f"\nLoading {get_history_name(args.heuristic1)} from {cat}...")
            data1 = load_progress_data(cat, args.heuristic1, base_dir, args.selector_type, args.n_lines)
            merged_data1['successful_bugs'].update(data1['successful_bugs'])
            merged_data1['total_bugs'] += data1['total_bugs']

            print(f"Loading {get_history_name(args.heuristic2)} from {cat}...")
            data2 = load_progress_data(cat, args.heuristic2, base_dir, args.selector_type, args.n_lines)
            merged_data2['successful_bugs'].update(data2['successful_bugs'])
            merged_data2['total_bugs'] += data2['total_bugs']

        # Calculate success rates for merged data
        merged_data1['success_rate'] = round(len(merged_data1['successful_bugs']) / merged_data1['total_bugs'] * 100, 2) if merged_data1['total_bugs'] > 0 else 0
        merged_data2['success_rate'] = round(len(merged_data2['successful_bugs']) / merged_data2['total_bugs'] * 100, 2) if merged_data2['total_bugs'] > 0 else 0

        # Create 2-way comparison (no HUNK4J for 'all' category)
        output_filename = f"heuristic_comparison_{args.heuristic1}_vs_{args.heuristic2}_all.pdf"
        output_path = output_dir / output_filename

        print(f"\nCreating merged Venn diagram...")
        comparison_stats = create_comparison_venn(
            merged_data1, merged_data2, args.heuristic1, args.heuristic2, "all", str(output_path)
        )

        print_comparison_analysis(merged_data1, merged_data2, args.heuristic1, args.heuristic2, "all", comparison_stats)

        # Save 2-way results
        all_results = {
            'all': {
                'heuristic1': {
                    'flag': args.heuristic1,
                    'name': get_history_name(args.heuristic1),
                    'total_bugs': merged_data1['total_bugs'],
                    'successful': len(merged_data1['successful_bugs']),
                    'success_rate': merged_data1['success_rate']
                },
                'heuristic2': {
                    'flag': args.heuristic2,
                    'name': get_history_name(args.heuristic2),
                    'total_bugs': merged_data2['total_bugs'],
                    'successful': len(merged_data2['successful_bugs']),
                    'success_rate': merged_data2['success_rate']
                },
                'comparison': comparison_stats,
                'output_file': output_filename
            }
        }
        print(f"\nVenn diagram saved to: {output_path}")
    else:
        categories_to_process = [args.bug_category]

    if args.bug_category not in ['multi_hunk', 'all']:
        all_results = {}
        for category in categories_to_process:
            print(f"\n{'='*50}")
            print(f"Processing {category}")
            print(f"{'='*50}")

            try:
                # Load data for both heuristics
                print(f"\nLoading {get_history_name(args.heuristic1)} data...")
                data1 = load_progress_data(category, args.heuristic1, base_dir, args.selector_type, args.n_lines)

                print(f"\nLoading {get_history_name(args.heuristic2)} data...")
                data2 = load_progress_data(category, args.heuristic2, base_dir, args.selector_type, args.n_lines)

                # Load HUNK4J data if requested
                hunk4j_data = None
                if args.include_hunk4j:
                    print(f"\nLoading HUNK4J data...")
                    hunk4j_data = load_hunk4j_data(
                        category,
                        selector_type=args.selector_type,
                        n_lines=args.n_lines,
                        hafix_results_dir=base_dir,
                    )

                # Create comparison
                if args.include_hunk4j:
                    output_filename = f"heuristic_comparison_{args.heuristic1}_vs_{args.heuristic2}_vs_hunk4j_{category}.pdf"
                    output_path = output_dir / output_filename

                    print(f"\nCreating 3-way Venn diagram...")
                    comparison_stats = create_3way_comparison_venn(
                        data1, data2, hunk4j_data, args.heuristic1, args.heuristic2, category, str(output_path)
                    )

                    # Print 3-way analysis
                    print_3way_comparison_analysis(data1, data2, hunk4j_data, args.heuristic1, args.heuristic2, category, comparison_stats)

                    # Save 3-way results
                    all_results[category] = {
                        'heuristic1': {
                            'flag': args.heuristic1,
                            'name': get_history_name(args.heuristic1),
                            'total_bugs': data1['total_bugs'],
                            'successful': len(data1['successful_bugs']),
                            'success_rate': data1['success_rate']
                        },
                        'heuristic2': {
                            'flag': args.heuristic2,
                            'name': get_history_name(args.heuristic2),
                            'total_bugs': data2['total_bugs'],
                            'successful': len(data2['successful_bugs']),
                            'success_rate': data2['success_rate']
                        },
                        'hunk4j': {
                            'total_bugs': hunk4j_data['total_bugs'],
                            'successful': len(hunk4j_data['successful_bugs']),
                            'success_rate': hunk4j_data['success_rate']
                        },
                        'comparison': comparison_stats,
                        'output_file': output_filename
                    }
                else:
                    output_filename = f"heuristic_comparison_{args.heuristic1}_vs_{args.heuristic2}_{category}.pdf"
                    output_path = output_dir / output_filename

                    print(f"\nCreating Venn diagram...")
                    comparison_stats = create_comparison_venn(
                        data1, data2, args.heuristic1, args.heuristic2, category, str(output_path)
                    )

                    # Print 2-way analysis
                    print_comparison_analysis(data1, data2, args.heuristic1, args.heuristic2, category, comparison_stats)

                    # Save 2-way results
                    all_results[category] = {
                        'heuristic1': {
                            'flag': args.heuristic1,
                            'name': get_history_name(args.heuristic1),
                            'total_bugs': data1['total_bugs'],
                            'successful': len(data1['successful_bugs']),
                            'success_rate': data1['success_rate']
                        },
                        'heuristic2': {
                            'flag': args.heuristic2,
                            'name': get_history_name(args.heuristic2),
                            'total_bugs': data2['total_bugs'],
                            'successful': len(data2['successful_bugs']),
                            'success_rate': data2['success_rate']
                        },
                        'comparison': comparison_stats,
                        'output_file': output_filename
                    }

                print(f"\nVenn diagram saved to: {output_path}")

            except Exception as e:
                print(f"Error processing {category}: {e}")
                continue

    # Save comprehensive results
    if all_results:
        if args.include_hunk4j:
            results_filename = f"heuristic_comparison_{args.heuristic1}_vs_{args.heuristic2}_vs_hunk4j_{args.bug_category}_results.json"
        else:
            results_filename = f"heuristic_comparison_{args.heuristic1}_vs_{args.heuristic2}_{args.bug_category}_results.json"
        results_path = output_dir / results_filename

        with open(results_path, 'w') as f:
            json.dump(all_results, f, indent=2, default=str)

        print(f"\n{'='*50}")
        print("SUMMARY")
        print(f"{'='*50}")
        print(f"Comparison results saved to: {results_path}")
        print(f"Generated {len(all_results)} comparison(s)")

if __name__ == "__main__":
    main()
