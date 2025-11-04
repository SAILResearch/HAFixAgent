#!/usr/bin/env python3
"""
Analyze the number of unique blame commits for each bug.

This script extracts ALL possible blame commits for each bug by:
- For blameable bugs: Running git blame on every blameable line in the patch
- For blameless bugs: Set commit count to 0 (no fallback strategy)

The goal is to understand the complete set of historical commits that could be
blamed for each bug, independent of selector strategies.
"""

import csv
import sys
import argparse
import concurrent.futures
from pathlib import Path
from typing import Dict, Set, Tuple

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from dataset.defects4j.defects4j_extractor import Defects4JExtractor
from dataset.defects4j.util import get_defects4j_work_dir, ensure_defects4j_docker_container
from hafix_agent.blame.core import run_git_blame
from hafix_agent.blame.patch_parser import PatchParser, PatchFormat


def convert_category_to_abbreviation(full_category: str) -> str:
    """Convert full category name to abbreviation matching CSV format."""
    category_map = {
        'single_line': 'SL',
        'single_hunk': 'SH',
        'single_file_multi_hunk': 'SFMH',
        'multi_file_multi_hunk': 'MFMH'
    }
    return category_map.get(full_category, full_category)


def extract_all_blame_commits_for_bug(
    project_name: str,
    bug_id: int,
    docker_env,
    work_dir: str
) -> Tuple[Set[str], Dict]:
    """
    Extract all unique blame commits for a bug.

    For blameless bugs (no removed/modified lines), commit count is set to 0.

    Args:
        project_name: Defects4J project name
        bug_id: Bug ID
        docker_env: Docker environment
        work_dir: Working directory in container

    Returns:
        Tuple of (set of commit hashes, metadata dict)
    """
    print(f"\n{'='*60}")
    print(f"Processing {project_name}_{bug_id}")
    print(f"{'='*60}")

    commit_hashes = set()
    metadata = {
        'project_name': project_name,
        'bug_id': bug_id,
        'total_blameable_lines': 0,
        'successful_blames': 0,
        'failed_blames': 0,
        'is_blameless': False,
        'used_pre_hunk_fallback': False,
        'total_hunks': 0,
        'error': None
    }

    try:
        # Get patch content directly from Defects4J patch files
        patch_result = docker_env.execute(f"cat /defects4j/framework/projects/{project_name}/patches/{bug_id}.src.patch")
        if patch_result['returncode'] != 0:
            metadata['error'] = f"Could not read patch file: {patch_result.get('output', 'Unknown error')}"
            return commit_hashes, metadata

        patch_content = patch_result['output']
        if not patch_content or not patch_content.strip():
            metadata['error'] = "Patch file is empty"
            return commit_hashes, metadata

        # Count total hunks in patch
        total_hunks = patch_content.count('@@')
        metadata['total_hunks'] = total_hunks // 2  # Each hunk has 2 @@ markers

        # Parse patch to extract all blameable lines
        parser = PatchParser(patch_format=PatchFormat.DEFECTS4J)
        all_lines = parser.extract_code_file_lines(patch_content)
        blamable_lines = parser.get_blamable_lines(all_lines)

        # Check if this is a blameless bug (no removed/modified lines)
        if not blamable_lines:
            metadata['is_blameless'] = True
            print(f"Blameless bug detected - setting commit count to 0")
            return commit_hashes, metadata

        metadata['total_blameable_lines'] = len(blamable_lines)
        print(f"Found {len(blamable_lines)} blameable lines")

        # Run git blame on ALL lines to collect commit hashes
        for i, line in enumerate(blamable_lines, 1):
            try:
                blame_info = run_git_blame(docker_env, work_dir, line.file_path, line.line_number)
                if blame_info and 'commit_hash' in blame_info:
                    commit_hash = blame_info['commit_hash']
                    commit_hashes.add(commit_hash)
                    metadata['successful_blames'] += 1

                    if i % 10 == 0 or i == len(blamable_lines):
                        print(f"  Progress: {i}/{len(blamable_lines)} lines blamed, "
                              f"{len(commit_hashes)} unique commits found")
                else:
                    metadata['failed_blames'] += 1
                    print(f"  Warning: Git blame failed for {line.file_path}:{line.line_number}")
            except Exception as e:
                metadata['failed_blames'] += 1
                print(f"  Error blaming {line.file_path}:{line.line_number}: {str(e)}")

        print(f"\nSummary for {project_name}_{bug_id}:")
        print(f"  Total blameable lines: {metadata['total_blameable_lines']}")
        print(f"  Successful blames: {metadata['successful_blames']}")
        print(f"  Failed blames: {metadata['failed_blames']}")
        print(f"  Unique commits: {len(commit_hashes)}")

    except Exception as e:
        metadata['error'] = f"Exception during processing: {str(e)}"
        print(f"Error processing {project_name}_{bug_id}: {str(e)}")

    return commit_hashes, metadata


def process_single_bug(
    project_name: str,
    bug_id: int,
    docker_image: str
) -> Dict:
    """
    Process a single bug - wrapper for parallel execution.

    Args:
        project_name: Project name
        bug_id: Bug ID
        docker_image: Docker image to use

    Returns:
        Result dictionary
    """
    extractor = Defects4JExtractor()
    docker_env = None

    # Get actual bug category from CSV and convert to abbreviation
    bug_category = extractor.get_bug_category(project_name, bug_id)
    if not bug_category:
        bug_category = "unknown"
    else:
        bug_category = convert_category_to_abbreviation(bug_category)

    try:
        # Create Docker container
        docker_env, error = ensure_defects4j_docker_container(
            project_name, str(bug_id), None,
            image=docker_image,
            use_existing_container=None,
            cleanup_on_exit=True
        )

        if error:
            print(f"Docker setup failed for {project_name}_{bug_id}: {error}")
            return {
                'Bug_ID': f"{project_name}_{bug_id}",
                'Project': project_name,
                'Bug_Number': bug_id,
                'Category': bug_category,
                'Total_Hunks': 0,
                'Unique_Commit_Count': 0,
                'Commit_Hashes': '',
                'Total_Blameable_Lines': 0,
                'Successful_Blames': 0,
                'Failed_Blames': 0,
                'Is_Blameless': False,
                'Used_Pre_Hunk_Fallback': False,
                'Error': f"Docker setup failed: {error}"
            }

        work_dir = get_defects4j_work_dir(project_name, str(bug_id))

        # Extract all blame commits
        commit_hashes, metadata = extract_all_blame_commits_for_bug(
            project_name, bug_id, extractor, docker_env, work_dir
        )

        # Return results (using naming convention from defects4j_blame_feasibility.csv)
        return {
            'Bug_ID': f"{project_name}_{bug_id}",
            'Project': project_name,
            'Bug_Number': bug_id,
            'Category': bug_category,
            'Total_Hunks': metadata['total_hunks'],
            'Unique_Commit_Count': len(commit_hashes),
            'Commit_Hashes': ','.join(sorted(commit_hashes)),
            'Total_Blameable_Lines': metadata['total_blameable_lines'],
            'Successful_Blames': metadata['successful_blames'],
            'Failed_Blames': metadata['failed_blames'],
            'Is_Blameless': metadata['is_blameless'],
            'Used_Pre_Hunk_Fallback': metadata['used_pre_hunk_fallback'],
            'Error': metadata['error'] or ''
        }

    except Exception as e:
        print(f"Error processing {project_name}_{bug_id}: {str(e)}")
        return {
            'Bug_ID': f"{project_name}_{bug_id}",
            'Project': project_name,
            'Bug_Number': bug_id,
            'Category': bug_category,
            'Total_Hunks': 0,
            'Unique_Commit_Count': 0,
            'Commit_Hashes': '',
            'Total_Blameable_Lines': 0,
            'Successful_Blames': 0,
            'Failed_Blames': 0,
            'Is_Blameless': False,
            'Used_Pre_Hunk_Fallback': False,
            'Error': str(e)
        }
    finally:
        # Cleanup Docker container
        if docker_env:
            try:
                docker_env.cleanup()
            except Exception as e:
                print(f"Warning: Failed to cleanup Docker container for {project_name}_{bug_id}: {e}")


def analyze_blame_commits(
    bug_category: str = "multi_file_multi_hunk",
    output_file: str = "results/blame_commit_analysis/commit_counts.csv",
    docker_image: str = "defects4j:latest",
    workers: int = 1
) -> None:
    """
    Main analysis function to count unique blame commits for bugs.

    Always processes all bugs (both blameable and blameless). Blameless bugs
    will have commit count set to 0.

    Args:
        bug_category: Bug category filter
        output_file: Output CSV file path
        docker_image: Docker image to use
        workers: Number of parallel workers
    """
    print("="*80)
    print("Blame Commit Count Analysis")
    print("="*80)
    print(f"Bug category: {bug_category}")
    print(f"Docker image: {docker_image}")
    print(f"Workers: {workers}")
    print("="*80)

    # Get bugs to analyze (always include both blameable and blameless)
    extractor = Defects4JExtractor()
    blameable_bugs, blameless_bugs = extractor.get_filtered_bugs_with_blameless(
        bug_category, "", 0, include_blameless=True
    )
    bugs_to_analyze = blameable_bugs + blameless_bugs
    print(f"Bugs to analyze: {len(blameable_bugs)} blameable + {len(blameless_bugs)} blameless")

    if not bugs_to_analyze:
        print("No bugs found matching criteria")
        return

    # Prepare output
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Process bugs in parallel
    print(f"\nStarting analysis with {workers} workers...")

    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        # Submit all tasks
        future_to_bug = {
            executor.submit(
                process_single_bug,
                project_name,
                bug_id,
                docker_image
            ): (project_name, bug_id)
            for project_name, bug_id in bugs_to_analyze
        }

        # Collect results as they complete
        for i, future in enumerate(concurrent.futures.as_completed(future_to_bug), 1):
            project_name, bug_id = future_to_bug[future]
            try:
                result = future.result()
                results.append(result)
                print(f"[{i}/{len(bugs_to_analyze)}] Completed {project_name}_{bug_id}")
            except Exception as e:
                print(f"[{i}/{len(bugs_to_analyze)}] Exception for {project_name}_{bug_id}: {e}")
                # Get category for error case and convert to abbreviation
                try:
                    error_category = extractor.get_bug_category(project_name, int(bug_id))
                    if error_category:
                        error_category = convert_category_to_abbreviation(error_category)
                    else:
                        error_category = "unknown"
                except:
                    error_category = "unknown"

                results.append({
                    'Bug_ID': f"{project_name}_{bug_id}",
                    'Project': project_name,
                    'Bug_Number': bug_id,
                    'Category': error_category,
                    'Total_Hunks': 0,
                    'Unique_Commit_Count': 0,
                    'Commit_Hashes': '',
                    'Total_Blameable_Lines': 0,
                    'Successful_Blames': 0,
                    'Failed_Blames': 0,
                    'Is_Blameless': False,
                    'Used_Pre_Hunk_Fallback': False,
                    'Error': str(e)
                })

    # Write results to CSV
    print(f"\n{'='*80}")
    print(f"Writing results to {output_file}")
    print(f"{'='*80}")

    with open(output_file, 'w', newline='') as f:
        fieldnames = [
            'Bug_ID', 'Project', 'Bug_Number', 'Category', 'Total_Hunks',
            'Unique_Commit_Count', 'Commit_Hashes', 'Total_Blameable_Lines',
            'Successful_Blames', 'Failed_Blames', 'Is_Blameless',
            'Used_Pre_Hunk_Fallback', 'Error'
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    # Print summary statistics
    print(f"\nAnalysis complete!")
    print(f"Total bugs processed: {len(results)}")

    successful = [r for r in results if not r['Error']]
    failed = [r for r in results if r['Error']]
    blameless = [r for r in results if r['Is_Blameless']]

    print(f"Successful: {len(successful)}")
    print(f"Failed: {len(failed)}")
    print(f"Blameless bugs: {len(blameless)}")

    if successful:
        commit_counts = [r['Unique_Commit_Count'] for r in successful]
        print(f"\nCommit count statistics:")
        print(f"  Min: {min(commit_counts)}")
        print(f"  Max: {max(commit_counts)}")
        print(f"  Mean: {sum(commit_counts) / len(commit_counts):.2f}")
        print(f"  Median: {sorted(commit_counts)[len(commit_counts)//2]}")

    print(f"\nResults saved to: {output_file}")


def print_commit_distribution_statistics(csv_file: str, output_dir: str = None) -> None:
    """
    Print and visualize the distribution of unique commit counts by bug category.

    Args:
        csv_file: Path to the CSV file with blame commit analysis results
        output_dir: Directory to save figures (if None, uses same dir as csv_file)
    """
    import pandas as pd
    import matplotlib.pyplot as plt
    from collections import Counter

    # Read CSV
    df = pd.read_csv(csv_file)

    # Count bugs with errors separately
    total_bugs = len(df)
    error_bugs = df[(df['Error'].notna()) & (df['Error'] != '')]
    num_error_bugs = len(error_bugs)

    # For bugs with errors, set commit count to 0
    df.loc[(df['Error'].notna()) & (df['Error'] != ''), 'Unique_Commit_Count'] = 0

    if output_dir is None:
        output_dir = Path(csv_file).parent
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Auto-detect category format (abbreviated or full names)
    sample_categories = df['Category'].unique()
    if any(cat in ['SL', 'SH', 'SFMH', 'MFMH'] for cat in sample_categories):
        # Abbreviated format
        categories = ['SL', 'SH', 'SFMH', 'MFMH']
    else:
        # Full names format
        categories = ['single_line', 'single_hunk', 'single_file_multi_hunk', 'multi_file_multi_hunk']

    print("\n" + "="*80)
    print("Blame Commit Distribution Statistics")
    print("="*80)

    # Overall statistics
    print("\n[ALL BUGS]")
    all_counts = df['Unique_Commit_Count'].values
    print(f"Total bugs: {len(all_counts)}")
    print(f"Mean commits: {all_counts.mean():.2f}")
    print(f"Median commits: {int(pd.Series(all_counts).median())}")
    print(f"Min commits: {all_counts.min()}")
    print(f"Max commits: {all_counts.max()}")

    # Distribution
    count_dist = Counter(all_counts)
    print("\nDistribution:")
    for commit_count in sorted(count_dist.keys()):
        num_bugs = count_dist[commit_count]
        percentage = (num_bugs / len(all_counts)) * 100
        print(f"  {commit_count} commit(s): {num_bugs} bugs ({percentage:.1f}%)")

    # Per-category statistics
    for category in categories:
        cat_df = df[df['Category'] == category]
        if len(cat_df) == 0:
            continue

        print(f"\n[{category}]")
        cat_counts = cat_df['Unique_Commit_Count'].values
        print(f"Total bugs: {len(cat_counts)}")
        print(f"Mean commits: {cat_counts.mean():.2f}")
        print(f"Median commits: {int(pd.Series(cat_counts).median())}")
        print(f"Min commits: {cat_counts.min()}")
        print(f"Max commits: {cat_counts.max()}")

        # Distribution
        count_dist = Counter(cat_counts)
        print("Distribution:")
        for commit_count in sorted(count_dist.keys()):
            num_bugs = count_dist[commit_count]
            percentage = (num_bugs / len(cat_counts)) * 100
            print(f"  {commit_count} commit(s): {num_bugs} bugs ({percentage:.1f}%)")

    # Create separate visualizations for each category
    print(f"\n{'='*80}")
    print("Generating figures...")
    print(f"{'='*80}")

    # Helper function to create and save a single figure
    def create_distribution_figure(counts, title, filename, output_dir):
        """Create a bar chart for commit distribution."""
        count_dist = Counter(counts)
        commits = sorted(count_dist.keys())
        frequencies = [count_dist[c] for c in commits]

        fig, ax = plt.subplots(figsize=(8, 6))

        # Use fixed absolute bar width for consistency across all figures
        # This ensures bars have the same visual width regardless of x-axis range
        bar_width = 0.35

        bars = ax.bar(commits, frequencies, color='#4A90E2', alpha=0.9, width=bar_width)
        ax.set_xlabel('Blame Commits', fontsize=13)
        ax.set_ylabel('Number of Bugs', fontsize=13)
        # ax.set_title(title, fontsize=14, fontweight='bold')  # Title commented out for research paper
        ax.grid(axis='y', alpha=0.3)

        # Add value labels on top of each bar
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{int(height)}',
                   ha='center', va='bottom', fontsize=11)

        # Set x-axis limits with consistent spacing
        # Use a fixed range to ensure consistent visual appearance
        max_commits = max(commits)
        min_commits = min(commits)

        # Set x-axis range from 0 to max observed across all categories (7)
        # This ensures consistent scale across all figures
        ax.set_xlim(-0.5, 7.5)

        # Set integer ticks for x-axis - only show ticks that have data
        ax.set_xticks(commits)

        plt.tight_layout()
        output_path = output_dir / filename
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"  Saved: {output_path}")
        plt.close()

    # Helper function to create grouped bar chart for all categories
    def create_grouped_distribution_figure(df, categories, output_dir):
        """Create a grouped bar chart showing all 4 categories together."""
        import numpy as np

        # Group commit counts: 0, 1, >=2
        commit_groups = ['0', '1', '≥2']

        # Prepare data for each category
        category_data = {}
        for category in categories:
            cat_df = df[df['Category'] == category]
            if len(cat_df) == 0:
                continue

            # Count bugs in each group
            count_0 = len(cat_df[cat_df['Unique_Commit_Count'] == 0])
            count_1 = len(cat_df[cat_df['Unique_Commit_Count'] == 1])
            count_2_plus = len(cat_df[cat_df['Unique_Commit_Count'] >= 2])

            frequencies = [count_0, count_1, count_2_plus]
            category_data[category] = frequencies

        # Set up the figure
        fig, ax = plt.subplots(figsize=(10, 6))

        # Set up bar positions
        x = np.arange(len(commit_groups))  # Label locations for '0', '1', '≥2'
        width = 0.2  # Width of bars
        multiplier = 0

        # Color palette for categories (consistent with RQ2/RQ3 figures)
        colors = {
            'SL': '#70AD47',      # Green (distinct from blues)
            'SH': '#B19CD9',      # Soft purple
            'SFMH': '#5B9BD5',    # Standard blue (matches RQ2)
            'MFMH': '#ED7D31'     # Standard orange (matches RQ2)
        }

        # Plot bars for each category
        for category in categories:
            if category not in category_data:
                continue

            frequencies = category_data[category]
            offset = width * multiplier
            bars = ax.bar(x + offset, frequencies, width, label=category,
                         color=colors.get(category, '#4A90E2'), alpha=0.9)

            # Add value labels on top of each bar
            for i, (bar, freq) in enumerate(zip(bars, frequencies)):
                height = bar.get_height()
                # For zero values, place label slightly above the x-axis
                y_pos = height if height > 0 else 1
                ax.text(bar.get_x() + bar.get_width()/2., y_pos,
                       f'{int(freq)}',
                       ha='center', va='bottom', fontsize=9)

            multiplier += 1

        # Customize the plot
        ax.set_xlabel('Number of Unique Blame Commits', fontsize=13)
        ax.set_ylabel('Number of Bugs', fontsize=13)
        ax.set_xticks(x + width * 1.5)  # Center the x-axis labels
        ax.set_xticklabels(commit_groups)  # Use grouped labels: '0', '1', '≥2'
        ax.legend(loc='upper right', fontsize=11)
        ax.grid(axis='y', alpha=0.3)

        plt.tight_layout()
        output_path = output_dir / 'rq1_blame_commit_distribution_all.pdf'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"  Saved: {output_path}")
        plt.close()

    # Helper function to create stacked bar chart for all categories
    def create_stacked_distribution_figure(df, categories, output_dir):
        """Create a stacked bar chart showing all 4 categories with commit count groups."""
        import numpy as np

        # Group commit counts: 0, 1, >=2
        commit_groups = ['0', '1', r'$\geq$2']
        commit_labels = ['0', '1', '≥2']

        # Prepare data for each category
        category_data = {}
        for category in categories:
            cat_df = df[df['Category'] == category]
            if len(cat_df) == 0:
                category_data[category] = [0, 0, 0]
                continue

            # Count bugs in each group
            count_0 = len(cat_df[cat_df['Unique_Commit_Count'] == 0])
            count_1 = len(cat_df[cat_df['Unique_Commit_Count'] == 1])
            count_2_plus = len(cat_df[cat_df['Unique_Commit_Count'] >= 2])

            category_data[category] = [count_0, count_1, count_2_plus]

        # Set up the figure
        fig, ax = plt.subplots(figsize=(10, 6))

        # Color palette for commit groups with high contrast for rare ≥2 cases
        # Light green for 0, medium green for 1, DISTINCT dark color for ≥2
        colors = ['#C8E6C9', '#66BB6A', '#D32F2F']  # Light green, medium green, RED (for visibility)

        # Position for bars
        x = np.arange(len(categories))
        width = 0.6

        # Create stacked bars
        bottom = np.zeros(len(categories))
        bars_list = []
        max_annotation_height = 0

        for i, (label, color) in enumerate(zip(commit_labels, colors)):
            heights = [category_data[cat][i] for cat in categories]
            bars = ax.bar(x, heights, width, label=label, color=color,
                         bottom=bottom, alpha=0.9, edgecolor='white', linewidth=1.5)
            bars_list.append((bars, heights, label))

            # Add value labels inside bars (only for non-zero values)
            for j, (bar, height) in enumerate(zip(bars, heights)):
                if height > 0:
                    y_pos = bottom[j] + height / 2
                    # For ≥2 segment (very small), add annotation with arrow if too small to fit text
                    if i == 2 and height < 10:  # ≥2 cases are rare and small
                        # Calculate annotation position dynamically
                        annotation_y = bottom[j] + height + 15
                        max_annotation_height = max(max_annotation_height, annotation_y + 5)

                        # Add arrow annotation pointing to the small segment
                        ax.annotate(f'{int(height)}',
                                   xy=(bar.get_x() + bar.get_width()/2., bottom[j] + height),
                                   xytext=(bar.get_x() + bar.get_width()/2., annotation_y),
                                   ha='center', va='bottom', fontsize=10,
                                   color='#D32F2F',
                                   arrowprops=dict(arrowstyle='->', color='#D32F2F', lw=1.5))
                    else:
                        ax.text(bar.get_x() + bar.get_width()/2., y_pos,
                               f'{int(height)}',
                               ha='center', va='center', fontsize=11,
                               color='white' if i == 2 else 'black')

            bottom += heights

        # Customize the plot
        ax.set_xlabel('Bug Category', fontsize=13)
        ax.set_ylabel('Number of Bugs', fontsize=13)
        ax.set_xticks(x)
        ax.set_xticklabels(categories, fontsize=12)
        ax.legend(title='Unique Blame Commits', loc='upper right', fontsize=11,
                 title_fontsize=11, framealpha=0.95)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.set_axisbelow(True)

        # Set y-axis to accommodate both bars and annotations
        y_max = max(max(bottom), max_annotation_height)
        ax.set_ylim(bottom=0, top=y_max * 1.02)

        plt.tight_layout()
        output_path = output_dir / 'rq1_blame_commit_distribution_stacked.pdf'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"  Saved: {output_path}")
        plt.close()

    # Plot stacked bar chart for all categories
    create_stacked_distribution_figure(df, categories, output_dir)

    # Plot grouped bar chart for all categories
    create_grouped_distribution_figure(df, categories, output_dir)

    # Plot for each category
    for category in categories:
        cat_df = df[df['Category'] == category]
        if len(cat_df) == 0:
            continue

        cat_counts = cat_df['Unique_Commit_Count'].values
        # Convert category name to filename-friendly format
        cat_name = category.replace('_', '-')
        create_distribution_figure(
            cat_counts,
            f'Distribution of Unique Blame Commits - {category.upper()} ({len(cat_counts)} bugs)',
            f'rq1_blame_commit_distribution_{cat_name}.pdf',
            output_dir
        )

    print(f"{'='*80}")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze the number of unique blame commits for each bug. "
                    "Always processes both blameable and blameless bugs (blameless bugs have commit count=0)."
    )
    parser.add_argument(
        "--bug-category",
        default="single_file_multi_hunk",
        help="Bug category filter: 'single_line', 'single_hunk', 'single_file_multi_hunk', 'multi_file_multi_hunk', or 'all'"
    )
    parser.add_argument(
        "--output",
        "-o",
        default="results/blame_commit_analysis/commit_counts.csv",
        help="Output CSV file path"
    )
    parser.add_argument(
        "--docker-image",
        default="defects4j:latest",
        help="Docker image to use (default: defects4j:latest)"
    )
    parser.add_argument(
        "--workers",
        "-w",
        type=int,
        default=1,
        help="Number of parallel workers (default: 1)"
    )
    parser.add_argument(
        "--stats",
        "-s",
        action="store_true",
        help="Generate distribution statistics from existing CSV file (no Docker needed)"
    )

    args = parser.parse_args()

    # If stats mode, just generate statistics from existing CSV
    if args.stats:
        print_commit_distribution_statistics(args.output)
    else:
        analyze_blame_commits(
            bug_category=args.bug_category,
            output_file=args.output,
            docker_image=args.docker_image,
            workers=args.workers
        )


if __name__ == "__main__":
    main()
