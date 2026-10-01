#!/usr/bin/env python3
"""
BugsInPy RQ0 Blame Commit Analysis - Count unique blame commits per bug.

This script:
1. Clones BugsInPy project repositories (17 Python projects)
2. For each bug, checks out at the buggy commit
3. Runs git blame on all blameable lines (deletions in forward patches)
4. Counts unique blame commit hashes

Unlike Defects4J (which uses Docker), this runs locally since Python projects
don't require special compilation environments.

Usage:
    # Run full analysis (clones repos, may take a while first time)
    python -m analysis.analyze_rq0_bugsinpy_blame_commit_count -p tqdm httpie

    # Generate statistics from existing CSV
    python -m analysis.analyze_rq0_bugsinpy_blame_commit_count --stats
"""

import csv
import subprocess
import argparse
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional
from collections import Counter

from dataset.bugsinpy.util import (
    get_all_bugs,
    get_bug_info,
    get_patch_file_path,
    bugsinpy_project_name_url_map,
    BUGSINPY_BASE_PATH,
)


# Default directory for cloned repositories
REPOS_DIR = BUGSINPY_BASE_PATH.parent / "repos"


def get_repos_dir() -> Path:
    """Get the directory for cloned BugsInPy project repositories."""
    return REPOS_DIR


def clone_or_update_repo(
    project_name: str, github_url: str, repos_dir: Path
) -> Tuple[bool, str]:
    """
    Clone a repository if it doesn't exist, or update it if it does.

    Args:
        project_name: Name of the project
        github_url: GitHub URL to clone from
        repos_dir: Directory to store cloned repos

    Returns:
        Tuple of (success, error_message)
    """
    repo_path = repos_dir / project_name
    repos_dir.mkdir(parents=True, exist_ok=True)

    if repo_path.exists():
        # Repo exists, just fetch updates
        try:
            result = subprocess.run(
                ["git", "fetch", "--all"],
                cwd=repo_path,
                capture_output=True,
                text=True,
                timeout=300,
            )
            if result.returncode != 0:
                return False, f"Git fetch failed: {result.stderr}"
            return True, ""
        except subprocess.TimeoutExpired:
            return False, "Git fetch timed out"
        except Exception as e:
            return False, f"Git fetch error: {str(e)}"
    else:
        # Clone the repository
        try:
            print(f"  Cloning {project_name} from {github_url}...")
            result = subprocess.run(
                ["git", "clone", "--depth=1", github_url, str(repo_path)],
                capture_output=True,
                text=True,
                timeout=600,
            )
            if result.returncode != 0:
                return False, f"Git clone failed: {result.stderr}"

            # Unshallow to get full history for blame
            print(f"  Fetching full history for {project_name}...")
            result = subprocess.run(
                ["git", "fetch", "--unshallow"],
                cwd=repo_path,
                capture_output=True,
                text=True,
                timeout=1800,  # 30 min for large repos
            )
            # unshallow may fail if already complete, ignore error

            return True, ""
        except subprocess.TimeoutExpired:
            return False, "Git clone timed out"
        except Exception as e:
            return False, f"Git clone error: {str(e)}"


def checkout_commit(repo_path: Path, commit_id: str) -> Tuple[bool, str]:
    """
    Checkout a specific commit in the repository.

    Args:
        repo_path: Path to the repository
        commit_id: Git commit hash to checkout

    Returns:
        Tuple of (success, error_message)
    """
    try:
        # First try to fetch the specific commit if not available
        result = subprocess.run(
            ["git", "cat-file", "-t", commit_id],
            cwd=repo_path,
            capture_output=True,
            text=True,
            timeout=30,
        )

        if result.returncode != 0:
            # Commit not found, try fetching it
            subprocess.run(
                ["git", "fetch", "origin", commit_id],
                cwd=repo_path,
                capture_output=True,
                text=True,
                timeout=120,
            )

        # Checkout the commit
        result = subprocess.run(
            ["git", "checkout", "-f", commit_id],
            cwd=repo_path,
            capture_output=True,
            text=True,
            timeout=60,
        )

        if result.returncode != 0:
            return False, f"Git checkout failed: {result.stderr}"

        return True, ""

    except subprocess.TimeoutExpired:
        return False, "Git checkout timed out"
    except Exception as e:
        return False, f"Git checkout error: {str(e)}"


def run_git_blame_local(
    repo_path: Path, file_path: str, line_number: int
) -> Optional[Dict]:
    """
    Run git blame on a specific line locally.

    Args:
        repo_path: Path to the repository
        file_path: Relative path to file in repo
        line_number: Line number to blame

    Returns:
        Dictionary with blame info or None if failed
    """
    try:
        full_path = repo_path / file_path
        if not full_path.exists():
            return None

        result = subprocess.run(
            [
                "git",
                "blame",
                "--porcelain",
                "-L",
                f"{line_number},{line_number}",
                file_path,
            ],
            cwd=repo_path,
            capture_output=True,
            text=True,
            timeout=30,
        )

        if result.returncode != 0:
            return None

        # Parse porcelain output
        lines = result.stdout.strip().split("\n")
        if not lines:
            return None

        # First line has commit hash
        commit_hash = lines[0].split()[0]

        # Parse metadata
        blame_info = {"commit_hash": commit_hash}
        for line in lines[1:]:
            if line.startswith("author "):
                blame_info["author"] = line[7:]
            elif line.startswith("summary "):
                blame_info["summary"] = line[8:]
            elif line.startswith("filename "):
                blame_info["filename"] = line[9:]

        return blame_info

    except subprocess.TimeoutExpired:
        return None
    except Exception:
        return None


def parse_patch_for_blameable_lines(patch_file: Path) -> List[Dict]:
    """
    Parse a patch file to extract blameable lines.

    For BugsInPy (forward patches), '-' lines are blameable.

    Args:
        patch_file: Path to the bug_patch.txt file

    Returns:
        List of dicts with file_path and line_number
    """
    blameable_lines = []

    with open(patch_file, "r", encoding="utf-8", errors="replace") as f:
        lines = f.readlines()

    current_file = None
    old_line_num = 0

    for line in lines:
        # Track current file
        if line.startswith("--- a/"):
            current_file = line[6:].strip()
        elif line.startswith("--- "):
            # Handle "--- a/path" or "--- /dev/null"
            path = line[4:].strip()
            if path.startswith("a/"):
                current_file = path[2:]
            elif path != "/dev/null":
                current_file = path

        # Parse hunk header to get line numbers
        elif line.startswith("@@"):
            # Format: @@ -start,count +start,count @@
            try:
                parts = line.split("@@")[1].strip().split()
                old_range = parts[0]  # e.g., "-10,5"
                old_start = int(old_range.split(",")[0].replace("-", ""))
                old_line_num = old_start
            except (IndexError, ValueError):
                continue

        # Track deletions (blameable in forward patches)
        elif line.startswith("-") and not line.startswith("---"):
            if current_file and old_line_num > 0:
                blameable_lines.append(
                    {"file_path": current_file, "line_number": old_line_num}
                )
            old_line_num += 1

        # Context lines advance old line counter
        elif not line.startswith("+") and not line.startswith("\\"):
            if line.startswith(" ") or (current_file and not line.startswith("diff")):
                old_line_num += 1

    return blameable_lines


def extract_all_blame_commits_for_bug(
    project_name: str, bug_id: int, repos_dir: Path
) -> Tuple[Set[str], Dict]:
    """
    Extract all unique blame commits for a bug.

    Args:
        project_name: BugsInPy project name
        bug_id: Bug ID
        repos_dir: Directory containing cloned repos

    Returns:
        Tuple of (set of commit hashes, metadata dict)
    """
    commit_hashes = set()
    metadata = {
        "project_name": project_name,
        "bug_id": bug_id,
        "total_blameable_lines": 0,
        "successful_blames": 0,
        "failed_blames": 0,
        "is_blameless": False,
        "error": None,
    }

    try:
        # Get bug info
        bug_info = get_bug_info(project_name, bug_id)
        buggy_commit = bug_info.get("buggy_commit_id", "")

        if not buggy_commit:
            metadata["error"] = "No buggy_commit_id found"
            return commit_hashes, metadata

        # Ensure repo is available and checkout buggy commit
        repo_path = repos_dir / project_name
        if not repo_path.exists():
            metadata["error"] = f"Repository not cloned: {project_name}"
            return commit_hashes, metadata

        success, error = checkout_commit(repo_path, buggy_commit)
        if not success:
            metadata["error"] = f"Checkout failed: {error}"
            return commit_hashes, metadata

        # Parse patch to get blameable lines
        patch_file = get_patch_file_path(project_name, bug_id)
        if not patch_file.exists():
            metadata["error"] = "Patch file not found"
            return commit_hashes, metadata

        blameable_lines = parse_patch_for_blameable_lines(patch_file)

        if not blameable_lines:
            metadata["is_blameless"] = True
            return commit_hashes, metadata

        metadata["total_blameable_lines"] = len(blameable_lines)

        # Run git blame on each blameable line
        for line_info in blameable_lines:
            blame_result = run_git_blame_local(
                repo_path, line_info["file_path"], line_info["line_number"]
            )

            if blame_result and "commit_hash" in blame_result:
                commit_hashes.add(blame_result["commit_hash"])
                metadata["successful_blames"] += 1
            else:
                metadata["failed_blames"] += 1

    except Exception as e:
        metadata["error"] = str(e)

    return commit_hashes, metadata


def get_bug_category(project_name: str, bug_id: int) -> str:
    """Get bug category from existing analysis."""
    patch_file = get_patch_file_path(project_name, bug_id)
    if not patch_file.exists():
        return "Unknown"

    from dataset.bugsinpy.bugsinpy_analysis import (
        is_single_line_bug,
        is_single_hunk_bug,
        analyze_patch_files,
    )

    if is_single_line_bug(str(patch_file)):
        return "SL"
    elif is_single_hunk_bug(str(patch_file)):
        return "SH"
    else:
        analysis = analyze_patch_files(str(patch_file))
        if analysis["files_changed"] == 1:
            return "SFMH"
        else:
            return "MFMH"


def process_single_bug(project_name: str, bug_id: int, repos_dir: Path) -> Dict:
    """
    Process a single bug - wrapper for parallel execution.

    Args:
        project_name: Project name
        bug_id: Bug ID
        repos_dir: Directory containing cloned repos

    Returns:
        Result dictionary
    """
    category = get_bug_category(project_name, bug_id)

    try:
        commit_hashes, metadata = extract_all_blame_commits_for_bug(
            project_name, bug_id, repos_dir
        )

        return {
            "Bug_ID": f"{project_name}_{bug_id}",
            "Project": project_name,
            "Bug_Number": bug_id,
            "Category": category,
            "Unique_Commit_Count": len(commit_hashes),
            "Commit_Hashes": ",".join(sorted(commit_hashes)),
            "Total_Blameable_Lines": metadata["total_blameable_lines"],
            "Successful_Blames": metadata["successful_blames"],
            "Failed_Blames": metadata["failed_blames"],
            "Is_Blameless": metadata["is_blameless"],
            "Error": metadata["error"] or "",
        }

    except Exception as e:
        return {
            "Bug_ID": f"{project_name}_{bug_id}",
            "Project": project_name,
            "Bug_Number": bug_id,
            "Category": category,
            "Unique_Commit_Count": 0,
            "Commit_Hashes": "",
            "Total_Blameable_Lines": 0,
            "Successful_Blames": 0,
            "Failed_Blames": 0,
            "Is_Blameless": False,
            "Error": str(e),
        }


def clone_all_repos(repos_dir: Path) -> Dict[str, bool]:
    """
    Clone all BugsInPy project repositories.

    Args:
        repos_dir: Directory to store cloned repos

    Returns:
        Dict mapping project names to success status
    """
    print("=" * 60)
    print("Cloning BugsInPy Project Repositories")
    print("=" * 60)

    results = {}

    for project_name, github_url in bugsinpy_project_name_url_map.items():
        print(f"\n[{project_name}]")
        success, error = clone_or_update_repo(project_name, github_url, repos_dir)
        results[project_name] = success
        if success:
            print("  ✓ Ready")
        else:
            print(f"  ✗ Failed: {error}")

    successful = sum(1 for s in results.values() if s)
    print(f"\n{'=' * 60}")
    print(f"Repository Status: {successful}/{len(results)} ready")
    print("=" * 60)

    return results


def analyze_blame_commits(
    output_file: str = None,
    repos_dir: Path = None,
    workers: int = 1,
    projects: List[str] = None,
) -> None:
    """
    Main analysis function to count unique blame commits for all BugsInPy bugs.

    Args:
        output_file: Output CSV file path
        repos_dir: Directory containing cloned repos
        workers: Number of parallel workers
        projects: Optional list of specific projects to analyze
    """
    if output_file is None:
        output_file = str(
            Path(__file__).parent.parent
            / "results"
            / "rq0_bugsinpy"
            / "bugsinpy_blame_commit_counts.csv"
        )

    if repos_dir is None:
        repos_dir = get_repos_dir()

    print("=" * 80)
    print("BugsInPy Blame Commit Count Analysis")
    print("=" * 80)
    print(f"Repos directory: {repos_dir}")
    print(f"Output file: {output_file}")
    print(f"Workers: {workers}")
    print("=" * 80)

    # Step 1: Clone/update all repositories
    clone_status = clone_all_repos(repos_dir)

    # Step 2: Get all bugs
    all_bugs = get_all_bugs()

    if projects:
        all_bugs = {p: bugs for p, bugs in all_bugs.items() if p in projects}

    # Filter to only projects with successful clones
    bugs_to_analyze = []
    for project_name, bug_ids in all_bugs.items():
        if clone_status.get(project_name, False):
            for bug_id in bug_ids:
                bugs_to_analyze.append((project_name, bug_id))
        else:
            print(f"Skipping {project_name} (repo not available)")

    print(f"\nTotal bugs to analyze: {len(bugs_to_analyze)}")

    if not bugs_to_analyze:
        print("No bugs to analyze!")
        return

    # Step 3: Process bugs (sequentially per project to avoid checkout conflicts)
    results = []

    # Group by project for sequential processing within same repo
    from collections import defaultdict

    bugs_by_project = defaultdict(list)
    for project_name, bug_id in bugs_to_analyze:
        bugs_by_project[project_name].append(bug_id)

    total_processed = 0
    for project_name, bug_ids in bugs_by_project.items():
        print(f"\n[{project_name}] Processing {len(bug_ids)} bugs...")

        for i, bug_id in enumerate(bug_ids, 1):
            result = process_single_bug(project_name, bug_id, repos_dir)
            results.append(result)
            total_processed += 1

            if i % 10 == 0 or i == len(bug_ids):
                print(
                    f"  Progress: {i}/{len(bug_ids)} ({total_processed}/{len(bugs_to_analyze)} total)"
                )

    # Step 4: Write results to CSV
    print(f"\n{'=' * 80}")
    print(f"Writing results to {output_file}")
    print("=" * 80)

    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "w", newline="") as f:
        fieldnames = [
            "Bug_ID",
            "Project",
            "Bug_Number",
            "Category",
            "Unique_Commit_Count",
            "Commit_Hashes",
            "Total_Blameable_Lines",
            "Successful_Blames",
            "Failed_Blames",
            "Is_Blameless",
            "Error",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    # Print summary
    print_summary(results)
    print(f"\nResults saved to: {output_file}")


def print_summary(results: List[Dict]) -> None:
    """Print summary statistics from results."""
    print(f"\n{'=' * 60}")
    print("Analysis Summary")
    print("=" * 60)

    total = len(results)
    successful = [r for r in results if not r["Error"]]
    failed = [r for r in results if r["Error"]]
    blameless = [r for r in results if r["Is_Blameless"]]

    print(f"Total bugs processed: {total}")
    print(f"Successful: {len(successful)}")
    print(f"Failed: {len(failed)}")
    print(f"Blameless bugs: {len(blameless)}")

    if successful:
        commit_counts = [r["Unique_Commit_Count"] for r in successful]
        print("\nCommit count statistics:")
        print(f"  Min: {min(commit_counts)}")
        print(f"  Max: {max(commit_counts)}")
        print(f"  Mean: {sum(commit_counts) / len(commit_counts):.2f}")

        # Distribution
        count_dist = Counter(commit_counts)
        print("\nDistribution:")
        for count in sorted(count_dist.keys()):
            num = count_dist[count]
            pct = num / len(commit_counts) * 100
            print(f"  {count} commit(s): {num} bugs ({pct:.1f}%)")


def print_statistics_from_csv(csv_file: str, output_dir: str = None) -> None:
    """
    Generate statistics, figures, and LaTeX tables from existing CSV file.

    Generates:
    - rq0_bugsinpy_blame_commit_distribution_stacked.pdf
    - rq0_bugsinpy_blame_distribution_table.tex
    - Console statistics
    """
    import pandas as pd

    df = pd.read_csv(csv_file)

    if output_dir is None:
        output_dir = Path(csv_file).parent
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 80)
    print("BugsInPy Blame Commit Distribution Statistics")
    print("=" * 80)

    # Overall statistics
    print("\n[ALL BUGS]")
    all_counts = df["Unique_Commit_Count"].values
    print(f"Total bugs: {len(all_counts)}")
    print(f"Mean commits: {all_counts.mean():.2f}")
    print(f"Median commits: {int(pd.Series(all_counts).median())}")
    print(f"Min commits: {all_counts.min()}")
    print(f"Max commits: {all_counts.max()}")

    # Distribution
    count_dist = Counter(all_counts)
    print("\nDistribution:")
    for count in sorted(count_dist.keys()):
        num = count_dist[count]
        pct = num / len(all_counts) * 100
        print(f"  {count} commit(s): {num} bugs ({pct:.1f}%)")

    # Per-category statistics
    categories = ["SL", "SH", "SFMH", "MFMH"]
    category_stats = {}

    for category in categories:
        cat_df = df[df["Category"] == category]
        if len(cat_df) == 0:
            continue

        # Count blameable (>=1 commit) vs blameless (0 commits)
        blameable = len(cat_df[cat_df["Unique_Commit_Count"] >= 1])
        blameless = len(cat_df[cat_df["Unique_Commit_Count"] == 0])
        total = len(cat_df)

        category_stats[category] = {
            "blameable": blameable,
            "blameless": blameless,
            "total": total,
        }

        print(f"\n[{category}]")
        cat_counts = cat_df["Unique_Commit_Count"].values
        print(f"Total bugs: {len(cat_counts)}")
        print(f"Blameable: {blameable} ({blameable / total * 100:.1f}%)")
        print(f"Blameless: {blameless} ({blameless / total * 100:.1f}%)")
        print(f"Mean commits: {cat_counts.mean():.2f}")

        count_dist = Counter(cat_counts)
        print("Distribution:")
        for count in sorted(count_dist.keys()):
            num = count_dist[count]
            pct = num / len(cat_counts) * 100
            print(f"  {count} commit(s): {num} bugs ({pct:.1f}%)")

    # Generate figures
    print(f"\n{'=' * 80}")
    print("Generating figures...")
    print("=" * 80)

    # 1. Stacked bar chart (matching Defects4J figure)
    _create_stacked_distribution_figure(df, categories, output_dir)

    # 2. Grouped bar chart
    _create_grouped_distribution_figure(df, categories, output_dir)

    # 3. LaTeX table (matching Defects4J table format)
    _generate_latex_table(df, categories, category_stats, output_dir)

    print(f"{'=' * 80}")


def _create_stacked_distribution_figure(df, categories, output_dir):
    """Create a stacked bar chart showing all 4 categories with commit count groups."""
    import matplotlib.pyplot as plt
    import numpy as np

    # Group commit counts: 0, 1, 2, 3, ≥4 (finer granularity than Defects4J)
    commit_labels = ["0", "1", "2", "3", "≥4"]

    # Prepare data for each category
    category_data = {}
    for category in categories:
        cat_df = df[df["Category"] == category]
        if len(cat_df) == 0:
            category_data[category] = [0, 0, 0, 0, 0]
            continue

        count_0 = len(cat_df[cat_df["Unique_Commit_Count"] == 0])
        count_1 = len(cat_df[cat_df["Unique_Commit_Count"] == 1])
        count_2 = len(cat_df[cat_df["Unique_Commit_Count"] == 2])
        count_3 = len(cat_df[cat_df["Unique_Commit_Count"] == 3])
        count_4_plus = len(cat_df[cat_df["Unique_Commit_Count"] >= 4])

        category_data[category] = [count_0, count_1, count_2, count_3, count_4_plus]

    # Set up the figure
    fig, ax = plt.subplots(figsize=(10, 6))

    # Color palette for commit groups (5 colors: light to dark with distinct ≥4)
    colors = ["#E8F5E9", "#A5D6A7", "#66BB6A", "#43A047", "#D32F2F"]

    # Position for bars
    x = np.arange(len(categories))
    width = 0.6

    # Create stacked bars
    bottom = np.zeros(len(categories))
    max_annotation_height = 0

    for i, (label, color) in enumerate(zip(commit_labels, colors)):
        heights = [category_data[cat][i] for cat in categories]
        bars = ax.bar(
            x,
            heights,
            width,
            label=label,
            color=color,
            bottom=bottom,
            alpha=0.9,
            edgecolor="white",
            linewidth=1.5,
        )

        # Add value labels inside bars
        for j, (bar, height) in enumerate(zip(bars, heights)):
            if height > 0:
                y_pos = bottom[j] + height / 2
                # For small segments in ≥4 group, use arrow annotation
                if i == 4 and height < 10:
                    annotation_y = bottom[j] + height + 15
                    max_annotation_height = max(max_annotation_height, annotation_y + 5)
                    ax.annotate(
                        f"{int(height)}",
                        xy=(bar.get_x() + bar.get_width() / 2.0, bottom[j] + height),
                        xytext=(bar.get_x() + bar.get_width() / 2.0, annotation_y),
                        ha="center",
                        va="bottom",
                        fontsize=10,
                        color="#D32F2F",
                        arrowprops=dict(arrowstyle="->", color="#D32F2F", lw=1.5),
                    )
                else:
                    ax.text(
                        bar.get_x() + bar.get_width() / 2.0,
                        y_pos,
                        f"{int(height)}",
                        ha="center",
                        va="center",
                        fontsize=10 if height < 15 else 11,
                        color="white" if i == 4 else "black",
                    )

        bottom += heights

    # Customize the plot
    ax.set_xlabel("Bug Category", fontsize=13)
    ax.set_ylabel("Number of Bugs", fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=12)
    ax.legend(
        title="Unique Blame Commits",
        loc="upper right",
        fontsize=11,
        title_fontsize=11,
        framealpha=0.95,
    )
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.set_axisbelow(True)

    y_max = max(max(bottom), max_annotation_height)
    ax.set_ylim(bottom=0, top=y_max * 1.02)

    plt.tight_layout()
    output_path = output_dir / "rq0_bugsinpy_blame_commit_distribution_stacked.pdf"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"  Saved: {output_path}")
    plt.close()


def _create_grouped_distribution_figure(df, categories, output_dir):
    """Create a grouped bar chart showing all 4 categories together."""
    import matplotlib.pyplot as plt
    import numpy as np

    commit_groups = ["0", "1", "2", "3", "≥4"]

    category_data = {}
    for category in categories:
        cat_df = df[df["Category"] == category]
        if len(cat_df) == 0:
            continue

        count_0 = len(cat_df[cat_df["Unique_Commit_Count"] == 0])
        count_1 = len(cat_df[cat_df["Unique_Commit_Count"] == 1])
        count_2 = len(cat_df[cat_df["Unique_Commit_Count"] == 2])
        count_3 = len(cat_df[cat_df["Unique_Commit_Count"] == 3])
        count_4_plus = len(cat_df[cat_df["Unique_Commit_Count"] >= 4])

        category_data[category] = [count_0, count_1, count_2, count_3, count_4_plus]

    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(commit_groups))
    width = 0.2
    multiplier = 0

    colors = {"SL": "#70AD47", "SH": "#B19CD9", "SFMH": "#5B9BD5", "MFMH": "#ED7D31"}

    for category in categories:
        if category not in category_data:
            continue

        frequencies = category_data[category]
        offset = width * multiplier
        bars = ax.bar(
            x + offset,
            frequencies,
            width,
            label=category,
            color=colors.get(category, "#4A90E2"),
            alpha=0.9,
        )

        for bar, freq in zip(bars, frequencies):
            height = bar.get_height()
            y_pos = height if height > 0 else 1
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                y_pos,
                f"{int(freq)}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

        multiplier += 1

    ax.set_xlabel("Number of Unique Blame Commits", fontsize=13)
    ax.set_ylabel("Number of Bugs", fontsize=13)
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(commit_groups)
    ax.legend(loc="upper right", fontsize=11)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    output_path = output_dir / "rq0_bugsinpy_blame_commit_distribution_grouped.pdf"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"  Saved: {output_path}")
    plt.close()


def _generate_latex_table(df, categories, category_stats, output_dir):
    """Generate LaTeX table matching Defects4J format."""

    # Calculate totals
    total_blameable = sum(s["blameable"] for s in category_stats.values())
    total_blameless = sum(s["blameless"] for s in category_stats.values())
    total_all = total_blameable + total_blameless

    latex = r"""\begin{table}[t]
    \centering
    \caption{Blame availability by bug category in BugsInPy, assuming perfect fault localization. Percentages in the Blameable or Blameless columns are within-category; Total shows the share relative to the full dataset.}
    \label{tab:bugsinpy-distribution}
    \begin{tabular}{lrrrrrr}
    \toprule
    \multirow{2}{*}{\textbf{Category}}
      & \multicolumn{2}{c}{\textbf{Blameable}}
      & \multicolumn{2}{c}{\textbf{Blameless}}
      & \multicolumn{2}{c}{\textbf{Total}} \\
    \cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}
      & \# & \% & \# & \% & \# & \% \\
    \midrule
"""

    for category in categories:
        if category not in category_stats:
            continue
        stats = category_stats[category]
        blameable = stats["blameable"]
        blameless = stats["blameless"]
        total = stats["total"]

        blameable_pct = blameable / total * 100 if total > 0 else 0
        blameless_pct = blameless / total * 100 if total > 0 else 0
        total_pct = total / total_all * 100 if total_all > 0 else 0

        latex += f"    {category:4s} & {blameable:3d} & {blameable_pct:4.1f} & {blameless:3d} & {blameless_pct:4.1f} & {total:3d} & {total_pct:4.1f} \\\\\n"

    blameable_total_pct = total_blameable / total_all * 100 if total_all > 0 else 0
    blameless_total_pct = total_blameless / total_all * 100 if total_all > 0 else 0

    latex += (
        r"""    \midrule
    \textbf{Total} & \textbf{"""
        + str(total_blameable)
        + r"""} & \textbf{"""
        + f"{blameable_total_pct:.1f}"
        + r"""} & \textbf{"""
        + str(total_blameless)
        + r"""} & \textbf{"""
        + f"{blameless_total_pct:.1f}"
        + r"""} & \textbf{"""
        + str(total_all)
        + r"""} & \textbf{100.0} \\
    \bottomrule
    \end{tabular}
\end{table}
"""
    )

    output_path = output_dir / "rq0_bugsinpy_blame_distribution_table.tex"
    with open(output_path, "w") as f:
        f.write(latex)
    print(f"  Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze unique blame commits for BugsInPy bugs."
    )
    parser.add_argument("--output", "-o", default=None, help="Output CSV file path")
    parser.add_argument(
        "--repos-dir", default=None, help="Directory for cloned repositories"
    )
    parser.add_argument(
        "--workers",
        "-w",
        type=int,
        default=1,
        help="Number of parallel workers (default: 1)",
    )
    parser.add_argument(
        "--projects",
        "-p",
        nargs="+",
        default=None,
        help="Specific projects to analyze (default: all)",
    )
    parser.add_argument(
        "--stats",
        "-s",
        action="store_true",
        help="Generate statistics from existing CSV file",
    )

    args = parser.parse_args()

    if args.stats:
        csv_file = args.output or str(
            Path(__file__).parent.parent
            / "results"
            / "rq0_bugsinpy"
            / "bugsinpy_blame_commit_counts.csv"
        )
        print_statistics_from_csv(csv_file)
    else:
        repos_dir = Path(args.repos_dir) if args.repos_dir else None
        analyze_blame_commits(
            output_file=args.output,
            repos_dir=repos_dir,
            workers=args.workers,
            projects=args.projects,
        )


if __name__ == "__main__":
    main()
