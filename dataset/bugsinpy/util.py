"""Utility functions for BugsInPy dataset."""

import os
from pathlib import Path
from typing import Dict, List, Optional


# Base paths
BASE_DIR = Path(__file__).resolve().parents[2]
BUGSINPY_BASE_PATH = BASE_DIR / "vendor" / "BugsInPy" / "projects"


def get_all_bugs(base_path: Optional[Path] = None) -> Dict[str, List[int]]:
    """
    Discover all projects and bug IDs by scanning BugsInPy directory structure.

    Args:
        base_path: Path to BugsInPy projects directory. Defaults to vendor/BugsInPy/projects.

    Returns:
        Dict mapping project names to list of bug IDs.
        Example: {'pandas': [1, 2, ..., 169], 'thefuck': [1, 2, ...], ...}
    """
    if base_path is None:
        base_path = BUGSINPY_BASE_PATH

    projects_bugs = {}

    for project_name in os.listdir(base_path):
        project_path = base_path / project_name
        if not project_path.is_dir():
            continue

        bugs_dir = project_path / "bugs"
        if not bugs_dir.exists():
            continue

        bug_ids = []
        for bug_dir in os.listdir(bugs_dir):
            bug_path = bugs_dir / bug_dir
            if bug_path.is_dir() and bug_dir.isdigit():
                # Verify it has required files
                if (bug_path / "bug.info").exists() and (
                    bug_path / "bug_patch.txt"
                ).exists():
                    bug_ids.append(int(bug_dir))

        if bug_ids:
            projects_bugs[project_name] = sorted(bug_ids)

    return projects_bugs


def get_bug_info(
    project_name: str, bug_id: int, base_path: Optional[Path] = None
) -> Dict:
    """
    Parse bug.info file to get bug metadata.

    Args:
        project_name: Name of the BugsInPy project
        bug_id: Bug ID number
        base_path: Path to BugsInPy projects directory

    Returns:
        Dict with buggy_commit_id, fixed_commit_id, python_version, test_file
    """
    if base_path is None:
        base_path = BUGSINPY_BASE_PATH

    bug_info_path = base_path / project_name / "bugs" / str(bug_id) / "bug.info"

    info = {}
    with open(bug_info_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if "=" in line:
                key, value = line.split("=", 1)
                # Remove quotes from values
                value = value.strip().strip('"').strip("'")
                info[key.strip()] = value

    return info


def get_project_info(project_name: str, base_path: Optional[Path] = None) -> Dict:
    """
    Parse project.info file to get project metadata (GitHub URL, etc.).

    Args:
        project_name: Name of the BugsInPy project
        base_path: Path to BugsInPy projects directory

    Returns:
        Dict with github_url, status, etc.
    """
    if base_path is None:
        base_path = BUGSINPY_BASE_PATH

    project_info_path = base_path / project_name / "project.info"

    info = {}
    with open(project_info_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if "=" in line:
                key, value = line.split("=", 1)
                value = value.strip().strip('"').strip("'")
                info[key.strip()] = value

    return info


def get_patch_file_path(
    project_name: str, bug_id: int, base_path: Optional[Path] = None
) -> Path:
    """
    Get path to bug patch file.

    Args:
        project_name: Name of the BugsInPy project
        bug_id: Bug ID number
        base_path: Path to BugsInPy projects directory

    Returns:
        Path to bug_patch.txt file
    """
    if base_path is None:
        base_path = BUGSINPY_BASE_PATH

    return base_path / project_name / "bugs" / str(bug_id) / "bug_patch.txt"


def build_project_url_map(base_path: Optional[Path] = None) -> Dict[str, str]:
    """
    Build mapping of project names to GitHub URLs by parsing project.info files.

    Args:
        base_path: Path to BugsInPy projects directory

    Returns:
        Dict mapping project names to GitHub URLs
    """
    if base_path is None:
        base_path = BUGSINPY_BASE_PATH

    url_map = {}
    for project_name in os.listdir(base_path):
        project_path = base_path / project_name
        if project_path.is_dir() and (project_path / "project.info").exists():
            info = get_project_info(project_name, base_path)
            if "github_url" in info:
                url_map[project_name] = info["github_url"]

    return url_map


# Build URL map at module load time
bugsinpy_project_name_url_map = build_project_url_map()
