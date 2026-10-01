"""
BugsInPy blame context extraction and bug information.

Implements BlameExtractor and BugInfoExtractor interfaces for BugsInPy dataset.
Uses Docker containers with pre-warmed images for fast execution.
"""

import csv
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional

from hafix_agent.blame.interface import BlameExtractor, BugInfoExtractor
from hafix_agent.blame.core import extract_blame_context
from hafix_agent.blame.patch_parser import PatchParser, PatchFormat

from .util import (
    get_all_bugs,
    get_bug_info,
    get_patch_file_path,
    get_project_info,
    BUGSINPY_BASE_PATH,
)

project_root = Path(__file__).parent.parent.parent


def get_bugsinpy_work_dir(project_name: str, bug_id: str) -> str:
    """Get the working directory path inside BugsInPy Docker container."""
    return f"/BugsInPy/framework/bin/temp/{project_name}_{bug_id}"


def ensure_bugsinpy_docker_container(
    project_name: str,
    bug_id: str,
    docker_env=None,
    image: str = None,
    use_existing_container: str = None,
    cleanup_on_exit: bool = True,
):
    """
    Ensure a BugsInPy Docker container is ready with the bug checked out.

    Uses pre-warmed images (bugsinpy:{project}_{bug_id}) if available,
    falls back to base image with checkout+compile.

    Returns:
        Tuple of (docker_env, error_dict or None)
    """
    if docker_env is not None:
        return docker_env, None

    from hafix_agent.environments.bugsinpy_docker import BugsInPyDocker

    # Try pre-warmed image first
    if image is None:
        prewarmed = f"bugsinpy:{project_name.lower()}_{bug_id}"
        import subprocess
        check = subprocess.run(
            ["docker", "image", "inspect", prewarmed],
            capture_output=True, text=True, timeout=10,
        )
        if check.returncode == 0:
            image = prewarmed
        else:
            image = "bugsinpy_image:clean"

    try:
        docker_env = BugsInPyDocker(
            use_existing_container=use_existing_container,
            cleanup_on_exit=cleanup_on_exit,
            image=image,
        )

        # Pre-warmed images already have checkout+compile done
        if "bugsinpy_image:clean" in image:
            work_dir = get_bugsinpy_work_dir(project_name, bug_id)
            checkout_result = docker_env.checkout_bug(project_name, int(bug_id), work_dir)
            if checkout_result['returncode'] != 0:
                return None, {
                    "bug_info": None, "blame_info": None,
                    "error": f"Checkout failed: {checkout_result['output']}"
                }
            compile_result = docker_env.compile_bug(work_dir)
            if compile_result['returncode'] != 0:
                return None, {
                    "bug_info": None, "blame_info": None,
                    "error": f"Compile failed: {compile_result['output']}"
                }

        return docker_env, None

    except Exception as e:
        return None, {
            "bug_info": None, "blame_info": None,
            "error": f"Docker setup failed: {str(e)}"
        }


class BugsInPyExtractor(BlameExtractor, BugInfoExtractor):
    """BugsInPy implementation of BlameExtractor and BugInfoExtractor."""

    def __init__(self):
        self._categorization = None

    def get_bug_category(self, project_name: str, bug_id: int) -> Optional[str]:
        """Get bug category from pre-computed CSV."""
        if self._categorization is None:
            csv_path = Path(__file__).parent / "bugsinpy_blame_feasibility.csv"
            if csv_path.exists():
                self._categorization = {}
                with open(csv_path) as f:
                    for row in csv.DictReader(f):
                        key = row.get('Bug_ID', '')
                        self._categorization[key] = row.get('Category', '')
            else:
                self._categorization = {}

        return self._categorization.get(f"{project_name}_{bug_id}")

    def get_all_bug_ids(self) -> List[str]:
        """Get all BugsInPy bug IDs."""
        all_bugs = get_all_bugs()
        return [f"{proj}_{bid}" for proj, bids in all_bugs.items() for bid in bids]

    def _extract_blame_context(
        self,
        project_name: str,
        bug_id: str,
        selector_type: str = "first",
        n_lines: int = 1,
        docker_env=None,
        image: str = None,
        use_existing_container: str = None,
        cleanup_on_exit: bool = True,
        bug_info: Dict = None,
        include_blameless: bool = True,
        **kwargs
    ) -> Dict:
        """Extract blame context for a BugsInPy bug."""
        try:
            docker_env, error_result = ensure_bugsinpy_docker_container(
                project_name, bug_id, docker_env, image, use_existing_container, cleanup_on_exit
            )
            if error_result:
                return error_result

            work_dir = get_bugsinpy_work_dir(project_name, bug_id)
            # BugsInPy checkout creates {work_dir}/{project}/ — git repo is inside
            repo_dir = f"{work_dir}/{project_name}"

            # Get buggy commit for checkout
            meta = get_bug_info(project_name, int(bug_id))
            buggy_commit = meta.get("buggy_commit_id", "")

            if not buggy_commit:
                return {"bug_info": None, "blame_info": None,
                        "error": f"No buggy_commit_id for {project_name}_{bug_id}"}

            # Checkout the buggy commit for blame (pre-warmed images are already at buggy version)
            # For blame we need to be at the buggy commit with full git history
            git_checkout = docker_env.execute(f"cd {repo_dir} && git checkout {buggy_commit}")

            # Read patch content
            patch_path = get_patch_file_path(project_name, int(bug_id))
            patch_content = patch_path.read_text(encoding="utf-8", errors="replace")

            # Run blame extraction (forward patch for BugsInPy)
            blame_info = extract_blame_context(
                patch_content=patch_content,
                docker_env=docker_env,
                work_dir=repo_dir,
                patch_format=PatchFormat.FORWARD,
                selector_type=selector_type,
                n_lines=n_lines,
                buggy_files=None,
                bug_info=bug_info,
                include_blameless=include_blameless,
                model_config=kwargs.get('model_config')
            )

            return {
                "bug_info": {"project": project_name, "bug_id": f"{project_name}_{bug_id}", "container_ready": True},
                "blame_info": blame_info,
                "docker_env": docker_env,
            }

        except Exception as e:
            return {"bug_info": None, "blame_info": None,
                    "error": f"Blame extraction failed: {str(e)}"}

    def _extract_bug_info_context(
        self,
        project_name: str,
        bug_id: str,
        docker_env=None,
        image: str = None,
        use_existing_container: str = None,
        cleanup_on_exit: bool = True,
        **kwargs
    ) -> Dict[str, Any]:
        """Extract bug information for BugsInPy."""
        try:
            docker_env, error = ensure_bugsinpy_docker_container(
                project_name, bug_id, docker_env, image, use_existing_container, cleanup_on_exit
            )
            if error:
                work_dir = get_bugsinpy_work_dir(project_name, bug_id)
                return {
                    'project': project_name,
                    'bug_id': f"{project_name}_{bug_id}",
                    'repo_path': f"{work_dir}/{project_name}",
                    'error': error.get('error', 'Unknown error')
                }

            work_dir = get_bugsinpy_work_dir(project_name, bug_id)
            # BugsInPy checkout creates {work_dir}/{project}/ — the actual git repo
            repo_dir = f"{work_dir}/{project_name}"

            # Parse metadata
            meta = get_bug_info(project_name, int(bug_id))

            # Extract fault locations from patch, enriched with function_name from source
            patch_path = get_patch_file_path(project_name, int(bug_id))
            fault_locations = self._extract_fault_locations(patch_path, docker_env, repo_dir)

            # Get failing tests from run_test.sh
            failing_tests = self._extract_failing_tests(project_name, int(bug_id))

            # Get bug category
            bug_category = self.get_bug_category(project_name, int(bug_id))

            # Get project GitHub URL
            project_info = get_project_info(project_name)
            github_url = project_info.get("github_url", "")

            # Extract commit message (stored in result for analysis, not injected into prompt)
            commit_message = ""
            fixed_commit = meta.get("fixed_commit_id", "")
            if docker_env and fixed_commit:
                try:
                    msg_result = docker_env.execute(
                        f"cd {repo_dir} && git log --format=%s -n 1 {fixed_commit}"
                    )
                    if msg_result.get('returncode') == 0:
                        commit_message = msg_result.get('output', '').strip()
                except Exception:
                    pass

            return {
                'project': project_name,
                'bug_id': f"{project_name}_{bug_id}",
                'repo_path': repo_dir,  # Actual git repo, not parent work_dir
                'github_url': github_url,
                'description': '',  # Empty: no bug report for BugsInPy, avoid leakage from commit message
                'commit_message': commit_message,  # For result analysis only, not in prompt
                'fault_locations': fault_locations,
                'failing_tests': failing_tests,
                'bug_category': bug_category,
                'golden_patch': patch_path.read_text(encoding="utf-8", errors="replace") if patch_path.exists() else "",
            }

        except Exception as e:
            return {
                'project': project_name,
                'bug_id': f"{project_name}_{bug_id}",
                'repo_path': f"{get_bugsinpy_work_dir(project_name, bug_id)}/{project_name}",
                'error': str(e)
            }

    def _extract_fault_locations(self, patch_path: Path, docker_env=None, repo_dir: str = None) -> List[Dict]:
        """Extract fault locations from BugsInPy patch file.

        If docker_env and repo_dir are provided, enriches locations with
        function_name by reading source files from the container.
        """
        if not patch_path.exists():
            return []

        from hafix_agent.blame.core import find_function_containing_line

        patch_content = patch_path.read_text(encoding="utf-8", errors="replace")
        parser = PatchParser(patch_format=PatchFormat.FORWARD)
        all_lines = parser.extract_code_file_lines(patch_content)
        blamable = parser.get_blamable_lines(all_lines)

        # Group consecutive lines by file into fault locations
        locations = []
        current_file = None
        current_start = None
        current_end = None

        for line in sorted(blamable, key=lambda l: (l.file_path, l.line_number)):
            if line.file_path != current_file or (current_end and line.line_number > current_end + 1):
                if current_file:
                    locations.append({
                        'file': current_file,
                        'start_line': current_start,
                        'end_line': current_end,
                        'function_name': '',
                        'class_name': '',
                    })
                current_file = line.file_path
                current_start = line.line_number
                current_end = line.line_number
            else:
                current_end = line.line_number

        if current_file:
            locations.append({
                'file': current_file,
                'start_line': current_start,
                'end_line': current_end,
                'function_name': '',
                'class_name': '',
            })

        # For blameless bugs (pure additions), extract insertion points
        if not locations:
            addition_lines = [l for l in all_lines if l.change_type == "+"]
            if addition_lines:
                for line in addition_lines:
                    locations.append({
                        'file': line.file_path,
                        'start_line': max(1, line.line_number - 1),
                        'end_line': line.line_number,
                        'function_name': '',
                        'class_name': '',
                    })

        # Enrich with function_name from source if Docker is available
        if docker_env and repo_dir:
            source_cache = {}
            for loc in locations:
                filepath = loc['file']
                if filepath not in source_cache:
                    result = docker_env.execute(f"cat {repo_dir}/{filepath}")
                    source_cache[filepath] = result.get('output', '') if result.get('returncode') == 0 else ''

                source = source_cache[filepath]
                if source:
                    from pathlib import Path as _P
                    ext = _P(filepath).suffix
                    func_name = find_function_containing_line(source, loc['start_line'], ext)
                    if func_name:
                        loc['function_name'] = func_name

        return locations

    def _extract_failing_tests(self, project_name: str, bug_id: int) -> List[str]:
        """Extract failing test names from run_test.sh."""
        run_test_path = BUGSINPY_BASE_PATH / project_name / "bugs" / str(bug_id) / "run_test.sh"
        if not run_test_path.exists():
            return []

        tests = []
        with open(run_test_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    tests.append(line)
        return tests
