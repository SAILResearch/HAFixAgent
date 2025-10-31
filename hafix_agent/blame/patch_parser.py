"""
Dataset-agnostic patch parsing to extract blamable lines from code files.

This module contains the core functionality to parse unified diff patches
and extract lines that can be used for git blame analysis across different datasets.

Supported patch formats:
- Defects4J: REVERSE patches (fixed -> buggy) - '+' lines are blamable
- SWE-Bench: FORWARD patches (buggy -> fixed) - '-' lines are blamable
"""

import re
from dataclasses import dataclass
from typing import List
from pathlib import Path
from enum import Enum


class PatchFormat(Enum):
    """
    Patch direction format for blame analysis.

    - REVERSE: Patch shows changes from fixed → buggy version
              ('+' lines exist in buggy code, can be blamed)
              Example: Defects4J .src.patch files

    - FORWARD: Patch shows changes from buggy → fixed version
              ('-' lines exist in buggy code, can be blamed)
              Example: Standard diff output, SWE-bench patches
    """
    REVERSE = "reverse"  # Fixed -> buggy (+ lines are buggy)
    FORWARD = "forward"  # Buggy -> fixed (- lines are buggy)

    # Backward compatibility aliases (deprecated)
    DEFECTS4J = "reverse"  # Alias for REVERSE
    SWE_BENCH = "forward"  # Alias for FORWARD


@dataclass(frozen=True)
class PatchLine:
    """Represents a single changed line in a patch."""
    file_path: str
    line_number: int  # Line number in respective file (old file for '-', new file for '+')
    content: str
    change_type: str  # "+", "-", " " (addition, deletion, context)
    patch_format: PatchFormat  # Dataset format to determine blamable logic
    
    def can_blame(self) -> bool:
        """Check if this line can be blamed in the buggy commit.

        Logic depends on patch direction:
        - REVERSE (fixed→buggy): '+' lines exist in buggy version → blamable
        - FORWARD (buggy→fixed): '-' lines exist in buggy version → blamable
        - Pre-hunk fallback lines are always blamable (exist in buggy version)
        """
        if self.change_type == "pre_hunk":
            return True  # Pre-hunk lines always exist in buggy version

        # Normalize to canonical values for comparison
        patch_value = self.patch_format.value

        if patch_value == "reverse":  # REVERSE or DEFECTS4J alias
            return self.change_type == "+"  # Reverse patch: + lines are buggy
        elif patch_value == "forward":  # FORWARD or SWE_BENCH alias
            return self.change_type == "-"  # Forward patch: - lines are buggy
        else:
            raise ValueError(f"Unsupported patch format: {self.patch_format}")
    
    def __str__(self) -> str:
        """
        Nice way to display for print() and f""
        """
        return f"{self.change_type} Line {self.line_number} in {self.file_path}: {self.content.strip()}"


class PatchParser:
    """Dataset-agnostic parser for unified diff patches."""

    def __init__(self, patch_format: PatchFormat = PatchFormat.REVERSE):
        """
        Initialize parser with patch format.

        Args:
            patch_format: Patch direction (REVERSE: fixed→buggy, FORWARD: buggy→fixed)
        """
        self.patch_format = patch_format
        # Only process Python and Java files (for SWE-Bench and Defects4J)
        self.code_extensions = {'.py', '.java'}
    
    def extract_code_file_lines(self, patch_content: str) -> List[PatchLine]:
        """
        Extract all changed lines from code files in the patch.

        Returns:
            List of PatchLine objects for all changed lines in code files
        """
        if not patch_content:
            return []
        
        lines = patch_content.split('\n')
        patch_lines = []
        
        current_file = None
        current_old_line = 0
        current_new_line = 0
        
        for line in lines:
            # Parse file header: --- a/path/to/file.py
            if line.startswith('--- a/'):
                current_file = line[6:]  # Remove '--- a/' prefix
                continue
            elif line.startswith('--- /dev/null'):
                current_file = None  # New file, skip
                continue
            elif line.startswith('+++ '):
                continue  # Skip +++ lines
            
            # Parse hunk header: @@ -start,count +start,count @@
            if line.startswith('@@'):
                if current_file and self._is_code_file(current_file):
                    hunk_match = re.match(r'@@\s*-(\d+)(?:,\d+)?\s*\+(\d+)(?:,\d+)?\s*@@', line)
                    if hunk_match:
                        current_old_line = int(hunk_match.group(1))
                        current_new_line = int(hunk_match.group(2))
                continue
            
            # Skip if not in a code file
            if not current_file or not self._is_code_file(current_file):
                continue
            
            # Process change lines
            if line.startswith(' '):
                # Context line - exists in both old and new files
                current_old_line += 1
                current_new_line += 1
            elif line.startswith('-'):
                # Deletion line - exists in old file only
                patch_lines.append(PatchLine(
                    file_path=current_file,
                    line_number=current_old_line,
                    content=line[1:],  # Remove '-' prefix
                    change_type='-',
                    patch_format=self.patch_format
                ))
                current_old_line += 1
            elif line.startswith('+'):
                # Addition line - exists in new file only
                patch_lines.append(PatchLine(
                    file_path=current_file,
                    line_number=current_new_line,
                    content=line[1:],  # Remove '+' prefix
                    change_type='+',
                    patch_format=self.patch_format
                ))
                current_new_line += 1
        
        return patch_lines
    
    @staticmethod
    def get_blamable_lines(patch_lines: List[PatchLine]) -> List[PatchLine]:
        """Filter to only lines that can be blamed (deletions)."""
        return [line for line in patch_lines if line.can_blame()]

    def extract_pre_hunk_blameable_lines(self, patch_content: str, docker_env, work_dir: str) -> List[PatchLine]:
        """
        For blameless patches, extract lines immediately before the first actual change as fallback blame targets.

        This method identifies the first actual change line (+ or -) in each hunk and attempts to blame
        the line just before that first change.

        Args:
            patch_content: Patch content to analyze
            docker_env: Docker environment for file access
            work_dir: Working directory in container

        Returns:
            List of PatchLine objects representing pre-hunk lines that can be blamed
        """
        if not patch_content:
            return []

        lines = patch_content.split('\n')
        pre_hunk_lines = []

        current_file = None
        current_old_line = 0
        in_hunk = False

        for i, line in enumerate(lines):
            # Parse file header: --- a/path/to/file.py
            if line.startswith('--- a/'):
                current_file = line[6:]  # Remove '--- a/' prefix
                continue
            elif line.startswith('--- /dev/null'):
                current_file = None  # New file, skip
                continue
            elif line.startswith('+++ '):
                continue  # Skip +++ lines

            # Parse hunk header: @@ -start,count +start,count @@
            if line.startswith('@@'):
                if current_file and self._is_code_file(current_file):
                    hunk_match = re.match(r'@@\s*-(\d+)(?:,\d+)?\s*\+(\d+)(?:,\d+)?\s*@@', line)
                    if hunk_match:
                        current_old_line = int(hunk_match.group(1))
                        in_hunk = True
                continue

            # Skip if not in a code file or not in a hunk
            if not current_file or not self._is_code_file(current_file) or not in_hunk:
                continue

            # Process hunk content to find first change
            if line.startswith(' '):
                # Context line - exists in both old and new files
                current_old_line += 1
            elif line.startswith('-') or line.startswith('+'):
                # Found first actual change! Search for the best available pre-hunk line
                best_line = self._find_best_pre_hunk_line(
                    docker_env, work_dir, current_file, current_old_line
                )

                if best_line:
                    pre_hunk_lines.append(best_line)

                # Reset for next hunk (only get one pre-hunk line per hunk)
                in_hunk = False
                continue

        return pre_hunk_lines

    def _find_best_pre_hunk_line(self, docker_env, work_dir: str, current_file: str,
                                current_old_line: int) -> PatchLine:
        """
        Find the best available pre-hunk line using tiered fallback strategy.

        Priority order:
        1. Substantive code lines (method calls, declarations, etc.)
        2. Structural boundaries ({, }, ;)
        3. Skip formatting artifacts ((, ))
        4. Skip blank/empty lines

        Args:
            docker_env: Docker environment instance
            work_dir: Working directory in container
            current_file: File path relative to work_dir
            current_old_line: Line number where first change occurs

        Returns:
            Best available PatchLine or None if no suitable line found
        """
        fallback_line = None

        # Search up to 5 lines before the first change
        for offset in range(1, 6):
            candidate_line_num = current_old_line - offset
            if candidate_line_num <= 0:
                break

            # Get line content from buggy file
            line_content = self._get_line_content_from_buggy_file(
                docker_env, work_dir, current_file, candidate_line_num
            )

            if not line_content:
                continue

            # Categorize the line
            if self._is_substantive_code_line(line_content):
                # Best case - found substantive code, return immediately
                return PatchLine(
                    file_path=current_file,
                    line_number=candidate_line_num,
                    content=line_content,
                    change_type="pre_hunk",
                    patch_format=self.patch_format
                )
            elif self._is_structural_line(line_content):
                # Good fallback - keep the first structural line found
                if not fallback_line:
                    fallback_line = PatchLine(
                        file_path=current_file,
                        line_number=candidate_line_num,
                        content=line_content,
                        change_type="pre_hunk",
                        patch_format=self.patch_format
                    )
            # Skip formatting artifacts and empty lines (continue searching)

        return fallback_line

    def _is_substantive_code_line(self, line_content: str) -> bool:
        """Check if line contains substantive code (not just structural elements)."""
        stripped = line_content.strip()

        # Skip empty lines
        if not stripped:
            return False

        # Skip comment-only lines
        if (stripped.startswith('//') or stripped.startswith('/*') or
            stripped.startswith('*') or stripped.startswith('#')):
            return False

        # Skip pure structural/formatting lines
        if stripped in ['{', '}', ';', '(', ')', 'else {', '} else {']:
            return False

        # Accept substantive code lines
        return True

    def _is_structural_line(self, line_content: str) -> bool:
        """Check if line contains structural boundaries (fallback option)."""
        stripped = line_content.strip()

        # Accept structural boundaries but not formatting artifacts
        return stripped in ['{', '}', ';', 'else {', '} else {']

    def _get_line_content_from_buggy_file(self, docker_env, work_dir: str,
                                         filepath: str, line_number: int) -> str:
        """
        Get the content of a specific line from the buggy version of a file.

        Args:
            docker_env: Docker environment instance
            work_dir: Working directory in container
            filepath: File path relative to work_dir
            line_number: Line number to extract (1-based)

        Returns:
            Line content or empty string if failed
        """
        try:
            # Use sed to extract specific line from file in buggy version
            cmd = f"cd {work_dir} && sed -n '{line_number}p' {filepath}"
            result = docker_env.execute(cmd)

            if result['returncode'] == 0 and result['output'].strip():
                return result['output'].strip()
            else:
                return ""

        except Exception as e:
            print(f"Error getting line content from {filepath}:{line_number}: {e}")
            return ""

    def _is_suitable_blame_line(self, line_content: str) -> bool:
        """
        Check if a line is suitable for blame analysis.

        Args:
            line_content: Content of the line

        Returns:
            True if line is suitable for blame (not comment, not whitespace, etc.)
        """
        stripped = line_content.strip()

        # Skip empty lines
        if not stripped:
            return False

        # Skip comment-only lines (Java and Python)
        if (stripped.startswith('//') or  # Java single-line comment
            stripped.startswith('/*') or  # Java multi-line comment start
            stripped.startswith('*') or   # Java multi-line comment continuation
            stripped.startswith('#')):    # Python comment
            return False

        # Skip lines that are only braces or semicolons
        if stripped in ['{', '}', ';', '(', ')']:
            return False

        # Accept any other non-empty code line
        return True
    
    def _is_code_file(self, filepath: str) -> bool:
        """Check if file is a code file we want to process."""
        try:
            extension = Path(filepath).suffix.lower()
            return extension in self.code_extensions
        except:
            return False