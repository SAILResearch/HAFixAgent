"""
HAFixAgent Blame Module - Core Contribution

This module contains the core blame-based historical context extraction functionality,
which is the primary contribution of HAFixAgent for improving automated program repair
through historical information from git blame.

Modules:
- core: Dataset-agnostic git blame utilities and history extraction
- patch_parser: Parse unified diff patches to extract blameable lines
- selection: Strategies for selecting which lines to blame (first, random, LLM-guided)
"""

from .core import (
    run_git_blame,
    get_commit_patch,
    get_changed_files_in_commit,
    extract_function_names_from_code,
    get_file_content_at_commit,
    get_changed_functions_in_commit,
    extract_all_history_context,
    extract_blame_context,
    extract_function_code_pairs,
)

# Import multi-line selection components
from .patch_parser import PatchParser, PatchLine
from .selection import get_selector, BlameLineSelector, FirstSelector, RandomSelector, LLMJudgeSelector


__all__ = [
    'run_git_blame',
    'get_commit_patch',
    'get_changed_files_in_commit',
    'extract_function_names_from_code',
    'get_file_content_at_commit',
    'get_changed_functions_in_commit',
    'extract_all_history_context',
    'extract_blame_context',
    'extract_function_code_pairs',
    # Multi-line selection components
    'PatchParser',
    'PatchLine',
    'get_selector',
    'BlameLineSelector',
    'FirstSelector',
    'RandomSelector',
    'LLMJudgeSelector',
]

__version__ = "1.0.0"
__author__ = "HAFixAgent"