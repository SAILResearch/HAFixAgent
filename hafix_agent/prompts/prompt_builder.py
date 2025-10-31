"""
Build prompts for HAFixAgent with blame history integration using YAML templates.
Implements baseline vs history augmentation architecture.
"""

from typing import Dict, Optional
from ..blame.core import HistoryCategory

# History content truncation limits based on empirical token usage analysis
#
# Rationale for limits (for paper reference):
# - Analyzed actual token usage from DeepSeek-V3.1 evaluation results
# - Baseline bugs range 18K-500K tokens, history adds minimal overhead (1-5K tokens)
# - Model context: 64K tokens, leaving substantial budget for history augmentation
# - Function/file lists are high signal, low cost: ~4-5 chars per token
# - Code snippets often empty in blame results, but when present should be complete
# - Prioritize comprehensive function/file lists over theoretical code snippet space
# HISTORY_LIMITS = {
#     'code_before': 2000,        # ~400-500 tokens - complete methods when available
#     'code_after': 2000,         # ~400-500 tokens - complete methods when available
#     'patch': 3000,              # ~600-800 tokens - multiple complete hunks with context
#     'functions_focused': 200,    # ~800 tokens - co-evolved/modified functions (high signal)
#     'functions_comprehensive': 400,  # ~1600 tokens - all functions (comprehensive context)
#     'files': 200                # ~1200 tokens - co-evolved files (structural context)
# }

# adjusted after caching blame commit data
HISTORY_LIMITS = {
    'code_before': 1500,  # ↑ from 2000 (covers 95% of fn_pair cases)
    'code_after': 1500,  # ↑ from 2000 (covers 95% of fn_pair cases)
    'patch': 3000,  # = same (already sufficient for fl_diff)
    'functions_focused': 600,  # ↓ from 200 (over-provisioned!)
    'functions_comprehensive': 600,  # ↑ from 400 (under-provisioned)
    'files': 600  # ↓ from 200 (over-provisioned!)
}


def build_hafix_prompt(
        bug_info: Dict,
        config: Dict,
        history_category: HistoryCategory = HistoryCategory.baseline,
        blame_info: Optional[Dict] = None
) -> Dict:
    """
    Build dataset-agnostic prompts for HAFixAgent using YAML templates.

    Args:
        bug_info: Bug information with fault locations (dataset-agnostic format)
        config: Pre-loaded YAML config for the target dataset
        history_category: Which history augmentation to use (baseline = no history)
        blame_info: Blame commit information (only used if not baseline)

    Returns:
        Dictionary with system and instance prompts
    """
    
    # Extract agent configuration from dataset-specific config
    agent_config = config.get('agent', {})
    
    # Extract bug details for template variables
    bug_id = bug_info.get("bug_id", "Unknown")
    project = bug_info.get("project", "")
    fault_locations = bug_info.get("fault_locations", [])
    failing_tests = bug_info.get("failing_tests", [])
    description = bug_info.get("description", "")
    repo_path = bug_info.get("repo_path", "/workspace")
    bug_category = bug_info.get("bug_category", None)
    
    # Base template variables (always available)
    template_vars = {
        'bug_id': bug_id,
        'project': project,
        'repo_path': repo_path,
        'description': description,
        'fault_locations': fault_locations,
        'failing_tests': failing_tests,
        'is_multi_hunk': len(fault_locations) > 1,
        'bug_category': bug_category,  # Category passed from evaluation script
        'has_blame_info': False,  # Default for baseline
        'history_augmentation': '',  # Empty for baseline
        # Initialize execution tracking variables (updated dynamically during agent run)
        'current_failing_tests': [],
        'tests_fixed': False,
        'test_failures_count': 0,
        'compilation_failures': 0,
        'compilation_status': None,
    }
    
    # Conditional history augmentation (only if not baseline)
    if history_category != HistoryCategory.baseline and blame_info:
        template_vars['has_blame_info'] = True
        template_vars['history_augmentation'] = build_history_augmentation(history_category, blame_info)
    
    # Return raw template strings and template variables
    # The agent will render them using Jinja2 with the task description
    return {
        "system_template": agent_config.get('system_template', ''),
        "instance_template": agent_config.get('instance_template', ''),
        "template_vars": template_vars
    }


def build_history_augmentation(history_category: HistoryCategory, blame_info: Dict) -> str:
    """Build history augmentation section based on category."""

    augmentation = "\n# Historical Context from Git Blame\n"

    commit_info = blame_info.get('blame_commit', {}).get('commit', {})
    function_info = blame_info.get('blame_commit', {}).get('function', {})
    file_info = blame_info.get('blame_commit', {}).get('file', {})

    # Basic commit info
    augmentation += f"Commit Date: {commit_info.get('commit_date', 'Unknown')}\n"
    augmentation += f"Commit Message: {commit_info.get('commit_message', 'No message')}\n\n"

    if history_category == HistoryCategory.baseline_function_code_pair_blame:
        # Show before/after code from blame commit
        augmentation += "## Previous change at this location:\n"

        code_before = function_info.get('function_code_before', '')
        code_after = function_info.get('function_code_after', '')

        if code_before:
            augmentation += f"**Before:**\n```java\n{code_before[:HISTORY_LIMITS['code_before']]}\n```\n\n"
        if code_after:
            augmentation += f"**After:**\n```java\n{code_after[:HISTORY_LIMITS['code_after']]}\n```\n\n"

    elif history_category == HistoryCategory.baseline_file_code_patch_blame:
        # Show patch from blame commit
        file_patches = file_info.get('file_patches', {})
        if file_patches:
            # Get the first patch (there should typically be one for the blamed file)
            patch = next(iter(file_patches.values())) if file_patches else ''
            if patch:
                augmentation += f"## Historical diff patch:\n```diff\n{patch[:HISTORY_LIMITS['patch']]}\n```\n\n"

    elif history_category in [
        HistoryCategory.baseline_co_evolved_functions_name_modified_file_blame,
        HistoryCategory.baseline_co_evolved_functions_name_all_files_blame
    ]:
        # Show co-evolved functions
        if history_category == HistoryCategory.baseline_co_evolved_functions_name_modified_file_blame:
            functions = function_info.get('functions_name_co_evolved_modified_file', [])
            augmentation += f"## Co-evolved functions in same file:\n"
        else:
            functions = function_info.get('functions_name_co_evolved_all_files', [])
            augmentation += f"## Co-evolved functions across files:\n"

        for func in functions[:HISTORY_LIMITS['functions_focused']]:
            augmentation += f"- {func}\n"

    elif history_category in [
        HistoryCategory.baseline_all_functions_name_modified_file_blame,
        HistoryCategory.baseline_all_functions_name_all_files_blame
    ]:
        # Show all functions
        if history_category == HistoryCategory.baseline_all_functions_name_modified_file_blame:
            functions = function_info.get('functions_name_modified_file', [])
            augmentation += f"## All functions in file at blame commit:\n"
        else:
            functions = function_info.get('functions_name_all_files', [])
            augmentation += f"## All functions across files at blame commit:\n"

        for func in functions[:HISTORY_LIMITS['functions_comprehensive']]:
            augmentation += f"- {func}\n"

    elif history_category == HistoryCategory.baseline_all_co_evolved_files_name_blame:
        # Show co-evolved files
        files = file_info.get('files_name_in_blame_commit', [])
        augmentation += f"## Co-evolved files in blame commit:\n"
        for file in files[:HISTORY_LIMITS['files']]:
            augmentation += f"- {file}\n"

    augmentation += "\nThis historical context may help understand the code's evolution and common issues.\n"

    return augmentation