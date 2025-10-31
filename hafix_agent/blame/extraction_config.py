"""
Path utilities for blame extraction cache structure.
Simple functions for multi-dataset support.
"""

from pathlib import Path

# Default values for extraction parameters
DEFAULT_SELECTOR = "first"
DEFAULT_N_LINES = 1
SUPPORTED_SELECTORS = ["first", "random", "llm_judge"]
SUPPORTED_N_LINES = [1]

def get_cache_path(dataset_base_path: str, bug_filter: str = None) -> Path:
    """Get the cache directory path for a dataset, optionally filtered by bug category."""
    base_path = Path(dataset_base_path) / "cached_context"
    if bug_filter:
        return base_path / bug_filter
    return base_path

def get_bug_info_path(dataset_base_path: str, bug_filter: str = None) -> Path:
    """Get bug info cache directory."""
    return get_cache_path(dataset_base_path, bug_filter) / "bug_info"

def get_blame_info_path(dataset_base_path: str, bug_filter: str = None) -> Path:
    """Get blame info cache directory."""
    return get_cache_path(dataset_base_path, bug_filter) / "blame_info"

def get_bug_info_filename(project_name: str, bug_id: str) -> str:
    """Generate consistent bug info filename."""
    return f"{project_name}_{bug_id}.json"

def get_blame_info_filename(project_name: str, bug_id: str, selector_type: str = DEFAULT_SELECTOR, n_lines: int = DEFAULT_N_LINES) -> str:
    """Generate consistent blame info filename with parameters."""
    return f"{project_name}_{bug_id}_{selector_type}_{n_lines}line.json"

def validate_extraction_params(selector_type: str, n_lines: int) -> None:
    """Validate that parameters are supported."""
    if selector_type not in SUPPORTED_SELECTORS:
        raise ValueError(f"Unsupported selector_type: {selector_type}. Supported: {SUPPORTED_SELECTORS}")
    if n_lines not in SUPPORTED_N_LINES:
        raise ValueError(f"Unsupported n_lines: {n_lines}. Supported: {SUPPORTED_N_LINES}")
