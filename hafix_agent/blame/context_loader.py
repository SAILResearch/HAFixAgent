"""
Context loader interface and implementations for blame context extraction.
Supports both runtime extraction and pre-cached data loading.
"""

import json
from abc import ABC, abstractmethod
from typing import Dict, Any

from .extraction_config import (
    get_bug_info_path, get_blame_info_path, get_bug_info_filename, get_blame_info_filename,
    DEFAULT_SELECTOR, DEFAULT_N_LINES, validate_extraction_params
)


class ContextLoader(ABC):
    """Abstract interface for loading bug info and blame contexts."""

    @abstractmethod
    def get_bug_info(self, project_name: str, bug_id: str, extractor, **kwargs) -> Dict[str, Any]:
        """Get bug information for the specified bug."""
        pass

    @abstractmethod
    def get_blame_context(self, project_name: str, bug_id: str, extractor,
                         selector_type: str, n_lines: int, bug_info: Dict[str, Any],
                         **kwargs) -> Dict[str, Any]:
        """Get blame context for the specified bug and parameters."""
        pass


class RuntimeContextLoader(ContextLoader):
    """Context loader that performs runtime extraction (original behavior)."""

    def get_bug_info(self, project_name: str, bug_id: str, extractor, docker_env=None, **kwargs) -> Dict[str, Any]:
        """Extract bug info at runtime."""
        try:
            # pylint: disable=protected-access  # Intentional access to private method for internal delegation
            return extractor._extract_bug_info_context(project_name, bug_id, docker_env=docker_env, **kwargs)
        except Exception as e:
            return {
                "bug_info": None,
                "error": f"Bug info extraction failed: {str(e)}"
            }

    def get_blame_context(self, project_name: str, bug_id: str, extractor,
                         selector_type: str, n_lines: int, bug_info: Dict[str, Any],
                         docker_env=None, **kwargs) -> Dict[str, Any]:
        """Extract blame context at runtime."""
        try:
            # pylint: disable=protected-access  # Intentional access to private method for internal delegation
            return extractor._extract_blame_context(
                project_name, bug_id,
                selector_type=selector_type, n_lines=n_lines, bug_info=bug_info,
                docker_env=docker_env, **kwargs
            )
        except Exception as e:
            return {
                "bug_info": None,
                "blame_info": None,
                "error": f"Extraction failed: {str(e)}"
            }


class CachedContextLoader(ContextLoader):
    """Context loader that reads from pre-cached data files."""

    def __init__(self, dataset_base_path: str, category_filter: str = None):
        """Initialize cached context loader."""
        self.dataset_base_path = dataset_base_path
        self.category_filter = category_filter

    def get_bug_info(self, project_name: str, bug_id: str, extractor=None, **kwargs) -> Dict[str, Any]:
        """Load bug info from cached file."""
        bug_info_path = get_bug_info_path(self.dataset_base_path, self.category_filter)
        filename = get_bug_info_filename(project_name, bug_id)
        file_path = bug_info_path / filename

        if not file_path.exists():
            raise FileNotFoundError(
                f"Cached bug info not found: {file_path}. "
                f"Please run pre-extraction for {project_name}_{bug_id}"
            )

        return self._load_json_file(file_path, f"bug info for {project_name}_{bug_id}")

    def get_blame_context(self, project_name: str, bug_id: str, extractor=None,
                         selector_type: str = DEFAULT_SELECTOR, n_lines: int = DEFAULT_N_LINES,
                         bug_info: Dict[str, Any] = None, **kwargs) -> Dict[str, Any]:
        """Load blame context from cached file."""

        # Validate parameters
        validate_extraction_params(selector_type, n_lines)

        # Find cached file
        blame_info_path = get_blame_info_path(self.dataset_base_path, self.category_filter)
        filename = get_blame_info_filename(project_name, bug_id, selector_type, n_lines)
        file_path = blame_info_path / filename

        if not file_path.exists():
            raise FileNotFoundError(
                f"Cached blame context not found: {file_path}. "
                f"Please run pre-extraction for {project_name}_{bug_id} "
                f"with selector={selector_type}, n_lines={n_lines}"
            )

        cached_data = self._load_json_file(file_path, f"blame context for {project_name}_{bug_id}")

        # Return only blame_info - bug_info comes from separate dedicated file
        return {
            'blame_info': cached_data.get('blame_info'),
            'bug_info': bug_info  # Use bug_info from separate loading
        }

    def _load_json_file(self, file_path, description: str) -> Dict[str, Any]:
        """Load and parse JSON file with consistent error handling."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            raise RuntimeError(f"Failed to load {description} from {file_path}: {e}")



def create_context_loader(mode: str, dataset_base_path: str = None, category_filter: str = None) -> ContextLoader:
    """
    Factory function to create appropriate context loader.

    Args:
        mode: 'runtime' or 'cached'
        dataset_base_path: Required for cached mode (e.g., 'dataset/defects4j')
        category_filter: Optional category filter for cached mode (e.g., 'multi_file_multi_hunk')

    Returns:
        ContextLoader instance
    """
    if mode == "runtime":
        return RuntimeContextLoader()
    elif mode == "cached":
        if dataset_base_path is None:
            raise ValueError("dataset_base_path is required for cached mode")
        return CachedContextLoader(dataset_base_path, category_filter)
    else:
        raise ValueError(f"Unsupported mode: {mode}. Use 'runtime' or 'cached'")