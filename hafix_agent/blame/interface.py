"""
Interfaces for dataset-agnostic extraction operations.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any


class BlameExtractor(ABC):
    """Dataset-agnostic interface for blame context extraction with containers."""

    @abstractmethod
    def _extract_blame_context(
        self,
        project_name: str,
        bug_id: str,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Extract blame context for a bug using containers.
        
        Returns:
            Dict with bug_info, blame_info, docker_env, or error
        """
        pass


class BugInfoExtractor(ABC):
    """Dataset-agnostic interface for lightweight bug information extraction."""
    
    @abstractmethod
    def _extract_bug_info_context(
        self,
        project_name: str,
        bug_id: str,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Extract lightweight bug information without Docker or blame analysis.
        
        Gets essential bug data from filesystem for baseline prompts:
        - Bug descriptions from dataset sources
        - Fault locations from patches/metadata  
        - Failing tests from dataset info
        
        Args:
            project_name: Project name
            bug_id: Bug identifier
            
        Returns:
            Dict with bug_info for baseline prompts
        """
        pass