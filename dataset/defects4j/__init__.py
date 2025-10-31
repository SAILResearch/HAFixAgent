"""
Defects4J dataset implementation for HAFixAgent.

This module provides Defects4J-specific implementations of HAFixAgent's
blame extraction and dataset management interfaces.
"""

from .defects4j_extractor import Defects4JExtractor

__all__ = ['Defects4JExtractor']