"""
HAFixAgent utility modules for evaluation, tracking, and common operations.
"""

# Core evaluation utilities
from .evaluation import (
    BugLogger,
    EvaluationProgressManager,
    extract_execution_metrics,
    save_trajectory_safe
)

# Token tracking utilities
from .token_tracking import (
    add_token_tracking_to_model,
    get_token_usage
)

# Common utilities
from .common import (
    get_timestamp,
    format_duration_human
)

__all__ = [
    # Evaluation classes
    'BugLogger',
    'EvaluationProgressManager',

    # Evaluation functions
    'extract_execution_metrics',
    'save_trajectory_safe',

    # Token tracking
    'add_token_tracking_to_model',
    'get_token_usage',

    # Common utilities
    'get_timestamp',
    'format_duration_human'
]