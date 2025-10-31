"""Model specifications and technical constants."""

from typing import Dict

# Context window limits for supported models (in tokens)
# These are fixed technical specifications, not user-configurable parameters
MODEL_CONTEXT_LIMITS: Dict[str, int] = {
    # OpenAI models
    "gpt-4o": 128000,
    "gpt-4o-mini": 128000,
    "gpt-5-mini": 128000,  # Expected limit, verify when available
    "gpt-4": 128000,
    "gpt-4-turbo": 128000,
    "gpt-3.5-turbo": 16384,

    # DeepSeek models
    "deepseek/deepseek-chat": 131072,
    "deepseek-chat": 131072,
    "deepseek/deepseek-coder": 131072,
    "deepseek-coder": 131072,

    # Qwen-Coder models
    "qwen2.5-coder-7b": 131072,
    "qwen2.5-coder-14b": 131072,
    "qwen2.5-coder-32b": 131072,
    "qwen/qwen2.5-coder-7b": 131072,
    "qwen/qwen2.5-coder-14b": 131072,
    "qwen/qwen2.5-coder-32b": 131072,

    # Claude models (for future use)
    "claude-3-5-sonnet": 200000,
    "claude-3-opus": 200000,
    "claude-3-haiku": 200000,
}

# Conservative fallback for unknown models
DEFAULT_CONTEXT_LIMIT: int = 128000


def get_context_limit(model_name: str) -> int:
    """
    Get the context window limit for a given model.

    Args:
        model_name: Name of the model (with or without provider prefix)

    Returns:
        Context limit in tokens
    """
    # Try exact match first
    if model_name in MODEL_CONTEXT_LIMITS:
        return MODEL_CONTEXT_LIMITS[model_name]

    # Try without provider prefix
    base_model = model_name.split('/')[-1] if '/' in model_name else model_name
    if base_model in MODEL_CONTEXT_LIMITS:
        return MODEL_CONTEXT_LIMITS[base_model]

    # Return conservative default for unknown models
    return DEFAULT_CONTEXT_LIMIT


def get_model_info(model_name: str) -> Dict[str, any]:
    """
    Get comprehensive model information.

    Args:
        model_name: Name of the model

    Returns:
        Dictionary with model specifications
    """
    return {
        "context_limit": get_context_limit(model_name),
        "model_name": model_name,
        "base_model": model_name.split('/')[-1] if '/' in model_name else model_name,
    }