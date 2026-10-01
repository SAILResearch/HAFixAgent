"""Model specifications and technical constants."""

import os
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
    "gpt-3.5-turbo-0125": 16384,
    "o4-mini": 200000,

    # DeepSeek models
    "deepseek/deepseek-chat": 131072,
    "deepseek-chat": 131072,
    "deepseek/deepseek-coder": 131072,
    "deepseek-coder": 131072,
    "openrouter/deepseek/deepseek-v3.2-exp": 163840,
    "deepseek-v3.2-exp": 163840,

    # Qwen-Coder models
    "qwen2.5-coder-7b": 131072,
    "qwen2.5-coder-14b": 131072,
    "qwen2.5-coder-32b": 131072,
    "qwen/qwen2.5-coder-7b": 131072,
    "qwen/qwen2.5-coder-14b": 131072,
    "qwen/qwen2.5-coder-32b": 131072,
    "qwen3-coder-next": 262144,            # ~256K HF default (SAIL vLLM)
    "openai/qwen3-coder-next": 262144,

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


def register_local_model_with_litellm(config: Dict) -> None:
    """Register a local OpenAI-compatible model (vLLM endpoint) with litellm's
    registry so its internal cost/context lookups don't raise "model isn't
    mapped yet". No-op for cloud models already known to litellm (e.g. DeepSeek
    via OpenRouter), so it is safe to call unconditionally."""
    model = config.get("model", {})
    model_name = model.get("model_name", "")
    kwargs = model.get("model_kwargs", {})
    # Local vLLM endpoint: LLM_API_BASE overrides api_base so the real serving
    # host stays out of version control (configs ship a localhost default).
    # Mutates the shared model_kwargs, so the repair model and the LLM judge
    # both pick it up.
    env_base = os.environ.get("LLM_API_BASE")
    if env_base and "api_base" in kwargs:
        kwargs["api_base"] = env_base
    # Only custom OpenAI-compatible endpoints: model="openai/<name>" + api_base.
    if not model_name.startswith("openai/") or "api_base" not in kwargs:
        return
    short = model_name.split("/", 1)[-1]
    try:
        import litellm
    except Exception:
        return
    try:
        litellm.get_model_info(short)
        return  # already known to litellm
    except Exception:
        pass
    max_out = kwargs.get("max_tokens", 8192)
    litellm.register_model({
        short: {
            "max_tokens": max_out,
            "max_input_tokens": get_context_limit(short),
            "max_output_tokens": max_out,
            "input_cost_per_token": 0.0,
            "output_cost_per_token": 0.0,
            "litellm_provider": "openai",
            "mode": "chat",
        }
    })