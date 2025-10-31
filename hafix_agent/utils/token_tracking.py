"""Token usage tracking utilities for evaluation."""

from typing import Any


def add_token_tracking_to_model(model: Any) -> None:
    """
    Add token tracking capabilities to a model instance.

    Tracks total, cached, and uncached token usage across all API calls.

    Args:
        model: Model instance to enhance with token tracking
    """
    if hasattr(model, 'total_input_tokens'):
        return  # Already has token tracking

    # Initialize token counters
    model.total_input_tokens = 0
    model.total_output_tokens = 0
    model.cached_input_tokens = 0
    model.uncached_input_tokens = 0

    # Wrap the query method to track token usage
    original_query = model.query

    def query_with_token_tracking(*args, **kwargs):
        try:
            response = original_query(*args, **kwargs)
        except Exception as e:
            # If the original query fails, re-raise the error
            print(f"Debug: Model query failed: {e}")
            raise

        # Extract token usage from response
        try:
            if (hasattr(response, 'get') and
                response.get('extra', {}).get('response') and
                'usage' in response['extra']['response']):
                usage = response['extra']['response']['usage']

                # Standard token counts
                input_tokens = usage.get('prompt_tokens', 0)
                output_tokens = usage.get('completion_tokens', 0)
                model.total_input_tokens += input_tokens
                model.total_output_tokens += output_tokens

                # Cache-aware token counts (if available)
                cached_tokens = 0

                # Check for cached tokens in different possible locations
                if 'prompt_tokens_details' in usage:
                    details = usage['prompt_tokens_details']
                    if isinstance(details, dict):
                        cached_tokens = details.get('cached_tokens', 0)

                # Also check for direct cache hit tokens (DeepSeek format)
                if 'prompt_cache_hit_tokens' in usage:
                    cached_tokens = max(cached_tokens, usage.get('prompt_cache_hit_tokens', 0))

                if cached_tokens > 0:
                    model.cached_input_tokens += cached_tokens
                    model.uncached_input_tokens += (input_tokens - cached_tokens)
                else:
                    # Fallback: assume all tokens are uncached if no cache info
                    model.uncached_input_tokens += input_tokens

        except Exception as e:
            # Silent fallback - token tracking shouldn't break the main flow
            print(f"Debug: Token tracking failed: {e}")

        return response

    model.query = query_with_token_tracking


def get_token_usage(model: Any) -> dict:
    """
    Get token usage statistics from a model.

    Args:
        model: Model instance with token tracking

    Returns:
        Dictionary with token usage statistics
    """
    return {
        "input_tokens": getattr(model, 'total_input_tokens', 0),
        "output_tokens": getattr(model, 'total_output_tokens', 0),
        "total_tokens": getattr(model, 'total_input_tokens', 0) + getattr(model, 'total_output_tokens', 0),
        "cached_input_tokens": getattr(model, 'cached_input_tokens', 0),
        "uncached_input_tokens": getattr(model, 'uncached_input_tokens', 0)
    }