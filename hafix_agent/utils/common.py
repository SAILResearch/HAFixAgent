"""
Common utilities for HAFixAgent framework.
"""

import time
from datetime import datetime
import pytz


def resolve_model_tag(config: dict) -> str:
    """Short directory tag for the active model, used to isolate output paths
    across LLMs. Honors an explicit ``model_tag`` in the model config; otherwise
    derives a tag from ``model.model_name``."""
    explicit = config.get("model_tag")
    if explicit:
        return str(explicit)
    model_name = config.get("model", {}).get("model_name", "")
    tag = model_name.split("/")[-1].replace("-", "").replace(".", "").replace("turbo0125", "").lower()
    tag_map = {
        "gpt35turbo0125": "gpt35",
        "o4mini": "o4mini",
        "deepseekv32exp": "deepseek_openrouter",
        "deepseekchat": "deepseek",
    }
    return tag_map.get(tag, tag)


def get_timestamp(timestamp: float = None, timezone: str = None) -> str:
    """Get formatted timestamp in specified timezone or host local time."""
    if timestamp is None:
        timestamp = time.time()

    if timezone:
        # Try to use the specified timezone
        try:
            tz = pytz.timezone(timezone)
            utc_dt = datetime.fromtimestamp(timestamp, tz=pytz.UTC)
            local_dt = utc_dt.astimezone(tz)
            return local_dt.strftime("%Y-%m-%d %H:%M:%S %Z")
        except:
            pass  # Fall back to host time

    # Default: use host local time
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(timestamp))


def format_duration_human(seconds: float) -> str:
    """Format duration in seconds to human readable format."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes}m {secs}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        if secs > 0:
            return f"{hours}h {minutes}m {secs}s"
        else:
            return f"{hours}h {minutes}m"