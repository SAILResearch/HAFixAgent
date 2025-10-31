"""
HAFixAgent evaluation utilities for logging and progress management.
"""

import json
import logging
import time
import threading
from pathlib import Path
from typing import Dict, Optional, Any
from rich.console import Console

from .common import get_timestamp, format_duration_human

console = Console()


class TimezoneAwareFormatter(logging.Formatter):
    """Custom formatter that uses consistent timezone handling."""

    def __init__(self, fmt=None, datefmt=None, timezone=None):
        super().__init__(fmt, datefmt)
        self.timezone = timezone

    def formatTime(self, record, datefmt=None):
        """Override formatTime to use consistent timezone."""
        # Use the same get_timestamp function as EvaluationProgressManager
        return get_timestamp(record.created, self.timezone)


class BugLogger:
    """Unified logger that handles both file and console output."""

    def __init__(self, project_bug: str, log_file: Path, history_category: str, timezone: str = None):
        self.project_bug = project_bug
        self.timezone = timezone
        self.logger = logging.getLogger(f"hafix.{project_bug}-{history_category}")
        self.logger.setLevel(logging.DEBUG)

        # Remove any existing handlers
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)

        # File handler
        file_handler = logging.FileHandler(log_file, mode='w')
        file_handler.setLevel(logging.DEBUG)
        # Use timezone-aware formatter to match EvaluationProgressManager
        formatter = TimezoneAwareFormatter('%(asctime)s - %(levelname)s - %(message)s', timezone=timezone)
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)
        self.logger.propagate = False

        # Store log file path for console output capture
        self.log_file = log_file
    
    def phase(self, phase_num: int, description: str):
        """Log phase with both file and console output."""
        msg = f"Phase {phase_num}: {description}"
        self.logger.info(msg)
    
    def info(self, message: str):
        """Log info message."""
        self.logger.info(message)
    
    def debug(self, message: str):
        """Log debug message (file only)."""
        self.logger.debug(message)
    
    def warning(self, message: str):
        """Log warning message."""
        self.logger.warning(message)
    
    def error(self, message: str, show_console: bool = True):
        """Log error message."""
        self.logger.error(f"{message}")
        if show_console:
            console.print(f"[red]{message}[/red]")
    
    def success(self, message: str):
        """Log success message."""
        self.logger.info(f"{message}")

    def console_print(self, message: str, level: str = "INFO"):
        """
        Log console output to file (for capturing Rich console.print output).

        Args:
            message: Console message to log
            level: Log level (INFO, DEBUG, WARNING, ERROR)
        """
        log_func = getattr(self.logger, level.lower(), self.logger.info)
        log_func(f"[CONSOLE] {message}")

    def cleanup(self):
        """Clean up logger handlers."""
        for handler in self.logger.handlers[:]:
            handler.close()
            self.logger.removeHandler(handler)


class EvaluationProgressManager:
    """Progress manager for HAFixAgent evaluation runs."""

    def __init__(self, total_bugs: int, output_file: str, timezone: str = None):
        self.total_bugs = total_bugs
        self.completed = 0
        self.failed = 0
        self.output_file = output_file
        self.timezone = timezone  # Use provided timezone or None for host time
        self.start_time = time.time()
        self.lock = threading.Lock()
        self.repair_results = {
            "successful_bugs": [],
            "failed_bugs": {"LimitsExceeded": [], "ExecutionError": []},
            "exit_status_breakdown": {"Submitted": 0, "LimitsExceeded": 0, "ExecutionError": 0},
            "total_model_cost": 0.0,
            "total_token_usage": {
                "input_tokens": 0,
                "output_tokens": 0,
                "total_tokens": 0,
                "cached_input_tokens": 0,
                "uncached_input_tokens": 0
            }
        }

        self.blame_failures = []
        
    def update_bug_status(self, project_bug: str, history_category: str, status: str,
                          repair_success: bool = None, exit_status: str = None):
        """Update status of a bug with history category context and repair results."""
        with self.lock:
            if status == "completed":
                self.completed += 1
            elif status == "failed":
                self.failed += 1

            # Progress update with history category
            elapsed = time.time() - self.start_time
            console.print(f"[{elapsed:.1f}s] {project_bug} ({history_category}): {status} "
                         f"({self.completed + self.failed}/{self.total_bugs})")

            # Save progress to JSON file (if output_file specified)
            if self.output_file:
                self._save_progress(project_bug, history_category, status, elapsed, repair_success, exit_status)

    def record_repair_result(self, project_bug: str, repair_result: Dict):
        """Record repair result at runtime for accurate progress tracking.

        Args:
            project_bug: Bug identifier (e.g., "Math_91")
            repair_result: Repair result dictionary from run_hafix_agent_repair()
        """
        with self.lock:
            success = repair_result.get('success', False)
            exit_status = repair_result.get('exit_status', 'Unknown')

            # Track repair success/failure by exit status
            if success and exit_status == "Submitted":
                self.repair_results["successful_bugs"].append(project_bug)
                self.repair_results["exit_status_breakdown"]["Submitted"] += 1
            elif exit_status == "LimitsExceeded":
                self.repair_results["failed_bugs"]["LimitsExceeded"].append(project_bug)
                self.repair_results["exit_status_breakdown"]["LimitsExceeded"] += 1
            else:
                # RuntimeError, ERROR, or other execution failures
                self.repair_results["failed_bugs"]["ExecutionError"].append(project_bug)
                self.repair_results["exit_status_breakdown"]["ExecutionError"] += 1

            # Aggregate model cost and token usage
            model_cost = repair_result.get('model_cost', 0.0)
            if isinstance(model_cost, (int, float)):
                self.repair_results["total_model_cost"] += model_cost

            token_usage = repair_result.get('token_usage', {})
            for key in self.repair_results["total_token_usage"]:
                if key in token_usage and isinstance(token_usage[key], (int, float)):
                    self.repair_results["total_token_usage"][key] += token_usage[key]

    def record_blame_failure(self, project_bug: str, failure_reason: str):
        """Record blame extraction failure at runtime.

        Args:
            project_bug: Bug identifier (e.g., "Math_91")
            failure_reason: Reason why blame extraction failed
        """
        with self.lock:
            self.blame_failures.append({
                "project_bug": project_bug,
                "failure_reason": failure_reason,
                "category": self._classify_failure_reason(failure_reason)
            })

    @staticmethod
    def _classify_failure_reason(reason: str) -> str:
        """Classify failure reason into categories based on actual error messages."""
        reason_lower = reason.lower()

        # Match actual error messages from the defects4j extractor
        if "could not find commit id" in reason_lower:
            return "missing_commit_id"
        elif "failed to checkout" in reason_lower:
            return "git_checkout_failed"
        elif "failed to clone" in reason_lower:
            return "git_clone_failed"
        elif "could not read patch file" in reason_lower:
            return "patch_file_missing"
        elif "container blame extraction failed" in reason_lower:
            return "extraction_exception"
        elif "extraction failed" in reason_lower:
            return "extraction_exception"
        elif "bug info extraction failed" in reason_lower:
            return "bug_info_extraction_failed"
        elif "exception during blame extraction" in reason_lower:
            return "extraction_exception"
        else:
            return "other"

    def _save_progress(self, project_bug: str, history_category: str, status: str, elapsed: float,
                      repair_success: bool = None, exit_status: str = None):
        """Save current progress to JSON file with runtime tracking."""

        total_repairs = len(self.repair_results["successful_bugs"]) + \
                       len(self.repair_results["failed_bugs"]["LimitsExceeded"]) + \
                       len(self.repair_results["failed_bugs"]["ExecutionError"])

        # Calculate timing information
        avg_seconds_per_bug = elapsed / max(1, self.completed + self.failed)

        # Calculate blame failure categories
        blame_categories = {}
        for failure in self.blame_failures:
            category = failure["category"]
            blame_categories[category] = blame_categories.get(category, 0) + 1

        progress_data = {
            "total_bugs": self.total_bugs,
            "execution_stats": {
                "completed": self.completed,
                "execution_failed": self.failed,
                "in_progress": self.total_bugs - self.completed - self.failed,
                "completion_rate": round((self.completed + self.failed) / self.total_bugs * 100, 1)
            },
            "repair_results": {
                "successful": len(self.repair_results["successful_bugs"]),
                "failed": len(self.repair_results["failed_bugs"]["LimitsExceeded"]) +
                         len(self.repair_results["failed_bugs"]["ExecutionError"]),
                "success_rate": round(len(self.repair_results["successful_bugs"]) / max(1, total_repairs) * 100, 1) if total_repairs > 0 else 0.0
            },
            "blame_failures": {
                "total_blame_failures": len(self.blame_failures),
                "failures": self.blame_failures,
                "failure_categories": blame_categories
            },
            "exit_status_breakdown": self.repair_results["exit_status_breakdown"],
            "successful_bugs": self.repair_results["successful_bugs"],
            "failed_bugs": self.repair_results["failed_bugs"],
            "total_model_cost": round(self.repair_results["total_model_cost"], 4),
            "total_token_usage": self.repair_results["total_token_usage"],
            "execution_timing": {
                "start_time": get_timestamp(self.start_time, self.timezone),
                "last_update_time": get_timestamp(timezone=self.timezone),
                "total_elapsed_seconds": round(elapsed, 1),
                "total_elapsed_human": format_duration_human(elapsed),
                "avg_seconds_per_bug": round(avg_seconds_per_bug, 1),
                "avg_human_per_bug": format_duration_human(avg_seconds_per_bug),
                "estimated_completion": self._estimate_completion_time(elapsed) if self.completed + self.failed > 0 else None
            },
            "last_update": {
                "project_bug": project_bug,
                "history_category": history_category,
                "status": status
            }
        }

        try:
            # Ensure parent directory exists
            Path(self.output_file).parent.mkdir(parents=True, exist_ok=True)
            with open(self.output_file, 'w') as f:
                json.dump(progress_data, f, indent=2)
        except Exception:
            pass  # Don't break evaluation if progress save fails

    def _estimate_completion_time(self, elapsed: float) -> str:
        """Estimate when all bugs will be completed."""
        completed_count = self.completed + self.failed
        if completed_count == 0:
            return None

        avg_time_per_bug = elapsed / completed_count
        remaining_bugs = self.total_bugs - completed_count
        estimated_remaining_time = remaining_bugs * avg_time_per_bug

        completion_timestamp = time.time() + estimated_remaining_time
        return get_timestamp(completion_timestamp, self.timezone)


def extract_execution_metrics(start_time: float, model=None, agent=None, env=None) -> Dict[str, Any]:
    """Extract execution metrics for both successful and failed HAFixAgent runs.

    Args:
        start_time: Execution start timestamp
        model: HAFixAgent model instance (for cost/token tracking)
        agent: HAFixAgent instance (for step counting)
        env: Environment instance (for patch extraction)

    Returns:
        Dictionary containing timing, cost, token usage, steps, and patch data
    """
    end_time = time.time()
    duration_seconds = round(end_time - start_time)

    # Extract model metrics
    model_cost = 0.0
    model_calls = 0
    token_usage = {}

    if model:
        try:
            model_cost = round(model.cost, 4)
            model_calls = model.n_calls
            # Import here to avoid circular dependencies
            from .token_tracking import get_token_usage
            token_usage = get_token_usage(model) if hasattr(model, 'total_input_tokens') else {}
        except Exception:
            pass

    # Extract agent metrics
    # agent_steps should equal model_calls (each LLM call = 1 step)
    # Note: len(messages) // 2 is incorrect because:
    # 1. System message breaks pairing
    # 2. Hafix-context creates extra user messages
    # 3. Final empty user message when agent stops
    agent_steps = model_calls

    # Extract patch content
    patch_content = ""
    if env:
        try:
            patch_result = env.execute("git diff HEAD")
            if patch_result.get('returncode') == 0:
                patch_content = patch_result.get('output', '')
                if patch_content:
                    console.print(f"📋 Code changes captured ({len(patch_content)} chars)")
            else:
                console.print(f"⚠️  Could not generate patch: {patch_result.get('output', 'Unknown error')}")
        except Exception as e:
            console.print(f"⚠️  Patch capture failed: {e}")

    return {
        "timing": {
            "start_time": get_timestamp(start_time),
            "end_time": get_timestamp(end_time),
            "duration_seconds": duration_seconds,
            "duration_human": format_duration_human(duration_seconds)
        },
        "model_cost": model_cost,
        "model_calls": model_calls,
        "token_usage": token_usage,
        "agent_steps": agent_steps,
        "checkout_success": env is not None,
        "patch": patch_content
    }


def save_trajectory_safe(agent, output_dir: Optional[Path], category_filter: str,
                        project_name: str, bug_id: int, history_category,
                        exit_status: str, message=None) -> bool:
    """Safely save trajectory with comprehensive error handling.

    Args:
        agent: HAFixAgent instance
        output_dir: Output directory path
        category_filter: Bug category for path organization
        project_name: Project name (e.g., "Math")
        bug_id: Bug ID number
        history_category: History category enum
        exit_status: Agent exit status
        message: Optional exit message

    Returns:
        True if trajectory was saved successfully, False otherwise
    """
    if not (agent and output_dir):
        return False

    try:
        # Import here to avoid circular dependencies
        from minisweagent.run.utils.save import save_traj

        trajectory_path = output_dir.parent / category_filter / "trajectories" / f"{project_name}_{bug_id}_{history_category.value}.traj.json"
        trajectory_path.parent.mkdir(parents=True, exist_ok=True)

        save_traj(
            agent,
            trajectory_path,
            exit_status=exit_status,
            result=str(message) if message else None,
            print_path=False
        )
        console.print(f"📁 Trajectory saved: {trajectory_path}")
        return True
    except Exception as e:
        console.print(f"⚠️  Failed to save trajectory: {e}")
        return False
