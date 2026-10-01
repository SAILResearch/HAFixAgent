"""
BugsInPy evaluation runner for HAFixAgent.

Separate from run_defects4j_evaluation.py to avoid risking the stable Defects4J pipeline.
Reuses shared infrastructure (blame extraction, prompt builder, context loader).

Usage:
    python -m evaluation.run_bugsinpy_evaluation --bug-category all --history baseline --workers 24
    python -m evaluation.run_bugsinpy_evaluation --history fl_diff --workers 8 --bugs "tqdm_1,pandas_5"
"""

import concurrent.futures
import json
import time
import traceback as tb
from pathlib import Path
from typing import Dict, Optional

from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent.parent / ".env", override=True)

import subprocess
import typer
import yaml
from rich.console import Console
from minisweagent.models import get_model

from dataset.bugsinpy.bugsinpy_extractor import (
    BugsInPyExtractor, ensure_bugsinpy_docker_container, get_bugsinpy_work_dir
)
from dataset.bugsinpy.util import get_all_bugs
from hafix_agent.blame.core import HistoryCategory, history_name_to_category
from hafix_agent.blame.context_loader import create_context_loader
from hafix_agent.utils import (
    BugLogger, EvaluationProgressManager, extract_execution_metrics, save_trajectory_safe,
    get_timestamp as _get_timestamp, add_token_tracking_to_model, resolve_model_tag
)
from hafix_agent.utils.model_specs import register_local_model_with_litellm
from hafix_agent.agents.hafix_agent import HAFixAgent
from hafix_agent.prompts.prompt_builder import build_hafix_prompt

DEFAULT_TIMEZONE = "America/Toronto"

_CONFIG_CACHE = None
_CONFIG_PATH = None
_MODEL_CONFIG_PATH = None


def set_config_path(config_path: str, model_config_path: str = None):
    global _CONFIG_PATH, _MODEL_CONFIG_PATH
    _CONFIG_PATH = config_path
    _MODEL_CONFIG_PATH = model_config_path


def get_config() -> Dict:
    global _CONFIG_CACHE
    if _CONFIG_CACHE is None:
        config_path = Path(_CONFIG_PATH) if _CONFIG_PATH else Path(__file__).parent.parent / "config" / "bugsinpy.yaml"
        _CONFIG_CACHE = yaml.safe_load(config_path.read_text())
        if _MODEL_CONFIG_PATH:
            model_config = yaml.safe_load(Path(_MODEL_CONFIG_PATH).read_text())
            _CONFIG_CACHE["model"] = model_config["model"]
            if model_config.get("model_tag"):
                _CONFIG_CACHE["model_tag"] = model_config["model_tag"]
        register_local_model_with_litellm(_CONFIG_CACHE)
    return _CONFIG_CACHE


def get_timestamp(timestamp: float = None) -> str:
    return _get_timestamp(timestamp, DEFAULT_TIMEZONE)


console = Console()
app = typer.Typer(rich_markup_mode="rich", add_completion=False)


def _get_docker_image(project_name: str, bug_id: int, fallback_image: str) -> str:
    """Get pre-warmed image if available, otherwise fallback to base."""
    prewarmed = f"bugsinpy:{project_name.lower()}_{bug_id}"
    check = subprocess.run(
        ["docker", "image", "inspect", prewarmed],
        capture_output=True, text=True, timeout=10,
    )
    return prewarmed if check.returncode == 0 else fallback_image


def process_bugsinpy_bug(
    project_name: str,
    bug_id: int,
    output_dir: Path,
    history_category: HistoryCategory,
    progress_manager: EvaluationProgressManager,
    selector_type: str = "llm_judge",
    n_lines: int = 1,
    bug_category: str = "all",
    context_mode: str = "runtime",
    include_blameless: bool = True,
) -> None:
    """Process a single BugsInPy bug with blame context."""

    project_bug = f"{project_name}_{bug_id}"
    bug_output_dir = output_dir / project_bug
    bug_output_dir.mkdir(parents=True, exist_ok=True)

    result_file = bug_output_dir / f"{project_bug}_{history_category.value}_result.json"
    log_file = bug_output_dir / f"{project_bug}_{history_category.value}.log"

    log = BugLogger(project_bug, log_file, history_category.name, timezone=DEFAULT_TIMEZONE)
    config = get_config()
    env_config = config.get('environment', {})

    shared_docker_env = None

    try:
        log.info(f"=== Starting HAFixAgent BugsInPy evaluation for {project_bug} ===")
        log.info(f"History: {history_category.name}, Selector: {selector_type}, N-lines: {n_lines}")
        log.info(f"Context mode: {context_mode}")

        progress_manager.update_bug_status(project_bug, history_category.name, "extracting_info")

        # Docker setup with pre-warmed image preference
        image = _get_docker_image(
            project_name, bug_id, env_config.get('image', 'bugsinpy_image:clean')
        )
        log.info(f"Docker image: {image}")

        shared_docker_env, error = ensure_bugsinpy_docker_container(
            project_name, str(bug_id), None,
            image=image,
            cleanup_on_exit=env_config.get('cleanup_on_exit', True),
        )
        if error:
            raise Exception(f"Docker setup failed: {error}")

        # Phase 1: Extract bug info
        log.phase(1, "Extracting bug information")
        extractor = BugsInPyExtractor()
        bug_info = extractor._extract_bug_info_context(
            project_name=project_name,
            bug_id=str(bug_id),
            docker_env=shared_docker_env,
        )
        bug_info['bug_category'] = {
            'SL': 'single_line', 'SH': 'single_hunk',
            'SFMH': 'single_file_multi_hunk', 'MFMH': 'multi_file_multi_hunk',
        }.get(bug_info.get('bug_category', ''), bug_category)

        if not bug_info or 'error' in bug_info:
            raise Exception(f"Bug info extraction failed: {bug_info.get('error')}")

        log.info(f"Fault locations: {len(bug_info.get('fault_locations', []))}, "
                 f"Tests: {len(bug_info.get('failing_tests', []))}")

        # Phase 2: Extract blame context (skip for baseline)
        blame_info = None
        if history_category != HistoryCategory.baseline:
            log.phase(2, "Extracting blame context")
            try:
                blame_result = extractor._extract_blame_context(
                    project_name=project_name,
                    bug_id=str(bug_id),
                    selector_type=selector_type,
                    n_lines=n_lines,
                    docker_env=shared_docker_env,
                    bug_info=bug_info,
                    include_blameless=include_blameless,
                    model_config=config.get('model', {}),
                )
                blame_info = blame_result.get("blame_info") if blame_result else None
                if blame_info:
                    log.info("Blame context extracted successfully")
                else:
                    log.warning("No blame context available")
            except Exception as e:
                log.warning(f"Blame extraction failed: {e}")

        # Phase 3: Run repair agent
        phase_num = 3 if history_category != HistoryCategory.baseline else 2
        log.phase(phase_num, "Running repair agent")
        progress_manager.update_bug_status(project_bug, history_category.name, "running_repair")

        repair_result = _run_repair(
            project_name, bug_id, bug_info, blame_info, history_category,
            shared_docker_env, config, output_dir, bug_category, log,
            selector_type, n_lines, include_blameless,
        )

        # Save result
        result_data = {
            "project_name": project_name,
            "bug_id": project_bug,
            "history_category": history_category.name,
            "repair_result": repair_result,
            "timestamp": get_timestamp(),
            "selector_type": selector_type,
            "n_lines": n_lines,
            "bug_info": {k: v for k, v in bug_info.items() if k != 'golden_patch'},
            "blame_info": {"has_blame": blame_info is not None} if blame_info else None,
        }
        with open(result_file, "w") as f:
            json.dump(result_data, f, indent=2, default=str)

        status = "success" if repair_result.get("success") else "failed"
        progress_manager.update_bug_status(project_bug, history_category.name, status)
        log.info(f"=== Evaluation complete: {status} ===")

    except Exception as e:
        log.error(f"Evaluation failed: {e}")
        log.error(tb.format_exc())
        progress_manager.update_bug_status(project_bug, history_category.name, "error")

        error_result = {
            "project_name": project_name,
            "bug_id": project_bug,
            "history_category": history_category.name,
            "repair_result": {"success": False, "error": str(e)},
            "timestamp": get_timestamp(),
        }
        with open(result_file, "w") as f:
            json.dump(error_result, f, indent=2, default=str)

    finally:
        if shared_docker_env and env_config.get('cleanup_on_exit', True):
            try:
                shared_docker_env.cleanup()
            except Exception:
                pass


def _run_repair(
    project_name: str,
    bug_id: int,
    bug_info: Dict,
    blame_info: Optional[Dict],
    history_category: HistoryCategory,
    docker_env=None,
    config: Optional[Dict] = None,
    output_dir: Optional[Path] = None,
    bug_category: str = "all",
    logger=None,
    selector_type: str = "llm_judge",
    n_lines: int = 1,
    include_blameless: bool = True,
) -> Dict:
    """Run HAFixAgent repair on a BugsInPy bug."""

    try:
        start_time = time.time()
        if config is None:
            config = get_config()
        model = get_model(None, config.get("model", {}))

        try:
            add_token_tracking_to_model(model)
        except Exception:
            pass

        env, error = ensure_bugsinpy_docker_container(project_name, str(bug_id), docker_env)
        if error:
            return {"success": False, "error": "docker_setup_failed",
                    "runtime": round(time.time() - start_time)}

        repo_path = bug_info.get('repo_path', get_bugsinpy_work_dir(project_name, str(bug_id)))
        env.config.cwd = repo_path

        try:
            prompt_data = build_hafix_prompt(
                bug_info=bug_info,
                history_category=history_category,
                blame_info=blame_info,
                config=config,
            )

            agent_config = config.get('agent', {})
            agent = HAFixAgent(
                model=model,
                env=env,
                logger=logger,
                system_template=prompt_data["system_template"],
                instance_template=prompt_data["instance_template"],
                step_limit=agent_config.get('step_limit', 50),
                cost_limit=agent_config.get('cost_limit', 1.0),
                dataset=agent_config.get('dataset', 'bugsinpy'),
                action_observation_template=agent_config.get('action_observation_template'),
                format_error_template=agent_config.get('format_error_template'),
                timeout_template=agent_config.get('timeout_template'),
            )
            agent.model_config = config.get('model', {})
            agent.bug_info = bug_info
            agent.adaptive_selector_type = selector_type
            agent.adaptive_n_lines = n_lines
            agent.adaptive_include_blameless = include_blameless

            agent.set_bug_context(bug_info, blame_info, history_category)
            agent.extra_template_vars.update(prompt_data["template_vars"])

            exit_status, submission = agent.run()

            metrics = extract_execution_metrics(start_time, model=model, agent=agent, env=env)

            if output_dir:
                save_trajectory_safe(agent, output_dir, bug_category, project_name, bug_id,
                                     history_category, exit_status, submission)

            return {
                "success": exit_status == "Submitted",
                "exit_status": exit_status,
                "message": submission or "",
                **metrics,
            }

        except Exception as e:
            metrics = extract_execution_metrics(
                start_time if 'start_time' in locals() else time.time(),
                model if 'model' in locals() else None,
                agent if 'agent' in locals() else None,
                env if 'env' in locals() else None
            )
            return {
                "success": False,
                "exit_status": "ExecutionError",
                "message": str(e),
                "traceback": tb.format_exc(),
                **metrics
            }

    except Exception as e:
        return {"success": False, "error": str(e)}


@app.command()
def main(
    bug_category: str = typer.Option("all", "--bug-category",
        help="Bug category: single_line/SL, single_hunk/SH, single_file_multi_hunk/SFMH, multi_file_multi_hunk/MFMH, all"),
    history_category: str = typer.Option("baseline", "--history",
        help="History heuristic: baseline, fn_all, fn_pair, fl_diff"),
    selector_type: str = typer.Option("llm_judge", "--selector-type",
        help="Line selection: first, random, llm_judge"),
    n_lines: int = typer.Option(1, "--n-lines", help="Number of blame lines (1-10)"),
    context_mode: str = typer.Option("runtime", "--context-mode", help="runtime or cached"),
    blame_category: str = typer.Option("both", "--blame-category", help="blameable, blameless, or both"),
    config: str = typer.Option("config/bugsinpy.yaml", "--config", help="Config YAML file"),
    model_config: str = typer.Option("", "--model-config", help="Model config override"),
    output: str = typer.Option("results/bugsinpy", "-o", "--output", help="Output directory"),
    custom_bugs: str = typer.Option("", "--bugs", help="Comma-separated bugs (e.g., tqdm_1,pandas_5)"),
    workers: int = typer.Option(1, "-w", "--workers", help="Parallel workers"),
) -> None:
    """Run HAFixAgent on BugsInPy bugs with blame context."""

    set_config_path(config, model_config if model_config else None)

    # Parse history category
    try:
        if history_category.isdigit():
            history_cat = HistoryCategory(int(history_category))
        elif history_category in history_name_to_category:
            history_cat = history_name_to_category[history_category]
        else:
            history_cat = HistoryCategory[history_category]
    except (KeyError, ValueError):
        console.print(f"[red]Invalid history category: {history_category}[/red]")
        return

    # Category normalization
    # Map CLI input to abbreviation for CSV matching
    abbrev_map = {
        "single_line": "SL", "SL": "SL",
        "single_hunk": "SH", "SH": "SH",
        "single_file_multi_hunk": "SFMH", "SFMH": "SFMH",
        "multi_file_multi_hunk": "MFMH", "MFMH": "MFMH",
        "all": "all",
    }
    # Map abbreviation back to long form for directory names (consistent with Defects4J)
    long_name_map = {
        "SL": "single_line", "SH": "single_hunk",
        "SFMH": "single_file_multi_hunk", "MFMH": "multi_file_multi_hunk",
        "all": "all",
    }
    cat_filter = abbrev_map.get(bug_category, bug_category)
    cat_dir_name = long_name_map.get(cat_filter, bug_category)

    # Discover bugs
    extractor = BugsInPyExtractor()
    if custom_bugs:
        bug_list = []
        for bug_str in [b.strip() for b in custom_bugs.split(",")]:
            parts = bug_str.rsplit("_", 1)
            if len(parts) == 2:
                bug_list.append((parts[0], int(parts[1])))
    else:
        all_bugs = get_all_bugs()
        bug_list = []
        for project, bug_ids in sorted(all_bugs.items()):
            for bid in bug_ids:
                if cat_filter != "all":
                    actual_cat = extractor.get_bug_category(project, bid)
                    if actual_cat != cat_filter:
                        continue
                bug_list.append((project, bid))

    console.print(f"[green]BugsInPy: {len(bug_list)} bugs, {history_cat.name}, {workers} workers[/green]")

    # Setup output directory
    selector_dir = f"{selector_type}_{n_lines}line"
    # Append model tag when using a non-default model config (isolates per-LLM output)
    if model_config:
        selector_dir = f"{selector_dir}_{resolve_model_tag(get_config())}"
    output_path = Path(output) / selector_dir / cat_dir_name
    output_path.mkdir(parents=True, exist_ok=True)

    # Log model
    cfg = get_config()
    model_name = cfg.get("model", {}).get("model_name", "unknown")
    console.print(f"[cyan]Model: {model_name}[/cyan]")

    progress_manager = EvaluationProgressManager(
        len(bug_list), str(output_path / f"progress_{history_cat.value}_{blame_category}.json"), DEFAULT_TIMEZONE
    )

    if workers <= 1:
        for project, bid in bug_list:
            process_bugsinpy_bug(
                project, bid, output_path, history_cat, progress_manager,
                selector_type, n_lines, bug_category, context_mode,
                blame_category != "blameable",
            )
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(
                    process_bugsinpy_bug,
                    project, bid, output_path, history_cat, progress_manager,
                    selector_type, n_lines, bug_category, context_mode,
                    blame_category != "blameable",
                ): f"{project}_{bid}"
                for project, bid in bug_list
            }
            for future in concurrent.futures.as_completed(futures):
                bug_label = futures[future]
                try:
                    future.result()
                except Exception as e:
                    console.print(f"[red]{bug_label} failed: {e}[/red]")

    console.print(f"\n[green]Evaluation complete. Results in {output_path}[/green]")


if __name__ == "__main__":
    app()
