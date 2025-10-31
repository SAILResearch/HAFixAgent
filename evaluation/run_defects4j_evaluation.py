import concurrent.futures
import json
import time
import traceback as tb
from pathlib import Path
from typing import Dict, Optional
import typer
import yaml
from rich.console import Console
from minisweagent.models import get_model

from dataset.defects4j.defects4j_extractor import Defects4JExtractor
from dataset.defects4j.util import get_defects4j_work_dir, ensure_defects4j_docker_container
from hafix_agent.blame.core import HistoryCategory, history_name_to_category
from hafix_agent.blame.context_loader import create_context_loader
from hafix_agent.utils import (
    BugLogger, EvaluationProgressManager, extract_execution_metrics, save_trajectory_safe,
    get_timestamp as _get_timestamp, add_token_tracking_to_model
)
from hafix_agent.agents.hafix_agent import HAFixAgent
from hafix_agent.prompts.prompt_builder import build_hafix_prompt

# Default timezone for Defects4J evaluation runs
DEFAULT_TIMEZONE = "America/Toronto"

# Load config once at module level to avoid repeated I/O
_CONFIG_CACHE = None
_CONFIG_PATH = None

def set_config_path(config_path: str):
    """Set the config path (must be called before get_config)."""
    global _CONFIG_PATH
    _CONFIG_PATH = config_path

def get_config() -> Dict:
    """Get cached config or load it if not cached."""
    global _CONFIG_CACHE
    if _CONFIG_CACHE is None:
        # Use provided config path or default to defects4j.yaml
        if _CONFIG_PATH:
            config_path = Path(_CONFIG_PATH)
        else:
            config_path = Path(__file__).parent.parent / "config" / "defects4j.yaml"
        _CONFIG_CACHE = yaml.safe_load(config_path.read_text())
    return _CONFIG_CACHE

def get_timestamp(timestamp: float = None) -> str:
    """Get timestamp in Toronto timezone for Defects4J evaluation."""
    return _get_timestamp(timestamp, DEFAULT_TIMEZONE)

console = Console()

_HELP_TEXT = """
[not dim]
Run HAFixAgent on Defects4J with blame-based historical context.
[/not dim]
"""

app = typer.Typer(rich_markup_mode="rich", add_completion=False)


def process_defects4j_bug_with_blame(
    project_name: str,
    bug_id: int,
    output_dir: Path,
    history_category: HistoryCategory,
    progress_manager: EvaluationProgressManager,
    selector_type: str = "first",
    n_lines: int = 1,
    bug_category: str = "single_hunk",
    context_mode: str = "runtime",
    cache_dataset_path: str = "dataset/defects4j",
    include_blameless: bool = True
) -> None:
    """Process a single Defects4J bug with blame context."""

    project_bug = f"{project_name}_{bug_id}"
    bug_output_dir = output_dir / project_bug
    bug_output_dir.mkdir(parents=True, exist_ok=True)

    # Output files
    result_file = bug_output_dir / f"{project_bug}_{history_category.value}_result.json"
    log_file = bug_output_dir / f"{project_bug}_{history_category.value}.log"

    # Setup unified logger with consistent timezone
    log = BugLogger(project_bug, log_file, history_category.name, timezone=DEFAULT_TIMEZONE)

    # Use cached config to avoid repeated I/O
    config = get_config()
    env_config = config.get('environment', {})
    docker_params = {k: v for k, v in env_config.items() if k != 'environment_class'}

    try:
        log.info(f"=== Starting HAFixAgent evaluation for {project_bug} ===")
        log.info(f"History category: {history_category.name}, Selector: {selector_type}, N-lines: {n_lines}")
        log.info(f"Context mode: {context_mode}")

        progress_manager.update_bug_status(project_bug, history_category.name, "extracting_info")

        # Create context loader and Docker container (using proper abstractions)
        log.phase(1, f"{'Loading bug information from cache' if context_mode == 'cached' else 'Creating shared Docker container and extracting bug information'}")

        # Create context loader with base path and category filter
        context_loader = create_context_loader(context_mode, cache_dataset_path, bug_category)

        if context_mode == "cached":
            log.info(f"Using cached context loader with base path: {cache_dataset_path}, category: {bug_category}")

        # Always create Docker container (needed for agent execution anyway)
        log.debug(f"Creating shared Docker container for {project_name}_{bug_id}")
        shared_docker_env, error = ensure_defects4j_docker_container(
            project_name, str(bug_id), None,
            image=docker_params.get('image'),
            use_existing_container=docker_params.get('use_existing_container'),
            cleanup_on_exit=docker_params.get('cleanup_on_exit')
        )

        if error:
            log.error(f"Docker setup failed: {error}")
            raise Exception(f"Docker setup failed for {project_name}_{bug_id}: {error}")

        log.info(f"Successfully created and checked out {project_name}_{bug_id} in shared container")

        # Extract bug information using context loader (cached or runtime)
        log.debug(f"Loading bug info for {project_name}_{bug_id} using {context_mode} mode")
        extractor = Defects4JExtractor()

        # Unified call - interface handles cached vs runtime differences
        bug_info = context_loader.get_bug_info(
            project_name=project_name,
            bug_id=str(bug_id),
            extractor=extractor,
            docker_env=shared_docker_env,  # Ignored in cached mode
            **docker_params                # Ignored in cached mode
        )

        # Add bug category for strategy selection in prompts
        bug_info['bug_category'] = bug_category
        
        if not bug_info or 'error' in bug_info:
            log.error("Failed to extract baseline bug info")
            raise Exception(f"Failed to extract baseline bug info: {bug_info.get('error', 'Unknown error')}")
        
        log.info(f"Baseline bug info extracted: {bug_info.get('bug_id')}")
        log.debug(f"   Description length: {len(bug_info.get('description', ''))}")
        log.debug(f"   Fault locations: {len(bug_info.get('fault_locations', []))}")
        log.debug(f"   Failing tests: {len(bug_info.get('failing_tests', []))}")
        
        blame_info = None
        blame_failure_reason = None
        # Skip upfront blame loading for baseline (no history) and adaptive (on-demand) modes
        if history_category != HistoryCategory.baseline and history_category != HistoryCategory.adaptive:
            log.debug(f"Loading blame context with selector={selector_type}, n_lines={n_lines} using {context_mode} mode")
            try:
                # Unified call - interface handles cached vs runtime differences
                blame_result = context_loader.get_blame_context(
                    project_name=project_name,
                    bug_id=str(bug_id),
                    extractor=extractor,
                    selector_type=selector_type,
                    n_lines=n_lines,
                    bug_info=bug_info,
                    include_blameless=include_blameless,
                    docker_env=shared_docker_env,  # Ignored in cached mode
                    model_config=config.get('model', {}),  # For LLMJudgeSelector
                    **docker_params                # Ignored in cached mode
                )

                # Check if blame extraction failed
                if 'error' in blame_result:
                    blame_failure_reason = blame_result['error']
                    log.warning(f"Blame context loading failed: {blame_failure_reason}")
                    log.info("Continuing with baseline mode due to blame failure")

                    # Record blame failure for runtime progress tracking
                    progress_manager.record_blame_failure(project_bug, blame_failure_reason)
                else:
                    blame_info = blame_result.get('blame_info')
                    log.info(f"Blame context loaded: blame_info={bool(blame_info)} ({context_mode} mode)")

            except Exception as e:
                blame_failure_reason = f"Exception during blame context loading: {str(e)}"
                log.warning(f"Blame context loading failed: {blame_failure_reason}")
                log.info("Continuing with baseline mode due to blame failure")

                # Record blame failure for runtime progress tracking
                progress_manager.record_blame_failure(project_bug, blame_failure_reason)
        elif history_category == HistoryCategory.adaptive:
            log.info("Adaptive mode - blame context will be loaded on-demand by agent")
        else:
            log.info("Baseline mode - skipping blame context loading")

        # History augmentation will be built in build_hafix_prompt() - no duplication needed
        log.phase(2, "Preparing for HAFixAgent repair")
        if blame_info and history_category != HistoryCategory.baseline:
            log.info(f"Will build history augmentation for category: {history_category.name}")
        else:
            log.info("Using baseline mode - no history augmentation")

        progress_manager.update_bug_status(project_bug, history_category.name, "running_repair")
        log.phase(3, "Running HAFixAgent repair")
        
        # Run HAFixAgent with enhanced prompts and blame context
        log.debug(f"Calling run_hafix_agent_repair with shared docker_env: {bool(shared_docker_env)}")
        repair_result = run_hafix_agent_repair(
            project_name, bug_id, bug_info, blame_info, history_category,
            docker_env=shared_docker_env, config=config, output_dir=output_dir,
            bug_category=bug_category, logger=log,
            selector_type=selector_type, n_lines=n_lines,
            include_blameless=include_blameless
        )
        
        log.info(f"Repair completed: success={repair_result.get('success', False)}, "
                  f"exit_status={repair_result.get('exit_status', 'unknown')}, "
                  f"runtime={repair_result.get('runtime', 0):.1f}s")

        # Save results
        log.phase(4, "Saving results")

        # Move blame_failure_reason inside blame_info
        if blame_info is None:
            blame_info = {}
        blame_info["blame_failure_reason"] = blame_failure_reason

        result_data = {
            "project_name": project_name,
            "bug_id": bug_id,
            "history_category": history_category.name,
            "repair_result": repair_result,
            "timestamp": get_timestamp(),
            "selector_type": selector_type,
            "n_lines": n_lines,
            "bug_info": bug_info,
            "blame_info": blame_info,
        }
        
        log.debug(f"Writing results to: {result_file}")
        with open(result_file, 'w') as f:
            json.dump(result_data, f, indent=2, default=str)
        
        log.success(f"Bug processing completed successfully: {project_bug}")

        # Record repair result for runtime progress tracking
        progress_manager.record_repair_result(project_bug, repair_result)

        progress_manager.update_bug_status(
            project_bug, history_category.name, "completed",
            repair_success=repair_result.get('success', False),
            exit_status=repair_result.get('exit_status', 'Unknown')
        )
        
    except Exception as e:
        error_msg = f"Error processing {project_bug}: {str(e)}"
        log.error(error_msg)
        log.debug(f"Full traceback: {tb.format_exc()}")

        # Classify the type of failure
        error_str = str(e).lower()
        is_fundamental_failure = (
            "bug not found" in error_str or
            "project not found" in error_str or
            "invalid bug" in error_str or
            "checkout" in error_str and "failed" in error_str or
            "docker" in error_str and ("setup" in error_str or "failed" in error_str)
        )

        # Save error info
        log.info("Saving error information to result file")
        error_data = {
            "project_name": project_name,
            "bug_id": bug_id,
            "error": str(e),
            "traceback": tb.format_exc(),
            "timestamp": get_timestamp()
        }

        with open(result_file, 'w') as f:
            json.dump(error_data, f, indent=2)

        if is_fundamental_failure:
            # Fundamental setup failure - couldn't even start processing
            log.error(f"Fundamental failure for {project_bug} - marking as execution failed")
            progress_manager.update_bug_status(project_bug, history_category.name, "failed")
        else:
            # Processing started but encountered error - mark as completed execution but failed repair
            log.warning(f"Execution error for {project_bug} - marking as completed execution with failed repair")

            # Create a synthetic repair result for progress tracking
            synthetic_repair_result = {
                "success": False,
                "exit_status": "ExecutionError",
                "error": str(e),
                "model_cost": 0.0,
                "token_usage": {}
            }

            # Record this as a failed repair (but completed execution)
            progress_manager.record_repair_result(project_bug, synthetic_repair_result)
            progress_manager.update_bug_status(project_bug, history_category.name, "completed",
                                              repair_success=False, exit_status="ExecutionError")
    
    finally:
        # Clean up logger to prevent handler leaks
        if 'log' in locals():
            log.cleanup()
        
        # Clean up shared docker environment after all stages are complete
        if 'shared_docker_env' in locals() and shared_docker_env:
            if docker_params.get('cleanup_on_exit', True):
                log.debug("Cleaning up shared Docker container")
                shared_docker_env.cleanup()
            else:
                log.debug("Keeping shared Docker container for debugging")


def run_hafix_agent_repair(
    project_name: str,
    bug_id: int,
    bug_info: Dict,
    blame_info: Optional[Dict],
    history_category: HistoryCategory,
    docker_env = None,
    config: Optional[Dict] = None,
    output_dir: Optional[Path] = None,
    bug_category: str = "single_hunk",
    logger = None,
    selector_type: str = "llm_judge",
    n_lines: int = 1,
    include_blameless: bool = True
) -> Dict:
    """
    Run HAFixAgent with blame-augmented prompts for a specific bug.
    
    Args:
        logger:
        project_name: Defects4J project name
        bug_id: Bug ID number
        bug_info: Bug information with fault locations
        blame_info: Git blame context information
        history_category: Which history heuristic to use
        docker_env: Optional Docker container from blame extraction (reused if provided)
        config: Optional YAML config dict (if None, loads from file)
        output_dir: Add output_dir parameter for trajectory saving
        bug_category: bug category
        selector_type:
        n_lines:
        include_blameless:
        
    Returns:
        Repair result dictionary with success status and details
    """

    try:
        start_time = time.time()
        # Use passed config or load cached config if not provided (for backward compatibility)
        if config is None:
            config = get_config()
        model = get_model(None, config.get("model", {}))

        # Add token tracking to model (optional, continue if it fails)
        try:
            add_token_tracking_to_model(model)
        except Exception as e:
            if logger:
                logger.warning(f"Token tracking setup failed: {e}")
                logger.info("Continuing without token tracking...")

        # Ensure Docker container is ready (shared or create new)
        env, error = ensure_defects4j_docker_container(project_name, str(bug_id), docker_env)
        if error:
            return {
                "success": False,
                "error": "docker_setup_failed",
                "checkout_output": error.get('error', 'Docker setup failed'),
                "runtime": round(time.time() - start_time),
            }

        # Set working directory for the container
        repo_path = bug_info.get('repo_path', get_defects4j_work_dir(project_name, str(bug_id)))
        env.config.cwd = repo_path

        try:
            # Build enhanced prompts with blame context using YAML templates
            prompt_data = build_hafix_prompt(
                bug_info=bug_info,
                history_category=history_category, 
                blame_info=blame_info,
                config=config  # Pass already-loaded config instead of path
            )
            
            # Initialize HAFixAgent using YAML config values
            agent_config = config.get('agent', {})
            agent = HAFixAgent(
                model=model,
                env=env,
                logger=logger,  # Pass logger to capture console output
                system_template=prompt_data["system_template"],
                instance_template=prompt_data["instance_template"],
                step_limit=agent_config.get('step_limit', 50),
                cost_limit=agent_config.get('cost_limit', 1.0),
                # Include other template overrides from YAML (only if they exist)
                action_observation_template=agent_config.get('action_observation_template'),
                format_error_template=agent_config.get('format_error_template'),
                timeout_template=agent_config.get('timeout_template')
            )
            # Set model config for adaptive context loading (RQ2)
            agent.model_config = config.get('model', {})

            # Set bug_info for adaptive context loading (required for hafix-context command)
            agent.bug_info = bug_info

            # Set selector and n_lines for RQ2 adaptive extraction (defaults: llm_judge, 1)
            agent.adaptive_selector_type = selector_type
            agent.adaptive_n_lines = n_lines
            agent.adaptive_include_blameless = include_blameless  # Pass from blame_category config

            # Set template variables from prompt builder (includes all bug context + history)
            agent.extra_template_vars.update(prompt_data["template_vars"])
            
            # Run the agent - no task description needed, all info is in templates
            if logger:
                logger.info(f"🤖 Starting HAFixAgent execution for {project_name}_{bug_id}...")
                logger.info(f"📋 Agent config: step_limit={agent_config.get('step_limit', 50)}, cost_limit=${agent_config.get('cost_limit', 1.0)}")

            exit_status, message = agent.run()

            if logger:
                logger.info(f"✅ Agent execution completed: {exit_status}")

            # Analyze results - "Submitted" means the agent successfully completed the task
            success = (exit_status == "Submitted")

            # Log execution summary
            steps_taken = len(agent.messages) // 2
            if logger:
                logger.info(f"📊 Execution summary:")
                logger.info(f"   • Steps taken: {steps_taken}")
                logger.info(f"   • Model calls: {model.n_calls}")
                logger.info(f"   • Total cost: ${model.cost:.4f}")
                logger.info(f"   • Success: {success}")
                if not success:
                    logger.info(f"   • Exit reason: {exit_status}")
                    logger.info(f"   • Last message: {str(message)[:100]}..." if message else "   • No final message")

            # Save trajectory for analysis
            save_trajectory_safe(
                agent, output_dir, bug_category, project_name, bug_id,
                history_category, exit_status, message
            )

            # Extract execution metrics
            metrics = extract_execution_metrics(start_time, model, agent, env)

            return {
                "success": success,
                "exit_status": exit_status,
                "message": str(message)[:1000] if message and str(message).strip() else None,
                **metrics
            }
            
        finally:
            # Cleanup container unless we're reusing one from previous stages
            if not docker_env and 'env' in locals():
                # Check config for cleanup setting
                env_config = config.get('environment', {})
                if env_config.get('cleanup_on_exit', True):
                    env.cleanup()
                else:
                    if logger:
                        logger.debug("Keeping agent container for debugging")
                
    except Exception as e:
        # Save trajectory if agent exists
        if 'agent' in locals() and 'output_dir' in locals() and output_dir:
            save_trajectory_safe(
                agent, output_dir, bug_category, project_name, bug_id,
                history_category, "ExecutionError", str(e)
            )

        # Extract execution metrics
        metrics = extract_execution_metrics(
            start_time if 'start_time' in locals() else time.time(),
            model if 'model' in locals() else None,
            agent if 'agent' in locals() else None,
            env if 'env' in locals() else None
        )

        # Return in exact same format as successful runs
        return {
            "success": False,
            "exit_status": "ExecutionError",
            "message": str(e),
            "traceback": tb.format_exc(),
            **metrics
        }


# fmt: off
@app.command(help=_HELP_TEXT)
def main(
    bug_category: str = typer.Option("single_file_multi_hunk", "--bug-category", help="Bug category filter: 'single_line', 'single_hunk', 'single_file_multi_hunk', 'multi_file_multi_hunk', or 'all'", rich_help_panel="Bug Selection"),
    history_category: str = typer.Option("baseline", "--history", help="History heuristic: baseline,cfn_modified,cfn_all,fn_modified,fn_all,fln_all,fn_pair,fl_diff", rich_help_panel="Blame Context Category"),
    selector_type: str = typer.Option("first", "--selector-type", help="Line selection strategy: 'first', 'random', 'llm_judge'", rich_help_panel="Multi-Line Selection"),
    n_lines: int = typer.Option(1, "--n-lines", help="Number of lines to select for blame (1-10)", rich_help_panel="Multi-Line Selection"),
    # Cache mode parameters
    context_mode: str = typer.Option("runtime", "--context-mode", help="Context loading mode: 'runtime' (extract at runtime) or 'cached' (use pre-extracted data)", rich_help_panel="Performance"),
    cache_dataset_path: str = typer.Option("dataset/defects4j", "--cache-dataset-path", help="Dataset path for cached contexts (only used with --context-mode=cached)", rich_help_panel="Performance"),
    #blame
    blame_category: str = typer.Option("both", "--blame-category", help="Which bugs to evaluate: 'blameable' (only blameable bugs), 'blameless' (only blameless bugs), 'both' (all bugs)", rich_help_panel="Blame Context Category"),

    config: str = typer.Option("config/defects4j.yaml", "--config", help="Path to config YAML file (use config/defects4j_adaptive.yaml for RQ2)", rich_help_panel="Basic"),
    output: str = typer.Option("results/defects4j", "-o", "--output", help="Output directory", rich_help_panel="Basic"),
    project_filter: str = typer.Option("", "--project", help="Specific project to evaluate (e.g., 'Math', 'Lang')", rich_help_panel="Bug Selection"),
    custom_bugs: str = typer.Option("", "--custom-bugs", help="Comma-separated list of specific bugs to run (e.g., 'Math_91,Lang_58,Cli_24')", rich_help_panel="Bug Selection"),
    limit: int = typer.Option(0, "--limit", help="Maximum number of bugs to evaluate (0 = no limit)", rich_help_panel="Bug Selection"),
    workers: int = typer.Option(1, "-w", "--workers", help="Number of worker threads", rich_help_panel="Basic")

) -> None:
    # fmt: on
    """Run HAFixAgent with blame context on Defects4J bugs."""

    # Set config path (must be called before any get_config() calls)
    set_config_path(config)

    # Parse history category - support both names and legacy numbers
    try:
        if history_category.isdigit():
            # Legacy number support
            history_cat = HistoryCategory(int(history_category))
            console.print(f"[yellow]Using legacy number {history_category}. Consider using named parameters.[/yellow]")
        elif history_category in history_name_to_category:
            # New named parameter support
            history_cat = history_name_to_category[history_category]
        else:
            # Try direct enum lookup
            history_cat = HistoryCategory[history_category]
    except (KeyError, ValueError):
        console.print(f"[red]Invalid history category: {history_category}[/red]")
        console.print("Available categories:")
        for cat in HistoryCategory:
            console.print(f"  {cat.value}: {cat.name}")
        return

    # RQ2 Adaptive Mode: Auto-configure parameters
    if history_cat == HistoryCategory.adaptive:
        console.print("[cyan]🔄 RQ2 Adaptive Mode Activated[/cyan]")
        console.print("[cyan]   On-demand historical context loading enabled[/cyan]")

        # Auto-set selector_type
        if selector_type != "llm_judge":
            console.print(f"[yellow]   → Auto-setting selector_type: '{selector_type}' → 'llm_judge'[/yellow]")
            selector_type = "llm_judge"

        # Auto-set n_lines
        if n_lines != 1:
            console.print(f"[yellow]   → Auto-setting n_lines: {n_lines} → 1[/yellow]")
            n_lines = 1

        # Auto-set context_mode
        if context_mode != "runtime":
            console.print(f"[yellow]   → Auto-setting context_mode: '{context_mode}' → 'runtime'[/yellow]")
            context_mode = "runtime"

        # Auto-set blame_category
        if blame_category != "both":
            console.print(f"[yellow]   → Auto-setting blame_category: '{blame_category}' → 'both'[/yellow]")
            blame_category = "both"

    # Validate parameters
    if selector_type not in ["first", "random", "llm_judge"]:
        console.print(f"[red]Invalid selector type: {selector_type}[/red]")
        return

    if not (1 <= n_lines <= 10):
        console.print(f"[red]Invalid n_lines: {n_lines}. Must be 1-10[/red]")
        return

    # Validate context mode
    if context_mode not in ["runtime", "cached"]:
        console.print(f"[red]Invalid context mode: {context_mode}. Use 'runtime' or 'cached'[/red]")
        return

    # Validate blame category
    if blame_category not in ["blameable", "blameless", "both"]:
        console.print(f"[red]Invalid blame category: {blame_category}. Use 'blameable', 'blameless', or 'both'[/red]")
        return

    # Simple cache validation for cached mode
    if context_mode == "cached":
        console.print(f"[green]Using cached mode with {cache_dataset_path}[/green]")

    # Validate custom_bugs have consistent categories
    if custom_bugs:
        console.print(f"[cyan]Using custom bug list: {custom_bugs}[/cyan]")
        console.print(f"[yellow]Important: Make sure all custom bugs belong to the same category ({bug_category})[/yellow]")
        console.print(f"[yellow]Categories are determined by defects4j_blame_feasibility.csv[/yellow]")

        # Parse and validate custom bugs early - batch validation for efficiency
        custom_bug_list = []
        invalid_bugs = []
        category_mismatches = []

        for bug_str in [bug.strip() for bug in custom_bugs.split(",")]:
            if "_" not in bug_str:
                invalid_bugs.append(f"{bug_str} (expected format: Project_ID)")
                continue

            project_name, bug_id_str = bug_str.split("_", 1)
            try:
                bug_id = int(bug_id_str)
                custom_bug_list.append(bug_str)

                # Look up the actual category for this bug
                actual_category = Defects4JExtractor.get_bug_category(project_name, bug_id)
                if actual_category is None:
                    invalid_bugs.append(f"{bug_str} (not found in defects4j_blame_feasibility.csv)")
                elif actual_category != bug_category:
                    category_mismatches.append(f"{bug_str} (has category '{actual_category}', expected '{bug_category}')")

            except ValueError:
                invalid_bugs.append(f"{bug_str} (invalid bug ID format)")

        # Report all validation errors at once
        if invalid_bugs or category_mismatches:
            if invalid_bugs:
                console.print(f"[red]Invalid bugs found:[/red]")
                for bug in invalid_bugs:
                    console.print(f"  - {bug}")
            if category_mismatches:
                console.print(f"[red]Category mismatches:[/red]")
                for bug in category_mismatches:
                    console.print(f"  - {bug}")
                console.print(f"[yellow]Please ensure all custom bugs match the specified category filter[/yellow]")
            return

    # Setup output directory with selector_type, n-lines, and bug category subdirectories
    base_output_path = Path(output)
    # Construct path: base / {selector_type}_{n_lines}line / {bug_category}
    selector_dir = f"{selector_type}_{n_lines}line"
    output_path = base_output_path / selector_dir / bug_category
    output_path.mkdir(parents=True, exist_ok=True)
    
    console.print(f"[green]HAFixAgent Defects4J Evaluation[/green]")
    console.print(f"Output: {output_path}")
    console.print(f"History category: {history_cat.name}")
    console.print(f"Multi-line: ({'LLM' if selector_type == 'llm_judge' else selector_type} selector, {n_lines} lines)")
    console.print("[dim]Container-based execution[/dim]")
    
    # Get bugs to evaluate
    console.print(f"\n[yellow]Getting bugs to evaluate...[/yellow]")

    if custom_bugs:
        # Parse custom bug list (already validated above)
        bugs_to_evaluate = []
        for bug_str in custom_bug_list:
            project_name, bug_id = bug_str.split("_", 1)
            bug_id = int(bug_id)
            bugs_to_evaluate.append((project_name, bug_id))
    else:
        # Use normal filtering based on blame category
        if blame_category == "both":
            # Get both blameable and blameless bugs
            blameable_bugs, blameless_bugs = Defects4JExtractor.get_filtered_bugs_with_blameless(
                bug_category, project_filter, limit, include_blameless=True
            )
            bugs_to_evaluate = blameable_bugs + blameless_bugs
            console.print(f"[cyan]All bugs: {len(blameable_bugs)} blameable + {len(blameless_bugs)} blameless bugs[/cyan]")
        elif blame_category == "blameless":
            # Get only blameless bugs
            _, blameless_bugs = Defects4JExtractor.get_filtered_bugs_with_blameless(
                bug_category, project_filter, limit, include_blameless=True
            )
            bugs_to_evaluate = blameless_bugs
            console.print(f"[cyan]Blameless only: {len(blameless_bugs)} bugs[/cyan]")
        else:  # blame_category == "blameable"
            # Get only blameable bugs (default)
            bugs_to_evaluate = Defects4JExtractor.get_filtered_bugs(bug_category, project_filter, limit, blame_suitable_only=True)
            console.print(f"[cyan]Blameable only: {len(bugs_to_evaluate)} bugs[/cyan]")

    console.print(f"[green]Found {len(bugs_to_evaluate)} bugs to evaluate[/green]")

    if not bugs_to_evaluate:
        console.print("[red]No bugs found matching criteria[/red]")
        return

    # Initialize progress manager with history and blame-category specific filename
    progress_manager = EvaluationProgressManager(
        len(bugs_to_evaluate),
        str(output_path / f"progress_{history_cat.value}_{blame_category}.json"),
        timezone=DEFAULT_TIMEZONE
    )
    
    console.print(f"\n[yellow]Starting evaluation with {workers} workers...[/yellow]")
    
    # Process bugs
    def process_futures(futures: dict[concurrent.futures.Future, str]):
        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()
            except Exception as e:
                project_bug = futures[future]
                console.print(f"[red]Error in future for {project_bug}: {e}[/red]")

    # Determine include_blameless flag based on blame_category
    include_blameless = blame_category in ["blameless", "both"]

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                process_defects4j_bug_with_blame,
                project_name,
                bug_id,
                output_path,
                history_cat,
                progress_manager,
                selector_type,
                n_lines,
                bug_category,
                context_mode,
                cache_dataset_path,
                include_blameless
            ): f"{project_name}_{bug_id}"
            for project_name, bug_id in bugs_to_evaluate
        }
        
        try:
            process_futures(futures)
        except KeyboardInterrupt:
            console.print("[yellow]Cancelling evaluation...[/yellow]")
            for future in futures:
                future.cancel()
    
    # Final summary
    elapsed = time.time() - progress_manager.start_time
    console.print(f"\n[green]Evaluation completed in {elapsed:.1f}s[/green]")
    console.print(f"Completed: {progress_manager.completed}")
    console.print(f"Failed: {progress_manager.failed}")

    # Show blame failure summary from runtime data
    if progress_manager.blame_failures:
        console.print(f"[yellow]Blame extraction failures: {len(progress_manager.blame_failures)}[/yellow]")
        console.print("Failed blame extractions:")
        for failure in progress_manager.blame_failures:
            console.print(f"  - {failure['project_bug']}: {failure['failure_reason']}")
    else:
        console.print(f"[green]All blame extractions successful[/green]")

    console.print(f"Results saved to: {output_path}")
    console.print(f"Progress data (includes blame failures): {output_path / f'progress_{history_cat.value}_{blame_category}.json'}")


if __name__ == "__main__":
    app()
