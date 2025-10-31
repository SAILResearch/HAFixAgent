#!/usr/bin/env python3
"""
Run HAFixAgent on Defects4J bugs.
Based on mini-swe-agent's run pattern but adapted for Defects4J.
"""

import json
import traceback
from pathlib import Path
from typing import Any, Dict, Optional
import typer
import yaml
from rich.console import Console
from minisweagent.environments.docker import DockerEnvironment
from minisweagent.models import get_model
from minisweagent.run.utils.save import save_traj
from minisweagent.utils.log import logger

from hafix_agent.agents.hafix_agent import HAFixAgent

# Default paths
DEFAULT_CONFIG = Path("configs/d4j_deepseek.litellm.yaml")
DEFAULT_DATASET = Path("dataset/defects4j/bugs.json")
DEFAULT_OUTPUT_DIR = Path("results")

console = Console(highlight=False)
app = typer.Typer(rich_markup_mode="rich")

_HELP_TEXT = """Run HAFixAgent on Defects4J bugs with perfect fault localization.

[bold green]Examples:[/bold green]

Run on a specific bug:
  [cyan]python scripts/run_hafix.py --bug Lang_1[/cyan]

Run on all multi-hunk bugs:
  [cyan]python scripts/run_hafix.py --multi-hunk[/cyan]

Run with custom config:
  [cyan]python scripts/run_hafix.py --bug Lang_1 --config configs/custom.yaml[/cyan]
"""


def load_bug_data(dataset_path: Path) -> Dict[str, Any]:
    """Load bug dataset with fault locations."""
    with open(dataset_path, 'r') as f:
        bugs_data = json.load(f)

    # Create a mapping from bug_id to bug_info
    bugs = {}
    for bug in bugs_data:
        bug_id = f"{bug['project']}_{bug['bug_id']}"
        bugs[bug_id] = {
            "bug_id": bug_id,
            "project": bug['project'],
            "bug_num": bug['bug_id'],
            "fault_locations": bug.get('fault_locations', []),
            "failing_tests": bug.get('failing_tests', []),
            "description": bug.get('description', ''),
            "is_multi_hunk": len(bug.get('fault_locations', [])) > 1
        }

    return bugs


def setup_docker_environment(config: Dict[str, Any], bug_info: Dict[str, Any]) -> DockerEnvironment:
    """Setup Docker environment for a specific bug."""
    env_config = config.get('environment', {})

    # Create environment with proper settings
    env = DockerEnvironment(
        image=env_config.get('image', 'defects4j:latest'),
        persistent_container=not env_config.get('container_per_bug', False),
        container_name=f"hafix_{bug_info['bug_id']}",
        env_vars=env_config.get('env', {})
    )

    # Setup and checkout the bug
    env.setup()

    # Checkout the buggy version
    checkout_cmd = (
        f"cd /workspace && "
        f"defects4j checkout -p {bug_info['project']} -v {bug_info['bug_num']}b -w ."
    )
    result = env.execute(checkout_cmd)

    if "error" in result.lower() and "already exists" not in result.lower():
        logger.error(f"Failed to checkout bug: {result}")
        raise RuntimeError(f"Failed to checkout {bug_info['bug_id']}")

    # Initial compilation
    compile_result = env.execute("cd /workspace && defects4j compile")
    if "BUILD FAILED" in compile_result:
        logger.warning("Initial compilation failed (expected for some bugs)")

    return env


def run_single_bug(
        bug_id: str,
        config: Dict[str, Any],
        bugs_data: Dict[str, Any],
        output_dir: Path,
        cost_limit: Optional[float] = None
) -> Dict[str, Any]:
    """Run HAFixAgent on a single bug."""

    # Get bug info
    if bug_id not in bugs_data:
        raise ValueError(f"Bug {bug_id} not found in dataset")

    bug_info = bugs_data[bug_id]
    console.print(f"[bold cyan]Running on {bug_id}[/bold cyan]")
    console.print(f"  Project: {bug_info['project']}")
    console.print(f"  Multi-hunk: {bug_info['is_multi_hunk']}")
    console.print(f"  Fault locations: {len(bug_info['fault_locations'])}")

    # Override cost limit if provided
    if cost_limit is not None:
        config.setdefault('agent', {})['cost_limit'] = cost_limit

    # Initialize model
    model_config = config.get('model', {})
    model = get_model(model_config.get('model_name'), model_config)

    # Setup environment
    console.print("[yellow]Setting up Docker environment...[/yellow]")
    env = setup_docker_environment(config, bug_info)

    # Initialize agent
    agent = HAFixAgent(model, env, **config.get('agent', {}))

    # Create task description
    task = f"""Fix the bug in {bug_info['project']} (bug #{bug_info['bug_num']}).
The project is checked out at /workspace.
{bug_info.get('description', '')}"""

    # Run agent
    console.print("[yellow]Running agent...[/yellow]")
    exit_status, result, extra_info = None, None, None

    try:
        exit_status, result = agent.run(bug_info=bug_info)

        # Check if bug was fixed
        if exit_status == "Submitted":
            # Validate fix by running tests
            test_result = env.execute("cd /workspace && defects4j test")
            if "Failing tests: 0" in test_result:
                console.print(f"[bold green]✓ {bug_id} FIXED![/bold green]")
                extra_info = {"status": "fixed", "test_output": test_result}
            else:
                console.print(f"[bold red]✗ {bug_id} tests still failing[/bold red]")
                extra_info = {"status": "not_fixed", "test_output": test_result}
        else:
            console.print(f"[bold yellow]⚠ {bug_id} {exit_status}[/bold yellow]")
            extra_info = {"status": exit_status}

    except Exception as e:
        logger.error(f"Error running agent: {e}", exc_info=True)
        exit_status, result = type(e).__name__, str(e)
        extra_info = {"traceback": traceback.format_exc()}
        console.print(f"[bold red]✗ {bug_id} ERROR: {e}[/bold red]")

    finally:
        # Save trajectory
        output_file = output_dir / f"{bug_id}.traj.json"
        save_traj(agent, output_file, exit_status=exit_status, result=result, extra_info=extra_info)

        # Cleanup environment if not persistent
        if config.get('environment', {}).get('container_per_bug', False):
            env.cleanup()

    return {
        "bug_id": bug_id,
        "exit_status": exit_status,
        "result": result,
        "extra_info": extra_info
    }


# fmt: off
@app.command(help=_HELP_TEXT)
def main(
        bug: Optional[str] = typer.Option(None, "--bug", "-b", help="Specific bug ID (e.g., Lang_1)"),
        multi_hunk: bool = typer.Option(False, "--multi-hunk", help="Run on all multi-hunk bugs"),
        all_bugs: bool = typer.Option(False, "--all", help="Run on all bugs in dataset"),
        config_path: Path = typer.Option(DEFAULT_CONFIG, "--config", "-c", help="Path to config file"),
        dataset_path: Path = typer.Option(DEFAULT_DATASET, "--dataset", "-d", help="Path to bug dataset"),
        output_dir: Path = typer.Option(DEFAULT_OUTPUT_DIR, "--output", "-o", help="Output directory"),
        cost_limit: Optional[float] = typer.Option(None, "--cost-limit", "-l", help="Override cost limit per bug"),
        max_bugs: Optional[int] = typer.Option(None, "--max-bugs", help="Maximum number of bugs to process"),
) -> None:
    # fmt: on

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load configuration
    if not config_path.exists():
        console.print(f"[bold red]Config file not found: {config_path}[/bold red]")
        raise typer.Exit(1)

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Load bug dataset
    if not dataset_path.exists():
        console.print(f"[bold red]Dataset not found: {dataset_path}[/bold red]")
        raise typer.Exit(1)

    bugs_data = load_bug_data(dataset_path)
    console.print(f"[green]Loaded {len(bugs_data)} bugs from dataset[/green]")

    # Determine which bugs to run
    bugs_to_run = []

    if bug:
        # Single bug mode
        if bug not in bugs_data:
            console.print(f"[bold red]Bug {bug} not found in dataset[/bold red]")
            raise typer.Exit(1)
        bugs_to_run = [bug]

    elif multi_hunk:
        # Multi-hunk bugs only
        bugs_to_run = [
            bug_id for bug_id, info in bugs_data.items()
            if info['is_multi_hunk']
        ]
        console.print(f"[cyan]Found {len(bugs_to_run)} multi-hunk bugs[/cyan]")

    elif all_bugs:
        # All bugs
        bugs_to_run = list(bugs_data.keys())

    else:
        console.print("[bold red]Please specify --bug, --multi-hunk, or --all[/bold red]")
        raise typer.Exit(1)

    # Apply max_bugs limit if specified
    if max_bugs and len(bugs_to_run) > max_bugs:
        bugs_to_run = bugs_to_run[:max_bugs]
        console.print(f"[yellow]Limited to {max_bugs} bugs[/yellow]")

    # Run on selected bugs
    console.print(f"[bold green]Running HAFixAgent on {len(bugs_to_run)} bug(s)[/bold green]\n")

    results = []
    for i, bug_id in enumerate(bugs_to_run, 1):
        console.print(f"\n[bold]Progress: {i}/{len(bugs_to_run)}[/bold]")
        console.print("=" * 60)

        result = run_single_bug(
            bug_id=bug_id,
            config=config,
            bugs_data=bugs_data,
            output_dir=output_dir,
            cost_limit=cost_limit
        )
        results.append(result)

    # Print summary
    console.print("\n" + "=" * 60)
    console.print("[bold]SUMMARY[/bold]")
    console.print("=" * 60)

    fixed = sum(1 for r in results if r.get('extra_info', {}).get('status') == 'fixed')
    total = len(results)

    console.print(f"Total bugs: {total}")
    console.print(f"Fixed: [bold green]{fixed}[/bold green]")
    console.print(f"Not fixed: [bold red]{total - fixed}[/bold red]")
    console.print(f"Success rate: [bold]{100 * fixed / total if total > 0 else 0:.1f}%[/bold]")

    # Save summary
    summary_file = output_dir / "summary.json"
    with open(summary_file, 'w') as f:
        json.dump({
            "total": total,
            "fixed": fixed,
            "results": results
        }, f, indent=2)

    console.print(f"\n[green]Results saved to {output_dir}/[/green]")


if __name__ == "__main__":
    app()