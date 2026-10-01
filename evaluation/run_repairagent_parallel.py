"""
Parallel runner for RepairAgent replication with DeepSeek.
Runs RepairAgent across multiple Docker containers on all 854 Defects4J bugs.

Usage:
    python evaluation/run_repairagent_parallel.py --workers 16
    python evaluation/run_repairagent_parallel.py --workers 4 --bugs "Chart_14,Math_1,Lang_7"
"""
import argparse
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
REPAIR_AGENT_DIR = PROJECT_ROOT / "baselines" / "RepairAgent" / "repair_agent"
DEFECTS4J_DIR = PROJECT_ROOT / "vendor" / "defects4j"
DOCKER_IMAGE = "repairagent-deepseek:latest"
DEFAULT_MODEL = "deepseek/deepseek-v3.2-exp"

# All 17 Defects4J v3.0.1 projects
DEFECTS4J_PROJECTS = [
    "Chart", "Cli", "Closure", "Codec", "Collections", "Compress", "Csv",
    "Gson", "JacksonCore", "JacksonDatabind", "JacksonXml", "Jsoup",
    "JxPath", "Lang", "Math", "Mockito", "Time",
]


def discover_bugs(custom_bugs: str = None):
    """Discover all bugs from Defects4J patches or parse custom list."""
    if custom_bugs:
        selected = [b.strip() for b in custom_bugs.split(",")]
        return [(b.rsplit("_", 1)[0], b.rsplit("_", 1)[1]) for b in selected]

    bugs = []
    for project in DEFECTS4J_PROJECTS:
        patches_dir = DEFECTS4J_DIR / "framework" / "projects" / project / "patches"
        if not patches_dir.exists():
            continue
        for patch in sorted(patches_dir.glob("*.src.patch")):
            bug_id = patch.stem.replace(".src", "")
            bugs.append((project, bug_id))
    return bugs


def get_completed_bugs(results_dir: Path):
    """Find already-completed bugs from results directories."""
    completed = set()
    if not results_dir.exists():
        return completed
    for bug_dir in results_dir.iterdir():
        if bug_dir.is_dir():
            # Check if plausible_patches or logs exist (indicates completion)
            logs_dir = bug_dir / "logs"
            if logs_dir.exists() and any(logs_dir.iterdir()):
                completed.add(bug_dir.name)
    return completed


def _safe_docker_rm(container_name: str):
    """Force-remove a container; never raise (a hung `docker rm` must not crash the run)."""
    try:
        subprocess.run(["docker", "rm", "-f", "-v", container_name],
                       capture_output=True, timeout=60)
    except Exception:
        pass


def run_bug(project: str, bug_id: str, model: str, api_key: str, results_base: Path,
            api_base: str = "https://openrouter.ai/api/v1", max_tokens: int = 163840,
            image: str = DOCKER_IMAGE):
    """Run RepairAgent on a single bug inside a Docker container."""
    bug_label = f"{project}-{bug_id}"
    bug_results = results_base / f"{project}_{bug_id}"
    bug_results.mkdir(parents=True, exist_ok=True)

    # Pre-create experiment subdirectories (RepairAgent expects these)
    for subdir in ["logs", "responses", "external_fixes", "saved_contexts",
                    "mutations_history", "plausible_patches"]:
        (bug_results / subdir).mkdir(exist_ok=True)

    REPAIR_AGENT_DATA = PROJECT_ROOT / "baselines" / "RepairAgent" / "data"

    # Setup defects4j matching RepairAgent's expected structure:
    # 1. Copy buggy-lines and buggy-methods from RepairAgent data (mounted at /mnt/)
    # 2. Link framework from base image's /defects4j/ (already installed)
    # This mirrors their README: "git clone defects4j && cp -r ../data/buggy-lines defects4j"
    inner_cmd = (
        f"cd /app && "
        f"cp -r /mnt/buggy-lines defects4j/buggy-lines && "
        f"cp -r /mnt/buggy-methods defects4j/buggy-methods && "
        f"ln -s /defects4j/framework defects4j/framework && "
        f"ln -s /defects4j/project_repos defects4j/project_repos && "
        f"echo 'OPENAI_API_KEY='$OPENAI_API_KEY > .env && "
        f"echo 'OPENAI_API_BASE_URL='$OPENAI_API_BASE_URL >> .env && "
        f"cp .env autogpt/.env && "
        f"export PATH=$PATH:/app/defects4j/framework/bin && "
        f"export LANG=C.UTF-8 && export LC_COLLATE=C && "
        f"echo 'experiment_1' > experimental_setups/experiments_list.txt && "
        f"python3 construct_commands_descriptions.py && "
        f"python3 prepare_ai_settings.py {project} {bug_id} && "
        f"python3 checkout_py.py {project} {bug_id} && "
        f"python3 -m autogpt "
        f"--ai-settings ai_settings.yaml "
        f"--model {model} "
        f"-c -l 40 "
        f"-m json_file "
        f"--experiment-file hyperparams.json"
    )

    container_name = f"repairagent-{project.lower()}-{bug_id}"

    cmd = [
        "docker", "run", "--rm",
        "--name", container_name,
        "-v", f"{REPAIR_AGENT_DATA}/buggy-lines:/mnt/buggy-lines:ro",
        "-v", f"{REPAIR_AGENT_DATA}/buggy-methods:/mnt/buggy-methods:ro",
        "-v", f"{bug_results}:/app/experimental_setups/experiment_1",
        "-e", f"OPENAI_API_KEY={api_key}",
        "-e", f"OPENAI_API_BASE_URL={api_base}",
        "-e", f"OPENAI_API_BASE={api_base}",
        "-e", f"LOCAL_MODEL_NAME={model}",
        "-e", f"LOCAL_MODEL_MAX_TOKENS={max_tokens}",
        image,
        "bash", "-c", inner_cmd,
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=2700)  # 45 min
        # Check if any plausible patches were found
        plausible_dir = bug_results / "plausible_patches"
        has_plausible = plausible_dir.exists() and any(plausible_dir.iterdir())
        return bug_label, "OK", result.returncode, has_plausible
    except subprocess.TimeoutExpired:
        # Kill the container on timeout - subprocess.run only kills the client, not the container.
        # The cleanup itself must never raise (a hung `docker rm` would otherwise crash the run).
        _safe_docker_rm(container_name)
        plausible_dir = bug_results / "plausible_patches"
        has_plausible = plausible_dir.exists() and any(plausible_dir.iterdir())
        return bug_label, "TIMEOUT", -1, has_plausible
    except Exception as e:
        _safe_docker_rm(container_name)
        return bug_label, f"ERROR: {e}", -1, False


def main():
    parser = argparse.ArgumentParser(description="Parallel RepairAgent runner with DeepSeek")
    parser.add_argument("--workers", type=int, default=4, help="Number of parallel workers")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help="LLM model name")
    parser.add_argument("--bugs", type=str, default=None,
                        help="Comma-separated bug list (e.g., Chart_14,Math_1)")
    parser.add_argument("--resume", action="store_true", help="Skip already-completed bugs")
    parser.add_argument("--api-base", type=str, default="https://openrouter.ai/api/v1",
                        help="OpenAI-compatible model-serving endpoint base URL (e.g. http://localhost:8000/v1 for local vLLM)")
    parser.add_argument("--api-key", type=str, default="",
                        help="API key (overrides env; use any non-empty value for local vLLM)")
    parser.add_argument("--max-tokens", type=int, default=163840,
                        help="Model context window, for registry (default DeepSeek; Qwen3-Coder = 262144)")
    parser.add_argument("--image", type=str, default=DOCKER_IMAGE,
                        help="Docker image (use repairagent-local:latest for non-DeepSeek LLMs)")
    args = parser.parse_args()

    # API key: explicit --api-key wins; else the DeepSeek/OpenRouter env var (default path)
    from dotenv import load_dotenv
    load_dotenv(PROJECT_ROOT / ".env")
    api_key = args.api_key or os.environ.get("OPENROUTER_DEEPSEEK_API_KEY_REPAIRAGENT", "")
    if not api_key:
        print("ERROR: provide --api-key or set OPENROUTER_DEEPSEEK_API_KEY_REPAIRAGENT in .env")
        sys.exit(1)

    # Results directory
    model_tag = args.model.replace("/", "_")
    results_base = PROJECT_ROOT / "results" / "defects4j" / "baselines" / f"repairagent_{model_tag}"
    results_base.mkdir(parents=True, exist_ok=True)

    # Discover bugs
    bugs = discover_bugs(args.bugs)
    print(f"Total bugs discovered: {len(bugs)}")

    # Resume support
    if args.resume:
        completed_set = get_completed_bugs(results_base)
        before = len(bugs)
        bugs = [(p, b) for p, b in bugs if f"{p}_{b}" not in completed_set]
        print(f"Resuming: {before - len(bugs)} already done, {len(bugs)} remaining")

    print(f"Running {len(bugs)} bugs with {args.workers} workers using {args.model}")
    print("-" * 60)

    completed = 0
    failed = 0
    plausible = 0

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(run_bug, project, bug_id, args.model, api_key, results_base,
                            args.api_base, args.max_tokens, args.image): (project, bug_id)
            for project, bug_id in bugs
        }

        for future in as_completed(futures):
            bug_label, status, rc, has_plausible = future.result()
            completed += 1
            if rc != 0:
                failed += 1
            if has_plausible:
                plausible += 1
            print(f"[{completed}/{len(bugs)}] {bug_label}: {status} (rc={rc})"
                  f"{' [PLAUSIBLE]' if has_plausible else ''}")

    print(f"\nDone. Completed: {completed}, Failed: {failed}, Plausible: {plausible}")
    print(f"Results directory: {results_base}")

    # Generate summary CSV from all per-bug results (including previous runs)
    summary_csv = results_base / "repairagent_summary.csv"
    total_plausible = 0
    total_counted = 0
    with open(summary_csv, "w") as out:
        out.write("bug,plausible,cycles,fixes_attempted\n")
        for bug_dir in sorted(results_base.iterdir()):
            if not bug_dir.is_dir() or bug_dir.name.endswith(".csv"):
                continue
            ctx_files = list((bug_dir / "saved_contexts").glob("saved_context_*"))
            plausible_files = list((bug_dir / "plausible_patches").glob("*.json"))
            has_p = len(plausible_files) > 0
            cycles = 0
            fixes = 0
            if ctx_files:
                try:
                    import json
                    with open(ctx_files[0]) as f:
                        ctx = json.load(f)
                    cycles = ctx.get("cycle_count", 0)
                    fixes = len(ctx.get("suggested_fixes", []))
                except Exception:
                    pass
            out.write(f"{bug_dir.name},{'Yes' if has_p else 'No'},{cycles},{fixes}\n")
            if has_p:
                total_plausible += 1
            total_counted += 1

    print(f"\nSummary: {summary_csv}")
    print(f"Plausible: {total_plausible}/{total_counted}")


if __name__ == "__main__":
    main()
