"""
Parallel runner for BIRCH-feedback replication with DeepSeek.
Runs d4j_code_repair_redwood.py across multiple Docker containers.

Usage:
    python run_birch_parallel.py --workers 4
    python run_birch_parallel.py --workers 8 --bugs "Chart_14,Math_1,Lang_5"
"""
import argparse
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

BIRCH_DIR = Path(__file__).resolve().parent.parent / "baselines" / "birch"
DATASET_PATH = BIRCH_DIR / "redwood" / "config" / "method_multihunk.json"
DOCKER_IMAGE = "birch-deepseek:latest"
DEFAULT_MODEL = "openrouter/deepseek/deepseek-v3.2-exp"


def load_bugs(dataset_path: Path, custom_bugs: str = None):
    """Load bug list from dataset JSON or custom list."""
    with open(dataset_path) as f:
        all_bugs = json.load(f)

    if custom_bugs:
        selected = [b.strip() for b in custom_bugs.split(",")]
        return [(b.rsplit("_", 1)[0], b.rsplit("_", 1)[1]) for b in selected if b in all_bugs]

    return [(k.rsplit("_", 1)[0], k.rsplit("_", 1)[1]) for k in sorted(all_bugs.keys())]


def get_completed_bugs(results_dir: Path):
    """Read already-completed bugs from the CSV to enable resume."""
    csv_path = results_dir / "test_results_mode_4.csv"
    if not csv_path.exists():
        return set()
    completed = set()
    with open(csv_path) as f:
        for line in f:
            if line.startswith("bug,"):
                continue
            bug = line.split(",")[0].strip()
            if bug:
                completed.add(bug)
    return completed


def run_bug(project: str, bug_id: str, model: str, env_key_val: str,
            api_base: str = "https://openrouter.ai/api/v1", api_key: str = ""):
    """Run BIRCH on a single bug inside a Docker container with isolated results."""
    bug_label = f"{project}-{bug_id}"
    # Each bug gets its own results directory to avoid CSV write conflicts
    results_path = f"/birch/redwood/results/mode_4_model_{model.replace('/', '_')}/{project}_{bug_id}"
    cmd = [
        "docker", "run", "--rm",
        "-v", f"{BIRCH_DIR}:/birch",
        "-v", f"{BIRCH_DIR}/work_dir:/tmp/WORK_DIR",
        "-v", f"{BIRCH_DIR}/work_dir_fixed:/tmp/WORK_DIR_FIXED",
        "-e", f"OPENROUTER_API_KEY={env_key_val}",
        # For openai/<model> (local vLLM): litellm reads these. Ignored by the
        # openrouter/<model> default path, so DeepSeek behavior is unchanged.
        "-e", f"OPENAI_API_KEY={api_key or env_key_val}",
        "-e", f"OPENAI_BASE_URL={api_base}",
        "-e", f"OPENAI_API_BASE={api_base}",
        "-e", "PYTHONPATH=/birch:/birch/birch",
        "-w", "/birch/redwood",
        DOCKER_IMAGE,
        "python", "d4j_code_repair_redwood.py",
        "--model", model,
        "--project", project,
        "--bug_id", bug_id,
        "--mode", "4",
        "--multihunk", "yes",
        "--scope", "method",
        "--method", "rag",
        "--max_iterations", "3",
        "--checkout_dir", "/tmp/WORK_DIR",
        "--fixed_dir", "/tmp/WORK_DIR_FIXED",
        "--results_path", results_path,
    ]
    # Local vLLM endpoint: pass it explicitly via redwood's --api_host (threaded to
    # invoke_llm -> litellm api_base). The default OpenRouter path passes nothing, so
    # DeepSeek behavior (openrouter/<model>, native litellm routing) is unchanged.
    if "openrouter.ai" not in api_base:
        cmd += ["--api_host", api_base]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)  # 30 min timeout
        passed = "Yes" in result.stdout and "compile_fail" not in result.stdout.lower()
        return bug_label, "OK", result.returncode
    except subprocess.TimeoutExpired:
        return bug_label, "TIMEOUT", -1
    except Exception as e:
        return bug_label, f"ERROR: {e}", -1


def main():
    parser = argparse.ArgumentParser(description="Parallel BIRCH-feedback runner")
    parser.add_argument("--workers", type=int, default=4, help="Number of parallel workers")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help="LLM model name")
    parser.add_argument("--bugs", type=str, default=None, help="Comma-separated bug list (e.g., Chart_14,Math_1)")
    parser.add_argument("--resume", action="store_true", help="Skip already-completed bugs")
    parser.add_argument("--api-base", type=str, default="https://openrouter.ai/api/v1",
                        help="OpenAI-compatible model-serving endpoint base URL (e.g. http://localhost:8000/v1 for local vLLM)")
    parser.add_argument("--api-key", type=str, default="",
                        help="API key for the endpoint (use any non-empty value for local vLLM)")
    args = parser.parse_args()

    # Load API key (OpenRouter for the default DeepSeek path)
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
    api_key = os.environ.get("OPENROUTER_API_KEY", "")
    if not api_key and not args.api_key:
        print("ERROR: set OPENROUTER_API_KEY in .env or pass --api-key")
        sys.exit(1)

    # Load bugs
    bugs = load_bugs(DATASET_PATH, args.bugs)
    print(f"Total bugs in dataset: {len(bugs)}")

    # Resume support
    if args.resume:
        model_tag = args.model.replace("/", "_")
        results_base = BIRCH_DIR / "redwood" / "results" / f"mode_4_model_{model_tag}"
        completed_set = set()
        if results_base.exists():
            for d in results_base.iterdir():
                if d.is_dir() and (d / "test_results_mode_4.csv").exists():
                    # Directory name is Project_BugId
                    completed_set.add(d.name.replace("_", "-", 1))  # Chart_14 → Chart-14
        before = len(bugs)
        bugs = [(p, b) for p, b in bugs if f"{p}-{b}" not in completed_set]
        print(f"Resuming: {before - len(bugs)} already done, {len(bugs)} remaining")

    print(f"Running {len(bugs)} bugs with {args.workers} workers using {args.model}")
    print("-" * 60)

    completed = 0
    failed = 0

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(run_bug, project, bug_id, args.model, api_key,
                            args.api_base, args.api_key): (project, bug_id)
            for project, bug_id in bugs
        }

        for future in as_completed(futures):
            bug_label, status, rc = future.result()
            completed += 1
            if rc != 0:
                failed += 1
            print(f"[{completed}/{len(bugs)}] {bug_label}: {status} (rc={rc})")

    print(f"\nDone. Completed: {completed}, Failed: {failed}")

    # Merge per-bug CSVs into a single results file
    model_tag = args.model.replace("/", "_")
    per_bug_base = BIRCH_DIR / "redwood" / "results" / f"mode_4_model_{model_tag}"
    # Write merged CSV to HAFixAgent results dir (we own this, avoids Docker root permission issues).
    # Model-specific dir; keep the documented "birch_deepseek" for the default DeepSeek run,
    # isolate other models (e.g. Qwen) under birch_<model_tag> so DeepSeek results are untouched.
    merged_name = "birch_deepseek" if "deepseek-v3.2-exp" in args.model else f"birch_{model_tag}"
    merged_dir = Path(__file__).resolve().parent.parent / "results" / "defects4j" / "baselines" / merged_name
    merged_dir.mkdir(parents=True, exist_ok=True)
    merged_csv = merged_dir / "test_results_mode_4_merged.csv"
    header_written = False
    total_pass = 0
    total_bugs_counted = 0

    with open(merged_csv, "w") as out:
        # Scan all per-bug directories (not just bugs from this run)
        if per_bug_base.exists():
            for bug_dir in sorted(per_bug_base.iterdir()):
                csv_path = bug_dir / "test_results_mode_4.csv"
                if not bug_dir.is_dir() or not csv_path.exists():
                    continue
                last_line = ""
                with open(csv_path) as f:
                    for line in f:
                        if line.startswith("bug,"):
                            if not header_written:
                                out.write(line)
                                header_written = True
                            continue
                        out.write(line)
                        last_line = line
                if last_line and "Yes" in last_line.split(",")[1]:
                    total_pass += 1
                if last_line:
                    total_bugs_counted += 1

    print(f"\nMerged results: {merged_csv}")
    print(f"Pass: {total_pass}/{total_bugs_counted} ({total_pass/total_bugs_counted*100:.1f}%)" if total_bugs_counted > 0 else "No results")

if __name__ == "__main__":
    main()
