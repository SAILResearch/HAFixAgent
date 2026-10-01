"""SBFL repair stage (Defects4J + BugsInPy) — RQ3.

Stage 2: reads the cached SBFL ranking produced by the FL stage
(`run_fl_defects4j.py` → GZoltar, or `run_fl_bugsinpy.py` → coverage engine), takes the top-N
suspicious lines as the fault localization, and runs the existing HARPER pipeline
(blame -> history -> agent) on it.

Replaces the developer-patch "perfect FL" of RQ1/RQ2 with realistic SBFL. Reuses the
existing extractor / blame / prompt / agent code read-only — only the source of
`fault_locations` changes. Dataset branching mirrors run_fl_sensitivity.py:204-240.

Usage:
    # validate wiring without API cost (baseline skips the LLM judge):
    python evaluation/run_sbfl_evaluation.py --dataset defects4j --history baseline --bugs Cli_1 --dry-run
    python evaluation/run_sbfl_evaluation.py --dataset bugsinpy --history baseline --bugs luigi_10 --dry-run
    # real run:
    python evaluation/run_sbfl_evaluation.py --dataset bugsinpy --history fl_diff --all --workers 8 --resume
"""

import argparse
import concurrent.futures
import json
import sys
import time
import traceback as tb
from pathlib import Path
from typing import Dict, List, Optional

from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent.parent / ".env", override=True)

import yaml
from rich.console import Console
from minisweagent.models import get_model

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "evaluation" / "sbfl"))

from hafix_agent.blame.core import (
    HistoryCategory, history_name_to_category,
    run_git_blame, find_function_containing_line, extract_all_history_context,
)
from hafix_agent.blame.patch_parser import PatchLine, PatchFormat
from hafix_agent.blame.selection import get_selector
from hafix_agent.prompts.prompt_builder import build_hafix_prompt
from hafix_agent.agents.hafix_agent import HAFixAgent
from hafix_agent.utils import (
    BugLogger, EvaluationProgressManager, extract_execution_metrics,
    save_trajectory_safe, get_timestamp as _get_timestamp, add_token_tracking_to_model,
    resolve_model_tag,
)
from parse_ranking import parse_gzoltar_ranking, parse_fauxpy_ranking  # noqa: E402
from hafix_agent.utils.model_specs import register_local_model_with_litellm

DEFAULT_TIMEZONE = "America/Toronto"
console = Console()

# bug_category is kept ONLY for post-hoc result grouping, never in the SBFL prompt.
_CATEGORY_MAP = {
    "SL": "single_line", "SH": "single_hunk",
    "SFMH": "single_file_multi_hunk", "MFMH": "multi_file_multi_hunk",
}

# Per-dataset cached-ranking filename (GZoltar vs coverage-engine FL stage output).
_RANKING_FILE = {"defects4j": "ochiai.ranking.csv", "bugsinpy": "Scores_Ochiai.csv"}

# One cached extractor per dataset for the (container-free) category lookup used to lay out
# results the same way RQ1 does (results/.../llm_judge_1line/<category>/<bug>/...).
_EXTRACTOR_CACHE: Dict[str, object] = {}


def _category_full(dataset: str, project_name: str, bug_id: str) -> str:
    """Full bug-category dir name (e.g. single_file_multi_hunk) — matches RQ1's layout."""
    ext = _EXTRACTOR_CACHE.get(dataset)
    if ext is None:
        if dataset == "defects4j":
            from dataset.defects4j.defects4j_extractor import Defects4JExtractor
            ext = Defects4JExtractor()
        else:
            from dataset.bugsinpy.bugsinpy_extractor import BugsInPyExtractor
            ext = BugsInPyExtractor()
        _EXTRACTOR_CACHE[dataset] = ext
    cat = ext.get_bug_category(project_name, int(bug_id) if dataset == "defects4j" else str(bug_id))
    return _CATEGORY_MAP.get(cat, cat or "unknown")


def read_line_from_container(docker_env, work_dir: str, file_path: str, line_number: int) -> str:
    """Read a single line from a file inside the container."""
    result = docker_env.execute(f"cd {work_dir} && sed -n '{line_number}p' {file_path}")
    if result.get("returncode") == 0:
        return result.get("output", "").rstrip("\n")
    return ""


def build_sbfl_fault_locations(suspicious) -> List[Dict]:
    """Convert ranked SuspiciousLine list into fault_location dicts (one per line)."""
    return [
        {
            "file": s.file,
            "start_line": s.line,
            "end_line": s.line,
            "suspiciousness": s.suspiciousness,
            "rank": s.rank,
        }
        for s in suspicious
    ]


def _create_container_with_retry(create_fn, log, attempts: int = 3, backoff: int = 5):
    """Create a container, retrying transient `docker run` timeouts (the daemon serializes
    creations under high concurrency, so a 60s timeout can fire in a burst). `create_fn`
    returns (docker_env, error); returns the same, or (None, last_error) after `attempts`.
    Only retries container creation — never API/repair errors."""
    last = None
    for i in range(attempts):
        try:
            docker_env, error = create_fn()
            if not error:
                return docker_env, None
            last = error
        except Exception as e:  # subprocess.TimeoutExpired / RuntimeError from docker run
            last = str(e)
        if i < attempts - 1:
            log.warning(f"Container creation failed (attempt {i+1}/{attempts}): "
                        f"{str(last)[:120]}; retrying in {backoff*(i+1)}s")
            time.sleep(backoff * (i + 1))
    return None, last


def process_sbfl_bug(
    project_name: str,
    bug_id: str,
    history_category: HistoryCategory,
    output_dir: Path,
    config: Dict,
    ranking_dir: Path,
    top_n: int,
    dataset: str = "defects4j",
    selector_type: str = "llm_judge",
    n_lines: int = 1,
    dry_run: bool = False,
    progress_manager: Optional[EvaluationProgressManager] = None,
) -> None:
    """Run one bug through the SBFL repair pipeline (defects4j or bugsinpy)."""
    bug_label = f"{project_name}_{bug_id}"
    bug_output_dir = output_dir / bug_label
    bug_output_dir.mkdir(parents=True, exist_ok=True)
    result_file = bug_output_dir / f"{bug_label}_{history_category.value}_result.json"
    log_file = bug_output_dir / f"{bug_label}_{history_category.value}.log"

    log = BugLogger(bug_label, log_file, history_category.name, timezone=DEFAULT_TIMEZONE)
    env_config = config.get("environment", {})
    docker_env = None

    try:
        log.info(f"=== SBFL repair: {bug_label}, {history_category.name}, top-{top_n} ===")

        # Step 1: load cached SBFL ranking + meta (GZoltar for D4J, coverage engine for BugsInPy)
        ranking_csv = ranking_dir / bug_label / _RANKING_FILE[dataset]
        meta_file = ranking_dir / bug_label / "meta.json"
        if not ranking_csv.exists():
            raise Exception(f"No cached ranking at {ranking_csv} (run FL stage first)")
        src_root = json.loads(meta_file.read_text()).get("src_root", "") if meta_file.exists() else ""
        parse_ranking = parse_gzoltar_ranking if dataset == "defects4j" else parse_fauxpy_ranking
        suspicious = parse_ranking(str(ranking_csv), src_root=src_root, top_n=top_n)
        if not suspicious:
            raise Exception("Ranking parsed to zero suspicious lines")
        log.info(f"Top-{top_n} suspicious: {suspicious[0].file}:{suspicious[0].line} (susp={suspicious[0].suspiciousness:.3f})")

        # Step 2: container + extractor (dataset-specific; mirrors run_fl_sensitivity.py)
        if dataset == "defects4j":
            from dataset.defects4j.util import (
                ensure_defects4j_docker_container, get_defects4j_work_dir,
            )
            from dataset.defects4j.defects4j_extractor import Defects4JExtractor

            docker_env, error = _create_container_with_retry(
                lambda: ensure_defects4j_docker_container(
                    project_name, str(bug_id), None,
                    image=env_config.get("image", "defects4j:latest"),
                    cleanup_on_exit=True,
                ), log)
            if error:
                raise Exception(f"Docker setup failed: {error}")
            work_dir = get_defects4j_work_dir(project_name, str(bug_id))
            extractor = Defects4JExtractor()
            patch_format = PatchFormat.REVERSE
        else:  # bugsinpy
            from dataset.bugsinpy.bugsinpy_extractor import (
                BugsInPyExtractor, ensure_bugsinpy_docker_container, get_bugsinpy_work_dir,
            )
            import subprocess

            image = f"bugsinpy:{project_name.lower()}_{bug_id}"
            if subprocess.run(["docker", "image", "inspect", image],
                              capture_output=True, text=True).returncode != 0:
                image = env_config.get("image", "bugsinpy_image:clean")
            docker_env, error = _create_container_with_retry(
                lambda: ensure_bugsinpy_docker_container(
                    project_name, str(bug_id), None, image=image, cleanup_on_exit=True,
                ), log)
            if error:
                raise Exception(f"Docker setup failed: {error}")
            work_dir = f"{get_bugsinpy_work_dir(project_name, str(bug_id))}/{project_name}"
            extractor = BugsInPyExtractor()
            patch_format = PatchFormat.FORWARD

        # Step 3: bug info (description / failing_tests / repo_path) — DISCARD its fault_locations
        bug_info = extractor._extract_bug_info_context(project_name, str(bug_id), docker_env=docker_env)
        if not bug_info or "error" in bug_info:
            raise Exception(f"Bug info extraction failed: {bug_info.get('error') if bug_info else 'none'}")

        # Step 4: override fault_locations with SBFL top-N
        bug_info["fault_locations"] = build_sbfl_fault_locations(suspicious)
        cat = extractor.get_bug_category(
            project_name, int(bug_id) if dataset == "defects4j" else str(bug_id))
        bug_info["bug_category"] = _CATEGORY_MAP.get(cat, cat or "unknown")  # grouping only

        # Step 5: blame on SBFL lines (history configs only)
        blame_info = None
        if history_category != HistoryCategory.baseline:
            blameable = []
            for s in suspicious:
                content = read_line_from_container(docker_env, work_dir, s.file, s.line)
                blameable.append(PatchLine(
                    file_path=s.file, line_number=s.line, content=content,
                    change_type="+", patch_format=patch_format,
                ))
            selector = get_selector(selector_type, config.get("model", {}))
            selected = selector.select(blameable, n_lines)
            if selected:
                sel = selected[0]
                log.info(f"Judge selected blame line: {sel.file_path}:{sel.line_number}")
                blame_result = run_git_blame(docker_env, work_dir, sel.file_path, sel.line_number)
                if blame_result and "commit_hash" in blame_result:
                    source = docker_env.execute(f"cd {work_dir} && cat {sel.file_path}")
                    func_name = ""
                    if source.get("returncode") == 0 and source.get("output"):
                        func_name = find_function_containing_line(
                            source["output"], sel.line_number, Path(sel.file_path).suffix) or ""
                    history_context = extract_all_history_context(
                        docker_env, work_dir, blame_result["commit_hash"],
                        [fl["file"] for fl in bug_info["fault_locations"]],
                        blamed_file=sel.file_path, blamed_line=sel.line_number,
                        function_name=func_name,
                    )
                    if history_context:
                        blame_info = {"blame_commit": history_context}
                        log.info(f"Blame commit: {blame_result['commit_hash'][:8]}")
                    else:
                        log.warning("History extraction returned empty")
                else:
                    log.warning("git blame failed on selected line")
            else:
                log.warning("Selector returned no lines")

        # Step 6: build prompt
        repo_path = bug_info.get("repo_path", work_dir)
        docker_env.config.cwd = repo_path
        prompt_data = build_hafix_prompt(
            bug_info=bug_info, history_category=history_category,
            blame_info=blame_info, config=config,
        )

        if dry_run:
            log.info("DRY RUN — skipping agent.run()")
            console.print(f"[cyan]== {bug_label} / {history_category.name} (dry-run) ==[/cyan]")
            console.print(f"src_root={src_root}  fault_locations={len(bug_info['fault_locations'])}  blame={'yes' if blame_info else 'no'}")
            console.print("[dim]--- instance prompt (rendered vars) ---[/dim]")
            for fl in bug_info["fault_locations"][:top_n]:
                console.print(f"  #{fl['rank']} {fl['file']}:{fl['start_line']} susp={fl['suspiciousness']:.3f}")
            if blame_info:
                bc = blame_info["blame_commit"].get("commit", {})
                console.print(f"  blame commit msg: {str(bc.get('commit_message',''))[:80]}")
            result_file = bug_output_dir / f"{bug_label}_{history_category.value}_dryrun.json"
            result_file.write_text(json.dumps({
                "bug_id": bug_label, "history_category": history_category.name,
                "src_root": src_root, "fault_locations": bug_info["fault_locations"],
                "has_blame": blame_info is not None,
            }, indent=2, default=str))
            return

        # Step 7: run agent
        if progress_manager:
            progress_manager.update_bug_status(bug_label, history_category.name, "running_repair")
        start_time = time.time()
        model = get_model(None, config.get("model", {}))
        try:
            add_token_tracking_to_model(model)
        except Exception:
            pass

        agent_config = config.get("agent", {})
        agent = HAFixAgent(
            model=model, env=docker_env, logger=log,
            system_template=prompt_data["system_template"],
            instance_template=prompt_data["instance_template"],
            step_limit=agent_config.get("step_limit", 50),
            cost_limit=agent_config.get("cost_limit", 1.0),
            dataset=agent_config.get("dataset", "defects4j"),
            action_observation_template=agent_config.get("action_observation_template"),
            format_error_template=agent_config.get("format_error_template"),
            timeout_template=agent_config.get("timeout_template"),
        )
        agent.model_config = config.get("model", {})
        agent.bug_info = bug_info
        agent.set_bug_context(bug_info, blame_info, history_category)
        agent.extra_template_vars.update(prompt_data["template_vars"])

        exit_status, submission = agent.run()
        metrics = extract_execution_metrics(start_time, model=model, agent=agent, env=docker_env)
        save_trajectory_safe(agent, output_dir, bug_info.get("bug_category", "unknown"),
                             project_name, bug_id, history_category, exit_status, submission)

        result_data = {
            "project_name": project_name, "bug_id": bug_label,
            "history_category": history_category.name,
            "fl_source": "sbfl_gzoltar" if dataset == "defects4j" else "sbfl_fauxpy",
            "dataset": dataset, "top_n": top_n,
            "repair_result": {
                "success": exit_status == "Submitted", "exit_status": exit_status,
                "message": str(submission)[:1000] if submission else None, **metrics,
            },
            "timestamp": _get_timestamp(timezone=DEFAULT_TIMEZONE),
            "selector_type": selector_type, "n_lines": n_lines,
            "bug_info": {k: v for k, v in bug_info.items() if k != "golden_patch"},
            "blame_info": {"has_blame": blame_info is not None} if blame_info else None,
            "sbfl_fault_locations": bug_info["fault_locations"],
        }
        result_file.write_text(json.dumps(result_data, indent=2, default=str))
        log.info(f"=== Complete: {'success' if exit_status == 'Submitted' else 'failed'} ===")
        if progress_manager:
            progress_manager.record_repair_result(bug_label, result_data["repair_result"])
            progress_manager.update_bug_status(
                bug_label, history_category.name, "completed",
                repair_success=(exit_status == "Submitted"), exit_status=exit_status)

    except Exception as e:
        log.error(f"Failed: {e}")
        log.error(tb.format_exc())
        result_file.write_text(json.dumps({
            "project_name": project_name, "bug_id": bug_label,
            "history_category": history_category.name,
            "repair_result": {"success": False, "error": str(e)},
            "timestamp": _get_timestamp(timezone=DEFAULT_TIMEZONE),
        }, indent=2, default=str))
        if progress_manager:
            progress_manager.update_bug_status(
                bug_label, history_category.name, "failed",
                repair_success=False, exit_status="ExecutionError")
    finally:
        if docker_env:
            try:
                docker_env.cleanup()
            except Exception:
                pass


def discover_cached_bugs(ranking_dir: Path, dataset: str) -> List[str]:
    """Bug labels (Project_Bug) that have a cached ranking (dataset-aware filename)."""
    ranking_file = _RANKING_FILE[dataset]
    return sorted(
        d.name for d in ranking_dir.iterdir()
        if d.is_dir() and (d / ranking_file).exists()
    )


def main() -> int:
    ap = argparse.ArgumentParser(description="SBFL repair stage (RQ3)")
    ap.add_argument("--dataset", choices=["defects4j", "bugsinpy"], default="defects4j")
    ap.add_argument("--history", required=True, help="baseline, fn_all, fn_pair, fl_diff")
    ap.add_argument("--all", action="store_true", help="all bugs with a cached ranking")
    ap.add_argument("--bugs", nargs="+", help="explicit bug labels, e.g. Cli_1 Lang_6")
    ap.add_argument("--sample", default=None,
                    help="JSON sample file (e.g. evaluation/sbfl/rq3_repair_sample.json); "
                         "runs the bugs it lists for --dataset")
    ap.add_argument("--top-n", type=int, default=10)
    ap.add_argument("--config", default=None,
                    help="default: config/<dataset>_sbfl.yaml")
    ap.add_argument("--model-config", default=None,
                    help="model config YAML overriding the model section and tagging the output "
                         "dir per LLM (e.g. config/models/qwen3_coder_next.yaml). "
                         "Default: the model defined in --config (DeepSeek).")
    ap.add_argument("--ranking-dir", default=None,
                    help="default: results/sbfl/<dataset>/fl_cache")
    ap.add_argument("--output", default=None,
                    help="default: results/sbfl/<dataset>/llm_judge_1line")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="build FL+blame, skip agent (no API cost)")
    args = ap.parse_args()

    # Resolve dataset-dependent path defaults. Output mirrors the RQ1 layout
    # (results/<dataset>/llm_judge_1line/<category>/<bug>/...) under results/sbfl/ so RQ1 and
    # RQ3 are directly comparable with the same analysis code.
    cfg_path = args.config or str(PROJECT_ROOT / "config" / f"{args.dataset}_sbfl.yaml")
    ranking_dir = Path(args.ranking_dir or PROJECT_ROOT / "results" / "sbfl" / args.dataset / "fl_cache")

    history_cat = history_name_to_category.get(args.history) or HistoryCategory[args.history]
    config = yaml.safe_load(Path(cfg_path).read_text())

    # Optional per-LLM model override, mirroring the RQ1 runners' --model-config: swap the
    # model section and tag the output dir so each LLM's SBFL results are isolated under
    # results/sbfl/<dataset>/llm_judge_1line[_<tag>]/...  DeepSeek (no tag) is never touched.
    if args.model_config:
        mc = yaml.safe_load(Path(args.model_config).read_text())
        config["model"] = mc["model"]
        if mc.get("model_tag"):
            config["model_tag"] = mc["model_tag"]

    # Register local OpenAI-compatible models (vLLM) with litellm so the agent's cost/context
    # lookups don't raise "model isn't mapped yet" (mirrors the RQ1 runners; no-op for DeepSeek).
    register_local_model_with_litellm(config)

    if args.output:
        out_base = Path(args.output)
    else:
        sub = "llm_judge_1line" + (f"_{resolve_model_tag(config)}" if args.model_config else "")
        out_base = PROJECT_ROOT / "results" / "sbfl" / args.dataset / sub

    if args.bugs:
        labels = args.bugs
    elif args.sample:
        ds_samp = json.loads(Path(args.sample).read_text()).get(args.dataset, {})
        labels = sorted(b for cat in ds_samp.values() if isinstance(cat, list) for b in cat)
        if not labels:
            ap.error(f"no bugs for dataset '{args.dataset}' in {args.sample}")
    elif args.all:
        labels = discover_cached_bugs(ranking_dir, args.dataset)
    else:
        ap.error("specify --sample, --all, or --bugs")

    # Category per bug (container-free CSV lookup, done single-threaded) -> RQ1-style path.
    catmap = {l: _category_full(args.dataset, *l.rsplit("_", 1)) for l in labels}

    def result_path(label):
        return out_base / catmap[label] / label / f"{label}_{history_cat.value}_result.json"

    if args.resume:
        before = len(labels)
        labels = [l for l in labels if not result_path(l).exists()]
        console.print(f"Resume: {before - len(labels)} done, {len(labels)} remaining")

    out_base.mkdir(parents=True, exist_ok=True)
    console.print(f"SBFL repair [{args.dataset}]: {len(labels)} bugs, {args.history}, "
                  f"top-{args.top_n}, {'DRY-RUN' if args.dry_run else f'{args.workers} workers'}")

    progress_manager = None
    progress_file = out_base / f"progress_{history_cat.value}_sbfl.json"
    if not args.dry_run:
        progress_manager = EvaluationProgressManager(
            len(labels), str(progress_file), timezone=DEFAULT_TIMEZONE)

    def submit(label):
        project, bug = label.rsplit("_", 1)
        process_sbfl_bug(project, bug, history_cat, out_base / catmap[label], config,
                         ranking_dir, args.top_n, args.dataset, "llm_judge", 1, args.dry_run,
                         progress_manager)

    if args.workers <= 1 or args.dry_run:
        for label in labels:
            submit(label)
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(submit, l): l for l in labels}
            for fut in concurrent.futures.as_completed(futs):
                label = futs[fut]
                try:
                    fut.result()
                except Exception as e:
                    console.print(f"[red]{label} failed: {e}[/red]")

    if progress_manager:
        rr = progress_manager.repair_results
        console.print(f"Completed: {progress_manager.completed}  Failed: {progress_manager.failed}")
        console.print(f"Total model cost: ${rr['total_model_cost']:.4f}  "
                      f"tokens: {rr['total_token_usage'].get('total_tokens', 0)}")
        console.print(f"Exit-status breakdown: {rr['exit_status_breakdown']}")
        console.print(f"Progress data: {progress_file}")
    console.print(f"Done. Results in {out_base}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
