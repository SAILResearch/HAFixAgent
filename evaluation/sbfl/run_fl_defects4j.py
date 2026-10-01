"""FL stage (Defects4J): run GZoltar SBFL per bug in throwaway containers, cache rankings.

Stage 1 of RQ3. Decoupled from the repair stage so expensive FL runs once per bug and
the agent can be re-run cheaply against the cached ranking.

For each bug it launches a disposable `defects4j:latest` container that checks out +
compiles the bug and runs `java/fl_one_bug.sh`, writing to
`results/sbfl/defects4j/fl_cache/<Project>_<Bug>/{ochiai.ranking.csv, meta.json, gzoltar.log}`.

Usage:
    python evaluation/sbfl/run_fl_defects4j.py --all --workers 8
    python evaluation/sbfl/run_fl_defects4j.py --project Cli --workers 4
    python evaluation/sbfl/run_fl_defects4j.py --bugs Cli_1 Lang_6
"""

import argparse
import concurrent.futures
import csv
import subprocess
from pathlib import Path
from typing import List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[2]
JAVA_DIR = PROJECT_ROOT / "evaluation" / "sbfl" / "java"
FEASIBILITY_CSV = PROJECT_ROOT / "dataset" / "defects4j" / "defects4j_blame_feasibility.csv"


def load_all_bugs() -> List[Tuple[str, str]]:
    """Return [(project, bug_number)] for every bug in the feasibility CSV."""
    bugs = []
    with open(FEASIBILITY_CSV, newline="") as f:
        for row in csv.DictReader(f):
            bugs.append((row["Project"], str(row["Bug_Number"])))
    return bugs


def run_one(project: str, bug_id: str, output_root: Path, scope: str,
            image: str, timeout: int) -> Tuple[str, str, str]:
    out_dir = output_root / f"{project}_{bug_id}"
    ranking = out_dir / "ochiai.ranking.csv"
    if ranking.exists() and ranking.stat().st_size > 0:
        return (project, bug_id, "cached")

    out_dir.mkdir(parents=True, exist_ok=True)
    cname = f"sbfl_{project}_{bug_id}"
    # Clear any stale same-named container from a previous interrupted run.
    subprocess.run(["docker", "rm", "-f", cname], capture_output=True)
    cmd = [
        "docker", "run", "--rm", "--name", cname,
        "-v", f"{JAVA_DIR}:/sbfl_java:ro",
        "-v", f"{out_dir}:/out",
        image, "bash", "/sbfl_java/fl_one_bug.sh", project, str(bug_id), scope,
    ]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return (project, bug_id, "timeout")
    finally:
        # On timeout/error the `docker run` client may die before --rm fires,
        # leaving an orphaned container burning CPU. Force-remove by name to be
        # sure (no-op if --rm already cleaned it up).
        subprocess.run(["docker", "rm", "-f", cname], capture_output=True)

    if ranking.exists() and ranking.stat().st_size > 0:
        return (project, bug_id, "ok")
    tail = (r.stderr or r.stdout or "").strip().splitlines()[-1:] or [""]
    return (project, bug_id, f"FAIL(rc={r.returncode}): {tail[0][:120]}")


def main() -> int:
    ap = argparse.ArgumentParser(description="RQ3 FL stage — GZoltar SBFL on Defects4J")
    ap.add_argument("--all", action="store_true", help="run all bugs in the feasibility CSV")
    ap.add_argument("--project", help="restrict to one project")
    ap.add_argument("--bugs", nargs="+", help="explicit bug ids, e.g. Cli_1 Lang_6")
    ap.add_argument("--exclude-chart", action="store_true",
                    help="skip Chart (no real git history)")
    ap.add_argument("--scope", default="relevant", choices=["relevant", "modified"])
    ap.add_argument("--output", default=str(PROJECT_ROOT / "results" / "sbfl" / "defects4j" / "fl_cache"))
    ap.add_argument("--image", default="defects4j:latest")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--timeout", type=int, default=1800, help="per-bug timeout (s)")
    args = ap.parse_args()

    if args.bugs:
        bugs = [(b.rsplit("_", 1)[0], b.rsplit("_", 1)[1]) for b in args.bugs]
    elif args.all or args.project:
        bugs = load_all_bugs()
        if args.project:
            bugs = [(p, b) for p, b in bugs if p == args.project]
    else:
        ap.error("specify --all, --project, or --bugs")

    if args.exclude_chart:
        bugs = [(p, b) for p, b in bugs if p != "Chart"]

    output_root = Path(args.output)
    print(f"FL stage: {len(bugs)} bugs, scope={args.scope}, workers={args.workers}")

    counts = {"ok": 0, "cached": 0, "fail": 0}
    total = len(bugs)
    done = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {
            ex.submit(run_one, p, b, output_root, args.scope, args.image, args.timeout):
            f"{p}_{b}" for p, b in bugs
        }
        for fut in concurrent.futures.as_completed(futs):
            project, bug_id, status = fut.result()
            done += 1
            counts[status if status in ("ok", "cached") else "fail"] += 1
            tag = status if status in ("ok", "cached") else f"FAIL {status}"
            print(f"[{done}/{total}] {project}_{bug_id} {tag}  "
                  f"(ok={counts['ok']} cached={counts['cached']} fail={counts['fail']})",
                  flush=True)

    print(f"\nDone. ok={counts['ok']} cached={counts['cached']} fail={counts['fail']}", flush=True)
    return 0 if counts["fail"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
