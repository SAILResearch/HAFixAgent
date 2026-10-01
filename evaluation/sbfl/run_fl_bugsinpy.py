"""FL stage (BugsInPy) — FauxPy SBFL engine over a decoupled coverage spectrum.

Two-step per bug:
  1. in the prewarmed container: collect_spectrum.sh -> coverage.json + outcomes.tsv + meta
  2. on the host (clean env): compute_ochiai.py -> FauxPy ranking -> Scores_Ochiai.csv + spectrum.json

Why this exists: FauxPy must be installed into the bug's fragile venv (pytest-version
conflicts) where it silently emits empty rankings for some bugs that have a perfectly valid
spectrum — the failure is in collection, not in FauxPy's ranking code. So we collect the
spectrum with `coverage` (only dep added to the bug env, universally compatible) and feed it
into FauxPy's own SbflDbManager/RankingMetricManager on the host (see compute_ochiai.py),
producing `spectrum.json`, the durable replication artifact (re-rank with no Docker).

Output: results/sbfl/bugsinpy/fl_cache/<Project>_<Bug>/{Scores_Ochiai.csv, spectrum.json,
meta.json, coverage.json, outcomes.tsv, pytest.log}. The Scores_Ochiai.csv uses the
standard Entity,Score Ochiai format, so parse_ranking / analyze / the repair stage are
unchanged.

Usage:
    python evaluation/sbfl/run_fl_bugsinpy.py --bugs scrapy_10 keras_3
    python evaluation/sbfl/run_fl_bugsinpy.py --all --workers 8
"""

import argparse
import concurrent.futures
import csv
import subprocess
import sys
from pathlib import Path
from typing import List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PYTHON_DIR = PROJECT_ROOT / "evaluation" / "sbfl" / "python"
FEASIBILITY_CSV = PROJECT_ROOT / "dataset" / "bugsinpy" / "bugsinpy_blame_feasibility.csv"

sys.path.insert(0, str(PROJECT_ROOT / "evaluation" / "sbfl"))
from compute_ochiai import compute  # noqa: E402


def load_all_bugs() -> List[Tuple[str, str]]:
    bugs = []
    with open(FEASIBILITY_CSV, newline="") as f:
        for row in csv.DictReader(f):
            bugs.append((row["Project"], str(row["Bug_Number"])))
    return bugs


def _has_real_ranking(out_dir: Path) -> bool:
    scores = out_dir / "Scores_Ochiai.csv"
    if not (out_dir / "meta.json").exists() or not scores.exists():
        return False
    with open(scores) as f:
        return sum(1 for _ in f) > 1


def run_one(project: str, bug_id: str, output_root: Path, timeout: int) -> Tuple[str, str, str]:
    out_dir = output_root / f"{project}_{bug_id}"
    if _has_real_ranking(out_dir):
        return (project, bug_id, "cached")

    image = f"bugsinpy:{project.lower()}_{bug_id}"
    if subprocess.run(["docker", "image", "inspect", image], capture_output=True).returncode != 0:
        return (project, bug_id, "FAIL no prewarmed image")

    out_dir.mkdir(parents=True, exist_ok=True)
    cname = f"hafixagent-bip-cov-{project}_{bug_id}"
    subprocess.run(["docker", "rm", "-f", cname], capture_output=True)
    cmd = [
        "docker", "run", "--rm", "--name", cname,
        "-v", f"{PYTHON_DIR}:/sbfl_py:ro",
        "-v", f"{out_dir}:/out",
        image, "bash", "/sbfl_py/collect_spectrum.sh", project, str(bug_id),
    ]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return (project, bug_id, "timeout")
    finally:
        subprocess.run(["docker", "rm", "-f", cname], capture_output=True)

    cov_json = out_dir / "coverage.json"
    outcomes = out_dir / "outcomes.tsv"
    if not (cov_json.exists() and outcomes.exists()):
        tail = (r.stderr or r.stdout or "").strip().splitlines()[-1:] or [""]
        return (project, bug_id, f"FAIL(rc={r.returncode}): {tail[0][:120]}")

    # Host-side ranking (clean env, no bug deps).
    try:
        n = compute(cov_json, outcomes, out_dir / "Scores_Ochiai.csv", out_dir / "spectrum.json")
    except Exception as e:  # noqa: BLE001
        return (project, bug_id, f"FAIL compute: {str(e)[:100]}")
    if n <= 0:
        return (project, bug_id, "FAIL no suspicious lines (no failing test covered src)")
    return (project, bug_id, "ok")


def main() -> int:
    ap = argparse.ArgumentParser(description="RQ3 FL stage — FauxPy Ochiai over coverage spectra on BugsInPy")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--project")
    ap.add_argument("--bugs", nargs="+", help="explicit bug ids, e.g. scrapy_10 keras_3")
    ap.add_argument("--output", default=str(PROJECT_ROOT / "results" / "sbfl" / "bugsinpy" / "fl_cache"))
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

    output_root = Path(args.output).resolve()  # docker -v needs an absolute path
    print(f"FL stage (BugsInPy/coverage): {len(bugs)} bugs, workers={args.workers}")
    counts = {"ok": 0, "cached": 0, "fail": 0}
    total, done = len(bugs), 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(run_one, p, b, output_root, args.timeout): f"{p}_{b}" for p, b in bugs}
        for fut in concurrent.futures.as_completed(futs):
            project, bug_id, status = fut.result()
            done += 1
            counts[status if status in ("ok", "cached") else "fail"] += 1
            tag = status if status in ("ok", "cached") else f"FAIL {status}"
            print(f"[{done}/{total}] {project}_{bug_id} {tag}  "
                  f"(ok={counts['ok']} cached={counts['cached']} fail={counts['fail']})", flush=True)

    print(f"\nDone. ok={counts['ok']} cached={counts['cached']} fail={counts['fail']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
