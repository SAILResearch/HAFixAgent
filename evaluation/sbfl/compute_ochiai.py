"""Rank suspicious lines with FauxPy's SBFL engine from a collected coverage spectrum.

This is the host-side *ranking* half of the decoupled FL design. FauxPy is the SBFL
tool of record for BugsInPy (Rezaalipour & Furia, EMSE; arXiv 2305.19834), but its
pytest-plugin coverage collection fails to reproduce on many BugsInPy environments
(yanked pre-release pins, unittest-style suites, import-timing of `pytest-cov`). Our
integration layer replaces only that fragile *collection* stage with a robust
`coverage.py`-based collector (see python/collect_spectrum.sh) and then feeds the result
straight into FauxPy's own ranking pipeline:

    our spectrum  ->  FauxPy SbflDbManager  ->  FauxPy RankingMetricManager
                      (counts ef/ep/nf/np)      (MetricOchiai, epsilon=0.1)

So the suspiciousness scores and the Ochiai ranking are computed by FauxPy's actual code
(`fauxpy.fault_localization.sbfl`), not a re-implementation. We supply only the spectrum
and read back FauxPy's `Ochiai` column. Requires `pip install fauxpy`.

Inputs (both portable, emitted by collect_spectrum.sh inside the bug container):
  * coverage.json : `coverage json --show-contexts` output. Per file, `contexts` maps
                    "<line>" -> [test-context, ...]; a context is a pytest nodeid,
                    optionally suffixed "|run"/"|setup"/"|teardown".
  * outcomes.tsv  : one row per test phase, "<nodeid>\t<outcome>" (passed/failed/error),
                    written by _sbfl_outcome_plugin.py. A nodeid is failing if any phase
                    failed or errored.

Output:
  * Scores_Ochiai.csv : "Entity,Score" rows "<relpath>::<line>,<score>" sorted desc — the
                        SAME format FauxPy's CLI emits, so parse_ranking.parse_fauxpy_ranking
                        and everything downstream are unchanged.
  * spectrum.json (optional) : FauxPy's per-line ef/ep/nf/np + score, the durable
                        replication artifact (immune to PyPI/image rot — re-rank with no env).
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from fauxpy.fault_localization.sbfl.db_manager import SbflDbManager
from fauxpy.fault_localization.sbfl.ranking_metric_manager import RankingMetricManager

# coverage contexts may carry a phase suffix; the nodeid is everything before the last '|'.
_PHASE = re.compile(r"\|(run|setup|teardown)$")


def _nodeid(context: str) -> str:
    return _PHASE.sub("", context).strip()


def load_failing(outcomes_path: Path) -> Tuple[Set[str], Set[str]]:
    """Return (failing_nodeids, all_nodeids) from outcomes.tsv.

    A test is failing if ANY of its phases failed or errored (setup error counts).
    """
    failing: Set[str] = set()
    allids: Set[str] = set()
    for line in outcomes_path.read_text(errors="ignore").splitlines():
        if "\t" not in line:
            continue
        nodeid, outcome = line.split("\t", 1)
        nodeid = nodeid.strip()
        allids.add(nodeid)
        if outcome.strip() in ("failed", "error"):
            failing.add(nodeid)
    return failing, allids


def build_per_test_coverage(
    coverage_json: Dict, allids: Set[str], failing: Set[str]
) -> Dict[str, Set[str]]:
    """nodeid -> {"<file>::<line>", ...}, restricted to entities a failing test covers.

    Entities never covered by a failing test get Ochiai 0 (FauxPy returns 0 for ef=0),
    so dropping them up front is equivalent and keeps FauxPy's DB small. We still keep
    every *passing* test that covers a kept entity so FauxPy's ep count stays exact.
    """
    # Pass 1: entities covered by >=1 failing test.
    failing_entities: Set[str] = set()
    for fpath, fdata in coverage_json.get("files", {}).items():
        for line_s, ctxs in fdata.get("contexts", {}).items():
            if not line_s.isdigit():
                continue
            entity = f"{fpath}::{line_s}"
            for c in ctxs:
                if _nodeid(c) in failing:
                    failing_entities.add(entity)
                    break

    # Pass 2: per-test traces over only those entities.
    per_test: Dict[str, Set[str]] = {}
    for fpath, fdata in coverage_json.get("files", {}).items():
        for line_s, ctxs in fdata.get("contexts", {}).items():
            entity = f"{fpath}::{line_s}"
            if entity not in failing_entities:
                continue
            for c in ctxs:
                nid = _nodeid(c)
                if nid and nid in allids:
                    per_test.setdefault(nid, set()).add(entity)
    return per_test


def rank_with_fauxpy(
    failing: Set[str], allids: Set[str], per_test: Dict[str, Set[str]]
) -> List[Tuple[str, int, int, int, int, float]]:
    """Drive FauxPy's DB + ranking. Return [(entity, ef, ep, nf, np, ochiai), ...].

    FauxPy does all the work: SbflDbManager counts ef/ep/nf/np via SQL, and
    RankingMetricManager scores every entity with its MetricOchiai (epsilon=0.1).
    We only insert the spectrum and read FauxPy's Score table back.
    """
    with tempfile.TemporaryDirectory() as tmp:
        db = SbflDbManager(Path(tmp))
        # Perf only: FauxPy commits (fsyncs) after every row insert. This temp DB is
        # discarded, so we disable durability fsync/journal — commits become cheap and
        # the ranking FauxPy computes is byte-for-byte identical.
        db._connection.execute("PRAGMA synchronous=OFF")
        db._connection.execute("PRAGMA journal_mode=MEMORY")
        try:
            for nid in sorted(allids):
                if nid in failing:
                    db.insert_test_case(nid, "failed", True)
                else:
                    db.insert_test_case(nid, "passed", False)
            for nid, entities in per_test.items():
                if entities:
                    db.insert_execution_trace(nid, sorted(entities))

            # FauxPy computes and persists ef/ep/nf/np + Ochiai for every entity.
            RankingMetricManager(db).compute_sorted_scores(-1)

            cur = db._connection.cursor()
            cur.execute("SELECT Entity, Ef, Ep, Nf, Np, Ochiai FROM Score")
            rows = [
                (e, int(ef), int(ep), int(nf), int(np), float(o))
                for (e, ef, ep, nf, np, o) in cur.fetchall()
            ]
        finally:
            db.end()
    return rows


def compute(
    coverage_json_path: Path,
    outcomes_path: Path,
    out_csv: Path,
    spectrum_out: Optional[Path] = None,
) -> int:
    """Write Scores_Ochiai.csv via FauxPy's engine; return the number of ranked lines."""
    failing, allids = load_failing(outcomes_path)
    cov = json.loads(coverage_json_path.read_text())
    per_test = build_per_test_coverage(cov, allids, failing)
    scored = rank_with_fauxpy(failing, allids, per_test)

    # Keep only suspicious lines (ef>0 => Ochiai>0). Deterministic order: score desc,
    # then entity asc for stable ties (FauxPy's SQL ORDER BY leaves ties arbitrary).
    ranked = [r for r in scored if r[5] > 0]
    ranked.sort(key=lambda r: (-r[5], r[0]))

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Entity", "Score"])
        for entity, _ef, _ep, _nf, _np, score in ranked:
            w.writerow([entity, f"{score:.6f}"])

    if spectrum_out is not None:
        spectrum_out.write_text(json.dumps({
            "engine": "fauxpy",
            "metric": "Ochiai",
            "epsilon": RankingMetricManager.EPSILON,
            "total_failing": len(failing),
            "total_tests": len(allids),
            "failing_tests": sorted(failing),
            "lines": [
                {"file": entity.rsplit("::", 1)[0], "line": int(entity.rsplit("::", 1)[1]),
                 "e_f": ef, "e_p": ep, "n_f": nf, "n_p": np, "ochiai": score}
                for entity, ef, ep, nf, np, score in sorted(ranked, key=lambda r: r[0])
            ],
        }, indent=2))

    return len(ranked)


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Rank Ochiai SBFL from a coverage spectrum using FauxPy's engine")
    ap.add_argument("--coverage-json", required=True)
    ap.add_argument("--outcomes", required=True)
    ap.add_argument("--out", required=True, help="Scores_Ochiai.csv path")
    ap.add_argument("--spectrum-out", default=None, help="optional spectrum.json (replication artifact)")
    args = ap.parse_args()
    n = compute(Path(args.coverage_json), Path(args.outcomes), Path(args.out),
                Path(args.spectrum_out) if args.spectrum_out else None)
    print(f"ranked {n} lines -> {args.out}")
    return 0 if n > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
