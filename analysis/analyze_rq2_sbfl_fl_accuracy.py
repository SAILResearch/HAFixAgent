"""RQ3 FL accuracy — how well SBFL localizes the developer-modified lines.

Offline analysis (no container, no API) over the cached rankings produced by the FL
stage (`run_fl_defects4j.py` → GZoltar, or `run_fl_bugsinpy.py` → coverage engine). For each bug
it compares the SBFL ranking against the developer-modified ("buggy") lines parsed from
the developer patch, reporting whether a true line falls in the SBFL top-1/5/10 and the
rank of the first hit. This contextualizes the RQ3 repair numbers (how often the agent
is even shown the right line).

The developer patch is used ONLY here for measurement — never fed to the agent.

Defects4J ground truth: the saved `dev.src.patch` (REVERSE; '+' = buggy line).
BugsInPy ground truth: the dataset's `bug_patch.txt` (FORWARD; '-' = buggy line).

Omission faults (fix only ADDS code) have no buggy line of their own. Following the
standard FL-evaluation convention (Pearson et al. ICSE'17 [49]; Zou et al. TSE'21 [78];
the FauxPy EMSE study, arXiv 2305.19834 §4.2), we map each inserted line to the
buggy-version line immediately preceding AND following the insertion point — the same
insertion-point heuristic HARPER's own pre-hunk fallback uses for repair. They are thus
included in the hit-rate (not excluded), so our Top-N is comparable to FauxPy's.

Usage:
    python analysis/analyze_rq2_sbfl_fl_accuracy.py --dataset defects4j
    python analysis/analyze_rq2_sbfl_fl_accuracy.py --dataset bugsinpy
"""

import argparse
import csv
import re
import sys
from collections import defaultdict
from math import comb
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

_CODE_EXT = {".py", ".java"}
_HUNK_RE = re.compile(r"@@\s*-(\d+)(?:,\d+)?\s*\+(\d+)(?:,\d+)?\s*@@")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "evaluation" / "sbfl"))

from hafix_agent.blame.patch_parser import PatchParser, PatchFormat  # noqa: E402
from parse_ranking import parse_gzoltar_ranking, parse_fauxpy_ranking  # noqa: E402
from dataset.bugsinpy.util import get_patch_file_path  # noqa: E402

FEASIBILITY_CSV = {
    "defects4j": PROJECT_ROOT / "dataset" / "defects4j" / "defects4j_blame_feasibility.csv",
    "bugsinpy": PROJECT_ROOT / "dataset" / "bugsinpy" / "bugsinpy_blame_feasibility.csv",
}
# Per-dataset cached-ranking filename + parser + ground-truth patch direction.
RANKING_FILE = {"defects4j": "ochiai.ranking.csv", "bugsinpy": "Scores_Ochiai.csv"}
PATCH_FORMAT = {"defects4j": PatchFormat.REVERSE, "bugsinpy": PatchFormat.FORWARD}
CATEGORY_ORDER = ["SL", "SH", "SFMH", "MFMH"]
CATEGORY_NORM = {
    "single_line": "SL", "single_hunk": "SH",
    "single_file_multi_hunk": "SFMH", "multi_file_multi_hunk": "MFMH",
    "SL": "SL", "SH": "SH", "SFMH": "SFMH", "MFMH": "MFMH",
}


def _norm_path(p: str) -> str:
    """Strip a leading a/ or b/ (diff prefix) so patch and ranking paths compare equal."""
    if p.startswith(("a/", "b/")):
        return p[2:]
    return p


def load_categories(csv_path: Path) -> Dict[str, str]:
    cats = {}
    if not csv_path.exists():
        return cats
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            label = f"{row['Project']}_{row['Bug_Number']}"
            cats[label] = CATEGORY_NORM.get(row.get("Category", ""), row.get("Category", "?"))
    return cats


def _direct_buggy_lines(patch_text: str, patch_format: PatchFormat) -> Set[Tuple[str, int]]:
    """(file, line) the developer modified/deleted — lines that exist in the buggy version.

    REVERSE (Defects4J dev.src.patch): '+' lines are the buggy-version lines.
    FORWARD (BugsInPy bug_patch.txt):  '-' lines are the buggy-version lines.
    """
    parser = PatchParser(patch_format=patch_format)
    lines = parser.extract_code_file_lines(patch_text)
    return {(_norm_path(pl.file_path), pl.line_number) for pl in lines if pl.can_blame()}


def _omission_adjacent_lines(patch_text: str, patch_format: PatchFormat) -> Set[Tuple[str, int]]:
    """Insertion-point ground truth for omission faults (Pearson/Zou/FauxPy convention).

    For each line the fix ADDS (which does not exist in the buggy version), record the
    buggy-version lines immediately PRECEDING and FOLLOWING the insertion point. The buggy
    side is the NEW side of a REVERSE patch (fixed->buggy) and the OLD side of a FORWARD
    patch (buggy->fixed); an inserted line is a '-' in a reverse patch and a '+' in a
    forward patch. We don't skip blank/comment neighbours — those simply never appear in a
    coverage/Ochiai ranking, so they harmlessly fail to match (a conservative undercount).
    """
    reverse = patch_format.value == "reverse"
    out: Set[Tuple[str, int]] = set()
    current_file: Optional[str] = None
    bug_ln = 0  # running buggy-version line number

    def is_code(fp: Optional[str]) -> bool:
        return bool(fp) and Path(fp).suffix.lower() in _CODE_EXT

    for line in patch_text.split("\n"):
        if line.startswith("--- a/"):
            current_file = line[6:]
            continue
        if line.startswith("--- /dev/null"):
            current_file = None
            continue
        if line.startswith("+++ "):
            continue
        if line.startswith("@@"):
            m = _HUNK_RE.match(line)
            if m and is_code(current_file):
                bug_ln = int(m.group(2)) if reverse else int(m.group(1))
            continue
        if not is_code(current_file):
            continue

        if line.startswith(" "):
            bug_ln += 1
        elif (line.startswith("-") and reverse) or (line.startswith("+") and not reverse):
            # inserted-by-fix line (omission): insertion sits between bug_ln-1 and bug_ln
            fp = _norm_path(current_file)
            if bug_ln - 1 >= 1:
                out.add((fp, bug_ln - 1))
            out.add((fp, bug_ln))
        else:  # a buggy-version line ('+' reverse / '-' forward) advances the buggy counter
            bug_ln += 1
    return out


def ground_truth_lines(patch_text: str, patch_format: PatchFormat):
    """Return (direct_modify/delete_lines, omission_adjacent_lines)."""
    return (_direct_buggy_lines(patch_text, patch_format),
            _omission_adjacent_lines(patch_text, patch_format))


def einspect(start: int, ties: int, faulty: int) -> float:
    """Zou et al. / FauxPy E_inspect rank (Eq. 6) for a faulty line in a tie group.

        I = start + sum_{k=1}^{ties-faulty} k * C(ties-k-1, faulty-1) / C(ties, faulty)

    start  = 1-based position of the first entity tying this score
    ties   = # entities sharing this score (incl. the faulty one)
    faulty = # of those tied entities that are faulty
    Coincides with the average rank start+(ties-1)/2 when faulty==1; gives the expected
    first-hit position when several faulty lines share one tie group.
    """
    denom = comb(ties, faulty)
    extra = sum(k * comb(ties - k - 1, faulty - 1) / denom
                for k in range(1, ties - faulty + 1))
    return start + extra


def hit_ranks(ranking, dev_lines: Set[Tuple[str, int]]):
    """Return (raw_rank, einspect_rank, total) for the best-localized developer line.

    raw_rank      : GZoltar's deterministic 1-based position (what the agent's top-N sees).
    einspect_rank : Zou et al. E_inspect rank (FauxPy Eq. 6) — the tie-aware "effort"
                    used for Top-N. I_b = min over faulty lines, which is attained at the
                    highest faulty suspiciousness score.
    (None, None, total) if no developer line is ranked.
    """
    total = len(ranking)
    susp_of = {(_norm_path(s.file), s.line): s.suspiciousness for s in ranking}
    raw_of = {(_norm_path(s.file), s.line): s.rank for s in ranking}
    susps = [s.suspiciousness for s in ranking]

    matched = [(k, susp_of[k]) for k in dev_lines if k in susp_of]
    if not matched:
        return None, None, total

    best_score = max(s for _, s in matched)          # min E_inspect is at highest faulty score
    ties = sum(1 for v in susps if v == best_score)
    faulty = sum(1 for _, s in matched if s == best_score)
    start = sum(1 for v in susps if v > best_score) + 1
    raw = min(raw_of[k] for k, _ in matched)
    return raw, einspect(start, ties, faulty), total


def main() -> int:
    import json
    ap = argparse.ArgumentParser(description="RQ3 SBFL FL-accuracy")
    ap.add_argument("--dataset", choices=["defects4j", "bugsinpy"], default="defects4j")
    ap.add_argument("--ranking-dir", default=None, help="default: results/sbfl/<dataset>/fl_cache")
    ap.add_argument("--top", nargs="+", type=int, default=[1, 3, 5, 10])
    ap.add_argument("--agent-n", type=int, default=10, help="N actually fed to the agent")
    args = ap.parse_args()

    dataset = args.dataset
    ranking_file = RANKING_FILE[dataset]
    patch_format = PATCH_FORMAT[dataset]
    parse_ranking = parse_gzoltar_ranking if dataset == "defects4j" else parse_fauxpy_ranking
    ranking_dir = Path(args.ranking_dir or PROJECT_ROOT / "results" / "sbfl" / dataset / "fl_cache")
    categories = load_categories(FEASIBILITY_CSV[dataset])
    bug_dirs = sorted(d for d in ranking_dir.iterdir()
                      if d.is_dir() and (d / ranking_file).exists())
    if not bug_dirs:
        print(f"No cached rankings under {ranking_dir} (run the FL stage first).")
        return 1

    by_cat: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
    eff_ranks: List[float] = []     # E_inspect rank of first hit (localizable & found)
    no_gt = []                      # no code ground truth at all (e.g. missing/non-code patch)
    missed = []                     # ground-truth line exists but never ranked

    for d in bug_dirs:
        label = d.name
        cat = categories.get(label, "?")
        meta = json.loads((d / "meta.json").read_text()) if (d / "meta.json").exists() else {}
        src_root = meta.get("src_root", "")

        # Ground-truth developer patch: D4J saves dev.src.patch beside the ranking;
        # BugsInPy reads the dataset's bug_patch.txt (project_bug -> project, bug).
        if dataset == "defects4j":
            patch = d / "dev.src.patch"
        else:
            project, bug = label.rsplit("_", 1)
            patch = get_patch_file_path(project, int(bug))

        by_cat[cat]["bugs"] += 1
        if not patch.exists():
            no_gt.append(label)
            continue
        direct, adjacent = ground_truth_lines(patch.read_text(errors="ignore"), patch_format)
        # modify/delete faults: use their actual buggy lines. pure-omission faults (no buggy
        # line at all): use the insertion-point adjacency (Pearson/Zou/FauxPy convention).
        gt = direct or adjacent
        if not gt:
            no_gt.append(label)
            continue
        if not direct:                  # pure insertion-only fault (localized via adjacency)
            by_cat[cat]["omission"] += 1

        ranking = parse_ranking(str(d / ranking_file), src_root=src_root)
        raw, eff, total = hit_ranks(ranking, gt)
        by_cat[cat]["localizable"] += 1
        if eff is None:
            missed.append(label)
            continue
        eff_ranks.append(eff)
        for n in args.top:                       # Top-N uses E_inspect rank
            if eff <= n:
                by_cat[cat][f"hit@{n}"] += 1
        if raw is not None and raw <= args.agent_n:  # faulty line in agent's actual top-N
            by_cat[cat]["agent"] += 1

    # ---- report ----
    cats_present = [c for c in CATEGORY_ORDER if c in by_cat] + \
                   [c for c in by_cat if c not in CATEGORY_ORDER]
    topcols = "".join(f"{'Top'+str(n):>10}" for n in args.top) + f"{'agent@'+str(args.agent_n):>11}"
    print(f"\n{'cat':<6}{'bugs':>6}{'local':>7}{'omiss':>8}{topcols}")
    print("-" * (27 + 10 * len(args.top) + 11))
    tot = defaultdict(float)

    def cell(h, loc):
        return f"{int(h):>3}/{loc} ({100*h/loc:.0f}%)" if loc else f"{'-':>9}"

    for c in cats_present:
        s = by_cat[c]
        loc = int(s.get("localizable", 0))
        line = f"{c:<6}{int(s.get('bugs',0)):>6}{loc:>7}{int(s.get('omission',0)):>8}"
        for n in args.top:
            line += f"{cell(s.get(f'hit@{n}',0), loc):>10}"
            tot[f"hit@{n}"] += s.get(f"hit@{n}", 0)
        line += f"{cell(s.get('agent',0), loc):>11}"
        tot["agent"] += s.get("agent", 0)
        tot["bugs"] += s.get("bugs", 0); tot["localizable"] += loc
        tot["omission"] += s.get("omission", 0)
        print(line)

    loc = int(tot["localizable"])
    print("-" * (27 + 10 * len(args.top) + 11))
    line = f"{'ALL':<6}{int(tot['bugs']):>6}{loc:>7}{int(tot['omission']):>8}"
    for n in args.top:
        line += f"{cell(tot[f'hit@{n}'], loc):>10}"
    line += f"{cell(tot['agent'], loc):>11}"
    print(line)

    if eff_ranks:
        eff_ranks.sort()
        med = eff_ranks[len(eff_ranks) // 2]
        mean = sum(eff_ranks) / len(eff_ranks)
        print(f"\nE_inspect rank of first hit: median={med:.1f}  mean={mean:.1f}  "
              f"best={eff_ranks[0]:.1f}  worst={eff_ranks[-1]:.1f}  n={len(eff_ranks)}")
    print(f"omission faults (mapped to insertion point, INCLUDED in hit-rate): {int(tot['omission'])}")
    print(f"no code ground truth (missing/non-code patch, excluded): {len(no_gt)}")
    print(f"localizable but ground-truth line never ranked: {len(missed)}"
          + (f"  e.g. {missed[:5]}" if missed else ""))
    tool = "GZoltar" if dataset == "defects4j" else "FauxPy"
    print(f"\nGround truth follows Pearson ICSE'17 / Zou TSE'21 / FauxPy EMSE: modify+delete lines "
          f"plus the buggy line immediately before & after each insertion (omission faults). "
          f"Top-N + E_inspect (tie-aware) match FauxPy's reference (literature_metrics.py). "
          f"agent@{args.agent_n} = faulty line in {tool}'s raw top-{args.agent_n} (what the agent sees). "
          f"EXAM omitted: FauxPy divides E_inspect by total project LINE_COUNT, which we don't yet cache.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
