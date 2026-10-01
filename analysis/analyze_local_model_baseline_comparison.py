"""Same-LLM baseline comparison for the locally-deployed models, tab:rq2-ablation subtable format.

Covers both locally-deployed base models via --model {qwen,devstral} (default qwen):
(A) HAFixAgent vs RepairAgent on all 854 Defects4J bugs.
(B) HAFixAgent vs BIRCH-feedback on BIRCH's multi-hunk common set.

HAFixAgent: results/defects4j/<ha>/all/<bug>/<bug>_<val>_result.json
RepairAgent: results/defects4j/baselines/<ra>/<bug>/plausible_patches/
BIRCH: results/defects4j/baselines/<birch>/test_results_mode_4_merged.csv
Categories: dataset/defects4j/defects4j_blame_feasibility.csv.
Bold = best per category among the four HAFixAgent configs (baseline excluded, matching the caption).
"""
import argparse, csv, json, sys
from pathlib import Path
from collections import defaultdict, Counter

MODELS = {
    "qwen": {
        "label": "Qwen3-Coder-Next",
        "ha": "llm_judge_1line_qwen3coder",
        "ra": "repairagent_qwen3-coder-next",
        "birch": "birch_openai_qwen3-coder-next",
    },
    "devstral": {
        "label": "Devstral-Small-2",
        "ha": "llm_judge_1line_devstral",
        "ra": "repairagent_devstral-small-2",
        "birch": "birch_openai_devstral-small-2",
    },
}

ap = argparse.ArgumentParser()
ap.add_argument("--model", choices=list(MODELS), default="qwen")
ap.add_argument("--latex", action="store_true")
args = ap.parse_args()
CFG = MODELS[args.model]

ROOT = Path(__file__).resolve().parent.parent
CSV_CAT = ROOT / "dataset" / "defects4j" / "defects4j_blame_feasibility.csv"
HA_DIR = ROOT / "results" / "defects4j" / CFG["ha"] / "all"
RA_DIR = ROOT / "results" / "defects4j" / "baselines" / CFG["ra"]
BIRCH_CSV = ROOT / "results" / "defects4j" / "baselines" / CFG["birch"] / "test_results_mode_4_merged.csv"
VALS = ["1", "5", "7", "8"]
CATS = ["SL", "SH", "SFMH", "MFMH"]

cat_of = {}
with open(CSV_CAT) as f:
    for row in csv.DictReader(f):
        cat_of[row["Bug_ID"]] = row["Category"]
cat_total = Counter(cat_of.values())

ha = {v: defaultdict(dict) for v in VALS}
for d in HA_DIR.iterdir():
    if not d.is_dir() or d.name not in cat_of:
        continue
    c = cat_of[d.name]
    for v in VALS:
        rf = d / f"{d.name}_{v}_result.json"
        if rf.exists():
            try:
                ha[v][c][d.name] = bool(json.load(open(rf)).get("repair_result", {}).get("success", False))
            except Exception:
                ha[v][c][d.name] = False

ra = defaultdict(dict)
for d in RA_DIR.iterdir():
    if not d.is_dir() or d.name not in cat_of:
        continue
    has = False
    pp = d / "plausible_patches"
    if pp.exists():
        for f in pp.iterdir():
            if f.suffix == ".json":
                try:
                    if json.load(open(f)):
                        has = True
                        break
                except Exception:
                    pass
    ra[cat_of[d.name]][d.name] = has

csv.field_size_limit(sys.maxsize)
birch = defaultdict(dict)
with open(BIRCH_CSV) as f:
    for row in csv.DictReader(f):
        bug = row["bug"].replace("-", "_")
        if bug not in cat_of:
            continue
        passed = row.get("pass", "").strip().lower() == "yes"
        birch[cat_of[bug]][bug] = birch[cat_of[bug]].get(bug, False) or passed


def n(d):
    return sum(1 for x in d.values() if x)


def bold(val, mx):
    return f"\\textbf{{{val}}}" if val == mx else str(val)


print(f"=== (A) All 854 Defects4J bugs: HAFixAgent({CFG['label']}) vs RepairAgent ===")
print(f"{'Cat':6}{'Total':>7}{'non-h':>7}{'fn_all':>7}{'fnpair':>7}{'fldiff':>7}{'RepairA':>9}")
texA = []
totA = {v: 0 for v in VALS}
ra_tot = 0
for c in CATS:
    hc = {v: n(ha[v][c]) for v in VALS}
    rc = n(ra[c])
    for v in VALS:
        totA[v] += hc[v]
    ra_tot += rc
    mx = max(hc.values())  # bold = best among the four HAFixAgent configs only (baseline excluded)
    print(f"{c:6}{cat_total[c]:>7}" + "".join(f"{hc[v]:>7}" for v in VALS) + f"{rc:>9}")
    texA.append(f"    {c} & {cat_total[c]} & " + " & ".join(bold(hc[v], mx) for v in VALS) + f" & {rc} \\\\")
mxT = max(totA.values())
print(f"{'Total':6}{854:>7}" + "".join(f"{totA[v]:>7}" for v in VALS) + f"{ra_tot:>9}")
totrowA = f"    Total & 854 & " + " & ".join(bold(totA[v], mxT) for v in VALS) + f" & {ra_tot} \\\\"

print(f"\n=== (B) BIRCH common multi-hunk: HAFixAgent({CFG['label']}) vs BIRCH ===")
print(f"{'Cat':6}{'Common':>7}{'non-h':>7}{'fn_all':>7}{'fnpair':>7}{'fldiff':>7}{'BIRCH':>7}")
texB = []
totB = {v: 0 for v in VALS}
b_tot = 0
common_tot = 0
for c in ["SFMH", "MFMH"]:
    common = set(birch[c].keys())
    common_tot += len(common)
    hc = {v: sum(1 for b in common if ha[v][c].get(b)) for v in VALS}
    bc = sum(1 for b in common if birch[c][b])
    for v in VALS:
        totB[v] += hc[v]
    b_tot += bc
    mx = max(hc.values())  # bold = best among the four HAFixAgent configs only (baseline excluded)
    print(f"{c:6}{len(common):>7}" + "".join(f"{hc[v]:>7}" for v in VALS) + f"{bc:>7}")
    texB.append(f"    {c} & {len(common)} & " + " & ".join(bold(hc[v], mx) for v in VALS) + f" & {bc} \\\\")
mxB = max(totB.values())
print(f"{'Total':6}{common_tot:>7}" + "".join(f"{totB[v]:>7}" for v in VALS) + f"{b_tot:>7}")
totrowB = f"    Total & {common_tot} & " + " & ".join(bold(totB[v], mxB) for v in VALS) + f" & {b_tot} \\\\"

if args.latex:
    print("\n% ---- (A) rows ----\n" + "\n".join(texA) + "\n    \\midrule\n" + totrowA)
    print("\n% ---- (B) rows ----\n" + "\n".join(texB) + "\n    \\midrule\n" + totrowB)
