"""Locally-deployed models (Qwen, Devstral) history ablation on BOTH datasets, in tab:rq1-ablation format.

Reads results/<dataset>/llm_judge_1line_<tag>/all/<bug>/<bug>_<val>_result.json
  val: 1 = non-history, 5 = fn_all, 7 = fn_pair, 8 = fl_diff
Category (SL/SH/SFMH/MFMH) from dataset/<dataset>/<dataset>_blame_feasibility.csv.
Only the canonical `all/` run is read (auxiliary re-run subdirs are ignored), so
counts match how DeepSeek and Qwen were scored for tab:rq1-ablation.

Each cell: #Pass with a +unique-fix superscript = the number of bugs a history config
repairs that non-history does not (set difference fix(config) minus fix(non-history)). Bold = best per
category among the four configs; Union and the superscripts are never bold. This matches
the merged main-text table tab:rq2-{qwen,devstral} exactly (the superscripts are wrapped
in a smash there only to keep the three-subtable float on one page; the numbers are the same).

Usage: python analysis/analyze_local_model_ablation.py [--tag <tag>] [--name <Name>] [--latex]
  Defaults: --tag qwen3coder --name Qwen  (backward compatible).
  Devstral: python analysis/analyze_local_model_ablation.py --tag devstral --name Devstral --latex
"""
import csv, json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
import analyze_ablation_significance as SIG  # noqa: E402

VALS = ["1", "5", "7", "8"]
HIST = ["5", "7", "8"]  # fn_all, fn_pair, fl_diff (Union = fixed by at least one)
VAL_NAME = {"5": "fn_all", "7": "fn_pair", "8": "fl_diff"}
CATS = ["SL", "SH", "SFMH", "MFMH"]
DATASETS = [("Defects4J", "defects4j"), ("BugsInPy", "bugsinpy")]


def _arg(flag, default):
    return sys.argv[sys.argv.index(flag) + 1] if flag in sys.argv else default


TAG = _arg("--tag", "qwen3coder")
NAME = _arg("--name", "Qwen")
# Per-config McNemar significance (perfect FL) for this model, keyed
# (dataset, cat, config_name); a True cell earns a * next to its +unique superscript.
SIGMAP = SIG.sig_lookup("perfect", f"_{TAG}")


def load(ds):
    res = ROOT / "results" / ds / f"llm_judge_1line_{TAG}" / "all"
    csvp = ROOT / "dataset" / ds / f"{ds}_blame_feasibility.csv"
    cat_of = {}
    with open(csvp) as f:
        for row in csv.DictReader(f):
            cat_of[row["Bug_ID"]] = row["Category"]
    succ = {v: {} for v in VALS}
    for bugdir in sorted(res.iterdir()):
        if not bugdir.is_dir():
            continue
        bug = bugdir.name
        for v in VALS:
            rf = bugdir / f"{bug}_{v}_result.json"
            if rf.exists():
                try:
                    succ[v][bug] = bool(json.load(open(rf)).get("repair_result", {}).get("success", False))
                except Exception:
                    succ[v][bug] = False
    allbugs = set()
    for v in VALS:
        allbugs |= set(succ[v])
    cat_bugs = {c: sorted(b for b in allbugs if cat_of.get(b) == c) for c in CATS}
    return succ, cat_bugs, sorted(allbugs)


latex = []
for label, ds in DATASETS:
    succ, cat_bugs, allbugs = load(ds)
    print(f"\n=== {NAME}  {label}  (bugs: {len(allbugs)}) ===")
    print(f"{'Cat':6}{'Total':>7}{'non-hist':>14}{'fn_all':>13}{'fn_pair':>13}{'fl_diff':>13}{'Union':>8}")
    rows = []
    for c in CATS + ["Total"]:
        bugs = cat_bugs[c] if c in CATS else allbugs
        tot = len(bugs)
        fix = {v: set(b for b in bugs if succ[v].get(b)) for v in VALS}
        cnt = {v: len(fix[v]) for v in VALS}
        nh = fix["1"]
        uniq = {v: len(fix[v] - nh) for v in HIST}  # bugs config v fixes that non-history does not
        union = len(fix["5"] | fix["7"] | fix["8"])
        best = max(cnt.values())

        def disp(v):
            pct = 100 * cnt[v] / tot if tot else 0
            sup = f" (+{uniq[v]})" if v in HIST else ""
            return f"{cnt[v]}{sup} ({pct:.1f}%)"

        print(f"{c:6}{tot:>7}" + "".join(f"{disp(v):>18}" for v in VALS) + f"{union:>8}")

        # LaTeX cells match the merged main-text table (tab:rq2-{qwen,devstral}):
        # #Pass, bold = best among the four configs, +unique superscript on history configs,
        # Union appended (never bold). The external Baseline column is added separately.
        def cell(v):
            s = f"\\textbf{{{cnt[v]}}}" if cnt[v] == best else str(cnt[v])
            if v in HIST:
                star = "\\,*" if SIGMAP.get((ds, c, VAL_NAME[v]), False) else ""
                s += f"\\,$^{{\\smash{{+{uniq[v]}{star}}}}}$"
            return s

        if c == "SL":
            lead = f"    \\multirow{{5}}{{*}}{{{label}}}\n     & SL"
        elif c == "Total":
            lead = "    \\cmidrule(lr){2-9}\n     & \\textbf{Total}"
        else:
            lead = f"     & {c}"
        ustar = "\\,$^{*}$" if SIGMAP.get((ds, c, "union"), False) else ""
        rows.append(f"{lead} & {tot} & " + " & ".join(cell(v) for v in VALS)
                    + f" & {union}{ustar} \\\\")
    latex.append("\n".join(rows))

if "--latex" in sys.argv:
    print(f"\n% ===== ablation + Union for {NAME}, both datasets =====")
    print(f"% (external Baseline column is added separately: RepairAgent + BIRCH-feedback)")
    print("\n    \\midrule\n".join(latex))
