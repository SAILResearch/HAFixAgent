"""Union-vs-non-history significance for the history-ablation tables.

Backs the ``*`` markers in the two history-ablation tables of the paper:
  * ``tab:rq2-ablation``  -- perfect fault localization  (--regime perfect)
  * ``tab:rq4-ablation``  -- realistic SBFL               (--regime sbfl)
across all three base models (DeepSeek, Qwen, Devstral) and both benchmarks.

For each (model, dataset, category, Total) it runs McNemar's exact test of the
UNION of the three history heuristics (fn_all OR fn_pair OR fl_diff) versus
non-history over the matched per-bug pass sets, and flags significance at the
Bonferroni threshold alpha = 0.05/4 = 0.0125 (four paired comparisons per cell:
each of the three heuristics and their union against non-history). A bug counts
as repaired by the union if at least one of the three heuristics repairs it.

For the perfect regime it also reports, per (model, dataset) Total, the best
single history configuration and its McNemar p-value against non-history. These
back the cross-model effectiveness claims in the RQ2 text (e.g. "no single
history configuration significantly beats non-history: p = 0.12 on Defects4J and
p = 0.79 on BugsInPy").

Universes (matching the two tables):
  * perfect : all evaluated bugs per dataset (854 Defects4J / 501 BugsInPy)
  * sbfl    : the FL-runnable bugs per dataset (827 Defects4J / 361 BugsInPy)
Categories come from dataset/<ds>/<ds>_blame_feasibility.csv (the canonical
category source, shared by every model since category is a property of the bug).

Result readers are reused from analyze_rq2_sbfl_repair (_read / _category_map /
_sbfl_runnable), so pass counts stay identical to tab:rq3_repair and
tab:rq4-ablation; this script adds only the perfect-FL full-universe enumeration
and the union-McNemar test. --validate asserts the DeepSeek perfect-FL grid
against the published numbers before trusting the Qwen/Devstral rows.

Usage:
    python analysis/analyze_ablation_significance.py                 # both regimes, all models
    python analysis/analyze_ablation_significance.py --regime perfect
    python analysis/analyze_ablation_significance.py --regime sbfl
    python analysis/analyze_ablation_significance.py --validate
"""
import argparse
import glob
import sys
from pathlib import Path

from scipy.stats import binomtest

# Reuse the on-disk readers of the RQ3 repair analysis so counts stay identical
# to tab:rq3_repair / tab:rq4-ablation (sibling module in analysis/).
sys.path.insert(0, str(Path(__file__).resolve().parent))
import analyze_rq2_sbfl_repair as R  # noqa: E402

NONHIST = 1
HIST = [5, 7, 8]                       # fn_all, fn_pair, fl_diff
HIST_NAME = {5: "fn_all", 7: "fn_pair", 8: "fl_diff"}
CATS = R.CATS                          # ["SL", "SH", "SFMH", "MFMH"]
DATASETS = [("Defects4J", "defects4j"), ("BugsInPy", "bugsinpy")]
MODELS = [("DeepSeek", ""), ("Qwen", "_qwen3coder"), ("Devstral", "_devstral")]
ALPHA = 0.05 / 4

# Published DeepSeek perfect-FL numbers (the deleted tab:rq2-mcnemar Union column
# and the tab:rq2-ablation-deepseek Union counts) used by --validate.
EXPECT_PERFECT_DEEPSEEK = {
    "defects4j": {"n": 854,
                  "union": {"SL": 156, "SH": 122, "SFMH": 340, "MFMH": 66, "Total": 684},
                  "sig": {"SL": True, "SH": True, "SFMH": True, "MFMH": True, "Total": True}},
    "bugsinpy": {"n": 501,
                 "union": {"SL": 80, "SH": 120, "SFMH": 187, "MFMH": 76, "Total": 463},
                 "sig": {"SL": False, "SH": True, "SFMH": True, "MFMH": True, "Total": True}},
}


def perfect_universe(dataset):
    """All evaluated bugs for the current model (R.TAG), from the perfect-FL tree.

    DeepSeek is laid out as <cat>/<bug>, Qwen/Devstral as all/<bug>; the "*/*"
    glob captures both. Union across configs is unnecessary: non-history ran on
    every bug, so any bug directory is in the universe.
    """
    base = R.PROJECT_ROOT / "results" / dataset / f"llm_judge_1line{R.TAG}"
    bugs = set()
    for d in glob.glob(str(base / "*" / "*")):
        p = Path(d)
        if p.is_dir():
            bugs.add(p.name)
    return bugs


def build_fixed(regime, dataset, bugs):
    """{val: {bug: bool}} pass sets for all four configs, one FL regime."""
    tree = dataset if regime == "perfect" else f"sbfl/{dataset}"
    return {v: {b: R._read(tree, b, v)["fixed"] for b in bugs}
            for v, _, _ in R.CONFIGS}


def _mcnemar(bugs, a_pass, b_pass):
    """b = A-pass & B-fail, c = A-fail & B-pass; exact binomial two-sided p."""
    b = c = 0
    for bug in bugs:
        ap, bp = a_pass(bug), b_pass(bug)
        if ap and not bp:
            b += 1
        elif bp and not ap:
            c += 1
    n = b + c
    p = binomtest(min(b, c), n, 0.5, alternative="two-sided").pvalue if n else 1.0
    return b, c, p


def union_mcnemar(bugs, fixed):
    return _mcnemar(bugs,
                    lambda x: any(fixed[v][x] for v in HIST),
                    lambda x: fixed[NONHIST][x])


def best_single_mcnemar(bugs, fixed):
    cnt = {v: sum(1 for x in bugs if fixed[v][x]) for v in HIST}
    best = max(HIST, key=lambda v: cnt[v])
    b, c, p = _mcnemar(bugs, lambda x: fixed[best][x], lambda x: fixed[NONHIST][x])
    return best, cnt[best], b, c, p


def per_config_mcnemar(bugs, fixed):
    """Each history config vs non-history: {v: dict(cnt, b_only, c_only, p, sig)}.

    b_only = config repairs & non-history fails; c_only = the reverse. Uses the
    SAME Bonferroni threshold as the union test (alpha = 0.05/4), since the /4
    already budgets one comparison for each of the three heuristics plus union.
    """
    out = {}
    for v in HIST:
        b, c, p = _mcnemar(bugs, lambda x: fixed[v][x], lambda x: fixed[NONHIST][x])
        cnt = sum(1 for x in bugs if fixed[v][x])
        out[v] = dict(cnt=cnt, b_only=b, c_only=c, p=p, sig=p < ALPHA)
    return out


def _pstr(p):
    return "<0.001" if p < 1e-3 else f"{p:.3f}"


def sig_lookup(regime, tag):
    """Per-config significance for one model, keyed for the LaTeX generators.

    Returns {(dataset, cat, config_name): bool}, config_name in
    {"fn_all","fn_pair","fl_diff"}, cat in CATS+["Total"], dataset the short id
    ("defects4j"/"bugsinpy"). ``True`` means that single history configuration
    significantly repairs bugs \\nonhistory cannot (McNemar exact, Bonferroni
    alpha = 0.05/4), i.e. the cell earns a ``*``. Reuses the same readers/universe
    as the pass-count generators, so stars align cell-for-cell with the tables.
    """
    R.TAG = tag
    out = {}
    for _label, ds in DATASETS:
        catmap = R._category_map(ds)
        universe = perfect_universe(ds) if regime == "perfect" else R._sbfl_runnable(ds)
        bugs = sorted(b for b in universe if b in catmap)
        fixed = build_fixed(regime, ds, bugs)
        for cat in CATS + ["Total"]:
            cb = bugs if cat == "Total" else [b for b in bugs if catmap[b] == cat]
            pc = per_config_mcnemar(cb, fixed)
            for v in HIST:
                out[(ds, cat, HIST_NAME[v])] = pc[v]["sig"]
            _b, _c, up = union_mcnemar(cb, fixed)
            out[(ds, cat, "union")] = up < ALPHA
    return out


def analyze_cell(bugs, fixed):
    """union count + union-vs-non McNemar for one cell (list of bugs)."""
    union = sum(1 for x in bugs if any(fixed[v][x] for v in HIST))
    b, c, p = union_mcnemar(bugs, fixed)
    return dict(n=len(bugs), union=union, b_only=b, c_only=c, p=p, sig=p < ALPHA,
                configs=per_config_mcnemar(bugs, fixed))


def run(regime, models=MODELS, verbose=True):
    """Return {(model, dataset): {cat: cell, ..., 'best': (...)}}; print a grid."""
    results = {}
    if verbose:
        title = "PERFECT FL" if regime == "perfect" else "REALISTIC SBFL"
        print(f"\n{'=' * 62}\n {title}   (* = union significant at "
              f"Bonferroni alpha = 0.05/4 = {ALPHA:.4f})\n{'=' * 62}")
    for model_name, tag in models:
        R.TAG = tag
        catmap = R._category_map(dataset="defects4j")  # placeholder, reset per dataset
        for label, ds in DATASETS:
            catmap = R._category_map(ds)
            universe = (perfect_universe(ds) if regime == "perfect"
                        else R._sbfl_runnable(ds))
            bugs = sorted(b for b in universe if b in catmap)
            fixed = build_fixed(regime, ds, bugs)
            cells = {}
            for cat in CATS:
                cells[cat] = analyze_cell([b for b in bugs if catmap[b] == cat], fixed)
            cells["Total"] = analyze_cell(bugs, fixed)
            best_v, best_cnt, bb, bc, bp = best_single_mcnemar(bugs, fixed)
            cells["best"] = (best_v, best_cnt, bb, bc, bp)
            results[(model_name, ds)] = cells
            if verbose:
                print(f"\n--- {model_name} | {label}  (n={len(bugs)}) ---")
                hdr = f"  {'cell':5s} {'fn_all':>14s} {'fn_pair':>14s} {'fl_diff':>14s} {'UNION':>10s}"
                print(hdr)
                for cat in CATS + ["Total"]:
                    d = cells[cat]
                    parts = []
                    for v in HIST:
                        cf = d["configs"][v]
                        star = "*" if cf["sig"] else " "
                        parts.append(f"{cf['cnt']:3d} p={_pstr(cf['p']):>6s}{star}")
                    umark = "*" if d["sig"] else " "
                    parts.append(f"{d['union']:3d}{umark}")
                    print(f"  {cat:5s} " + " ".join(f"{p:>14s}" for p in parts[:3])
                          + f" {parts[3]:>10s}")
                # compact star grids for quick transcription into the tables
                for v in HIST:
                    g = ["*" if cells[cat]["configs"][v]["sig"] else "-"
                         for cat in CATS + ["Total"]]
                    print(f"  -> {HIST_NAME[v]:7s} * grid [SL SH SFMH MFMH Total]: {g}")
                ug = ["*" if cells[cat]["sig"] else "-" for cat in CATS + ["Total"]]
                print(f"  -> {'Union':7s} * grid [SL SH SFMH MFMH Total]: {ug}")
    return results


def validate():
    """Assert the DeepSeek perfect-FL grid matches the published paper numbers."""
    res = run("perfect", models=[("DeepSeek", "")], verbose=False)
    ok = True
    for ds, exp in EXPECT_PERFECT_DEEPSEEK.items():
        cells = res[("DeepSeek", ds)]
        n = cells["Total"]["n"]
        if n != exp["n"]:
            print(f"[FAIL] {ds}: universe n={n}, expected {exp['n']}")
            ok = False
        for cat in CATS + ["Total"]:
            got_u, got_s = cells[cat]["union"], cells[cat]["sig"]
            exp_u, exp_s = exp["union"][cat], exp["sig"][cat]
            tag = "ok" if (got_u == exp_u and got_s == exp_s) else "FAIL"
            if tag == "FAIL":
                ok = False
            print(f"  [{tag}] DeepSeek {ds:9s} {cat:5s}: "
                  f"union {got_u} (exp {exp_u}), sig {got_s} (exp {exp_s})")
    print("\nVALIDATION", "PASSED" if ok else "FAILED")
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--regime", choices=["perfect", "sbfl", "both"], default="both")
    ap.add_argument("--validate", action="store_true",
                    help="assert the DeepSeek perfect-FL grid against the paper, then exit")
    args = ap.parse_args()

    if args.validate:
        sys.exit(0 if validate() else 1)

    regimes = ["perfect", "sbfl"] if args.regime == "both" else [args.regime]
    for regime in regimes:
        run(regime)


if __name__ == "__main__":
    main()
