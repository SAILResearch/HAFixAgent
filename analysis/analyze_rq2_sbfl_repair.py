"""RQ3 repair effectiveness + cost/steps — SBFL (realistic FL) vs RQ1 (perfect FL).

For each config and bug in the RQ3 sample (evaluation/sbfl/rq3_repair_sample.json), compare
plausible@1 (exit == "Submitted") under:
  * RQ1 perfect FL    : results/<dataset>/llm_judge_1line/<cat>/<bug>/<bug>_<val>_result.json
  * RQ3 realistic SBFL: results/sbfl/<dataset>/llm_judge_1line/<cat>/<bug>/<bug>_<val>_result.json

Errors and missing results count as failures (consistent across configs). Cost/steps use the
tracked `model_cost` and `agent_steps` fields (same as the RQ2b cost code) over successful
repairs. Produces console tables and, with --latex, the two paper tables:
  * tab:rq3_repair  — success rate (%) per category, perfect vs SBFL, both datasets
  * tab:rq3_cost    — median steps + cost (USD) for successful repairs, perfect vs SBFL

Usage:
    python analysis/analyze_rq2_sbfl_repair.py
    python analysis/analyze_rq2_sbfl_repair.py --latex
"""
import argparse
import csv
import glob
import json
import statistics
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
# (value, name, LaTeX macro)
CONFIGS = [(1, "baseline", r"\nonhistory"), (5, "fn_all", r"\fnall"),
           (7, "fn_pair", r"\fnpair"), (8, "fl_diff", r"\fldiff")]
CATS = ["SL", "SH", "SFMH", "MFMH"]

# Model result subdir suffix: "" = DeepSeek (default), "_qwen3coder", "_devstral".
# Set from --tag in main(); both perfect (results/<ds>/llm_judge_1line<TAG>) and
# SBFL (results/sbfl/<ds>/llm_judge_1line<TAG>) trees follow the same convention.
TAG = ""


def _read(tree, label, val):
    fs = glob.glob(str(PROJECT_ROOT / "results" / tree / f"llm_judge_1line{TAG}" /
                       "*" / label / f"{label}_{val}_result.json"))
    if not fs:
        return {"fixed": False, "steps": None, "cost": None}   # missing -> failure
    rr = json.loads(Path(fs[0]).read_text()).get("repair_result", {})
    if "error" in rr:
        return {"fixed": False, "steps": None, "cost": None}   # error -> failure
    return {"fixed": bool(rr.get("success")), "steps": rr.get("agent_steps", 0),
            "cost": rr.get("model_cost", 0) or 0}


def _category_map(dataset):
    cat_of = {}
    csvp = PROJECT_ROOT / "dataset" / dataset / f"{dataset}_blame_feasibility.csv"
    with open(csvp) as f:
        for r in csv.DictReader(f):
            cat_of[r["Bug_ID"]] = r["Category"]
    return cat_of


def _sbfl_runnable(dataset):
    """Full FL-runnable set: bugs with a cached SBFL baseline (val 1) result."""
    base = PROJECT_ROOT / "results" / "sbfl" / dataset / f"llm_judge_1line{TAG}"
    bugs = set()
    for f in glob.glob(str(base / "*" / "*" / "*_1_result.json")):
        bug, val = Path(f).name[:-len("_result.json")].rsplit("_", 1)
        if val == "1":
            bugs.add(bug)
    return bugs


def analyze(dataset):
    # Full scale: all SBFL-runnable bugs (827 D4J / 361 BugsInPy), categories from the
    # blame-feasibility CSV. (The earlier 200-bug rq3_repair_sample.json is superseded.)
    cat_all = _category_map(dataset)
    cat_of = {b: cat_all[b] for b in _sbfl_runnable(dataset) if b in cat_all}
    labels = sorted(cat_of)
    res = {}
    for val, name, _ in CONFIGS:
        perf = {l: _read(dataset, l, val) for l in labels}
        sbfl = {l: _read(f"sbfl/{dataset}", l, val) for l in labels}

        def count(d, cat=None):
            sub = [l for l in labels if cat is None or cat_of[l] == cat]
            return sum(d[l]["fixed"] for l in sub)

        def rate(d, cat=None):
            sub = [l for l in labels if cat is None or cat_of[l] == cat]
            return 100.0 * count(d, cat) / len(sub)

        def med(d, key):
            vals = [d[l][key] for l in labels if d[l]["fixed"] and d[l][key] is not None]
            return statistics.median(vals) if vals else float("nan")

        res[name] = {
            "perf_overall": rate(perf), "sbfl_overall": rate(sbfl),
            "perf_cat": {c: rate(perf, c) for c in CATS},
            "sbfl_cat": {c: rate(sbfl, c) for c in CATS},
            "perf_cat_n": {c: count(perf, c) for c in CATS},
            "sbfl_cat_n": {c: count(sbfl, c) for c in CATS},
            "perf_steps": med(perf, "steps"), "sbfl_steps": med(sbfl, "steps"),
            "perf_cost": med(perf, "cost"), "sbfl_cost": med(sbfl, "cost"),
            "n_perf": sum(perf[l]["fixed"] for l in labels),
            "n_sbfl": sum(sbfl[l]["fixed"] for l in labels),
            "n_total": len(labels),
            "cat_total": {c: sum(1 for l in labels if cat_of[l] == c) for c in CATS},
        }
    return res


def paired_cost(dataset):
    """Strict paired perfect-vs-SBFL cost/steps: per config, restrict to bugs fixed
    under BOTH FL settings, then Wilcoxon signed-rank on the matched pairs."""
    from scipy.stats import wilcoxon
    cat_all = _category_map(dataset)
    labels = sorted(b for b in _sbfl_runnable(dataset) if b in cat_all)
    res = {}
    for val, name, _ in CONFIGS:
        perf = {l: _read(dataset, l, val) for l in labels}
        sbfl = {l: _read(f"sbfl/{dataset}", l, val) for l in labels}
        m = [l for l in labels if perf[l]["fixed"] and sbfl[l]["fixed"]
             and perf[l]["cost"] is not None and sbfl[l]["cost"] is not None]
        pc, sc = [perf[l]["cost"] for l in m], [sbfl[l]["cost"] for l in m]
        ps, ss = [perf[l]["steps"] for l in m], [sbfl[l]["steps"] for l in m]

        def wp(a, b):
            if not a or all(x == y for x, y in zip(a, b)):
                return float("nan")
            try:
                return wilcoxon(a, b).pvalue
            except Exception:
                return float("nan")

        res[name] = {
            "n": len(m),
            "perf_cost": statistics.median(pc) if pc else float("nan"),
            "sbfl_cost": statistics.median(sc) if sc else float("nan"),
            "perf_steps": statistics.median(ps) if ps else float("nan"),
            "sbfl_steps": statistics.median(ss) if ss else float("nan"),
            "p_cost": wp(pc, sc), "p_steps": wp(ps, ss),
        }
    return res


def sbfl_config_friedman(dataset):
    """Config axis UNDER SBFL: Friedman across the 4 configs on cost, on the
    matched-across-configs set (bugs fixed by all 4 configs under SBFL). Tests
    'does history inflate cost under SBFL?' (complement to tab:rq2b's perfect-FL test)."""
    from scipy.stats import friedmanchisquare
    cat_all = _category_map(dataset)
    labels = sorted(b for b in _sbfl_runnable(dataset) if b in cat_all)
    names = [n for _, n, _ in CONFIGS]
    data = {n: {l: _read(f"sbfl/{dataset}", l, v) for l in labels}
            for v, n, _ in CONFIGS}

    from scipy.stats import wilcoxon

    def fried(subset):
        m = [l for l in subset
             if all(data[n][l]["fixed"] and data[n][l]["cost"] is not None for n in names)]
        if len(m) < 3:
            return {"p": float("nan"), "n": len(m), "med": {}, "pair": {}}
        cols = {n: [data[n][l]["cost"] for l in m] for n in names}
        try:
            p = friedmanchisquare(*[cols[n] for n in names]).pvalue
        except Exception:
            p = float("nan")
        med = {n: statistics.median(cols[n]) for n in names}
        # pairwise vs non-history (baseline), Bonferroni a=0.0167
        pair = {}
        for n in names[1:]:
            try:
                pair[n] = wilcoxon(cols["baseline"], cols[n]).pvalue
            except Exception:
                pair[n] = float("nan")
        return {"p": p, "n": len(m), "med": med, "pair": pair}

    return {"overall": fried(labels),
            "cats": {c: fried([l for l in labels if cat_all[l] == c]) for c in CATS}}


def sbfl_resilience_tests(dataset):
    """RESILIENCE (difference-in-differences): does each history config degrade LESS than
    \\nonhistory from perfect->SBFL, per category? This is the test for the supervisor's
    'history is more resilient' reading of the shaded table (a 2 FL x 2 config interaction on
    the binary fix outcome, paired within bug). Per bug: d = (non degraded) - (hist degraded),
    degraded = fixed-under-perfect but not-under-SBFL. One-sided exact sign test (McNemar-type)
    on discordant bugs + one-sided Wilcoxon signed-rank; Holm across the category x config grid."""
    from scipy.stats import binomtest, wilcoxon
    cat_all = _category_map(dataset)
    labels = sorted(b for b in _sbfl_runnable(dataset) if b in cat_all)
    perf = {n: {l: int(_read(dataset, l, v)["fixed"]) for l in labels} for v, n, _ in CONFIGS}
    sbfl = {n: {l: int(_read(f"sbfl/{dataset}", l, v)["fixed"]) for l in labels} for v, n, _ in CONFIGS}
    hist = [n for _, n, _ in CONFIGS if n != "baseline"]
    rows = []
    for c in CATS + ["ALL"]:
        sub = labels if c == "ALL" else [l for l in labels if cat_all[l] == c]
        m = len(sub)
        for h in hist:
            diff = [(perf["baseline"][l] - sbfl["baseline"][l]) - (perf[h][l] - sbfl[h][l]) for l in sub]
            adv = 100.0 * sum(diff) / m if m else float("nan")
            npos = sum(1 for d in diff if d > 0)   # non degraded more -> history more resilient here
            nneg = sum(1 for d in diff if d < 0)   # history degraded more
            nd = npos + nneg
            p_sign = binomtest(npos, nd, 0.5, alternative="greater").pvalue if nd else float("nan")
            nz = [d for d in diff if d != 0]
            try:
                p_w = wilcoxon(nz, alternative="greater").pvalue if nz else float("nan")
            except Exception:
                p_w = float("nan")
            rows.append({"cat": c, "config": h, "n": m, "adv": adv,
                         "npos": npos, "nneg": nneg, "p_sign": p_sign, "p_w": p_w})
    # Holm over the per-category family only (the 12 cat x config tests); "ALL" is a separate
    # pre-specified omnibus and keeps its raw p.
    idx = [i for i, r in enumerate(rows)
           if r["cat"] != "ALL" and r["p_sign"] == r["p_sign"]]
    order = sorted(idx, key=lambda i: rows[i]["p_sign"])
    k, prev = len(order), 0.0
    for rank, i in enumerate(order):
        prev = min(1.0, max(prev, (k - rank) * rows[i]["p_sign"]))
        rows[i]["p_holm"] = prev
    for r in rows:
        if r["cat"] == "ALL":
            r["p_holm"] = r["p_sign"]   # display raw (no family correction)
    return rows


def sbfl_resilience_report(dataset, rows):
    print(f"\n[{dataset.upper()}] RESILIENCE: history degrades LESS than non-history (perfect->SBFL)?")
    print(f"   {'cat':5}{'config':9}{'N':>4}{'adv_pp':>8}{'non>h':>7}{'h>non':>7}"
          f"{'p_sign':>8}{'p_holm':>8}{'p_wilc':>8}")
    for r in rows:
        star = " *" if r.get("p_holm", 1.0) < 0.05 else ""
        print(f"   {r['cat']:5}{r['config']:9}{r['n']:>4}{r['adv']:>8.1f}{r['npos']:>7}{r['nneg']:>7}"
              f"{r['p_sign']:>8.3f}{r.get('p_holm', float('nan')):>8.3f}{r['p_w']:>8.3f}{star}")


def sbfl_effectiveness_tests(dataset):
    """Effectiveness significance UNDER SBFL: is history better than non-history at fixing bugs?
    Cochran's Q across the 4 configs (omnibus, paired binary) + exact McNemar (best-history vs
    non-history), per dataset and per category. Parallels the cost Friedman/Wilcoxon tests."""
    from scipy.stats import chi2, binomtest
    cat_all = _category_map(dataset)
    labels = sorted(b for b in _sbfl_runnable(dataset) if b in cat_all)
    names = [n for _, n, _ in CONFIGS]
    fixed = {n: {l: _read(f"sbfl/{dataset}", l, v)["fixed"] for l in labels}
             for v, n, _ in CONFIGS}

    def cochran_q(sub):
        rows = [[1 if fixed[n][l] else 0 for n in names] for l in sub]
        k, N = len(names), len(rows)
        Cj = [sum(r[j] for r in rows) for j in range(k)]
        Ri = [sum(r) for r in rows]
        T = sum(Cj)
        denom = k * T - sum(x * x for x in Ri)
        if denom == 0 or N == 0:
            return float("nan"), N
        Q = (k - 1) * (k * sum(c * c for c in Cj) - T * T) / denom
        return chi2.sf(Q, k - 1), N

    def mcnemar(a, base, sub):
        b_only = sum(1 for l in sub if fixed[a][l] and not fixed[base][l])
        c_only = sum(1 for l in sub if fixed[base][l] and not fixed[a][l])
        n = b_only + c_only
        p = binomtest(min(b_only, c_only), n, 0.5).pvalue if n else float("nan")
        return p, b_only, c_only

    cnt = {n: sum(1 for l in labels if fixed[n][l]) for n in names}
    best_hist = max(names[1:], key=lambda n: cnt[n])
    res = {"best_hist": best_hist, "cnt": cnt,
           "cochran": cochran_q(labels), "mcnemar": mcnemar(best_hist, "baseline", labels),
           "cat": {}}
    for c in CATS:
        sub = [l for l in labels if cat_all[l] == c]
        res["cat"][c] = {"cochran": cochran_q(sub),
                         "mcnemar": mcnemar(best_hist, "baseline", sub)}
    return res


def sbfl_effectiveness_report(dataset, r):
    bh, mp, b_only, c_only = r["best_hist"], *r["mcnemar"]
    print(f"\n[{dataset.upper()}] effectiveness UNDER SBFL (history vs non-history):")
    print(f"   best-history = {bh} (fixes {r['cnt'][bh]} vs non-history {r['cnt']['baseline']})")
    print(f"   Cochran's Q (4 configs) overall p={r['cochran'][0]:.4f} (N={r['cochran'][1]})")
    print(f"   McNemar {bh} vs non-history: p={mp:.4f}  ({bh}-only={b_only}, non-only={c_only})")
    for c in CATS:
        cc = r["cat"][c]
        mcp, bo, co = cc["mcnemar"]
        print(f"   {c}: Cochran p={cc['cochran'][0]:.3f} (N={cc['cochran'][1]}) | "
              f"McNemar p={mcp:.3f} ({bh}-only={bo}, non-only={co})")


def paired_report(dataset, res):
    print(f"\n=== {dataset.upper()} — PAIRED cost/steps (bugs fixed under BOTH perfect & SBFL) ===")
    print(f"{'config':9s} {'Nmatch':>6s} {'Pcost':>7s} {'Scost':>7s} {'p_cost':>7s} "
          f"{'Pstep':>6s} {'Sstep':>6s} {'p_step':>7s}")
    for _, name, _ in CONFIGS:
        r = res[name]
        print(f"{name:9s} {r['n']:6d} {r['perf_cost']:7.4f} {r['sbfl_cost']:7.4f} "
              f"{r['p_cost']:7.4f} {r['perf_steps']:6.1f} {r['sbfl_steps']:6.1f} {r['p_steps']:7.4f}")


def report(dataset, res):
    print(f"\n=== {dataset.upper()} — success rate %% (plausible@1; errors=failures) ===")
    print(f"{'config':9s} {'FL':7s} {'Ovr':>5s} " + " ".join(f"{c:>5s}" for c in CATS))
    for _, name, _ in CONFIGS:
        r = res[name]
        print(f"{name:9s} {'Perfect':7s} {r['perf_overall']:5.1f} " +
              " ".join(f"{r['perf_cat'][c]:5.0f}" for c in CATS))
        print(f"{'':9s} {'SBFL':7s} {r['sbfl_overall']:5.1f} " +
              " ".join(f"{r['sbfl_cat'][c]:5.0f}" for c in CATS))
    print(f"\n{'config':9s} {'P.steps':>8s} {'S.steps':>8s} {'P.cost':>8s} {'S.cost':>8s}")
    for _, name, _ in CONFIGS:
        r = res[name]
        print(f"{name:9s} {r['perf_steps']:8.1f} {r['sbfl_steps']:8.1f} "
              f"{r['perf_cost']:8.4f} {r['sbfl_cost']:8.4f}")


def effectiveness_latex(d4j, bip):
    # Cells are success RATES; SBFL cells carry a perfect->SBFL drop subscript (pp).
    # Bold = best Overall rate per (dataset, FL). Shading encodes the INSIGHT: each history
    # SBFL cell is shaded by its robustness advantage over \nonhistory (how much LESS it drops
    # than the baseline in that category); darker = larger advantage. \nonhistory is the
    # unshaded reference; history cells no better than baseline stay unshaded. Grayscale-safe.
    names = [name for _, name, _ in CONFIGS]
    DSS = {"d4j": d4j, "bip": bip}
    best = {(ds, fl): max(DSS[ds][n][f"{fl}_overall"] for n in names)
            for ds in DSS for fl in ("perf", "sbfl")}

    def shade(adv, is_base):
        # adv = history drop minus baseline drop (positive = history more robust).
        # Threshold at 1pp so negligible advantages don't add visual noise.
        if is_base or adv <= 1.0:
            return ""
        return f"\\cellcolor{{gray!{min(55, round(3.0 * adv))}}}"

    def _sbfl_ovr(data, name, ds, is_base):
        cfg, base = data[name], data["baseline"]
        val, perf = cfg["sbfl_overall"], cfg["perf_overall"]
        d = val - perf
        s = f"{val:.1f}\\%$_{{{d:+.1f}}}$"
        if abs(val - best[(ds, "sbfl")]) < 0.05:
            s = f"\\textbf{{{s}}}"
        adv = d - (base["sbfl_overall"] - base["perf_overall"])
        return shade(adv, is_base) + s

    def _sbfl_cat(data, name, c, is_base):
        cfg, base = data[name], data["baseline"]
        rate, perf = cfg["sbfl_cat"][c], cfg["perf_cat"][c]
        d = rate - perf
        adv = d - (base["sbfl_cat"][c] - base["perf_cat"][c])
        return shade(adv, is_base) + f"{rate:.0f}\\%$_{{{d:+.0f}}}$"

    dc, bc = d4j["baseline"]["cat_total"], bip["baseline"]["cat_total"]
    cap = (f"Repair effectiveness under realistic SBFL, paired on the FL-runnable bugs per dataset "
           f"({d4j['baseline']['n_total']} Defects4J, {bip['baseline']['n_total']} BugsInPy). Each "
           f"cell is the plausible@1 success rate (\\%) under SBFL; the subscript is the change in "
           f"percentage points from perfect FL. \\textbf{{Bold}} marks the best Overall rate per "
           f"dataset. The \\nonhistory row is the unshaded reference; each history cell is shaded by "
           f"its robustness advantage over \\nonhistory (darker = degrades less than the baseline in "
           f"that category; cells no better than the baseline are unshaded), so history's robustness "
           f"on multi-hunk bugs reads as the darker cells. Category sizes (SL/SH/SFMH/MFMH): "
           f"Defects4J {dc['SL']}/{dc['SH']}/{dc['SFMH']}/{dc['MFMH']}, BugsInPy "
           f"{bc['SL']}/{bc['SH']}/{bc['SFMH']}/{bc['MFMH']}.")
    L = [r"\begin{table*}[t]", r"\centering",
         r"\caption{" + cap + "}",
         r"\label{tab:rq3_repair}",
         r"\begin{tabular}{l ccccc ccccc}", r"\toprule",
         r" & \multicolumn{5}{c}{\textbf{Defects4J}} & \multicolumn{5}{c}{\textbf{BugsInPy}} \\",
         r"\cmidrule(lr){2-6}\cmidrule(lr){7-11}",
         r"Config & Overall & SL & SH & SFMH & MFMH & Overall & SL & SH & SFMH & MFMH \\",
         r"\midrule"]
    for _, name, macro in CONFIGS:
        is_base = name == "baseline"
        L.append(f"{macro} & {_sbfl_ovr(d4j, name, 'd4j', is_base)} & " +
                 " & ".join(_sbfl_cat(d4j, name, c, is_base) for c in CATS) +
                 f" & {_sbfl_ovr(bip, name, 'bip', is_base)} & " +
                 " & ".join(_sbfl_cat(bip, name, c, is_base) for c in CATS) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    return "\n".join(L)


def cost_latex(d4j, bip):
    L = [r"\begin{table*}[t]", r"\centering",
         r"\caption{Median agent steps and inference cost (USD) for successful repairs, under "
         r"perfect FL (RQ1) versus realistic SBFL (RQ2).}",
         r"\label{tab:rq3_cost}",
         r"\begin{tabular}{l cccc cccc}", r"\toprule",
         r" & \multicolumn{4}{c}{\textbf{Defects4J}} & \multicolumn{4}{c}{\textbf{BugsInPy}} \\",
         r"\cmidrule(lr){2-5}\cmidrule(lr){6-9}",
         r" & \multicolumn{2}{c}{Steps} & \multicolumn{2}{c}{Cost} & "
         r"\multicolumn{2}{c}{Steps} & \multicolumn{2}{c}{Cost} \\",
         r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}\cmidrule(lr){8-9}",
         r"Config & Perf. & SBFL & Perf. & SBFL & Perf. & SBFL & Perf. & SBFL \\",
         r"\midrule"]
    for _, name, macro in CONFIGS:
        d, b = d4j[name], bip[name]
        L.append(f"{macro} & {d['perf_steps']:.1f} & {d['sbfl_steps']:.1f} & "
                 f"{d['perf_cost']:.3f} & {d['sbfl_cost']:.3f} & "
                 f"{b['perf_steps']:.1f} & {b['sbfl_steps']:.1f} & "
                 f"{b['perf_cost']:.3f} & {b['sbfl_cost']:.3f} \\\\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    return "\n".join(L)


def _pfmt(p):
    if p != p:
        return "--"
    return "$<$0.001" if p < 0.001 else f"{p:.3f}"


def paired_cost_latex(d4j, bip):
    L = [r"\begin{table*}[t]", r"\centering",
         r"\caption{Paired effect of realistic fault localization on repair cost (FL axis). For "
         r"each configuration, on the bugs solved under \emph{both} perfect and SBFL FL ($N$), we "
         r"report median cost (USD) and agent steps in each regime; the $p$ columns are Wilcoxon "
         r"signed-rank tests (perfect vs SBFL) on the matched pairs. SBFL significantly raises both "
         r"cost and effort for every configuration on both benchmarks (all $p<0.05$).}",
         r"\label{tab:rq3_sbfl_cost}",
         r"\setlength{\tabcolsep}{5pt}",
         r"\begin{tabular}{l l r rr r rr r}", r"\toprule",
         r" & & & \multicolumn{3}{c}{\textbf{Median Cost (USD)}} & "
         r"\multicolumn{3}{c}{\textbf{Median Steps}} \\",
         r"\cmidrule(lr){4-6}\cmidrule(lr){7-9}",
         r"\textbf{Dataset} & \textbf{Config} & \textbf{N} & Perf. & SBFL & $p$ & "
         r"Perf. & SBFL & $p$ \\", r"\midrule"]
    for tag, data in (("Defects4J", d4j), ("BugsInPy", bip)):
        for j, (_, name, macro) in enumerate(CONFIGS):
            r = data[name]
            lead = f"\\multirow{{4}}{{*}}{{{tag}}}" if j == 0 else ""
            L.append(f"{lead} & {macro} & {r['n']} & {r['perf_cost']:.3f} & {r['sbfl_cost']:.3f} & "
                     f"{_pfmt(r['p_cost'])} & {r['perf_steps']:.0f} & {r['sbfl_steps']:.0f} & "
                     f"{_pfmt(r['p_steps'])} \\\\")
        L.append(r"\midrule" if tag == "Defects4J" else r"\bottomrule")
    L += [r"\end{tabular}", r"\end{table*}"]
    return "\n".join(L)


def _sbfl_fixed_sets(dataset):
    """Per-config sets of bug IDs plausibly repaired under realistic SBFL (val 1/5/7/8)."""
    cat_all = _category_map(dataset)
    labels = sorted(b for b in _sbfl_runnable(dataset) if b in cat_all)
    cat_of = {b: cat_all[b] for b in labels}
    fixed = {name: {l for l in labels if _read(f"sbfl/{dataset}", l, val)["fixed"]}
             for val, name, _ in CONFIGS}
    return cat_of, fixed


def ablation_report(dataset):
    """Console: SBFL #Pass, unique-vs-non, and 3-history union per category."""
    cat_of, fixed = _sbfl_fixed_sets(dataset)
    allbugs = set(cat_of)
    print(f"\n=== {dataset.upper()} — SBFL ablation (#Pass, unique-vs-non, union) ===")
    for c in CATS + ["Total"]:
        cb = allbugs if c == "Total" else {b for b in allbugs if cat_of[b] == c}
        cnt = {n: len(fixed[n] & cb) for n in fixed}
        uq = {n: len((fixed[n] & cb) - fixed["baseline"]) for n in ("fn_all", "fn_pair", "fl_diff")}
        union = len(cb & (fixed["fn_all"] | fixed["fn_pair"] | fixed["fl_diff"]))
        print(f"  {c:5} N={len(cb):3} non={cnt['baseline']:3} "
              f"fnall={cnt['fn_all']:3}(+{uq['fn_all']}) fnpair={cnt['fn_pair']:3}(+{uq['fn_pair']}) "
              f"fldiff={cnt['fl_diff']:3}(+{uq['fl_diff']}) union={union}")


def ablation_latex():
    """tab:rq4-ablation: SBFL #Pass per config with unique-vs-non superscripts + Union column."""
    HIST = ["fn_all", "fn_pair", "fl_diff"]
    # Per-config McNemar significance (SBFL) for the current model (TAG); a True
    # cell earns a * beside its +unique superscript, same test/threshold as Union.
    import analyze_ablation_significance as SIG
    sigmap = SIG.sig_lookup("sbfl", TAG)
    L = [r"\begin{table}[t]", r"\centering", r"\small", r"\setlength{\tabcolsep}{6pt}",
         r"\caption{History ablation under \textbf{realistic SBFL} (Defects4J 827, BugsInPy 361 "
         r"FL-runnable bugs). Each cell is \#Pass; the superscript is the number of bugs a "
         r"configuration uniquely repairs that \nonhistory does not. Union $=$ fixed by at least "
         r"one of the three history heuristics (a complementarity diagnostic, not a single-heuristic "
         r"result). \textbf{Bold} $=$ best single configuration per category.}",
         r"\label{tab:rq4-ablation}",
         r"\resizebox{\textwidth}{!}{%",
         r"\begin{tabular}{l l r r r r r r}", r"\toprule",
         r"\textbf{Dataset} & \textbf{Category} & \textbf{Total} & \textbf{\nonhistory} & "
         r"\textbf{\fnall} & \textbf{\fnpair} & \textbf{\fldiff} & \textbf{Union} \\", r"\midrule"]
    for di, (dataset, dlabel) in enumerate([("defects4j", "Defects4J"), ("bugsinpy", "BugsInPy")]):
        cat_of, fixed = _sbfl_fixed_sets(dataset)
        allbugs = set(cat_of)
        for i, c in enumerate(CATS + ["Total"]):
            cb = allbugs if c == "Total" else {b for b in allbugs if cat_of[b] == c}
            counts = {n: len(fixed[n] & cb) for n in fixed}
            best = max(counts.values())

            def cell(n):
                v = counts[n]
                s = f"\\textbf{{{v}}}" if v == best else str(v)
                if n in HIST:
                    uniq = len((fixed[n] & cb) - fixed['baseline'])
                    star = "\\,*" if sigmap.get((dataset, c, n), False) else ""
                    s += f"\\,$^{{\\smash{{+{uniq}{star}}}}}$"
                return s
            union = len(cb & (fixed["fn_all"] | fixed["fn_pair"] | fixed["fl_diff"]))
            ustar = "\\,$^{*}$" if sigmap.get((dataset, c, "union"), False) else ""
            head = f"\\multirow{{5}}{{*}}{{{dlabel}}}" if i == 0 else ""
            name = r"\textbf{Total}" if c == "Total" else c
            if c == "Total":
                L.append(r"\cmidrule(lr){2-8}")
            L.append(f" {head} & {name} & {len(cb)} & {cell('baseline')} & {cell('fn_all')} "
                     f"& {cell('fn_pair')} & {cell('fl_diff')} & {union}{ustar} \\\\")
        L.append(r"\midrule" if di == 0 else r"\bottomrule")
    L += [r"\end{tabular}}", r"\end{table}"]
    return "\n".join(L)


def union_cost(dataset, regime):
    """Early-stop union cost of the three history heuristics, one FL regime.

    regime 'perfect' reads the full bug set from results/<dataset> (reproduces
    the RQ2 union, 684 D4J / 463 BugsInPy); regime 'sbfl' reads the FL-runnable
    set from results/sbfl/<dataset> (matches tab:rq4-ablation). The portfolio
    runs fn_all, fn_pair, fl_diff in order and stops at the first test-passing
    patch; per-bug cost/steps sum over the runs executed, and a never-fixed bug
    pays for all three. Effectiveness is order-independent (we stop only on a
    pass), so which bugs end fixed reproduces the union; only cost is
    order-dependent. Paired Wilcoxon vs \\nonhistory on per-bug cost and steps
    is the cost counterpart of the union-vs-non-history effectiveness McNemar.
    """
    from scipy.stats import wilcoxon
    cat_all = _category_map(dataset)
    if regime == "perfect":
        tree, labels = dataset, sorted(cat_all)
    else:
        tree = f"sbfl/{dataset}"
        labels = sorted(b for b in _sbfl_runnable(dataset) if b in cat_all)
    order = ["fn_all", "fn_pair", "fl_diff"]
    val_of = {n: v for v, n, _ in CONFIGS}
    reads = {n: {l: _read(tree, l, val_of[n]) for l in labels} for n in ["baseline"] + order}

    es_cost, es_steps, runs, non_cost, non_steps = [], [], [], [], []
    ceil_cost, ceil_steps, fld_cost, fld_steps = [], [], [], []
    n_fixed = fld_fixed = 0
    stop = {1: 0, 2: 0, 3: 0}
    for l in labels:
        if any(reads[n][l]["cost"] is None for n in ["baseline"] + order):
            continue  # portfolio needs all four runs present
        c = s = 0.0
        r = 0
        fx = False
        for name in order:
            d = reads[name][l]
            c += d["cost"]; s += d["steps"]; r += 1
            if d["fixed"]:
                fx = True
                break
        es_cost.append(c); es_steps.append(s); runs.append(r); stop[r] += 1
        n_fixed += fx
        ceil_cost.append(sum(reads[n][l]["cost"] for n in order))
        ceil_steps.append(sum(reads[n][l]["steps"] for n in order))
        non_cost.append(reads["baseline"][l]["cost"]); non_steps.append(reads["baseline"][l]["steps"])
        fld_cost.append(reads["fl_diff"][l]["cost"]); fld_steps.append(reads["fl_diff"][l]["steps"])
        fld_fixed += reads["fl_diff"][l]["fixed"]

    def mean(xs):
        return sum(xs) / len(xs) if xs else float("nan")

    def wp(a, b):
        try:
            return wilcoxon(a, b).pvalue
        except Exception:
            return float("nan")

    return {
        "regime": regime, "n": len(es_cost), "n_fixed": n_fixed, "fld_fixed": fld_fixed,
        "runs": mean(runs), "stop": stop,
        "es_cost": mean(es_cost), "es_steps": mean(es_steps),
        "ceil_cost": mean(ceil_cost), "ceil_steps": mean(ceil_steps),
        "non_cost": mean(non_cost), "non_steps": mean(non_steps),
        "fld_cost": mean(fld_cost), "fld_steps": mean(fld_steps),
        "p_cost": wp(es_cost, non_cost), "p_steps": wp(es_steps, non_steps),
    }


def union_cost_report(dataset, regime, r):
    print(f"\n[{dataset.upper()}] union cost, {regime} FL "
          f"(order fn_all->fn_pair->fl_diff, N={r['n']}):")
    print(f"   union-fixed={r['n_fixed']}  fl_diff-only-fixed={r['fld_fixed']}  "
          f"runs/bug={r['runs']:.2f}  stop@1/2/3={r['stop'][1]}/{r['stop'][2]}/{r['stop'][3]}")
    print(f"   early-stop  cost/bug ${r['es_cost']:.4f}  steps/bug {r['es_steps']:.1f}")
    print(f"   run-all-3   cost/bug ${r['ceil_cost']:.4f}  steps/bug {r['ceil_steps']:.1f}")
    ratio = f"{r['es_cost']/r['fld_cost']:.2f}x" if r['fld_cost'] else "n/a (no cost data)"
    print(f"   fl_diff     cost/bug ${r['fld_cost']:.4f}  steps/bug {r['fld_steps']:.1f}  "
          f"(early-stop = {ratio})")
    print(f"   non-history cost/bug ${r['non_cost']:.4f}  steps/bug {r['non_steps']:.1f}")
    print(f"   Wilcoxon early-stop-union vs non-history: cost p={r['p_cost']:.2e}, steps p={r['p_steps']:.2e}")


def union_cost_latex(stats):
    """tab:rq4-union-cost: cost of the complementarity strategy, perfect + SBFL.

    stats: {regime: {"defects4j": r, "bugsinpy": r}} from union_cost().
    """
    L = [r"\begin{table}[t]", r"\centering", r"\small", r"\setlength{\tabcolsep}{5pt}",
         r"\caption{Cost of the complementarity union under perfect and realistic (SBFL) fault "
         r"localization. The union runs \fnall, \fnpair, \fldiff and keeps any test-passing patch; "
         r"\emph{early stopping} halts at the first pass in that fixed order. \emph{Fixed} counts "
         r"repaired bugs, \emph{Runs} the mean configurations executed per bug, and cost and steps "
         r"are per-bug means over all $N$ bugs (including failed attempts), not the "
         r"median-of-success basis of Table~\ref{tab:rq2b-cost-crosslang}; \fldiff\ alone is the "
         r"reference. SBFL $N$ is the FL-runnable subset whose four runs all recorded cost. The "
         r"union costs significantly more than one \nonhistory run (paired Wilcoxon on per-bug "
         r"cost, all $p<0.01$).}",
         r"\label{tab:rq4-union-cost}",
         r"\begin{tabular}{l l l r r r r}", r"\toprule",
         r"\textbf{FL} & \textbf{Dataset ($N$)} & \textbf{Policy} & \textbf{Fixed} & \textbf{Runs} "
         r"& \textbf{Cost (\$)} & \textbf{Steps} \\", r"\midrule"]
    regimes = [("perfect", "Perfect"), ("sbfl", "SBFL")]
    for ri, (rk, rlabel) in enumerate(regimes):
        for di, ds in enumerate(["defects4j", "bugsinpy"]):
            r = stats[rk][ds]
            dlabel = "Defects4J" if ds == "defects4j" else "BugsInPy"
            dcell = r"\shortstack[l]{" + dlabel + r"\\ \scriptsize $N{=}" + str(r["n"]) + r"$}"
            fl_col = r"\multirow{6}{*}{" + rlabel + "}" if di == 0 else ""
            rows = [
                (r"\fldiff\ only", r["fld_fixed"], 1.0, r["fld_cost"], r["fld_steps"]),
                (r"Union, early-stop", r["n_fixed"], r["runs"], r["es_cost"], r["es_steps"]),
                (r"Union, run-all-3", r["n_fixed"], 3.0, r["ceil_cost"], r["ceil_steps"]),
            ]
            for j, (pol, fx, runs, cost, steps) in enumerate(rows):
                lead = fl_col if j == 0 else ""
                ds_col = r"\multirow{3}{*}{" + dcell + "}" if j == 0 else ""
                L.append(f"{lead} & {ds_col} & {pol} & {fx} & {runs:.2f} & {cost:.3f} & {steps:.1f} \\\\")
            if di == 0:
                L.append(r"\cmidrule(lr){2-7}")
        L.append(r"\midrule" if ri == 0 else r"\bottomrule")
    L += [r"\end{tabular}", r"\end{table}"]
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser(description="RQ3 repair: SBFL vs perfect FL + cost/steps")
    ap.add_argument("--latex", action="store_true", help="emit LaTeX for tab:rq3_repair + tab:rq3_cost")
    ap.add_argument("--tag", default="", help="model result subdir suffix (e.g. qwen3coder, devstral); default = DeepSeek")
    args = ap.parse_args()
    global TAG
    TAG = f"_{args.tag}" if args.tag else ""
    d4j, bip = analyze("defects4j"), analyze("bugsinpy")
    report("defects4j", d4j)
    report("bugsinpy", bip)
    pc_d4j, pc_bip = paired_cost("defects4j"), paired_cost("bugsinpy")
    paired_report("defects4j", pc_d4j)
    paired_report("bugsinpy", pc_bip)
    for ds in ("defects4j", "bugsinpy"):
        sbfl_resilience_report(ds, sbfl_resilience_tests(ds))
    for ds in ("defects4j", "bugsinpy"):
        sbfl_effectiveness_report(ds, sbfl_effectiveness_tests(ds))
    for ds in ("defects4j", "bugsinpy"):
        ablation_report(ds)
    union_stats = {rk: {ds: union_cost(ds, rk) for ds in ("defects4j", "bugsinpy")}
                   for rk in ("perfect", "sbfl")}
    for rk in ("perfect", "sbfl"):
        for ds in ("defects4j", "bugsinpy"):
            union_cost_report(ds, rk, union_stats[rk][ds])
    for ds in ("defects4j", "bugsinpy"):
        fr = sbfl_config_friedman(ds)
        o = fr["overall"]
        print(f"\n[{ds.upper()}] config-axis cost Friedman UNDER SBFL: overall p={o['p']:.4f} "
              f"(N={o['n']} matched-across-configs)")
        med = o["med"]
        print("   overall median SBFL cost:", {n: round(med[n], 4) for n in med})
        for c in CATS:
            fc = fr["cats"][c]
            sig = "  <-- sig" if fc["p"] == fc["p"] and fc["p"] < 0.05 else ""
            line = f"   {c} p={fc['p']:.3f} (N={fc['n']})"
            if fc["med"]:
                line += "  med=" + str({n: round(fc['med'][n], 4) for n in fc['med']})
            if fc["pair"] and fc["p"] == fc["p"] and fc["p"] < 0.05:
                line += "  pair_vs_non=" + str({n: round(fc['pair'][n], 4) for n in fc['pair']})
            print(line + sig)
    if args.latex:
        print("\n% ===== tab:rq3_repair =====\n" + effectiveness_latex(d4j, bip))
        print("\n% ===== tab:rq3_cost =====\n" + cost_latex(d4j, bip))
        print("\n% ===== tab:rq3_sbfl_cost (paired perfect-vs-SBFL) =====\n" + paired_cost_latex(pc_d4j, pc_bip))
        print("\n% ===== tab:rq4-ablation (SBFL unique-pass + union) =====\n" + ablation_latex())
        print("\n% ===== tab:rq4-union-cost (early-stop union cost, perfect + SBFL) =====\n" + union_cost_latex(union_stats))


if __name__ == "__main__":
    main()
