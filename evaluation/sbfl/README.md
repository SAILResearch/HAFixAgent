# SBFL — Realistic Fault Localization for RQ3

This module provides **real** spectrum-based fault localization (SBFL) so HARPER can be
evaluated without assuming perfect knowledge of the buggy lines. RQ1/RQ2 take fault
locations from the developer patch (perfect FL); RQ3 replaces that with a real SBFL
ranking, directly answering the "perfect FL is unrealistic" reviewer concern.

**One engine per language, both producing an Ochiai ranking:**

| Language | Benchmark | Engine | Output |
|---|---|---|---|
| Java | Defects4J | **GZoltar 1.7.3** (bundled jars) | `ochiai.ranking.csv` |
| Python | BugsInPy | **FauxPy** (its SBFL ranking code) + our collection layer | `Scores_Ochiai.csv` + `spectrum.json` |

The Java side bundles the official GZoltar jars; the Python side uses **FauxPy's actual
ranking engine** (`fauxpy.fault_localization.sbfl`) with a thin integration layer that
replaces only FauxPy's fragile coverage *collection* (see below). No runtime dependency on
`vendor/ITER` (its cached rankings are used only as parser test fixtures).

## How SBFL works

When a bug has failing and passing tests, the faulty line is usually one that
**failing tests execute but passing tests mostly avoid**. For each source line `e`, over
the whole test run: `ef`/`ep` = #failing/#passing tests that execute `e`, `nf` = #failing
that do not. The **Ochiai** score (the de-facto SBFL standard; GZoltar, FauxPy, ITER all
use it) is:

```
suspiciousness(e) = ef / ( sqrt( (ef + nf) * (ef + ep) ) + epsilon )
```

FauxPy carries `epsilon = 0.1` as a zero/tie stabilizer (GZoltar omits it). We use FauxPy's
own `MetricOchiai` for the Python ranking, so the BugsInPy scores carry this epsilon.
Sorting lines by this score yields a ranked suspicious-line list. It uses only tests +
code — it **never sees the developer's fix**, which is what makes it realistic FL.

## Java / Defects4J — GZoltar

GZoltar runs three steps inside the `defects4j:latest` container: (1) instrument & run
each relevant test in isolation → `gzoltar.ser` coverage matrix; (2) build the per-line
spectrum; (3) Ochiai report → `ochiai.ranking.csv`. Defects4J supplies compiled classes,
tests, and classpath via `defects4j compile` + `defects4j export`, so each bug rebuilds
its exact environment deterministically.

## Python / BugsInPy — FauxPy engine + collection integration

FauxPy is the SBFL tool of record for BugsInPy (Rezaalipour & Furia, EMSE; arXiv
2305.19834), but it ships as a **pytest plugin** that must be installed *into* each bug's
fragile venv, where it hits pytest-version conflicts and import-timing failures and
silently emits empty rankings. The failure is in **collection, not computation** — FauxPy's
ranking code is standalone. So we **keep FauxPy's ranking engine and replace only its
collection stage**:

```
┌─ bug container (messy) ──────────────┐      ┌─ host (clean, versioned) ──────────┐
│ provision declared deps (relax        │ spec │ compute_ochiai.py (integration):  │
│   yanked pre-release pins)            │ trum │   spectrum → FauxPy SbflDbManager  │
│ coverage run -m pytest <test module>  │ .json│   → FauxPy RankingMetricManager    │
│   + _sbfl_context_plugin (per-test    │ ───► │     (MetricOchiai, epsilon=0.1)    │
│   context=nodeid, record pass/fail)   │      │   → Scores_Ochiai.csv + spectrum   │
│ coverage json --show-contexts         │      │ FauxPy does all counting + scoring │
└───────────────────────────────────────┘      └────────────────────────────────────┘
```

The **only** dependency added to the bug env is `coverage` (pure-Python, supports 3.6+);
`coverage run` starts measurement *before* any import (pytest-cov's `--cov` starts late and
records nothing for already-imported packages). On the host, `compute_ochiai.py` feeds the
collected spectrum into FauxPy's **own** `SbflDbManager` + `RankingMetricManager`, which
count `ef/ep/nf/np` and score every line with FauxPy's `MetricOchiai`. We supply only the
spectrum and read back FauxPy's `Ochiai` column — the suspiciousness values are computed by
FauxPy's actual code, not a re-implementation (host needs `pip install fauxpy==0.7.0`; the
only perf tweak is disabling SQLite fsync on the throwaway temp DB — identical results).

The portable `spectrum.json` (per-line `ef/ep/nf/np` + FauxPy's Ochiai + total failing) is
the **durable replication artifact**: rankings recompute from it with no Docker and no
re-running of the bug env. The resulting `Scores_Ochiai.csv` matches FauxPy's CLI format, so
everything downstream is identical across languages.

## From raw ranking to FL input (`parse_ranking.py`)

`parse_gzoltar_ranking` translates GZoltar entity names to `(file, line)`:
```
com.google.javascript.jscomp$Compiler#<clinit>():70;1.0  ->  src/.../Compiler.java : 70
```
`parse_fauxpy_ranking` parses the `Entity,Score` CSV (`<path>::<line>,<score>`) FauxPy
emits. Both dedup `(file, line)`, preserve rank order, and take the **top-10** (top-1/5 as
sensitivity points), per the standard top-k FL convention.

## Two stages

**Stage 1 — FL (run once per bug, cached):**
```
# Defects4J
run_fl_defects4j.py  -> docker defects4j:latest -> fl_one_bug.sh (checkout+compile+GZoltar)
                     -> results/sbfl/defects4j/fl_cache/<Project_Bug>/{ochiai.ranking.csv, meta.json}
# BugsInPy
run_fl_bugsinpy.py -> docker bugsinpy:<p>_<b> -> collect_spectrum.sh (coverage)
                     -> compute_ochiai.py (host)
                     -> results/sbfl/bugsinpy/fl_cache/<Project_Bug>/{Scores_Ochiai.csv, spectrum.json, meta.json}
```

**Stage 2 — repair (per config, reuses cached ranking):**
```
run_sbfl_evaluation.py --dataset {defects4j,bugsinpy}:
   parse_ranking.py (top-10) -> suspicious (file:line) list
   -> override fault_locations (replaces perfect FL)
        ├─ agent prompt (all 4 configs): "suspicious locations: file:line ..."
        └─ history configs only: candidate lines -> LLM judge -> git blame -> history
   -> run repair agent   (config/<dataset>_sbfl.yaml)
```

FL accuracy (whether a developer-patched line appears in the top-N) is computed offline by
`analysis/analyze_rq2_sbfl_fl_accuracy.py --dataset {defects4j,bugsinpy}`.

## Replication

### Stage 2 — repair experiment (RQ3)

`run_sbfl_evaluation.py` reads the cached top-10 ranking per bug, overrides `fault_locations`
with it, and runs the existing HARPER pipeline (LLM-judge → git blame → history → agent). Run
**one config at a time**; `--history` takes `baseline` (no history) or `fn_all` / `fn_pair` /
`fl_diff` (the three history heuristics). Prereq: `OPENROUTER_API_KEY` in `.env` (DeepSeek-V3.2-Exp).

**Step 0 — build the stratified sample (one-time, no API).** RQ3 repair runs on a
stratified, SBFL-runnable sample (200/dataset, 50/category, seed=42, Chart excluded) — same
sampler as RQ2a plus a "has a cached ranking" filter, so every sampled bug is runnable:
```
python evaluation/sbfl/create_rq3_sample.py        # -> evaluation/sbfl/rq3_repair_sample.json
```

**Run one config at a time** on that sample (`--workers 16`; the wrapper retries transient
Docker-creation timeouts, so the burst that hit the full alphabetical run is handled):
```
python evaluation/run_sbfl_evaluation.py --dataset defects4j \
    --sample evaluation/sbfl/rq3_repair_sample.json --history baseline --workers 16 --resume
# one bug (debug):    --bugs Cli_1        wiring only (no API): add --dry-run --history baseline
# --all instead of --sample runs every bug with a ranking (827 D4J / 361 BugsInPy; ~$850, not sampled)
```

Complete sampled experiment — run these 8 one by one (4 configs × 2 datasets, ~$290 total):
```
S=evaluation/sbfl/rq3_repair_sample.json
python evaluation/run_sbfl_evaluation.py --dataset defects4j --sample $S --history baseline --workers 16 --resume
python evaluation/run_sbfl_evaluation.py --dataset defects4j --sample $S --history fn_all   --workers 16 --resume
python evaluation/run_sbfl_evaluation.py --dataset defects4j --sample $S --history fn_pair  --workers 16 --resume
python evaluation/run_sbfl_evaluation.py --dataset defects4j --sample $S --history fl_diff  --workers 16 --resume
python evaluation/run_sbfl_evaluation.py --dataset bugsinpy  --sample $S --history baseline --workers 16 --resume
python evaluation/run_sbfl_evaluation.py --dataset bugsinpy  --sample $S --history fn_all   --workers 16 --resume
python evaluation/run_sbfl_evaluation.py --dataset bugsinpy  --sample $S --history fn_pair  --workers 16 --resume
python evaluation/run_sbfl_evaluation.py --dataset bugsinpy  --sample $S --history fl_diff  --workers 16 --resume
```

Output mirrors the RQ1 layout so the same analysis code works on both
(`history_category.value`: baseline=1, fn_all=5, fn_pair=7, fl_diff=8):
```
results/sbfl/<dataset>/llm_judge_1line/<category>/<bug>/<bug>_<value>_result.json   (+ .log)
results/sbfl/<dataset>/llm_judge_1line/<category>/trajectories/<bug>_<value>.traj.json
results/sbfl/<dataset>/llm_judge_1line/progress_<value>_sbfl.json    # live aggregate stats
```
Each `_result.json` carries `repair_result` with `success` (exit==Submitted), `exit_status`,
`model_cost`, `agent_steps`, `token_usage`, and the `patch` — same fields as RQ1.

Monitor a run (per-bug status with the N/total counter):
```
grep -a "(baseline):" <logfile> | tail -20        # or the config you launched
```

### Stage 1 — regenerate the FL rankings from scratch

Run one driver per dataset. Each
reads its bug list from `dataset/<ds>/<ds>_blame_feasibility.csv`, launches a disposable
container per bug, and writes the cached ranking. Both are parallel (`--workers`, default 4)
and resumable (a bug with an existing valid ranking is skipped).

**Defects4J — GZoltar (deterministic):**
```
# Prereq: docker image `defects4j:latest`. GZoltar jars are bundled (evaluation/sbfl/java/lib).
python evaluation/sbfl/run_fl_defects4j.py --all            # all 854 bugs in the feasibility CSV
python evaluation/sbfl/run_fl_defects4j.py --bugs Cli_1     # one bug (or --project Lang)
#   options: --exclude-chart  --scope relevant|modified(default relevant)  --workers N
# -> results/sbfl/defects4j/fl_cache/<Project_Bug>/{ochiai.ranking.csv, meta.json, gzoltar.log}
```

**BugsInPy — FauxPy engine over a coverage spectrum (fragile env):**
```
# Prereq: host `pip install fauxpy==0.7.0`; one prewarmed docker image `bugsinpy:<project>_<bug>`
#         per bug (built by the BugsInPy provisioning pipeline). `coverage` is installed into
#         the container by collect_spectrum.sh; a bug with no prewarmed image is skipped.
python evaluation/sbfl/run_fl_bugsinpy.py --all            # all bugs in the feasibility CSV
python evaluation/sbfl/run_fl_bugsinpy.py --bugs scrapy_10 # one bug (or --project keras)
#   options: --workers N  --timeout S
# -> results/sbfl/bugsinpy/fl_cache/<Project_Bug>/{Scores_Ochiai.csv, spectrum.json, meta.json,
#                                         coverage.json, outcomes.tsv, pytest.log}
```

Defects4J is fully deterministic. BugsInPy depends on the prewarmed `bugsinpy:*` images and
is inherently fragile (the 140 non-reproduced bugs are documented in RESULTS.md). Because the
spectrum is cached, `compute_ochiai.py` can **re-run FauxPy's ranking from the saved
`coverage.json`/`spectrum.json` with no container** — useful if you only need to re-rank.

Then run the offline FL-accuracy report (Top-N + E_inspect) on the cached rankings:
```
python analysis/analyze_rq2_sbfl_fl_accuracy.py --dataset defects4j
python analysis/analyze_rq2_sbfl_fl_accuracy.py --dataset bugsinpy --ranking-dir results/sbfl/bugsinpy/fl_cache
```

## Files

```
config/{defects4j,bugsinpy}_sbfl.yaml   realistic-FL prompts (category-agnostic, Decision C)
evaluation/run_sbfl_evaluation.py       Stage 2: dataset-agnostic repair from cached rankings
evaluation/sbfl/
├── java/lib/*.jar          official GZoltar v1.7.3 jars (provenance below)
├── java/run_gzoltar.sh     GZoltar 3-step runner
├── java/fl_one_bug.sh      checkout+compile+GZoltar+meta (in-container)
├── run_fl_defects4j.py     Stage 1 driver (Java) — parallel, resumable
├── python/collect_spectrum.sh    Stage 1 collector (Python) — coverage spectrum in-container
├── python/_sbfl_context_plugin.py per-test context + outcome pytest plugin
├── python/resolve_test_target.py  run_test.sh -> pytest module (handles unittest specs)
├── compute_ochiai.py       spectrum -> FauxPy SbflDbManager/RankingMetricManager -> ranking + spectrum.json
├── run_fl_bugsinpy.py  Stage 1 driver (Python) — parallel, resumable
├── parse_ranking.py        GZoltar / Ochiai-CSV ranking -> top-N (file, line)
├── README.md / RESULTS.md  this file / coverage numbers + exclusions
test/{test_compute_ochiai,test_resolve_test_target,test_sbfl_parse_ranking}.py
```

Quick start:
```
# Defects4J:  python evaluation/sbfl/run_fl_defects4j.py --bugs Cli_1
# BugsInPy:   python evaluation/sbfl/run_fl_bugsinpy.py --bugs scrapy_10
# Repair (dry-run, no API): python evaluation/run_sbfl_evaluation.py --dataset bugsinpy --history baseline --bugs scrapy_10 --dry-run
# Tests:      python -m pytest test/test_compute_ochiai.py test/test_resolve_test_target.py test/test_sbfl_parse_ranking.py
```

## GZoltar jars — provenance

`java/lib/` holds the **official GZoltar v1.7.3 release** jars, bundled so replication
needs no source build or network access.
- Release: https://github.com/GZoltar/gzoltar/releases/tag/v1.7.3
  (`gzoltar-1.7.3.202203230348.zip` → `{gzoltarcli,gzoltaragent,gzoltarant}.jar`)
```
sha256:
  3e2ff55576801b989cc98a86c79ad0e2a682c96ea28f45bdd9aef4942dd4c6b1  gzoltarcli.jar
  3cd6ee0147269db6c706debfad79d677d8f7fb8ca81e6008ed636777568f0ef7  gzoltaragent.jar
  72ee3c083facbe06a5ae6dfba4e7e809a56a43c2b181e8ae430ca569ff420f6f  gzoltarant.jar
```
Byte-identical to the official release (the copies in `vendor/ITER` are the same jars).
