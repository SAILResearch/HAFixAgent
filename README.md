# HAFixAgent

[![Paper](https://img.shields.io/badge/Paper-arXiv:2511.01047-red)](https://arxiv.org/pdf/2511.01047)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

## 📖 Project Overview
HAFixAgent is an automated program repair (APR) agent that augments an LLM repair loop with
history-aware blame context. Given a localized fault, it uses `git blame` to retrieve blame
commits, extract historical context, and injects them (one of three historical heuristics) into
the repair prompt. This repository is the replication package of HAFixAgent.

It currently supports:
- **Two benchmarks**: Defects4J (854 Java bugs) and BugsInPy (501 Python bugs).
- **Four configurations**: `non-history` (control), `fn_all`, `fn_pair`, `fl_diff` (three history representations).
- **Fault localization**: perfect FL (from the developer patch) and realistic SBFL (GZoltar for Java, FauxPy for Python).
- **Models**: DeepSeek-V3.2-Exp (primary, via OpenRouter), Qwen3-Coder and Devstral-Small-2 (locally deployed) and any OpenAI-compatible LLMs.

### Authors
- Yu Shi, Hao Li, Bram Adams, Ahmed E. Hassan
- [Lab on Maintenance, Construction and Intelligence of Software (MCIS)](https://mcis.cs.queensu.ca)
- [Software Analysis and Intelligence Lab (SAIL)](https://sail.cs.queensu.ca)
- School of Computing, Queen's University, Canada

## 🏗️ Repository Structure
```
HAFixAgent/
├── hafix_agent/                    # Core agent (dataset-agnostic)
│   ├── agents/hafix_agent.py       # Main agent, extends mini-swe-agent
│   ├── blame/                      # Blame extraction: interfaces, core, selection, context loader
│   ├── environments/               # Docker container management
│   ├── prompts/prompt_builder.py   # History-aware prompt construction
│   └── utils/                      # Logging, token tracking, model specs
├── dataset/
│   ├── defects4j/                  # Defects4J extractor, analysis, mined bug_description/
│   └── bugsinpy/                   # BugsInPy extractor, analysis, bug discovery
├── config/
│   ├── defects4j.yaml              # Perfect-FL config (Java)
│   ├── defects4j_sbfl.yaml         # Realistic-SBFL config (Java)
│   ├── bugsinpy.yaml               # Perfect-FL config (Python)
│   ├── bugsinpy_sbfl.yaml          # Realistic-SBFL config (Python)
│   └── models/                     # DeepSeek / Qwen3-Coder / Devstral model specs
├── evaluation/
│   ├── run_defects4j_evaluation.py # Defects4J repair runner (console entry: hafixagent)
│   ├── run_bugsinpy_evaluation.py  # BugsInPy repair runner
│   ├── run_sbfl_evaluation.py      # SBFL repair runner (reads cached rankings)
│   ├── run_repairagent_parallel.py # RepairAgent baseline driver
│   ├── run_birch_parallel.py       # BIRCH-feedback baseline driver
│   ├── cache_context.py            # Pre-extract blame contexts
│   └── sbfl/                        # Self-contained FL module: GZoltar jars + FauxPy layer
├── analysis/                       # One script per paper RQ table/figure (see below)
├── results/                        # Curated result subset backing the paper
├── Dockerfile.defects4j-enhanced   # Defects4J image with extra bash tools
├── Dockerfile.bugsinpy-enhanced    # BugsInPy image with extra bash tools
└── pyproject.toml
```

## 🚀 Environment Setup
```bash
git clone https://github.com/SAILResearch/HAFixAgent.git
cd HAFixAgent
conda create -n hafixagent python=3.11
conda activate hafixagent
pip install -e .
# Optional: SBFL fault-localization stage only (FauxPy for BugsInPy; GZoltar jars are bundled)
pip install -e ".[sbfl]"
```

- **LLM API.** We extend [mini-swe-agent](https://mini-swe-agent.com/latest/) and call models through
  OpenRouter. The model is set in each config's `model` section (see `config/models/`). Set your key:
  ```bash
  mini-extra config set OPENROUTER_API_KEY <OPENROUTER_API_KEY>
  ```
  Any [LiteLLM](https://docs.litellm.ai/docs/providers)-supported provider works by editing the model config.

- **Defects4J Docker image.**
  ```bash
  mkdir -p vendor && cd vendor
  git clone https://github.com/rjust/defects4j.git
  cd defects4j && docker build -f Dockerfile -t defects4j:base .
  cd ../../ && docker build -f Dockerfile.defects4j-enhanced -t defects4j:latest .
  ```

- **BugsInPy Docker image.** Build the base `bugsinpy_image:clean` from the
  [BugsInPy](https://github.com/soarsmu/BugsInPy) projects, then enhance it:
  ```bash
  docker build -f Dockerfile.bugsinpy-enhanced -t bugsinpy_image:clean .
  ```

## 🔍 RQ0: Blame Availability
How often does buggy code have blame history, and how concentrated is it?
```bash
python analysis/analyze_rq0_blame_availability.py
```

## 🐳 RQ1: Effectiveness, Complementarity, and Cross-Model Generalization
Run the four configurations under perfect FL on Defects4J:
```bash
# hafixagent is the console entry for python evaluation/run_defects4j_evaluation.py
# --bug-category: single_line | single_hunk | single_file_multi_hunk | multi_file_multi_hunk | all
hafixagent --bug-category all --history baseline --selector-type llm_judge --workers 4
python evaluation/run_defects4j_evaluation.py --bug-category all --history fn_all  --selector-type llm_judge --workers 4
python evaluation/run_defects4j_evaluation.py --bug-category all --history fn_pair --selector-type llm_judge --workers 4
python evaluation/run_defects4j_evaluation.py --bug-category all --history fl_diff --selector-type llm_judge --workers 4

# Scope to a single category instead of all:
python evaluation/run_defects4j_evaluation.py --bug-category single_file_multi_hunk --history fl_diff --selector-type llm_judge --workers 4

# BugsInPy (same four configs)
python evaluation/run_bugsinpy_evaluation.py --bug-category all --history fl_diff --workers 4

# Cross-model: pass a model config to any runner
hafixagent --bug-category all --history fn_pair --model-config config/models/qwen3_coder_next.yaml --workers 4
```
Analysis (ablation tables, Venn complementarity, external baselines, significance):
```bash
python analysis/analyze_rq1_effectiveness.py                       # history ablation + complementarity
python analysis/analyze_rq1_external_baselines.py -b repairagent   # choices: repairagent, hunk4j
python analysis/analyze_local_model_ablation.py --tag qwen3coder   # cross-model ablation (--tag devstral for Devstral)
python analysis/analyze_ablation_significance.py --regime perfect  # per-config and Union McNemar
```

## 📊 RQ2: Realistic Fault Localization (SBFL)
```bash
# 1. (optional) regenerate the FL cache: GZoltar (Java) / FauxPy (Python)
python evaluation/sbfl/run_fl_defects4j.py --all --workers 8
python evaluation/sbfl/run_fl_bugsinpy.py  --all --workers 4

# 2. repair with SBFL-ranked top-10 locations (repeat --history for baseline, fn_all, fn_pair, fl_diff)
python evaluation/run_sbfl_evaluation.py --dataset defects4j --history fl_diff --all --workers 4

# 3. FL accuracy (top-N hit rate) and repair outcomes
python analysis/analyze_rq2_sbfl_fl_accuracy.py --dataset defects4j
python analysis/analyze_rq2_sbfl_repair.py --latex
```

## 💰 RQ3: Cost and Efficiency
```bash
python analysis/analyze_rq3_cost_efficiency.py
```

## 📚 Citation
If you found this work helpful, please kindly consider citing it using the following:

<details>
<summary>HAFixAgent</summary>

```bibtex
@article{shi2025hafixagent,
  title={HAFixAgent: History-aware program repair agent},
  author={Shi, Yu and Li, Hao and Adams, Bram and Hassan, Ahmed E},
  journal={arXiv preprint arXiv:2511.01047},
  year={2025}
}
```

</details>

<details>
<summary>HAFix (Prior Foundation Work)</summary>

```bibtex
@article{shi2025hafix,
  title={HAFix: History-Augmented Large Language Models for Bug Fixing},
  author={Shi, Yu and Bangash, Abdul Ali and Fallahzadeh, Emad and Adams, Bram and Hassan, Ahmed E},
  journal={arXiv preprint arXiv:2501.09135},
  year={2025}
}
```

</details>

## 📧 Contact
For questions or issues, please open a [GitHub Issue](https://github.com/SAILResearch/HAFixAgent/issues).

## 🙏 Acknowledgement
- [Defects4J](https://github.com/rjust/defects4j)
- [BugsInPy](https://github.com/soarsmu/bugsinpy)
- [GZoltar](https://gzoltar.com/)
- [FauxPy](https://github.com/atom-sw/fauxpy)
- [mini-swe-agent](https://mini-swe-agent.com/latest/)
- [RepairAgent](https://github.com/sola-st/RepairAgent)
- [BIRCH](https://arxiv.org/pdf/2506.04418)
