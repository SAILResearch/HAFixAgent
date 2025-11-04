# HAFixAgent

[![Paper](https://img.shields.io/badge/Paper-arXiv:2501.09135-red)](https://arxiv.org/abs/2501.09135)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

## 📖 Project Overview
HAFixAgent is an automated program repair agent with history-aware blame context extraction. It currently supports Defects4J through a dataset-agnostic architecture.
### Authors
- Yu Shi, Hao Li, Bram Adams, Ahmed E. Hassan
- [Lab on Maintenance, Construction and Intelligence of Software (MCIS)](https://mcis.cs.queensu.ca)
- [Software Analysis and Intelligence Lab (SAIL)](https://sail.cs.queensu.ca)
- School of Computing, Queen's University, Canada

## 🏗️ Repository Structure
```
HAFixAgent/
├── 📁 hafix_agent/                    # Core HAFixAgent components, dataset-agnostic design
│   ├── 📁 agents/
│   │   └── 📄 hafix_agent.py          # Main HAFixAgent implementation extending mini-swe-agent
│   ├── 📁 blame/                      # Blame extraction architecture
│   │   ├── 📄 __init__.py
│   │   ├── 📄 interface.py            # BlameExtractor & BugInfoExtractor interfaces
│   │   ├── 📄 context_loader.py       # Context loading factory with runtime and cached implementations
│   │   ├── 📄 core.py                 # Core blame extraction of different history heuristic with container integration
│   │   ├── 📄 patch_parser.py         # Patch file parsing utilities (dataset-agnostic)
│   │   ├── 📄 selection.py            # Blame line selection strategies
│   │   └── 📄 extraction_config.py    # Path utilities and configuration for cache structure
│   ├── 📁 environments/
│   │   └── 📄 defects4j_docker.py     # Docker container management
│   ├── 📁 prompts/
│   │   └── 📄 prompt_builder.py       # History-aware prompt construction, prepare placeholders for rendering
│   └── 📁 utils/                      # Common utilities for evaluation and framework operation
│       ├── 📄 __init__.py
│       ├── 📄 evaluation.py           # Custom logging (BugLogger) and progress management (EvaluationProgressManager)
│       ├── 📄 token_tracking.py       # Token usage tracking utilities for model evaluation
│       ├── 📄 common.py               # Common utilities (timestamp formatting, duration helpers)
│       └── 📄 model_specs.py          # Model specifications and context window limits
├── 📁 dataset/                        # Dataset-specific implementations
│   ├── 📁 defects4j/
│   │   ├── 📄 defects4j_analysis.py   # Bug category analysis and blame feasibility analysis
│   │   ├── 📄 defects4j_extractor.py  # Defects4JExtractor (both interfaces of BlameExtractor and BugInfoExtractor)
│   │   └── 📁 bug_description/        # Mined bug descriptions (JSON files)
├── 📁 evaluation/                     # Evaluation scripts for running experiments
│   ├── 📄 run_defects4j_evaluation.py # Defects4J evaluation on baseline and blame-augmented mode
│   └── 📄 run_hafix_agent.py          # General agent runner
├── 📁 config/
│   └── 📄 defects4j.yaml              # Configuration of prompt template, model, agent, environment
├── 📁 analysis/                       # Result analysis and visualization scripts
│   └── 📄 utils.py                    # Shared helpers for analysis scripts
├── 📁 results/                        # Evaluation results
├── 📁 vendor/                         # Reference projects
└── 📄 pyproject.toml                  # Project metadata and dependencies

```


## 🚀 Environment Setup
- Conda environment
```
git clone https://github.com/SAILResearch/HAFixAgent.git
cd HAFixAgent
conda create -n hafixagent python=3.11
conda activate hafixagent
pip install -e .
```
- Load the docker image from anywhere else
```
docker load -i defects4j_latest.tar.gz
```

## 🔧 RQ0: Blameable Analysis
```
python analysis/analyze_blame_commit_count.py --bug-category all -o results/blame_commit_analysis/defects4j_blame_commit_counts.csv --workers 8
python analysis/analyze_blame_commit_count.py --stats -o results/blame_commit_analysis/defects4j_blame_commit_counts.csv
```

## 🐳 RQ1: Effectiveness Evaluation
- Run HAFixAgent
```
# Example1: single_hunk, runtime mode
hafixagent --bug-category single_hunk --history baseline --selector-type llm_judge --context-mode runtime --blame-category both --workers 4
hafixagent --bug-category single_hunk --history fn_all --selector-type llm_judge --context-mode runtime --blame-category both --workers 4
```
- Result analysis and generate analysis figures
```
# 1. Internal baseline, for all 4 bug categories separately, and one figure with 4 sub-figures
python analysis/analyze_rq1_heuristics_comparison.py -h1 1 -h2 5 -h3 7 -h4 8 -c all -s llm_judge -n 1 --grid

# 2. External baseline, for Bar chart vs HUNK4J (371 common bugs)
# 2.1 HAFixAgent vs RepairAgent in table
python analysis/analyze_rq1_external_baselines.py --baseline repairagent -s llm_judge -n 1
# 2.2 HAFixAgent vs RepairAgent in table
python analysis/analyze_rq1_external_baselines.py --baseline hunk4j -s llm_judge -n 1
```

## 📊 RQ2 Efficiency Analysis
```
# separately
python analysis/analyze_rq3_cost_step_comparison.py --mode multi-config --rq1-dir results/defects4j --rq2-dir results/defects4j_adaptive --bug-category single_line --no-adaptive --output results/rq3_analysis

# Pareto frontier one-panel
python analysis/analyze_rq3_cost_effectiveness_tradeoff.py --bug-category all --single-panel --no-pareto --cost-metric avg
```

## 📚 Citation
If you found this work helpful, please consider citing it using the following:
```bibtex
@misc{shi2025hafixagent,
      title={HAFixAgent: History-Aware Automated Program Repair Agent}, 
      author={Yu Shi and Hao Li and Bram Adams and Ahmed E. Hassan},
      year={2025},
      eprint={2511.01047},
      archivePrefix={arXiv},
      primaryClass={cs.SE},
      url={https://arxiv.org/abs/2511.01047}, 
}
```
<details>
<summary> HAFix (Prior Work) </summary>

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
For questions or issues, please:
- Open a GitHub issue

##  Acknowledgement 

- [mini-swe-agent](https://mini-swe-agent.com/latest/)
- [SWE-bench](https://www.swebench.com/)
