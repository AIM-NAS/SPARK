# SPARK: Structured Progressive Knowledge Activation for LLM-Driven Neural Architecture Search




This repository is the official implementation of SPARK.


## 🗞️ Release Notes
- [2026/04/23] 🚀 We’re thrilled to release the SPARK! The code is now open to the community.
- [2026/05/01] 🎉 SPARK has been accepted to ICML 2026!





## Repository Structure

The current codebase is organized as follows:

```text
.
├── llm/                    # LLM clients and OpenAI-compatible API wrappers
├── prompt/                 # Prompt templates for ASR, RC, SAR, and baseline editing
├── utils/                  # Utilities for parsing, diff checking, logging, and evaluation helpers
├── __init__.py
├── _version.py
├── cli.py                  # Command-line interface helpers
├── config.py               # Python-side configuration definitions
├── config.yaml             # Default experiment configuration
├── controller.py           # Main evolutionary search controller
├── database.py             # Candidate archive, population, and result storage
├── evaluation_result.py    # Evaluation result data structures
├── evaluator.py            # CLRS/NAS evaluator entry point
├── initial_for.py          # Optional initial/search helper program
├── initial_program.py      # Seed architecture program with factor-scoped regions
├── iteration.py            # Per-iteration state, logging, and evolution records
├── openevolve-run.py       # Main executable script
└── process_parallel.py     # Parallel evaluation / process management utilities
```


## Installation

### 1. Clone the repository

```bash
git clone https://github.com/AIM-NAS/SPARK.git
cd SPARK
```

### 2. Create environment

```bash
conda create -n spark_nas python=3.10 -y
conda activate spark_nas
```

### 3. Install dependencies

```bash
pip install -e .
```

If your evaluator depends on CLRS or a local CLRS-PyTorch/JAX environment, install the corresponding benchmark dependencies according to your local setup.

### 4. Configure LLM API

SPARK uses an OpenAI-compatible API interface. Set your API key before running experiments:

```bash
export OPENAI_API_KEY="your_api_key_here"
```

Then edit `config.yaml` to specify your model and endpoint, for example:

```yaml
llm:
  model: "your-model-name"
  api_base: "https://your-openai-compatible-endpoint/v1"
  temperature: 0.7
```

Do **not** commit API keys to GitHub.



## Quick Start

Run a SPARK search from the initial CLRS architecture:

```bash
python openevolve-run.py \
  initial_program.py \
  evaluator.py \
  --config config.yaml \
  --output runs/spark_dfs_seed0 \
  --iterations 100
```

A typical run will:

1. load the seed architecture from `initial_program.py`;
2. sample a parent from the archive;
3. use ASR to select `OPERATOR` or `ACTION`;
4. use RC to generate a factor-local refinement directive;
5. use SAR to generate a complete updated program;
6. run syntax/interface/shape feasibility checks;
7. train and evaluate valid candidates with `evaluator.py`;
8. update the archive if the candidate improves fitness.



## Running Multiple Seeds

```bash
for seed in 0 1 2 3 4; do
  python openevolve-run.py \
    initial_program.py \
    evaluator.py \
    --config config.yaml \
    --output runs/spark_dfs_seed${seed} \
    --iterations 100 \
    --seed ${seed}
done
```



## Configuration

Most experiment settings can be changed in `config.yaml`, including:

```yaml
max_iterations: 100
population_size: 100
num_islands: 5
archive_size: 100

llm:
  model: "your-model-name"
  temperature: 0.7
  timeout: 120
  retries: 3

spark:
  enable: true
  factors: ["OPERATOR", "ACTION"]
  router_retries: 3
  freeze_non_selected_region: true
  feasibility_check: true
```

The exact fields may differ depending on your local branch. Please check `config.py` and `config.yaml` for the final supported options.



## Factor-Scoped Region Markers

`initial_program.py` should contain explicit editable regions. A typical structure is:

```python
# ===== BEGIN OPERATOR REGION =====
# Define architecture modules, projections, gates, or operator parameters here.
# ===== END OPERATOR REGION =====

# ===== BEGIN ACTION REGION =====
# Define how operators are invoked, routed, masked, and aggregated here.
# ===== END ACTION REGION =====
```

During SPARK evolution, if ASR selects `OPERATOR`, SAR should only modify the operator region. If ASR selects `ACTION`, SAR should only modify the action region. Changes outside the selected region are rejected before full evaluation.



## Results

### CLRS-DFS Search

| Method | DFS OOD Acc. (%) | Notes |
|---|---:|---|
| CLRS reference | 46.78 | Original reference architecture |
| OpenEvolve baseline | 32.54 | Free-form LLM evolution under the same search setting |
| EvoPrompting | 68.14 | Prior LLM-driven NAS baseline |
| FunSearch-style baseline | 74.50 | Program-search baseline |
| EoH-style baseline | 77.27 | Evolution-of-heuristics baseline |
| **SPARK (Ours)** | **83.74** | Factor-scoped where-then-how evolution |

SPARK reaches its best DFS OOD accuracy with 57 evaluated candidates, corresponding to a 28.1× evaluation-efficiency improvement over the 1600-evaluation EvoPrompting setting.

### 10-Task CLRS Transfer Results

After searching on DFS, the best architecture is transferred to 9 additional CLRS tasks and trained/evaluated from scratch.

| Method | Avg. OOD Acc. (%) | Avg. MACs (K) |
|---|---:|---:|
| CLRS reference | 71.22 | 450 |
| OpenEvolve baseline | 68.30 | 481 |
| EvoPrompting | 74.42 | 448 |
| FunSearch-style baseline | 77.61 | 469 |
| EoH-style baseline | 79.43 | 463 |
| **SPARK (Ours)** | **83.92** | **453** |

The results suggest that the improvement mainly comes from more reliable search dynamics and reduced functional entanglement, rather than simply increasing model size or compute.



## Ablation

| Variant | LLM | DFS OOD Acc. (%) | Gain over CLRS |
|---|---|---:|---:|
| RC + SAR only | DeepSeek-R1 | 56.79 | +10.01 |
| ASR only | DeepSeek-R1 | 65.28 | +18.50 |
| **SPARK: ASR + RC + SAR** | DeepSeek-R1 | **83.74** | **+36.96** |
| RC + SAR only | Qwen-Plus | 56.00 | +9.22 |
| ASR only | Qwen-Plus | 64.50 | +17.72 |
| **SPARK: ASR + RC + SAR** | Qwen-Plus | **80.50** | **+33.72** |

These results show that both scope selection and scope-local refinement are useful, while their combination gives the strongest and most stable improvement.



## Logs and Outputs

Each experiment directory may contain:

```text
runs/spark_dfs_seed0/
├── logs/                  # Runtime logs and per-iteration metrics
├── programs/              # Generated candidate programs
├── checkpoints/           # Optional evaluator/model checkpoints
├── results/               # Candidate scores and summary metrics
└── config.yaml            # Copied experiment configuration
```

Useful quantities to track include:

- `ood_acc`: CLRS out-of-distribution accuracy;
- `macs`: multiply-accumulate operations;
- `param_count`: model parameter count;
- `valid_rate`: fraction of proposals passing feasibility checks;
- `entanglement_rate`: fraction of non-factor-local edits;
- `best_so_far`: best archived architecture score over iterations.



## Reproducing Main Experiments

A typical reproduction workflow is:

```bash
# 1. Run DFS search
python openevolve-run.py initial_program.py evaluator.py \
  --config config.yaml \
  --output runs/spark_dfs_seed0 \
  --iterations 100 \
  --seed 0

# 2. Parse the best architecture from the run directory
# 3. Train/evaluate the selected architecture on other CLRS tasks
# 4. Aggregate results across tasks/seeds
```

The exact aggregation script depends on your local evaluator implementation. Please refer to the scripts under `utils/` or your experiment-specific parsing scripts.




## 🤝 Acknowledgements

We would like to express our sincere gratitude to [OpenEvolve](https://github.com/algorithmicsuperintelligence/openevolve) for providing open-source resources that contributed to the development of this project.

## Citation

If you find this repository useful, please cite:

```bibtex
@article{liu2026spark,
  title   = {Structured Progressive Knowledge Activation for LLM-Driven Neural Architecture Search},
  author  = {Liu, Zhen and Liu, Yuhan and Fu, Jingwen},
  journal = {Preprint},
  year    = {2026}
}
```

