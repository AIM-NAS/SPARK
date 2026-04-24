# SPARK: Structured Progressive Knowledge Activation for LLM-Driven Neural Architecture Search

<p align="center">
  <b>Factor-scoped LLM-driven neural architecture search for reliable program evolution</b>
</p>

<p align="center">
  <a href="#overview">Overview</a> •
  <a href="#method">Method</a> •
  <a href="#installation">Installation</a> •
  <a href="#quick-start">Quick Start</a> •
  <a href="#results">Results</a> •
  <a href="#citation">Citation</a>
</p>

---

## Overview

This repository contains the official implementation of **SPARK** (**S**tructured **P**rogressive **A**ctivation of **R**elevant **K**nowledge), a structure-guided editing framework for **LLM-driven Neural Architecture Search (NAS)**.

LLM-based architecture search can generate executable model code directly, but free-form code rewriting often modifies multiple architectural factors at once. In NAS programs, this can lead to **functional entanglement**: a single edit may simultaneously change *which operator is used* and *how the operator is invoked or wired*, causing unpredictable behavior, invalid candidates, or unstable performance.

SPARK addresses this issue by converting free-form architecture rewriting into a **where-then-how** editing process:

1. **Where to edit**: select an explicit architectural factor, such as `OPERATOR` or `ACTION`.
2. **How to edit**: generate a factor-conditioned code patch only inside the selected region.
3. **What to keep fixed**: freeze all non-selected regions and public training/evaluation interfaces.

This repository builds on the OpenEvolve-style evolutionary coding pipeline and adapts it to program-structured NAS on the CLRS algorithmic reasoning benchmark.

---

## Highlights

- **Factor-scoped LLM editing**: decomposes architecture evolution into `OPERATOR` and `ACTION` factors.
- **ASR + RC + SAR pipeline**:
  - `ASR`: Architecture Scope Router, selects the target edit scope.
  - `RC`: Refinement Compass, converts search feedback into a scope-local directive.
  - `SAR`: Scoped Architecture Refiner, generates a constrained code patch.
- **Factor-respecting feasibility checks**: rejects proposals that modify frozen regions, break interfaces, or violate tensor-shape/masking constraints.
- **OpenEvolve-compatible search backbone**: uses archive-based evolutionary search with elite/diverse candidate sampling.
- **CLRS NAS evaluation**: searches on DFS and transfers the best architecture to other CLRS tasks.

---

## Method

SPARK treats each candidate architecture as an executable Python program. Each evolution step is factorized as:

```text
f_t       = ASR(a_t, H_t)          # choose edit factor: OPERATOR or ACTION
d_t       = RC(a_t, f_t, H_t)      # generate refinement directive under the factor
a_{t+1}   = SAR(a_t, f_t, d_t, H_t)# produce factor-conditioned code patch
```

where:

- `a_t` is the parent architecture program;
- `H_t` is the evolution context, including parent code, recent outcomes, and archive examples;
- `f_t` is the selected functional factor;
- `d_t` is the refinement directive;
- `a_{t+1}` is the proposed offspring architecture.

In the CLRS implementation, the editable architecture is split into two disjoint factors:

| Factor | Meaning | Typical edits |
|---|---|---|
| `OPERATOR` | What computation modules are defined | projections, gates, residual blocks, message operators |
| `ACTION` | How modules are invoked and composed | message construction, masking, routing, aggregation, control flow |

The feasibility checker accepts a proposal only when the code diff is local to the selected factor and the frozen regions remain unchanged.

---

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

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/AIM-NAS/SPARK.git
<<<<<<< HEAD
cd <SPARK>
=======
cd <your-repo>
>>>>>>> a9359cb92058708f5500d0120df87c665b360a72
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

---

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

---

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

---

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

---

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

---

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

---

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

---

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

---

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

---

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

This repository also builds on OpenEvolve. Please consider citing or acknowledging the upstream project:

```bibtex
@misc{sharma2025openevolve,
  title        = {OpenEvolve: An Open-Source Evolutionary Coding Agent},
  author       = {Sharma, A.},
  year         = {2025},
  howpublished = {\url{https://github.com/algorithmicsuperintelligence/openevolve}}
}
```

---

## Acknowledgements

This project is inspired by recent progress in LLM-driven code optimization, evolutionary program search, and neural architecture search. The implementation is adapted from the OpenEvolve-style evolutionary coding framework and extended with SPARK's factor-scoped architecture editing modules.

---

## License

Please refer to `LICENSE` for licensing information. If this repository is derived from OpenEvolve, make sure the upstream license and attribution requirements are preserved.
