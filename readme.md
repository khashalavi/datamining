# Natural Language Inference — SNLI Fine-tuning Pipeline

3-way NLI classification (Entailment / Neutral / Contradiction) using a causal
language model fine-tuned on the Stanford NLI corpus.  The pipeline covers
training, baseline evaluation, two bonus robustness tasks, result aggregation,
visualisation, and qualitative error analysis.

---

## Table of Contents

1. [Task Overview](#task-overview)
2. [Project Structure](#project-structure)
3. [Environment Setup](#environment-setup)
4. [Pipeline Scripts](#pipeline-scripts)
5. [Outputs Reference](#outputs-reference)
6. [Prompt Format](#prompt-format)
7. [Label Parsing](#label-parsing)
8. [Bonus Tasks](#bonus-tasks)
9. [Running the Full Pipeline](#running-the-full-pipeline)

---

## Task Overview

Given a **premise** sentence and a **hypothesis** sentence, the model must
predict one of three labels:

| Label | Meaning |
|---|---|
| `entailment` | The hypothesis must be true given the premise |
| `neutral` | The truth of the hypothesis cannot be determined |
| `contradiction` | The hypothesis must be false given the premise |

**Dataset:** [stanfordnlp/snli](https://huggingface.co/datasets/stanfordnlp/snli)
— ~570 000 human-written sentence pairs.

**Model:** `Qwen/Qwen3-1.7B` (or `Qwen/Qwen3-0.6B`) fine-tuned as a causal
language model.  Change `MODEL_NAME` at the top of any script to switch model.

---

## Project Structure

```
datamining/
├── 0.finetuning_pipeline.py        # Fine-tune the model on SNLI
├── 1.baseline_performance.py       # Zero-shot base-model evaluation on SNLI test
├── 2.1-bonus-generalization.py     # Cross-genre eval on Multi-NLI
├── 2.2-bonus-negation.py           # Adversarial negation eval
├── 3.combine_results.py            # Aggregate all results into one summary table
├── 4.generate_graphs.py            # Produce visualisation charts
├── 5.output_analysis.py            # Qualitative hard-case & error analysis
├── negation_set_generation.py      # LLM-based negated-hypothesis generator
│
├── data/                           # Cached datasets (auto-created)
│   ├── multi_nli_val_mismatched/   # Multi-NLI split saved to disk
│   └── negated_validation_set.xlsx # Negated hypotheses from Multi-NLI
│
├── models/                         # Saved model checkpoints (auto-created)
│   └── <MODEL_SLUG>/
│       ├── final_model/            # Fine-tuned weights + tokenizer
│       ├── training_losses.txt     # Step-by-step loss log
│       └── loss_plot.png           # Training vs validation loss curve
│
├── results/                        # Evaluation outputs (auto-created)
│   └── <MODEL_SLUG>/
│       ├── baseline.xlsx                     # Base model on SNLI test
│       ├── finetuned_predictions.xlsx        # Fine-tuned model on SNLI test
│       ├── generalization-baseline.xlsx      # Base model on Multi-NLI
│       ├── generalization-finetuned.xlsx     # Fine-tuned model on Multi-NLI
│       ├── negation-baseline.xlsx            # Base model on negated set
│       └── negation-finetuned.xlsx           # Fine-tuned model on negated set
│   └── combined_results/
│       ├── final_table.xlsx                  # Aggregated metrics (all models)
│       ├── final_table.csv
│       ├── accuracy_by_task.png
│       ├── f1_by_task.png
│       ├── metrics_heatmap.png
│       ├── baseline_vs_finetuned.png
│       ├── unknown_rate.png
│       ├── confusion_<model>_<task>.png      # One per model × task
│       ├── error_type_distribution.png
│       ├── hard_cases_distribution.png
│       ├── error_analysis.xlsx               # Per-model error sheets
│       └── hard_cases.xlsx                   # Consensus failures across models
│
├── snli-project/                   # uv virtual environment
│   ├── .venv/
│   └── pyproject.toml
│
└── history/                        # Previous experiment notebooks
```

---

## Environment Setup

The project uses [uv](https://github.com/astral-sh/uv) for fast, reproducible
environments.

### 1. Create the environment

```powershell
# From the project root
cd snli-project
uv venv --python 3.11
.\.venv\Scripts\Activate.ps1
```

### 2. Install dependencies

```powershell
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
uv pip install transformers datasets accelerate pandas openpyxl matplotlib scikit-learn tqdm
```

For the negation-generation script only (`negation_set_generation.py`):

```powershell
uv pip install openai python-dotenv
```

### 3. Activate before running any script

```powershell
cd snli-project
.\.venv\Scripts\Activate.ps1
cd ..
```



---

## Pipeline Scripts

### `0.finetuning_pipeline.py` — Fine-tune

Fine-tunes `Qwen3-1.7B` on 50 000 SNLI training examples for one epoch using
the HuggingFace `Trainer`.

Key settings at the top of the file:

| Variable | Default | Description |
|---|---|---|
| `MODEL_NAME` | `Qwen/Qwen3-1.7B` | Base model from HuggingFace Hub |
| `TRAIN_SAMPLES` | `50000` | Training examples to use |
| `VAL_SAMPLES` | `5000` | Validation examples during training |
| `BATCH_SIZE` | `8` | Per-device batch size |
| `GRAD_ACCUM_STEPS` | `4` | Gradient accumulation (effective batch = 32) |
| `EPOCHS` | `1` | Training epochs |
| `LEARNING_RATE` | `2e-5` | Peak learning rate (cosine schedule) |
| `MAX_LENGTH` | `128` | Max token length per example |

After training the script automatically runs inference on the full SNLI test set
and saves predictions to `results/<MODEL_SLUG>/finetuned_predictions.xlsx`.

```powershell
python 0.finetuning_pipeline.py
```

---

### `1.baseline_performance.py` — Zero-shot Baseline

Loads the same base model without any fine-tuning and evaluates it on the SNLI
test set.  Useful to quantify how much the fine-tuning actually helped.

```powershell
python 1.baseline_performance.py
```

Output: `results/<MODEL_SLUG>/baseline.xlsx`

---

### `2.1-bonus-generalization.py` — Cross-Genre Generalisation

Evaluates both the base model and the fine-tuned model on the
[Multi-NLI](https://huggingface.co/datasets/nyu-mll/multi_nli)
`validation_mismatched` split, which covers ten different genres (fiction,
government letters, telephone transcripts, etc.) not seen during training.

The split is downloaded once and cached to `./data/multi_nli_val_mismatched/`.

```powershell
python 2.1-bonus-generalization.py
```

Outputs:
- `results/<MODEL_SLUG>/generalization-baseline.xlsx`
- `results/<MODEL_SLUG>/generalization-finetuned.xlsx`

---

### `2.2-bonus-negation.py` — Adversarial Negation Robustness

Tests whether the model is fooled by negated hypotheses.  The negated dataset
is generated by an LLM (see below) and stored in `data/negated_validation_set.xlsx`.

**Negation logic:**
- Neutral examples are excluded (no well-defined gold label after negation).
- Entailment (0) → Contradiction (2) and vice versa.

If `negated_validation_set.xlsx` exists in the project root (from a previous
run of `negation_set_generation.py`) it is copied to `./data/` automatically.

```powershell
python 2.2-bonus-negation.py
```

Outputs:
- `results/<MODEL_SLUG>/negation-baseline.xlsx`
- `results/<MODEL_SLUG>/negation-finetuned.xlsx`

---

### `negation_set_generation.py` — Generate Negated Dataset

Uses the NVIDIA-hosted `openai/gpt-oss-120b` model (free tier) to negate the
verb of each hypothesis in the Multi-NLI `validation_mismatched` split.
Processes in batches of 20 and resumes from where it left off if interrupted.

Requires `NVIDIA_API_KEY` in `.env`.

```powershell
python negation_set_generation.py
```

Output: `negated_validation_set.xlsx` (project root, then copy to `./data/`).

---

### `3.combine_results.py` — Aggregate Metrics

Walks every subfolder of `./results/` (skipping `combined_results/` to avoid
circular reads), computes accuracy, macro precision, macro recall, and macro F1
for each result file, and writes a single summary table.

```powershell
python 3.combine_results.py
```

Outputs:
- `results/combined_results/final_table.xlsx`
- `results/combined_results/final_table.csv`

Safe to re-run at any time — it always rewrites from scratch.

---

### `4.generate_graphs.py` — Visualisations

Reads `final_table.csv` and produces five charts.  Run after
`3.combine_results.py`.

```powershell
python 4.generate_graphs.py
```

| Chart | Description |
|---|---|
| `accuracy_by_task.png` | Grouped bar — Baseline vs Fine-tuned accuracy per task × model |
| `f1_by_task.png` | Same layout for macro-F1 |
| `metrics_heatmap.png` | Green/red heatmap of all four metrics across model × task |
| `baseline_vs_finetuned.png` | Scatter — points above the diagonal mean fine-tuning helped |
| `unknown_rate.png` | Bar chart of unparseable-output rate per model × task |

---

### `5.output_analysis.py` — Error Analysis

Identifies hard cases and characterises the error patterns across all models
and tasks.  Run after all result files have been generated.

```powershell
python 5.output_analysis.py
```

| Output | Description |
|---|---|
| `error_analysis.xlsx` | One sheet per model×task of misclassified rows + `Summary` sheet |
| `hard_cases.xlsx` | Examples wrong across **every** model on the same task |
| `confusion_*.png` | Normalised confusion matrix per model × task |
| `error_type_distribution.png` | Top-10 gold→predicted confusion pairs across all models |
| `hard_cases_distribution.png` | Hard-case counts by task and gold label |

A qualitative digest is also printed to the console, showing the top confusion
pairs and three sampled misclassified examples per model×task.

---

## Outputs Reference

All per-example Excel files share a common column schema:

| Column | Present in | Description |
|---|---|---|
| `premise` | all | Premise sentence |
| `hypothesis` | SNLI, generalization | Hypothesis sentence |
| `original_hypothesis` | negation | Hypothesis before negation |
| `negated_hypothesis` | negation | Hypothesis after negation |
| `original_label` | negation | Gold label of the original pair |
| `correct_label` | SNLI, generalization | Gold label |
| `correct_label_negated` | negation | Gold label after label-flip |
| `predicted_label` | all | Model output (`entailment` / `neutral` / `contradiction` / `unknown`) |
| `genre` | generalization | Multi-NLI genre (fiction, telephone, …) |

---

## Prompt Format

Every script uses an identical prompt so that results are comparable:

```
Premis: <premise sentence>. Hypothesis: <hypothesis sentence>. Label:
```

The model generates up to 5 new tokens; only those new tokens are decoded.

---

## Label Parsing

The generated text is lowercased and the **first word** is matched against
`{entailment, neutral, contradiction}`.  If no first-word match is found, a
substring search is tried.  If that also fails the prediction is recorded as
`unknown` and excluded from metric computation (but counted in `n_unknown`).

---

## Bonus Tasks

| Script | Task | Dataset | Evaluates |
|---|---|---|---|
| `2.1-bonus-generalization.py` | Cross-genre generalisation | Multi-NLI val_mismatched | Base + Fine-tuned |
| `2.2-bonus-negation.py` | Adversarial negation | Multi-NLI negated hypotheses | Base + Fine-tuned |
| `5.output_analysis.py` | Qualitative error analysis | All result files | Hard cases + confusion patterns |

---

## Running the Full Pipeline

### Option A — One command (recommended)

Set the model name once in `run_pipeline.py` and execute everything:

```python
# run_pipeline.py — only line you need to edit
MODEL_NAME = "Qwen/Qwen3-0.6B"   # or "Qwen/Qwen3-1.7B"
```

```powershell
cd snli-project && .\.venv\Scripts\Activate.ps1 && cd ..
python run_pipeline.py
```

`run_pipeline.py` patches `MODEL_NAME` / `BASE_MODEL_NAME` into each child
script before running it, so every script automatically uses the same model.
It skips the negation step if the negated dataset is not found yet, and asks
whether to continue if any step fails.

At the end it prints a summary table with the status and wall-clock time of
every step.

---

### Option B — Step by step

```powershell
# 1. Activate environment
cd snli-project && .\.venv\Scripts\Activate.ps1 && cd ..

# 2. Fine-tune (writes models/Qwen3-0.6B/ and results/Qwen3-0.6B/finetuned_predictions.xlsx)
python 0.finetuning_pipeline.py

# 3. Zero-shot baseline (writes results/Qwen3-0.6B/baseline.xlsx)
python 1.baseline_performance.py

# 4a. Generate negated dataset — only needed once (requires NVIDIA_API_KEY)
python negation_set_generation.py

# 4b. Bonus: cross-genre generalisation
python 2.1-bonus-generalization.py

# 4c. Bonus: negation robustness
python 2.2-bonus-negation.py

# 5. Aggregate all metrics
python 3.combine_results.py

# 6. Generate charts
python 4.generate_graphs.py

# 7. Error analysis
python 5.output_analysis.py
```

Steps 4b and 4c are independent of each other and can be run in parallel.
Steps 5–7 require all result files to be present but are fast (CPU-only).
