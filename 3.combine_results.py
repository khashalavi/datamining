"""
3.combine_results.py — Aggregate all model results into one summary table.

Walks ./results/<model_name>/ (skips ./results/combined_results/) and reads
every known result Excel file.  For each file it computes accuracy, macro
precision, macro recall, and macro F1, then writes a single summary to:
  ./results/combined_results/final_table.xlsx
  ./results/combined_results/final_table.csv
"""

import os
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

RESULTS_ROOT   = "./results"
COMBINED_DIR   = os.path.join(RESULTS_ROOT, "combined_results")
SKIP_DIRS      = {"combined_results"}

os.makedirs(COMBINED_DIR, exist_ok=True)

# Maps filename → (task, dataset, model_type, correct_col)
FILE_META = {
    "baseline.xlsx": (
        "NLI Classification", "SNLI Test", "Baseline", "correct_label"
    ),
    "finetuned_predictions.xlsx": (
        "NLI Classification", "SNLI Test", "Fine-tuned", "correct_label"
    ),
    "generalization-baseline.xlsx": (
        "Generalization", "Multi-NLI val_mismatched", "Baseline", "correct_label"
    ),
    "generalization-finetuned.xlsx": (
        "Generalization", "Multi-NLI val_mismatched", "Fine-tuned", "correct_label"
    ),
    "negation-baseline.xlsx": (
        "Negation Robustness", "Multi-NLI negated", "Baseline", "correct_label_negated"
    ),
    "negation-finetuned.xlsx": (
        "Negation Robustness", "Multi-NLI negated", "Fine-tuned", "correct_label_negated"
    ),
}


def compute_metrics(df: pd.DataFrame, correct_col: str) -> dict:
    y_true = df[correct_col].astype(str)
    y_pred = df["predicted_label"].astype(str)

    # Drop rows where prediction is unknown (model output couldn't be parsed)
    mask = y_pred != "unknown"
    n_unknown = (~mask).sum()
    y_true, y_pred = y_true[mask], y_pred[mask]

    labels = sorted(y_true.unique().tolist())

    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
    rec  = recall_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
    f1   = f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)

    return {
        "accuracy":  round(acc,  4),
        "precision": round(prec, 4),
        "recall":    round(rec,  4),
        "f1_macro":  round(f1,   4),
        "n_examples": len(df),
        "n_unknown":  int(n_unknown),
    }


rows = []

for model_name in sorted(os.listdir(RESULTS_ROOT)):
    if model_name in SKIP_DIRS:
        continue
    model_dir = os.path.join(RESULTS_ROOT, model_name)
    if not os.path.isdir(model_dir):
        continue

    for filename, (task, dataset, model_type, correct_col) in FILE_META.items():
        filepath = os.path.join(model_dir, filename)
        if not os.path.exists(filepath):
            continue

        try:
            df = pd.read_excel(filepath)
        except Exception as e:
            print(f"  [WARN] Could not read {filepath}: {e}")
            continue

        if correct_col not in df.columns or "predicted_label" not in df.columns:
            print(f"  [WARN] Expected columns missing in {filepath} — skipping.")
            continue

        metrics = compute_metrics(df, correct_col)
        rows.append({
            "model":       model_name,
            "model_type":  model_type,
            "task":        task,
            "dataset":     dataset,
            **metrics,
        })
        print(
            f"  {model_name} | {model_type:10s} | {task:25s} | "
            f"acc={metrics['accuracy']:.4f}  f1={metrics['f1_macro']:.4f}"
        )

if not rows:
    print("No result files found — nothing to combine.")
else:
    summary = pd.DataFrame(rows, columns=[
        "model", "model_type", "task", "dataset",
        "accuracy", "precision", "recall", "f1_macro",
        "n_examples", "n_unknown",
    ])
    summary = summary.sort_values(["task", "model", "model_type"]).reset_index(drop=True)

    xlsx_path = os.path.join(COMBINED_DIR, "final_table.xlsx")
    csv_path  = os.path.join(COMBINED_DIR, "final_table.csv")
    summary.to_excel(xlsx_path, index=False)
    summary.to_csv(csv_path, index=False)

    print(f"\nSaved {len(summary)} rows →")
    print(f"  {xlsx_path}")
    print(f"  {csv_path}")
    print("\n" + summary.to_string(index=False))
