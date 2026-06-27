"""
5.output_analysis.py — Qualitative error analysis of hard cases.

For every result file in ./results/<model_name>/ it finds predictions that
differ from the gold label, characterises the error patterns, and identifies
"hard cases" — examples that are misclassified by ALL available models on a
given task (i.e. consensus failures).

Outputs (all in ./results/combined_results/):
  error_analysis.xlsx   — one sheet per model+task with misclassified rows
                          plus summary sheets
  hard_cases.xlsx       — examples wrong across every model evaluated on the
                          same task/dataset
"""

import os
import sys
import textwrap
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from sklearn.metrics import confusion_matrix

RESULTS_ROOT = "./results"
COMBINED_DIR = os.path.join(RESULTS_ROOT, "combined_results")
SKIP_DIRS    = {"combined_results"}
os.makedirs(COMBINED_DIR, exist_ok=True)

# Same registry as 3.combine_results.py
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

LABEL_ORDER = ["entailment", "neutral", "contradiction"]


# ── Load all result files ─────────────────────────────────────────────────────
records = []   # list of dicts with full DataFrame + metadata

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
            print(f"[WARN] Cannot read {filepath}: {e}")
            continue
        if correct_col not in df.columns or "predicted_label" not in df.columns:
            print(f"[WARN] Missing columns in {filepath} — skipping.")
            continue

        df = df.copy()
        df["_correct"]    = df[correct_col].astype(str).str.strip().str.lower()
        df["_predicted"]  = df["predicted_label"].astype(str).str.strip().str.lower()
        df["_error"]      = df["_correct"] != df["_predicted"]
        df["_model"]      = model_name
        df["_model_type"] = model_type
        df["_task"]       = task
        df["_dataset"]    = dataset
        df["_correct_col"] = correct_col

        records.append({
            "model":      model_name,
            "model_type": model_type,
            "task":       task,
            "dataset":    dataset,
            "correct_col": correct_col,
            "df":         df,
        })
        n_err = df["_error"].sum()
        acc   = 1 - n_err / len(df)
        print(f"  {model_name:20s} | {model_type:10s} | {task:25s} "
              f"| {n_err:5d} errors / {len(df)} ({acc:.1%} acc)")

if not records:
    sys.exit("No result files found — run the pipeline scripts first.")

print(f"\nLoaded {len(records)} result file(s).\n")


# ── Per-file error breakdown ──────────────────────────────────────────────────
def error_type_label(row):
    return f"{row['_correct']} → {row['_predicted']}"

summary_rows = []
confusion_data = {}   # key: (model, model_type, task) → (y_true, y_pred)

for rec in records:
    df       = rec["df"]
    key      = (rec["model"], rec["model_type"], rec["task"])
    errors   = df[df["_error"]].copy()
    errors["error_type"] = errors.apply(error_type_label, axis=1)

    y_true = df["_correct"].tolist()
    y_pred = df["_predicted"].tolist()
    confusion_data[key] = (y_true, y_pred)

    total   = len(df)
    n_err   = len(errors)
    unknown = (df["_predicted"] == "unknown").sum()

    # Count each confusion pair
    error_counts = (
        errors.groupby(["_correct", "_predicted"])
              .size()
              .reset_index(name="count")
              .sort_values("count", ascending=False)
    )
    error_counts["error_type"] = (
        error_counts["_correct"] + " → " + error_counts["_predicted"]
    )
    error_counts["pct_of_errors"] = (error_counts["count"] / n_err * 100).round(1)

    summary_rows.append({
        "model":            rec["model"],
        "model_type":       rec["model_type"],
        "task":             rec["task"],
        "dataset":          rec["dataset"],
        "total_examples":   total,
        "n_correct":        total - n_err,
        "n_errors":         n_err,
        "error_rate":       round(n_err / total, 4),
        "n_unknown":        int(unknown),
        "top_error_type":   error_counts.iloc[0]["error_type"] if len(error_counts) else "—",
        "top_error_count":  int(error_counts.iloc[0]["count"]) if len(error_counts) else 0,
        "top_error_pct":    error_counts.iloc[0]["pct_of_errors"] if len(error_counts) else 0,
    })

summary_df = pd.DataFrame(summary_rows).sort_values(
    ["task", "model", "model_type"]
).reset_index(drop=True)


# ── Hard cases: wrong across ALL models on the same task ─────────────────────
# Group records by (task, dataset) — find premise+hypothesis present in all
# models' error sets for that task.

# Build a unique sentence-pair key per record
def pair_key(df):
    if "premise" in df.columns and "hypothesis" in df.columns:
        return df["premise"].astype(str) + "|||" + df["hypothesis"].astype(str)
    if "premise" in df.columns and "negated_hypothesis" in df.columns:
        return df["premise"].astype(str) + "|||" + df["negated_hypothesis"].astype(str)
    # fallback: use first two text columns
    text_cols = [c for c in df.columns if df[c].dtype == object and not c.startswith("_")][:2]
    return df[text_cols[0]].astype(str) + "|||" + df[text_cols[1]].astype(str)

from collections import defaultdict
task_records = defaultdict(list)
for rec in records:
    task_records[rec["task"]].append(rec)

hard_case_frames = []
for task, recs in task_records.items():
    if len(recs) < 2:
        continue  # need at least two models to compare

    # Error sets per model
    error_sets = []
    for rec in recs:
        df  = rec["df"]
        err = df[df["_error"]].copy()
        err["_pair_key"] = pair_key(err)
        error_sets.append(set(err["_pair_key"]))

    # Intersection = wrong in every model
    consensus_keys = set.intersection(*error_sets)
    if not consensus_keys:
        continue

    # Build a combined hard-case table
    base_rec = recs[0]
    base_df  = base_rec["df"].copy()
    base_df["_pair_key"] = pair_key(base_df)
    hard = base_df[base_df["_pair_key"].isin(consensus_keys)].copy()

    # Add each model's prediction as a separate column
    hard = hard.rename(columns={
        "_correct":    "gold_label",
        "_predicted":  f"pred_{base_rec['model']}_{base_rec['model_type']}",
    })
    for rec in recs[1:]:
        rec_df = rec["df"].copy()
        rec_df["_pair_key"] = pair_key(rec_df)
        col_name = f"pred_{rec['model']}_{rec['model_type']}"
        lookup = rec_df.set_index("_pair_key")["_predicted"]
        hard[col_name] = hard["_pair_key"].map(lookup)

    # Keep only readable columns
    keep = [c for c in hard.columns if not c.startswith("_")]
    # Always include premise / hypothesis / negated_hypothesis if present
    text_cols = [c for c in ["premise", "hypothesis", "original_hypothesis",
                              "negated_hypothesis", "original_label"]
                 if c in hard.columns]
    pred_cols = [c for c in keep if c.startswith("pred_")]
    final_cols = text_cols + ["gold_label"] + pred_cols
    hard = hard[[c for c in final_cols if c in hard.columns]].reset_index(drop=True)
    hard.insert(0, "task", task)

    hard_case_frames.append(hard)
    print(f"Hard cases [{task}]: {len(hard)} examples wrong across all {len(recs)} model(s)")

hard_cases_df = (
    pd.concat(hard_case_frames, ignore_index=True)
    if hard_case_frames else pd.DataFrame(columns=["task", "note"])
)


# ── Confusion-matrix plots ────────────────────────────────────────────────────
def plot_confusion(y_true, y_pred, title, filepath):
    labels = [l for l in LABEL_ORDER if l in set(y_true) | set(y_pred)]
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm_norm, vmin=0, vmax=1, cmap="Blues")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set_xticks(range(len(labels))); ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=9)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("Predicted", fontsize=10)
    ax.set_ylabel("Gold", fontsize=10)
    ax.set_title(title, fontsize=10, pad=8)

    for r in range(len(labels)):
        for c in range(len(labels)):
            ax.text(c, r, f"{cm[r,c]}\n({cm_norm[r,c]:.0%})",
                    ha="center", va="center", fontsize=8,
                    color="white" if cm_norm[r, c] > 0.6 else "black")
    fig.tight_layout()
    fig.savefig(filepath, dpi=130, bbox_inches="tight")
    plt.close(fig)

print("\nGenerating confusion matrices …")
for (model, model_type, task), (y_true, y_pred) in confusion_data.items():
    fname = f"confusion_{model}_{model_type}_{task.replace(' ','_')}.png"
    fpath = os.path.join(COMBINED_DIR, fname)
    plot_confusion(y_true, y_pred,
                   f"{model} · {model_type}\n{task}",
                   fpath)
    print(f"  Saved: {fpath}")


# ── Error-type distribution plot ──────────────────────────────────────────────
print("\nGenerating error-type distribution …")

all_errors = []
for rec in records:
    err = rec["df"][rec["df"]["_error"]].copy()
    err["error_type"]  = err.apply(error_type_label, axis=1)
    err["source_label"] = f"{rec['model']} · {rec['model_type']}\n{rec['task']}"
    all_errors.append(err[["error_type", "source_label"]])

if all_errors:
    err_df  = pd.concat(all_errors, ignore_index=True)
    top_types = (
        err_df["error_type"]
        .value_counts()
        .head(10)
        .reset_index()
    )
    top_types.columns = ["error_type", "count"]

    fig, ax = plt.subplots(figsize=(8, 4))
    colors = plt.get_cmap("tab10")(np.linspace(0, 0.9, len(top_types)))
    bars = ax.barh(top_types["error_type"][::-1], top_types["count"][::-1],
                   color=colors[::-1], alpha=0.88)
    ax.bar_label(bars, padding=3, fontsize=9)
    ax.set_xlabel("Number of occurrences (across all models & tasks)", fontsize=10)
    ax.set_title("Top-10 Confusion Pairs  (gold → predicted)", fontsize=12)
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    etype_path = os.path.join(COMBINED_DIR, "error_type_distribution.png")
    fig.savefig(etype_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {etype_path}")


# ── Hard-case label distribution plot ────────────────────────────────────────
if not hard_cases_df.empty and "gold_label" in hard_cases_df.columns:
    fig, ax = plt.subplots(figsize=(6, 3.5))
    hard_cases_df.groupby(["task", "gold_label"]).size().unstack(fill_value=0).plot(
        kind="bar", ax=ax, colormap="Set2", alpha=0.88
    )
    ax.set_xlabel("Task", fontsize=10)
    ax.set_ylabel("Hard-case count", fontsize=10)
    ax.set_title("Hard Cases by Task and Gold Label", fontsize=12)
    ax.legend(title="Gold label", fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    plt.xticks(rotation=20, ha="right")
    fig.tight_layout()
    hc_dist_path = os.path.join(COMBINED_DIR, "hard_cases_distribution.png")
    fig.savefig(hc_dist_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {hc_dist_path}")


# ── Print qualitative digest ──────────────────────────────────────────────────
print("\n" + "=" * 70)
print("ERROR ANALYSIS DIGEST")
print("=" * 70)

for rec in records:
    df    = rec["df"]
    errors = df[df["_error"] & (df["_predicted"] != "unknown")]
    if errors.empty:
        continue

    print(f"\n▸ {rec['model']} · {rec['model_type']} — {rec['task']}")
    print(f"  {len(errors)} errors / {len(df)} examples  "
          f"({len(errors)/len(df):.1%} error rate)")

    # Top confusion pairs
    pair_counts = (
        errors.groupby(["_correct", "_predicted"])
              .size()
              .sort_values(ascending=False)
    )
    print("  Top confusion pairs:")
    for (gold, pred), cnt in pair_counts.head(4).items():
        pct = cnt / len(errors) * 100
        print(f"    {gold:15s} → {pred:15s}  {cnt:5d}  ({pct:.1f}% of errors)")

    # Sample hard cases (most confident wrong: long sentences, clear categories)
    text_col = next(
        (c for c in ["premise", "premises"] if c in errors.columns), None
    )
    hyp_col = next(
        (c for c in ["negated_hypothesis", "hypothesis"] if c in errors.columns), None
    )
    if text_col and hyp_col:
        sample = errors.sample(n=min(3, len(errors)), random_state=1)
        print("  Sample misclassified examples:")
        for _, row in sample.iterrows():
            prem = textwrap.shorten(str(row[text_col]), width=70, placeholder="…")
            hyp  = textwrap.shorten(str(row[hyp_col]),  width=70, placeholder="…")
            print(f"    Premise   : {prem}")
            print(f"    Hypothesis: {hyp}")
            print(f"    Gold={row['_correct']}  Predicted={row['_predicted']}")
            print()


# ── Write Excel workbook ──────────────────────────────────────────────────────
print("Writing Excel workbook …")
xlsx_path = os.path.join(COMBINED_DIR, "error_analysis.xlsx")
hc_path   = os.path.join(COMBINED_DIR, "hard_cases.xlsx")

with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
    summary_df.to_excel(writer, sheet_name="Summary", index=False)
    for rec in records:
        df     = rec["df"]
        errors = df[df["_error"]].copy()
        errors["error_type"] = errors.apply(error_type_label, axis=1)
        # drop internal _ columns before saving
        out = errors[[c for c in errors.columns if not c.startswith("_")]]
        sheet = f"{rec['model_type'][:4]}_{rec['task'][:18]}"
        sheet = sheet.replace(" ", "_")[:31]   # Excel sheet name limit
        out.to_excel(writer, sheet_name=sheet, index=False)

print(f"  Saved: {xlsx_path}")

hard_cases_df.to_excel(hc_path, index=False)
print(f"  Saved: {hc_path}")

print("\nDone.")
