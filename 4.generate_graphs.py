"""
4.generate_graphs.py — Visualise combined results from final_table.csv

Reads ./results/combined_results/final_table.csv and produces:
  1. accuracy_by_task.png      — grouped bar: baseline vs fine-tuned per task
  2. f1_by_task.png            — same layout for macro-F1
  3. metrics_heatmap.png       — heatmap of all four metrics per model×task
  4. baseline_vs_finetuned.png — scatter: baseline accuracy vs fine-tuned accuracy
  5. unknown_rate.png          — bar chart of unparseable-prediction rates
"""

import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

COMBINED_DIR = "./results/combined_results"
TABLE_PATH   = os.path.join(COMBINED_DIR, "final_table.csv")

if not os.path.exists(TABLE_PATH):
    sys.exit(f"[ERROR] {TABLE_PATH} not found — run 3.combine_results.py first.")

df = pd.read_csv(TABLE_PATH)
print(f"Loaded {len(df)} rows from {TABLE_PATH}")

PALETTE = {"Baseline": "#5b8dd9", "Fine-tuned": "#e07b54"}
METRICS  = ["accuracy", "precision", "recall", "f1_macro"]
METRIC_LABELS = {
    "accuracy":  "Accuracy",
    "precision": "Precision (macro)",
    "recall":    "Recall (macro)",
    "f1_macro":  "F1 (macro)",
}


def save(fig, name):
    path = os.path.join(COMBINED_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ── Helper: grouped bar chart ─────────────────────────────────────────────────
def grouped_bar(metric: str, ylabel: str, filename: str):
    tasks   = sorted(df["task"].unique())
    models  = sorted(df["model"].unique())
    types   = ["Baseline", "Fine-tuned"]

    # number of groups = tasks × models; two bars per group
    n_groups = len(tasks) * len(models)
    x        = np.arange(n_groups)
    width    = 0.35

    fig, ax = plt.subplots(figsize=(max(8, n_groups * 1.4), 5))

    for i, mtype in enumerate(types):
        values = []
        for task in tasks:
            for model in models:
                row = df[(df["task"] == task) & (df["model"] == model) & (df["model_type"] == mtype)]
                values.append(row[metric].values[0] if len(row) else float("nan"))
        offset = (i - 0.5) * width
        bars = ax.bar(x + offset, values, width, label=mtype, color=PALETTE[mtype], alpha=0.88)
        ax.bar_label(bars, fmt="%.3f", padding=2, fontsize=7)

    # x-tick labels: "task\n(model)"
    tick_labels = [f"{task}\n({model})" for task in tasks for model in models]
    ax.set_xticks(x)
    ax.set_xticklabels(tick_labels, fontsize=8, ha="center")
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_ylim(0, 1.12)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    ax.set_title(f"{ylabel} — Baseline vs Fine-tuned", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    save(fig, filename)


# ── 1 & 2: Accuracy and F1 grouped bar charts ─────────────────────────────────
print("\nGenerating bar charts …")
grouped_bar("accuracy", "Accuracy",       "accuracy_by_task.png")
grouped_bar("f1_macro", "F1 Score (macro)", "f1_by_task.png")


# ── 3: Heatmap — all metrics per (model + type) × task ───────────────────────
print("Generating metrics heatmap …")

df["row_label"] = df["model"] + " · " + df["model_type"]
pivot_dfs = []
for metric in METRICS:
    p = df.pivot_table(index="row_label", columns="task", values=metric, aggfunc="first")
    p.columns = [f"{c}\n({METRIC_LABELS[metric]})" for c in p.columns]
    pivot_dfs.append(p)

heat_df = pd.concat(pivot_dfs, axis=1)
heat_df = heat_df.sort_index()

fig, ax = plt.subplots(figsize=(max(10, len(heat_df.columns) * 1.6), max(4, len(heat_df) * 0.7 + 1.5)))
im = ax.imshow(heat_df.values.astype(float), vmin=0, vmax=1, cmap="RdYlGn", aspect="auto")
plt.colorbar(im, ax=ax, fraction=0.02, pad=0.02)

ax.set_xticks(range(len(heat_df.columns)))
ax.set_xticklabels(heat_df.columns, fontsize=8, rotation=30, ha="right")
ax.set_yticks(range(len(heat_df.index)))
ax.set_yticklabels(heat_df.index, fontsize=9)

for r in range(heat_df.shape[0]):
    for c in range(heat_df.shape[1]):
        val = heat_df.values[r, c]
        if not np.isnan(val):
            ax.text(c, r, f"{val:.3f}", ha="center", va="center",
                    fontsize=8, color="black" if 0.3 < val < 0.85 else "white")

ax.set_title("All Metrics — Model × Task Heatmap", fontsize=13, pad=12)
fig.tight_layout()
save(fig, "metrics_heatmap.png")


# ── 4: Scatter — baseline accuracy vs fine-tuned accuracy ────────────────────
print("Generating baseline vs fine-tuned scatter …")

base_df = df[df["model_type"] == "Baseline"][["model", "task", "accuracy"]].rename(
    columns={"accuracy": "baseline_acc"}
)
ft_df = df[df["model_type"] == "Fine-tuned"][["model", "task", "accuracy"]].rename(
    columns={"accuracy": "finetuned_acc"}
)
scatter_df = base_df.merge(ft_df, on=["model", "task"], how="inner")

if scatter_df.empty:
    print("  [SKIP] Not enough paired data for scatter plot.")
else:
    tasks_list = scatter_df["task"].unique()
    cmap = plt.get_cmap("tab10")
    task_colors = {t: cmap(i) for i, t in enumerate(tasks_list)}

    fig, ax = plt.subplots(figsize=(6, 6))
    for _, row in scatter_df.iterrows():
        ax.scatter(row["baseline_acc"], row["finetuned_acc"],
                   color=task_colors[row["task"]], s=120, zorder=3,
                   label=row["task"])
        ax.annotate(
            f"{row['model']}\n{row['task']}",
            (row["baseline_acc"], row["finetuned_acc"]),
            fontsize=7, textcoords="offset points", xytext=(6, 4)
        )

    lims = [
        min(scatter_df[["baseline_acc", "finetuned_acc"]].min()) - 0.05,
        max(scatter_df[["baseline_acc", "finetuned_acc"]].max()) + 0.05,
    ]
    ax.plot(lims, lims, "k--", alpha=0.4, linewidth=1, label="no change")
    ax.set_xlim(lims); ax.set_ylim(lims)
    ax.set_xlabel("Baseline Accuracy", fontsize=11)
    ax.set_ylabel("Fine-tuned Accuracy", fontsize=11)
    ax.set_title("Fine-tuning Impact on Accuracy", fontsize=13)
    # deduplicate legend entries
    handles, labels = ax.get_legend_handles_labels()
    seen = {}
    for h, l in zip(handles, labels):
        seen.setdefault(l, h)
    ax.legend(seen.values(), seen.keys(), fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    save(fig, "baseline_vs_finetuned.png")


# ── 5: Unknown-prediction rate ────────────────────────────────────────────────
print("Generating unknown-rate chart …")

df["unknown_rate"] = df["n_unknown"] / df["n_examples"]
df["bar_label"]    = df["model"] + "\n" + df["model_type"] + "\n" + df["task"].str[:14]

fig, ax = plt.subplots(figsize=(max(6, len(df) * 0.9), 4))
colors = [PALETTE.get(mt, "#aaaaaa") for mt in df["model_type"]]
bars = ax.bar(range(len(df)), df["unknown_rate"], color=colors, alpha=0.88)
ax.bar_label(bars, fmt="%.3f", padding=2, fontsize=7)
ax.set_xticks(range(len(df)))
ax.set_xticklabels(df["bar_label"], fontsize=7)
ax.set_ylabel("Unknown-prediction rate", fontsize=11)
ax.set_ylim(0, max(df["unknown_rate"].max() * 1.3, 0.05))
ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
ax.set_title("Rate of Unparseable Model Outputs", fontsize=13)
# manual legend
from matplotlib.patches import Patch
ax.legend(handles=[Patch(color=v, label=k) for k, v in PALETTE.items()], fontsize=9)
ax.grid(axis="y", alpha=0.3)
fig.tight_layout()
save(fig, "unknown_rate.png")

print("\nAll graphs saved to", COMBINED_DIR)
