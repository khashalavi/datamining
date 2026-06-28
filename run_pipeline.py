"""
run_pipeline.py — Run the full NLI pipeline end-to-end.

Set MODEL_NAME below and run:
    python run_pipeline.py

Steps executed in order:
  0. finetuning_pipeline       — fine-tune the model on SNLI
  1. baseline_performance      — zero-shot base model on SNLI test set
  2. bonus-generalization      — base + fine-tuned on Multi-NLI
  3. bonus-negation            — base + fine-tuned on negated hypotheses
  4. combine_results           — aggregate all metrics into final_table
  5. generate_graphs           — produce all visualisation charts

Skips step 3 automatically if the negated dataset is not found.
"""

import os
import re
import sys
import time
import tempfile
import subprocess

# ── The only setting you need to change ──────────────────────────────────────
MODEL_NAME = "Qwen/Qwen3-0.6B"
# MODEL_NAME = "Qwen/Qwen3-1.7B"
# ─────────────────────────────────────────────────────────────────────────────

SCRIPTS = [
    "0.finetuning_pipeline.py",
    "1.baseline_performance.py",
    "2.1-bonus-generalization.py",
    "2.2-bonus-negation.py",
    "3.combine_results.py",
    "4.generate_graphs.py",
]

# Patterns to rewrite in each child script so they pick up MODEL_NAME
REPLACEMENTS = [
    # non-commented MODEL_NAME = "..."
    (r'^(MODEL_NAME\s*=\s*)["\'].*?["\']', rf'\g<1>"{MODEL_NAME}"'),
    # non-commented BASE_MODEL_NAME = "..."
    (r'^(BASE_MODEL_NAME\s*=\s*)["\'].*?["\']', rf'\g<1>"{MODEL_NAME}"'),
]

NEGATED_DATASET_PATHS = [
    "./data/negated_validation_set.xlsx",
    "./negated_validation_set.xlsx",
]


def inject_model_name(source: str) -> str:
    """Replace MODEL_NAME / BASE_MODEL_NAME assignments in script source."""
    for pattern, replacement in REPLACEMENTS:
        source = re.sub(pattern, replacement, source, flags=re.MULTILINE)
    return source


def run_script(script_path: str) -> bool:
    """Patch model name, write to a temp file, run as subprocess. Returns True on success."""
    with open(script_path, "r", encoding="utf-8") as f:
        source = f.read()

    patched = inject_model_name(source)

    # Write patched source to a temp file so there are no arg-length limits
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False, encoding="utf-8"
    ) as tmp:
        tmp.write(patched)
        tmp_path = tmp.name

    try:
        result = subprocess.run(
            [sys.executable, tmp_path],
            check=False,  # we handle the return code ourselves
        )
        return result.returncode == 0
    finally:
        os.unlink(tmp_path)


def hms(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def negated_dataset_exists() -> bool:
    return any(os.path.exists(p) for p in NEGATED_DATASET_PATHS)


# ── Run ───────────────────────────────────────────────────────────────────────
slug = MODEL_NAME.split("/")[-1]
print("=" * 60)
print(f"  NLI Pipeline")
print(f"  Model : {MODEL_NAME}")
print(f"  Slug  : {slug}")
print("=" * 60)

pipeline_start = time.time()
step_results = []   # (script, status, duration)

for script in SCRIPTS:
    # Guard: skip negation script if the dataset is missing
    if script == "2.2-bonus-negation.py" and not negated_dataset_exists():
        print(f"\n[SKIP] {script}")
        print(
            "  Negated dataset not found in ./data/ or project root.\n"
            "  Run negation_set_generation.py first (requires NVIDIA_API_KEY)."
        )
        step_results.append((script, "skipped", 0.0))
        continue

    print(f"\n{'─' * 60}")
    print(f"[RUN ] {script}")
    print(f"{'─' * 60}")

    t0 = time.time()
    ok = run_script(script)
    elapsed = time.time() - t0

    status = "ok" if ok else "FAILED"
    step_results.append((script, status, elapsed))
    print(f"\n[{'OK  ' if ok else 'FAIL'}] {script}  ({hms(elapsed)})")

    if not ok:
        print("\n  Script exited with a non-zero return code.")
        answer = input("  Continue with the next step? [y/N] ").strip().lower()
        if answer != "y":
            print("\nPipeline aborted.")
            break

# ── Summary ───────────────────────────────────────────────────────────────────
total = time.time() - pipeline_start
print(f"\n{'=' * 60}")
print(f"  PIPELINE SUMMARY  (total: {hms(total)})")
print(f"{'=' * 60}")
col_w = max(len(s) for s, _, _ in step_results)
for script, status, dur in step_results:
    icon = {"ok": "✓", "FAILED": "✗", "skipped": "–"}.get(status, "?")
    dur_str = hms(dur) if dur else "      "
    print(f"  {icon}  {script:<{col_w}}  {status:<7}  {dur_str}")
print()
