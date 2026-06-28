"""
Bonus Task 2.1 — Cross-Genre Generalization
Evaluates the base model and the fine-tuned model on the Multi-NLI
validation_mismatched split and saves per-example results to Excel.

Outputs:
  ./results/<MODEL_SLUG>/generalization-baseline.xlsx
  ./results/<MODEL_SLUG>/generalization-finetuned.xlsx
"""

import os
import re
import torch
import pandas as pd
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")


# ── Configuration ─────────────────────────────────────────────────────────────
BASE_MODEL_NAME   = "Qwen/Qwen3-0.6B"
MODEL_SLUG        = BASE_MODEL_NAME.split("/")[-1]       # e.g. "Qwen3-0.6B"
FINETUNED_DIR     = f"./models/{MODEL_SLUG}/final_model"

MAX_LENGTH        = 256  # Increased to 256 to fit baseline instructions
INFER_BATCH_SIZE  = 16
NUM_SAMPLES       = None   # set to an int (e.g. 500) to limit; None = full split

DATA_DIR          = "./data"
RESULTS_DIR       = f"./results/{MODEL_SLUG}"
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

LABEL_MAP   = {0: "entailment", 1: "neutral", 2: "contradiction"}
VALID_LABELS = set(LABEL_MAP.values())


# ── Dataset ───────────────────────────────────────────────────────────────────
mnli_cache = os.path.join(DATA_DIR, "multi_nli_val_mismatched")
if os.path.exists(mnli_cache):
    from datasets import load_from_disk
    val_set = load_from_disk(mnli_cache)
    print(f"Loaded Multi-NLI validation_mismatched from cache ({len(val_set)} examples)")
else:
    print("Downloading Multi-NLI …")
    mnli = load_dataset("nyu-mll/multi_nli")
    val_set = mnli["validation_mismatched"]
    val_set.save_to_disk(mnli_cache)
    print(f"Saved to {mnli_cache} ({len(val_set)} examples)")

# Multi-NLI labels: 0=entailment, 1=neutral, 2=contradiction (same as SNLI)
val_set = val_set.filter(lambda x: x["label"] != -1)
if NUM_SAMPLES:
    val_set = val_set.shuffle(seed=42).select(range(min(NUM_SAMPLES, len(val_set))))
print(f"Evaluating on {len(val_set)} examples")


# ── Inference helpers ─────────────────────────────────────────────────────────
def load_model_and_tokenizer(model_path_or_name: str):
    print(f"\nLoading model: {model_path_or_name}")
    tok = AutoTokenizer.from_pretrained(model_path_or_name, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    mdl = AutoModelForCausalLM.from_pretrained(
        model_path_or_name,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    mdl.eval()
    mdl.config.use_cache = True
    print(f"  Parameters: {mdl.num_parameters() / 1e6:.1f}M")
    return mdl, tok


def predict_batch(model, tokenizer, premises, hypotheses, is_baseline=False):
    # Branch prompt formats based on model type
    if is_baseline:
        prompts = [
            f"Answer with exactly one word: 'entailment' (the premise proves the hypothesis is true), 'contradiction' (the premise proves the hypothesis is false), or 'neutral' (there is no definitive proof either way).\n\n"
            f"Premise: {p}\n"
            f"Hypothesis: {h}\n"
            f"Label: "
            for p, h in zip(premises, hypotheses)
        ]
        max_tokens = 15
    else:
        prompts = [
            f"Premis: {p}. Hypothesis: {h}. Label:"
            for p, h in zip(premises, hypotheses)
        ]
        max_tokens = 5

    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=MAX_LENGTH,
    ).to(model.device)

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=False,
            temperature=1.0,
            pad_token_id=tokenizer.pad_token_id,
        )

    preds = []
    for seq in out:
        new_tokens = seq[inputs["input_ids"].shape[1]:]
        text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip().lower()
        
        # Branch parsing logic based on model type
        if is_baseline:
            if "entail" in text:
                preds.append("entailment")
            elif "contradict" in text or "contr" in text:
                preds.append("contradiction")
            elif "neutral" in text:
                preds.append("neutral")
            else:
                preds.append("unknown")
        else:
            first_word = text.split()[0] if text.split() else ""
            if first_word in VALID_LABELS:
                preds.append(first_word)
            else:
                matched = next((l for l in VALID_LABELS if l in text), "unknown")
                preds.append(matched)
                
    return preds


def run_evaluation(model, tokenizer, dataset, label: str) -> pd.DataFrame:
    rows = []
    n = len(dataset)
    is_baseline = (label == "baseline")
    
    print(f"\nEvaluating [{label}] on {n} examples …")
    for start in range(0, n, INFER_BATCH_SIZE):
        end = min(start + INFER_BATCH_SIZE, n)
        batch = dataset.select(range(start, end))
        premises   = batch["premise"]
        hypotheses = batch["hypothesis"]
        genres     = batch["genre"] if "genre" in batch.column_names else [""] * (end - start)
        correct    = [LABEL_MAP[l] for l in batch["label"]]
        
        predicted  = predict_batch(model, tokenizer, premises, hypotheses, is_baseline=is_baseline)
        
        for p, h, g, c, pr in zip(premises, hypotheses, genres, correct, predicted):
            rows.append({
                "premise":         p,
                "hypothesis":      h,
                "genre":           g,
                "correct_label":   c,
                "predicted_label": pr,
            })
        if (start // INFER_BATCH_SIZE) % 20 == 0:
            print(f"  {end}/{n} done …")
            
    df = pd.DataFrame(rows)
    acc = (df["correct_label"] == df["predicted_label"]).mean()
    unknown_rate = (df["predicted_label"] == "unknown").mean()
    
    print(f"  Accuracy: {acc:.4f} ({acc * 100:.2f}%)")
    print(f"  Unknown rate: {unknown_rate:.4f} ({unknown_rate * 100:.2f}%)")
    return df


# ── Baseline (base model, no fine-tuning) ─────────────────────────────────────
base_model, base_tok = load_model_and_tokenizer(BASE_MODEL_NAME)
df_base = run_evaluation(base_model, base_tok, val_set, "baseline")
out_base = os.path.join(RESULTS_DIR, "generalization-baseline.xlsx")
df_base.to_excel(out_base, index=False)
print(f"Saved: {out_base}")

# Free GPU memory before loading the second model
del base_model
torch.cuda.empty_cache()


# ── Fine-tuned model ──────────────────────────────────────────────────────────
if not os.path.exists(FINETUNED_DIR):
    print(f"\nFine-tuned model not found at {FINETUNED_DIR} — skipping.")
else:
    ft_model, ft_tok = load_model_and_tokenizer(FINETUNED_DIR)
    df_ft = run_evaluation(ft_model, ft_tok, val_set, "finetuned")
    out_ft = os.path.join(RESULTS_DIR, "generalization-finetuned.xlsx")
    df_ft.to_excel(out_ft, index=False)
    print(f"Saved: {out_ft}")
    del ft_model
    torch.cuda.empty_cache()

print("\nDone.")