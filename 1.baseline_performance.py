import os
import torch
import pandas as pd
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")


# Configuration — change MODEL_NAME to evaluate a different base model
# MODEL_NAME = "Qwen/Qwen3-1.7B"
MODEL_NAME = "Qwen/Qwen3-0.6B"
MODEL_SLUG = MODEL_NAME.split("/")[-1]
MAX_LENGTH = 128
INFER_BATCH_SIZE = 16

LABEL_MAP = {0: "entailment", 1: "neutral", 2: "contradiction"}
VALID_LABELS = set(LABEL_MAP.values())

results_dir = f"./results/{MODEL_SLUG}"
os.makedirs(results_dir, exist_ok=True)
output_path = os.path.join(results_dir, "baseline.xlsx")


# Load base model and tokenizer (no fine-tuning)
print(f"\nLoading base model: {MODEL_NAME}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"  # required for batched causal LM inference

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
model.eval()
model.config.use_cache = True
print(f"Model loaded: {model.num_parameters() / 1e6:.1f}M parameters")


# Load SNLI test set (filter invalid labels)
print("\nLoading SNLI test set...")
dataset = load_dataset("stanfordnlp/snli", split="test")
dataset = dataset.filter(lambda x: x["label"] != -1)
print(f"Test examples: {len(dataset)}")


def predict_batch(premises, hypotheses):
    prompts = [
        f"Answer with exactly one word: 'entailment' (the premise proves the hypothesis is true), 'contradiction' (the premise proves the hypothesis is false), or 'neutral' (there is no definitive proof either way).\n\n"
        f"Premise: {p}\n"
        f"Hypothesis: {h}\n"
        f"Label: "  
        for p, h in zip(premises, hypotheses)
    ]
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
            max_new_tokens=15,  
            do_sample=False,
            temperature=1.0,
            pad_token_id=tokenizer.pad_token_id,
        )

    preds = []
    for i, seq in enumerate(out):
        new_tokens = seq[inputs["input_ids"].shape[1]:]
        text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip().lower()
        
        if i == 0: 
            print(f"\n[DEBUG] Raw output: '{text}'")

        # Smart parsing for base model hallucinations
        if "entail" in text:  # Catches entailment, entailing, entailed
            preds.append("entailment")
        elif "contradict" in text or "contr" in text: # Catches contradiction, contr
            preds.append("contradiction")
        elif "neutral" in text:
            preds.append("neutral")
        else:
            preds.append("unknown")
            
    return preds


# Run inference over full test set
rows = []
n = len(dataset)
print(f"\nRunning inference on {n} test examples (batch size {INFER_BATCH_SIZE})...")

for start in range(0, n, INFER_BATCH_SIZE):
    end = min(start + INFER_BATCH_SIZE, n)
    batch = dataset.select(range(start, end))
    premises = batch["premise"]
    hypotheses = batch["hypothesis"]
    correct_labels = [LABEL_MAP[l] for l in batch["label"]]
    predicted_labels = predict_batch(premises, hypotheses)
    for p, h, correct, pred in zip(premises, hypotheses, correct_labels, predicted_labels):
        rows.append({
            "premise": p,
            "hypothesis": h,
            "correct_label": correct,
            "predicted_label": pred,
        })
    if (start // INFER_BATCH_SIZE) % 20 == 0:
        print(f"  {end}/{n} done...")

df = pd.DataFrame(rows)
accuracy = (df["correct_label"] == df["predicted_label"]).mean()
unknown_rate = (df["predicted_label"] == "unknown").mean()

print(f"\nResults for {MODEL_SLUG} (base, no fine-tuning):")
print(f"  Accuracy:     {accuracy:.4f} ({accuracy * 100:.2f}%)")
print(f"  Unknown rate: {unknown_rate:.4f} ({unknown_rate * 100:.2f}%)")
print(f"\nPer-label breakdown:")
for label in VALID_LABELS:
    subset = df[df["correct_label"] == label]
    label_acc = (subset["correct_label"] == subset["predicted_label"]).mean()
    print(f"  {label:15s}: {label_acc:.4f} ({len(subset)} examples)")

df.to_excel(output_path, index=False)
print(f"\nSaved to: {output_path}")
