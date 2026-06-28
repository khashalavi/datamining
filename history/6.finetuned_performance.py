import os
import torch
import pandas as pd
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

# 1. Load your ALREADY FINETUNED model and tokenizer
MODEL_DIR = "./models/Qwen3-0.6B/final_model"
MODEL_SLUG = "Qwen3-0.6B"
MAX_LENGTH = 128

print(f"Loading finetuned model from {MODEL_DIR}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Use left-padding for batch inference with causal LM
tokenizer.padding_side = "left"

model = AutoModelForCausalLM.from_pretrained(
    MODEL_DIR,
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
model.config.use_cache = True
model.eval()

# 2. Load the dataset again
dataset = load_dataset("stanfordnlp/snli")
test_dataset = dataset["test"]

# Filter out invalid labels (-1) if they exist in the test set
def filter_valid(example):
    return example["label"] != -1
test_dataset = test_dataset.filter(filter_valid)

LABEL_MAP = {0: "entailment", 1: "neutral", 2: "contradiction"}
VALID_LABELS = {"entailment", "neutral", "contradiction"}
INFER_BATCH_SIZE = 16

# 3. Define the prediction function
def predict_batch(premises, hypotheses):
    prompts = [
        f"Premis: {p}. Hypothesis: {h}. Label:"
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
            max_new_tokens=5,
            do_sample=False,
            temperature=1.0,
            pad_token_id=tokenizer.pad_token_id,
        )

    preds = []
    for seq in out:
        new_tokens = seq[inputs["input_ids"].shape[1]:]
        text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip().lower()
        first_word = text.split()[0] if text.split() else ""
        if first_word in VALID_LABELS:
            preds.append(first_word)
        else:
            matched = next((l for l in VALID_LABELS if l in text), "unknown")
            preds.append(matched)
    return preds

# 4. Run inference
rows = []
n = len(test_dataset)
print(f"Running inference on {n} test examples...")
for start in range(0, n, INFER_BATCH_SIZE):
    end = min(start + INFER_BATCH_SIZE, n)
    batch = test_dataset.select(range(start, end))
    premises = batch["premise"]
    hypotheses = batch["hypothesis"]
    labels = [LABEL_MAP[l] for l in batch["label"]]
    
    preds = predict_batch(premises, hypotheses)
    
    for p, h, lbl, pred in zip(premises, hypotheses, labels, preds):
        rows.append({"premise": p, "hypothesis": h, "correct_label": lbl, "predicted_label": pred})
        
    if (start // INFER_BATCH_SIZE) % 20 == 0:
        print(f"  {end}/{n} done...")

# 5. Save the results (This will work now!)
df = pd.DataFrame(rows)
accuracy = (df["correct_label"] == df["predicted_label"]).mean()
print(f"Test accuracy: {accuracy:.4f} ({(accuracy*100):.2f}%)")

results_dir = f"./results/{MODEL_SLUG}"
os.makedirs(results_dir, exist_ok=True)
results_path = os.path.join(results_dir, "finetuned_predictions.xlsx")

df.to_excel(results_path, index=False)
print(f"Predictions saved to: {results_path}")