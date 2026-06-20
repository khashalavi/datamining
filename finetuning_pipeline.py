import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version (built with): {torch.version.cuda}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")


    # Cell 2: Imports
import os
import json
import torch
import matplotlib.pyplot as plt
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    TrainerCallback,
)

print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"Device: {torch.cuda.get_device_name(0)}")



# Cell 3: Configuration
MODEL_NAME = "Qwen/Qwen3-1.7B"
# MODEL_NAME = "Qwen/Qwen3-0.6B"
OUTPUT_DIR = f"./{MODEL_NAME}_finetuned"
LOG_FILE = f"./{OUTPUT_DIR}/training_losses.txt"
MAX_LENGTH = 128
TRAIN_SAMPLES = 50000   # Adjust based on your GPU/time constraints
VAL_SAMPLES = 5000
BATCH_SIZE = 8
GRAD_ACCUM_STEPS = 4
EPOCHS = 1
LEARNING_RATE = 2e-5

# Cell 4: Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

# Enable gradient checkpointing to save memory
model.gradient_checkpointing_enable()
model.config.use_cache = False

print(f"Model loaded: {MODEL_NAME}")
print(f"Parameters: {model.num_parameters() / 1e6:.1f}M")

# Cell 5: Load and preprocess dataset
dataset = load_dataset("stanfordnlp/snli")
print(dataset)
print(dataset["train"][0])

# Label mapping
LABEL_MAP = {0: "entailment", 1: "neutral", 2: "contradiction"}

def filter_valid(example):
    """Filter out examples with label -1 (no gold label)."""
    return example["label"] != -1

# Filter invalid labels
dataset = dataset.filter(filter_valid)
print(f"\nAfter filtering invalid labels:")
print(f"Train: {len(dataset['train'])}, Validation: {len(dataset['validation'])}")


# Cell 6: Subsample and format dataset
# Subsample for manageable training
train_dataset = dataset["train"].shuffle(seed=42).select(range(min(TRAIN_SAMPLES, len(dataset["train"]))))
val_dataset = dataset["validation"].shuffle(seed=42).select(range(min(VAL_SAMPLES, len(dataset["validation"]))))

def format_and_tokenize(examples):
    """Format into prompt and tokenize."""
    texts = []
    for premise, hypothesis, label in zip(examples["premise"], examples["hypothesis"], examples["label"]):
        label_text = LABEL_MAP[label]
        text = f"Premis: {premise}. Hypothesis: {hypothesis}. Label: {label_text}{tokenizer.eos_token}"
        texts.append(text)
    
    tokenized = tokenizer(
        texts,
        truncation=True,
        max_length=MAX_LENGTH,
        padding="max_length",
    )
    # For causal LM, labels = input_ids (shifted internally by the model)
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized

# Tokenize datasets
train_tokenized = train_dataset.map(
    format_and_tokenize,
    batched=True,
    batch_size=1000,
    remove_columns=train_dataset.column_names,
    desc="Tokenizing train",
)

val_tokenized = val_dataset.map(
    format_and_tokenize,
    batched=True,
    batch_size=1000,
    remove_columns=val_dataset.column_names,
    desc="Tokenizing validation",
)

print(f"Train samples: {len(train_tokenized)}")
print(f"Validation samples: {len(val_tokenized)}")

# Verify a sample
sample_ids = train_tokenized[0]["input_ids"]
print(f"\nSample decoded:\n{tokenizer.decode(sample_ids, skip_special_tokens=False)}")


# Cell 7: Custom callback to log losses to file
class LossLoggerCallback(TrainerCallback):
    def __init__(self, log_file):
        self.log_file = log_file
        self.train_losses = []
        self.eval_losses = []
        self.train_steps = []
        self.eval_steps = []
        
        os.makedirs(os.path.dirname(self.log_file), exist_ok=True)
        # Initialize log file
        with open(self.log_file, "w") as f:
            f.write("=" * 60 + "\n")
            f.write("TRAINING AND VALIDATION LOSS LOG\n")
            f.write("=" * 60 + "\n\n")
    
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return
        
        with open(self.log_file, "a") as f:
            if "loss" in logs:
                step = state.global_step
                loss = logs["loss"]
                self.train_losses.append(loss)
                self.train_steps.append(step)
                f.write(f"[Step {step}] Train Loss: {loss:.4f}\n")
            
            if "eval_loss" in logs:
                step = state.global_step
                eval_loss = logs["eval_loss"]
                self.eval_losses.append(eval_loss)
                self.eval_steps.append(step)
                f.write(f"[Step {step}] Validation Loss: {eval_loss:.4f}\n")
                f.write("-" * 40 + "\n")
    
    def on_train_end(self, args, state, control, **kwargs):
        with open(self.log_file, "a") as f:
            f.write("\n" + "=" * 60 + "\n")
            f.write("TRAINING COMPLETE\n")
            f.write(f"Total steps: {state.global_step}\n")
            if self.train_losses:
                f.write(f"Final train loss: {self.train_losses[-1]:.4f}\n")
            if self.eval_losses:
                f.write(f"Final eval loss: {self.eval_losses[-1]:.4f}\n")
            f.write("=" * 60 + "\n")

loss_logger = LossLoggerCallback(LOG_FILE)


# Cell 8: Training arguments
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM_STEPS,
    learning_rate=LEARNING_RATE,
    weight_decay=0.01,
    warmup_ratio=0.05,
    lr_scheduler_type="cosine",
    logging_steps=50,
    eval_strategy="steps",
    eval_steps=250,
    save_strategy="steps",
    save_steps=500,
    save_total_limit=2,
    bf16=torch.cuda.is_available(),
    report_to="none",
    dataloader_num_workers=0,
    gradient_checkpointing=True,
    optim="adamw_torch",
)

print("Training arguments configured.")


# Cell 9: Initialize Trainer and train
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,  # Causal LM
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_tokenized,
    eval_dataset=val_tokenized,
    data_collator=data_collator,
    callbacks=[loss_logger],
)

print("Starting training...")
train_result = trainer.train()
print("Training complete!")
print(f"Losses saved to: {LOG_FILE}")


# Cell 10: Save final model
trainer.save_model(os.path.join(OUTPUT_DIR, "final_model"))
tokenizer.save_pretrained(os.path.join(OUTPUT_DIR, "final_model"))
print(f"Model saved to: {os.path.join(OUTPUT_DIR, 'final_model')}")


# Cell 11: Visualize training and validation losses
fig, ax = plt.subplots(1, 1, figsize=(12, 6))

# Plot train loss
if loss_logger.train_losses:
    ax.plot(
        loss_logger.train_steps,
        loss_logger.train_losses,
        label="Train Loss",
        color="blue",
        alpha=0.7,
    )

# Plot validation loss
if loss_logger.eval_losses:
    ax.plot(
        loss_logger.eval_steps,
        loss_logger.eval_losses,
        label="Validation Loss",
        color="red",
        marker="o",
        linewidth=2,
    )

ax.set_xlabel("Steps", fontsize=12)
ax.set_ylabel("Loss", fontsize=12)
ax.set_title("Training and Validation Loss - Qwen3-0.6B on SNLI", fontsize=14)
ax.legend(fontsize=12)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("./loss_plot.png", dpi=150, bbox_inches="tight")
plt.show()

print("Plot saved to: ./loss_plot.png")


# Cell 12: Quick inference test
model.config.use_cache = True
model.eval()

test_prompt = "Premis: A man is playing guitar on stage. Hypothesis: A man is performing music. Label:"

inputs = tokenizer(test_prompt, return_tensors="pt").to(model.device)

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=5,
        do_sample=False,
        temperature=1.0,
    )

response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
print(f"Prompt: {test_prompt}")
print(f"Prediction: {response.strip()}")