from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_dataset
import torch
import evaluate
from tqdm.notebook import tqdm
import numpy as np

@torch.no_grad()
def evaluate_performance(model,dataset, batch_size=4):
  # Batch evaluation on NLI datasets such as Multi-NLI dataset.
  # Params:
  # model: Sequence Classification Model
  # dataset: huggingface datset
  # batch_size: how many samples to feed the model at once.
  metric = evaluate.load('f1')
  predictions = []
  labels = dataset['label']
  for i in tqdm(range(0,len(dataset),batch_size),desc="progress"):
    batch = dataset[i:i+batch_size]
    inputs = tokenizer(list(batch['premise']),list(batch['hypothesis']),return_tensors='pt',padding=True,truncation=True)
    inputs = {k:v.to(model.device) for k,v in inputs.items()}
    logits = model(**inputs).logits
    ids = logits.argmax(-1)
    predictions.extend(ids.cpu().tolist())
  accuracy = np.sum(np.array(predictions)==np.array(labels))/len(labels)
  macro_f1 = metric.compute(predictions=predictions,references=labels,average='macro')
  return accuracy, macro_f1

if __name__=='__main__':
  model_name = "facebook/bart-large-mnli"
  tokenizer = AutoTokenizer.from_pretrained(model_name)
  model = AutoModelForSequenceClassification.from_pretrained(model_name)
  ds = load_dataset("nyu-mll/multi_nli")
  device = "cuda" if torch.cuda.is_available() else "cpu"
  model.to(device)

  validation_mismatched=ds['validation_mismatched']
  accuracy, macro_f1=evaluate_performance(model,validation_mismatched.select(range(100)))
  print(f"Accuracy: {accuracy}; Macro F1 score: {macro_f1}")
