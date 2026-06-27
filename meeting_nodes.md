model selections

qwen 3-0.6B

- qwen/qwen3.5-2b
- qwen/qwen3.5-2b-base
- google/gemma-4-2B
- google/gemma-4-2B-it

- meta-llama/llam-3.2-1B
- meta-llama/llam-3.2-1B-Instruct

----

finetuning via Lora
- --> GPT write code 


Prompt: 
Premis:{Premis}. Hypothesis {Hypothesis}. Label: {label}


Inference: Premis:{Premis}. Hypothesis {Hypothesis}. Label: _____
outputs: actual label pos, neg, neutral 
<!--  -->
Dataset-link: https://huggingface.co/datasets/stanfordnlp/snli


------
Bonus task



-------

Prompt LLM: 

I want to use hugginface model Qwen/Qwen3-0.6B and use the dataset https://huggingface.co/datasets/stanfordnlp/snli
and use huggingface libary to train on this dataset with the prompt: Premis:{Premis}. Hypothesis {Hypothesis}. Label: {label}

Training dataset is: """
from datasets import load_dataset

dataset = load_dataset("stanfordnlp/snli")
print(dataset)
print(dataset["train"][0])"""

write me a python pipeline, that i can execute in jupyter notebook, to finetune this model on this dataset. During finetuning i want to save the training and validation losses in a .txt file of the folder and at the end to also visualize the training and validation loss. 
