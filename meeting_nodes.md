model selections

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