# Use LLM to generate negated hypothesis
import pandas as pd
import os
import ast
from datasets import load_dataset
import numpy as np
from tqdm.notebook import tqdm
from openai import OpenAI
from dotenv import load_dotenv


#NVIDIA appears to be offering generous free tokens.
load_dotenv()
openai_client=OpenAI(base_url="https://integrate.api.nvidia.com/v1",api_key=os.getenv('NVIDIA_API_KEY'))
ds=load_dataset("nyu-mll/multi_nli")
valset=ds['validation_mismatched']

def generate_negated_hypothesis(client,dataset, batch_size=20,fname="/content/drive/MyDrive/negated_validation_set.xlsx"):
  premises=dataset['premise']
  hypotheses=dataset['hypothesis']

  if os.access(fname, os.F_OK):
    existing_df=pd.read_excel(fname)
  else:
    existing_df=pd.DataFrame({"premises":premises,"hypothesis":hypotheses,"negated_hypothesis":np.nan})
  negated_hypothesis=existing_df['negated_hypothesis']

  # Feed the model batches of premises and hypotheses
  for i in tqdm(range(0,len(valset),batch_size),desc="Progress"):
    if not pd.isna(existing_df.loc[i,'negated_hypothesis']):
      print(f"batch {i} is not none; skipping")
      continue
    batch=valset[i:i+batch_size]
    premises=batch['premise']
    hypotheses=batch['hypothesis']
    content=""
    for p in range(len(premises)):
      content+=f"Pair {p}: Premise: {premises[p]}; Hypothesis: {hypotheses[p]}\n"
    #response=client.models.generate_content(model=model,contents=prompt)
    prompt=f"negate the following hypotheses by only changing the verb. return the result in a python list. You must keep the original order. return only the list, nothing else, not even quotes and back ticks.\n {content}"
    #response=call_api(groq_client,prompt)
    completion=client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[
            {"role":"user","content":prompt}
        ],
        temperature=0.2,
        top_p=1,
        max_tokens=6000,
        stream=False
    )
    response=completion.choices[0].message.content
    try:
      new_nh=ast.literal_eval(response)
    except Exception as e:
      print(e)
      continue
    negated_hypothesis[i:i+batch_size]=new_nh
    existing_df['negated_hypothesis']=negated_hypothesis
    existing_df['label']=valset['label']
    existing_df.to_excel(fname,index=False)
generate_negated_hypothesis(openai_client,valset)
