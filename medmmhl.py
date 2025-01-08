# %%
from dotenv import load_dotenv
from utils import LLM, Results, extract_json, normalize_string
import pandas as pd
from tqdm import tqdm
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-m', type=str, required=True, help="LLM to use")
parser.add_argument('-v', type=str, default='1', help="Execution version")
args = parser.parse_args()

load_dotenv()

llm = LLM(
    os.getenv('OPENAI_API_KEY'),
    os.getenv('TOGETHER_API_KEY')
)

# %%
dataset_file = "./data/processed/medmmhl.csv"
df = pd.read_csv(dataset_file)
LIMIT = 2000 # remove this line to process the entire dataset
df = df.head(LIMIT)
df.loc[0]

# %%
PROMPT = """\
ARTICLE/CLAIM:
{article}
"""

SYSTEM_PROMPT = """\
Answer in a JSON format with the following structure:
{
  "stance": "fake/real (whether the claim is fake or real)",
  "stance_reason": "explanation of the stance of the article towards the claim"
}
"""

VERSION_PROMPT = args.v
MODEL = args.m

print(f"Executing medmmhl-v{VERSION_PROMPT} with model {MODEL}")

# %%
results_file = f"./results/medmmhl-v{VERSION_PROMPT}_{normalize_string(MODEL)}.csv"

results = Results(
    results_file,
    columns=["article_id","article","stance","model","predicted_stance","stance_reason","input_tokens","output_tokens"],
    sep=";",
    ids=["article_id"]
)
print(f"{len(results.df)} results collected")

# %%
for idx, row in tqdm(df.iterrows(), total=len(df)):
    article_id = row["id"]
    if results.get(article_id):
        continue

    article = row["content"]
    
    prompt = PROMPT.format(article=article)
    try:
        result_txt, usage = llm(prompt, MODEL, SYSTEM_PROMPT, track_usage=True)
        result = extract_json(result_txt)
        stance = result["stance"]
        stance_reason = result["stance_reason"]
    except Exception as e:
        prompt = f"{SYSTEM_PROMPT}\n\n{prompt}"
        result_txt, usage_2 = llm(prompt, MODEL, track_usage=True, max_retries=10, retry_delay=15)
        try:
            usage["input"] += usage_2["input"]
            usage["output"] += usage_2["output"]
        except:
            usage = usage_2
        try:
            result = extract_json(result_txt)
            stance = result["stance"]
            stance_reason = result["stance_reason"]
        except:
            result = {"stance": "error", "stance_reason": str(e)}

    results.add({
        "article_id": article_id,
        "article": article,
        "stance": "fake" if row["det_fake_label"] == 1 else "real",
        "model": MODEL, 
        "predicted_stance": result["stance"],
        "stance_reason": result["stance_reason"],
        "input_tokens": usage["input"],
        "output_tokens": usage["output"]
    })

# %%
results.display_score()