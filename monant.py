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
dataset_file = "./data/processed/monant.csv"
df = pd.read_csv(dataset_file)
LIMIT = 500 # remove this line to process the entire dataset
df = df.head(LIMIT)
df.loc[0]

# %%
PROMPT = """\
ARTICLE:
{article}

CLAIM:
{claim}
"""

SYSTEM_PROMPT = """\
Answer in a JSON format with the following structure:
{
  "presence": "yes/no (whether the claim is discussed in the article)",
  "stance": "supporting/contradicting/neutral (article stance towards the claim)",
  "stance_reason": "explanation of the stance of the article towards the claim"
}

- The ONLY CASE WHERE stance should be 'neutral' is if the topic is completely unrelated to the claim; otherwise, it should be 'supporting' or 'contradicting'.
"""

# changing these lines will change the results file:
VERSION_PROMPT = args.v
MODEL = args.m

print(f"Executing monant-v{VERSION_PROMPT} with model {MODEL}")

# %%
results_file = f"./results/monant-v{VERSION_PROMPT}_{normalize_string(MODEL)}.csv"

results = Results(
    results_file,
    columns=["article_id","claim_id","article","claim","presence","stance","model","predicted_presence","predicted_stance","stance_reason","input_tokens","output_tokens"],
    sep=";",
    ids=["article_id","claim_id"]
)
print(f"{len(results.df)} results collected")

# %%
for idx, row in tqdm(df.iterrows(), total=len(df)):
    article_id = row["articles_id"]
    claim_id = row["claims_id"]
    if results.get(article_id, claim_id):
        continue

    article_title = row["articles_title"].strip() if isinstance(row["articles_title"], str) else ""
    article_body = row["articles_body"].strip() if isinstance(row["articles_body"], str) else ""
    article_date = row["articles_published_at"].split()[0] if isinstance(row["articles_published_at"], str) else ""
    claim_statement = row["claims_statement"].strip() if isinstance(row["claims_statement"], str) else ""
    claim_description = row["claims_description"].strip() if isinstance(row["claims_description"], str) else ""
    claim_date = row["claims_created_at"].split()[0] if isinstance(row["claims_created_at"], str) else ""

    article = f"({article_date})\n# {article_title}\n\n{article_body}"
    claim = f"({claim_date})\n# {claim_statement}\n\n{claim_description}"
    
    prompt = PROMPT.format(article=article, claim=claim)
    try:
        result_txt, usage = llm(prompt, MODEL, SYSTEM_PROMPT, track_usage=True)
        result = extract_json(result_txt)
        presence = result["presence"]
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
            presence = result["presence"]
            stance = result["stance"]
            stance_reason = result["stance_reason"]
        except:
            result = {"presence": "error", "stance": "error", "stance_reason": str(e)}

    results.add({
        "article_id": article_id,
        "claim_id": claim_id,
        "article": article,
        "claim": claim,
        "presence": row["presence"],
        "stance": row["stance"],
        "model": MODEL, 
        "predicted_presence": result["presence"],
        "predicted_stance": result["stance"],
        "stance_reason": result["stance_reason"],
        "input_tokens": usage["input"],
        "output_tokens": usage["output"]
    })

# %%
results.display_score()