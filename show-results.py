from collections import defaultdict
from utils import Results
import pandas as pd
import os

board = defaultdict(lambda: defaultdict(dict))

prices = {
    "gpt-4o": {"input":2.5, "output":10},
    "meta-llama/Llama-3.3-70B-Instruct-Turbo": {"input":0.88, "output":0.88},
    "meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo": {"input":3.5, "output":3.5},
    "nvidia/Llama-3.1-Nemotron-70B-Instruct-HF": {"input":0.88, "output":0.88},
    "Qwen/QwQ-32B-Preview": {"input":1.2, "output":1.2},
}

files = os.listdir('results')
files.sort()
for fpath in files:
    if not fpath.endswith('.csv'):
        continue
    dataset = fpath.split('-')[0]
    ids = ["article_id"] if dataset == "medmmhl" else ["article_id","claim_id"]
    results = Results(
        f"./results/{fpath}",
        sep=';',
        ids=ids
    )
    model = results.df.loc[0].model
    macro = results.get_macro_score()
    tokens = results.get_tokens()
    cost = round((tokens["input"] * prices[model]["input"] + tokens["output"] * prices[model]["output"])/1e6, 2)
    for target in macro:
        macro[target]["cost"] = float(cost)
        macro[target]["# executions"] = len(results.df)
        board[dataset][target][model] = macro[target]

    # save results.df in a ./results/excel folder
    #results.df.to_excel(f"./results/excel/{fpath.replace('csv','xlsx')}", index=False)

for category, subcategories in board.items():
    for subcategory, models in subcategories.items():
        # Convert to DataFrame
        df = pd.DataFrame.from_dict(models, orient='index')
        df.index.name = 'Model'
        df.reset_index(inplace=True)
        print(f"\nTable: {category}-{subcategory}")
        print(df.to_string(index=False))
