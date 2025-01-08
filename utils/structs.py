from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import pandas as pd
import time
import os

class Results():
    def __init__(self, filename:str, columns:list=[], sep:str=';', ids:list=[]):
        self.filename = filename
        self.ids = ids

        if not os.path.exists(filename):
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            with open(filename, "w") as f:
                f.write(sep.join(columns) + '\n')

        self.df = pd.read_csv(filename, sep=sep)
        columns = self.df.columns
        self.target = [col[10:] for col in columns if col.startswith('predicted_')]
        self.sep = sep
    
    def add(self, data:dict):
        existing = self.get(*[data[id_col] for id_col in self.ids])
        if existing is None:
            new_row = pd.DataFrame([data])
            new_row.to_csv(self.filename, sep=self.sep, mode='a', header=False, index=False)
            self.df = pd.concat([self.df, new_row], ignore_index=True)
        else:
            mask = True
            for id_col in self.ids:
                mask &= (self.df[id_col] == data[id_col])
            for col, val in data.items():
                self.df.loc[mask, col] = val
            self.df.to_csv(self.filename, sep=self.sep, index=False)

    def get(self, *ids_values):
        mask = True
        for col, val in zip(self.ids, ids_values):
            mask &= (self.df[col] == val)
        result = self.df[mask]
        if result.empty:
            return None
        return result.iloc[0].to_dict()
    
    def display_score(self, logfile:str=None):
        y = self.df[self.target]
        y_pred = self.df[[f'predicted_{col}' for col in self.target]]

        for tg in self.target:
            txt = (f"----------- {tg} -----------\n")
            txt += (f"Confusion Matrix:\n{confusion_matrix(y[tg], y_pred[f'predicted_{tg}'])}")
            txt += (f"\n{classification_report(y[tg], y_pred[f'predicted_{tg}'], zero_division=0)}")
            txt += (f"\nAccuracy: {accuracy_score(y[tg], y_pred[f'predicted_{tg}'])}\n")
            if logfile:
                with open(logfile, 'a') as f:
                    txt = (f"[{time.strftime('%Y-%m-%d %H:%M:%S')}]:\n{txt}")
                    f.write(txt)
            print(txt)
    
    def get_macro_score(self):
        y = self.df[self.target]
        y_pred = self.df[[f'predicted_{col}' for col in self.target]]

        for tg in self.target:
            y_pred.loc[:, f'predicted_{tg}'] = y_pred[f'predicted_{tg}'].fillna(y_pred[f'predicted_{tg}'].mode()[0])

        macro = {}
        for tg in self.target:
            report = classification_report(y[tg], y_pred[f'predicted_{tg}'], zero_division=0, output_dict=True)
            macro[tg] = {
                "precision": round(report["macro avg"]["precision"], 2),
                "recall": round(report["macro avg"]["recall"], 2),
                "f1-score": round(report["macro avg"]["f1-score"], 2),
                "accuracy": round(accuracy_score(y[tg], y_pred[f'predicted_{tg}']), 2)
            }
        return macro

    def get_tokens(self):
        return {
            "input": self.df['input_tokens'].sum(),
            "output": self.df['output_tokens'].sum()
        }

if __name__ == "__main__":
    results = Results(
        "results/sample.csv", 
        columns=["article_id","claim_id","article","claim"],
        sep=';',
        ids=["article_id","claim_id"]
    )
    results.add({
        "article_id": 1,
        "claim_id": 1,
        "article": "article 1",
        "claim": "claim 1"
    })
    results.add({
        "article_id": 1,
        "claim_id": 2,
        "article": "article 1",
        "claim": "claim 2"
    })
    print(results.get(1, 2))
    print(results.get(1, 3))
