import pandas as pd
from pathlib import Path

def main():
    res_dir = Path("metrics/results")
    lex_df = pd.read_csv(res_dir / "metric_scores_lexical.csv")
    neu_df = pd.read_csv(res_dir / "metric_scores_neural.csv")

    merged = pd.merge(lex_df, neu_df, on=["system", "language"], how="outer")
    merged.sort_values(by=["system", "language"], inplace=True)
    merged.to_csv(res_dir / "metric_scores.csv", index=False)
    print(f"Merged into {res_dir / 'metric_scores.csv'} ({len(merged)} rows)")

if __name__ == "__main__":
    main()