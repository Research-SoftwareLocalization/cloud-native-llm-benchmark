import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path

def main():
    df = pd.read_csv("metrics/results/metric_scores.csv")
    kiwi_path = Path("metrics/cometkiwi/cometkiwi_scores.csv")
    if kiwi_path.exists():
        kiwi_df = pd.read_csv(kiwi_path)
        df = pd.merge(df, kiwi_df, on=["system", "language"], how="left")

    # 1. Generate comet_by_system.png
    plt.figure(figsize=(12, 6), dpi=300)
    sns.barplot(data=df, x="language", y="COMET", hue="system", palette="viridis")
    plt.title("COMET Scores by LLM System and Target Language", fontsize=14, fontweight="bold")
    plt.xlabel("Target Language", fontsize=12)
    plt.ylabel("COMET Score", fontsize=12)
    plt.legend(title="LLM System", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig("comet_by_system.png", dpi=300)
    plt.close()
    print("Saved comet_by_system.png")

    # 2. Generate metric_correlation.png
    metric_cols = [c for c in ["BLEU", "chrF", "METEOR", "ROUGE_L", "TER", "BERTScore", "COMET", "COMETKiwi"] if c in df.columns]
    corr = df[metric_cols].corr(method="spearman")

    plt.figure(figsize=(8, 6), dpi=300)
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1, square=True)
    plt.title("Spearman Correlation Across Evaluation Metrics (LLM Benchmark)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig("metric_correlation.png", dpi=300)
    plt.close()
    print("Saved metric_correlation.png")

if __name__ == "__main__":
    main()