import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path

def main():
    reports_dir = Path("reports")
    reports_dir.mkdir(parents=True, exist_ok=True)
    kiwi_path = Path("metrics/cometkiwi/cometkiwi_scores.csv")
    if not kiwi_path.exists():
        print("⚠️ metrics/cometkiwi/cometkiwi_scores.csv not found yet.")
        return

    df = pd.read_csv(kiwi_path)
    plt.figure(figsize=(14, 6), dpi=300)
    sns.barplot(data=df, x="language", y="cometkiwi", hue="system", palette="viridis")
    plt.title("Reference-Free Semantic Quality (CometKiwi) Across Baseline & Repaired Suites", fontsize=13, fontweight="bold")
    plt.xlabel("Target Language", fontsize=11)
    plt.ylabel("CometKiwi Score", fontsize=11)
    plt.legend(title="Suite", bbox_to_anchor=(1.01, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(reports_dir / "cometkiwi_by_system.png", dpi=300)
    plt.close()
    print("✅ Saved publication figure: reports/cometkiwi_by_system.png")

if __name__ == "__main__":
    main()
