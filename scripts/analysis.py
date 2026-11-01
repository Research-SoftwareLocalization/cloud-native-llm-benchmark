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
    plt.title("Reference-Free Semantic Quality (CometKiwi) Across Baseline & Self-Repaired Suites", fontsize=13, fontweight="bold")
    plt.xlabel("Target Language", fontsize=11)
    plt.ylabel("CometKiwi Score", fontsize=11)
    plt.legend(title="Suite", bbox_to_anchor=(1.01, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(reports_dir / "cometkiwi_by_system.png", dpi=300)
    plt.close()
    print("✅ Saved publication figure: reports/cometkiwi_by_system.png")

    summary_path = Path("metrics/cometkiwi/cometkiwi_model_summary.csv")
    if summary_path.exists():
        sum_df = pd.read_csv(summary_path)
        melted = sum_df.melt(
            id_vars=["model"],
            value_vars=["mean_baseline_cometkiwi", "mean_repaired_cometkiwi"],
            var_name="Tier",
            value_name="Mean CometKiwi"
        )
        melted["Tier"] = melted["Tier"].map({
            "mean_baseline_cometkiwi": "Raw Zero-Shot Baseline",
            "mean_repaired_cometkiwi": "Compiler-Guided Self-Repaired"
        })
        plt.figure(figsize=(9, 5), dpi=300)
        sns.barplot(data=melted, x="model", y="Mean CometKiwi", hue="Tier", palette=["#e07a5f", "#3d85c6"])
        plt.title("Mean Reference-Free CometKiwi Score: Baseline vs. Self-Repaired (12 Languages)", fontsize=12, fontweight="bold")
        plt.xlabel("Model Architecture", fontsize=11)
        plt.ylabel("Mean CometKiwi Score", fontsize=11)
        plt.ylim(0, 1.0)
        plt.legend(title="Evaluation Tier", loc="lower right")
        plt.tight_layout()
        plt.savefig(reports_dir / "cometkiwi_before_vs_after.png", dpi=300)
        plt.close()
        print("✅ Saved publication figure: reports/cometkiwi_before_vs_after.png")

if __name__ == "__main__":
    main()
