import matplotlib.pyplot as plt
import seaborn as sns

from config import REPORTS_DIR, TARGET
from src.data import clean_dataset, load_dataset


def run_eda() -> None:
    df = clean_dataset(load_dataset())
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    df.describe().to_csv(REPORTS_DIR / "statistical_summary.csv")

    plt.figure(figsize=(8, 5))
    sns.countplot(data=df, x=TARGET)
    plt.title("Heart Disease Target Distribution")
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "target_distribution.png")
    plt.close()

    plt.figure(figsize=(12, 8))
    sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Feature Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "correlation_heatmap.png")
    plt.close()

    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df, x=TARGET, y="thalach")
    plt.title("Maximum Heart Rate by Disease Target")
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "thalach_vs_target.png")
    plt.close()

    plt.figure(figsize=(10, 6))
    sns.histplot(data=df, x="age", hue=TARGET, multiple="stack", bins=20)
    plt.title("Age Distribution by Heart Disease Target")
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "age_distribution.png")
    plt.close()


if __name__ == "__main__":
    run_eda()
    print(f"EDA reports saved to {REPORTS_DIR}")

