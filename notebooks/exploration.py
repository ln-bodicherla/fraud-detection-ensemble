"""
Exploratory data analysis for insurance claims fraud detection.

Script-style exploration showing data distributions, correlation analysis,
class balance assessment, and feature relationships. Generates visualizations
for understanding the claims dataset before modeling.
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data_preprocessing import DataPreprocessor


def main():
    """Run full exploratory data analysis."""
    output_dir = Path("reports/exploration")
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------------
    # 1. Generate and load data
    # ---------------------------------------------------------------
    print("Generating synthetic claims data...")
    preprocessor = DataPreprocessor(random_state=42)
    df = preprocessor.generate_synthetic_data(n_samples=50000, fraud_rate=0.05)

    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print(f"\nFirst 5 rows:")
    print(df.head())
    print(f"\nData types:")
    print(df.dtypes)
    print(f"\nBasic statistics:")
    print(df.describe())

    # ---------------------------------------------------------------
    # 2. Class distribution
    # ---------------------------------------------------------------
    print("\n--- Class Distribution ---")
    fraud_counts = df["is_fraud"].value_counts()
    fraud_rate = df["is_fraud"].mean()
    print(f"Legitimate: {fraud_counts.get(0, 0):,}")
    print(f"Fraudulent: {fraud_counts.get(1, 0):,}")
    print(f"Fraud rate:  {fraud_rate:.4f} ({fraud_rate * 100:.2f}%)")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    fraud_counts.plot(kind="bar", ax=axes[0], color=["#4CAF50", "#F44336"])
    axes[0].set_title("Class Distribution")
    axes[0].set_xlabel("Is Fraud")
    axes[0].set_ylabel("Count")
    axes[0].set_xticklabels(["Legitimate", "Fraud"], rotation=0)

    axes[1].pie(
        fraud_counts.values,
        labels=["Legitimate", "Fraud"],
        autopct="%1.2f%%",
        colors=["#4CAF50", "#F44336"],
        startangle=90,
    )
    axes[1].set_title("Class Proportions")

    fig.tight_layout()
    fig.savefig(output_dir / "class_distribution.png", dpi=150)
    plt.close(fig)

    # ---------------------------------------------------------------
    # 3. Claim amount distributions
    # ---------------------------------------------------------------
    print("\n--- Claim Amount Analysis ---")
    for label, group in df.groupby("is_fraud"):
        label_name = "Fraud" if label == 1 else "Legitimate"
        print(f"\n{label_name} claims:")
        print(f"  Mean:   ${group['claim_amount'].mean():,.2f}")
        print(f"  Median: ${group['claim_amount'].median():,.2f}")
        print(f"  Std:    ${group['claim_amount'].std():,.2f}")
        print(f"  Min:    ${group['claim_amount'].min():,.2f}")
        print(f"  Max:    ${group['claim_amount'].max():,.2f}")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    legit = df[df["is_fraud"] == 0]["claim_amount"]
    fraud = df[df["is_fraud"] == 1]["claim_amount"]

    axes[0, 0].hist(legit, bins=50, alpha=0.7, color="#4CAF50", label="Legitimate", density=True)
    axes[0, 0].hist(fraud, bins=50, alpha=0.7, color="#F44336", label="Fraud", density=True)
    axes[0, 0].set_title("Claim Amount Distribution")
    axes[0, 0].set_xlabel("Claim Amount ($)")
    axes[0, 0].legend()

    axes[0, 1].hist(np.log1p(legit), bins=50, alpha=0.7, color="#4CAF50", label="Legitimate", density=True)
    axes[0, 1].hist(np.log1p(fraud), bins=50, alpha=0.7, color="#F44336", label="Fraud", density=True)
    axes[0, 1].set_title("Log Claim Amount Distribution")
    axes[0, 1].set_xlabel("Log(Claim Amount)")
    axes[0, 1].legend()

    sns.boxplot(data=df, x="is_fraud", y="claim_amount", ax=axes[1, 0], palette=["#4CAF50", "#F44336"])
    axes[1, 0].set_title("Claim Amount by Class")
    axes[1, 0].set_xticklabels(["Legitimate", "Fraud"])

    sns.boxplot(data=df, x="is_fraud", y="service_days", ax=axes[1, 1], palette=["#4CAF50", "#F44336"])
    axes[1, 1].set_title("Service Days by Class")
    axes[1, 1].set_xticklabels(["Legitimate", "Fraud"])

    fig.tight_layout()
    fig.savefig(output_dir / "amount_distributions.png", dpi=150)
    plt.close(fig)

    # ---------------------------------------------------------------
    # 4. Numeric feature correlations
    # ---------------------------------------------------------------
    print("\n--- Correlation Analysis ---")
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    corr_matrix = df[numeric_cols].corr()

    fraud_corr = corr_matrix["is_fraud"].drop("is_fraud").sort_values(key=abs, ascending=False)
    print("Top features correlated with fraud:")
    for feat, corr in fraud_corr.head(10).items():
        print(f"  {feat:30s}: {corr:+.4f}")

    fig, ax = plt.subplots(figsize=(12, 10))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(
        corr_matrix,
        mask=mask,
        cmap="RdBu_r",
        center=0,
        annot=False,
        square=True,
        linewidths=0.5,
        ax=ax,
    )
    ax.set_title("Feature Correlation Matrix", fontsize=14)
    fig.tight_layout()
    fig.savefig(output_dir / "correlation_matrix.png", dpi=150)
    plt.close(fig)

    # ---------------------------------------------------------------
    # 5. Categorical feature analysis
    # ---------------------------------------------------------------
    print("\n--- Categorical Feature Analysis ---")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    specialty_fraud = df.groupby("specialty")["is_fraud"].mean().sort_values(ascending=False)
    specialty_fraud.plot(kind="barh", ax=axes[0, 0], color="#FF9800")
    axes[0, 0].set_title("Fraud Rate by Specialty")
    axes[0, 0].set_xlabel("Fraud Rate")

    state_fraud = df.groupby("state")["is_fraud"].mean().sort_values(ascending=False).head(15)
    state_fraud.plot(kind="barh", ax=axes[0, 1], color="#9C27B0")
    axes[0, 1].set_title("Fraud Rate by State (Top 15)")
    axes[0, 1].set_xlabel("Fraud Rate")

    claim_type_fraud = df.groupby("claim_type")["is_fraud"].mean().sort_values(ascending=False)
    claim_type_fraud.plot(kind="barh", ax=axes[1, 0], color="#2196F3")
    axes[1, 0].set_title("Fraud Rate by Claim Type")
    axes[1, 0].set_xlabel("Fraud Rate")

    place_fraud = df.groupby("place_of_service")["is_fraud"].mean().sort_values(ascending=False)
    place_fraud.plot(kind="barh", ax=axes[1, 1], color="#009688")
    axes[1, 1].set_title("Fraud Rate by Place of Service")
    axes[1, 1].set_xlabel("Fraud Rate")

    fig.tight_layout()
    fig.savefig(output_dir / "categorical_analysis.png", dpi=150)
    plt.close(fig)

    # ---------------------------------------------------------------
    # 6. Temporal patterns
    # ---------------------------------------------------------------
    print("\n--- Temporal Patterns ---")
    df["claim_date"] = pd.to_datetime(df["claim_date"])
    df["day_of_week"] = df["claim_date"].dt.dayofweek
    df["month"] = df["claim_date"].dt.month

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    dow_fraud = df.groupby("day_of_week")["is_fraud"].mean()
    day_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    dow_fraud.index = [day_names[i] for i in dow_fraud.index]
    dow_fraud.plot(kind="bar", ax=axes[0], color="#FF5722")
    axes[0].set_title("Fraud Rate by Day of Week")
    axes[0].set_xlabel("Day")
    axes[0].set_ylabel("Fraud Rate")
    axes[0].tick_params(axis="x", rotation=0)

    monthly_fraud = df.groupby("month")["is_fraud"].mean()
    monthly_fraud.plot(kind="bar", ax=axes[1], color="#3F51B5")
    axes[1].set_title("Fraud Rate by Month")
    axes[1].set_xlabel("Month")
    axes[1].set_ylabel("Fraud Rate")
    axes[1].tick_params(axis="x", rotation=0)

    fig.tight_layout()
    fig.savefig(output_dir / "temporal_patterns.png", dpi=150)
    plt.close(fig)

    # ---------------------------------------------------------------
    # 7. Provider analysis
    # ---------------------------------------------------------------
    print("\n--- Provider Analysis ---")
    provider_stats = df.groupby("provider_id").agg(
        total_claims=("claim_amount", "count"),
        mean_amount=("claim_amount", "mean"),
        fraud_rate=("is_fraud", "mean"),
        unique_patients=("patient_id", "nunique"),
    ).reset_index()

    high_fraud_providers = provider_stats[provider_stats["fraud_rate"] > 0.1]
    print(f"Providers with >10% fraud rate: {len(high_fraud_providers)}")
    print(f"Total providers: {len(provider_stats)}")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].scatter(
        provider_stats["total_claims"],
        provider_stats["fraud_rate"],
        alpha=0.5,
        s=20,
        c=provider_stats["fraud_rate"],
        cmap="RdYlGn_r",
    )
    axes[0].set_title("Provider Fraud Rate vs. Claim Volume")
    axes[0].set_xlabel("Total Claims")
    axes[0].set_ylabel("Fraud Rate")

    axes[1].scatter(
        provider_stats["mean_amount"],
        provider_stats["fraud_rate"],
        alpha=0.5,
        s=20,
        c=provider_stats["fraud_rate"],
        cmap="RdYlGn_r",
    )
    axes[1].set_title("Provider Fraud Rate vs. Mean Claim Amount")
    axes[1].set_xlabel("Mean Claim Amount ($)")
    axes[1].set_ylabel("Fraud Rate")

    fig.tight_layout()
    fig.savefig(output_dir / "provider_analysis.png", dpi=150)
    plt.close(fig)

    # ---------------------------------------------------------------
    # 8. Missing data assessment
    # ---------------------------------------------------------------
    print("\n--- Missing Data ---")
    missing = df.isnull().sum()
    missing_pct = (missing / len(df) * 100).round(2)
    missing_report = pd.DataFrame({"count": missing, "percent": missing_pct})
    missing_report = missing_report[missing_report["count"] > 0].sort_values(
        "count", ascending=False
    )
    if len(missing_report) > 0:
        print(missing_report)
    else:
        print("No missing values found.")

    # ---------------------------------------------------------------
    # Summary
    # ---------------------------------------------------------------
    print("\n" + "=" * 60)
    print("EXPLORATION COMPLETE")
    print("=" * 60)
    print(f"Visualizations saved to: {output_dir.resolve()}")
    print(f"Dataset: {len(df):,} claims, {len(df.columns)} columns")
    print(f"Fraud rate: {fraud_rate:.2%}")
    print(f"Top correlated feature: {fraud_corr.index[0]} ({fraud_corr.iloc[0]:+.4f})")


if __name__ == "__main__":
    main()
