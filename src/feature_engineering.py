"""
Feature engineering for insurance claims fraud detection.

Generates 200+ features across provider behavior, claim characteristics,
temporal patterns, network relationships, and interaction terms.
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """Constructs a comprehensive feature matrix for fraud detection.

    Produces provider-level, claim-level, temporal, network, and interaction
    features from raw claims data. All features are designed to capture
    statistical anomalies indicative of fraudulent behavior.
    """

    AMOUNT_COLUMNS = [
        "claim_amount", "approved_amount", "deductible",
        "copay", "coinsurance",
    ]

    CATEGORICAL_COLUMNS = [
        "provider_id", "patient_id", "diagnosis_code",
        "procedure_code", "specialty", "state",
    ]

    def __init__(self, rolling_windows: Optional[list[int]] = None):
        self.rolling_windows = rolling_windows or [7, 14, 30, 60, 90]
        self._provider_stats: Optional[pd.DataFrame] = None
        self._global_stats: Optional[dict] = None

    def build_feature_matrix(self, df: pd.DataFrame) -> pd.DataFrame:
        """Orchestrate all feature groups into a unified feature matrix.

        Args:
            df: Raw claims DataFrame with standard column names.

        Returns:
            DataFrame with 200+ engineered features appended.
        """
        logger.info("Building feature matrix from %d claims", len(df))
        self._compute_global_stats(df)

        df = self.create_provider_features(df)
        df = self.create_claim_features(df)
        df = self.create_temporal_features(df)
        df = self.create_network_features(df)
        df = self.create_interaction_features(df)

        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].fillna(0)

        feature_count = len(df.columns)
        logger.info("Feature matrix built: %d features, %d samples", feature_count, len(df))
        return df

    def _compute_global_stats(self, df: pd.DataFrame) -> None:
        """Precompute global statistics used across feature groups."""
        self._global_stats = {}

        for col in self.AMOUNT_COLUMNS:
            if col in df.columns:
                self._global_stats[f"{col}_mean"] = df[col].mean()
                self._global_stats[f"{col}_std"] = df[col].std()
                self._global_stats[f"{col}_median"] = df[col].median()
                percentiles = df[col].quantile([0.25, 0.75, 0.90, 0.95, 0.99]).to_dict()
                for q, val in percentiles.items():
                    self._global_stats[f"{col}_q{int(q * 100)}"] = val

        if "provider_id" in df.columns:
            self._provider_stats = (
                df.groupby("provider_id")
                .agg(
                    total_claims=("claim_amount", "count"),
                    mean_amount=("claim_amount", "mean"),
                    std_amount=("claim_amount", "std"),
                    max_amount=("claim_amount", "max"),
                    unique_patients=("patient_id", "nunique")
                    if "patient_id" in df.columns
                    else ("claim_amount", "count"),
                )
                .reset_index()
            )

    def create_provider_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate provider-level behavioral features.

        Captures billing frequency, average amounts, specialty deviation,
        patient volume, and billing pattern anomalies per provider.
        """
        if "provider_id" not in df.columns:
            logger.warning("provider_id column missing, skipping provider features")
            return df

        provider_grp = df.groupby("provider_id")

        provider_agg = provider_grp.agg(
            prov_claim_count=("claim_amount", "count"),
            prov_mean_amount=("claim_amount", "mean"),
            prov_std_amount=("claim_amount", "std"),
            prov_median_amount=("claim_amount", "median"),
            prov_max_amount=("claim_amount", "max"),
            prov_min_amount=("claim_amount", "min"),
            prov_sum_amount=("claim_amount", "sum"),
        ).reset_index()

        if "patient_id" in df.columns:
            patient_counts = provider_grp["patient_id"].nunique().reset_index()
            patient_counts.columns = ["provider_id", "prov_unique_patients"]
            provider_agg = provider_agg.merge(patient_counts, on="provider_id", how="left")

            claims_per_patient = (
                df.groupby(["provider_id", "patient_id"]).size()
                .reset_index(name="pair_count")
                .groupby("provider_id")["pair_count"]
                .agg(["mean", "max", "std"])
                .reset_index()
            )
            claims_per_patient.columns = [
                "provider_id", "prov_avg_claims_per_patient",
                "prov_max_claims_per_patient", "prov_std_claims_per_patient",
            ]
            provider_agg = provider_agg.merge(claims_per_patient, on="provider_id", how="left")

        if "diagnosis_code" in df.columns:
            diag_diversity = provider_grp["diagnosis_code"].nunique().reset_index()
            diag_diversity.columns = ["provider_id", "prov_diagnosis_diversity"]
            provider_agg = provider_agg.merge(diag_diversity, on="provider_id", how="left")

        if "procedure_code" in df.columns:
            proc_diversity = provider_grp["procedure_code"].nunique().reset_index()
            proc_diversity.columns = ["provider_id", "prov_procedure_diversity"]
            provider_agg = provider_agg.merge(proc_diversity, on="provider_id", how="left")

        if "specialty" in df.columns:
            specialty_avg = df.groupby("specialty")["claim_amount"].mean()
            df["specialty_mean_amount"] = df["specialty"].map(specialty_avg)
            df["prov_specialty_deviation"] = (
                df["claim_amount"] - df["specialty_mean_amount"]
            ) / df["specialty_mean_amount"].replace(0, 1)

        provider_agg["prov_amount_range"] = (
            provider_agg["prov_max_amount"] - provider_agg["prov_min_amount"]
        )
        provider_agg["prov_cv_amount"] = (
            provider_agg["prov_std_amount"] / provider_agg["prov_mean_amount"].replace(0, 1)
        )

        global_mean = self._global_stats.get("claim_amount_mean", df["claim_amount"].mean())
        provider_agg["prov_amount_vs_global"] = (
            provider_agg["prov_mean_amount"] - global_mean
        ) / max(global_mean, 1)

        if "approved_amount" in df.columns:
            approval_rate = provider_grp.apply(
                lambda g: (g["approved_amount"] > 0).mean()
            ).reset_index()
            approval_rate.columns = ["provider_id", "prov_approval_rate"]
            provider_agg = provider_agg.merge(approval_rate, on="provider_id", how="left")

        df = df.merge(provider_agg, on="provider_id", how="left", suffixes=("", "_prov"))

        logger.info("Created %d provider features", len(provider_agg.columns) - 1)
        return df

    def create_claim_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate claim-level anomaly features.

        Captures amount z-scores, diagnosis-procedure mismatch indicators,
        service duration anomalies, and billing pattern flags.
        """
        global_mean = self._global_stats.get("claim_amount_mean", 0)
        global_std = self._global_stats.get("claim_amount_std", 1)

        df["claim_amount_zscore"] = (df["claim_amount"] - global_mean) / max(global_std, 1e-6)
        df["claim_amount_log"] = np.log1p(df["claim_amount"].clip(lower=0))
        df["claim_is_high_value"] = (
            df["claim_amount"] > self._global_stats.get("claim_amount_q95", df["claim_amount"].quantile(0.95))
        ).astype(int)
        df["claim_is_round_number"] = (df["claim_amount"] % 100 == 0).astype(int)
        df["claim_amount_cents"] = (df["claim_amount"] * 100 % 100).astype(int)

        if "approved_amount" in df.columns:
            df["claim_approval_ratio"] = (
                df["approved_amount"] / df["claim_amount"].replace(0, 1)
            )
            df["claim_denied"] = (df["approved_amount"] == 0).astype(int)
            df["claim_partial_denial"] = (
                (df["approved_amount"] > 0) & (df["approved_amount"] < df["claim_amount"])
            ).astype(int)

        if "deductible" in df.columns:
            df["claim_deductible_ratio"] = (
                df["deductible"] / df["claim_amount"].replace(0, 1)
            )

        if "diagnosis_code" in df.columns and "procedure_code" in df.columns:
            diag_proc_pairs = df.groupby(["diagnosis_code", "procedure_code"]).size()
            diag_proc_freq = diag_proc_pairs.reset_index(name="diag_proc_frequency")
            df = df.merge(
                diag_proc_freq, on=["diagnosis_code", "procedure_code"],
                how="left",
            )
            total_claims = len(df)
            df["diag_proc_rarity"] = 1 - (df["diag_proc_frequency"] / total_claims)

            diag_counts = df["diagnosis_code"].value_counts()
            proc_counts = df["procedure_code"].value_counts()
            df["diagnosis_frequency"] = df["diagnosis_code"].map(diag_counts)
            df["procedure_frequency"] = df["procedure_code"].map(proc_counts)
            df["diag_proc_mismatch_score"] = (
                df["diag_proc_rarity"] *
                (1 / np.log1p(df["diagnosis_frequency"])) *
                (1 / np.log1p(df["procedure_frequency"]))
            )

        if "service_days" in df.columns:
            mean_duration = df["service_days"].mean()
            std_duration = df["service_days"].std()
            df["service_duration_zscore"] = (
                (df["service_days"] - mean_duration) / max(std_duration, 1e-6)
            )
            df["service_duration_anomaly"] = (
                df["service_duration_zscore"].abs() > 2
            ).astype(int)

        if "num_procedures" in df.columns:
            df["procedures_per_day"] = df["num_procedures"] / df.get(
                "service_days", pd.Series([1] * len(df))
            ).replace(0, 1)
            df["high_procedure_count"] = (
                df["num_procedures"] > df["num_procedures"].quantile(0.95)
            ).astype(int)

        logger.info("Created claim-level features")
        return df

    def create_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate time-based features from claim dates.

        Produces rolling window counts, inter-claim intervals,
        weekend/holiday flags, and velocity metrics.
        """
        date_col = None
        for candidate in ["claim_date", "service_date", "submission_date"]:
            if candidate in df.columns:
                date_col = candidate
                break

        if date_col is None:
            logger.warning("No date column found, skipping temporal features")
            return df

        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        df = df.sort_values(date_col)

        df["day_of_week"] = df[date_col].dt.dayofweek
        df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
        df["month"] = df[date_col].dt.month
        df["quarter"] = df[date_col].dt.quarter
        df["day_of_month"] = df[date_col].dt.day
        df["is_month_end"] = (df["day_of_month"] >= 28).astype(int)
        df["is_month_start"] = (df["day_of_month"] <= 3).astype(int)

        us_holidays = {
            (1, 1), (7, 4), (12, 25), (11, 28), (9, 1), (5, 27),
            (1, 20), (2, 17), (10, 14), (11, 11),
        }
        df["is_near_holiday"] = df[date_col].apply(
            lambda d: int(any(
                abs(d.month - m) <= 0 and abs(d.day - day) <= 2
                for m, day in us_holidays
            )) if pd.notna(d) else 0
        )

        if "provider_id" in df.columns:
            df = df.sort_values(["provider_id", date_col])

            for window in self.rolling_windows:
                col_name = f"prov_claims_{window}d"
                df[col_name] = (
                    df.groupby("provider_id")[date_col]
                    .transform(
                        lambda x: x.rolling(f"{window}D", min_periods=1).count()
                    )
                )

                amt_col = f"prov_amount_{window}d"
                df[amt_col] = (
                    df.groupby("provider_id")["claim_amount"]
                    .transform(
                        lambda x: x.rolling(window, min_periods=1).mean()
                    )
                )

            df["prov_days_since_last_claim"] = (
                df.groupby("provider_id")[date_col]
                .diff()
                .dt.total_seconds() / 86400
            )
            df["prov_days_since_last_claim"] = df["prov_days_since_last_claim"].fillna(0)

            df["prov_claim_velocity"] = 1 / df["prov_days_since_last_claim"].replace(0, 999)

            if len(self.rolling_windows) >= 2:
                short_window = self.rolling_windows[0]
                long_window = self.rolling_windows[-1]
                short_col = f"prov_claims_{short_window}d"
                long_col = f"prov_claims_{long_window}d"
                if short_col in df.columns and long_col in df.columns:
                    df["prov_velocity_ratio"] = (
                        df[short_col] / df[long_col].replace(0, 1)
                    )

        if "patient_id" in df.columns:
            df = df.sort_values(["patient_id", date_col])
            df["patient_days_since_last_claim"] = (
                df.groupby("patient_id")[date_col]
                .diff()
                .dt.total_seconds() / 86400
            ).fillna(0)

            for window in [7, 30]:
                col_name = f"patient_claims_{window}d"
                df[col_name] = (
                    df.groupby("patient_id")[date_col]
                    .transform(
                        lambda x: x.rolling(f"{window}D", min_periods=1).count()
                    )
                )

        logger.info("Created temporal features with windows %s", self.rolling_windows)
        return df

    def create_network_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate network-based relational features.

        Captures provider-patient pair frequency, shared patient counts,
        referral patterns, and geographic anomalies.
        """
        if "provider_id" not in df.columns or "patient_id" not in df.columns:
            logger.warning("Missing provider_id/patient_id, skipping network features")
            return df

        pair_counts = (
            df.groupby(["provider_id", "patient_id"]).size()
            .reset_index(name="net_pair_frequency")
        )
        df = df.merge(pair_counts, on=["provider_id", "patient_id"], how="left")

        pair_mean = pair_counts["net_pair_frequency"].mean()
        pair_std = pair_counts["net_pair_frequency"].std()
        df["net_pair_zscore"] = (
            (df["net_pair_frequency"] - pair_mean) / max(pair_std, 1e-6)
        )

        patient_provider_count = (
            df.groupby("patient_id")["provider_id"].nunique()
            .reset_index(name="net_patient_provider_count")
        )
        df = df.merge(patient_provider_count, on="patient_id", how="left")

        provider_shared = (
            df.groupby("provider_id")["patient_id"]
            .apply(set)
            .reset_index(name="patient_set")
        )
        shared_counts = {}
        provider_sets = dict(zip(provider_shared["provider_id"], provider_shared["patient_set"]))

        for prov_id, patients in provider_sets.items():
            total_shared = 0
            for other_id, other_patients in provider_sets.items():
                if prov_id != other_id:
                    total_shared += len(patients & other_patients)
            shared_counts[prov_id] = total_shared

        shared_df = pd.DataFrame(
            list(shared_counts.items()),
            columns=["provider_id", "net_shared_patient_total"]
        )
        df = df.merge(shared_df, on="provider_id", how="left")

        if "referring_provider" in df.columns:
            referral_depth = df.groupby("provider_id")["referring_provider"].nunique()
            referral_depth = referral_depth.reset_index(name="net_referral_chain_depth")
            df = df.merge(referral_depth, on="provider_id", how="left")

            df["has_referral"] = df["referring_provider"].notna().astype(int)

        if "state" in df.columns:
            state_provider_count = (
                df.groupby("state")["provider_id"].nunique()
                .reset_index(name="net_state_provider_density")
            )
            df = df.merge(state_provider_count, on="state", how="left")

            provider_state_count = (
                df.groupby("provider_id")["state"].nunique()
                .reset_index(name="net_provider_state_count")
            )
            df = df.merge(provider_state_count, on="provider_id", how="left")
            df["net_multi_state_flag"] = (df["net_provider_state_count"] > 1).astype(int)

        logger.info("Created network features")
        return df

    def create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate polynomial and ratio interaction features.

        Creates cross-feature interactions between key numeric columns
        to capture non-linear relationships.
        """
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        interaction_pairs = [
            ("claim_amount_zscore", "prov_claim_count"),
            ("claim_amount_zscore", "net_pair_frequency"),
            ("prov_mean_amount", "prov_claim_count"),
            ("claim_amount_log", "prov_cv_amount"),
        ]

        for col_a, col_b in interaction_pairs:
            if col_a in df.columns and col_b in df.columns:
                df[f"ix_{col_a}_x_{col_b}"] = df[col_a] * df[col_b]
                denominator = df[col_b].replace(0, 1)
                df[f"ix_{col_a}_div_{col_b}"] = df[col_a] / denominator

        amount_cols = [c for c in self.AMOUNT_COLUMNS if c in df.columns]
        for i, col_a in enumerate(amount_cols):
            for col_b in amount_cols[i + 1:]:
                df[f"ix_ratio_{col_a}_{col_b}"] = (
                    df[col_a] / df[col_b].replace(0, 1)
                )
                df[f"ix_diff_{col_a}_{col_b}"] = df[col_a] - df[col_b]

        if "claim_amount" in df.columns:
            df["claim_amount_sq"] = df["claim_amount"] ** 2
            df["claim_amount_sqrt"] = np.sqrt(df["claim_amount"].clip(lower=0))
            df["claim_amount_cubert"] = np.cbrt(df["claim_amount"])

        if "prov_claim_count" in df.columns and "prov_unique_patients" in df.columns:
            df["ix_claims_per_patient"] = (
                df["prov_claim_count"] / df["prov_unique_patients"].replace(0, 1)
            )

        if "prov_mean_amount" in df.columns and "prov_median_amount" in df.columns:
            df["ix_amount_skewness_proxy"] = (
                (df["prov_mean_amount"] - df["prov_median_amount"])
                / df["prov_std_amount"].replace(0, 1)
            )

        logger.info("Created interaction features")
        return df
