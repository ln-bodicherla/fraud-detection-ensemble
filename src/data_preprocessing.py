"""
Data preprocessing for insurance claims fraud detection.

Handles data loading, missing value imputation, categorical encoding,
class imbalance treatment, and train/test splitting with temporal awareness.
"""

import logging
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)


class DataPreprocessor:
    """Preprocesses insurance claims data for fraud detection modeling.

    Supports CSV and Parquet input, multiple imputation strategies,
    target/frequency encoding for high-cardinality categoricals,
    SMOTE/ADASYN oversampling, and stratified temporal splitting.
    """

    TARGET_COLUMN = "is_fraud"

    NUMERIC_COLUMNS = [
        "claim_amount", "approved_amount", "deductible", "copay",
        "coinsurance", "service_days", "num_procedures", "patient_age",
    ]

    CATEGORICAL_COLUMNS = [
        "provider_id", "patient_id", "diagnosis_code", "procedure_code",
        "specialty", "state", "claim_type", "place_of_service",
    ]

    DATE_COLUMNS = ["claim_date", "service_date", "submission_date"]

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self._target_encodings: dict[str, dict] = {}
        self._frequency_encodings: dict[str, dict] = {}
        self._imputation_values: dict[str, Any] = {}

    def load_claims_data(
        self,
        path: Union[str, Path],
        date_columns: Optional[list[str]] = None,
    ) -> pd.DataFrame:
        """Load claims data from CSV or Parquet files.

        Args:
            path: File path to the data file.
            date_columns: Columns to parse as datetime.

        Returns:
            DataFrame with parsed date columns.
        """
        path = Path(path)
        parse_dates = date_columns or [
            col for col in self.DATE_COLUMNS
        ]

        if path.suffix == ".parquet":
            df = pd.read_parquet(path)
            for col in parse_dates:
                if col in df.columns:
                    df[col] = pd.to_datetime(df[col], errors="coerce")
        elif path.suffix == ".csv":
            existing_date_cols = []
            sample = pd.read_csv(path, nrows=5)
            for col in parse_dates:
                if col in sample.columns:
                    existing_date_cols.append(col)
            df = pd.read_csv(path, parse_dates=existing_date_cols)
        else:
            raise ValueError(f"Unsupported file format: {path.suffix}")

        logger.info(
            "Loaded %d claims with %d columns from %s",
            len(df), len(df.columns), path,
        )
        return df

    def handle_missing_values(
        self,
        df: pd.DataFrame,
        numeric_strategy: str = "median",
        categorical_strategy: str = "mode",
    ) -> pd.DataFrame:
        """Impute missing values using configurable strategies.

        Args:
            df: Input DataFrame.
            numeric_strategy: One of 'mean', 'median', 'zero'.
            categorical_strategy: One of 'mode', 'missing', 'drop'.

        Returns:
            DataFrame with imputed values.
        """
        df = df.copy()
        missing_before = df.isnull().sum().sum()

        for col in df.select_dtypes(include=[np.number]).columns:
            if df[col].isnull().any():
                if numeric_strategy == "median":
                    fill_value = df[col].median()
                elif numeric_strategy == "mean":
                    fill_value = df[col].mean()
                elif numeric_strategy == "zero":
                    fill_value = 0
                else:
                    fill_value = df[col].median()

                self._imputation_values[col] = fill_value
                df[col] = df[col].fillna(fill_value)

        for col in df.select_dtypes(include=["object", "category"]).columns:
            if df[col].isnull().any():
                if categorical_strategy == "mode":
                    fill_value = df[col].mode().iloc[0] if not df[col].mode().empty else "UNKNOWN"
                elif categorical_strategy == "missing":
                    fill_value = "MISSING"
                elif categorical_strategy == "drop":
                    df = df.dropna(subset=[col])
                    continue
                else:
                    fill_value = "UNKNOWN"

                self._imputation_values[col] = fill_value
                df[col] = df[col].fillna(fill_value)

        missing_after = df.isnull().sum().sum()
        logger.info(
            "Missing values: %d -> %d (imputed %d)",
            missing_before, missing_after, missing_before - missing_after,
        )
        return df

    def encode_categoricals(
        self,
        df: pd.DataFrame,
        target_encode_columns: Optional[list[str]] = None,
        frequency_encode_columns: Optional[list[str]] = None,
        max_cardinality_onehot: int = 20,
    ) -> pd.DataFrame:
        """Encode categorical variables using target and frequency encoding.

        High-cardinality columns use target or frequency encoding instead
        of one-hot encoding to avoid feature explosion.

        Args:
            df: Input DataFrame.
            target_encode_columns: Columns for target encoding.
            frequency_encode_columns: Columns for frequency encoding.
            max_cardinality_onehot: Max unique values for one-hot encoding.

        Returns:
            DataFrame with encoded categorical columns.
        """
        df = df.copy()

        if target_encode_columns is None:
            target_encode_columns = ["provider_id", "patient_id"]

        if frequency_encode_columns is None:
            frequency_encode_columns = ["diagnosis_code", "procedure_code"]

        if self.TARGET_COLUMN in df.columns:
            for col in target_encode_columns:
                if col in df.columns:
                    df = self._target_encode(df, col)

        for col in frequency_encode_columns:
            if col in df.columns:
                df = self._frequency_encode(df, col)

        remaining_cats = df.select_dtypes(include=["object", "category"]).columns
        encoded_cols = set(target_encode_columns) | set(frequency_encode_columns)

        for col in remaining_cats:
            if col in encoded_cols or col == self.TARGET_COLUMN:
                continue
            n_unique = df[col].nunique()
            if n_unique <= max_cardinality_onehot:
                dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)
                df = pd.concat([df, dummies], axis=1)
                df = df.drop(columns=[col])
            else:
                df = self._frequency_encode(df, col)

        logger.info("Categorical encoding complete")
        return df

    def _target_encode(
        self, df: pd.DataFrame, column: str, smoothing: float = 10.0
    ) -> pd.DataFrame:
        """Apply smoothed target encoding to a categorical column."""
        global_mean = df[self.TARGET_COLUMN].mean()
        group_stats = df.groupby(column)[self.TARGET_COLUMN].agg(["mean", "count"])

        smoothed_mean = (
            (group_stats["count"] * group_stats["mean"] + smoothing * global_mean)
            / (group_stats["count"] + smoothing)
        )

        encoding = smoothed_mean.to_dict()
        self._target_encodings[column] = encoding
        self._target_encodings[f"{column}_global_mean"] = global_mean

        encoded_col = f"{column}_target_enc"
        df[encoded_col] = df[column].map(encoding).fillna(global_mean)
        df = df.drop(columns=[column])

        return df

    def _frequency_encode(
        self, df: pd.DataFrame, column: str
    ) -> pd.DataFrame:
        """Apply frequency encoding to a categorical column."""
        freq = df[column].value_counts(normalize=True).to_dict()
        self._frequency_encodings[column] = freq

        encoded_col = f"{column}_freq_enc"
        df[encoded_col] = df[column].map(freq).fillna(0)
        df = df.drop(columns=[column])

        return df

    def handle_class_imbalance(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        method: str = "smote",
        sampling_strategy: float = 0.3,
    ) -> tuple[pd.DataFrame, pd.Series]:
        """Address class imbalance using oversampling techniques.

        Args:
            X: Feature matrix.
            y: Target labels.
            method: One of 'smote', 'adasyn', 'class_weight', 'none'.
            sampling_strategy: Target minority ratio.

        Returns:
            Resampled (X, y) tuple.
        """
        fraud_rate = y.mean()
        logger.info("Original fraud rate: %.4f (%d / %d)", fraud_rate, y.sum(), len(y))

        if method == "none" or method == "class_weight":
            return X, y

        if method == "smote":
            from imblearn.over_sampling import SMOTE

            sampler = SMOTE(
                sampling_strategy=sampling_strategy,
                random_state=self.random_state,
                k_neighbors=5,
            )
        elif method == "adasyn":
            from imblearn.over_sampling import ADASYN

            sampler = ADASYN(
                sampling_strategy=sampling_strategy,
                random_state=self.random_state,
                n_neighbors=5,
            )
        else:
            raise ValueError(f"Unknown resampling method: {method}")

        numeric_X = X.select_dtypes(include=[np.number])
        X_resampled, y_resampled = sampler.fit_resample(numeric_X, y)

        X_resampled = pd.DataFrame(X_resampled, columns=numeric_X.columns)
        y_resampled = pd.Series(y_resampled, name=y.name)

        new_fraud_rate = y_resampled.mean()
        logger.info(
            "After %s: fraud rate %.4f (%d / %d)",
            method, new_fraud_rate, y_resampled.sum(), len(y_resampled),
        )
        return X_resampled, y_resampled

    def create_train_test_split(
        self,
        df: pd.DataFrame,
        test_size: float = 0.2,
        temporal_column: Optional[str] = None,
        stratify: bool = True,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Split data into training and test sets.

        Supports both random stratified splitting and temporal splitting
        where the test set contains the most recent claims.

        Args:
            df: Full DataFrame with features and target.
            test_size: Fraction of data for test set.
            temporal_column: Date column for temporal split.
            stratify: Whether to stratify by target (random split only).

        Returns:
            Tuple of (X_train, X_test, y_train, y_test).
        """
        if self.TARGET_COLUMN not in df.columns:
            raise ValueError(f"Target column '{self.TARGET_COLUMN}' not found")

        y = df[self.TARGET_COLUMN]
        X = df.drop(columns=[self.TARGET_COLUMN])

        date_cols_in_X = [c for c in self.DATE_COLUMNS if c in X.columns]
        if date_cols_in_X:
            X = X.drop(columns=date_cols_in_X)

        if temporal_column and temporal_column in df.columns:
            df_sorted = df.sort_values(temporal_column)
            split_idx = int(len(df_sorted) * (1 - test_size))

            train_df = df_sorted.iloc[:split_idx]
            test_df = df_sorted.iloc[split_idx:]

            y_train = train_df[self.TARGET_COLUMN]
            y_test = test_df[self.TARGET_COLUMN]
            X_train = train_df.drop(
                columns=[self.TARGET_COLUMN] + [
                    c for c in self.DATE_COLUMNS if c in train_df.columns
                ]
            )
            X_test = test_df.drop(
                columns=[self.TARGET_COLUMN] + [
                    c for c in self.DATE_COLUMNS if c in test_df.columns
                ]
            )

            logger.info(
                "Temporal split: train=%d (fraud=%.3f), test=%d (fraud=%.3f)",
                len(X_train), y_train.mean(), len(X_test), y_test.mean(),
            )
        else:
            stratify_col = y if stratify else None
            X_train, X_test, y_train, y_test = train_test_split(
                X, y,
                test_size=test_size,
                random_state=self.random_state,
                stratify=stratify_col,
            )
            logger.info(
                "Random split: train=%d (fraud=%.3f), test=%d (fraud=%.3f)",
                len(X_train), y_train.mean(), len(X_test), y_test.mean(),
            )

        return X_train, X_test, y_train, y_test

    def generate_synthetic_data(
        self,
        n_samples: int = 50000,
        fraud_rate: float = 0.05,
        n_providers: int = 200,
        n_patients: int = 10000,
    ) -> pd.DataFrame:
        """Generate realistic synthetic insurance claims data for demos.

        Creates a complete claims dataset with correlated features and
        injected fraud patterns that mimic real-world fraud schemes.

        Args:
            n_samples: Total number of claims to generate.
            fraud_rate: Proportion of fraudulent claims.
            n_providers: Number of distinct providers.
            n_patients: Number of distinct patients.

        Returns:
            DataFrame with all required columns including is_fraud target.
        """
        rng = np.random.RandomState(self.random_state)

        n_fraud = int(n_samples * fraud_rate)
        n_legit = n_samples - n_fraud

        providers = [f"PROV_{i:04d}" for i in range(n_providers)]
        patients = [f"PAT_{i:06d}" for i in range(n_patients)]
        specialties = [
            "Internal Medicine", "Orthopedics", "Cardiology", "Dermatology",
            "Neurology", "General Surgery", "Psychiatry", "Radiology",
            "Anesthesiology", "Emergency Medicine",
        ]
        states = [
            "CA", "TX", "FL", "NY", "PA", "IL", "OH", "GA", "NC", "MI",
            "NJ", "VA", "WA", "AZ", "MA", "TN", "IN", "MO", "MD", "WI",
        ]
        claim_types = ["Professional", "Institutional", "DME", "Pharmacy"]
        places = ["Office", "Hospital Inpatient", "Hospital Outpatient", "ER", "Lab"]

        diagnosis_codes = [f"D{rng.randint(100, 999)}.{rng.randint(0, 9)}" for _ in range(500)]
        procedure_codes = [f"P{rng.randint(10000, 99999)}" for _ in range(300)]

        start_date = pd.Timestamp("2022-01-01")
        date_offsets = pd.to_timedelta(rng.randint(0, 730, size=n_samples), unit="D")
        claim_dates = start_date + date_offsets

        legit_amounts = rng.lognormal(mean=6.0, sigma=1.2, size=n_legit).clip(50, 50000)
        fraud_amounts = rng.lognormal(mean=7.5, sigma=1.5, size=n_fraud).clip(200, 200000)
        amounts = np.concatenate([legit_amounts, fraud_amounts])

        is_fraud = np.concatenate([np.zeros(n_legit), np.ones(n_fraud)]).astype(int)

        shuffle_idx = rng.permutation(n_samples)
        amounts = amounts[shuffle_idx]
        is_fraud = is_fraud[shuffle_idx]

        fraud_providers = rng.choice(providers, size=int(n_providers * 0.1), replace=False)
        provider_ids = []
        for i in range(n_samples):
            if is_fraud[i] and rng.random() < 0.7:
                provider_ids.append(rng.choice(fraud_providers))
            else:
                provider_ids.append(rng.choice(providers))

        df = pd.DataFrame({
            "claim_id": [f"CLM_{i:08d}" for i in range(n_samples)],
            "provider_id": provider_ids,
            "patient_id": rng.choice(patients, size=n_samples),
            "claim_date": claim_dates[shuffle_idx] if len(claim_dates) == n_samples else claim_dates,
            "claim_amount": np.round(amounts, 2),
            "approved_amount": np.round(amounts * rng.uniform(0.6, 1.0, size=n_samples), 2),
            "deductible": np.round(rng.uniform(0, 500, size=n_samples), 2),
            "copay": np.round(rng.choice([20, 30, 40, 50, 75, 100], size=n_samples).astype(float), 2),
            "coinsurance": np.round(rng.uniform(0, 0.3, size=n_samples), 4),
            "diagnosis_code": rng.choice(diagnosis_codes, size=n_samples),
            "procedure_code": rng.choice(procedure_codes, size=n_samples),
            "specialty": rng.choice(specialties, size=n_samples),
            "state": rng.choice(states, size=n_samples),
            "claim_type": rng.choice(claim_types, size=n_samples),
            "place_of_service": rng.choice(places, size=n_samples),
            "service_days": rng.poisson(lam=3, size=n_samples).clip(1, 60),
            "num_procedures": rng.poisson(lam=2, size=n_samples).clip(1, 20),
            "patient_age": rng.normal(loc=55, scale=18, size=n_samples).clip(18, 95).astype(int),
            "is_fraud": is_fraud,
        })

        fraud_mask = df["is_fraud"] == 1
        df.loc[fraud_mask, "service_days"] = (
            df.loc[fraud_mask, "service_days"] * rng.uniform(1.5, 3.0, size=fraud_mask.sum())
        ).astype(int).clip(1, 60)
        df.loc[fraud_mask, "num_procedures"] = (
            df.loc[fraud_mask, "num_procedures"] * rng.uniform(1.5, 3.0, size=fraud_mask.sum())
        ).astype(int).clip(1, 20)

        n_missing = int(n_samples * 0.02)
        missing_idx = rng.choice(n_samples, size=n_missing, replace=False)
        df.loc[missing_idx[:n_missing // 2], "approved_amount"] = np.nan
        df.loc[missing_idx[n_missing // 2:], "deductible"] = np.nan

        logger.info(
            "Generated %d synthetic claims (fraud rate: %.3f)",
            n_samples, df["is_fraud"].mean(),
        )
        return df
