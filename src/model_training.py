"""
Model training for insurance claims fraud detection.

Implements XGBoost, LightGBM, CatBoost, and a stacking ensemble
with automated hyperparameter optimization via Optuna.
"""

import logging
from pathlib import Path
from typing import Any, Optional, Union

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score

logger = logging.getLogger(__name__)


class FraudDetectionModel:
    """Trains and manages gradient-boosted ensemble models for fraud detection.

    Supports individual XGBoost, LightGBM, and CatBoost models as well
    as a stacking ensemble that combines all three with a logistic
    regression meta-learner.
    """

    def __init__(
        self,
        config_path: Optional[str] = None,
        random_state: int = 42,
    ):
        self.random_state = random_state
        self.config = self._load_config(config_path)
        self.models: dict[str, Any] = {}
        self.feature_importances: dict[str, pd.Series] = {}
        self.stacking_model = None
        self._is_fitted = False

    @staticmethod
    def _load_config(config_path: Optional[str]) -> dict[str, Any]:
        """Load model configuration from YAML."""
        if config_path is None:
            return {}
        path = Path(config_path)
        if not path.exists():
            return {}
        with open(path) as f:
            return yaml.safe_load(f) or {}

    def train_xgboost(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
        params: Optional[dict] = None,
    ) -> Any:
        """Train an XGBoost classifier with scale_pos_weight for imbalance.

        Args:
            X_train: Training features.
            y_train: Training labels.
            X_val: Optional validation features for early stopping.
            y_val: Optional validation labels.
            params: Override default XGBoost parameters.

        Returns:
            Fitted XGBClassifier instance.
        """
        from xgboost import XGBClassifier

        default_params = {
            "n_estimators": 500,
            "max_depth": 6,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "scale_pos_weight": max(1, int((y_train == 0).sum() / max((y_train == 1).sum(), 1))),
            "eval_metric": "aucpr",
            "random_state": self.random_state,
            "n_jobs": -1,
            "verbosity": 0,
        }

        config_params = self.config.get("xgboost", {})
        default_params.update(config_params)
        if params:
            default_params.update(params)

        model = XGBClassifier(**default_params)

        fit_params = {}
        if X_val is not None and y_val is not None:
            fit_params["eval_set"] = [(X_val, y_val)]
            fit_params["verbose"] = False

        model.fit(X_train, y_train, **fit_params)

        self.models["xgboost"] = model
        self.feature_importances["xgboost"] = pd.Series(
            model.feature_importances_, index=X_train.columns
        ).sort_values(ascending=False)

        logger.info("XGBoost trained: %d estimators", model.n_estimators)
        return model

    def train_lightgbm(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
        params: Optional[dict] = None,
    ) -> Any:
        """Train a LightGBM classifier with is_unbalance for imbalance.

        Args:
            X_train: Training features.
            y_train: Training labels.
            X_val: Optional validation features.
            y_val: Optional validation labels.
            params: Override parameters.

        Returns:
            Fitted LGBMClassifier instance.
        """
        from lightgbm import LGBMClassifier

        default_params = {
            "n_estimators": 500,
            "max_depth": 7,
            "learning_rate": 0.05,
            "num_leaves": 63,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "is_unbalance": True,
            "metric": "average_precision",
            "random_state": self.random_state,
            "n_jobs": -1,
            "verbose": -1,
        }

        config_params = self.config.get("lightgbm", {})
        default_params.update(config_params)
        if params:
            default_params.update(params)

        model = LGBMClassifier(**default_params)

        fit_params = {}
        if X_val is not None and y_val is not None:
            fit_params["eval_set"] = [(X_val, y_val)]

        model.fit(X_train, y_train, **fit_params)

        self.models["lightgbm"] = model
        self.feature_importances["lightgbm"] = pd.Series(
            model.feature_importances_, index=X_train.columns
        ).sort_values(ascending=False)

        logger.info("LightGBM trained: %d estimators", model.n_estimators)
        return model

    def train_catboost(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
        params: Optional[dict] = None,
    ) -> Any:
        """Train a CatBoost classifier with auto_class_weights.

        Args:
            X_train: Training features.
            y_train: Training labels.
            X_val: Optional validation features.
            y_val: Optional validation labels.
            params: Override parameters.

        Returns:
            Fitted CatBoostClassifier instance.
        """
        from catboost import CatBoostClassifier

        default_params = {
            "iterations": 500,
            "depth": 6,
            "learning_rate": 0.05,
            "auto_class_weights": "Balanced",
            "random_seed": self.random_state,
            "verbose": 0,
        }

        config_params = self.config.get("catboost", {})
        default_params.update(config_params)
        if params:
            default_params.update(params)

        model = CatBoostClassifier(**default_params)

        eval_set = None
        if X_val is not None and y_val is not None:
            eval_set = (X_val, y_val)

        model.fit(X_train, y_train, eval_set=eval_set)

        self.models["catboost"] = model
        self.feature_importances["catboost"] = pd.Series(
            model.feature_importances_, index=X_train.columns
        ).sort_values(ascending=False)

        logger.info("CatBoost trained: %d iterations", model.tree_count_)
        return model

    def train_stacking_ensemble(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
        cv_folds: int = 5,
    ) -> Any:
        """Train a stacking ensemble with XGB + LGB + CatBoost base learners.

        Uses out-of-fold predictions from each base model as features
        for a logistic regression meta-learner.

        Args:
            X_train: Training features.
            y_train: Training labels.
            X_val: Optional validation features.
            y_val: Optional validation labels.
            cv_folds: Number of cross-validation folds for stacking.

        Returns:
            Dictionary with base models and meta-learner.
        """
        stacking_config = self.config.get("stacking", {})
        cv_folds = stacking_config.get("cv_folds", cv_folds)

        logger.info("Training stacking ensemble with %d folds", cv_folds)

        self.train_xgboost(X_train, y_train, X_val, y_val)
        self.train_lightgbm(X_train, y_train, X_val, y_val)
        self.train_catboost(X_train, y_train, X_val, y_val)

        kfold = StratifiedKFold(
            n_splits=cv_folds, shuffle=True, random_state=self.random_state
        )

        n_samples = len(X_train)
        oof_predictions = np.zeros((n_samples, 3))

        for fold_idx, (train_idx, val_idx) in enumerate(kfold.split(X_train, y_train)):
            logger.info("Stacking fold %d / %d", fold_idx + 1, cv_folds)

            X_fold_train = X_train.iloc[train_idx]
            y_fold_train = y_train.iloc[train_idx]
            X_fold_val = X_train.iloc[val_idx]

            from xgboost import XGBClassifier
            from lightgbm import LGBMClassifier
            from catboost import CatBoostClassifier

            xgb_fold = XGBClassifier(
                **{k: v for k, v in self.config.get("xgboost", {}).items()},
                random_state=self.random_state,
                n_jobs=-1,
                verbosity=0,
                scale_pos_weight=max(1, int((y_fold_train == 0).sum() / max((y_fold_train == 1).sum(), 1))),
            )
            xgb_fold.fit(X_fold_train, y_fold_train)
            oof_predictions[val_idx, 0] = xgb_fold.predict_proba(X_fold_val)[:, 1]

            lgb_fold = LGBMClassifier(
                **{k: v for k, v in self.config.get("lightgbm", {}).items()},
                random_state=self.random_state,
                n_jobs=-1,
                verbose=-1,
            )
            lgb_fold.fit(X_fold_train, y_fold_train)
            oof_predictions[val_idx, 1] = lgb_fold.predict_proba(X_fold_val)[:, 1]

            cb_fold = CatBoostClassifier(
                **{k: v for k, v in self.config.get("catboost", {}).items()},
                random_seed=self.random_state,
                verbose=0,
            )
            cb_fold.fit(X_fold_train, y_fold_train)
            oof_predictions[val_idx, 2] = cb_fold.predict_proba(X_fold_val)[:, 1]

        meta_learner = LogisticRegression(
            C=1.0,
            max_iter=1000,
            random_state=self.random_state,
        )
        meta_learner.fit(oof_predictions, y_train)

        self.stacking_model = {
            "base_models": {
                "xgboost": self.models["xgboost"],
                "lightgbm": self.models["lightgbm"],
                "catboost": self.models["catboost"],
            },
            "meta_learner": meta_learner,
        }
        self._is_fitted = True

        logger.info("Stacking ensemble trained successfully")
        return self.stacking_model

    def predict(
        self,
        X: pd.DataFrame,
        model_name: Optional[str] = None,
    ) -> np.ndarray:
        """Generate fraud probability predictions.

        Args:
            X: Feature matrix.
            model_name: Specific model to use, or None for stacking ensemble.

        Returns:
            Array of fraud probabilities.
        """
        if model_name and model_name in self.models:
            return self.models[model_name].predict_proba(X)[:, 1]

        if self.stacking_model is not None:
            base_preds = np.column_stack([
                self.stacking_model["base_models"]["xgboost"].predict_proba(X)[:, 1],
                self.stacking_model["base_models"]["lightgbm"].predict_proba(X)[:, 1],
                self.stacking_model["base_models"]["catboost"].predict_proba(X)[:, 1],
            ])
            return self.stacking_model["meta_learner"].predict_proba(base_preds)[:, 1]

        if self.models:
            first_model = next(iter(self.models.values()))
            return first_model.predict_proba(X)[:, 1]

        raise RuntimeError("No models have been trained")

    def predict_binary(
        self,
        X: pd.DataFrame,
        threshold: float = 0.5,
        model_name: Optional[str] = None,
    ) -> np.ndarray:
        """Generate binary fraud predictions at a given threshold."""
        probas = self.predict(X, model_name)
        return (probas >= threshold).astype(int)

    def optimize_hyperparameters(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        model_name: str = "xgboost",
        n_trials: int = 100,
        timeout: int = 3600,
        cv_folds: int = 5,
    ) -> dict[str, Any]:
        """Optimize hyperparameters using Optuna with cross-validation.

        Args:
            X_train: Training features.
            y_train: Training labels.
            model_name: Model to optimize ('xgboost', 'lightgbm', 'catboost').
            n_trials: Number of Optuna trials.
            timeout: Maximum optimization time in seconds.
            cv_folds: Cross-validation folds.

        Returns:
            Dictionary of best parameters.
        """
        import optuna
        from sklearn.metrics import average_precision_score

        optuna_config = self.config.get("optuna", {})
        n_trials = optuna_config.get("n_trials", n_trials)
        timeout = optuna_config.get("timeout", timeout)

        def objective(trial):
            if model_name == "xgboost":
                params = self._sample_xgboost_params(trial, y_train)
            elif model_name == "lightgbm":
                params = self._sample_lightgbm_params(trial)
            elif model_name == "catboost":
                params = self._sample_catboost_params(trial)
            else:
                raise ValueError(f"Unknown model: {model_name}")

            kfold = StratifiedKFold(
                n_splits=cv_folds, shuffle=True, random_state=self.random_state
            )

            scores = []
            for train_idx, val_idx in kfold.split(X_train, y_train):
                X_fold_train = X_train.iloc[train_idx]
                y_fold_train = y_train.iloc[train_idx]
                X_fold_val = X_train.iloc[val_idx]
                y_fold_val = y_train.iloc[val_idx]

                model = self._create_model(model_name, params)
                model.fit(X_fold_train, y_fold_train)
                preds = model.predict_proba(X_fold_val)[:, 1]
                scores.append(average_precision_score(y_fold_val, preds))

            return np.mean(scores)

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=n_trials, timeout=timeout)

        best_params = study.best_params
        logger.info(
            "Optuna optimization complete: best AP=%.4f, params=%s",
            study.best_value, best_params,
        )
        return best_params

    def _sample_xgboost_params(self, trial, y_train) -> dict:
        """Sample XGBoost hyperparameters for an Optuna trial."""
        return {
            "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "gamma": trial.suggest_float("gamma", 0.0, 5.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            "scale_pos_weight": max(1, int((y_train == 0).sum() / max((y_train == 1).sum(), 1))),
        }

    @staticmethod
    def _sample_lightgbm_params(trial) -> dict:
        """Sample LightGBM hyperparameters for an Optuna trial."""
        return {
            "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 15, 127),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            "is_unbalance": True,
        }

    @staticmethod
    def _sample_catboost_params(trial) -> dict:
        """Sample CatBoost hyperparameters for an Optuna trial."""
        return {
            "iterations": trial.suggest_int("iterations", 100, 1000),
            "depth": trial.suggest_int("depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-8, 10.0, log=True),
            "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 10.0),
            "random_strength": trial.suggest_float("random_strength", 0.0, 10.0),
            "auto_class_weights": "Balanced",
        }

    def _create_model(self, model_name: str, params: dict) -> Any:
        """Instantiate a model from name and parameters."""
        if model_name == "xgboost":
            from xgboost import XGBClassifier
            return XGBClassifier(**params, random_state=self.random_state, verbosity=0, n_jobs=-1)
        elif model_name == "lightgbm":
            from lightgbm import LGBMClassifier
            return LGBMClassifier(**params, random_state=self.random_state, verbose=-1, n_jobs=-1)
        elif model_name == "catboost":
            from catboost import CatBoostClassifier
            return CatBoostClassifier(**params, random_seed=self.random_state, verbose=0)
        raise ValueError(f"Unknown model: {model_name}")

    def save_models(self, directory: str) -> None:
        """Save all trained models to disk."""
        save_dir = Path(directory)
        save_dir.mkdir(parents=True, exist_ok=True)

        for name, model in self.models.items():
            path = save_dir / f"{name}_model.joblib"
            joblib.dump(model, path)
            logger.info("Saved %s to %s", name, path)

        if self.stacking_model:
            path = save_dir / "stacking_meta_learner.joblib"
            joblib.dump(self.stacking_model["meta_learner"], path)

    def load_models(self, directory: str) -> None:
        """Load previously saved models from disk."""
        load_dir = Path(directory)

        for model_file in load_dir.glob("*_model.joblib"):
            name = model_file.stem.replace("_model", "")
            self.models[name] = joblib.load(model_file)
            logger.info("Loaded %s from %s", name, model_file)

        meta_path = load_dir / "stacking_meta_learner.joblib"
        if meta_path.exists():
            meta_learner = joblib.load(meta_path)
            self.stacking_model = {
                "base_models": self.models,
                "meta_learner": meta_learner,
            }
            self._is_fitted = True
