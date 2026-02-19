"""
Model evaluation for insurance claims fraud detection.

Provides comprehensive evaluation metrics, visualizations, threshold
analysis, and report generation for fraud detection models.
"""

import logging
from pathlib import Path
from typing import Any, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold

logger = logging.getLogger(__name__)


class ModelEvaluator:
    """Comprehensive evaluation suite for fraud detection models.

    Computes classification metrics, generates visualizations for ROC
    and precision-recall curves, analyzes threshold sensitivity, and
    produces full evaluation reports.
    """

    def __init__(
        self,
        model: Any,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        model_name: str = "ensemble",
    ):
        self.model = model
        self.X_test = X_test
        self.y_test = y_test
        self.model_name = model_name

        self.y_proba = self._get_probabilities()
        self.y_pred = (self.y_proba >= 0.5).astype(int)

    def _get_probabilities(self) -> np.ndarray:
        """Obtain fraud probability predictions from the model."""
        if hasattr(self.model, "predict"):
            return self.model.predict(self.X_test)
        elif hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(self.X_test)[:, 1]
        raise ValueError("Model must have predict or predict_proba method")

    def classification_metrics(self) -> dict[str, float]:
        """Compute comprehensive classification metrics.

        Returns:
            Dictionary with precision, recall, F1, AUC-ROC, AUC-PR,
            accuracy, and support counts.
        """
        metrics = {
            "accuracy": accuracy_score(self.y_test, self.y_pred),
            "precision": precision_score(self.y_test, self.y_pred, zero_division=0),
            "recall": recall_score(self.y_test, self.y_pred, zero_division=0),
            "f1_score": f1_score(self.y_test, self.y_pred, zero_division=0),
            "auc_roc": roc_auc_score(self.y_test, self.y_proba),
            "auc_pr": average_precision_score(self.y_test, self.y_proba),
            "total_samples": len(self.y_test),
            "total_fraud": int(self.y_test.sum()),
            "total_legitimate": int((self.y_test == 0).sum()),
            "predicted_fraud": int(self.y_pred.sum()),
            "true_positives": int(((self.y_pred == 1) & (self.y_test == 1)).sum()),
            "false_positives": int(((self.y_pred == 1) & (self.y_test == 0)).sum()),
            "true_negatives": int(((self.y_pred == 0) & (self.y_test == 0)).sum()),
            "false_negatives": int(((self.y_pred == 0) & (self.y_test == 1)).sum()),
        }

        logger.info(
            "Metrics - AUC-ROC: %.4f, AUC-PR: %.4f, F1: %.4f",
            metrics["auc_roc"], metrics["auc_pr"], metrics["f1_score"],
        )
        return metrics

    def detailed_classification_report(self) -> str:
        """Generate sklearn classification report string."""
        return classification_report(
            self.y_test, self.y_pred,
            target_names=["Legitimate", "Fraud"],
            digits=4,
        )

    def plot_roc_curve(
        self,
        save_path: Optional[str] = None,
        figsize: tuple[int, int] = (8, 6),
    ) -> plt.Figure:
        """Plot the Receiver Operating Characteristic curve.

        Args:
            save_path: Optional file path to save the figure.
            figsize: Figure dimensions.

        Returns:
            Matplotlib Figure object.
        """
        fpr, tpr, thresholds = roc_curve(self.y_test, self.y_proba)
        auc_score = roc_auc_score(self.y_test, self.y_proba)

        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(fpr, tpr, color="#2196F3", linewidth=2, label=f"{self.model_name} (AUC = {auc_score:.4f})")
        ax.plot([0, 1], [0, 1], color="gray", linestyle="--", linewidth=1, label="Random")
        ax.fill_between(fpr, tpr, alpha=0.1, color="#2196F3")

        ax.set_xlabel("False Positive Rate", fontsize=12)
        ax.set_ylabel("True Positive Rate", fontsize=12)
        ax.set_title("ROC Curve", fontsize=14)
        ax.legend(loc="lower right", fontsize=11)
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.grid(True, alpha=0.3)
        fig.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            logger.info("ROC curve saved to %s", save_path)

        return fig

    def plot_precision_recall_curve(
        self,
        save_path: Optional[str] = None,
        figsize: tuple[int, int] = (8, 6),
    ) -> plt.Figure:
        """Plot the Precision-Recall curve.

        Args:
            save_path: Optional file path to save the figure.
            figsize: Figure dimensions.

        Returns:
            Matplotlib Figure object.
        """
        precision, recall, thresholds = precision_recall_curve(self.y_test, self.y_proba)
        ap_score = average_precision_score(self.y_test, self.y_proba)

        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(recall, precision, color="#4CAF50", linewidth=2, label=f"{self.model_name} (AP = {ap_score:.4f})")
        ax.fill_between(recall, precision, alpha=0.1, color="#4CAF50")

        baseline = self.y_test.mean()
        ax.axhline(y=baseline, color="gray", linestyle="--", linewidth=1, label=f"Baseline ({baseline:.4f})")

        ax.set_xlabel("Recall", fontsize=12)
        ax.set_ylabel("Precision", fontsize=12)
        ax.set_title("Precision-Recall Curve", fontsize=14)
        ax.legend(loc="upper right", fontsize=11)
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.grid(True, alpha=0.3)
        fig.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            logger.info("PR curve saved to %s", save_path)

        return fig

    def plot_confusion_matrix(
        self,
        save_path: Optional[str] = None,
        figsize: tuple[int, int] = (7, 6),
        normalize: bool = False,
    ) -> plt.Figure:
        """Plot the confusion matrix as a heatmap.

        Args:
            save_path: Optional file path to save the figure.
            figsize: Figure dimensions.
            normalize: Whether to normalize by true label counts.

        Returns:
            Matplotlib Figure object.
        """
        cm = confusion_matrix(self.y_test, self.y_pred)
        if normalize:
            cm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
            fmt = ".2%"
        else:
            fmt = "d"

        fig, ax = plt.subplots(figsize=figsize)
        sns.heatmap(
            cm,
            annot=True,
            fmt=fmt,
            cmap="Blues",
            xticklabels=["Legitimate", "Fraud"],
            yticklabels=["Legitimate", "Fraud"],
            ax=ax,
            cbar_kws={"shrink": 0.8},
        )

        ax.set_xlabel("Predicted Label", fontsize=12)
        ax.set_ylabel("True Label", fontsize=12)
        ax.set_title("Confusion Matrix", fontsize=14)
        fig.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            logger.info("Confusion matrix saved to %s", save_path)

        return fig

    def plot_feature_importance(
        self,
        feature_importances: pd.Series,
        top_n: int = 20,
        save_path: Optional[str] = None,
        figsize: tuple[int, int] = (10, 8),
    ) -> plt.Figure:
        """Plot the top N most important features.

        Args:
            feature_importances: Series of feature name to importance score.
            top_n: Number of top features to display.
            save_path: Optional file path to save the figure.
            figsize: Figure dimensions.

        Returns:
            Matplotlib Figure object.
        """
        top_features = feature_importances.head(top_n)

        fig, ax = plt.subplots(figsize=figsize)
        top_features.sort_values().plot(kind="barh", ax=ax, color="#FF9800")

        ax.set_xlabel("Importance Score", fontsize=12)
        ax.set_ylabel("Feature", fontsize=12)
        ax.set_title(f"Top {top_n} Feature Importances", fontsize=14)
        ax.grid(True, alpha=0.3, axis="x")
        fig.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            logger.info("Feature importance plot saved to %s", save_path)

        return fig

    def cross_validate(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        cv_folds: int = 5,
    ) -> dict[str, Any]:
        """Perform stratified K-fold cross-validation.

        Args:
            X: Full feature matrix.
            y: Full target vector.
            cv_folds: Number of folds.

        Returns:
            Dictionary with per-fold and aggregate metrics.
        """
        kfold = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)

        fold_metrics = []
        for fold_idx, (train_idx, val_idx) in enumerate(kfold.split(X, y)):
            X_fold_val = X.iloc[val_idx]
            y_fold_val = y.iloc[val_idx]

            y_proba = self._get_fold_predictions(X_fold_val)

            fold_result = {
                "fold": fold_idx + 1,
                "auc_roc": roc_auc_score(y_fold_val, y_proba),
                "auc_pr": average_precision_score(y_fold_val, y_proba),
                "f1": f1_score(y_fold_val, (y_proba >= 0.5).astype(int), zero_division=0),
                "precision": precision_score(y_fold_val, (y_proba >= 0.5).astype(int), zero_division=0),
                "recall": recall_score(y_fold_val, (y_proba >= 0.5).astype(int), zero_division=0),
            }
            fold_metrics.append(fold_result)

        fold_df = pd.DataFrame(fold_metrics)
        summary = {
            "per_fold": fold_metrics,
            "mean_auc_roc": fold_df["auc_roc"].mean(),
            "std_auc_roc": fold_df["auc_roc"].std(),
            "mean_auc_pr": fold_df["auc_pr"].mean(),
            "std_auc_pr": fold_df["auc_pr"].std(),
            "mean_f1": fold_df["f1"].mean(),
            "std_f1": fold_df["f1"].std(),
        }

        logger.info(
            "CV Results - AUC-ROC: %.4f (+/- %.4f), AUC-PR: %.4f (+/- %.4f)",
            summary["mean_auc_roc"], summary["std_auc_roc"],
            summary["mean_auc_pr"], summary["std_auc_pr"],
        )
        return summary

    def _get_fold_predictions(self, X: pd.DataFrame) -> np.ndarray:
        """Get predictions for a validation fold."""
        if hasattr(self.model, "predict"):
            return self.model.predict(X)
        return self.model.predict_proba(X)[:, 1]

    def threshold_analysis(
        self,
        thresholds: Optional[list[float]] = None,
        save_path: Optional[str] = None,
    ) -> pd.DataFrame:
        """Analyze precision and recall at different decision thresholds.

        Args:
            thresholds: List of thresholds to evaluate.
            save_path: Optional file path to save the analysis plot.

        Returns:
            DataFrame with metrics at each threshold.
        """
        if thresholds is None:
            thresholds = np.arange(0.05, 1.0, 0.05).tolist()

        results = []
        for threshold in thresholds:
            y_pred = (self.y_proba >= threshold).astype(int)

            n_flagged = y_pred.sum()
            tp = ((y_pred == 1) & (self.y_test == 1)).sum()
            fp = ((y_pred == 1) & (self.y_test == 0)).sum()

            results.append({
                "threshold": round(threshold, 2),
                "precision": precision_score(self.y_test, y_pred, zero_division=0),
                "recall": recall_score(self.y_test, y_pred, zero_division=0),
                "f1": f1_score(self.y_test, y_pred, zero_division=0),
                "flagged_claims": int(n_flagged),
                "true_positives": int(tp),
                "false_positives": int(fp),
                "flag_rate": n_flagged / len(self.y_test),
            })

        df = pd.DataFrame(results)

        if save_path:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(df["threshold"], df["precision"], "b-", linewidth=2, label="Precision")
            ax.plot(df["threshold"], df["recall"], "r-", linewidth=2, label="Recall")
            ax.plot(df["threshold"], df["f1"], "g--", linewidth=2, label="F1 Score")

            ax.set_xlabel("Decision Threshold", fontsize=12)
            ax.set_ylabel("Score", fontsize=12)
            ax.set_title("Threshold Analysis", fontsize=14)
            ax.legend(fontsize=11)
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            plt.close(fig)

        return df

    def generate_evaluation_report(
        self,
        output_dir: str,
        include_plots: bool = True,
    ) -> str:
        """Generate a comprehensive evaluation report with metrics and plots.

        Args:
            output_dir: Directory to save report files.
            include_plots: Whether to generate and save visualization plots.

        Returns:
            Path to the generated report file.
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        metrics = self.classification_metrics()
        report_text = self.detailed_classification_report()
        threshold_df = self.threshold_analysis()

        if include_plots:
            self.plot_roc_curve(save_path=str(output_path / "roc_curve.png"))
            self.plot_precision_recall_curve(save_path=str(output_path / "pr_curve.png"))
            self.plot_confusion_matrix(save_path=str(output_path / "confusion_matrix.png"))
            plt.close("all")

        report_lines = [
            "# Fraud Detection Model Evaluation Report",
            "",
            "## Classification Metrics",
            "",
            f"- AUC-ROC: {metrics['auc_roc']:.4f}",
            f"- AUC-PR: {metrics['auc_pr']:.4f}",
            f"- Precision: {metrics['precision']:.4f}",
            f"- Recall: {metrics['recall']:.4f}",
            f"- F1 Score: {metrics['f1_score']:.4f}",
            f"- Accuracy: {metrics['accuracy']:.4f}",
            "",
            "## Confusion Matrix Summary",
            "",
            f"- True Positives: {metrics['true_positives']}",
            f"- False Positives: {metrics['false_positives']}",
            f"- True Negatives: {metrics['true_negatives']}",
            f"- False Negatives: {metrics['false_negatives']}",
            "",
            "## Detailed Classification Report",
            "",
            "```",
            report_text,
            "```",
            "",
            "## Threshold Analysis",
            "",
            threshold_df.to_string(index=False),
            "",
        ]

        report_content = "\n".join(report_lines)
        report_path = output_path / "evaluation_report.md"
        report_path.write_text(report_content)

        metrics_path = output_path / "metrics.csv"
        pd.DataFrame([metrics]).to_csv(metrics_path, index=False)

        threshold_path = output_path / "threshold_analysis.csv"
        threshold_df.to_csv(threshold_path, index=False)

        logger.info("Evaluation report saved to %s", output_path)
        return str(report_path)
