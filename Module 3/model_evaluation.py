"""
Model evaluation utilities for binary classifiers.

Computes AUROC/F1 metrics for a given dataset (e.g. train, validation, or
test), and optionally plots a confusion matrix + ROC curve side by side.

Main entry point: evaluate_model(model, X, y, dataset_name='Train', plot=True)
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from sklearn.base import ClassifierMixin
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    f1_score,
    roc_auc_score,
)


def get_predictions(
    model: ClassifierMixin, X: pd.DataFrame
) -> tuple[np.ndarray, np.ndarray]:
    """Return predicted labels and positive-class probabilities for X."""
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)[:, 1]
    return y_pred, y_proba


def compute_classification_metrics(
    y_true: pd.Series, y_pred: np.ndarray, y_proba: np.ndarray
) -> dict[str, float]:
    """Compute AUROC and F1 score for a set of predictions."""
    return {
        "auroc": roc_auc_score(y_true, y_proba),
        "f1": f1_score(y_true, y_pred),
    }


def print_classification_metrics(metrics: dict[str, float], dataset_name: str = "Train") -> None:
    """Print AUROC/F1 metrics in a readable, labeled block."""
    print(f"=== {dataset_name} ===")
    print(f"AUROC:    {metrics['auroc']:.4f}")
    print(f"F1 Score: {metrics['f1']:.4f}")


def plot_confusion_and_roc(
    y_true: pd.Series, y_pred: np.ndarray, y_proba: np.ndarray, dataset_name: str = "Train"
) -> Figure:
    """Plot a normalized confusion matrix and ROC curve side by side."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle(f"{dataset_name} Metrics", fontsize=14)

    ConfusionMatrixDisplay.from_predictions(y_true, y_pred, ax=axes[0], normalize="true")
    axes[0].set_title("Confusion Matrix")

    RocCurveDisplay.from_predictions(y_true, y_proba, ax=axes[1])
    axes[1].set_title("ROC Curve")

    plt.tight_layout()
    return fig


def evaluate_model(
    model: ClassifierMixin,
    X: pd.DataFrame,
    y: pd.Series,
    dataset_name: str = "Train",
    plot: bool = True,
    verbose: bool = True,
) -> dict[str, float]:
    """
    Evaluate a fitted binary classifier on a dataset: compute AUROC/F1,
    optionally print them, and optionally plot a confusion matrix + ROC curve.

    Args:
        model: A fitted classifier exposing .predict and .predict_proba.
        X: Feature matrix to evaluate on.
        y: True labels corresponding to X.
        dataset_name: Label used in printed output and plot title
            (e.g. 'Train', 'Validation', 'Test').
        plot: Whether to plot the confusion matrix and ROC curve. Set to
            False to only compute (and optionally print) the metrics.
        verbose: Whether to print the AUROC/F1 metrics.

    Returns:
        A dictionary with keys "auroc" and "f1" holding the computed metrics.
    """
    y_pred, y_proba = get_predictions(model, X)
    metrics = compute_classification_metrics(y, y_pred, y_proba)

    if verbose:
        print_classification_metrics(metrics, dataset_name)

    if plot:
        plot_confusion_and_roc(y, y_pred, y_proba, dataset_name)
        plt.show()

    return metrics