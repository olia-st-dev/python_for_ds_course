"""
Hyperparameter sweep utilities: train DecisionTreeClassifier models across a
range of max_depth values and compare train/validation AUROC.

Main entry points:
  - train_models_by_max_depth(...): fits one model per depth, returns models
    and metrics.
  - plot_auroc_by_max_depth(...): plots train vs. validation AUROC curves
    across depth, to visualize overfitting.
"""

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.figure import Figure
from sklearn.tree import DecisionTreeClassifier

from model_evaluation import evaluate_model


def train_decision_tree(
    X_train: pd.DataFrame,
    train_targets: pd.Series,
    max_depth: int,
    random_state: int = 42,
) -> DecisionTreeClassifier:
    """Fit a single DecisionTreeClassifier with the given max_depth."""
    model = DecisionTreeClassifier(random_state=random_state, max_depth=max_depth)
    model.fit(X_train, train_targets)
    return model


def train_models_by_max_depth(
    X_train: pd.DataFrame,
    train_targets: pd.Series,
    X_val: pd.DataFrame,
    val_targets: pd.Series,
    max_depths: range = range(1, 21),
    random_state: int = 42,
) -> dict:
    """
    Fit one DecisionTreeClassifier per max_depth value and evaluate each on
    both train and validation sets.

    Args:
        X_train: Training feature matrix.
        train_targets: Training targets.
        X_val: Validation feature matrix.
        val_targets: Validation targets.
        max_depths: Sequence of max_depth values to try.
        random_state: Random state used for every tree, for reproducibility.

    Returns:
        A dictionary with keys:
            "models" (dict[int, DecisionTreeClassifier]): Fitted model per depth.
            "train_metrics" (dict[int, dict[str, float]]): {"auroc", "f1"} per depth, on train.
            "val_metrics" (dict[int, dict[str, float]]): {"auroc", "f1"} per depth, on validation.
    """
    models: dict[int, DecisionTreeClassifier] = {}
    train_metrics: dict[int, dict[str, float]] = {}
    val_metrics: dict[int, dict[str, float]] = {}

    for depth in max_depths:
        model = train_decision_tree(X_train, train_targets, max_depth=depth, random_state=random_state)
        models[depth] = model
        train_metrics[depth] = evaluate_model(model, X_train, train_targets, plot=False, verbose=False)
        val_metrics[depth] = evaluate_model(model, X_val, val_targets, plot=False, verbose=False)

    return {
        "models": models,
        "train_metrics": train_metrics,
        "val_metrics": val_metrics,
    }


def extract_metric_by_depth(metrics_by_depth: dict[int, dict[str, float]], metric: str = "auroc") -> dict[int, float]:
    """Pull out a single metric (e.g. 'auroc') from a {depth: metrics_dict} mapping."""
    return {depth: values[metric] for depth, values in metrics_by_depth.items()}


def plot_auroc_by_max_depth(
    train_metrics: dict[int, dict[str, float]],
    val_metrics: dict[int, dict[str, float]],
) -> Figure:
    """
    Plot train vs. validation AUROC across max_depth values on one chart,
    useful for spotting the point where the model starts overfitting.

    Args:
        train_metrics: {depth: {"auroc": ..., "f1": ...}} from train_models_by_max_depth.
        val_metrics: {depth: {"auroc": ..., "f1": ...}} from train_models_by_max_depth.

    Returns:
        The matplotlib Figure containing the plot.
    """
    train_aurocs = extract_metric_by_depth(train_metrics, "auroc")
    val_aurocs = extract_metric_by_depth(val_metrics, "auroc")
    depths = sorted(train_aurocs.keys())

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(depths, [train_aurocs[d] for d in depths], marker="o", label="Train AUROC")
    ax.plot(depths, [val_aurocs[d] for d in depths], marker="o", label="Validation AUROC")
    ax.set_xlabel("max_depth")
    ax.set_ylabel("AUROC")
    ax.set_title("Train vs. Validation AUROC by Tree Depth")
    ax.set_xticks(depths)
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    return fig


if __name__ == "__main__":
    # Example usage (assumes X_train, train_targets, X_val, val_targets exist,
    # e.g. from data_processing.process_data):
    # result = train_models_by_max_depth(X_train, train_targets, X_val, val_targets)
    # plot_auroc_by_max_depth(result["train_metrics"], result["val_metrics"])
    # plt.show()
    pass