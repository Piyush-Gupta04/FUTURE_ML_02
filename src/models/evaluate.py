"""Model evaluation metrics and diagnostic plotting utilities.

Computes accuracy, precision, recall, macro-F1, classification reports,
and generates confusion matrix visualizations saved to reports/figures/.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

logger = logging.getLogger(__name__)


def compute_metrics(
    y_true: Union[List[str], np.ndarray, pd.Series],
    y_pred: Union[List[str], np.ndarray, pd.Series],
    labels: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Computes comprehensive classification metrics.

    Args:
        y_true: True ground truth labels.
        y_pred: Predicted class labels.
        labels: Optional explicit ordering of unique class labels.

    Returns:
        Dictionary containing accuracy, macro and weighted precision/recall/F1,
        full classification report, and confusion matrix.
    """
    if labels is None:
        labels = sorted(list(set(y_true).union(set(y_pred))))

    acc = float(accuracy_score(y_true, y_pred))
    macro_prec = float(precision_score(y_true, y_pred, labels=labels, average="macro", zero_division=0))
    macro_rec = float(recall_score(y_true, y_pred, labels=labels, average="macro", zero_division=0))
    macro_f1 = float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0))

    weighted_f1 = float(f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0))

    report_dict = classification_report(
        y_true, y_pred, labels=labels, output_dict=True, zero_division=0
    )
    report_text = classification_report(
        y_true, y_pred, labels=labels, zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    return {
        "accuracy": acc,
        "macro_precision": macro_prec,
        "macro_recall": macro_rec,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "labels": labels,
        "report_dict": report_dict,
        "report_text": report_text,
        "confusion_matrix": cm,
    }


def plot_confusion_matrix(
    cm: np.ndarray,
    labels: List[str],
    title: str,
    output_path: Optional[Path] = None,
    normalize: bool = False,
    figsize: tuple[int, int] = (7, 6),
) -> plt.Figure:
    """Plots and optionally saves a heatmap visualization of the confusion matrix.

    Args:
        cm: Confusion matrix array.
        labels: Class label names.
        title: Plot title.
        output_path: Path to save figure (e.g. reports/figures/cm.png).
        normalize: Whether to normalize by row (recall per class).
        figsize: Figure dimensions.

    Returns:
        matplotlib Figure object.
    """
    if normalize:
        cm_display = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
        cm_display = np.nan_to_num(cm_display)
        fmt = ".2%"
    else:
        cm_display = cm
        fmt = "d"

    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        cm_display,
        annot=True,
        fmt=fmt,
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        cbar=True,
        ax=ax,
    )
    ax.set_title(title, fontsize=12, fontweight="bold", pad=12)
    ax.set_xlabel("Predicted Label", fontsize=11, fontweight="semibold")
    ax.set_ylabel("True Ground-Truth Label", fontsize=11, fontweight="semibold")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        logger.info("Saved confusion matrix plot to %s", output_path)

    return fig
