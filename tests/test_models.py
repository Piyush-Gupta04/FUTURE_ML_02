"""Unit tests for ML pipelines, metrics, and prediction interface."""

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.features.build_features import get_tfidf_vectorizer, normalize_text
from src.models.evaluate import compute_metrics
from src.models.predict import load_artifacts, predict_ticket


def test_normalize_text():
    """Test text normalization removes email, URL, and non-alphanumerics."""
    text = "Error at support@domain.com! Visit http://help.me now #123"
    norm = normalize_text(text)
    assert "support@domain.com" not in norm
    assert "http://help.me" not in norm
    assert "#" not in norm
    assert "error at" in norm
    assert "123" in norm


def test_compute_metrics():
    """Test that compute_metrics produces required accuracy, precision, recall, and macro-F1."""
    y_true = ["A", "B", "A", "C", "B"]
    y_pred = ["A", "B", "B", "C", "B"]
    labels = ["A", "B", "C"]

    metrics = compute_metrics(y_true, y_pred, labels=labels)

    assert "accuracy" in metrics
    assert "macro_precision" in metrics
    assert "macro_recall" in metrics
    assert "macro_f1" in metrics
    assert "confusion_matrix" in metrics
    assert metrics["accuracy"] == 4 / 5
    assert metrics["confusion_matrix"].shape == (3, 3)


def test_pipeline_fit_predict():
    """Test that TF-IDF pipeline fits inside pipeline and predicts without leakage."""
    texts = [
        "Refund my payment please",
        "The battery does not charge",
        "Cancel my subscription now",
        "Is this compatible with windows",
    ]
    labels = ["Refund", "Tech", "Cancel", "Inquiry"]

    pipe = Pipeline([
        ("tfidf", get_tfidf_vectorizer(max_features=100, min_df=1)),
        ("clf", LogisticRegression()),
    ])

    pipe.fit(texts, labels)
    preds = pipe.predict(["Need a refund for payment", "Battery issue"])
    assert len(preds) == 2
    assert isinstance(preds[0], str)


def test_predict_ticket_interface():
    """Test the full CLI predict_ticket function works end-to-end with loaded artifacts."""
    artifacts = load_artifacts()
    res = predict_ticket(
        subject="Payment duplicate charge",
        description="I noticed duplicate charge on my credit card. Please refund immediately.",
        artifacts=artifacts,
    )

    assert "ml_predictions" in res
    assert "rule_based_baseline" in res
    assert "ticket_type" in res["ml_predictions"]
    assert "ticket_priority_4class" in res["ml_predictions"]
    assert "ticket_priority_3class" in res["ml_predictions"]
    assert res["rule_based_baseline"]["predicted_type"] == "Billing inquiry"
