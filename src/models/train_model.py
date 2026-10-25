"""Model training, hyperparameter configuration, and comparative evaluation.

Trains and evaluates scikit-learn Pipelines using TF-IDF text features
across multiple classifiers (Logistic Regression, Linear SVC, Complement Naive Bayes)
and compares against the rule-based baseline.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Tuple

import joblib
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from src.data.make_dataset import build_and_save_dataset
from src.features.build_features import get_tfidf_vectorizer
from src.models.evaluate import compute_metrics, plot_confusion_matrix
from src.models.rule_based import RuleBasedClassifier

logger = logging.getLogger(__name__)


def create_candidate_pipelines() -> Dict[str, Pipeline]:
    """Builds candidate scikit-learn text classification pipelines."""
    return {
        "LogisticRegression": Pipeline(
            [
                ("tfidf", get_tfidf_vectorizer(max_features=6000, ngram_range=(1, 2))),
                (
                    "clf",
                    LogisticRegression(
                        C=1.0,
                        max_iter=1000,
                        class_weight="balanced",
                        random_state=42,
                    ),
                ),
            ]
        ),
        "LinearSVC": Pipeline(
            [
                ("tfidf", get_tfidf_vectorizer(max_features=6000, ngram_range=(1, 2))),
                (
                    "clf",
                    CalibratedClassifierCV(
                        LinearSVC(
                            C=0.5,
                            class_weight="balanced",
                            random_state=42,
                            dual="auto",
                        )
                    ),
                ),
            ]
        ),
        "ComplementNB": Pipeline(
            [
                ("tfidf", get_tfidf_vectorizer(max_features=6000, ngram_range=(1, 2))),
                ("clf", ComplementNB(alpha=0.5)),
            ]
        ),
    }


def evaluate_models_on_target(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target_col: str,
    output_fig_path: Path,
) -> Tuple[Dict[str, Any], Pipeline, str]:
    """Trains and compares multiple classifiers on a given target column.

    Returns:
        Tuple of (results_dict, best_fitted_pipeline, best_model_name).
    """
    X_train = train_df["text"]
    y_train = train_df[target_col]
    X_test = test_df["text"]
    y_test = test_df[target_col]

    unique_labels = sorted(y_train.unique().tolist())
    candidates = create_candidate_pipelines()
    results: Dict[str, Any] = {}

    best_macro_f1 = -1.0
    best_name = ""
    best_pipe: Pipeline = None  # type: ignore

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    logger.info("Evaluating models for target: '%s'", target_col)

    for name, pipeline in candidates.items():
        # Perform Stratified 5-Fold Cross-Validation on training data
        cv_scores = cross_validate(
            pipeline,
            X_train,
            y_train,
            cv=cv,
            scoring=["accuracy", "f1_macro"],
            n_jobs=-1,
        )

        # Fit on full training set
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)

        metrics = compute_metrics(y_test, y_pred, labels=unique_labels)
        metrics["cv_accuracy_mean"] = float(cv_scores["test_accuracy"].mean())
        metrics["cv_accuracy_std"] = float(cv_scores["test_accuracy"].std())
        metrics["cv_macro_f1_mean"] = float(cv_scores["test_f1_macro"].mean())
        metrics["cv_macro_f1_std"] = float(cv_scores["test_f1_macro"].std())

        results[name] = metrics

        logger.info(
            "[%s] %s -> Test Acc: %.4f, Macro-F1: %.4f (CV F1: %.4f +/- %.4f)",
            target_col,
            name,
            metrics["accuracy"],
            metrics["macro_f1"],
            metrics["cv_macro_f1_mean"],
            metrics["cv_macro_f1_std"],
        )

        if metrics["macro_f1"] > best_macro_f1:
            best_macro_f1 = metrics["macro_f1"]
            best_name = name
            best_pipe = pipeline

    # Also evaluate Rule-Based Baseline
    rule_engine = RuleBasedClassifier()
    if target_col == "ticket_type":
        y_rule_pred = [rule_engine.predict_category(t)[0] for t in X_test]
    elif target_col == "ticket_priority":
        y_rule_pred = [rule_engine.predict_priority_4class(t)[0] for t in X_test]
    elif target_col == "priority_3class":
        y_rule_pred = [rule_engine.predict_priority_3class(t)[0] for t in X_test]
    else:
        y_rule_pred = [unique_labels[0]] * len(X_test)

    rule_metrics = compute_metrics(y_test, y_rule_pred, labels=unique_labels)
    results["RuleBasedBaseline"] = rule_metrics
    logger.info(
        "[%s] RuleBasedBaseline -> Test Acc: %.4f, Macro-F1: %.4f",
        target_col,
        rule_metrics["accuracy"],
        rule_metrics["macro_f1"],
    )

    # Plot and save confusion matrix for the best model
    best_cm = results[best_name]["confusion_matrix"]
    plot_confusion_matrix(
        cm=best_cm,
        labels=unique_labels,
        title=f"Confusion Matrix: {target_col} ({best_name})",
        output_path=output_fig_path,
        normalize=True,
    )

    return results, best_pipe, best_name


def run_training_experiment(
    data_dir: Optional[Path] = None,
    models_dir: Optional[Path] = None,
    reports_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """Runs full reproducible training experiment across Category and Priority targets."""
    base_dir = Path(__file__).resolve().parents[2]
    models_dir = models_dir or (base_dir / "models")
    reports_dir = reports_dir or (base_dir / "reports")
    figures_dir = reports_dir / "figures"

    models_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    # Ingest and prepare clean train/test data
    train_df, test_df = build_and_save_dataset()

    full_evaluation: Dict[str, Any] = {}

    # 1. Train Ticket Category Classifier
    type_results, type_best_model, type_best_name = evaluate_models_on_target(
        train_df=train_df,
        test_df=test_df,
        target_col="ticket_type",
        output_fig_path=figures_dir / "cm_ticket_type.png",
    )
    joblib.dump(type_best_model, models_dir / "ticket_type_model.joblib")
    full_evaluation["ticket_type"] = {
        "best_model": type_best_name,
        "results": {
            k: {
                m: v
                for m, v in val.items()
                if m not in ["confusion_matrix", "report_dict"]
            }
            for k, val in type_results.items()
        },
    }

    # 2. Train Ticket Priority Classifier (4-class original: Critical, High, Medium, Low)
    p4_results, p4_best_model, p4_best_name = evaluate_models_on_target(
        train_df=train_df,
        test_df=test_df,
        target_col="ticket_priority",
        output_fig_path=figures_dir / "cm_ticket_priority.png",
    )
    joblib.dump(p4_best_model, models_dir / "ticket_priority_model.joblib")
    full_evaluation["ticket_priority_4class"] = {
        "best_model": p4_best_name,
        "results": {
            k: {
                m: v
                for m, v in val.items()
                if m not in ["confusion_matrix", "report_dict"]
            }
            for k, val in p4_results.items()
        },
    }

    # 3. Train Ticket Priority Classifier (3-class mapped: High, Medium, Low)
    p3_results, p3_best_model, p3_best_name = evaluate_models_on_target(
        train_df=train_df,
        test_df=test_df,
        target_col="priority_3class",
        output_fig_path=figures_dir / "cm_priority_3class.png",
    )
    joblib.dump(p3_best_model, models_dir / "ticket_priority_3class_model.joblib")
    full_evaluation["ticket_priority_3class"] = {
        "best_model": p3_best_name,
        "results": {
            k: {
                m: v
                for m, v in val.items()
                if m not in ["confusion_matrix", "report_dict"]
            }
            for k, val in p3_results.items()
        },
    }

    # Save metrics JSON to reports
    metrics_path = reports_dir / "metrics_summary.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(full_evaluation, f, indent=2)

    logger.info("Saved evaluation summary to %s", metrics_path)
    return full_evaluation


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    summary = run_training_experiment()
    print("\n=== TRAINING COMPLETED SUCCESSFULLY ===")
    print(json.dumps(summary, indent=2))
