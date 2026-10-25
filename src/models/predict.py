"""Command-Line Interface (CLI) for support ticket classification and prioritization.

Accepts new ticket subject and description, runs inference through trained
scikit-learn pipelines, and displays both machine learning predictions
and deterministic rule-based baseline recommendations.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any, Dict, Optional

import joblib

from src.data.make_dataset import sanitize_text
from src.models.rule_based import RuleBasedClassifier

logger = logging.getLogger(__name__)


def load_artifacts(
    models_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """Loads trained model artifacts from the models directory."""
    if models_dir is None:
        models_dir = Path(__file__).resolve().parents[2] / "models"

    artifacts: Dict[str, Any] = {
        "rule_engine": RuleBasedClassifier(),
        "type_model": None,
        "p4_model": None,
        "p3_model": None,
    }

    type_model_path = models_dir / "ticket_type_model.joblib"
    p4_model_path = models_dir / "ticket_priority_model.joblib"
    p3_model_path = models_dir / "ticket_priority_3class_model.joblib"

    if type_model_path.is_file():
        artifacts["type_model"] = joblib.load(type_model_path)
    if p4_model_path.is_file():
        artifacts["p4_model"] = joblib.load(p4_model_path)
    if p3_model_path.is_file():
        artifacts["p3_model"] = joblib.load(p3_model_path)

    return artifacts


def predict_ticket(
    subject: str,
    description: str,
    artifacts: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Predicts category and priority for a single ticket text.

    Distinguishes machine-learned predictions from deterministic rule-based results.
    """
    if artifacts is None:
        artifacts = load_artifacts()

    clean_subj = sanitize_text(subject)
    clean_desc = sanitize_text(description)
    combined_text = f"{clean_subj}. {clean_desc}".strip()

    # Rule-Based deterministic inference
    rule_engine: RuleBasedClassifier = artifacts["rule_engine"]
    rule_results = rule_engine.predict_all(combined_text)

    # Machine Learning Pipeline predictions
    ml_type_pred = "N/A (Model not fitted)"
    ml_type_prob = None
    if artifacts["type_model"] is not None:
        model = artifacts["type_model"]
        ml_type_pred = model.predict([combined_text])[0]
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba([combined_text])[0]
            max_idx = probs.argmax()
            ml_type_prob = float(probs[max_idx])

    ml_p4_pred = "N/A (Model not fitted)"
    ml_p4_prob = None
    if artifacts["p4_model"] is not None:
        model = artifacts["p4_model"]
        ml_p4_pred = model.predict([combined_text])[0]
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba([combined_text])[0]
            max_idx = probs.argmax()
            ml_p4_prob = float(probs[max_idx])

    ml_p3_pred = "N/A (Model not fitted)"
    if artifacts["p3_model"] is not None:
        ml_p3_pred = artifacts["p3_model"].predict([combined_text])[0]

    return {
        "input_subject": subject,
        "input_description": description,
        "combined_text": combined_text,
        "ml_predictions": {
            "ticket_type": ml_type_pred,
            "type_confidence": f"{ml_type_prob:.2%}" if ml_type_prob else "N/A",
            "ticket_priority_4class": ml_p4_pred,
            "priority_4class_confidence": f"{ml_p4_prob:.2%}" if ml_p4_prob else "N/A",
            "ticket_priority_3class": ml_p3_pred,
        },
        "rule_based_baseline": rule_results,
    }


def format_prediction_report(results: Dict[str, Any]) -> str:
    """Formats prediction results into an easy-to-read terminal report."""
    ml = results["ml_predictions"]
    rb = results["rule_based_baseline"]

    lines = [
        "=" * 68,
        "       SUPPORT TICKET CLASSIFICATION & TRIAGE REPORT",
        "=" * 68,
        f"Input Subject     : {results['input_subject']}",
        f"Input Description : {results['input_description']}",
        "-" * 68,
        " [1] SUPERVISED MACHINE LEARNING MODEL PREDICTIONS",
        f"     * Predicted Category (Type) : {ml['ticket_type']} (Conf: {ml['type_confidence']})",
        f"     * Predicted Priority (4-tier): {ml['ticket_priority_4class']} (Conf: {ml['priority_4class_confidence']})",
        f"     * Mapped Priority (3-tier)   : {ml['ticket_priority_3class']}",
        "-" * 68,
        " [2] DETERMINISTIC RULE-BASED BASELINE (Intent & Urgency Rules)",
        f"     * Recommended Category      : {rb['predicted_type']} (Conf: {rb['type_confidence']})",
        f"     * Recommended Priority (4c) : {rb['priority_4class']}",
        f"       Rationale                : {rb['priority_4class_rationale']}",
        f"     * Recommended Priority (3c) : {rb['priority_3class']}",
        f"       Rationale                : {rb['priority_3class_rationale']}",
        "=" * 68,
    ]
    return "\n".join(lines)


def main() -> None:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(
        description="Predict ticket category and priority for incoming customer support tickets."
    )
    parser.add_argument(
        "--subject",
        "-s",
        type=str,
        default=None,
        help="Ticket subject line (e.g. 'Payment duplicate charge')",
    )
    parser.add_argument(
        "--description",
        "-d",
        type=str,
        default=None,
        help="Ticket description text (e.g. 'I was charged twice on my credit card. Please refund immediately.')",
    )

    args = parser.parse_args()

    subject = args.subject
    description = args.description

    if not subject:
        print("\nEnter Ticket Subject:")
        subject = input("> ").strip() or "Product setup"

    if not description:
        print("\nEnter Ticket Description:")
        description = input("> ").strip() or "I am having an issue with the product. Please assist."

    artifacts = load_artifacts()
    res = predict_ticket(subject, description, artifacts)
    print("\n" + format_prediction_report(res) + "\n")


if __name__ == "__main__":
    main()
