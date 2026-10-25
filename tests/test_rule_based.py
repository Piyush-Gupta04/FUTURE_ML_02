"""Unit tests for the heuristic rule-based baseline engine."""

import pytest
from src.models.rule_based import RuleBasedClassifier


@pytest.fixture
def classifier() -> RuleBasedClassifier:
    return RuleBasedClassifier()


def test_rule_based_category_refund(classifier):
    """Test refund keyword detection."""
    text = "Refund inquiry. I want my money back for this returned product."
    cat, conf, matches = classifier.predict_category(text)
    assert cat == "Refund request"
    assert conf > 0.5
    assert matches["Refund request"] >= 2


def test_rule_based_category_technical(classifier):
    """Test technical issue detection."""
    text = "Hardware bug. The display is broken and does not turn on."
    cat, conf, matches = classifier.predict_category(text)
    assert cat == "Technical issue"
    assert matches["Technical issue"] >= 2


def test_rule_based_category_billing(classifier):
    """Test billing inquiry detection."""
    text = "Invoice issue. I was charged twice on my credit card statement."
    cat, conf, matches = classifier.predict_category(text)
    assert cat == "Billing inquiry"


def test_rule_based_category_cancellation(classifier):
    """Test cancellation detection."""
    text = "Cancel subscription. I want to terminate my account immediately."
    cat, conf, matches = classifier.predict_category(text)
    assert cat == "Cancellation request"


def test_rule_based_priority_critical(classifier):
    """Test urgency keywords trigger Critical priority."""
    text = "System down! Emergency outage and data loss reported."
    p4, r4 = classifier.predict_priority_4class(text)
    assert p4 == "Critical"
    assert "emergency" in r4 or "data loss" in r4 or "system down" in r4

    p3, r3 = classifier.predict_priority_3class(text)
    assert p3 == "High"


def test_rule_based_priority_low(classifier):
    """Test inquiry keywords trigger Low priority."""
    text = "General inquiry. Question regarding how to use the feature."
    p4, r4 = classifier.predict_priority_4class(text)
    assert p4 == "Low"

    p3, r3 = classifier.predict_priority_3class(text)
    assert p3 == "Low"


def test_rule_based_fallback(classifier):
    """Test fallback when no explicit keywords are matched."""
    text = "Hello there lorem ipsum dolor sit amet."
    res = classifier.predict_all(text)
    assert res["predicted_type"] in ["Product inquiry", "Technical issue", "Refund request"]
    assert res["priority_4class"] == "Medium"
    assert res["priority_3class"] == "Medium"
