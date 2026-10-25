"""Deterministic Rule-Based Baseline Engine for Support Ticket Classification & Prioritization.

Provides explicit, transparent, keyword- and intent-driven classification
without machine learning weights. Used as a benchmark baseline and robust
fallback in production environments.
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple


# Keyword dictionaries mapped to canonical Ticket Types
CATEGORY_RULES: Dict[str, List[str]] = {
    "Refund request": [
        "refund", "money back", "reimburse", "reimbursement", "return item",
        "returned product", "credit back", "chargeback", "send back", "return policy",
    ],
    "Cancellation request": [
        "cancel", "cancellation", "terminate", "unsubscribe", "stop service",
        "deactivate account", "close account", "opt out", "discontinue",
    ],
    "Billing inquiry": [
        "billing", "invoice", "overcharge", "overcharged", "duplicate charge",
        "charged twice", "receipt", "payment failed", "payment declined", "credit card",
        "pricing", "subscription fee", "statement",
    ],
    "Technical issue": [
        "not working", "broken", "won't turn on", "does not turn on", "crash",
        "freezing", "freeze", "bug", "error code", "malfunction", "failure",
        "glitch", "screen black", "hardware issue", "connectivity", "network problem",
        "battery drained", "troubleshoot", "won't boot", "reboot",
    ],
    "Product inquiry": [
        "how to", "feature", "specs", "specification", "compatible", "compatibility",
        "manual", "guide", "warranty", "setup instructions", "recommendation",
        "is it possible", "how do i", "information regarding", "user guide",
    ],
}

# Urgency heuristics mapped to Priority levels
URGENCY_KEYWORDS = {
    "Critical": [
        "immediately", "emergency", "critical", "system down", "data loss",
        "security breach", "severe", "completely unusable", "danger", "urgent", "asap",
    ],
    "High": [
        "broken", "not working", "cannot access", "blocked", "deadline", "fails",
        "charged unauthorized", "major error", "stopped working", "cannot login",
    ],
    "Medium": [
        "issue", "problem", "glitch", "intermittent", "slow", "delay",
        "discrepancy", "difficulty", "trouble", "incorrect",
    ],
    "Low": [
        "question", "inquiry", "curious", "feedback", "suggestion", "minor",
        "cosmetic", "clarification", "how to", "manual", "general inquiry",
    ],
}


class RuleBasedClassifier:
    """Heuristic rule-based baseline classifier for tickets."""

    def __init__(
        self,
        category_rules: Optional[Dict[str, List[str]]] = None,
        urgency_rules: Optional[Dict[str, List[str]]] = None,
    ):
        self.category_rules = category_rules or CATEGORY_RULES
        self.urgency_rules = urgency_rules or URGENCY_KEYWORDS

    def predict_category(self, text: str) -> Tuple[str, float, Dict[str, int]]:
        """Predicts ticket category based on keyword density.

        Returns:
            Tuple of (predicted_category, confidence_score, keyword_matches_dict).
        """
        clean_text = text.lower()
        scores: Dict[str, int] = {}

        for category, keywords in self.category_rules.items():
            count = sum(1 for kw in keywords if re.search(r"\b" + re.escape(kw) + r"\b", clean_text))
            scores[category] = count

        max_count = max(scores.values())
        if max_count == 0:
            # Default fallback when no explicit keywords match
            return "Product inquiry", 0.20, scores

        best_category = max(scores, key=scores.get)
        confidence = min(0.95, 0.40 + (max_count * 0.15))
        return best_category, confidence, scores

    def predict_priority_4class(self, text: str) -> Tuple[str, str]:
        """Predicts 4-class priority (Critical, High, Medium, Low) based on urgency keywords.

        Returns:
            Tuple of (priority, rationale).
        """
        clean_text = text.lower()

        for level in ["Critical", "High", "Medium", "Low"]:
            matched = [kw for kw in self.urgency_rules[level] if re.search(r"\b" + re.escape(kw) + r"\b", clean_text)]
            if matched:
                return level, f"Urgency signal matched: {', '.join(matched)}"

        # Default fallback
        return "Medium", "No explicit urgency trigger found; defaulted to Medium standard priority."

    def predict_priority_3class(self, text: str) -> Tuple[str, str]:
        """Predicts 3-class priority (High, Medium, Low) where Critical & High map to High.

        Returns:
            Tuple of (priority_3class, rationale).
        """
        p4, rationale = self.predict_priority_4class(text)
        if p4 in ["Critical", "High"]:
            return "High", f"High priority tier assigned ({p4}): {rationale}"
        return p4, rationale

    def predict_all(self, text: str) -> Dict[str, str]:
        """Runs full heuristic evaluation on input text."""
        cat, conf, _ = self.predict_category(text)
        p4, r4 = self.predict_priority_4class(text)
        p3, r3 = self.predict_priority_3class(text)
        return {
            "predicted_type": cat,
            "type_confidence": f"{conf:.2f}",
            "priority_4class": p4,
            "priority_4class_rationale": r4,
            "priority_3class": p3,
            "priority_3class_rationale": r3,
        }
