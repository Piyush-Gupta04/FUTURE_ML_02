"""Feature extraction and text representation utilities.

Configures leakage-free TF-IDF vectorization transformers to be embedded
directly within scikit-learn Pipelines.
"""

from __future__ import annotations

import re
from typing import Callable, Optional
from sklearn.feature_extraction.text import TfidfVectorizer


def normalize_text(text: str) -> str:
    """Lightweight text normalizer for ticket text.

    Converts to lowercase, strips URLs, emails, special punctuation,
    while preserving standard ticket vocabulary and error codes.
    """
    if not isinstance(text, str):
        return ""
    text = text.lower()
    # Remove email addresses if any remain
    text = re.sub(r"\S+@\S+", " ", text)
    # Remove urls
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    # Remove non-alphanumeric except spaces
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)
    # Collapse multiple whitespaces
    return " ".join(text.split())


def get_tfidf_vectorizer(
    max_features: int = 5000,
    ngram_range: tuple[int, int] = (1, 2),
    min_df: int = 2,
    sublinear_tf: bool = True,
    preprocessor: Optional[Callable[[str], str]] = normalize_text,
) -> TfidfVectorizer:
    """Creates a configured TfidfVectorizer instance.

    Args:
        max_features: Maximum vocabulary size.
        ngram_range: Range of n-grams (unigrams, bigrams).
        min_df: Minimum document frequency threshold.
        sublinear_tf: Apply sublinear scaling (1 + log(tf)).
        preprocessor: Text normalization callable.

    Returns:
        Configured, unfitted TfidfVectorizer ready for Pipeline inclusion.
    """
    return TfidfVectorizer(
        preprocessor=preprocessor,
        stop_words="english",
        ngram_range=ngram_range,
        max_features=max_features,
        min_df=min_df,
        sublinear_tf=sublinear_tf,
    )
