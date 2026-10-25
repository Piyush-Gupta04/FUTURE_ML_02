"""Data ingestion, validation, and preprocessing pipeline.

Ensures strict separation of features and targets, excludes target leakage
and personally identifiable information (PII), and performs reproducible
stratified train/test splits.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)

# PII fields that must never be processed into modeling features or logged
PII_COLUMNS = [
    "Customer Name",
    "Customer Email",
    "Customer Age",
    "Customer Gender",
]

# Post-outcome lifecycle fields that cause target leakage if used as model inputs
LEAKAGE_COLUMNS = [
    "Ticket Status",
    "Resolution",
    "First Response Time",
    "Time to Resolution",
    "Customer Satisfaction Rating",
]

# Canonical priority mapping: maps 4-level priority to 3-level (High / Medium / Low)
# Rationale: Critical tickets represent urgent escalations requiring immediate action,
# aligning operationally with the 'High' severity tier in standard 3-tier triage systems.
PRIORITY_4_TO_3_MAPPING = {
    "Critical": "High",
    "High": "High",
    "Medium": "Medium",
    "Low": "Low",
}


def find_raw_dataset_path(base_dir: Optional[Path] = None) -> Path:
    """Locates the raw customer support tickets CSV file."""
    if base_dir is None:
        base_dir = Path(__file__).resolve().parents[2]

    candidates = [
        base_dir / "Data" / "Raw" / "customer_support_tickets.csv",
        base_dir / "data" / "raw" / "customer_support_tickets.csv",
        base_dir / "Data" / "customer_support_tickets.csv",
        base_dir / "data" / "customer_support_tickets.csv",
    ]

    for path in candidates:
        if path.is_file():
            return path

    raise FileNotFoundError(
        f"Raw dataset not found in expected locations. Checked: {[str(p) for p in candidates]}"
    )


def load_raw_data(file_path: Optional[Path] = None) -> pd.DataFrame:
    """Loads the raw customer support ticket dataset without logging sensitive rows."""
    path = file_path or find_raw_dataset_path()
    df = pd.read_csv(path)
    logger.info("Loaded raw dataset from %s with %d rows and %d columns", path, len(df), len(df.columns))
    return df


def sanitize_text(text: Optional[str]) -> str:
    """Sanitizes text by stripping whitespace and replacing template artifacts."""
    if text is None or pd.isna(text):
        return ""
    text_str = str(text).strip()
    # Normalize placeholder tokens frequently seen in synthetic/template tickets
    text_str = text_str.replace("{product_purchased}", "product")
    # Collapse multiple whitespace/newlines
    return " ".join(text_str.split())


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """Preprocesses the raw tickets into a clean DataFrame ready for NLP modeling.

    - Strips all Customer PII (Name, Email, Age, Gender).
    - Removes post-outcome fields that introduce target leakage.
    - Concatenates Subject and Description into a sanitized `text` feature.
    - Preserves ground-truth `ticket_type` and `ticket_priority`.
    - Creates a justified `priority_3class` mapping (High, Medium, Low).
    """
    clean_df = df.copy()

    # Verify expected raw text and target columns exist
    required_cols = ["Ticket Subject", "Ticket Description", "Ticket Type", "Ticket Priority"]
    for col in required_cols:
        if col not in clean_df.columns:
            raise KeyError(f"Missing required column in dataset: '{col}'")

    # Combine text inputs avoiding leakage
    subjects = clean_df["Ticket Subject"].apply(sanitize_text)
    descriptions = clean_df["Ticket Description"].apply(sanitize_text)

    # Build primary text input: "Subject: <subject>. Description: <description>"
    clean_df["text"] = subjects + ". " + descriptions
    clean_df["text"] = clean_df["text"].str.strip()

    # Standardize target names
    clean_df["ticket_type"] = clean_df["Ticket Type"].astype(str).str.strip()
    clean_df["ticket_priority"] = clean_df["Ticket Priority"].astype(str).str.strip()
    clean_df["priority_3class"] = clean_df["ticket_priority"].map(PRIORITY_4_TO_3_MAPPING)

    # Optional pre-outcome metadata feature
    if "Product Purchased" in clean_df.columns:
        clean_df["product_purchased"] = clean_df["Product Purchased"].fillna("Unknown").astype(str).str.strip()

    # Retain ticket_id if present for indexing, but remove PII & leakage fields
    keep_cols = ["text", "ticket_type", "ticket_priority", "priority_3class"]
    if "Ticket ID" in clean_df.columns:
        clean_df["ticket_id"] = clean_df["Ticket ID"]
        keep_cols.insert(0, "ticket_id")
    if "product_purchased" in clean_df.columns:
        keep_cols.append("product_purchased")

    # Filter out empty text rows if any
    clean_df = clean_df[clean_df["text"].str.len() > 0].copy()

    return clean_df[keep_cols]


def split_data(
    df: pd.DataFrame,
    target_col: str = "ticket_type",
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Performs a reproducible stratified train/test split on the given target column."""
    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=df[target_col],
    )
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)


def build_and_save_dataset(
    raw_path: Optional[Path] = None,
    output_dir: Optional[Path] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """End-to-end data pipeline saving processed sanitized train/test data."""
    if output_dir is None:
        output_dir = Path(__file__).resolve().parents[2] / "data" / "processed"
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_df = load_raw_data(raw_path)
    clean_df = preprocess_data(raw_df)

    train_df, test_df = split_data(clean_df, target_col="ticket_type")

    train_path = output_dir / "train.csv"
    test_path = output_dir / "test.csv"
    full_path = output_dir / "tickets_clean.csv"

    clean_df.to_csv(full_path, index=False)
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    logger.info("Saved processed clean data to %s", output_dir)
    return train_df, test_df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    train, test = build_and_save_dataset()
    print(f"Preprocessed dataset saved successfully. Train: {len(train)}, Test: {len(test)}")
