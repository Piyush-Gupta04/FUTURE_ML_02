"""Unit tests for data ingestion, PII stripping, and leakage prevention."""

import pandas as pd
import pytest

from src.data.make_dataset import (
    LEAKAGE_COLUMNS,
    PII_COLUMNS,
    PRIORITY_4_TO_3_MAPPING,
    preprocess_data,
    sanitize_text,
    split_data,
)


@pytest.fixture
def sample_raw_df() -> pd.DataFrame:
    """Creates a synthetic raw DataFrame matching the raw CSV schema."""
    return pd.DataFrame(
        {
            "Ticket ID": [101, 102, 103, 104],
            "Customer Name": ["Alice Smith", "Bob Jones", "Carol White", "David Lee"],
            "Customer Email": ["alice@example.com", "bob@test.com", "c@w.com", "d@l.com"],
            "Customer Age": [34, 45, 29, 52],
            "Customer Gender": ["Female", "Male", "Female", "Male"],
            "Product Purchased": ["iPhone", "GoPro Hero", "Dell XPS", "LG Smart TV"],
            "Date of Purchase": ["2023-01-01", "2023-02-01", "2023-03-01", "2023-04-01"],
            "Ticket Type": ["Technical issue", "Refund request", "Billing inquiry", "Technical issue"],
            "Ticket Subject": ["Network bug", "Return product", "Double billed", "Screen black"],
            "Ticket Description": [
                "My {product_purchased} network is failing.\n\nThanks.",
                "I want a refund for the item.",
                "Charged twice on credit card.",
                "Display does not turn on.",
            ],
            "Ticket Status": ["Closed", "Open", "In Progress", "Closed"],
            "Resolution": ["Rebooted device", None, None, "Replaced cable"],
            "Ticket Priority": ["Critical", "Low", "High", "Medium"],
            "Ticket Channel": ["Email", "Phone", "Chat", "Email"],
            "First Response Time": ["2023-01-01 10:00", None, "2023-03-01 11:00", None],
            "Time to Resolution": ["2023-01-01 12:00", None, None, "2023-04-01 15:00"],
            "Customer Satisfaction Rating": [4.0, None, None, 5.0],
        }
    )


def test_sanitize_text():
    """Test text sanitization cleans templates and collapses spaces."""
    raw = "  Issue with   {product_purchased}!\n\nPlease help.  "
    cleaned = sanitize_text(raw)
    assert "{product_purchased}" not in cleaned
    assert "Issue with product! Please help." == cleaned
    assert sanitize_text(None) == ""


def test_pii_and_leakage_exclusion(sample_raw_df):
    """Test that all PII and leakage fields are strictly excluded from processed output."""
    clean_df = preprocess_data(sample_raw_df)

    # Assert PII fields excluded
    for col in PII_COLUMNS:
        assert col not in clean_df.columns, f"PII column '{col}' leaked into features!"

    # Assert Leakage columns excluded
    for col in LEAKAGE_COLUMNS:
        assert col not in clean_df.columns, f"Leakage column '{col}' leaked into features!"

    # Assert primary text column exists and combines subject and description
    assert "text" in clean_df.columns
    assert "Network bug. My product network is failing. Thanks." in clean_df["text"].iloc[0]


def test_priority_mapping(sample_raw_df):
    """Test that priority is correctly mapped from 4-class to 3-class."""
    clean_df = preprocess_data(sample_raw_df)

    assert PRIORITY_4_TO_3_MAPPING["Critical"] == "High"
    assert PRIORITY_4_TO_3_MAPPING["High"] == "High"
    assert PRIORITY_4_TO_3_MAPPING["Medium"] == "Medium"
    assert PRIORITY_4_TO_3_MAPPING["Low"] == "Low"

    # Row 0 was Critical -> mapped to High
    assert clean_df["priority_3class"].iloc[0] == "High"
    # Row 1 was Low -> mapped to Low
    assert clean_df["priority_3class"].iloc[1] == "Low"
    # Row 2 was High -> mapped to High
    assert clean_df["priority_3class"].iloc[2] == "High"
    # Row 3 was Medium -> mapped to Medium
    assert clean_df["priority_3class"].iloc[3] == "Medium"


def test_split_data(sample_raw_df):
    """Test stratified split functionality."""
    clean_df = preprocess_data(sample_raw_df)
    # Duplicate rows to allow stratified splitting
    larger_df = pd.concat([clean_df] * 5, ignore_index=True)
    train_df, test_df = split_data(larger_df, target_col="ticket_type", test_size=0.2, random_state=42)

    assert len(train_df) + len(test_df) == len(larger_df)
    assert len(test_df) == int(len(larger_df) * 0.2)
