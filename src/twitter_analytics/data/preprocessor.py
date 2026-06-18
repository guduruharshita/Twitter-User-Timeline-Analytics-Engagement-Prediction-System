"""Data cleaning and deduplication."""
from __future__ import annotations

import logging

import pandas as pd

logger = logging.getLogger(__name__)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Drop duplicates, coerce numeric columns, remove outliers."""
    original_len = len(df)

    df = df.drop_duplicates()

    for col in ("Likes", "RetweetCount"):
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).clip(lower=0)

    df["Hour"] = pd.to_numeric(df["Hour"], errors="coerce")
    df = df.dropna(subset=["Hour"])
    df["Hour"] = df["Hour"].astype(int) % 24

    if "text" not in df.columns:
        df["text"] = ""
    df["text"] = df["text"].fillna("")

    logger.info("Cleaned %d → %d rows (dropped %d)", original_len, len(df), original_len - len(df))
    return df.reset_index(drop=True)
