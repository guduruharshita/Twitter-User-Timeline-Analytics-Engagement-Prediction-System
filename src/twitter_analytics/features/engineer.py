"""Feature engineering for Twitter engagement prediction."""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

from twitter_analytics.config import settings

FEATURE_COLS = [
    "text_len",
    "num_hashtags",
    "num_mentions",
    "num_urls",
    "sin_hour",
    "cos_hour",
    "Likes_mean",
    "RetweetCount_mean",
]

TARGET_COL = "engagement_score"


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add engineered features and engagement score to DataFrame."""
    df = df.copy()

    # Text features
    df["text_len"]      = df["text"].str.len().fillna(0)
    df["num_hashtags"]  = df["text"].str.count(r"#\w+").fillna(0)
    df["num_mentions"]  = df["text"].str.count(r"@\w+").fillna(0)
    df["num_urls"]      = df["text"].str.count(r"https?://\S+").fillna(0)

    # Cyclical hour encoding
    df["sin_hour"] = np.sin(2 * np.pi * df["Hour"] / 24)
    df["cos_hour"] = np.cos(2 * np.pi * df["Hour"] / 24)

    # User-level aggregate features (mean Likes/Retweets per user)
    user_means = df.groupby("UserID")[["Likes", "RetweetCount"]].transform("mean")
    df["Likes_mean"]        = user_means["Likes"]
    df["RetweetCount_mean"] = user_means["RetweetCount"]

    # Engagement score
    df[TARGET_COL] = (
        df["Likes"] * settings.engagement_likes_weight
        + df["RetweetCount"] * settings.engagement_retweet_weight
    )

    return df


def get_X_y(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    return df[FEATURE_COLS], df[TARGET_COL]
