"""Business-intelligence aggregations on the engagement-scored DataFrame."""
from __future__ import annotations

import pandas as pd

from twitter_analytics.config import settings


def user_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Per-user engagement statistics."""
    return (
        df.groupby("UserID")
        .agg(
            num_tweets=("TweetID", "count"),
            total_likes=("Likes", "sum"),
            total_retweets=("RetweetCount", "sum"),
            avg_likes=("Likes", "mean"),
            avg_retweets=("RetweetCount", "mean"),
            avg_engagement=("engagement_score", "mean"),
        )
        .reset_index()
        .sort_values("avg_engagement", ascending=False)
    )


def best_posting_hours(df: pd.DataFrame) -> pd.DataFrame:
    """Per-user best posting hour (highest avg engagement)."""
    hour_eng = (
        df.groupby(["UserID", "Hour"])["engagement_score"]
        .mean()
        .reset_index()
        .rename(columns={"engagement_score": "avg_hour_engagement"})
    )
    idx = hour_eng.groupby("UserID")["avg_hour_engagement"].idxmax()
    return hour_eng.loc[idx].reset_index(drop=True)


def top_tweets(df: pd.DataFrame, limit: int | None = None) -> pd.DataFrame:
    """Return the top `limit` tweets ranked by engagement score."""
    limit = limit or settings.top_tweets_limit
    cols = ["TweetID", "UserID", "engagement_score", "Hour"]
    if "text" in df.columns:
        cols.append("text")
    return df[cols].sort_values("engagement_score", ascending=False).head(limit).reset_index(drop=True)


def global_best_hour(df: pd.DataFrame) -> int:
    """Hour with the highest average engagement across all users."""
    return int(df.groupby("Hour")["engagement_score"].mean().idxmax())
