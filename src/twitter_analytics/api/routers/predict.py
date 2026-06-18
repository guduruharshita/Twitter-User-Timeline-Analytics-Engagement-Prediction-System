"""POST /api/predict — predict engagement for a batch of tweets."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from twitter_analytics.features.engineer import FEATURE_COLS, build_features
from twitter_analytics.models import xgboost_model

import pandas as pd

router = APIRouter(prefix="/api/predict", tags=["predict"])


class TweetInput(BaseModel):
    tweet_id: str = Field(..., description="Unique tweet identifier")
    user_id: str
    hour: int = Field(..., ge=0, le=23)
    likes: float = Field(default=0.0, ge=0)
    retweet_count: float = Field(default=0.0, ge=0)
    text: str = Field(default="")
    likes_mean: float = Field(default=0.0, ge=0)
    retweet_count_mean: float = Field(default=0.0, ge=0)


class PredictionOutput(BaseModel):
    tweet_id: str
    predicted_engagement: float


@router.post("", response_model=list[PredictionOutput])
def predict_engagement(tweets: list[TweetInput]) -> list[PredictionOutput]:
    if not xgboost_model.is_trained():
        raise HTTPException(
            status_code=503,
            detail="Model not trained yet. Run the training pipeline first.",
        )
    if not tweets:
        return []

    import numpy as np

    rows = [
        {
            "TweetID": t.tweet_id,
            "UserID": t.user_id,
            "Hour": t.hour,
            "Likes": t.likes,
            "RetweetCount": t.retweet_count,
            "text": t.text,
            "Likes_mean": t.likes_mean,
            "RetweetCount_mean": t.retweet_count_mean,
        }
        for t in tweets
    ]
    df = pd.DataFrame(rows)
    df["sin_hour"] = np.sin(2 * np.pi * df["Hour"] / 24)
    df["cos_hour"] = np.cos(2 * np.pi * df["Hour"] / 24)
    df["text_len"]     = df["text"].str.len().fillna(0)
    df["num_hashtags"] = df["text"].str.count(r"#\w+").fillna(0)
    df["num_mentions"] = df["text"].str.count(r"@\w+").fillna(0)
    df["num_urls"]     = df["text"].str.count(r"https?://\S+").fillna(0)

    preds = xgboost_model.predict(df[FEATURE_COLS])

    return [
        PredictionOutput(tweet_id=t.tweet_id, predicted_engagement=float(p))
        for t, p in zip(tweets, preds)
    ]
