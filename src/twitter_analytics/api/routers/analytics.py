"""GET /api/analytics/* — on-demand analytics from an uploaded dataset."""
from __future__ import annotations

import io
import logging

import pandas as pd
from fastapi import APIRouter, HTTPException, UploadFile

from twitter_analytics.analytics.reporter import (
    best_posting_hours,
    global_best_hour,
    top_tweets,
    user_summary,
)
from twitter_analytics.data.loader import load_csv
from twitter_analytics.data.preprocessor import clean
from twitter_analytics.features.engineer import build_features
from twitter_analytics.models import prophet_model

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/analytics", tags=["analytics"])


def _process_upload(file: UploadFile) -> pd.DataFrame:
    contents = file.file.read()
    df = pd.read_csv(io.BytesIO(contents), low_memory=False)
    df.columns = df.columns.str.strip()
    df = clean(df)
    df = build_features(df)
    return df


@router.post("/users")
def get_user_summary(file: UploadFile) -> list[dict]:
    df = _process_upload(file)
    return user_summary(df).to_dict(orient="records")


@router.post("/hours")
def get_best_hours(file: UploadFile) -> list[dict]:
    df = _process_upload(file)
    return best_posting_hours(df).to_dict(orient="records")


@router.post("/top-tweets")
def get_top_tweets(file: UploadFile, limit: int = 20) -> list[dict]:
    df = _process_upload(file)
    return top_tweets(df, limit=limit).to_dict(orient="records")


@router.post("/global-best-hour")
def get_global_best_hour(file: UploadFile) -> dict[str, int]:
    df = _process_upload(file)
    return {"best_hour": global_best_hour(df)}


@router.get("/forecast")
def get_forecast(periods: int = 24) -> list[dict]:
    if not prophet_model.is_trained():
        raise HTTPException(status_code=503, detail="Prophet model not trained yet.")
    df = prophet_model.forecast(periods=periods)
    return df.to_dict(orient="records")
