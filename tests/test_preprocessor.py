import pandas as pd
import pytest

from twitter_analytics.data.loader import generate_sample
from twitter_analytics.data.preprocessor import clean


def test_clean_removes_duplicates():
    df = generate_sample(n_rows=100)
    df = pd.concat([df, df.iloc[:10]], ignore_index=True)  # introduce 10 dupes
    cleaned = clean(df)
    assert len(cleaned) <= 100


def test_clean_clips_negative_values():
    df = generate_sample(n_rows=50)
    df.loc[0, "Likes"] = -100
    df.loc[1, "RetweetCount"] = -50
    cleaned = clean(df)
    assert (cleaned["Likes"] >= 0).all()
    assert (cleaned["RetweetCount"] >= 0).all()


def test_clean_handles_missing_text():
    df = generate_sample(n_rows=50)
    df.drop(columns=["text"], inplace=True)
    cleaned = clean(df)
    assert "text" in cleaned.columns
    assert cleaned["text"].notna().all()


def test_clean_hour_range():
    df = generate_sample(n_rows=100)
    cleaned = clean(df)
    assert cleaned["Hour"].between(0, 23).all()
