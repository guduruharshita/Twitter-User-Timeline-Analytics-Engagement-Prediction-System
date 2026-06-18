import numpy as np
import pandas as pd

from twitter_analytics.features.engineer import FEATURE_COLS, TARGET_COL, build_features, get_X_y


def test_feature_columns_present(featured_df: pd.DataFrame):
    for col in FEATURE_COLS:
        assert col in featured_df.columns, f"Missing feature: {col}"


def test_engagement_score_non_negative(featured_df: pd.DataFrame):
    assert (featured_df[TARGET_COL] >= 0).all()


def test_cyclical_hour_encoding(featured_df: pd.DataFrame):
    assert featured_df["sin_hour"].between(-1.0, 1.0).all()
    assert featured_df["cos_hour"].between(-1.0, 1.0).all()


def test_get_X_y_shapes(featured_df: pd.DataFrame):
    X, y = get_X_y(featured_df)
    assert X.shape[0] == y.shape[0]
    assert set(X.columns) == set(FEATURE_COLS)


def test_num_hashtags_count(clean_df: pd.DataFrame):
    df = clean_df.copy()
    df.loc[0, "text"] = "#python #ml #data"
    featured = build_features(df)
    assert featured.loc[0, "num_hashtags"] == 3


def test_num_mentions_count(clean_df: pd.DataFrame):
    df = clean_df.copy()
    df.loc[0, "text"] = "@alice @bob hello"
    featured = build_features(df)
    assert featured.loc[0, "num_mentions"] == 2
