import pandas as pd
import pytest

from twitter_analytics.data.loader import generate_sample
from twitter_analytics.data.preprocessor import clean
from twitter_analytics.features.engineer import build_features


@pytest.fixture(scope="session")
def raw_df() -> pd.DataFrame:
    return generate_sample(n_rows=200, seed=42)


@pytest.fixture(scope="session")
def clean_df(raw_df: pd.DataFrame) -> pd.DataFrame:
    return clean(raw_df)


@pytest.fixture(scope="session")
def featured_df(clean_df: pd.DataFrame) -> pd.DataFrame:
    return build_features(clean_df)
