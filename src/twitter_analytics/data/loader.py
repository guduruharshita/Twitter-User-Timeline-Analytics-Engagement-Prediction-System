"""Data loading and validation for Twitter CSV data."""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

REQUIRED_COLUMNS = {"Hour", "UserID", "TweetID", "RetweetCount", "Likes"}
OPTIONAL_COLUMNS = {"Weekday", "text"}


def load_csv(path: str | Path, low_memory: bool = False) -> pd.DataFrame:
    """Load and do minimal normalisation on a Twitter CSV file."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")

    df = pd.read_csv(path, low_memory=low_memory)

    # Normalise column names
    df.columns = df.columns.str.strip()

    # Resolve duplicate column names
    seen: dict[str, int] = {}
    new_cols = []
    for col in df.columns:
        if col in seen:
            seen[col] += 1
            new_cols.append(f"{col}__{seen[col]}")
        else:
            seen[col] = 0
            new_cols.append(col)
    df.columns = new_cols

    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing required columns: {missing}")

    df = df.dropna(how="all")
    df = df.dropna(subset=list(REQUIRED_COLUMNS))

    logger.info("Loaded %d rows from %s", len(df), path)
    return df


def generate_sample(n_rows: int = 500, seed: int = 42) -> pd.DataFrame:
    """Generate a synthetic Twitter dataset for testing."""
    import numpy as np

    rng = np.random.default_rng(seed)
    n_users = 50
    user_ids = [f"user_{i}" for i in range(n_users)]

    return pd.DataFrame(
        {
            "TweetID": range(n_rows),
            "UserID": rng.choice(user_ids, n_rows),
            "Hour": rng.integers(0, 24, n_rows),
            "Weekday": rng.integers(0, 7, n_rows),
            "Likes": rng.integers(0, 1000, n_rows).astype(float),
            "RetweetCount": rng.integers(0, 300, n_rows).astype(float),
            "text": [
                f"Sample tweet #{i} #analytics @mention http://example.com"
                if rng.random() > 0.5
                else f"Plain tweet #{i}"
                for i in range(n_rows)
            ],
        }
    )
