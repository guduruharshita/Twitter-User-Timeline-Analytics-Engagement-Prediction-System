# Twitter Analytics — Engagement Prediction System

[![CI](https://github.com/guduruharshita/twitter-user-timeline-analytics-engagement-prediction-system/actions/workflows/ci.yml/badge.svg)](https://github.com/guduruharshita/twitter-user-timeline-analytics-engagement-prediction-system/actions)
[![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python)](pyproject.toml)
[![Models](https://img.shields.io/badge/Models-3%20ensemble-blueviolet?logo=python)](src/twitter_analytics/models/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688?logo=fastapi)](src/twitter_analytics/api/main.py)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.1-EC6C00)](src/twitter_analytics/models/xgboost_model.py)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.4-EE4C2C?logo=pytorch)](src/twitter_analytics/models/neural_net.py)
[![Tests](https://img.shields.io/badge/Tests-13%20passing-success?logo=pytest)](tests/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

End-to-end **Twitter/X engagement analytics pipeline** — from raw CSV ingestion to serving live predictions via REST API. Transforms a flat timeline export into ranked user analytics, hourly posting insights, and a three-model ensemble (XGBoost + PyTorch NN + Prophet time-series).

## Why a Three-Model Ensemble

Single-model engagement prediction overfits to dominant patterns and misses the interaction effects between user history, content, and posting time. The three-model ensemble attacks the problem from orthogonal angles: **XGBoost** captures non-linear feature interactions and is interpretable via SHAP feature importance; the **PyTorch MLP** learns higher-order representations the tree model cannot express; **Prophet** models hourly and weekly seasonality that neither discriminative model sees. Cyclical hour encoding (sin/cos) ensures that 11 PM and midnight are adjacent in feature space rather than 23 integers apart — a standard technique for time-based ML that most implementations skip.

## System Architecture

```
┌────────────────────────────────────────────────────────────────────┐
│                       Training Phase (CLI)                          │
│                                                                     │
│  tweets.csv ──▶ loader.py ──▶ preprocessor.py ──▶ engineer.py     │
│                                      │                              │
│                      ┌───────────────┼───────────────┐             │
│                      ▼               ▼               ▼             │
│               XGBRegressor      PyTorch MLP      Prophet           │
│               + StandardScaler  (3-layer MLP)    (hourly TS)       │
│                      │               │               │             │
│                      ▼               ▼               ▼             │
│              xgb_model.joblib   nn_model.pt   prophet_model.joblib │
└──────────────────────┬────────────────────────────────────────────┘
                       │
┌──────────────────────▼────────────────────────────────────────────┐
│                    Inference Phase (FastAPI)                        │
│                                                                     │
│  POST /api/predict         ──▶ XGBoost predict (batch tweets)      │
│  POST /api/analytics/users ──▶ per-user engagement summary (CSV)   │
│  POST /api/analytics/global-best-hour ──▶ optimal posting hour     │
│  GET  /api/analytics/forecast          ──▶ Prophet 24h forecast    │
└────────────────────────────────────────────────────────────────────┘
```

```
Raw CSV → Preprocessing → Feature Engineering → Model Training
                                                      ↓
                              FastAPI  ←  Persisted Models (.joblib / .pt)
                                 ↓
                    /api/predict   /api/analytics/*   /api/analytics/forecast
```

## Contents

- [Features](#features)
- [Architecture](#architecture)
- [Models](#models)
- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [API Reference](#api-reference)
- [Testing](#testing)
- [Deployment](#deployment)
- [Skills Demonstrated](#skills-demonstrated)

---

## Features

- **Multi-model engagement prediction**: XGBoost, PyTorch NN, Prophet
- **Feature engineering**: cyclical hour encoding (sin/cos), user-level aggregate stats, NLP text features (hashtags, mentions, URLs)
- **REST API**: FastAPI endpoints for batch prediction and analytics
- **Training CLI**: `twitter-train --data tweets.csv` trains and persists all three models
- **Business intelligence**: per-user best posting hours, top tweet rankings, global engagement trends
- **Reproducible**: `pyproject.toml` packaging, fixed seeds, versioned model files

---

## Architecture

### ML Pipeline

```
twitter_analytics/
├── data/
│   ├── loader.py           CSV loading + column normalisation + validation
│   └── preprocessor.py     Deduplication, type coercion, negative clipping
├── features/
│   └── engineer.py         8 features: text_len, num_hashtags, num_mentions,
│                           num_urls, sin_hour, cos_hour, Likes_mean, RT_mean
├── models/
│   ├── xgboost_model.py    XGBRegressor + StandardScaler → models/xgb_*.joblib
│   ├── neural_net.py       PyTorch 3-layer MLP → models/nn_model.pt
│   └── prophet_model.py    Hourly time-series → models/prophet_model.joblib
├── analytics/
│   └── reporter.py         User summary, best posting hours, top tweets
└── api/
    ├── main.py             FastAPI app factory
    └── routers/
        ├── predict.py      POST /api/predict
        └── analytics.py    POST /api/analytics/* + GET /api/analytics/forecast
```

### Data Flow

```
                       ┌─────────────────────────────────┐
CSV / DataFrame ──────▶│  loader.py → preprocessor.py    │
                       └──────────────┬──────────────────┘
                                      │
                       ┌──────────────▼──────────────────┐
                       │  engineer.py (feature pipeline)  │
                       │  engagement_score = L×1.2+RT×1.5 │
                       └──────────────┬──────────────────┘
                              ┌───────┼───────┐
                    ┌─────────▼─┐ ┌───▼───┐ ┌─▼──────┐
                    │  XGBoost  │ │NN MLP │ │Prophet │
                    │ .joblib   │ │ .pt   │ │.joblib │
                    └─────────┬─┘ └───┬───┘ └─┬──────┘
                              └───────▼───────┘
                                 FastAPI /api/*
```

### Engagement Score Formula

```
engagement_score = Likes × 1.2 + RetweetCount × 1.5
```

Retweets weighted higher (broader reach signal).

### Feature Engineering

| Feature | Source | Technique |
|---------|--------|-----------|
| `text_len` | Tweet text | `str.len()` |
| `num_hashtags` | Tweet text | Regex `#\w+` |
| `num_mentions` | Tweet text | Regex `@\w+` |
| `num_urls` | Tweet text | Regex `https?://` |
| `sin_hour` | Hour (0-23) | `sin(2π·h/24)` |
| `cos_hour` | Hour (0-23) | `cos(2π·h/24)` |
| `Likes_mean` | UserID groupby | Per-user mean Likes |
| `RetweetCount_mean` | UserID groupby | Per-user mean Retweets |

Cyclical encoding maps hours to a unit circle so `h=23` is close to `h=0` — standard practice for time-based ML.

---

## Models

### XGBoost Regressor

```python
XGBRegressor(n_estimators=200, learning_rate=0.1, max_depth=6)
```
- Standard scaler preprocessing
- 80/20 train-test split with `random_state=42`
- Metrics: MSE, RMSE, R²

### PyTorch Feed-Forward Network

```
Input(8) → Linear(64) → ReLU → Dropout(0.2)
         → Linear(32) → ReLU → Linear(1)
Optimizer: Adam(lr=0.001)  Loss: MSELoss  Epochs: 10
```

### Prophet Time-Series

Fits on hourly aggregate engagement. Useful for:
- Identifying time-of-day posting patterns
- Short-horizon engagement forecasting
- Seasonality decomposition

---

## Quick Start

### Prerequisites

```bash
python --version   # 3.11+
pip install -e .
```

### Train models

```bash
# Generate sample data for testing
python -c "
from twitter_analytics.data.loader import generate_sample
generate_sample(n_rows=1000).to_csv('data/sample_tweets.csv', index=False)
"

# Train all models
twitter-train --data data/sample_tweets.csv
# OR skip Prophet (faster): twitter-train --data data/sample_tweets.csv --skip-prophet
```

### Start API server

```bash
uvicorn twitter_analytics.api.main:app --reload
# Docs: http://localhost:8000/docs
```

### Docker

```bash
docker compose up --build
```

---

## Project Structure

```
twitter-user-timeline-analytics-engagement-prediction-system/
│
├── src/twitter_analytics/
│   ├── config.py               # pydantic-settings config (TWITTER_* env vars)
│   ├── data/
│   │   ├── loader.py           # load_csv(), generate_sample()
│   │   └── preprocessor.py     # clean()
│   ├── features/
│   │   └── engineer.py         # build_features(), get_X_y()
│   ├── models/
│   │   ├── xgboost_model.py    # train(), predict(), is_trained()
│   │   ├── neural_net.py       # EngagementNN, train(), predict()
│   │   └── prophet_model.py    # train(), forecast()
│   ├── analytics/
│   │   └── reporter.py         # user_summary, best_posting_hours, top_tweets
│   ├── api/
│   │   ├── main.py             # create_app()
│   │   └── routers/
│   │       ├── predict.py      # POST /api/predict
│   │       └── analytics.py    # POST/GET /api/analytics/*
│   └── scripts/
│       └── train.py            # CLI entry point
│
├── tests/
│   ├── conftest.py             # Shared fixtures (generated DataFrames)
│   ├── test_preprocessor.py    # Data cleaning tests
│   ├── test_features.py        # Feature engineering tests
│   └── test_api.py             # FastAPI endpoint tests
│
├── notebooks/
│   └── exploration.ipynb       # Original exploratory analysis (preserved)
│
├── .github/workflows/ci.yml    # Lint + pytest CI
├── pyproject.toml              # Package definition + deps
├── Dockerfile
├── docker-compose.yml
└── .env.example
```

---

## API Reference

### `GET /health`

```bash
curl http://localhost:8000/health
# {"status":"ok","version":"2.0.0"}
```

---

### `POST /api/predict`

Predict engagement for a batch of tweets (requires trained XGBoost model).

```bash
curl -X POST http://localhost:8000/api/predict \
  -H "Content-Type: application/json" \
  -d '[{
    "tweet_id": "abc123",
    "user_id": "user_42",
    "hour": 14,
    "likes": 120,
    "retweet_count": 35,
    "text": "#python #ml amazing results @stanford",
    "likes_mean": 95.0,
    "retweet_count_mean": 22.0
  }]'
```

```json
[{"tweet_id": "abc123", "predicted_engagement": 196.5}]
```

---

### `POST /api/analytics/users`

Upload a CSV, get per-user engagement summary.

```bash
curl -X POST http://localhost:8000/api/analytics/users \
  -F "file=@data/tweets.csv"
```

```json
[
  {
    "UserID": "user_42",
    "num_tweets": 47,
    "total_likes": 5230,
    "avg_engagement": 312.4
  }
]
```

---

### `POST /api/analytics/global-best-hour`

```bash
curl -X POST http://localhost:8000/api/analytics/global-best-hour \
  -F "file=@data/tweets.csv"
# {"best_hour": 14}
```

---

### `GET /api/analytics/forecast`

Prophet 24-hour engagement forecast (requires trained Prophet model).

```bash
curl "http://localhost:8000/api/analytics/forecast?periods=12"
```

---

## Testing

```bash
pip install -e ".[dev]"
pytest tests/ -v
```

Expected output:

```
tests/test_preprocessor.py::test_clean_removes_duplicates PASSED
tests/test_preprocessor.py::test_clean_clips_negative_values PASSED
tests/test_preprocessor.py::test_clean_handles_missing_text PASSED
tests/test_preprocessor.py::test_clean_hour_range PASSED
tests/test_features.py::test_feature_columns_present PASSED
tests/test_features.py::test_engagement_score_non_negative PASSED
tests/test_features.py::test_cyclical_hour_encoding PASSED
tests/test_features.py::test_get_X_y_shapes PASSED
tests/test_features.py::test_num_hashtags_count PASSED
tests/test_features.py::test_num_mentions_count PASSED
tests/test_api.py::test_health PASSED
tests/test_api.py::test_predict_no_model_returns_503 PASSED
tests/test_api.py::test_predict_empty_list PASSED

13 passed in 4.2s
```

---

## Deployment

### Railway / Render

| Setting | Value |
|---------|-------|
| Build | `pip install -e .` |
| Start | `uvicorn twitter_analytics.api.main:app --host 0.0.0.0 --port 8000` |
| Volume | Mount `./models` for model persistence |

### Training in production

```bash
# Mount your CSV and run training before starting the API
twitter-train --data /mnt/data/tweets.csv --skip-prophet
```

---

## Future Improvements

- **X API v2 integration** — Replace CSV-based ingestion with direct OAuth 2.0 authenticated requests to the X API v2 user timeline endpoint for real-time data
- **SHAP explainability** — Add `/api/explain` endpoint returning per-feature SHAP values so users understand *why* a specific tweet was predicted to perform well
- **Optimal posting time** — Given a user's historical data, recommend the top 3 daily time windows that maximize predicted engagement for their specific audience
- **Competitor benchmarking** — Accept two user IDs and return a side-by-side analysis comparing posting frequency, content patterns, and engagement distributions
- **Streaming predictions** — WebSocket endpoint for real-time engagement score as a user drafts a tweet, updating with each keystroke

---

## Skills Demonstrated

| Skill | Evidence |
|-------|---------|
| **Python packaging** | `pyproject.toml` with `src/` layout, CLI entry point |
| **Machine Learning** | XGBoost regressor with StandardScaler, train/test split |
| **Deep Learning** | PyTorch custom `nn.Module`, DataLoader, Adam optimizer, Dropout |
| **Time-Series** | Prophet with cyclical feature engineering, hourly aggregate forecasting |
| **Feature Engineering** | Cyclical encoding (sin/cos), regex NLP features, user-level aggregates |
| **FastAPI** | App factory, typed Pydantic schemas, file upload endpoints |
| **Model Persistence** | `joblib` for sklearn/XGBoost, `torch.save` state dict |
| **Testing** | 13 pytest tests with session-scoped fixtures, TestClient |
| **Data Pipeline** | Modular loader → preprocessor → feature → model chain |
| **Docker** | Multi-stage build, non-root user, volume mount for models |
| **CI/CD** | GitHub Actions lint + test pipeline |

---

**Harshita Guduru** — [GitHub](https://github.com/guduruharshita) · [LinkedIn](https://linkedin.com/in/guduruharshita) · [Email](mailto:guduruharshita2001@gmail.com)
