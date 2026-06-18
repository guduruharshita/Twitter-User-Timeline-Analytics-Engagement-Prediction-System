from fastapi.testclient import TestClient

from twitter_analytics.api.main import app

client = TestClient(app)


def test_health():
    res = client.get("/health")
    assert res.status_code == 200
    assert res.json()["status"] == "ok"


def test_predict_no_model_returns_503():
    """Before training, /api/predict should return 503."""
    payload = [
        {
            "tweet_id": "1",
            "user_id": "u1",
            "hour": 9,
            "likes": 100,
            "retweet_count": 20,
            "text": "#hello @world",
            "likes_mean": 50.0,
            "retweet_count_mean": 10.0,
        }
    ]
    res = client.post("/api/predict", json=payload)
    assert res.status_code == 503


def test_predict_empty_list():
    """Empty payload should return an empty list (200)."""
    res = client.post("/api/predict", json=[])
    assert res.status_code == 200
    assert res.json() == []
