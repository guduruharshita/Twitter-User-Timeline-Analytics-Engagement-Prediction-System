from pathlib import Path
from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_dir: Path = Path("models")
    data_dir: Path = Path("data")
    log_level: str = "INFO"
    environment: str = "development"
    allowed_origins: str = "http://localhost:3000,http://localhost:5173"

    # Model hyperparameters
    xgb_n_estimators: int = 200
    xgb_learning_rate: float = 0.1
    xgb_max_depth: int = 6
    nn_hidden1: int = 64
    nn_hidden2: int = 32
    nn_epochs: int = 10
    nn_batch_size: int = 1024
    nn_lr: float = 0.001

    # Engagement weights
    engagement_likes_weight: float = 1.2
    engagement_retweet_weight: float = 1.5

    # API
    api_title: str = "Twitter Analytics API"
    api_version: str = "2.0.0"
    top_tweets_limit: int = 100

    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="TWITTER_",
        extra="ignore",
    )

    @property
    def cors_origins(self) -> list[str]:
        return [o.strip() for o in self.allowed_origins.split(",")]


settings = Settings()
