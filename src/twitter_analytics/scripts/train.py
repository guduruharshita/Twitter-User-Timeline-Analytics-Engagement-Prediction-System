"""CLI: train all models on a Twitter CSV dataset.

Usage:
    python -m twitter_analytics.scripts.train --data data/tweets.csv
    twitter-train --data data/tweets.csv
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)


def main(argv: list[str] | None = None) -> None:
    import logging as _logging
    _logging.basicConfig(level="INFO", format="%(asctime)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="Train Twitter engagement models")
    parser.add_argument("--data", required=True, help="Path to tweets CSV file")
    parser.add_argument("--skip-prophet", action="store_true", help="Skip Prophet (slow)")
    args = parser.parse_args(argv)

    from twitter_analytics.data.loader import load_csv
    from twitter_analytics.data.preprocessor import clean
    from twitter_analytics.features.engineer import build_features, get_X_y
    from twitter_analytics.models import xgboost_model, neural_net, prophet_model

    logger.info("Loading data from %s", args.data)
    df = load_csv(args.data)
    df = clean(df)
    df = build_features(df)
    X, y = get_X_y(df)

    logger.info("Training XGBoost on %d samples, %d features …", len(X), X.shape[1])
    xgb_metrics = xgboost_model.train(X, y)
    logger.info("XGBoost → %s", xgb_metrics)

    logger.info("Training Neural Network …")
    nn_metrics = neural_net.train(X, y)
    logger.info("NN → %s", nn_metrics)

    if not args.skip_prophet:
        logger.info("Fitting Prophet time-series model …")
        prophet_metrics = prophet_model.train(df)
        logger.info("Prophet → %s", prophet_metrics)

    logger.info("All models saved to %s/", Path("models").resolve())


if __name__ == "__main__":
    main()
