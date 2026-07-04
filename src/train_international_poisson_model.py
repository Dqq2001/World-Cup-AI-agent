import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error

from src.international_features import chronological_split, prepare_feature_matrix
from src.paths import (
    INTERNATIONAL_POISSON_AWAY_PATH,
    INTERNATIONAL_POISSON_HOME_PATH,
    INTERNATIONAL_TRAINING_PATH,
    MODELS_DIR,
)

try:
    from xgboost import XGBRegressor
except ImportError as exc:
    raise ImportError("Missing dependency: install xgboost with `pip install xgboost`.") from exc


def load_training_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"International training data not found: {path}")
    data = pd.read_csv(path, encoding="utf-8")
    required = ["date", "home_goals", "away_goals", "sample_weight"]
    missing = [column for column in required if column not in data.columns]
    if missing:
        raise ValueError(f"International training data missing columns: {missing}")
    data["date"] = pd.to_datetime(data["date"], errors="coerce")
    data["home_goals"] = pd.to_numeric(data["home_goals"], errors="coerce")
    data["away_goals"] = pd.to_numeric(data["away_goals"], errors="coerce")
    return data.dropna(subset=["date", "home_goals", "away_goals"]).sort_values("date").reset_index(drop=True)


def build_model() -> XGBRegressor:
    return XGBRegressor(
        objective="count:poisson",
        eval_metric="poisson-nloglik",
        n_estimators=350,
        learning_rate=0.04,
        max_depth=3,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=42,
    )


def poisson_log_likelihood(actual: pd.Series, expected: np.ndarray) -> float:
    expected = np.clip(expected, 1e-9, None)
    values = actual.to_numpy()
    log_likelihood = values * np.log(expected) - expected - np.array([math.lgamma(value + 1) for value in values])
    return float(log_likelihood.mean())


def train(input_path: Path = INTERNATIONAL_TRAINING_PATH) -> dict:
    data = load_training_data(input_path)
    train_data, test_data = chronological_split(data)
    X_train = prepare_feature_matrix(train_data)
    X_test = prepare_feature_matrix(test_data)

    home_model = build_model()
    away_model = build_model()
    home_model.fit(X_train, train_data["home_goals"], sample_weight=train_data["sample_weight"])
    away_model.fit(X_train, train_data["away_goals"], sample_weight=train_data["sample_weight"])

    home_expected = home_model.predict(X_test)
    away_expected = away_model.predict(X_test)
    metrics = {
        "home_goals_mae": float(mean_absolute_error(test_data["home_goals"], home_expected)),
        "away_goals_mae": float(mean_absolute_error(test_data["away_goals"], away_expected)),
        "home_poisson_log_likelihood": poisson_log_likelihood(test_data["home_goals"], home_expected),
        "away_poisson_log_likelihood": poisson_log_likelihood(test_data["away_goals"], away_expected),
    }

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    home_model.save_model(INTERNATIONAL_POISSON_HOME_PATH)
    away_model.save_model(INTERNATIONAL_POISSON_AWAY_PATH)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=INTERNATIONAL_TRAINING_PATH)
    args = parser.parse_args()

    metrics = train(args.input)
    print("國家隊 Poisson 模型評估")
    for name, value in metrics.items():
        print(f"{name}: {value:.4f}")
    print(f"主隊進球模型已儲存到: {INTERNATIONAL_POISSON_HOME_PATH}")
    print(f"客隊進球模型已儲存到: {INTERNATIONAL_POISSON_AWAY_PATH}")


if __name__ == "__main__":
    main()
