import argparse
import sys
from pathlib import Path

import joblib
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.feature_groups import numeric_features
from src.paths import (
    FEATURES_PATH,
    MARKET_WDL_MODEL_PATH,
    NO_MARKET_WDL_MODEL_PATH,
    POISSON_MODEL_PATH,
    PROCESSED_DATA_DIR,
)


IDENTITY_COLUMNS = ["date", "league", "home_team", "away_team"]


def _load_features(features_path: Path) -> pd.DataFrame:
    if not features_path.exists():
        raise FileNotFoundError(f"Feature file not found: {features_path}")
    features = pd.read_csv(features_path, encoding="utf-8")
    missing = [column for column in IDENTITY_COLUMNS + ["result"] if column not in features.columns]
    if missing:
        raise ValueError(f"Feature file is missing required columns: {missing}")
    features["date"] = pd.to_datetime(features["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    return features


def _predict_wdl(features: pd.DataFrame, model_path: Path, prefix: str) -> pd.DataFrame:
    artifact = joblib.load(model_path)
    X = numeric_features(features, artifact["feature_columns"])
    probabilities = artifact["model"].predict_proba(X)
    output = features[IDENTITY_COLUMNS].copy()

    for class_name in ["H", "D", "A"]:
        class_index = list(artifact["classes"]).index(class_name)
        output[f"{prefix}_{class_name}"] = probabilities[:, class_index]

    return output


def _predict_poisson(features: pd.DataFrame, model_path: Path) -> pd.DataFrame:
    artifact = joblib.load(model_path)
    X = numeric_features(features, artifact["feature_cols"])
    output = features[IDENTITY_COLUMNS].copy()
    output["poisson_home_xg"] = artifact["home_model"].predict(X)
    output["poisson_away_xg"] = artifact["away_model"].predict(X)
    return output


def _actual_results(features: pd.DataFrame) -> pd.DataFrame:
    output = features[IDENTITY_COLUMNS].copy()
    output["actual_result"] = features["result"]
    return output


def export_prediction_csvs(
    features_path: Path = FEATURES_PATH,
    output_dir: Path = PROCESSED_DATA_DIR,
) -> dict[str, Path]:
    features = _load_features(features_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    outputs = {
        "market": output_dir / "market_predictions.csv",
        "nomarket": output_dir / "nomarket_predictions.csv",
        "poisson": output_dir / "poisson_predictions.csv",
        "actual": output_dir / "actual_results.csv",
    }

    _predict_wdl(features, MARKET_WDL_MODEL_PATH, "market").to_csv(outputs["market"], index=False, encoding="utf-8")
    _predict_wdl(features, NO_MARKET_WDL_MODEL_PATH, "nomarket").to_csv(outputs["nomarket"], index=False, encoding="utf-8")
    _predict_poisson(features, POISSON_MODEL_PATH).to_csv(outputs["poisson"], index=False, encoding="utf-8")
    _actual_results(features).to_csv(outputs["actual"], index=False, encoding="utf-8")
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=Path, default=FEATURES_PATH)
    parser.add_argument("--output-dir", type=Path, default=PROCESSED_DATA_DIR)
    args = parser.parse_args()

    outputs = export_prediction_csvs(features_path=args.features, output_dir=args.output_dir)
    print("已輸出模型預測 CSV：")
    for path in outputs.values():
        print(path)


if __name__ == "__main__":
    main()
