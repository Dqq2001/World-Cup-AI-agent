import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, log_loss

from src.paths import META_FEATURES_PATH, MODELS_DIR, XGB_META_MODEL_PATH

try:
    from xgboost import XGBClassifier
except ImportError as exc:
    raise ImportError("Missing dependency: install xgboost with `pip install xgboost`.") from exc


REQUIRED_COLUMNS = [
    "date",
    "league",
    "home_team",
    "away_team",
    "market_H",
    "market_D",
    "market_A",
    "nomarket_H",
    "nomarket_D",
    "nomarket_A",
    "poisson_home_xg",
    "poisson_away_xg",
    "actual_result",
]

CLASS_TO_ID = {"H": 0, "D": 1, "A": 2}
ID_TO_CLASS = {value: key for key, value in CLASS_TO_ID.items()}
PICK_TO_ID = CLASS_TO_ID

BASE_FEATURE_COLUMNS = [
    "market_H",
    "market_D",
    "market_A",
    "nomarket_H",
    "nomarket_D",
    "nomarket_A",
    "poisson_home_xg",
    "poisson_away_xg",
    "poisson_diff",
    "market_pick_encoded",
    "nomarket_pick_encoded",
    "market_confidence",
    "nomarket_confidence",
    "models_agree",
    "market_nomarket_disagree",
]


def validate_columns(data: pd.DataFrame) -> None:
    missing_columns = [column for column in REQUIRED_COLUMNS if column not in data.columns]
    if missing_columns:
        raise ValueError(f"CSV is missing required columns: {missing_columns}")


def load_meta_data(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"Meta-model training CSV not found: {csv_path}")

    data = pd.read_csv(csv_path, encoding="utf-8")
    validate_columns(data)
    return data


def add_meta_features(data: pd.DataFrame) -> pd.DataFrame:
    data = data.copy()
    data["date"] = pd.to_datetime(data["date"], errors="coerce")

    if data["date"].isna().any():
        bad_rows = data.index[data["date"].isna()].tolist()
        raise ValueError(f"Invalid date values found at rows: {bad_rows[:10]}")

    data["actual_result"] = data["actual_result"].astype(str).str.upper()
    invalid_results = sorted(set(data["actual_result"]) - set(CLASS_TO_ID))
    if invalid_results:
        raise ValueError(f"actual_result contains invalid classes: {invalid_results}")

    numeric_columns = [
        "market_H",
        "market_D",
        "market_A",
        "nomarket_H",
        "nomarket_D",
        "nomarket_A",
        "poisson_home_xg",
        "poisson_away_xg",
    ]
    for column in numeric_columns:
        data[column] = pd.to_numeric(data[column], errors="coerce")

    if data[numeric_columns].isna().any().any():
        bad_columns = data[numeric_columns].columns[data[numeric_columns].isna().any()].tolist()
        raise ValueError(f"Numeric feature columns contain missing or invalid values: {bad_columns}")

    market_columns = ["market_H", "market_D", "market_A"]
    nomarket_columns = ["nomarket_H", "nomarket_D", "nomarket_A"]

    data["poisson_diff"] = data["poisson_away_xg"] - data["poisson_home_xg"]
    data["market_pick"] = data[market_columns].idxmax(axis=1).str.replace("market_", "", regex=False)
    data["nomarket_pick"] = data[nomarket_columns].idxmax(axis=1).str.replace("nomarket_", "", regex=False)
    data["market_confidence"] = data[market_columns].max(axis=1)
    data["nomarket_confidence"] = data[nomarket_columns].max(axis=1)
    data["models_agree"] = (data["market_pick"] == data["nomarket_pick"]).astype(int)
    data["market_nomarket_disagree"] = (data["market_pick"] != data["nomarket_pick"]).astype(int)
    data["market_pick_encoded"] = data["market_pick"].map(PICK_TO_ID)
    data["nomarket_pick_encoded"] = data["nomarket_pick"].map(PICK_TO_ID)
    data["target"] = data["actual_result"].map(CLASS_TO_ID)

    return data.sort_values("date").reset_index(drop=True)


def chronological_split(data: pd.DataFrame, test_size: float = 0.2) -> tuple[pd.DataFrame, pd.DataFrame]:
    split_index = int(len(data) * (1 - test_size))
    if split_index <= 0 or split_index >= len(data):
        raise ValueError("Not enough rows for chronological train/test split.")
    return data.iloc[:split_index].copy(), data.iloc[split_index:].copy()


def train_xgb_meta_model(train_data: pd.DataFrame) -> XGBClassifier:
    model = XGBClassifier(
        objective="multi:softprob",
        num_class=3,
        eval_metric="mlogloss",
        n_estimators=300,
        learning_rate=0.05,
        max_depth=3,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=42,
    )
    model.fit(train_data[BASE_FEATURE_COLUMNS], train_data["target"])
    return model


def _decode_predictions(predictions: np.ndarray) -> list[str]:
    return [ID_TO_CLASS[int(prediction)] for prediction in predictions]


def evaluate_model(model: XGBClassifier, test_data: pd.DataFrame) -> dict:
    X_test = test_data[BASE_FEATURE_COLUMNS]
    y_test = test_data["target"]

    probabilities = model.predict_proba(X_test)
    predictions = model.predict(X_test)
    xgb_result_predictions = _decode_predictions(predictions)

    market_predictions = test_data["market_pick"]
    nomarket_predictions = test_data["nomarket_pick"]
    disagreement_mask = test_data["market_nomarket_disagree"] == 1

    disagreement_accuracy = np.nan
    if disagreement_mask.any():
        disagreement_accuracy = accuracy_score(
            test_data.loc[disagreement_mask, "actual_result"],
            np.array(xgb_result_predictions)[disagreement_mask.to_numpy()],
        )

    return {
        "accuracy": accuracy_score(y_test, predictions),
        "log_loss": log_loss(y_test, probabilities, labels=[0, 1, 2]),
        "confusion_matrix": confusion_matrix(y_test, predictions, labels=[0, 1, 2]),
        "classification_report": classification_report(
            y_test,
            predictions,
            labels=[0, 1, 2],
            target_names=["H", "D", "A"],
            zero_division=0,
        ),
        "market_accuracy": accuracy_score(test_data["actual_result"], market_predictions),
        "nomarket_accuracy": accuracy_score(test_data["actual_result"], nomarket_predictions),
        "disagreement_xgb_accuracy": disagreement_accuracy,
        "disagreement_count": int(disagreement_mask.sum()),
    }


def save_model(model: XGBClassifier, model_path: Path) -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model.save_model(model_path)


def print_metrics(metrics: dict, model_path: Path) -> None:
    print("XGBoost meta-model 評估結果")
    print(f"accuracy: {metrics['accuracy']:.4f}")
    print(f"log_loss: {metrics['log_loss']:.4f}")
    print(f"market vs XGBoost accuracy: {metrics['market_accuracy']:.4f} vs {metrics['accuracy']:.4f}")
    print(f"no-market vs XGBoost accuracy: {metrics['nomarket_accuracy']:.4f} vs {metrics['accuracy']:.4f}")
    if np.isnan(metrics["disagreement_xgb_accuracy"]):
        print("market 和 no-market 無分歧樣本，無法計算分歧時 XGBoost accuracy。")
    else:
        print(
            "market 和 no-market 分歧時的 XGBoost accuracy: "
            f"{metrics['disagreement_xgb_accuracy']:.4f} "
            f"({metrics['disagreement_count']} matches)"
        )
    print("confusion matrix (rows=true H/D/A, cols=pred H/D/A):")
    print(metrics["confusion_matrix"])
    print("classification report:")
    print(metrics["classification_report"])
    print(f"模型已儲存到: {model_path}")


def run_training(csv_path: Path, model_path: Path) -> dict:
    raw_data = load_meta_data(csv_path)
    data = add_meta_features(raw_data)
    train_data, test_data = chronological_split(data)
    model = train_xgb_meta_model(train_data)
    metrics = evaluate_model(model, test_data)
    save_model(model, model_path)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, default=META_FEATURES_PATH)
    parser.add_argument("--model-path", type=Path, default=XGB_META_MODEL_PATH)
    args = parser.parse_args()

    metrics = run_training(args.csv, args.model_path)
    print_metrics(metrics, args.model_path)


if __name__ == "__main__":
    main()
