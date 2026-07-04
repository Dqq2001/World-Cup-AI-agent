import argparse
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, log_loss

from src.international_features import RESULT_TO_ID, chronological_split, prepare_feature_matrix
from src.paths import INTERNATIONAL_TRAINING_PATH, INTERNATIONAL_WDL_MODEL_PATH, MODELS_DIR

try:
    from xgboost import XGBClassifier
except ImportError as exc:
    raise ImportError("Missing dependency: install xgboost with `pip install xgboost`.") from exc


def load_training_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"International training data not found: {path}")
    data = pd.read_csv(path, encoding="utf-8")
    required = ["date", "result", "sample_weight"]
    missing = [column for column in required if column not in data.columns]
    if missing:
        raise ValueError(f"International training data missing columns: {missing}")
    data["date"] = pd.to_datetime(data["date"], errors="coerce")
    data = data.dropna(subset=["date", "result"])
    data = data[data["result"].isin(RESULT_TO_ID)]
    return data.sort_values("date").reset_index(drop=True)


def build_model() -> XGBClassifier:
    return XGBClassifier(
        objective="multi:softprob",
        num_class=3,
        eval_metric="mlogloss",
        n_estimators=350,
        learning_rate=0.04,
        max_depth=3,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=42,
    )


def evaluate(model: XGBClassifier, data: pd.DataFrame) -> dict:
    X = prepare_feature_matrix(data)
    y = data["result"].map(RESULT_TO_ID)
    probabilities = model.predict_proba(X)
    predictions = model.predict(X)
    return {
        "accuracy": float(accuracy_score(y, predictions)),
        "log_loss": float(log_loss(y, probabilities, labels=[0, 1, 2])),
        "confusion_matrix": confusion_matrix(y, predictions, labels=[0, 1, 2]),
    }


def train(input_path: Path = INTERNATIONAL_TRAINING_PATH, model_path: Path = INTERNATIONAL_WDL_MODEL_PATH) -> dict:
    data = load_training_data(input_path)
    train_data, test_data = chronological_split(data)
    model = build_model()
    model.fit(
        prepare_feature_matrix(train_data),
        train_data["result"].map(RESULT_TO_ID),
        sample_weight=train_data["sample_weight"],
    )

    metrics = evaluate(model, test_data)
    worldcup_subset = test_data[test_data["is_worldcup"].astype(bool)]
    metrics["worldcup_subset"] = evaluate(model, worldcup_subset) if not worldcup_subset.empty else None

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model.save_model(model_path)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=INTERNATIONAL_TRAINING_PATH)
    parser.add_argument("--model", type=Path, default=INTERNATIONAL_WDL_MODEL_PATH)
    args = parser.parse_args()

    metrics = train(args.input, args.model)
    print("國家隊 H/D/A 模型評估")
    print(f"accuracy: {metrics['accuracy']:.4f}")
    print(f"log_loss: {metrics['log_loss']:.4f}")
    print("confusion matrix rows=true H/D/A cols=pred H/D/A:")
    print(metrics["confusion_matrix"])
    if metrics["worldcup_subset"] is not None:
        wc = metrics["worldcup_subset"]
        print("World Cup subset 評估")
        print(f"accuracy: {wc['accuracy']:.4f}")
        print(f"log_loss: {wc['log_loss']:.4f}")
        print(wc["confusion_matrix"])
    else:
        print("World Cup subset: test split 中沒有 World Cup matches。")
    print(f"模型已儲存到: {args.model}")


if __name__ == "__main__":
    main()
