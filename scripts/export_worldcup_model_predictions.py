import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.international_features import ID_TO_RESULT
from src.international_fixture_features import build_fixture_features, fixture_model_matrix, load_fixtures, load_history
from src.paths import INTERNATIONAL_WDL_MODEL_PATH, PROCESSED_DATA_DIR

try:
    from xgboost import XGBClassifier
except ImportError as exc:
    raise ImportError("Missing dependency: install xgboost with `pip install xgboost`.") from exc


DEFAULT_OUTPUT = PROCESSED_DATA_DIR / "worldcup_model_predictions.csv"
STANDARD_COLUMNS = ["date", "group", "home_team", "away_team", "model_H", "model_D", "model_A"]


def export_predictions(fixtures_path: Path | None, model_path: Path, output_path: Path):
    if not model_path.exists():
        raise FileNotFoundError(f"找不到國家隊 H/D/A 模型: {model_path}")

    fixtures = load_fixtures(fixtures_path)
    features = build_fixture_features(fixtures, load_history())
    model = XGBClassifier()
    model.load_model(model_path)
    probabilities = model.predict_proba(fixture_model_matrix(features))

    classes = [int(value) for value in getattr(model, "classes_", [0, 1, 2])]
    identity_cols = [column for column in ["date", "group", "home_team", "away_team"] if column in features.columns]
    rows = features[identity_cols].copy()
    for class_id, probability_column in zip(classes, probabilities.T):
        rows[f"model_{ID_TO_RESULT[class_id]}"] = probability_column

    for column in ["model_H", "model_D", "model_A"]:
        if column not in rows.columns:
            rows[column] = 0.0
    for column in STANDARD_COLUMNS:
        if column not in rows.columns:
            rows[column] = ""
    rows["date"] = pd.to_datetime(rows["date"], errors="coerce").dt.date.astype(str)
    for column in ["group", "home_team", "away_team"]:
        rows[column] = rows[column].fillna("").astype(str).str.strip()
    rows = rows[STANDARD_COLUMNS]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows.to_csv(output_path, index=False, encoding="utf-8")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixtures", type=Path, default=None)
    parser.add_argument("--model", type=Path, default=INTERNATIONAL_WDL_MODEL_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    try:
        predictions = export_predictions(args.fixtures, args.model, args.output)
    except (FileNotFoundError, ValueError) as exc:
        raise SystemExit(f"錯誤: {exc}") from exc

    print(f"已輸出 World Cup H/D/A 模型預測: {args.output}")
    print(f"輸出筆數: {len(predictions)}")
    print(f"MODEL_PREDICTIONS_ROWS={len(predictions)}")
    print(f"MODEL_PREDICTIONS_COLUMNS={','.join(predictions.columns)}")
    example = predictions[
        (predictions["home_team"].astype(str).str.casefold() == "portugal")
        & (predictions["away_team"].astype(str).str.casefold() == "spain")
    ]
    if not example.empty:
        print(f"MODEL_EXAMPLE_PORTUGAL_VS_SPAIN={example.iloc[-1].to_dict()}")


if __name__ == "__main__":
    main()
