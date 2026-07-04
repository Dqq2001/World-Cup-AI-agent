import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.international_fixture_features import build_fixture_features, fixture_model_matrix, load_fixtures, load_history, top_scorelines
from src.paths import INTERNATIONAL_POISSON_AWAY_PATH, INTERNATIONAL_POISSON_HOME_PATH, PROCESSED_DATA_DIR

try:
    from xgboost import XGBRegressor
except ImportError as exc:
    raise ImportError("Missing dependency: install xgboost with `pip install xgboost`.") from exc


DEFAULT_OUTPUT = PROCESSED_DATA_DIR / "worldcup_poisson_predictions.csv"


def load_regressor(path: Path) -> XGBRegressor:
    if not path.exists():
        raise FileNotFoundError(f"找不到國家隊 Poisson 模型: {path}")
    model = XGBRegressor()
    model.load_model(path)
    return model


def export_predictions(fixtures_path: Path | None, home_model_path: Path, away_model_path: Path, output_path: Path):
    fixtures = load_fixtures(fixtures_path)
    features = build_fixture_features(fixtures, load_history())
    X = fixture_model_matrix(features)

    home_model = load_regressor(home_model_path)
    away_model = load_regressor(away_model_path)
    home_xg = np.clip(home_model.predict(X), 0, None)
    away_xg = np.clip(away_model.predict(X), 0, None)

    identity_cols = [
        column
        for column in ["date", "group", "stage", "round", "match_id", "home_team", "away_team", "home_slot", "away_slot", "status"]
        if column in features.columns
    ]
    rows = features[identity_cols].copy()
    rows["poisson_home_xg"] = home_xg
    rows["poisson_away_xg"] = away_xg
    rows["poisson_top_scores"] = [
        json.dumps(top_scorelines(home_value, away_value), ensure_ascii=False)
        for home_value, away_value in zip(home_xg, away_xg)
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows.to_csv(output_path, index=False, encoding="utf-8")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixtures", type=Path, default=None)
    parser.add_argument("--home-model", type=Path, default=INTERNATIONAL_POISSON_HOME_PATH)
    parser.add_argument("--away-model", type=Path, default=INTERNATIONAL_POISSON_AWAY_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    try:
        predictions = export_predictions(args.fixtures, args.home_model, args.away_model, args.output)
    except (FileNotFoundError, ValueError) as exc:
        raise SystemExit(f"錯誤: {exc}") from exc

    print(f"已輸出 World Cup Poisson 預測: {args.output}")
    print(f"輸出筆數: {len(predictions)}")


if __name__ == "__main__":
    main()
