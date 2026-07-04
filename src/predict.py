import argparse

import joblib
import pandas as pd

from src.feature_groups import FEATURE_GROUPS, numeric_features
from src.poisson_model import predict_scoreline_distribution
from src.paths import (
    FEATURES_PATH,
    MARKET_WDL_MODEL_PATH,
    NO_MARKET_WDL_MODEL_PATH,
    POISSON_MODEL_PATH,
)


MODEL_PATHS = {
    "market": MARKET_WDL_MODEL_PATH,
    "no_market": NO_MARKET_WDL_MODEL_PATH,
}


def _odds_available(feature_row: pd.DataFrame) -> bool:
    odds_columns = FEATURE_GROUPS["odds"]
    if any(column not in feature_row.columns for column in odds_columns):
        return False
    odds_values = numeric_features(feature_row, odds_columns).iloc[0]
    return bool(odds_values.notna().all())


def predict_wdl(feature_row: pd.DataFrame, model_type: str) -> pd.DataFrame:
    artifact = joblib.load(MODEL_PATHS[model_type])
    model = artifact["model"]
    classes = artifact["classes"]
    X = numeric_features(feature_row, artifact["feature_columns"])
    probabilities = model.predict_proba(X)[0]
    return pd.DataFrame({"result": classes, "probability": probabilities})


def predict_poisson(feature_row: pd.DataFrame) -> dict:
    return predict_scoreline_distribution(feature_row, model_path=POISSON_MODEL_PATH)


def _load_features() -> pd.DataFrame:
    features = pd.read_csv(FEATURES_PATH, encoding="utf-8")
    features["date"] = pd.to_datetime(features["date"], errors="coerce").dt.date.astype(str)
    return features


def _select_by_match_index(features: pd.DataFrame, match_index: int) -> pd.DataFrame | None:
    if match_index < 0 or match_index >= len(features):
        print(f"錯誤：match-index 超出範圍。可用範圍是 0 到 {len(features) - 1}。")
        return None
    return features.iloc[[match_index]]


def _select_by_match_details(
    features: pd.DataFrame,
    home_team: str,
    away_team: str,
    match_date: str,
) -> pd.DataFrame | None:
    candidates = features[
        (features["home_team"].str.lower() == home_team.lower())
        & (features["away_team"].str.lower() == away_team.lower())
        & (features["date"] == match_date)
    ]

    if candidates.empty:
        print("錯誤：找不到符合條件的比賽。")
        print(f"查詢條件：{match_date} {home_team} vs {away_team}")
        return None

    if len(candidates) > 1:
        print("找到多筆符合條件的比賽，請改用 --match-index 指定其中一筆：")
        print(candidates[["date", "league", "home_team", "away_team"]].to_string())
        return None

    return candidates


def select_match(args: argparse.Namespace) -> pd.DataFrame | None:
    features = _load_features()
    if args.match_index is not None:
        return _select_by_match_index(features, args.match_index)

    details = [args.home_team, args.away_team, args.date]
    if any(value is not None for value in details):
        if not all(details):
            print("錯誤：使用球隊與日期查詢時，必須同時提供 --home-team、--away-team、--date。")
            return None
        return _select_by_match_details(features, args.home_team, args.away_team, args.date)

    return features.tail(1)


def _print_match_header(feature_row: pd.DataFrame) -> None:
    row = feature_row.iloc[0]
    print("比賽資訊")
    print(f"日期: {row['date']}")
    print(f"聯賽: {row['league']}")
    print(f"主隊: {row['home_team']}")
    print(f"客隊: {row['away_team']}")


def _print_wdl_prediction(title: str, prediction: pd.DataFrame) -> None:
    print(f"\n{title}")
    print(prediction.to_string(index=False))


def _print_poisson_prediction(prediction: dict) -> None:
    print("\nPoisson 比分模型")
    print(f"主隊期望進球: {prediction['expected_home_goals']:.3f}")
    print(f"客隊期望進球: {prediction['expected_away_goals']:.3f}")
    print("前五比分:")
    for scoreline in prediction["top_5_scorelines"]:
        print(f"{scoreline['scoreline']}: {scoreline['probability']:.4f}")
    print(f"主勝機率: {prediction['home_win_probability']:.4f}")
    print(f"和局機率: {prediction['draw_probability']:.4f}")
    print(f"客勝機率: {prediction['away_win_probability']:.4f}")
    print(f"大於 2.5 球機率: {prediction['over_2_5_probability']:.4f}")
    print(f"雙方進球機率: {prediction['both_teams_to_score_probability']:.4f}")


def run_predictions(feature_row: pd.DataFrame, model_type: str | None = None) -> None:
    _print_match_header(feature_row)

    if model_type in (None, "market"):
        if _odds_available(feature_row):
            market_prediction = predict_wdl(feature_row, "market")
            _print_wdl_prediction("市場勝和負模型（使用模型: market）", market_prediction)
        else:
            print("\n市場模型：無賠率資料，已跳過。")

    if model_type in (None, "no_market"):
        no_market_prediction = predict_wdl(feature_row, "no_market")
        _print_wdl_prediction("非市場勝和負模型（使用模型: no_market）", no_market_prediction)

    if model_type in (None, "poisson"):
        poisson_prediction = predict_poisson(feature_row)
        _print_poisson_prediction(poisson_prediction)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--match-index", type=int)
    parser.add_argument("--home-team")
    parser.add_argument("--away-team")
    parser.add_argument("--date", help="比賽日期，格式 YYYY-MM-DD")
    parser.add_argument(
        "--model-type",
        choices=["market", "no_market", "poisson"],
        default=None,
        help="可選。未指定時會同時輸出 market/no_market/poisson。",
    )
    args = parser.parse_args()

    feature_row = select_match(args)
    if feature_row is None:
        return

    run_predictions(feature_row, model_type=args.model_type)


if __name__ == "__main__":
    main()
