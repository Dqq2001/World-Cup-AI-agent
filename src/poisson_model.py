import math

import joblib
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import PoissonRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.feature_groups import get_group_feature_columns, numeric_features
from src.features import get_model_feature_columns
from src.paths import FEATURES_PATH, MODELS_DIR, POISSON_MODEL_PATH


def _build_poisson_pipeline() -> Pipeline:
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", PoissonRegressor(alpha=0.001, max_iter=1000)),
        ]
    )


def train_poisson_models(
    features_path: str = FEATURES_PATH,
    model_path: str = POISSON_MODEL_PATH,
    data: pd.DataFrame | None = None,
    group_names: list[str] | None = None,
) -> dict:
    matches = data.copy() if data is not None else pd.read_csv(features_path, encoding="utf-8")
    train_data = matches.dropna(subset=["home_goals", "away_goals"]).copy()
    feature_cols = (
        get_group_feature_columns(train_data, group_names)
        if group_names is not None
        else get_model_feature_columns(train_data)
    )

    home_model = _build_poisson_pipeline()
    away_model = _build_poisson_pipeline()
    X = numeric_features(train_data, feature_cols)

    home_model.fit(X, train_data["home_goals"])
    away_model.fit(X, train_data["away_goals"])

    artifact = {
        "home_model": home_model,
        "away_model": away_model,
        "feature_cols": feature_cols,
    }

    if model_path is not None:
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump(artifact, model_path)

    return artifact


def _poisson_probability(goals: int, expected_goals: float) -> float:
    return math.exp(-expected_goals) * expected_goals**goals / math.factorial(goals)


def _scoreline_grid(expected_home_goals: float, expected_away_goals: float, max_goals: int) -> pd.DataFrame:
    rows = []
    for home_goals in range(max_goals + 1):
        for away_goals in range(max_goals + 1):
            probability = _poisson_probability(home_goals, expected_home_goals) * _poisson_probability(
                away_goals, expected_away_goals
            )
            rows.append(
                {
                    "home_goals": home_goals,
                    "away_goals": away_goals,
                    "scoreline": f"{home_goals}-{away_goals}",
                    "probability": probability,
                }
            )

    grid = pd.DataFrame(rows)
    total_probability = grid["probability"].sum()
    if total_probability > 0:
        grid["probability"] = grid["probability"] / total_probability
    return grid


def predict_scoreline_distribution(
    feature_row: pd.DataFrame,
    model_artifact: dict | None = None,
    model_path: str = POISSON_MODEL_PATH,
    max_goals: int = 5,
) -> dict:
    artifact = model_artifact if model_artifact is not None else joblib.load(model_path)
    X = numeric_features(feature_row, artifact["feature_cols"])

    expected_home_goals = float(artifact["home_model"].predict(X)[0])
    expected_away_goals = float(artifact["away_model"].predict(X)[0])
    grid = _scoreline_grid(expected_home_goals, expected_away_goals, max_goals)

    home_win_probability = float(grid.loc[grid["home_goals"] > grid["away_goals"], "probability"].sum())
    draw_probability = float(grid.loc[grid["home_goals"] == grid["away_goals"], "probability"].sum())
    away_win_probability = float(grid.loc[grid["home_goals"] < grid["away_goals"], "probability"].sum())
    over_2_5_probability = float(grid.loc[(grid["home_goals"] + grid["away_goals"]) > 2.5, "probability"].sum())
    both_teams_to_score_probability = float(
        grid.loc[(grid["home_goals"] > 0) & (grid["away_goals"] > 0), "probability"].sum()
    )

    top_5_scorelines = (
        grid.sort_values("probability", ascending=False)
        .head(5)[["scoreline", "probability"]]
        .to_dict("records")
    )

    return {
        "expected_home_goals": expected_home_goals,
        "expected_away_goals": expected_away_goals,
        "top_5_scorelines": top_5_scorelines,
        "home_win_probability": home_win_probability,
        "draw_probability": draw_probability,
        "away_win_probability": away_win_probability,
        "over_2_5_probability": over_2_5_probability,
        "both_teams_to_score_probability": both_teams_to_score_probability,
        "scoreline_grid": grid,
    }


def main() -> None:
    artifact = train_poisson_models()
    features = pd.read_csv(FEATURES_PATH, encoding="utf-8")
    prediction = predict_scoreline_distribution(features.tail(1), model_artifact=artifact)

    print(f"已儲存 Poisson 比分模型到 {POISSON_MODEL_PATH}")
    print(f"範例主隊期望進球: {prediction['expected_home_goals']:.3f}")
    print(f"範例客隊期望進球: {prediction['expected_away_goals']:.3f}")
    print("範例前五比分:")
    for scoreline in prediction["top_5_scorelines"]:
        print(f"{scoreline['scoreline']}: {scoreline['probability']:.4f}")
    print(f"主勝機率: {prediction['home_win_probability']:.4f}")
    print(f"和局機率: {prediction['draw_probability']:.4f}")
    print(f"客勝機率: {prediction['away_win_probability']:.4f}")
    print(f"大於 2.5 球機率: {prediction['over_2_5_probability']:.4f}")
    print(f"雙方進球機率: {prediction['both_teams_to_score_probability']:.4f}")


if __name__ == "__main__":
    main()
