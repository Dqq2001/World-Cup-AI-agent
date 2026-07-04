import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, PoissonRegressor
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.poisson_model import _scoreline_grid
from src.paths import FEATURES_PATH


FEATURE_GROUPS = {
    "basic_form": [
        "home_recent_points_5",
        "away_recent_points_5",
        "home_recent_goals_for_5",
        "away_recent_goals_for_5",
        "home_recent_goals_against_5",
        "away_recent_goals_against_5",
        "home_win_rate_5",
        "away_win_rate_5",
    ],
    "elo": [
        "home_elo_before",
        "away_elo_before",
        "elo_diff_before",
    ],
    "rest_days": [
        "home_rest_days_capped",
        "away_rest_days_capped",
        "rest_days_diff_capped",
    ],
    "rolling_stats": [
        "home_recent_shots_5",
        "away_recent_shots_5",
        "home_recent_shots_on_target_5",
        "away_recent_shots_on_target_5",
        "home_recent_corners_5",
        "away_recent_corners_5",
        "home_recent_fouls_5",
        "away_recent_fouls_5",
        "home_recent_yellow_cards_5",
        "away_recent_yellow_cards_5",
        "home_recent_red_cards_5",
        "away_recent_red_cards_5",
    ],
    "odds": [
        "odds_implied_home_prob",
        "odds_implied_draw_prob",
        "odds_implied_away_prob",
    ],
}

CONFIGS = {
    "basic_form": ["basic_form"],
    "basic_form + elo": ["basic_form", "elo"],
    "basic_form + elo + rest_days": ["basic_form", "elo", "rest_days"],
    "basic_form + elo + rest_days + rolling_stats": [
        "basic_form",
        "elo",
        "rest_days",
        "rolling_stats",
    ],
    "odds only": ["odds"],
    "all features": ["basic_form", "elo", "rest_days", "rolling_stats", "odds"],
}


def _feature_cols(data: pd.DataFrame, group_names: list[str]) -> list[str]:
    cols = []
    for group_name in group_names:
        for column in FEATURE_GROUPS[group_name]:
            if column in data.columns and column not in cols:
                cols.append(column)
    return cols


def _numeric_features(data: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    return data[feature_cols].apply(pd.to_numeric, errors="coerce")


def _classification_pipeline() -> Pipeline:
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=1000)),
        ]
    )


def _poisson_pipeline() -> Pipeline:
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", PoissonRegressor(alpha=0.001, max_iter=1000)),
        ]
    )


def _brier_score(y_true: pd.Series, probabilities, classes: list[str]) -> float:
    scores = []
    for index, class_name in enumerate(classes):
        scores.append(brier_score_loss((y_true == class_name).astype(int), probabilities[:, index]))
    return float(sum(scores) / len(scores))


def evaluate_wdl_configs(data: pd.DataFrame) -> pd.DataFrame:
    data = data.dropna(subset=["result"]).copy()
    train_data, test_data = train_test_split(data, test_size=0.2, shuffle=False)
    rows = []

    for config_name, group_names in CONFIGS.items():
        feature_cols = _feature_cols(train_data, group_names)
        if not feature_cols:
            continue

        model = _classification_pipeline()
        model.fit(_numeric_features(train_data, feature_cols), train_data["result"])

        X_test = _numeric_features(test_data, feature_cols)
        probabilities = model.predict_proba(X_test)
        predictions = model.predict(X_test)
        classes = list(model.classes_)

        rows.append(
            {
                "config": config_name,
                "accuracy": float(accuracy_score(test_data["result"], predictions)),
                "log_loss": float(log_loss(test_data["result"], probabilities, labels=classes)),
                "brier_score": _brier_score(test_data["result"], probabilities, classes),
                "feature_count": len(feature_cols),
            }
        )

    return pd.DataFrame(rows).sort_values("log_loss", ascending=True).reset_index(drop=True)


def _poisson_predictions_to_wdl(home_expected, away_expected) -> tuple[list[str], list[str]]:
    results = []
    scorelines = []
    for home_goals_expected, away_goals_expected in zip(home_expected, away_expected):
        grid = _scoreline_grid(float(home_goals_expected), float(away_goals_expected), max_goals=5)
        home_win_probability = grid.loc[grid["home_goals"] > grid["away_goals"], "probability"].sum()
        draw_probability = grid.loc[grid["home_goals"] == grid["away_goals"], "probability"].sum()
        away_win_probability = grid.loc[grid["home_goals"] < grid["away_goals"], "probability"].sum()
        results.append(
            max(
                [
                    ("H", home_win_probability),
                    ("D", draw_probability),
                    ("A", away_win_probability),
                ],
                key=lambda item: item[1],
            )[0]
        )

        top_scoreline = grid.sort_values("probability", ascending=False).iloc[0]
        scorelines.append(f"{int(top_scoreline['home_goals'])}-{int(top_scoreline['away_goals'])}")

    return results, scorelines


def evaluate_poisson_configs(data: pd.DataFrame) -> pd.DataFrame:
    data = data.dropna(subset=["home_goals", "away_goals", "result"]).copy()
    train_data, test_data = train_test_split(data, test_size=0.2, shuffle=False)
    rows = []

    for config_name, group_names in CONFIGS.items():
        feature_cols = _feature_cols(train_data, group_names)
        if not feature_cols:
            continue

        home_model = _poisson_pipeline()
        away_model = _poisson_pipeline()
        X_train = _numeric_features(train_data, feature_cols)
        X_test = _numeric_features(test_data, feature_cols)

        home_model.fit(X_train, train_data["home_goals"])
        away_model.fit(X_train, train_data["away_goals"])

        home_expected = home_model.predict(X_test)
        away_expected = away_model.predict(X_test)
        predicted_results, predicted_scorelines = _poisson_predictions_to_wdl(home_expected, away_expected)
        actual_scorelines = (
            test_data["home_goals"].astype(int).astype(str)
            + "-"
            + test_data["away_goals"].astype(int).astype(str)
        )

        rows.append(
            {
                "config": config_name,
                "home_goals_mae": float(mean_absolute_error(test_data["home_goals"], home_expected)),
                "away_goals_mae": float(mean_absolute_error(test_data["away_goals"], away_expected)),
                "total_goals_mae": float(
                    mean_absolute_error(
                        test_data["home_goals"] + test_data["away_goals"],
                        home_expected + away_expected,
                    )
                ),
                "exact_score_accuracy": float(accuracy_score(actual_scorelines, predicted_scorelines)),
                "poisson_wdl_accuracy": float(accuracy_score(test_data["result"], predicted_results)),
                "feature_count": len(feature_cols),
            }
        )

    return pd.DataFrame(rows).sort_values("total_goals_mae", ascending=True).reset_index(drop=True)


def run_ablation(features_path: str = FEATURES_PATH) -> dict[str, pd.DataFrame]:
    data = pd.read_csv(features_path, encoding="utf-8")
    return {
        "wdl": evaluate_wdl_configs(data),
        "poisson": evaluate_poisson_configs(data),
    }


def main() -> None:
    results = run_ablation()

    print("勝和負特徵消融評估（依 log_loss 由低到高）")
    print(results["wdl"].to_string(index=False))

    print("\nPoisson 特徵消融評估（依 total_goals_mae 由低到高）")
    print(results["poisson"].to_string(index=False))


if __name__ == "__main__":
    main()
