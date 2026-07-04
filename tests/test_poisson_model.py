import pandas as pd

from src.features import add_rolling_features, get_model_feature_columns
from src.poisson_model import predict_scoreline_distribution, train_poisson_models


POST_MATCH_COLUMNS = {
    "home_goals",
    "away_goals",
    "home_shots",
    "away_shots",
    "home_shots_on_target",
    "away_shots_on_target",
    "home_corners",
    "away_corners",
}


def _tiny_feature_data() -> pd.DataFrame:
    matches = pd.DataFrame(
        [
            {
                "date": "2024-01-01",
                "league": "Premier League",
                "home_team": "A",
                "away_team": "B",
                "home_goals": 2,
                "away_goals": 0,
                "result": "H",
                "home_shots": 10,
                "away_shots": 3,
                "home_corners": 5,
                "away_corners": 1,
                "odds_home": 2.0,
                "odds_draw": 3.0,
                "odds_away": 4.0,
            },
            {
                "date": "2024-01-08",
                "league": "Premier League",
                "home_team": "B",
                "away_team": "C",
                "home_goals": 1,
                "away_goals": 1,
                "result": "D",
                "home_shots": 8,
                "away_shots": 7,
                "home_corners": 4,
                "away_corners": 3,
                "odds_home": 2.2,
                "odds_draw": 3.1,
                "odds_away": 3.5,
            },
            {
                "date": "2024-01-15",
                "league": "Premier League",
                "home_team": "C",
                "away_team": "A",
                "home_goals": 0,
                "away_goals": 2,
                "result": "A",
                "home_shots": 6,
                "away_shots": 9,
                "home_corners": 2,
                "away_corners": 6,
                "odds_home": 3.0,
                "odds_draw": 3.2,
                "odds_away": 2.4,
            },
            {
                "date": "2024-01-22",
                "league": "Premier League",
                "home_team": "A",
                "away_team": "C",
                "home_goals": 3,
                "away_goals": 1,
                "result": "H",
                "home_shots": 12,
                "away_shots": 5,
                "home_corners": 7,
                "away_corners": 2,
                "odds_home": 1.8,
                "odds_draw": 3.4,
                "odds_away": 4.2,
            },
        ]
    )
    return add_rolling_features(matches)


def test_poisson_model_can_train_on_tiny_sample_dataset() -> None:
    features = _tiny_feature_data()

    artifact = train_poisson_models(data=features, model_path=None)

    assert set(artifact) == {"home_model", "away_model", "feature_cols"}
    assert artifact["feature_cols"]


def test_poisson_prediction_returns_probabilities_and_top_5_scorelines() -> None:
    features = _tiny_feature_data()
    artifact = train_poisson_models(data=features, model_path=None)

    prediction = predict_scoreline_distribution(features.tail(1), model_artifact=artifact)

    assert prediction["expected_home_goals"] >= 0
    assert prediction["expected_away_goals"] >= 0
    assert len(prediction["top_5_scorelines"]) == 5
    assert all("scoreline" in row and "probability" in row for row in prediction["top_5_scorelines"])
    assert 0 <= prediction["home_win_probability"] <= 1
    assert 0 <= prediction["draw_probability"] <= 1
    assert 0 <= prediction["away_win_probability"] <= 1
    assert 0 <= prediction["over_2_5_probability"] <= 1
    assert 0 <= prediction["both_teams_to_score_probability"] <= 1


def test_training_feature_cols_exclude_current_match_post_match_columns() -> None:
    features = _tiny_feature_data()
    feature_cols = set(get_model_feature_columns(features))

    assert feature_cols.isdisjoint(POST_MATCH_COLUMNS)
    artifact = train_poisson_models(data=features, model_path=None)
    assert set(artifact["feature_cols"]).isdisjoint(POST_MATCH_COLUMNS)
