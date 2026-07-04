import pandas as pd
import pytest

from src.features import add_rolling_features


def test_rolling_features_only_use_past_matches() -> None:
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
                "odds_home": 2.0,
                "odds_draw": 3.0,
                "odds_away": 4.0,
            },
            {
                "date": "2024-01-08",
                "league": "Premier League",
                "home_team": "A",
                "away_team": "C",
                "home_goals": 1,
                "away_goals": 1,
                "result": "D",
                "odds_home": 2.0,
                "odds_draw": 3.0,
                "odds_away": 4.0,
            },
        ]
    )

    features = add_rolling_features(matches)

    assert features.loc[0, "home_recent_points_5"] == 0
    assert features.loc[0, "home_recent_goals_for_5"] == 0
    assert features.loc[1, "home_recent_points_5"] == 3
    assert features.loc[1, "home_recent_goals_for_5"] == 2
    assert features.loc[1, "home_recent_goals_against_5"] == 0
    assert features.loc[1, "away_recent_points_5"] == 0
    assert "home_recent_shots_5" not in features.columns


def test_elo_and_rest_days_only_use_past_matches() -> None:
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
                "odds_home": 2.0,
                "odds_draw": 3.0,
                "odds_away": 4.0,
            },
            {
                "date": "2024-01-08",
                "league": "Premier League",
                "home_team": "A",
                "away_team": "C",
                "home_goals": 0,
                "away_goals": 1,
                "result": "A",
                "odds_home": 2.0,
                "odds_draw": 3.0,
                "odds_away": 4.0,
            },
            {
                "date": "2024-01-10",
                "league": "Premier League",
                "home_team": "C",
                "away_team": "B",
                "home_goals": 1,
                "away_goals": 1,
                "result": "D",
                "odds_home": 2.0,
                "odds_draw": 3.0,
                "odds_away": 4.0,
            },
        ]
    )

    features = add_rolling_features(matches)

    assert features.loc[0, "home_elo_before"] == 1500
    assert features.loc[0, "away_elo_before"] == 1500
    assert pd.isna(features.loc[0, "home_rest_days"])
    assert pd.isna(features.loc[0, "away_rest_days"])
    assert features.loc[1, "home_elo_before"] == pytest.approx(1510)
    assert features.loc[1, "away_elo_before"] == 1500
    assert features.loc[1, "elo_diff_before"] == pytest.approx(10)
    assert features.loc[1, "home_rest_days"] == 7
    assert features.loc[1, "home_rest_days_capped"] == 7
    assert pd.isna(features.loc[1, "away_rest_days"])
    assert pd.isna(features.loc[1, "away_rest_days_capped"])
    assert features.loc[2, "home_rest_days"] == 2
    assert features.loc[2, "away_rest_days"] == 9
    assert features.loc[2, "rest_days_diff"] == -7
    assert features.loc[2, "rest_days_diff_capped"] == -7


def test_rest_day_caps_are_model_safe() -> None:
    matches = pd.DataFrame(
        [
            {
                "date": "2024-01-01",
                "league": "Premier League",
                "home_team": "A",
                "away_team": "B",
                "home_goals": 1,
                "away_goals": 0,
                "result": "H",
            },
            {
                "date": "2024-03-15",
                "league": "Premier League",
                "home_team": "A",
                "away_team": "B",
                "home_goals": 0,
                "away_goals": 1,
                "result": "A",
            },
        ]
    )

    features = add_rolling_features(matches)

    assert features.loc[1, "home_rest_days"] == 74
    assert features.loc[1, "away_rest_days"] == 74
    assert features.loc[1, "home_rest_days_capped"] == 30
    assert features.loc[1, "away_rest_days_capped"] == 30
    assert features.loc[1, "rest_days_diff_capped"] == 0
    assert "home_rest_days_capped" in features.columns
    assert "away_rest_days_capped" in features.columns
    assert "rest_days_diff_capped" in features.columns
    assert features["home_rest_days_capped"].dropna().between(0, 30).all()
    assert features["away_rest_days_capped"].dropna().between(0, 30).all()
    assert features["rest_days_diff_capped"].dropna().between(-14, 14).all()


def test_rolling_stat_features_only_use_previous_matches() -> None:
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
                "home_shots_on_target": 4,
                "away_shots_on_target": 1,
                "home_corners": 7,
                "away_corners": 2,
                "home_fouls": 9,
                "away_fouls": 12,
                "home_yellow_cards": 1,
                "away_yellow_cards": 3,
                "home_red_cards": 0,
                "away_red_cards": 1,
            },
            {
                "date": "2024-01-08",
                "league": "Premier League",
                "home_team": "A",
                "away_team": "B",
                "home_goals": 1,
                "away_goals": 1,
                "result": "D",
                "home_shots": 20,
                "away_shots": 30,
                "home_shots_on_target": 8,
                "away_shots_on_target": 10,
                "home_corners": 9,
                "away_corners": 8,
                "home_fouls": 5,
                "away_fouls": 6,
                "home_yellow_cards": 0,
                "away_yellow_cards": 1,
                "home_red_cards": 0,
                "away_red_cards": 0,
            },
        ]
    )

    features = add_rolling_features(matches)

    assert features.loc[0, "home_recent_shots_5"] == 0
    assert features.loc[1, "home_recent_shots_5"] == 10
    assert features.loc[1, "away_recent_shots_5"] == 3
    assert features.loc[1, "home_recent_shots_on_target_5"] == 4
    assert features.loc[1, "away_recent_corners_5"] == 2
    assert features.loc[1, "home_recent_fouls_5"] == 9
    assert features.loc[1, "away_recent_yellow_cards_5"] == 3
    assert features.loc[1, "away_recent_red_cards_5"] == 1
