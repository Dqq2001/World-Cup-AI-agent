import pandas as pd

from src.predict import _odds_available, _select_by_match_details


def test_odds_are_unavailable_when_market_features_are_missing() -> None:
    feature_row = pd.DataFrame(
        [
            {
                "home_recent_points_5": 8,
                "away_recent_points_5": 5,
            }
        ]
    )

    assert not _odds_available(feature_row)


def test_odds_are_available_when_market_features_exist() -> None:
    feature_row = pd.DataFrame(
        [
            {
                "odds_implied_home_prob": 0.45,
                "odds_implied_draw_prob": 0.25,
                "odds_implied_away_prob": 0.30,
            }
        ]
    )

    assert _odds_available(feature_row)


def test_select_match_by_home_away_and_date() -> None:
    features = pd.DataFrame(
        [
            {
                "date": "2024-01-01",
                "league": "Premier League",
                "home_team": "A",
                "away_team": "B",
            }
        ]
    )

    selected = _select_by_match_details(features, "a", "b", "2024-01-01")

    assert selected is not None
    assert selected.iloc[0]["home_team"] == "A"
