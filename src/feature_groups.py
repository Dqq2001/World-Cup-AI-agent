import pandas as pd


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

PRODUCTION_MODEL_GROUPS = {
    "market": ["odds"],
    "no_market": ["basic_form", "elo", "rest_days", "rolling_stats"],
    "poisson": ["basic_form", "elo", "rest_days", "rolling_stats", "odds"],
}


def get_group_feature_columns(data: pd.DataFrame, group_names: list[str]) -> list[str]:
    columns = []
    for group_name in group_names:
        for column in FEATURE_GROUPS[group_name]:
            if column in data.columns and column not in columns:
                columns.append(column)
    return columns


def numeric_features(data: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    features = data.copy()
    for column in feature_columns:
        if column not in features.columns:
            features[column] = pd.NA
    return features[feature_columns].apply(pd.to_numeric, errors="coerce")
