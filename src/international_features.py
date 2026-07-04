import pandas as pd


INTERNATIONAL_FEATURE_COLUMNS = [
    "neutral",
    "home_elo",
    "away_elo",
    "elo_diff",
    "home_recent_form",
    "away_recent_form",
    "home_recent_goals_for",
    "home_recent_goals_against",
    "away_recent_goals_for",
    "away_recent_goals_against",
    "is_worldcup",
    "is_qualifier",
    "is_friendly",
]

RESULT_TO_ID = {"H": 0, "D": 1, "A": 2}
ID_TO_RESULT = {value: key for key, value in RESULT_TO_ID.items()}


def prepare_feature_matrix(data: pd.DataFrame) -> pd.DataFrame:
    features = data.copy()
    for column in INTERNATIONAL_FEATURE_COLUMNS:
        if column not in features.columns:
            features[column] = 0
    return features[INTERNATIONAL_FEATURE_COLUMNS].apply(pd.to_numeric, errors="coerce").fillna(0)


def chronological_split(data: pd.DataFrame, test_size: float = 0.2) -> tuple[pd.DataFrame, pd.DataFrame]:
    data = data.sort_values("date").reset_index(drop=True)
    split_index = int(len(data) * (1 - test_size))
    if split_index <= 0 or split_index >= len(data):
        raise ValueError("Not enough rows for chronological train/test split.")
    return data.iloc[:split_index].copy(), data.iloc[split_index:].copy()
