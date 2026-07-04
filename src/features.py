import pandas as pd

from src.paths import FEATURES_PATH, MATCHES_CLEAN_PATH, PROCESSED_DATA_DIR


FEATURE_COLUMNS = [
    "home_recent_points_5",
    "away_recent_points_5",
    "home_recent_goals_for_5",
    "away_recent_goals_for_5",
    "home_recent_goals_against_5",
    "away_recent_goals_against_5",
    "home_win_rate_5",
    "away_win_rate_5",
    "odds_implied_home_prob",
    "odds_implied_draw_prob",
    "odds_implied_away_prob",
    "home_elo_before",
    "away_elo_before",
    "elo_diff_before",
    "home_rest_days_capped",
    "away_rest_days_capped",
    "rest_days_diff_capped",
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
]

INITIAL_ELO = 1500.0
ELO_K = 20.0

ROLLING_STAT_SOURCES = {
    "shots": ("home_shots", "away_shots"),
    "shots_on_target": ("home_shots_on_target", "away_shots_on_target"),
    "corners": ("home_corners", "away_corners"),
    "fouls": ("home_fouls", "away_fouls"),
    "yellow_cards": ("home_yellow_cards", "away_yellow_cards"),
    "red_cards": ("home_red_cards", "away_red_cards"),
}


def get_model_feature_columns(data: pd.DataFrame) -> list[str]:
    return [column for column in FEATURE_COLUMNS if column in data.columns]


def _recent_stats(history: list[dict], window: int = 5) -> dict[str, float]:
    recent = history[-window:]
    if not recent:
        return {
            "points": 0.0,
            "goals_for": 0.0,
            "goals_against": 0.0,
            "win_rate": 0.0,
        }

    return {
        "points": float(sum(match["points"] for match in recent)),
        "goals_for": float(sum(match["goals_for"] for match in recent)),
        "goals_against": float(sum(match["goals_against"] for match in recent)),
        "win_rate": float(sum(match["won"] for match in recent) / len(recent)),
    }


def _recent_sum(history: list[dict], key: str, window: int = 5) -> float:
    values = [match.get(key, pd.NA) for match in history[-window:]]
    numeric_values = pd.to_numeric(pd.Series(values), errors="coerce").dropna()
    if numeric_values.empty:
        return 0.0
    return float(numeric_values.sum())


def _points(result: str, is_home: bool) -> int:
    if result == "D":
        return 1
    if (result == "H" and is_home) or (result == "A" and not is_home):
        return 3
    return 0


def _append_team_match(
    history: dict[str, list[dict]],
    team: str,
    goals_for: float,
    goals_against: float,
    points: int,
    extra_stats: dict[str, float],
) -> None:
    match = {
        "goals_for": goals_for,
        "goals_against": goals_against,
        "points": points,
        "won": 1 if points == 3 else 0,
    }
    match.update(extra_stats)
    history.setdefault(team, []).append(match)


def _elo_expected_rating(team_elo: float, opponent_elo: float) -> float:
    return 1 / (1 + 10 ** ((opponent_elo - team_elo) / 400))


def _elo_scores(result: str) -> tuple[float, float]:
    if result == "H":
        return 1.0, 0.0
    if result == "A":
        return 0.0, 1.0
    return 0.5, 0.5


def _updated_elo(home_elo: float, away_elo: float, result: str, k: float = ELO_K) -> tuple[float, float]:
    home_score, away_score = _elo_scores(result)
    home_expected = _elo_expected_rating(home_elo, away_elo)
    away_expected = _elo_expected_rating(away_elo, home_elo)
    return (
        home_elo + k * (home_score - home_expected),
        away_elo + k * (away_score - away_expected),
    )


def _rest_days(current_date: pd.Timestamp, previous_date: pd.Timestamp | None) -> float:
    if previous_date is None or pd.isna(previous_date):
        return pd.NA
    return float((current_date - previous_date).days)


def _clip_or_na(value: float, lower: float, upper: float) -> float:
    if pd.isna(value):
        return pd.NA
    return float(min(max(value, lower), upper))


def _available_stat_sources(matches: pd.DataFrame) -> dict[str, tuple[str, str]]:
    available = {}
    for stat_name, (home_column, away_column) in ROLLING_STAT_SOURCES.items():
        if home_column not in matches.columns or away_column not in matches.columns:
            continue
        values = pd.concat([matches[home_column], matches[away_column]], ignore_index=True)
        if pd.to_numeric(values, errors="coerce").notna().any():
            available[stat_name] = (home_column, away_column)
    return available


def _implied_probabilities(odds_home: float, odds_draw: float, odds_away: float) -> dict[str, float]:
    odds = pd.to_numeric(pd.Series([odds_home, odds_draw, odds_away]), errors="coerce")
    if odds.isna().any() or (odds <= 0).any():
        return {
            "odds_implied_home_prob": pd.NA,
            "odds_implied_draw_prob": pd.NA,
            "odds_implied_away_prob": pd.NA,
        }

    inverse = 1 / odds
    probabilities = inverse / inverse.sum()
    return {
        "odds_implied_home_prob": float(probabilities.iloc[0]),
        "odds_implied_draw_prob": float(probabilities.iloc[1]),
        "odds_implied_away_prob": float(probabilities.iloc[2]),
    }


def add_rolling_features(matches: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    matches = matches.copy()
    matches["date"] = pd.to_datetime(matches["date"], errors="coerce")
    matches = matches.sort_values(["date", "league", "home_team", "away_team"]).reset_index(drop=True)
    stat_sources = _available_stat_sources(matches)

    history: dict[str, list[dict]] = {}
    elo_ratings: dict[str, float] = {}
    previous_match_dates: dict[str, pd.Timestamp] = {}
    feature_rows = []

    for row in matches.itertuples(index=False):
        home_stats = _recent_stats(history.get(row.home_team, []), window)
        away_stats = _recent_stats(history.get(row.away_team, []), window)
        home_elo = elo_ratings.get(row.home_team, INITIAL_ELO)
        away_elo = elo_ratings.get(row.away_team, INITIAL_ELO)
        home_rest_days = _rest_days(row.date, previous_match_dates.get(row.home_team))
        away_rest_days = _rest_days(row.date, previous_match_dates.get(row.away_team))
        rest_days_diff = (
            pd.NA
            if pd.isna(home_rest_days) or pd.isna(away_rest_days)
            else home_rest_days - away_rest_days
        )

        rolling_stat_features = {}
        for stat_name in stat_sources:
            rolling_stat_features[f"home_recent_{stat_name}_5"] = _recent_sum(
                history.get(row.home_team, []), stat_name, window
            )
            rolling_stat_features[f"away_recent_{stat_name}_5"] = _recent_sum(
                history.get(row.away_team, []), stat_name, window
            )

        odds_home = getattr(row, "odds_home", pd.NA)
        odds_draw = getattr(row, "odds_draw", pd.NA)
        odds_away = getattr(row, "odds_away", pd.NA)
        implied = _implied_probabilities(odds_home, odds_draw, odds_away)

        feature_rows.append(
            {
                "home_recent_points_5": home_stats["points"],
                "away_recent_points_5": away_stats["points"],
                "home_recent_goals_for_5": home_stats["goals_for"],
                "away_recent_goals_for_5": away_stats["goals_for"],
                "home_recent_goals_against_5": home_stats["goals_against"],
                "away_recent_goals_against_5": away_stats["goals_against"],
                "home_win_rate_5": home_stats["win_rate"],
                "away_win_rate_5": away_stats["win_rate"],
                **implied,
                "home_elo_before": home_elo,
                "away_elo_before": away_elo,
                "elo_diff_before": home_elo - away_elo,
                "home_rest_days": home_rest_days,
                "away_rest_days": away_rest_days,
                "rest_days_diff": rest_days_diff,
                "home_rest_days_capped": _clip_or_na(home_rest_days, 0, 30),
                "away_rest_days_capped": _clip_or_na(away_rest_days, 0, 30),
                "rest_days_diff_capped": _clip_or_na(rest_days_diff, -14, 14),
                **rolling_stat_features,
            }
        )

        home_points = _points(row.result, is_home=True)
        away_points = _points(row.result, is_home=False)
        home_extra_stats = {}
        away_extra_stats = {}
        for stat_name, (home_column, away_column) in stat_sources.items():
            home_extra_stats[stat_name] = getattr(row, home_column, pd.NA)
            away_extra_stats[stat_name] = getattr(row, away_column, pd.NA)

        _append_team_match(
            history,
            row.home_team,
            row.home_goals,
            row.away_goals,
            home_points,
            home_extra_stats,
        )
        _append_team_match(
            history,
            row.away_team,
            row.away_goals,
            row.home_goals,
            away_points,
            away_extra_stats,
        )
        elo_ratings[row.home_team], elo_ratings[row.away_team] = _updated_elo(home_elo, away_elo, row.result)
        previous_match_dates[row.home_team] = row.date
        previous_match_dates[row.away_team] = row.date

    features = pd.concat([matches, pd.DataFrame(feature_rows)], axis=1)
    return features


def build_features(input_path: str = MATCHES_CLEAN_PATH, output_path: str = FEATURES_PATH) -> pd.DataFrame:
    matches = pd.read_csv(input_path, encoding="utf-8")
    features = add_rolling_features(matches)
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    features.to_csv(output_path, index=False, encoding="utf-8")
    return features


def main() -> None:
    features = build_features()
    print(f"Saved {len(features)} feature rows to {FEATURES_PATH}")


if __name__ == "__main__":
    main()
