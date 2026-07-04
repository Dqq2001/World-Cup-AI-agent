from __future__ import annotations

import math
from collections import defaultdict
from pathlib import Path

import pandas as pd

from src.international_features import INTERNATIONAL_FEATURE_COLUMNS, prepare_feature_matrix
from src.paths import INTERNATIONAL_TRAINING_PATH, PROCESSED_DATA_DIR


INITIAL_ELO = 1500.0
ELO_K = 20.0
RECENT_WINDOW = 5
FIXTURE_CANDIDATES = [
    PROCESSED_DATA_DIR / "worldcup_fixtures_resolved.csv",
    PROCESSED_DATA_DIR / "worldcup_features.csv",
    PROCESSED_DATA_DIR / "worldcup_schedule.csv",
    PROCESSED_DATA_DIR / "worldcup_fixtures.csv",
]


def result_from_goals(home_goals: float, away_goals: float) -> str:
    if home_goals > away_goals:
        return "H"
    if home_goals < away_goals:
        return "A"
    return "D"


def points_for(result: str, is_home: bool) -> int:
    if result == "D":
        return 1
    if (result == "H" and is_home) or (result == "A" and not is_home):
        return 3
    return 0


def elo_expected(team_elo: float, opponent_elo: float) -> float:
    return 1 / (1 + 10 ** ((opponent_elo - team_elo) / 400))


def update_elo(home_elo: float, away_elo: float, result: str) -> tuple[float, float]:
    if result == "H":
        home_score, away_score = 1.0, 0.0
    elif result == "A":
        home_score, away_score = 0.0, 1.0
    else:
        home_score, away_score = 0.5, 0.5
    home_expected = elo_expected(home_elo, away_elo)
    away_expected = elo_expected(away_elo, home_elo)
    return (
        home_elo + ELO_K * (home_score - home_expected),
        away_elo + ELO_K * (away_score - away_expected),
    )


def recent_stats(history: list[dict]) -> dict:
    recent = history[-RECENT_WINDOW:]
    if not recent:
        return {"form": 0.0, "goals_for": 0.0, "goals_against": 0.0}
    return {
        "form": float(sum(match["points"] for match in recent)),
        "goals_for": float(sum(match["goals_for"] for match in recent)),
        "goals_against": float(sum(match["goals_against"] for match in recent)),
    }


def append_history(histories: dict[str, list[dict]], team: str, goals_for: float, goals_against: float, points: int) -> None:
    histories[team].append(
        {
            "goals_for": goals_for,
            "goals_against": goals_against,
            "points": points,
        }
    )


def load_fixtures(path: Path | None = None) -> pd.DataFrame:
    fixture_path = path
    if fixture_path is None:
        fixture_path = next((candidate for candidate in FIXTURE_CANDIDATES if candidate.exists()), None)
    if fixture_path is None:
        searched = ", ".join(str(path) for path in FIXTURE_CANDIDATES)
        raise FileNotFoundError(f"找不到 World Cup fixtures。請先提供其中之一: {searched}")

    fixtures = pd.read_csv(fixture_path, encoding="utf-8")
    required = ["date", "home_team", "away_team"]
    missing = [column for column in required if column not in fixtures.columns]
    if missing:
        raise ValueError(f"World Cup fixtures 缺少必要欄位: {missing}")

    fixtures = fixtures.copy()
    if "group" not in fixtures.columns:
        fixtures["group"] = ""
    for column in ["stage", "round", "match_id", "home_slot", "away_slot", "status"]:
        if column not in fixtures.columns:
            fixtures[column] = ""
    fixtures["date"] = pd.to_datetime(fixtures["date"], errors="coerce")
    fixtures = fixtures.dropna(subset=["date", "home_team", "away_team"])
    fixtures = fixtures[
        ~fixtures["home_team"].astype(str).str.upper().eq("TBD")
        & ~fixtures["away_team"].astype(str).str.upper().eq("TBD")
    ]
    return fixtures.sort_values(["date", "stage", "round", "group", "home_team", "away_team"]).reset_index(drop=True)


def load_history(path: Path = INTERNATIONAL_TRAINING_PATH) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"找不到國際賽訓練資料: {path}")
    history = pd.read_csv(path, encoding="utf-8")
    required = ["date", "home_team", "away_team", "home_goals", "away_goals"]
    missing = [column for column in required if column not in history.columns]
    if missing:
        raise ValueError(f"國際賽訓練資料缺少必要欄位: {missing}")
    history = history.copy()
    history["date"] = pd.to_datetime(history["date"], errors="coerce")
    return history.dropna(subset=["date", "home_goals", "away_goals"]).sort_values("date").reset_index(drop=True)


def build_fixture_features(fixtures: pd.DataFrame, history: pd.DataFrame) -> pd.DataFrame:
    elo_ratings: dict[str, float] = {}
    histories: dict[str, list[dict]] = defaultdict(list)
    rows = []
    history_index = 0
    fixtures = fixtures.sort_values("date").reset_index(drop=True)
    history = history.sort_values("date").reset_index(drop=True)

    for fixture in fixtures.itertuples(index=False):
        fixture_date = fixture.date
        while history_index < len(history) and history.loc[history_index, "date"] < fixture_date:
            match = history.loc[history_index]
            home_team = match["home_team"]
            away_team = match["away_team"]
            home_elo = elo_ratings.get(home_team, INITIAL_ELO)
            away_elo = elo_ratings.get(away_team, INITIAL_ELO)
            result = result_from_goals(match["home_goals"], match["away_goals"])
            append_history(histories, home_team, match["home_goals"], match["away_goals"], points_for(result, True))
            append_history(histories, away_team, match["away_goals"], match["home_goals"], points_for(result, False))
            elo_ratings[home_team], elo_ratings[away_team] = update_elo(home_elo, away_elo, result)
            history_index += 1

        home_team = fixture.home_team
        away_team = fixture.away_team
        home_elo = elo_ratings.get(home_team, INITIAL_ELO)
        away_elo = elo_ratings.get(away_team, INITIAL_ELO)
        home_recent = recent_stats(histories[home_team])
        away_recent = recent_stats(histories[away_team])
        neutral = bool(getattr(fixture, "neutral", getattr(fixture, "neutral_venue", True)))

        rows.append(
            {
                "date": fixture_date.strftime("%Y-%m-%d"),
                "group": getattr(fixture, "group", ""),
                "stage": getattr(fixture, "stage", ""),
                "round": getattr(fixture, "round", ""),
                "match_id": getattr(fixture, "match_id", ""),
                "home_slot": getattr(fixture, "home_slot", ""),
                "away_slot": getattr(fixture, "away_slot", ""),
                "status": getattr(fixture, "status", ""),
                "home_team": home_team,
                "away_team": away_team,
                "neutral": neutral,
                "home_elo": home_elo,
                "away_elo": away_elo,
                "elo_diff": home_elo - away_elo,
                "home_recent_form": home_recent["form"],
                "away_recent_form": away_recent["form"],
                "home_recent_goals_for": home_recent["goals_for"],
                "home_recent_goals_against": home_recent["goals_against"],
                "away_recent_goals_for": away_recent["goals_for"],
                "away_recent_goals_against": away_recent["goals_against"],
                "is_worldcup": True,
                "is_qualifier": False,
                "is_friendly": False,
            }
        )

    return pd.DataFrame(rows)


def fixture_model_matrix(feature_data: pd.DataFrame) -> pd.DataFrame:
    return prepare_feature_matrix(feature_data[INTERNATIONAL_FEATURE_COLUMNS])


def poisson_pmf(goal_count: int, expected_goals: float) -> float:
    expected_goals = max(float(expected_goals), 1e-9)
    return math.exp(-expected_goals) * expected_goals**goal_count / math.factorial(goal_count)


def top_scorelines(home_xg: float, away_xg: float, max_goals: int = 5, top_n: int = 5) -> list[dict]:
    scorelines = []
    for home_goals in range(max_goals + 1):
        home_prob = poisson_pmf(home_goals, home_xg)
        for away_goals in range(max_goals + 1):
            probability = home_prob * poisson_pmf(away_goals, away_xg)
            scorelines.append({"scoreline": f"{home_goals}-{away_goals}", "probability": round(probability, 6)})
    return sorted(scorelines, key=lambda item: item["probability"], reverse=True)[:top_n]
