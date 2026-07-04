import argparse
from pathlib import Path

import pandas as pd


RAW_INPUT = Path("data/raw/international_results.csv")
PROCESSED_OUTPUT = Path("data/processed/international_training_data.csv")
INITIAL_ELO = 1500.0
ELO_K = 20.0
RECENT_WINDOW = 5


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


def elo_scores(result: str) -> tuple[float, float]:
    if result == "H":
        return 1.0, 0.0
    if result == "A":
        return 0.0, 1.0
    return 0.5, 0.5


def update_elo(home_elo: float, away_elo: float, result: str) -> tuple[float, float]:
    home_score, away_score = elo_scores(result)
    home_expected = elo_expected(home_elo, away_elo)
    away_expected = elo_expected(away_elo, home_elo)
    return (
        home_elo + ELO_K * (home_score - home_expected),
        away_elo + ELO_K * (away_score - away_expected),
    )


def competition_flags(competition: str) -> dict:
    name = str(competition).lower()
    return {
        "is_worldcup": "fifa world cup" in name and "qualification" not in name and "qualifier" not in name,
        "is_qualifier": "qualification" in name or "qualifier" in name,
        "is_friendly": "friendly" in name,
    }


def sample_weight(flags: dict) -> float:
    if flags["is_worldcup"]:
        return 3.0
    if flags["is_qualifier"]:
        return 2.0
    if flags["is_friendly"]:
        return 0.5
    return 1.0


def recent_stats(history: list[dict]) -> dict:
    recent = history[-RECENT_WINDOW:]
    if not recent:
        return {
            "form": 0.0,
            "goals_for": 0.0,
            "goals_against": 0.0,
        }
    return {
        "form": float(sum(match["points"] for match in recent)),
        "goals_for": float(sum(match["goals_for"] for match in recent)),
        "goals_against": float(sum(match["goals_against"] for match in recent)),
    }


def append_history(
    histories: dict[str, list[dict]],
    team: str,
    goals_for: float,
    goals_against: float,
    points: int,
) -> None:
    histories.setdefault(team, []).append(
        {
            "goals_for": goals_for,
            "goals_against": goals_against,
            "points": points,
        }
    )


def load_results(input_path: Path) -> pd.DataFrame:
    if not input_path.exists():
        raise FileNotFoundError(f"International results not found: {input_path}")
    data = pd.read_csv(input_path, encoding="utf-8")
    required = ["date", "competition", "home_team", "away_team", "home_goals", "away_goals", "neutral"]
    missing = [column for column in required if column not in data.columns]
    if missing:
        raise ValueError(f"International results missing columns: {missing}")

    data = data.copy()
    data["date"] = pd.to_datetime(data["date"], errors="coerce")
    data["home_goals"] = pd.to_numeric(data["home_goals"], errors="coerce")
    data["away_goals"] = pd.to_numeric(data["away_goals"], errors="coerce")
    data = data.dropna(subset=["date", "home_goals", "away_goals", "home_team", "away_team"])
    return data.sort_values(["date", "competition", "home_team", "away_team"]).reset_index(drop=True)


def build_training_data(input_path: Path = RAW_INPUT, output_path: Path = PROCESSED_OUTPUT) -> pd.DataFrame:
    matches = load_results(input_path)
    elo_ratings: dict[str, float] = {}
    histories: dict[str, list[dict]] = {}
    rows = []

    for row in matches.itertuples(index=False):
        home_team = row.home_team
        away_team = row.away_team
        home_elo = elo_ratings.get(home_team, INITIAL_ELO)
        away_elo = elo_ratings.get(away_team, INITIAL_ELO)
        home_recent = recent_stats(histories.get(home_team, []))
        away_recent = recent_stats(histories.get(away_team, []))
        result = result_from_goals(row.home_goals, row.away_goals)
        flags = competition_flags(row.competition)

        rows.append(
            {
                "date": row.date.strftime("%Y-%m-%d"),
                "competition": row.competition,
                "home_team": home_team,
                "away_team": away_team,
                "neutral": bool(row.neutral),
                "home_goals": int(row.home_goals),
                "away_goals": int(row.away_goals),
                "result": result,
                "home_elo": home_elo,
                "away_elo": away_elo,
                "elo_diff": home_elo - away_elo,
                "home_recent_form": home_recent["form"],
                "away_recent_form": away_recent["form"],
                "home_recent_goals_for": home_recent["goals_for"],
                "home_recent_goals_against": home_recent["goals_against"],
                "away_recent_goals_for": away_recent["goals_for"],
                "away_recent_goals_against": away_recent["goals_against"],
                "is_worldcup": flags["is_worldcup"],
                "is_qualifier": flags["is_qualifier"],
                "is_friendly": flags["is_friendly"],
                "sample_weight": sample_weight(flags),
            }
        )

        home_points = points_for(result, is_home=True)
        away_points = points_for(result, is_home=False)
        append_history(histories, home_team, row.home_goals, row.away_goals, home_points)
        append_history(histories, away_team, row.away_goals, row.home_goals, away_points)
        elo_ratings[home_team], elo_ratings[away_team] = update_elo(home_elo, away_elo, result)

    output = pd.DataFrame(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_path, index=False, encoding="utf-8")
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=RAW_INPUT)
    parser.add_argument("--output", type=Path, default=PROCESSED_OUTPUT)
    args = parser.parse_args()

    data = build_training_data(args.input, args.output)
    print(f"已建立國際賽訓練資料: {args.output}")
    print(f"輸出筆數: {len(data)}")


if __name__ == "__main__":
    main()
