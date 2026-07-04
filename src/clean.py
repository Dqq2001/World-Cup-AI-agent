from pathlib import Path

import pandas as pd

from src.paths import MATCHES_CLEAN_PATH, PROCESSED_DATA_DIR, RAW_DATA_DIR


COLUMN_MAP = {
    "Date": "date",
    "Div": "league",
    "HomeTeam": "home_team",
    "AwayTeam": "away_team",
    "FTHG": "home_goals",
    "FTAG": "away_goals",
    "FTR": "result",
    "HS": "home_shots",
    "AS": "away_shots",
    "HST": "home_shots_on_target",
    "AST": "away_shots_on_target",
    "HC": "home_corners",
    "AC": "away_corners",
    "HF": "home_fouls",
    "AF": "away_fouls",
    "HY": "home_yellow_cards",
    "AY": "away_yellow_cards",
    "HR": "home_red_cards",
    "AR": "away_red_cards",
    "B365H": "odds_home",
    "B365D": "odds_draw",
    "B365A": "odds_away",
}

CLEAN_COLUMNS = [
    "date",
    "league",
    "season",
    "home_team",
    "away_team",
    "home_goals",
    "away_goals",
    "result",
    "home_shots",
    "away_shots",
    "home_shots_on_target",
    "away_shots_on_target",
    "home_corners",
    "away_corners",
    "home_fouls",
    "away_fouls",
    "home_yellow_cards",
    "away_yellow_cards",
    "home_red_cards",
    "away_red_cards",
    "odds_home",
    "odds_draw",
    "odds_away",
    "source_file",
]

LEAGUE_NAMES = {
    "E0": "Premier League",
    "SP1": "La Liga",
    "I1": "Serie A",
    "D1": "Bundesliga",
    "F1": "Ligue 1",
}


def clean_file(csv_path: Path, raw_dir: Path = RAW_DATA_DIR) -> pd.DataFrame:
    raw = pd.read_csv(csv_path, encoding="utf-8")
    renamed = raw.rename(columns=COLUMN_MAP)
    cleaned = pd.DataFrame(
        {column: renamed[column] if column in renamed.columns else pd.NA for column in CLEAN_COLUMNS}
    )

    cleaned["date"] = pd.to_datetime(cleaned["date"], dayfirst=True, errors="coerce")
    cleaned["league"] = cleaned["league"].map(LEAGUE_NAMES).fillna(cleaned["league"])
    cleaned["season"] = csv_path.stem
    cleaned["source_file"] = str(csv_path.relative_to(raw_dir))

    numeric_columns = [
        "home_goals",
        "away_goals",
        "home_shots",
        "away_shots",
        "home_shots_on_target",
        "away_shots_on_target",
        "home_corners",
        "away_corners",
        "home_fouls",
        "away_fouls",
        "home_yellow_cards",
        "away_yellow_cards",
        "home_red_cards",
        "away_red_cards",
        "odds_home",
        "odds_draw",
        "odds_away",
    ]
    for column in numeric_columns:
        cleaned[column] = pd.to_numeric(cleaned[column], errors="coerce")

    cleaned = cleaned[CLEAN_COLUMNS]
    cleaned = cleaned.dropna(subset=["date", "home_team", "away_team", "result"])
    cleaned = cleaned[cleaned["result"].isin(["H", "D", "A"])]
    return cleaned


def clean_matches(raw_dir: Path = RAW_DATA_DIR, output_path: Path = MATCHES_CLEAN_PATH) -> pd.DataFrame:
    csv_files = sorted(raw_dir.glob("**/*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {raw_dir}")

    matches = pd.concat((clean_file(path, raw_dir) for path in csv_files), ignore_index=True)
    matches = matches.sort_values(["date", "league", "home_team", "away_team"]).reset_index(drop=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    matches.to_csv(output_path, index=False, encoding="utf-8")
    return matches


def main() -> None:
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    matches = clean_matches()
    print(f"Saved {len(matches)} cleaned matches to {MATCHES_CLEAN_PATH}")


if __name__ == "__main__":
    main()
