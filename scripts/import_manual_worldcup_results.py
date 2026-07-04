import argparse
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANUAL_INPUT = PROJECT_ROOT / "data" / "manual" / "worldcup_results_manual.csv"
FIXTURES_PATH = PROJECT_ROOT / "data" / "processed" / "worldcup_fixtures.csv"
OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "worldcup_results.csv"

REQUIRED_COLUMNS = ["date", "group", "home_team", "away_team", "home_goals", "away_goals", "status"]
KEY_COLUMNS = ["date", "group", "home_team", "away_team"]
VALID_STATUS = {"completed", "scheduled"}


def normalize_keys(data: pd.DataFrame) -> pd.DataFrame:
    data = data.copy()
    data["date"] = pd.to_datetime(data["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    for column in ["group", "home_team", "away_team", "status"]:
        data[column] = data[column].astype(str).str.strip()
    data["status"] = data["status"].str.lower()
    return data


def load_fixtures() -> pd.DataFrame:
    if not FIXTURES_PATH.exists():
        raise FileNotFoundError(f"World Cup fixtures not found: {FIXTURES_PATH}")
    fixtures = pd.read_csv(FIXTURES_PATH, encoding="utf-8")
    missing = [column for column in KEY_COLUMNS if column not in fixtures.columns]
    if missing:
        raise ValueError(f"worldcup_fixtures.csv missing columns: {missing}")
    return normalize_keys(fixtures[KEY_COLUMNS]).drop_duplicates()


def load_manual_results(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Manual results file not found: {path}")
    results = pd.read_csv(path, encoding="utf-8")
    missing = [column for column in REQUIRED_COLUMNS if column not in results.columns]
    if missing:
        raise ValueError(f"manual results missing columns: {missing}")

    results = normalize_keys(results[REQUIRED_COLUMNS])
    invalid_status = sorted(set(results["status"].dropna()) - VALID_STATUS)
    if invalid_status:
        raise ValueError(f"invalid status values: {invalid_status}; expected completed or scheduled")

    for column in ["home_goals", "away_goals"]:
        results[column] = pd.to_numeric(results[column], errors="coerce")

    completed = results["status"] == "completed"
    missing_scores = completed & results[["home_goals", "away_goals"]].isna().any(axis=1)
    if missing_scores.any():
        examples = results.loc[missing_scores, KEY_COLUMNS].head(10).to_dict(orient="records")
        raise ValueError(f"completed rows require home_goals and away_goals; examples={examples}")

    scheduled = results["status"] == "scheduled"
    results.loc[scheduled, ["home_goals", "away_goals"]] = pd.NA
    return results


def import_manual_results(manual_path: Path) -> pd.DataFrame:
    fixtures = load_fixtures()
    manual = load_manual_results(manual_path)

    aligned = manual.merge(fixtures, on=KEY_COLUMNS, how="inner")
    missing_fixture_rows = len(manual) - len(aligned)
    if missing_fixture_rows:
        missing = manual.merge(fixtures, on=KEY_COLUMNS, how="left", indicator=True)
        missing = missing[missing["_merge"] == "left_only"][KEY_COLUMNS]
        examples = missing.head(10).to_dict(orient="records")
        raise ValueError(f"{missing_fixture_rows} manual result rows do not match World Cup fixtures; examples={examples}")

    aligned = aligned[REQUIRED_COLUMNS].sort_values(KEY_COLUMNS)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    aligned.to_csv(OUTPUT_PATH, index=False, encoding="utf-8")
    return aligned


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manual-csv", type=Path, default=DEFAULT_MANUAL_INPUT)
    args = parser.parse_args()

    imported = import_manual_results(args.manual_csv)
    completed_count = int((imported["status"] == "completed").sum())
    scheduled_count = int((imported["status"] == "scheduled").sum())
    print(f"Imported World Cup results: {OUTPUT_PATH}")
    print(f"Rows: {len(imported)} completed={completed_count} scheduled={scheduled_count}")


if __name__ == "__main__":
    main()
