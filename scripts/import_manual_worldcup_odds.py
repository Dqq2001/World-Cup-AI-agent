import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANUAL_INPUT = PROJECT_ROOT / "data" / "manual" / "worldcup_odds_manual.csv"
FIXTURES_PATH = PROJECT_ROOT / "data" / "processed" / "worldcup_fixtures.csv"
BOOKMAKER_OUTPUT = PROJECT_ROOT / "data" / "processed" / "worldcup_odds.csv"
CONSENSUS_OUTPUT = PROJECT_ROOT / "data" / "processed" / "worldcup_consensus_odds.csv"
MARKET_OUTPUT = PROJECT_ROOT / "data" / "processed" / "worldcup_market_predictions.csv"
MISSING_REPORT = PROJECT_ROOT / "reports" / "manual_worldcup_odds_missing_report.csv"

MANUAL_COLUMNS = ["date", "home_team", "away_team", "home_odds", "draw_odds", "away_odds", "bookmaker"]
KEY_COLUMNS = ["date", "group", "home_team", "away_team"]
BOOKMAKER_COLUMNS = KEY_COLUMNS + ["bookmaker", "home_odds", "draw_odds", "away_odds"]
CONSENSUS_COLUMNS = KEY_COLUMNS + ["home_odds", "draw_odds", "away_odds"]
MARKET_COLUMNS = KEY_COLUMNS + ["market_H", "market_D", "market_A"]


def team_key(value: str) -> str:
    return "".join(character for character in str(value).lower() if character.isalnum())


def normalize_dates(data: pd.DataFrame) -> pd.DataFrame:
    data = data.copy()
    data["date"] = pd.to_datetime(data["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    for column in ["home_team", "away_team"]:
        data[column] = data[column].astype(str).str.strip()
    return data


def write_missing_report(rows: list[dict]) -> None:
    MISSING_REPORT.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(MISSING_REPORT, index=False, encoding="utf-8")


def load_fixtures(path: Path = FIXTURES_PATH) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"找不到 World Cup fixtures: {path}")
    fixtures = pd.read_csv(path, encoding="utf-8")
    missing = [column for column in KEY_COLUMNS if column not in fixtures.columns]
    if missing:
        raise ValueError(f"fixtures 缺少欄位: {missing}")
    fixtures = normalize_dates(fixtures[KEY_COLUMNS])
    fixtures["_home_key"] = fixtures["home_team"].map(team_key)
    fixtures["_away_key"] = fixtures["away_team"].map(team_key)
    return fixtures


def load_manual_odds(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"找不到 manual odds: {path}")
    manual = pd.read_csv(path, encoding="utf-8")
    missing = [column for column in MANUAL_COLUMNS if column not in manual.columns]
    if missing:
        raise ValueError(f"manual odds 缺少欄位: {missing}")
    manual = normalize_dates(manual[MANUAL_COLUMNS])
    manual["bookmaker"] = manual["bookmaker"].astype(str).str.strip()
    for column in ["home_odds", "draw_odds", "away_odds"]:
        manual[column] = pd.to_numeric(manual[column], errors="coerce")
    manual = manual.dropna(subset=MANUAL_COLUMNS)
    manual = manual[(manual["home_odds"] > 1) & (manual["draw_odds"] > 1) & (manual["away_odds"] > 1)]
    manual["_home_key"] = manual["home_team"].map(team_key)
    manual["_away_key"] = manual["away_team"].map(team_key)
    return manual


def align_manual_to_fixtures(manual: pd.DataFrame, fixtures: pd.DataFrame) -> pd.DataFrame:
    aligned = fixtures.merge(
        manual[["date", "_home_key", "_away_key", "bookmaker", "home_odds", "draw_odds", "away_odds"]],
        on=["date", "_home_key", "_away_key"],
        how="left",
    )
    missing_mask = aligned[["home_odds", "draw_odds", "away_odds"]].isna().any(axis=1)
    if missing_mask.any():
        missing = aligned.loc[missing_mask, KEY_COLUMNS].drop_duplicates()
        write_missing_report(
            [
                {
                    "missing_rows": len(missing),
                    "message": "manual odds 未覆蓋所有 World Cup fixtures；未補假 odds，改用 no-odds mode。",
                    "examples": missing.head(10).to_json(orient="records", force_ascii=False),
                }
            ]
        )
        raise RuntimeError(f"manual odds 只覆蓋 {len(fixtures) - len(missing)}/{len(fixtures)} 場。")
    return aligned[BOOKMAKER_COLUMNS].drop_duplicates(BOOKMAKER_COLUMNS).sort_values(KEY_COLUMNS + ["bookmaker"])


def build_consensus(bookmaker_odds: pd.DataFrame) -> pd.DataFrame:
    return (
        bookmaker_odds.groupby(KEY_COLUMNS, as_index=False)[["home_odds", "draw_odds", "away_odds"]]
        .mean()
        .sort_values(KEY_COLUMNS)
    )[CONSENSUS_COLUMNS]


def build_market(consensus: pd.DataFrame) -> pd.DataFrame:
    market = consensus[KEY_COLUMNS].copy()
    raw_h = 1 / consensus["home_odds"]
    raw_d = 1 / consensus["draw_odds"]
    raw_a = 1 / consensus["away_odds"]
    overround = raw_h + raw_d + raw_a
    market["market_H"] = raw_h / overround
    market["market_D"] = raw_d / overround
    market["market_A"] = raw_a / overround
    return market[MARKET_COLUMNS]


def run_command(command: list[str]) -> bool:
    print(f"執行: {' '.join(command)}")
    completed = subprocess.run(command, cwd=PROJECT_ROOT, check=False)
    if completed.returncode != 0:
        print(f"命令失敗，exit code={completed.returncode}: {' '.join(command)}")
        return False
    return True


def run_no_odds_mode() -> None:
    run_command([sys.executable, "scripts/run_worldcup_betting_agents.py", "--model-only"])


def import_manual_odds(manual_path: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    fixtures = load_fixtures()
    manual = load_manual_odds(manual_path)
    bookmaker_odds = align_manual_to_fixtures(manual, fixtures)
    consensus = build_consensus(bookmaker_odds)
    market = build_market(consensus)

    BOOKMAKER_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    bookmaker_odds.to_csv(BOOKMAKER_OUTPUT, index=False, encoding="utf-8")
    consensus.to_csv(CONSENSUS_OUTPUT, index=False, encoding="utf-8")
    market.to_csv(MARKET_OUTPUT, index=False, encoding="utf-8")
    write_missing_report([])
    return bookmaker_odds, consensus, market


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manual-csv", type=Path, default=DEFAULT_MANUAL_INPUT)
    parser.add_argument("--skip-downstream", action="store_true")
    args = parser.parse_args()

    try:
        bookmaker_odds, consensus, market = import_manual_odds(args.manual_csv)
    except Exception as exc:
        print(f"無法匯入 manual odds: {exc}")
        print("改用 no-odds mode。")
        run_no_odds_mode()
        return

    print(f"已輸出 bookmaker odds: {BOOKMAKER_OUTPUT} ({len(bookmaker_odds)} rows)")
    print(f"已輸出 consensus odds: {CONSENSUS_OUTPUT} ({len(consensus)} rows)")
    print(f"已輸出 market probabilities: {MARKET_OUTPUT} ({len(market)} rows)")

    if not args.skip_downstream:
        if run_command([sys.executable, "scripts/build_worldcup_features.py"]):
            run_command([sys.executable, "scripts/run_worldcup_betting_agents.py"])


if __name__ == "__main__":
    main()
