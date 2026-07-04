import argparse
from io import StringIO
import json
import os
import re
import sys
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pandas as pd


PROCESSED_DIR = Path("data/processed")
RAW_DIR = Path("data/raw")
REPORT_PATH = Path("reports/worldcup_data_fetch_missing_report.csv")
SCHEDULE_OUTPUT = PROCESSED_DIR / "worldcup_schedule.csv"
ODDS_OUTPUT = PROCESSED_DIR / "worldcup_odds.csv"
ENV_PATH = Path(".env")
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import NO_PAID_API_MODE

GROUP_PAGES = {
    "A": "https://en.wikipedia.org/wiki/2026_FIFA_World_Cup_Group_A",
    "B": "https://en.wikipedia.org/wiki/2026_FIFA_World_Cup_Group_B",
    "C": "https://en.wikipedia.org/wiki/2026_FIFA_World_Cup_Group_C",
    "D": "https://en.wikipedia.org/wiki/2026_FIFA_World_Cup_Group_D",
    "E": "https://en.wikipedia.org/wiki/2026_FIFA_World_Cup_Group_E",
    "F": "https://en.wikipedia.org/wiki/2026_FIFA_World_Cup_Group_F",
    "G": "https://en.wikipedia.org/wiki/2026_FIFA_World_Cup_Group_G",
    "H": "https://en.wikipedia.org/wiki/2026_FIFA_World_Cup_Group_H",
    "I": "https://en.wikipedia.org/wiki/2026_FIFA_World_Cup_Group_I",
    "J": "https://en.wikipedia.org/wiki/2026_FIFA_World_Cup_Group_J",
    "K": "https://en.wikipedia.org/wiki/2026_FIFA_World_Cup_Group_K",
    "L": "https://en.wikipedia.org/wiki/2026_FIFA_World_Cup_Group_L",
}


def load_dotenv(path: Path = ENV_PATH) -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def report(rows: list[dict]) -> None:
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(REPORT_PATH, index=False, encoding="utf-8")


def find_local_file(candidates: list[str]) -> Path | None:
    for directory in [PROCESSED_DIR, RAW_DIR]:
        for candidate in candidates:
            path = directory / candidate
            if path.exists():
                return path
    return None


def copy_local_schedule(path: Path) -> pd.DataFrame:
    data = pd.read_csv(path, encoding="utf-8")
    required = ["date", "group", "home_team", "away_team"]
    missing = [column for column in required if column not in data.columns]
    if missing:
        raise ValueError(f"local schedule is missing columns: {missing}")
    output = data[required].copy()
    output["date"] = pd.to_datetime(output["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    output.to_csv(SCHEDULE_OUTPUT, index=False, encoding="utf-8")
    return output


def copy_local_odds(path: Path) -> pd.DataFrame:
    data = pd.read_csv(path, encoding="utf-8")
    required = ["date", "group", "home_team", "away_team", "home_odds", "draw_odds", "away_odds"]
    missing = [column for column in required if column not in data.columns]
    if missing:
        raise ValueError(f"local odds is missing columns: {missing}")
    output = data[required].copy()
    output["date"] = pd.to_datetime(output["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    output.to_csv(ODDS_OUTPUT, index=False, encoding="utf-8")
    return output


def clean_team_name(value: str) -> str:
    value = re.sub(r"\[[^\]]+\]", "", str(value))
    value = value.replace("(H)", "").replace("(A)", "")
    return " ".join(value.split()).strip()


def parse_wikipedia_group_page(group: str, url: str) -> list[dict]:
    matches = []
    request = Request(url, headers={"User-Agent": "worldcup-ai-agent/1.0"})
    with urlopen(request, timeout=30) as response:
        html = response.read().decode("utf-8", errors="ignore")
    tables = pd.read_html(StringIO(html))
    for table in tables:
        flattened_columns = [str(column).lower() for column in table.columns]
        text_columns = " ".join(flattened_columns)
        if "team 1" not in text_columns or "team 2" not in text_columns:
            continue

        table.columns = [str(column).lower().replace(" ", "_") for column in table.columns]
        date_col = next((column for column in table.columns if "date" in column), None)
        team1_col = next((column for column in table.columns if "team_1" in column), None)
        team2_col = next((column for column in table.columns if "team_2" in column), None)
        if not date_col or not team1_col or not team2_col:
            continue

        for _, row in table.iterrows():
            date = pd.to_datetime(row.get(date_col), errors="coerce")
            home_team = clean_team_name(row.get(team1_col))
            away_team = clean_team_name(row.get(team2_col))
            if pd.isna(date) or not home_team or not away_team or home_team == "nan" or away_team == "nan":
                continue
            matches.append(
                {
                    "date": date.strftime("%Y-%m-%d"),
                    "group": group,
                    "home_team": home_team,
                    "away_team": away_team,
                }
            )
    return matches


def fetch_schedule_from_wikipedia() -> pd.DataFrame:
    rows = []
    errors = []
    for group, url in GROUP_PAGES.items():
        try:
            rows.extend(parse_wikipedia_group_page(group, url))
        except (HTTPError, URLError, ValueError) as exc:
            errors.append(f"Group {group}: {exc}")

    if not rows:
        raise RuntimeError("Unable to fetch World Cup schedule from public sources. " + "; ".join(errors))

    data = pd.DataFrame(rows).drop_duplicates(["date", "group", "home_team", "away_team"])
    data = data.sort_values(["date", "group", "home_team", "away_team"]).reset_index(drop=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    data.to_csv(SCHEDULE_OUTPUT, index=False, encoding="utf-8")
    return data


def fetch_odds_from_the_odds_api(api_key: str) -> pd.DataFrame:
    params = urlencode(
        {
            "apiKey": api_key,
            "regions": "us,uk,eu",
            "markets": "h2h",
            "oddsFormat": "decimal",
            "dateFormat": "iso",
        }
    )
    url = f"https://api.the-odds-api.com/v4/sports/soccer_fifa_world_cup/odds?{params}"
    request = Request(url, headers={"User-Agent": "worldcup-ai-agent/1.0"})
    with urlopen(request, timeout=30) as response:
        payload = json.loads(response.read().decode("utf-8"))

    rows = []
    for event in payload:
        home_team = event.get("home_team")
        away_team = event.get("away_team")
        date = pd.to_datetime(event.get("commence_time"), errors="coerce")
        if pd.isna(date) or not home_team or not away_team:
            continue

        prices = {}
        for bookmaker in event.get("bookmakers", []):
            for market in bookmaker.get("markets", []):
                if market.get("key") != "h2h":
                    continue
                for outcome in market.get("outcomes", []):
                    prices[outcome.get("name")] = outcome.get("price")
                break
            if prices:
                break

        home_odds = prices.get(home_team)
        away_odds = prices.get(away_team)
        draw_odds = prices.get("Draw")
        if not home_odds or not draw_odds or not away_odds:
            continue

        rows.append(
            {
                "date": date.strftime("%Y-%m-%d"),
                "group": pd.NA,
                "home_team": home_team,
                "away_team": away_team,
                "home_odds": home_odds,
                "draw_odds": draw_odds,
                "away_odds": away_odds,
            }
        )

    if not rows:
        raise RuntimeError("Odds API returned no usable World Cup h2h odds.")

    data = pd.DataFrame(rows)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    data.to_csv(ODDS_OUTPUT, index=False, encoding="utf-8")
    return data


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-build", action="store_true")
    args = parser.parse_args()

    load_dotenv()
    missing_rows = []

    schedule_path = find_local_file(
        [
            "worldcup_schedule.csv",
            "worldcup_fixtures.csv",
            "world_cup_schedule.csv",
            "world_cup_fixtures.csv",
        ]
    )
    if schedule_path:
        schedule = copy_local_schedule(schedule_path)
        print(f"已使用本機 World Cup schedule: {schedule_path} ({len(schedule)} rows)")
    else:
        try:
            schedule = fetch_schedule_from_wikipedia()
            print(f"已從公開穩定來源取得 World Cup schedule: {SCHEDULE_OUTPUT} ({len(schedule)} rows)")
        except Exception as exc:
            missing_rows.append(
                {
                    "data_type": "schedule",
                    "missing": "worldcup_schedule.csv",
                    "message": f"無法取得 World Cup schedule: {exc}",
                }
            )

    odds_path = find_local_file(["worldcup_odds.csv", "world_cup_odds.csv"])
    if odds_path:
        odds = copy_local_odds(odds_path)
        print(f"已使用本機 World Cup odds: {odds_path} ({len(odds)} rows)")
    elif NO_PAID_API_MODE:
        missing_rows.append(
            {
                "data_type": "odds",
                "missing": "manual odds",
                "message": "NO_PAID_API_MODE=True: paid odds APIs are disabled. Use data/manual/worldcup_odds_manual.csv.",
            }
        )
        print("NO_PAID_API_MODE=True: skipping paid odds API fetch.")
    else:
        api_key = os.environ.get("ODDS_API_KEY")
        if not api_key:
            missing_rows.append(
                {
                    "data_type": "odds",
                    "missing": "ODDS_API_KEY",
                    "message": "缺少 odds API key，無法取得賠率。",
                }
            )
        else:
            try:
                odds = fetch_odds_from_the_odds_api(api_key)
                print(f"已從 Odds API 取得 World Cup odds: {ODDS_OUTPUT} ({len(odds)} rows)")
            except Exception as exc:
                missing_rows.append(
                    {
                        "data_type": "odds",
                        "missing": "worldcup_odds.csv",
                        "message": f"無法取得 World Cup odds: {exc}",
                    }
                )

    if missing_rows:
        report(missing_rows)
        print(f"有資料無法取得，已輸出缺資料報告: {REPORT_PATH}")
        return

    if not args.skip_build:
        import subprocess
        import sys

        result = subprocess.run([sys.executable, "scripts/build_worldcup_features.py"], check=False)
        if result.returncode != 0:
            raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
