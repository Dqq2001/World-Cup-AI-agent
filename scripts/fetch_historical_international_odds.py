import argparse
import json
import os
import sys
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import NO_PAID_API_MODE

ENV_PATH = PROJECT_ROOT / ".env"
RAW_OUTPUT = PROJECT_ROOT / "data" / "raw" / "historical_international_odds_2022_2024.csv"
PROCESSED_OUTPUT = PROJECT_ROOT / "data" / "processed" / "international_training_with_odds.csv"
TRAINING_PATH = PROJECT_ROOT / "data" / "processed" / "international_training_data.csv"
STATE_PATH = PROJECT_ROOT / "data" / "raw" / "historical_international_odds_fetch_state.json"
REPORT_PATH = PROJECT_ROOT / "reports" / "historical_international_odds_fetch_report.csv"

SEASONS = [2022, 2023, 2024]
ODDS_COLUMNS = ["date", "competition", "home_team", "away_team", "home_odds", "draw_odds", "away_odds", "bookmaker"]
MARKET_COLUMNS = ["market_H", "market_D", "market_A"]

COMPETITIONS = [
    {"league_id": 1, "competition": "FIFA World Cup", "priority": 1},
    {"league_id": 29, "competition": "World Cup - Qualification Africa", "priority": 2},
    {"league_id": 30, "competition": "World Cup - Qualification Asia", "priority": 2},
    {"league_id": 31, "competition": "World Cup - Qualification CONCACAF", "priority": 2},
    {"league_id": 32, "competition": "World Cup - Qualification Europe", "priority": 2},
    {"league_id": 33, "competition": "World Cup - Qualification Oceania", "priority": 2},
    {"league_id": 34, "competition": "World Cup - Qualification South America", "priority": 2},
    {"league_id": 4, "competition": "Euro Championship", "priority": 3},
    {"league_id": 9, "competition": "Copa America", "priority": 3},
    {"league_id": 6, "competition": "Africa Cup of Nations", "priority": 3},
    {"league_id": 7, "competition": "Asian Cup", "priority": 3},
    {"league_id": 5, "competition": "UEFA Nations League", "priority": 4},
    {"league_id": 10, "competition": "Friendlies", "priority": 5},
]


def load_dotenv(path: Path = ENV_PATH) -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def request_json(path: str, api_key: str, params: dict) -> tuple[dict, int]:
    url = f"https://v3.football.api-sports.io{path}?{urlencode(params)}"
    request = Request(url, headers={"x-apisports-key": api_key})
    try:
        with urlopen(request, timeout=30) as response:
            return json.loads(response.read().decode("utf-8")), response.status
    except HTTPError as exc:
        if exc.code == 429:
            raise RuntimeError("api_football quota_limit: HTTP 429")
        body = exc.read().decode("utf-8", errors="ignore")
        raise RuntimeError(f"api_football HTTP {exc.code}: {body}") from exc


def load_state() -> dict:
    if not STATE_PATH.exists():
        return {"completed": []}
    state = json.loads(STATE_PATH.read_text(encoding="utf-8"))
    state.setdefault("completed", [])
    state.setdefault("completed_seasons", [])
    return state


def save_state(state: dict) -> None:
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8")


def task_key(league_id: int, season: int, page: int) -> str:
    return f"{league_id}:{season}:{page}"


def season_key(league_id: int, season: int) -> str:
    return f"{league_id}:{season}"


def normalize_team(value: str) -> str:
    return " ".join(str(value).replace("\xa0", " ").split()).strip()


def parse_1x2(values: list[dict], home_team: str, away_team: str) -> tuple[float | None, float | None, float | None]:
    home_odds = draw_odds = away_odds = None
    for value in values:
        label = str(value.get("value", "")).strip().lower()
        odd = value.get("odd")
        if label in {"home", "1"} or normalize_team(label).lower() == home_team.lower():
            home_odds = odd
        elif label in {"draw", "x"}:
            draw_odds = odd
        elif label in {"away", "2"} or normalize_team(label).lower() == away_team.lower():
            away_odds = odd
    return home_odds, draw_odds, away_odds


def parse_odds_response(payload: dict, fallback_competition: str) -> list[dict]:
    rows = []
    for item in payload.get("response", []):
        fixture = item.get("fixture", {})
        league = item.get("league", {})
        home_team = normalize_team(item.get("teams", {}).get("home", {}).get("name", ""))
        away_team = normalize_team(item.get("teams", {}).get("away", {}).get("name", ""))
        date = pd.to_datetime(fixture.get("date"), errors="coerce")
        if pd.isna(date) or not home_team or not away_team:
            continue
        competition = league.get("name") or fallback_competition
        for bookmaker in item.get("bookmakers", []):
            bookmaker_name = bookmaker.get("name") or bookmaker.get("id")
            for bet in bookmaker.get("bets", []):
                bet_name = str(bet.get("name", "")).strip().lower()
                if bet_name not in {"match winner", "1x2"}:
                    continue
                home_odds, draw_odds, away_odds = parse_1x2(bet.get("values", []), home_team, away_team)
                if home_odds and draw_odds and away_odds:
                    rows.append(
                        {
                            "date": date.strftime("%Y-%m-%d"),
                            "competition": competition,
                            "home_team": home_team,
                            "away_team": away_team,
                            "home_odds": home_odds,
                            "draw_odds": draw_odds,
                            "away_odds": away_odds,
                            "bookmaker": bookmaker_name,
                        }
                    )
    return rows


def normalize_odds(data: pd.DataFrame) -> pd.DataFrame:
    if data.empty:
        return pd.DataFrame(columns=ODDS_COLUMNS)
    data = data.copy()
    data["date"] = pd.to_datetime(data["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    for column in ["competition", "home_team", "away_team", "bookmaker"]:
        data[column] = data[column].astype(str).str.strip()
    for column in ["home_odds", "draw_odds", "away_odds"]:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    data = data.dropna(subset=ODDS_COLUMNS)
    data = data[(data["home_odds"] > 1) & (data["draw_odds"] > 1) & (data["away_odds"] > 1)]
    return data[ODDS_COLUMNS].drop_duplicates(ODDS_COLUMNS).sort_values(["date", "competition", "home_team", "away_team", "bookmaker"])


def append_rows(rows: list[dict]) -> None:
    if not rows:
        return
    existing = pd.read_csv(RAW_OUTPUT, encoding="utf-8") if RAW_OUTPUT.exists() else pd.DataFrame(columns=ODDS_COLUMNS)
    output = normalize_odds(pd.concat([existing, pd.DataFrame(rows)], ignore_index=True))
    RAW_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(RAW_OUTPUT, index=False, encoding="utf-8")


def ensure_raw_output_exists() -> None:
    if RAW_OUTPUT.exists():
        return
    RAW_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(columns=ODDS_COLUMNS).to_csv(RAW_OUTPUT, index=False, encoding="utf-8")


def write_report(rows: list[dict]) -> None:
    if not rows:
        return
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    existing = pd.read_csv(REPORT_PATH, encoding="utf-8") if REPORT_PATH.exists() else pd.DataFrame()
    pd.concat([existing, pd.DataFrame(rows)], ignore_index=True).to_csv(REPORT_PATH, index=False, encoding="utf-8")


def fetch_odds(max_api_calls: int | None = None) -> pd.DataFrame:
    if NO_PAID_API_MODE:
        write_report(
            [
                {
                    "status": "no_paid_api_mode",
                    "message": "NO_PAID_API_MODE=True: historical paid odds fetch is disabled.",
                }
            ]
        )
        print("NO_PAID_API_MODE=True: skipping historical odds API fetch.")
        return pd.DataFrame(columns=ODDS_COLUMNS)

    load_dotenv()
    provider = os.environ.get("ODDS_PROVIDER", "").strip()
    api_key = os.environ.get("ODDS_API_KEY", "").strip()
    if provider != "api_football":
        raise RuntimeError("ODDS_PROVIDER 必須是 api_football。")
    if not api_key:
        raise RuntimeError("缺少 ODDS_API_KEY。")

    state = load_state()
    completed = set(state.get("completed", []))
    completed_seasons = set(state.get("completed_seasons", []))
    report_rows = []
    api_calls = 0

    for competition in sorted(COMPETITIONS, key=lambda item: item["priority"]):
        for season in SEASONS:
            season_done_key = season_key(competition["league_id"], season)
            if season_done_key in completed_seasons:
                continue
            page = 1
            while True:
                key = task_key(competition["league_id"], season, page)
                if key in completed:
                    page += 1
                    continue
                if max_api_calls is not None and api_calls >= max_api_calls:
                    write_report(report_rows)
                    print(f"已達 max_api_calls={max_api_calls}，可重跑腳本繼續 resume。")
                    ensure_raw_output_exists()
                    return pd.read_csv(RAW_OUTPUT, encoding="utf-8")

                try:
                    payload, status = request_json(
                        "/odds",
                        api_key,
                        {"league": competition["league_id"], "season": season, "page": page},
                    )
                    api_calls += 1
                except (RuntimeError, URLError, TimeoutError) as exc:
                    report_rows.append(
                        {
                            "league_id": competition["league_id"],
                            "competition": competition["competition"],
                            "season": season,
                            "page": page,
                            "status": "quota_or_provider_error",
                            "message": str(exc),
                        }
                    )
                    write_report(report_rows)
                    print(f"API 停止: {exc}")
                    ensure_raw_output_exists()
                    return pd.read_csv(RAW_OUTPUT, encoding="utf-8")

                errors = payload.get("errors")
                if errors:
                    report_rows.append(
                        {
                            "league_id": competition["league_id"],
                            "competition": competition["competition"],
                            "season": season,
                            "page": page,
                            "status": "provider_error",
                            "message": json.dumps(errors, ensure_ascii=False),
                        }
                    )
                    completed.add(key)
                    state["completed"] = sorted(completed)
                    save_state(state)
                    break

                rows = parse_odds_response(payload, competition["competition"])
                append_rows(rows)
                report_rows.append(
                    {
                        "league_id": competition["league_id"],
                        "competition": competition["competition"],
                        "season": season,
                        "page": page,
                        "status": "ok" if rows else "no_odds_data",
                        "message": f"api_status={status}; results={payload.get('results', 0)}; rows={len(rows)}",
                    }
                )
                completed.add(key)
                state["completed"] = sorted(completed)
                if int(payload.get("paging", {}).get("current", page)) >= int(payload.get("paging", {}).get("total", page)):
                    completed_seasons.add(season_done_key)
                    state["completed_seasons"] = sorted(completed_seasons)
                save_state(state)

                paging = payload.get("paging", {})
                if int(paging.get("current", page)) >= int(paging.get("total", page)):
                    break
                page += 1

    write_report(report_rows)
    ensure_raw_output_exists()
    return pd.read_csv(RAW_OUTPUT, encoding="utf-8")


def team_key(value: str) -> str:
    return "".join(character for character in str(value).lower() if character.isalnum())


def build_market_probabilities(data: pd.DataFrame) -> pd.DataFrame:
    raw_h = 1 / data["home_odds"]
    raw_d = 1 / data["draw_odds"]
    raw_a = 1 / data["away_odds"]
    overround = raw_h + raw_d + raw_a
    data["market_H"] = raw_h / overround
    data["market_D"] = raw_d / overround
    data["market_A"] = raw_a / overround
    return data


def merge_training_with_odds() -> pd.DataFrame:
    if not TRAINING_PATH.exists():
        raise FileNotFoundError(f"找不到 training data: {TRAINING_PATH}")
    training = pd.read_csv(TRAINING_PATH, encoding="utf-8")
    odds = pd.read_csv(RAW_OUTPUT, encoding="utf-8") if RAW_OUTPUT.exists() else pd.DataFrame(columns=ODDS_COLUMNS)
    odds = normalize_odds(odds)

    if odds.empty:
        output = training.copy()
        for column in ["home_odds", "draw_odds", "away_odds", "bookmaker", *MARKET_COLUMNS]:
            output[column] = pd.NA
    else:
        consensus = (
            odds.groupby(["date", "home_team", "away_team"], as_index=False)[["home_odds", "draw_odds", "away_odds"]]
            .mean()
            .assign(bookmaker="consensus")
        )
        consensus = build_market_probabilities(consensus)
        training = training.copy()
        training["date"] = pd.to_datetime(training["date"], errors="coerce").dt.strftime("%Y-%m-%d")
        training["_home_key"] = training["home_team"].map(team_key)
        training["_away_key"] = training["away_team"].map(team_key)
        consensus["_home_key"] = consensus["home_team"].map(team_key)
        consensus["_away_key"] = consensus["away_team"].map(team_key)
        output = training.merge(
            consensus[["date", "_home_key", "_away_key", "home_odds", "draw_odds", "away_odds", "bookmaker", *MARKET_COLUMNS]],
            on=["date", "_home_key", "_away_key"],
            how="left",
        ).drop(columns=["_home_key", "_away_key"])

    PROCESSED_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(PROCESSED_OUTPUT, index=False, encoding="utf-8")
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-api-calls", type=int, default=None)
    parser.add_argument("--skip-fetch", action="store_true")
    args = parser.parse_args()

    if args.skip_fetch:
        ensure_raw_output_exists()
        odds = pd.read_csv(RAW_OUTPUT, encoding="utf-8")
    else:
        odds = fetch_odds(args.max_api_calls)
    merged = merge_training_with_odds()
    odds_rows = len(odds)
    matched_rows = int(merged["home_odds"].notna().sum()) if "home_odds" in merged.columns else 0
    print(f"historical odds rows: {odds_rows}")
    print(f"training rows with odds: {matched_rows}")
    print(f"raw output: {RAW_OUTPUT}")
    print(f"processed output: {PROCESSED_OUTPUT}")
    print(f"fetch report: {REPORT_PATH}")


if __name__ == "__main__":
    main()
