import argparse
import json
import os
import subprocess
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

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
RAW_DIR = PROJECT_ROOT / "data" / "raw"
FIXTURES_PATH = PROCESSED_DIR / "worldcup_fixtures.csv"
ODDS_OUTPUT = PROCESSED_DIR / "worldcup_odds.csv"
CONSENSUS_OUTPUT = PROCESSED_DIR / "worldcup_consensus_odds.csv"
MARKET_OUTPUT = PROCESSED_DIR / "worldcup_market_predictions.csv"
REPORT_PATH = PROJECT_ROOT / "reports" / "worldcup_odds_missing_report.csv"
ENV_PATH = PROJECT_ROOT / ".env"

KEY_COLUMNS = ["date", "group", "home_team", "away_team"]
BOOKMAKER_ODDS_COLUMNS = KEY_COLUMNS + ["bookmaker", "home_odds", "draw_odds", "away_odds"]
CONSENSUS_ODDS_COLUMNS = KEY_COLUMNS + ["home_odds", "draw_odds", "away_odds"]
MARKET_COLUMNS = KEY_COLUMNS + ["market_H", "market_D", "market_A"]
LOCAL_ODDS_CANDIDATES = ["worldcup_odds.csv", "world_cup_odds.csv", "worldcup_consensus_odds.csv"]
SUPPORTED_PROVIDERS = {"the_odds_api", "api_football"}


def load_dotenv(path: Path = ENV_PATH) -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def write_missing_report(rows: list[dict]) -> None:
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(REPORT_PATH, index=False, encoding="utf-8")


def stop_for_no_paid_api_mode() -> bool:
    if not NO_PAID_API_MODE:
        return False
    write_missing_report(
        [
            {
                "data_type": "worldcup_odds",
                "missing_columns": ", ".join(BOOKMAKER_ODDS_COLUMNS),
                "message": "NO_PAID_API_MODE=True: paid odds APIs are disabled. Use data/manual/worldcup_odds_manual.csv or OCR/manual import.",
                "required_source": "data/manual/worldcup_odds_manual.csv",
            }
        ]
    )
    print("NO_PAID_API_MODE=True: skipping paid odds API fetch. Use manual odds import instead.")
    return True


def classify_provider_error(error: Exception) -> str:
    message = str(error)
    if "Free plans do not have access to this season" in message:
        return "provider_limit"
    if "quota" in message.lower() or "rate limit" in message.lower():
        return "quota_limit"
    return "provider_error"


def normalize_key_columns(data: pd.DataFrame) -> pd.DataFrame:
    data = data.copy()
    data["date"] = pd.to_datetime(data["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    for column in ["group", "home_team", "away_team"]:
        if column in data.columns:
            data[column] = data[column].astype(str).str.strip()
    return data


def load_fixtures(path: Path = FIXTURES_PATH) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"找不到 World Cup fixtures: {path}，請先執行 scripts/fetch_worldcup_fixtures.py")
    fixtures = pd.read_csv(path, encoding="utf-8")
    missing = [column for column in KEY_COLUMNS if column not in fixtures.columns]
    if missing:
        raise ValueError(f"World Cup fixtures 缺少欄位: {missing}")
    return normalize_key_columns(fixtures[KEY_COLUMNS]).dropna(subset=["date"])


def find_local_odds() -> Path | None:
    for directory in [PROCESSED_DIR, RAW_DIR]:
        for filename in LOCAL_ODDS_CANDIDATES:
            path = directory / filename
            if path.exists():
                return path
    return None


def validate_bookmaker_odds(data: pd.DataFrame, source: str) -> pd.DataFrame:
    data = data.copy()
    if "bookmaker" not in data.columns:
        data["bookmaker"] = "local"
    missing = [column for column in BOOKMAKER_ODDS_COLUMNS if column not in data.columns]
    if missing:
        raise ValueError(f"{source} 缺少欄位: {missing}")

    odds = normalize_key_columns(data[BOOKMAKER_ODDS_COLUMNS])
    odds["bookmaker"] = odds["bookmaker"].astype(str).str.strip()
    for column in ["home_odds", "draw_odds", "away_odds"]:
        odds[column] = pd.to_numeric(odds[column], errors="coerce")
    odds = odds.dropna(subset=BOOKMAKER_ODDS_COLUMNS)
    odds = odds[(odds["home_odds"] > 1) & (odds["draw_odds"] > 1) & (odds["away_odds"] > 1)]
    return odds.drop_duplicates(KEY_COLUMNS + ["bookmaker"], keep="first")


def load_local_odds(path: Path) -> pd.DataFrame:
    return validate_bookmaker_odds(pd.read_csv(path, encoding="utf-8"), str(path))


def team_key(value: str) -> str:
    return "".join(character for character in str(value).lower() if character.isalnum())


def request_json(url: str, headers: dict | None = None) -> tuple[object, dict]:
    request = Request(url, headers=headers or {"User-Agent": "worldcup-ai-agent/1.0"})
    try:
        with urlopen(request, timeout=30) as response:
            payload = json.loads(response.read().decode("utf-8"))
            response_headers = dict(response.headers.items())
    except HTTPError as exc:
        if exc.code == 429:
            raise RuntimeError("API quota 用完或 rate limit 觸發。") from exc
        raise
    return payload, response_headers


def match_event_to_fixture(event_date: object, home_team: str, away_team: str, fixtures: pd.DataFrame) -> dict | None:
    parsed_date = pd.to_datetime(event_date, errors="coerce")
    if pd.isna(parsed_date):
        return None
    event_date_string = parsed_date.strftime("%Y-%m-%d")
    event_teams = {team_key(home_team), team_key(away_team)}

    same_date = fixtures[fixtures["date"] == event_date_string]
    for fixture in same_date.itertuples(index=False):
        fixture_teams = {team_key(fixture.home_team), team_key(fixture.away_team)}
        if fixture_teams == event_teams:
            return fixture._asdict()
    return None


def fetch_the_odds_api(api_key: str, fixtures: pd.DataFrame) -> pd.DataFrame:
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
    payload, headers = request_json(url)
    if int(headers.get("x-requests-remaining", "1")) <= 0:
        raise RuntimeError("The Odds API quota 用完。")
    if not isinstance(payload, list):
        raise RuntimeError(f"The Odds API 回應格式不是 list: {payload}")

    rows = []
    for event in payload:
        fixture = match_event_to_fixture(
            event.get("commence_time"),
            event.get("home_team", ""),
            event.get("away_team", ""),
            fixtures,
        )
        if fixture is None:
            continue

        for bookmaker in event.get("bookmakers", []):
            bookmaker_name = bookmaker.get("title") or bookmaker.get("key")
            for market in bookmaker.get("markets", []):
                if market.get("key") != "h2h":
                    continue
                prices = {outcome.get("name"): outcome.get("price") for outcome in market.get("outcomes", [])}
                home_odds = prices.get(fixture["home_team"]) or prices.get(event.get("home_team"))
                away_odds = prices.get(fixture["away_team"]) or prices.get(event.get("away_team"))
                draw_odds = prices.get("Draw")
                if home_odds and draw_odds and away_odds:
                    rows.append(
                        {
                            **fixture,
                            "bookmaker": bookmaker_name,
                            "home_odds": home_odds,
                            "draw_odds": draw_odds,
                            "away_odds": away_odds,
                        }
                    )
    return validate_bookmaker_odds(pd.DataFrame(rows), "The Odds API") if rows else pd.DataFrame(columns=BOOKMAKER_ODDS_COLUMNS)


def fetch_api_football_fixture_ids(api_key: str, fixtures: pd.DataFrame) -> dict[tuple, int]:
    url = "https://v3.football.api-sports.io/fixtures?league=1&season=2026"
    payload, _ = request_json(url, headers={"x-apisports-key": api_key})
    if payload.get("errors"):
        raise RuntimeError(f"API-Football fixtures error: {payload['errors']}")
    if payload.get("paging", {}).get("current") == 1 and payload.get("paging", {}).get("total") == 0:
        raise RuntimeError("API-Football quota 用完或無可用 fixtures 回應。")

    fixture_ids = {}
    for item in payload.get("response", []):
        fixture_info = item.get("fixture", {})
        teams = item.get("teams", {})
        home_team = teams.get("home", {}).get("name", "")
        away_team = teams.get("away", {}).get("name", "")
        fixture = match_event_to_fixture(fixture_info.get("date"), home_team, away_team, fixtures)
        if fixture:
            fixture_ids[tuple(fixture[column] for column in KEY_COLUMNS)] = fixture_info.get("id")
    return fixture_ids


def parse_api_football_1x2(values: list[dict], home_team: str, away_team: str) -> tuple[float | None, float | None, float | None]:
    home_odds = draw_odds = away_odds = None
    for value in values:
        label = str(value.get("value", "")).lower()
        odd = value.get("odd")
        if label in {"home", "1"} or team_key(label) == team_key(home_team):
            home_odds = odd
        elif label in {"draw", "x"}:
            draw_odds = odd
        elif label in {"away", "2"} or team_key(label) == team_key(away_team):
            away_odds = odd
    return home_odds, draw_odds, away_odds


def fetch_api_football(api_key: str, fixtures: pd.DataFrame) -> pd.DataFrame:
    fixture_ids = fetch_api_football_fixture_ids(api_key, fixtures)
    rows = []
    for fixture in fixtures.itertuples(index=False):
        fixture_dict = fixture._asdict()
        fixture_id = fixture_ids.get(tuple(fixture_dict[column] for column in KEY_COLUMNS))
        if not fixture_id:
            continue

        url = f"https://v3.football.api-sports.io/odds?fixture={fixture_id}"
        payload, _ = request_json(url, headers={"x-apisports-key": api_key})
        if payload.get("errors"):
            raise RuntimeError(f"API-Football odds error: {payload['errors']}")
        for item in payload.get("response", []):
            for bookmaker in item.get("bookmakers", []):
                bookmaker_name = bookmaker.get("name")
                for bet in bookmaker.get("bets", []):
                    bet_name = str(bet.get("name", "")).lower()
                    if bet_name not in {"match winner", "1x2"}:
                        continue
                    home_odds, draw_odds, away_odds = parse_api_football_1x2(
                        bet.get("values", []),
                        fixture_dict["home_team"],
                        fixture_dict["away_team"],
                    )
                    if home_odds and draw_odds and away_odds:
                        rows.append(
                            {
                                **fixture_dict,
                                "bookmaker": bookmaker_name,
                                "home_odds": home_odds,
                                "draw_odds": draw_odds,
                                "away_odds": away_odds,
                            }
                        )
    return validate_bookmaker_odds(pd.DataFrame(rows), "API-Football") if rows else pd.DataFrame(columns=BOOKMAKER_ODDS_COLUMNS)


def align_odds_to_fixtures(bookmaker_odds: pd.DataFrame, fixtures: pd.DataFrame) -> pd.DataFrame:
    covered = bookmaker_odds[KEY_COLUMNS].drop_duplicates()
    merged = fixtures.merge(covered.assign(has_odds=True), on=KEY_COLUMNS, how="left", validate="one_to_one")
    missing_rows = merged["has_odds"].isna()
    if missing_rows.any():
        missing = merged.loc[missing_rows, KEY_COLUMNS]
        write_missing_report(
            [
                {
                    "data_type": "worldcup_odds",
                    "missing_rows": int(missing_rows.sum()),
                    "message": "odds 未覆蓋所有 72 場 fixtures；未補假 odds。",
                    "examples": missing.head(10).to_json(orient="records", force_ascii=False),
                }
            ]
        )
        raise RuntimeError(
            f"odds 只覆蓋 {len(fixtures) - int(missing_rows.sum())}/{len(fixtures)} 場，已輸出 missing report: {REPORT_PATH}"
        )
    return bookmaker_odds[BOOKMAKER_ODDS_COLUMNS].sort_values(KEY_COLUMNS + ["bookmaker"]).reset_index(drop=True)


def build_consensus_odds(bookmaker_odds: pd.DataFrame) -> pd.DataFrame:
    consensus = (
        bookmaker_odds.groupby(KEY_COLUMNS, as_index=False)[["home_odds", "draw_odds", "away_odds"]]
        .mean()
        .sort_values(KEY_COLUMNS)
        .reset_index(drop=True)
    )
    return consensus[CONSENSUS_ODDS_COLUMNS]


def build_market_probabilities(consensus_odds: pd.DataFrame) -> pd.DataFrame:
    market = consensus_odds[KEY_COLUMNS].copy()
    raw_h = 1 / consensus_odds["home_odds"]
    raw_d = 1 / consensus_odds["draw_odds"]
    raw_a = 1 / consensus_odds["away_odds"]
    overround = raw_h + raw_d + raw_a
    market["market_H"] = raw_h / overround
    market["market_D"] = raw_d / overround
    market["market_A"] = raw_a / overround
    return market[MARKET_COLUMNS]


def save_outputs(bookmaker_odds: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    consensus = build_consensus_odds(bookmaker_odds)
    market = build_market_probabilities(consensus)
    bookmaker_odds.to_csv(ODDS_OUTPUT, index=False, encoding="utf-8")
    consensus.to_csv(CONSENSUS_OUTPUT, index=False, encoding="utf-8")
    market.to_csv(MARKET_OUTPUT, index=False, encoding="utf-8")
    return bookmaker_odds, consensus, market


def fetch_or_load_odds() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if stop_for_no_paid_api_mode():
        raise SystemExit(0)

    fixtures = load_fixtures()
    local_path = find_local_odds()
    if local_path:
        local_odds = load_local_odds(local_path)
        bookmaker_odds = align_odds_to_fixtures(local_odds, fixtures)
        print(f"已使用本機 World Cup odds: {local_path} ({len(bookmaker_odds)} bookmaker rows)")
        return save_outputs(bookmaker_odds)

    load_dotenv()
    provider = os.environ.get("ODDS_PROVIDER", "the_odds_api").strip().lower()
    api_key = os.environ.get("ODDS_API_KEY", "").strip()
    if provider not in SUPPORTED_PROVIDERS:
        raise RuntimeError(f"不支援的 ODDS_PROVIDER: {provider}，可用值: {sorted(SUPPORTED_PROVIDERS)}")
    if not api_key:
        write_missing_report(
            [
                {
                    "data_type": "worldcup_odds",
                    "missing_rows": len(fixtures),
                    "provider": provider,
                    "message": "缺少 ODDS_API_KEY，無法取得賠率；未產生假 odds。",
                    "required_env": "ODDS_API_KEY",
                }
            ]
        )
        raise RuntimeError(f"缺少 ODDS_API_KEY，已輸出 missing report: {REPORT_PATH}")

    try:
        if provider == "the_odds_api":
            api_odds = fetch_the_odds_api(api_key, fixtures)
        else:
            api_odds = fetch_api_football(api_key, fixtures)
    except (HTTPError, URLError, TimeoutError, RuntimeError, ValueError) as exc:
        error_type = classify_provider_error(exc)
        write_missing_report(
            [
                {
                    "data_type": "worldcup_odds",
                    "missing_rows": len(fixtures),
                    "provider": provider,
                    "error_type": error_type,
                    "message": f"{provider} 無法取得可用賠率或 quota 已用完: {exc}",
                }
            ]
        )
        raise RuntimeError(f"{provider} 無法取得可用賠率，已輸出 missing report: {REPORT_PATH}") from exc

    bookmaker_odds = align_odds_to_fixtures(api_odds, fixtures)
    print(f"已從 {provider} 取得 World Cup odds: {len(bookmaker_odds)} bookmaker rows")
    return save_outputs(bookmaker_odds)


def run_command(command: list[str]) -> bool:
    print(f"執行: {' '.join(command)}")
    completed = subprocess.run(command, cwd=PROJECT_ROOT, check=False)
    if completed.returncode != 0:
        print(f"命令失敗，exit code={completed.returncode}: {' '.join(command)}")
        return False
    return True


def run_downstream_pipeline() -> None:
    commands = [
        [sys.executable, "scripts/build_worldcup_features.py"],
        [sys.executable, "scripts/run_worldcup_betting_agents.py"],
    ]
    for command in commands:
        if not run_command(command):
            break


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-downstream", action="store_true")
    args = parser.parse_args()

    try:
        bookmaker_odds, consensus, market = fetch_or_load_odds()
    except Exception as exc:
        print(f"錯誤: {exc}")
        print("缺少 odds，改用 no-odds model-only mode 繼續產生觀察報告。")
        if not args.skip_downstream:
            run_command([sys.executable, "scripts/run_worldcup_betting_agents.py", "--model-only"])
        return

    print(f"已輸出 bookmaker odds: {ODDS_OUTPUT} ({len(bookmaker_odds)} rows)")
    print(f"已輸出 consensus odds: {CONSENSUS_OUTPUT} ({len(consensus)} rows)")
    print(f"已輸出 market probabilities: {MARKET_OUTPUT} ({len(market)} rows)")

    if not args.skip_downstream:
        run_downstream_pipeline()


if __name__ == "__main__":
    main()
