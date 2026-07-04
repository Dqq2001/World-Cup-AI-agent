import argparse
import hashlib
import json
import random
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIXTURES_PATH = PROJECT_ROOT / "data" / "processed" / "worldcup_fixtures.csv"
RESOLVED_FIXTURES_PATH = PROJECT_ROOT / "data" / "processed" / "worldcup_fixtures_resolved.csv"
OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "worldcup_results.csv"
CACHE_PATH = PROJECT_ROOT / "data" / "cache" / "worldcup_results_cache.json"
MISSING_REPORT = PROJECT_ROOT / "reports" / "worldcup_results_fetch_missing_report.csv"
RESULTS_DEBUG_PATH = PROJECT_ROOT / "reports" / "results_debug.csv"
FETCH_DEBUG_PATH = PROJECT_ROOT / "reports" / "results_fetch_debug.csv"

OUTPUT_COLUMNS = [
    "date",
    "group",
    "stage",
    "round",
    "match_id",
    "home_team",
    "away_team",
    "home_goals",
    "away_goals",
    "status",
    "aet",
    "penalties_home",
    "penalties_away",
    "winner",
    "source_url",
    "source_event_date",
    "matched_date_source",
    "fetched_at",
]

ESPN_SCOREBOARD_URL = "https://site.api.espn.com/apis/site/v2/sports/soccer/fifa.world/scoreboard?dates={date}"
CACHE_TTL_SECONDS = 6 * 60 * 60

TEAM_ALIASES = {
    "usa": "unitedstates",
    "unitedstatesofamerica": "unitedstates",
    "czechia": "czechrepublic",
    "czechrepublic": "czechrepublic",
    "southkorea": "southkorea",
    "korearepublic": "southkorea",
    "republicofkorea": "southkorea",
    "cotedivoire": "ivorycoast",
    "côtedivoire": "ivorycoast",
    "ivorycoast": "ivorycoast",
    "curacao": "curacao",
    "curaçao": "curacao",
    "drcongo": "drcongo",
    "congodr": "drcongo",
    "congodemocraticrepublic": "drcongo",
    "bosniaherzegovina": "bosniaandherzegovina",
    "bosniaandherzegovina": "bosniaandherzegovina",
}


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def normalize_team(value: str) -> str:
    key = "".join(character for character in str(value).lower() if character.isalnum())
    return TEAM_ALIASES.get(key, key)


def latest_fixtures_path() -> Path:
    if RESOLVED_FIXTURES_PATH.exists():
        return RESOLVED_FIXTURES_PATH
    return FIXTURES_PATH


def file_hash(path: Path) -> str:
    if not path.exists():
        return ""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_date(value: str | None) -> pd.Timestamp:
    if value:
        parsed = pd.to_datetime(value, errors="raise")
    else:
        parsed = pd.Timestamp.today()
    return parsed.normalize()


def date_range_strings(end_date: pd.Timestamp, days_back: int) -> list[str]:
    start_date = end_date - pd.Timedelta(days=days_back)
    return [(start_date + pd.Timedelta(days=offset)).strftime("%Y%m%d") for offset in range(days_back + 1)]


def date_range_between(start_date: pd.Timestamp, end_date: pd.Timestamp) -> list[str]:
    days = int((end_date - start_date).days)
    return [(start_date + pd.Timedelta(days=offset)).strftime("%Y%m%d") for offset in range(days + 1)]


def normalize_fixtures(data: pd.DataFrame) -> pd.DataFrame:
    data = data.copy()
    data["date"] = pd.to_datetime(data["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    data["fixture_date_dt"] = pd.to_datetime(data["date"], errors="coerce")
    if "group" not in data.columns:
        data["group"] = ""
    for column in ["stage", "round", "match_id"]:
        if column not in data.columns:
            data[column] = ""
    for column in ["group", "stage", "round", "match_id", "home_team", "away_team"]:
        data[column] = data[column].astype(str).str.strip()
    data["_home_key"] = data["home_team"].map(normalize_team)
    data["_away_key"] = data["away_team"].map(normalize_team)
    return data


def load_fixtures() -> pd.DataFrame:
    fixtures_path = latest_fixtures_path()
    if not fixtures_path.exists():
        raise FileNotFoundError(f"World Cup fixtures not found: {fixtures_path}")
    fixtures = pd.read_csv(fixtures_path, encoding="utf-8")
    required = ["date", "home_team", "away_team"]
    missing = [column for column in required if column not in fixtures.columns]
    if missing:
        raise ValueError(f"{fixtures_path.name} missing columns: {missing}")
    columns = [column for column in ["date", "group", "stage", "round", "match_id", "home_team", "away_team"] if column in fixtures.columns]
    return normalize_fixtures(fixtures[columns]).drop_duplicates(["date", "home_team", "away_team"])


def fixture_start_date(fixtures: pd.DataFrame) -> pd.Timestamp:
    start_date = fixtures["fixture_date_dt"].min()
    if pd.isna(start_date):
        raise ValueError("worldcup_fixtures.csv does not contain a valid start date")
    return start_date.normalize()


def read_cache() -> dict | None:
    if not CACHE_PATH.exists():
        return None
    try:
        return json.loads(CACHE_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def cache_is_fresh(cache: dict, requested_dates: list[str], fixtures_hash: str) -> bool:
    fetched_at = cache.get("fetched_at")
    cached_dates = cache.get("requested_dates")
    cached_fixtures_hash = cache.get("fixtures_hash", "")
    if not fetched_at or cached_dates != requested_dates:
        return False
    if cached_fixtures_hash != fixtures_hash:
        return False
    try:
        fetched = datetime.fromisoformat(fetched_at)
    except ValueError:
        return False
    return (datetime.now(timezone.utc) - fetched).total_seconds() < CACHE_TTL_SECONDS


def write_cache(rows: list[dict], requested_dates: list[str], fixtures_hash: str, stopped_reason: str | None = None) -> None:
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "fetched_at": now_iso(),
        "source": "espn_scoreboard",
        "requested_dates": requested_dates,
        "fixtures_hash": fixtures_hash,
        "stopped_reason": stopped_reason,
        "rows": rows,
    }
    CACHE_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def normalize_output_data(data: pd.DataFrame) -> pd.DataFrame:
    if data.empty:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)
    data = data.copy()
    for column in OUTPUT_COLUMNS:
        if column not in data.columns:
            data[column] = pd.NA
    data = data[OUTPUT_COLUMNS]
    data["date"] = pd.to_datetime(data["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    data["home_goals"] = pd.to_numeric(data["home_goals"], errors="coerce")
    data["away_goals"] = pd.to_numeric(data["away_goals"], errors="coerce")
    data["penalties_home"] = pd.to_numeric(data["penalties_home"], errors="coerce")
    data["penalties_away"] = pd.to_numeric(data["penalties_away"], errors="coerce")
    data["status"] = data["status"].astype(str).str.strip().str.lower()
    return data


def read_existing_output() -> pd.DataFrame:
    if not OUTPUT_PATH.exists():
        return pd.DataFrame(columns=OUTPUT_COLUMNS)
    try:
        return normalize_output_data(pd.read_csv(OUTPUT_PATH, encoding="utf-8"))
    except (pd.errors.EmptyDataError, UnicodeDecodeError):
        return pd.DataFrame(columns=OUTPUT_COLUMNS)


def merge_result_rows(new_data: pd.DataFrame, existing_data: pd.DataFrame) -> pd.DataFrame:
    combined = pd.concat([existing_data, new_data], ignore_index=True)
    combined = normalize_output_data(combined)
    combined = combined[
        combined["date"].notna()
        & combined["home_team"].notna()
        & combined["away_team"].notna()
        & combined["home_goals"].notna()
        & combined["away_goals"].notna()
    ]
    if combined.empty:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)
    combined["_completed_rank"] = (combined["status"] == "completed").astype(int)
    combined = combined.sort_values(["date", "home_team", "away_team", "_completed_rank", "fetched_at"])
    combined = combined.drop_duplicates(["date", "home_team", "away_team"], keep="last")
    combined = combined.drop(columns=["_completed_rank"])
    return combined.sort_values(["date", "group", "home_team", "away_team"]).reset_index(drop=True)


def write_output(rows: list[dict], merge_existing: bool = True) -> pd.DataFrame:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    data = normalize_output_data(pd.DataFrame(rows, columns=OUTPUT_COLUMNS))
    if merge_existing:
        data = merge_result_rows(data, read_existing_output())
    data.to_csv(OUTPUT_PATH, index=False, encoding="utf-8")
    return data


def write_missing_report(rows: list[dict]) -> None:
    MISSING_REPORT.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(MISSING_REPORT, index=False, encoding="utf-8")


def write_results_debug(data: pd.DataFrame) -> None:
    RESULTS_DEBUG_PATH.parent.mkdir(parents=True, exist_ok=True)
    if data.empty:
        pd.DataFrame(columns=["date", "home_team", "away_team", "status", "home_goals", "away_goals", "source"]).to_csv(
            RESULTS_DEBUG_PATH,
            index=False,
            encoding="utf-8",
        )
        return
    debug = data.copy()
    debug["source"] = debug.get("source_url", "")
    columns = ["date", "home_team", "away_team", "status", "home_goals", "away_goals", "source"]
    debug[columns].to_csv(RESULTS_DEBUG_PATH, index=False, encoding="utf-8")


def write_fetch_debug(rows: list[dict]) -> None:
    FETCH_DEBUG_PATH.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        rows,
        columns=[
            "source",
            "requested_dates",
            "matches_found",
            "completed_found",
            "parsed_success",
            "missing_dates",
            "error_message",
        ],
    ).to_csv(FETCH_DEBUG_PATH, index=False, encoding="utf-8")


def fetch_json(url: str) -> dict:
    request = Request(
        url,
        headers={
            "User-Agent": "worldcup-ai-agent/1.0 (+local dashboard results fetch)",
            "Accept": "application/json",
        },
    )
    with urlopen(request, timeout=20) as response:
        return json.loads(response.read().decode("utf-8"))


def is_completed_status(status_type: dict) -> bool:
    status_name = str(status_type.get("name", "")).lower()
    status_description = str(status_type.get("description", "")).lower()
    return bool(status_type.get("completed")) or status_name in {"final", "full_time"} or status_description == "final"


def read_optional_int(mapping: dict, keys: list[str]):
    for key in keys:
        value = mapping.get(key)
        if value in (None, ""):
            continue
        try:
            return int(value)
        except (TypeError, ValueError):
            continue
    return pd.NA


def winner_from_competitors(home: dict, away: dict, home_goals: int, away_goals: int) -> str:
    if home.get("winner") is True:
        return (home.get("team") or {}).get("displayName", "")
    if away.get("winner") is True:
        return (away.get("team") or {}).get("displayName", "")
    if home_goals > away_goals:
        return (home.get("team") or {}).get("displayName", "")
    if away_goals > home_goals:
        return (away.get("team") or {}).get("displayName", "")
    return ""


def parse_espn_events(payload: dict, source_url: str, fetched_at: str) -> tuple[list[dict], dict]:
    rows = []
    stats = {"events": 0, "completed_events": 0, "parsed_completed_events": 0}
    for event in payload.get("events", []):
        stats["events"] += 1
        competition = (event.get("competitions") or [{}])[0]
        status = competition.get("status") or event.get("status") or {}
        status_type = status.get("type") or {}
        if not is_completed_status(status_type):
            continue
        stats["completed_events"] += 1

        competitors = competition.get("competitors") or []
        home = next((item for item in competitors if item.get("homeAway") == "home"), None)
        away = next((item for item in competitors if item.get("homeAway") == "away"), None)
        if not home or not away:
            continue

        try:
            home_goals = int(home.get("score"))
            away_goals = int(away.get("score"))
        except (TypeError, ValueError):
            continue

        # ESPN timestamps are generally UTC ISO strings. We normalize to a date and
        # later allow ±1 day against the local World Cup fixture date.
        event_dt = pd.to_datetime(event.get("date"), errors="coerce", utc=True)
        if pd.isna(event_dt):
            continue

        event_date = event_dt.date().isoformat()
        status_text = f"{status_type.get('name', '')} {status_type.get('description', '')}".lower()
        penalties_home = read_optional_int(home, ["shootoutScore", "penaltyScore", "penalties"])
        penalties_away = read_optional_int(away, ["shootoutScore", "penaltyScore", "penalties"])
        rows.append(
            {
                "source_event_date": event_date,
                "source_event_date_dt": pd.Timestamp(event_date),
                "_home_key": normalize_team((home.get("team") or {}).get("displayName", "")),
                "_away_key": normalize_team((away.get("team") or {}).get("displayName", "")),
                "home_goals": home_goals,
                "away_goals": away_goals,
                "status": "completed",
                "aet": "extra" in status_text or "aet" in status_text,
                "penalties_home": penalties_home,
                "penalties_away": penalties_away,
                "winner": winner_from_competitors(home, away, home_goals, away_goals),
                "source_url": source_url,
                "fetched_at": fetched_at,
            }
        )
        stats["parsed_completed_events"] += 1
    return rows, stats


def describe_date_match(fixture_date: pd.Timestamp, source_event_date: pd.Timestamp) -> str:
    delta = int((source_event_date - fixture_date).days)
    if delta == 0:
        return "exact"
    if delta == 1:
        return "source_event_date_plus_1"
    if delta == -1:
        return "source_event_date_minus_1"
    return f"source_event_date_delta_{delta}"


def align_to_fixtures(parsed_rows: list[dict], fixtures: pd.DataFrame) -> list[dict]:
    if not parsed_rows:
        return []
    parsed = pd.DataFrame(parsed_rows)
    joined = fixtures.merge(parsed, on=["_home_key", "_away_key"], how="inner")
    if joined.empty:
        return []

    joined["date_delta_abs"] = (joined["source_event_date_dt"] - joined["fixture_date_dt"]).abs().dt.days
    joined = joined[joined["date_delta_abs"] <= 1].copy()
    if joined.empty:
        return []

    joined["matched_date_source"] = joined.apply(
        lambda row: describe_date_match(row["fixture_date_dt"], row["source_event_date_dt"]),
        axis=1,
    )
    joined = joined.sort_values(["date_delta_abs", "date", "home_team", "away_team"]).drop_duplicates(
        ["date", "group", "home_team", "away_team"],
        keep="first",
    )
    return joined[OUTPUT_COLUMNS].drop_duplicates(OUTPUT_COLUMNS).to_dict(orient="records")


def fetch_results(
    date: str | None = None,
    days_back: int | None = None,
    force_refresh: bool = False,
    all_completed: bool = False,
) -> tuple[pd.DataFrame, str]:
    end_date = parse_date(date)
    fixtures_path = latest_fixtures_path()
    fixtures_hash = file_hash(fixtures_path)
    fixtures = load_fixtures()
    if all_completed or days_back is None:
        start_date = fixture_start_date(fixtures)
        requested_dates = date_range_between(start_date, end_date)
    else:
        requested_dates = date_range_strings(end_date, days_back)
    cache = read_cache()
    fetch_debug_rows = []
    if cache and cache_is_fresh(cache, requested_dates, fixtures_hash) and not force_refresh:
        data = write_output(cache.get("rows", []), merge_existing=True)
        write_results_debug(data)
        write_fetch_debug(
            [
                {
                    "source": "espn_scoreboard_cache",
                    "requested_dates": "|".join(requested_dates),
                    "matches_found": "",
                    "completed_found": "",
                    "parsed_success": len(cache.get("rows", [])),
                    "missing_dates": "",
                    "error_message": "cache_hit",
                }
            ]
        )
        write_missing_report(
            [
                {
                    "status": "cache_hit",
                    "message": "Using cached World Cup results fetch; cache TTL is 6 hours.",
                    "rows": len(data),
                    "requested_dates": "|".join(requested_dates),
                }
            ]
        )
        return data, "cache"

    all_rows: list[dict] = []
    report_rows: list[dict] = []
    stopped_reason = None
    total_events = 0
    total_completed = 0
    total_parsed = 0

    for index, date_text in enumerate(requested_dates):
        if index > 0:
            time.sleep(random.uniform(5, 10))
        source_url = ESPN_SCOREBOARD_URL.format(date=date_text)
        fetched_at = now_iso()
        try:
            payload = fetch_json(source_url)
        except HTTPError as exc:
            if exc.code in {403, 429}:
                stopped_reason = f"provider_blocked_http_{exc.code}"
                report_rows.append(
                    {
                        "status": stopped_reason,
                        "message": "Stopped fetching after provider returned 403/429. Kept already fetched rows.",
                        "source_url": source_url,
                    }
                )
                break
            report_rows.append({"status": f"http_{exc.code}", "message": str(exc), "source_url": source_url})
            fetch_debug_rows.append(
                {
                    "source": "espn_scoreboard",
                    "requested_dates": date_text,
                    "matches_found": 0,
                    "completed_found": 0,
                    "parsed_success": 0,
                    "missing_dates": date_text,
                    "error_message": str(exc),
                }
            )
            continue
        except URLError as exc:
            report_rows.append({"status": "network_error", "message": str(exc), "source_url": source_url})
            fetch_debug_rows.append(
                {
                    "source": "espn_scoreboard",
                    "requested_dates": date_text,
                    "matches_found": 0,
                    "completed_found": 0,
                    "parsed_success": 0,
                    "missing_dates": date_text,
                    "error_message": str(exc),
                }
            )
            continue

        parsed, stats = parse_espn_events(payload, source_url, fetched_at)
        aligned = align_to_fixtures(parsed, fixtures)
        all_rows.extend(aligned)
        total_events += stats["events"]
        total_completed += stats["completed_events"]
        total_parsed += stats["parsed_completed_events"]
        missing_date = date_text if stats["completed_events"] and not aligned else ""
        fetch_debug_rows.append(
            {
                "source": "espn_scoreboard",
                "requested_dates": date_text,
                "matches_found": stats["events"],
                "completed_found": stats["completed_events"],
                "parsed_success": len(aligned),
                "missing_dates": missing_date,
                "error_message": "" if not missing_date else "completed events returned but no fixture match after normalization",
            }
        )
        report_rows.append(
            {
                "status": "fetched",
                "message": (
                    f"date={date_text}; espn_events={stats['events']}; "
                    f"completed={stats['completed_events']}; parsed={stats['parsed_completed_events']}; "
                    f"matched_worldcup_2026={len(aligned)}"
                ),
                "source_url": source_url,
            }
        )
        write_cache(all_rows, requested_dates, fixtures_hash, stopped_reason=None)

    all_rows = pd.DataFrame(all_rows, columns=OUTPUT_COLUMNS).drop_duplicates(OUTPUT_COLUMNS).to_dict(orient="records")
    write_cache(all_rows, requested_dates, fixtures_hash, stopped_reason=stopped_reason)
    data = write_output(all_rows, merge_existing=True)
    write_results_debug(data)
    write_fetch_debug(fetch_debug_rows)

    if data.empty:
        if total_events or total_completed or total_parsed:
            message = "ESPN returned events but none matched local fixtures after normalization."
        else:
            message = "No completed World Cup 2026 results were found. No scores were fabricated."
        report_rows.append(
            {
                "status": "no_completed_results",
                "message": message,
                "source_url": "",
                "espn_events": total_events,
                "completed_events": total_completed,
                "parsed_completed_events": total_parsed,
            }
        )
    write_missing_report(report_rows)
    return data, "network"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", default=None, help="End date in YYYY-MM-DD. Defaults to today.")
    parser.add_argument("--days-back", type=int, default=None)
    parser.add_argument("--all-completed", action="store_true")
    parser.add_argument("--force-refresh", action="store_true")
    args = parser.parse_args()

    data, mode = fetch_results(
        date=args.date,
        days_back=args.days_back,
        force_refresh=args.force_refresh,
        all_completed=args.all_completed,
    )
    print(f"World Cup results fetch mode: {mode}")
    print(f"Output: {OUTPUT_PATH}")
    print(f"Rows: {len(data)}")
    print(f"Missing report: {MISSING_REPORT}")


if __name__ == "__main__":
    main()
