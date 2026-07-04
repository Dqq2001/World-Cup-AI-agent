import argparse
import json
import random
import sys
import time
from datetime import date, datetime, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import quote_plus
from urllib.request import Request, urlopen

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.paths import PROCESSED_DATA_DIR
from agents.article_intel_parser import ArticleIntelParser


FIXTURES_PATH = PROCESSED_DATA_DIR / "worldcup_fixtures.csv"
RESOLVED_FIXTURES_PATH = PROCESSED_DATA_DIR / "worldcup_fixtures_resolved.csv"
CACHE_PATH = PROJECT_ROOT / "data" / "cache" / "worldcup_intel_search_cache.json"
OUTPUT_PATH = PROCESSED_DATA_DIR / "worldcup_news_intel.csv"
MISSING_REPORT = PROJECT_ROOT / "reports" / "worldcup_intel_search_missing_report.csv"
ESPN_TEAM_NEWS_PATH = PROCESSED_DATA_DIR / "worldcup_espn_team_news.csv"
FIFA_MATCH_CENTRE_PATH = PROCESSED_DATA_DIR / "worldcup_fifa_match_centre_intel.csv"
CACHE_TTL_HOURS = 48
REQUEST_SLEEP_MIN_SECONDS = 10
REQUEST_SLEEP_MAX_SECONDS = 20
OUTPUT_COLUMNS = [
    "date",
    "home_team",
    "away_team",
    "team_news_home",
    "team_news_away",
    "injuries_home",
    "injuries_away",
    "suspensions_home",
    "suspensions_away",
    "expected_lineup_home",
    "expected_lineup_away",
    "coach_comments_home",
    "coach_comments_away",
    "source_url",
    "source_type",
    "source_status",
    "intel_has_content",
    "intel_confidence_level",
    "confidence",
    "fetched_at",
]
CONTENT_COLUMNS = [
    "team_news_home",
    "team_news_away",
    "injuries_home",
    "injuries_away",
    "suspensions_home",
    "suspensions_away",
    "expected_lineup_home",
    "expected_lineup_away",
    "coach_comments_home",
    "coach_comments_away",
]


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def load_cache() -> dict:
    if not CACHE_PATH.exists():
        return {}
    try:
        return json.loads(CACHE_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def save_cache(cache: dict) -> None:
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    CACHE_PATH.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")


def latest_fixtures_path() -> Path:
    if RESOLVED_FIXTURES_PATH.exists():
        return RESOLVED_FIXTURES_PATH
    return FIXTURES_PATH


def cache_key(row) -> str:
    return f"{row.date}|{row.home_team}|{row.away_team}"


def cache_is_fresh(entry: dict) -> bool:
    fetched_at = pd.to_datetime(entry.get("fetched_at"), utc=True, errors="coerce")
    if pd.isna(fetched_at):
        return False
    return (utc_now() - fetched_at.to_pydatetime()).total_seconds() < CACHE_TTL_HOURS * 3600


def cached_row(entry: dict | None) -> dict | None:
    if not entry or "row" not in entry:
        return None
    row = dict(entry["row"])
    row = finalize_intel_row(row)
    row["source_status"] = "cached_search_results" if row["intel_has_content"] else "cached_search_results_no_content"
    return row


def has_content(row: dict) -> bool:
    for column in CONTENT_COLUMNS:
        value = str(row.get(column, "")).strip().lower()
        if value and value not in {"unknown", "nan", "none", "<na>"}:
            return True
    return False


def finalize_intel_row(row: dict) -> dict:
    for column in OUTPUT_COLUMNS:
        row.setdefault(column, "unknown")
    content = has_content(row)
    row["intel_has_content"] = bool(content)
    row["intel_confidence_level"] = "MEDIUM" if content and row.get("source_url") != "unknown" else "LOW"
    return row


def resolve_match_date(args: argparse.Namespace) -> str:
    if args.date:
        return pd.to_datetime(args.date, errors="raise").strftime("%Y-%m-%d")
    return date.today().strftime("%Y-%m-%d")


def load_matchday_fixtures(path: Path, match_date: str, start_index: int, max_matches: int | None) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing fixtures: {path}")
    fixtures = pd.read_csv(path, encoding="utf-8")
    required = ["date", "home_team", "away_team"]
    missing = [column for column in required if column not in fixtures.columns]
    if missing:
        raise ValueError(f"fixtures missing columns: {missing}")
    fixtures = fixtures.copy()
    fixtures["date"] = pd.to_datetime(fixtures["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    selected = fixtures[fixtures["date"] == match_date].sort_values(["date", "home_team", "away_team"]).reset_index(drop=True)
    if start_index:
        selected = selected.iloc[start_index:].reset_index(drop=True)
    if max_matches is not None:
        selected = selected.head(max_matches).reset_index(drop=True)
    return selected


def load_single_match_fixture(path: Path, match_key: str | None, match_date: str | None, team: str | None) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing fixtures: {path}")
    fixtures = pd.read_csv(path, encoding="utf-8")
    required = ["date", "home_team", "away_team"]
    missing = [column for column in required if column not in fixtures.columns]
    if missing:
        raise ValueError(f"fixtures missing columns: {missing}")
    fixtures = fixtures.copy()
    fixtures["date"] = pd.to_datetime(fixtures["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    for column in ["home_team", "away_team"]:
        fixtures[column] = fixtures[column].fillna("").astype(str).str.strip()

    if match_key:
        parts = [part.strip() for part in match_key.split("|")]
        if len(parts) != 3:
            raise ValueError('--match-key must use "date|home_team|away_team"')
        key_date, key_home, key_away = parts
        selected = fixtures[
            (fixtures["date"] == pd.to_datetime(key_date, errors="raise").strftime("%Y-%m-%d"))
            & (fixtures["home_team"].str.casefold() == key_home.casefold())
            & (fixtures["away_team"].str.casefold() == key_away.casefold())
        ]
        return selected.reset_index(drop=True)

    if match_date and team:
        key_date = pd.to_datetime(match_date, errors="raise").strftime("%Y-%m-%d")
        team_key = team.strip().casefold()
        selected = fixtures[
            (fixtures["date"] == key_date)
            & (
                fixtures["home_team"].str.casefold().str.contains(team_key, regex=False)
                | fixtures["away_team"].str.casefold().str.contains(team_key, regex=False)
            )
        ]
        return selected.reset_index(drop=True)

    return pd.DataFrame(columns=fixtures.columns)


def gdelt_search(query: str) -> list[dict]:
    url = (
        "https://api.gdeltproject.org/api/v2/doc/doc"
        f"?query={quote_plus(query)}&mode=ArtList&format=json&maxrecords=5&sort=HybridRel"
    )
    request = Request(url, headers={"User-Agent": "worldcup-ai-agent/1.0"})
    with urlopen(request, timeout=30) as response:
        payload = json.loads(response.read().decode("utf-8", errors="ignore"))
    return payload.get("articles", [])


def queries_for_match(home_team: str, away_team: str) -> list[str]:
    return [
        f'"{home_team}" "{away_team}" World Cup team news injuries lineup',
        f'"{home_team}" "{away_team}" World Cup suspensions coach comments expected lineup',
    ]


def articles_to_team_intel(articles: list[dict]) -> dict:
    return ArticleIntelParser().best_intel_from_articles(articles)


def source_row_from_csv(path: Path, fixture, source_type: str, source_status: str) -> dict | None:
    if not path.exists():
        return None
    try:
        data = pd.read_csv(path, encoding="utf-8")
    except pd.errors.EmptyDataError:
        return None
    required = ["date", "home_team", "away_team"]
    if not all(column in data.columns for column in required):
        return None
    data = data.copy()
    data["date"] = pd.to_datetime(data["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    matched = data[
        (data["date"] == fixture.date)
        & (data["home_team"].astype(str).str.casefold() == str(fixture.home_team).casefold())
        & (data["away_team"].astype(str).str.casefold() == str(fixture.away_team).casefold())
    ]
    if matched.empty:
        return None
    source = matched.iloc[0].to_dict()
    row = empty_row(fixture, utc_now().isoformat())
    for column in OUTPUT_COLUMNS:
        if column in source and pd.notna(source[column]):
            row[column] = source[column]
    row["source_type"] = source_type
    row["source_status"] = source_status
    return finalize_intel_row(row)


def structured_fallback_row(fixture) -> dict | None:
    return (
        source_row_from_csv(ESPN_TEAM_NEWS_PATH, fixture, "espn", "espn_team_news_available")
        or source_row_from_csv(FIFA_MATCH_CENTRE_PATH, fixture, "fifa", "fifa_match_centre_available")
    )


def empty_row(fixture, fetched_at: str) -> dict:
    return {
        "date": fixture.date,
        "home_team": fixture.home_team,
        "away_team": fixture.away_team,
        "team_news_home": "unknown",
        "team_news_away": "unknown",
        "injuries_home": "unknown",
        "injuries_away": "unknown",
        "suspensions_home": "unknown",
        "suspensions_away": "unknown",
        "expected_lineup_home": "unknown",
        "expected_lineup_away": "unknown",
        "coach_comments_home": "unknown",
        "coach_comments_away": "unknown",
        "source_url": "unknown",
        "source_type": "search",
        "source_status": "not_run",
        "intel_has_content": False,
        "intel_confidence_level": "LOW",
        "confidence": "low",
        "fetched_at": fetched_at,
    }


def build_row(fixture, articles: list[dict], fetched_at: str, source_status: str) -> dict:
    intel = articles_to_team_intel(articles)
    resolved_source_status = (
        "article_body_parsed"
        if intel.get("intel_has_content") and float(intel.get("confidence", 0)) >= 0.75
        else source_status
    )
    row = empty_row(fixture, fetched_at)
    row.update(
        {
            "team_news_home": intel["team_news"],
            "team_news_away": intel["team_news"],
            "injuries_home": intel["injuries"],
            "injuries_away": intel["injuries"],
            "suspensions_home": intel["suspensions"],
            "suspensions_away": intel["suspensions"],
            "expected_lineup_home": intel["expected_lineup"],
            "expected_lineup_away": intel["expected_lineup"],
            "coach_comments_home": intel["coach_comments"],
            "coach_comments_away": intel["coach_comments"],
            "source_url": intel["source_url"],
            "source_status": resolved_source_status,
            "confidence": intel.get("confidence", "medium" if intel["source_url"] != "unknown" else "low"),
        }
    )
    return finalize_intel_row(row)


def merge_with_existing_output(new_rows: list[dict]) -> pd.DataFrame:
    new_rows = [finalize_intel_row(dict(row)) for row in new_rows]
    new_output = pd.DataFrame(new_rows, columns=OUTPUT_COLUMNS)
    if OUTPUT_PATH.exists():
        try:
            existing = pd.read_csv(OUTPUT_PATH, encoding="utf-8")
        except pd.errors.EmptyDataError:
            existing = pd.DataFrame(columns=OUTPUT_COLUMNS)
    else:
        existing = pd.DataFrame(columns=OUTPUT_COLUMNS)
    for column in OUTPUT_COLUMNS:
        if column not in existing.columns:
            existing[column] = pd.NA
    combined = pd.concat([existing[OUTPUT_COLUMNS], new_output], ignore_index=True)
    combined["date"] = pd.to_datetime(combined["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    combined = combined.drop_duplicates(["date", "home_team", "away_team"], keep="last")
    return combined[OUTPUT_COLUMNS].sort_values(["date", "home_team", "away_team"]).reset_index(drop=True)


def write_missing_report(rows: list[dict]) -> None:
    MISSING_REPORT.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(MISSING_REPORT, index=False, encoding="utf-8")


def sleep_between_requests() -> None:
    time.sleep(random.uniform(REQUEST_SLEEP_MIN_SECONDS, REQUEST_SLEEP_MAX_SECONDS))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixtures", type=Path, default=None)
    parser.add_argument("--date", help="Match date in YYYY-MM-DD format.")
    parser.add_argument("--match-key", help='Single match key in "date|home_team|away_team" format.')
    parser.add_argument("--team", help="Team name filter used together with --date.")
    parser.add_argument("--today", action="store_true", help="Use local date. This is also the default.")
    parser.add_argument("--max-matches", type=int, default=3)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--force-refresh", action="store_true")
    args = parser.parse_args()

    match_date = resolve_match_date(args)
    cache = load_cache()
    fixtures_path = args.fixtures or latest_fixtures_path()
    if args.match_key or args.team:
        fixtures = load_single_match_fixture(fixtures_path, args.match_key, args.date, args.team)
    else:
        fixtures = load_matchday_fixtures(fixtures_path, match_date, args.start_index, args.max_matches)
    rows = []
    missing_rows = []
    request_count = 0
    stop_fetching = False

    for fixture in fixtures.itertuples(index=False):
        key = cache_key(fixture)
        cached = cache.get(key)
        if cached and cache_is_fresh(cached) and not args.force_refresh:
            rows.append(finalize_intel_row(dict(cached["row"])))
            continue
        if stop_fetching:
            fallback = cached_row(cached) or structured_fallback_row(fixture) or empty_row(fixture, utc_now().isoformat())
            if fallback["source_status"] == "not_run":
                fallback["source_status"] = "rate_limited"
            rows.append(finalize_intel_row(fallback))
            continue

        fallback = structured_fallback_row(fixture)
        if fallback is not None and not args.force_refresh:
            rows.append(fallback)
            cache[key] = {"fetched_at": fallback["fetched_at"], "row": fallback}
            save_cache(cache)
            continue

        articles = []
        source_status = "no_search_results"
        for query in queries_for_match(fixture.home_team, fixture.away_team)[:2]:
            if request_count > 0:
                sleep_between_requests()
            try:
                articles.extend(gdelt_search(query))
                request_count += 1
            except HTTPError as exc:
                if exc.code == 429:
                    source_status = "rate_limited"
                    fallback = cached_row(cached) or structured_fallback_row(fixture)
                    if fallback is not None:
                        if fallback["source_status"] == "cached_search_results_no_content":
                            fallback["source_status"] = "rate_limited_cached_no_content"
                        rows.append(fallback)
                        missing_rows.append(
                            {
                                "date": fixture.date,
                                "home_team": fixture.home_team,
                                "away_team": fixture.away_team,
                                "query": query,
                                "message": "HTTP 429 encountered; used cached/structured fallback. Use manual intel if urgent.",
                            }
                        )
                        stop_fetching = True
                        articles = []
                        break
                    missing_rows.append(
                        {
                            "date": fixture.date,
                            "home_team": fixture.home_team,
                            "away_team": fixture.away_team,
                            "query": query,
                            "message": "HTTP 429 encountered; no cache available. Use manual intel if urgent.",
                        }
                    )
                    stop_fetching = True
                    break
                if exc.code == 403:
                    source_status = "source_blocked"
                else:
                    source_status = "parser_failed"
                missing_rows.append(
                    {
                        "date": fixture.date,
                        "home_team": fixture.home_team,
                        "away_team": fixture.away_team,
                        "query": query,
                        "message": f"HTTP error: {exc}",
                    }
                )
            except (URLError, TimeoutError, json.JSONDecodeError) as exc:
                source_status = "parser_failed"
                missing_rows.append(
                    {
                        "date": fixture.date,
                        "home_team": fixture.home_team,
                        "away_team": fixture.away_team,
                        "query": query,
                        "message": f"search failed: {exc}",
                    }
                )

        fetched_at = utc_now().isoformat()
        if rows and len(rows) > 0 and rows[-1].get("date") == fixture.date and rows[-1].get("home_team") == fixture.home_team and rows[-1].get("away_team") == fixture.away_team:
            continue
        if articles:
            source_status = "search_results_found"
            row = build_row(fixture, articles, fetched_at, source_status)
        else:
            row = empty_row(fixture, fetched_at)
            row["source_status"] = source_status
            row = finalize_intel_row(row)
            missing_rows.append(
                {
                    "date": fixture.date,
                    "home_team": fixture.home_team,
                    "away_team": fixture.away_team,
                    "query": "",
                    "message": source_status,
                }
            )
        row = finalize_intel_row(row)
        rows.append(row)
        cache[key] = {"fetched_at": fetched_at, "row": row}
        save_cache(cache)

    output = merge_with_existing_output(rows)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(OUTPUT_PATH, index=False, encoding="utf-8")
    if not missing_rows:
        missing_rows.append(
            {
                "date": match_date,
                "home_team": "",
                "away_team": "",
                "query": "",
                "message": "No search errors. Unknown fields mean no relevant public article was found, or no fixtures existed for this date.",
            }
        )
    if not ESPN_TEAM_NEWS_PATH.exists():
        missing_rows.append(
            {
                "date": match_date,
                "home_team": "",
                "away_team": "",
                "query": "",
                "message": f"ESPN team news fallback unavailable: {ESPN_TEAM_NEWS_PATH}",
            }
        )
    if not FIFA_MATCH_CENTRE_PATH.exists():
        missing_rows.append(
            {
                "date": match_date,
                "home_team": "",
                "away_team": "",
                "query": "",
                "message": f"FIFA match centre fallback unavailable: {FIFA_MATCH_CENTRE_PATH}",
            }
        )
    write_missing_report(missing_rows)
    print(f"已輸出 search intel: {OUTPUT_PATH} ({len(output)} rows)")
    print(f"已輸出 missing report: {MISSING_REPORT}")
    print(f"match date: {match_date}")
    print(f"requests sent: {request_count}")
    if stop_fetching:
        print("遇到 HTTP 429，已 graceful stop，cache 未清空。")


if __name__ == "__main__":
    main()
