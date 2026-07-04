import argparse
import json
import re
import sys
from datetime import date, datetime, timezone
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agents.article_intel_parser import ArticleIntelParser
from src.paths import PROCESSED_DATA_DIR


FIXTURES_PATH = PROCESSED_DATA_DIR / "worldcup_fixtures.csv"
RESOLVED_FIXTURES_PATH = PROCESSED_DATA_DIR / "worldcup_fixtures_resolved.csv"
OUTPUT_PATH = PROCESSED_DATA_DIR / "worldcup_article_intel.csv"
DEBUG_PATH = PROJECT_ROOT / "reports" / "theanalyst_intel_debug.csv"
CACHE_PATH = PROJECT_ROOT / "data" / "cache" / "worldcup_theanalyst_intel_cache.json"
CACHE_TTL_HOURS = 24
SOURCE_NAME = "The Analyst"

OUTPUT_COLUMNS = [
    "match_key",
    "date",
    "home_team",
    "away_team",
    "source_name",
    "source_url",
    "source_status",
    "title",
    "body_char_count",
    "injuries_home",
    "injuries_away",
    "suspensions_home",
    "suspensions_away",
    "expected_lineup_home",
    "expected_lineup_away",
    "coach_comments_home",
    "coach_comments_away",
    "intel_has_content",
    "confidence",
    "fetched_at",
    "article_source_count",
    "article_content_count",
    "cross_source_agreement",
    "conflict_detected",
    "final_article_confidence",
    "quality_reason",
]
DEBUG_COLUMNS = [
    "match_key",
    "query_or_slug",
    "url",
    "status_code",
    "fetch_success",
    "parser_success",
    "intel_has_content",
    "confidence",
    "error_message",
]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def latest_fixtures_path() -> Path:
    if RESOLVED_FIXTURES_PATH.exists():
        return RESOLVED_FIXTURES_PATH
    return FIXTURES_PATH


def match_key(row) -> str:
    return f"{row.date}|{row.home_team}|{row.away_team}"


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


def cache_is_fresh(entry: dict | None) -> bool:
    if not entry:
        return False
    fetched_at = pd.to_datetime(entry.get("fetched_at"), utc=True, errors="coerce")
    if pd.isna(fetched_at):
        return False
    return (datetime.now(timezone.utc) - fetched_at.to_pydatetime()).total_seconds() < CACHE_TTL_HOURS * 3600


def load_fixtures(path: Path, match_date: str | None, match_key_arg: str | None, max_matches: int | None) -> pd.DataFrame:
    fixtures = pd.read_csv(path, encoding="utf-8")
    required = ["date", "home_team", "away_team"]
    missing = [column for column in required if column not in fixtures.columns]
    if missing:
        raise ValueError(f"fixtures missing columns: {missing}")
    fixtures = fixtures.copy()
    fixtures["date"] = pd.to_datetime(fixtures["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    for column in ["home_team", "away_team"]:
        fixtures[column] = fixtures[column].fillna("").astype(str).str.strip()
    if match_key_arg:
        parts = [part.strip() for part in match_key_arg.split("|")]
        if len(parts) != 3:
            raise ValueError('--match-key must use "date|home_team|away_team"')
        key_date, home_team, away_team = parts
        selected = fixtures[
            (fixtures["date"] == pd.to_datetime(key_date, errors="raise").strftime("%Y-%m-%d"))
            & (fixtures["home_team"].str.casefold() == home_team.casefold())
            & (fixtures["away_team"].str.casefold() == away_team.casefold())
        ]
    else:
        selected_date = pd.to_datetime(match_date or date.today(), errors="raise").strftime("%Y-%m-%d")
        selected = fixtures[fixtures["date"] == selected_date]
    selected = selected.sort_values(["date", "home_team", "away_team"]).reset_index(drop=True)
    if max_matches is not None:
        selected = selected.head(max_matches)
    return selected


def team_slug(team: str) -> str:
    aliases = {
        "United States": "usa",
        "USA": "usa",
        "Bosnia and Herzegovina": "bosnia-herzegovina",
        "Bosnia & Herzegovina": "bosnia-herzegovina",
        "Bosnia-Herzegovina": "bosnia-herzegovina",
        "Czech Republic": "czech-republic",
        "Czechia": "czech-republic",
        "Ivory Coast": "ivory-coast",
        "Côte d'Ivoire": "ivory-coast",
        "DR Congo": "dr-congo",
        "Congo DR": "dr-congo",
        "South Korea": "south-korea",
        "Korea Republic": "south-korea",
        "Curacao": "curacao",
        "Curaçao": "curacao",
    }
    value = aliases.get(team, team)
    return re.sub(r"[^A-Za-z0-9]+", "-", value).strip("-").lower()


def theanalyst_urls(home_team: str, away_team: str) -> list[tuple[str, str]]:
    home = team_slug(home_team)
    away = team_slug(away_team)
    base = "https://theanalyst.com/articles"
    return [
        ("direct_slug_home_away", f"{base}/{home}-vs-{away}-prediction-world-cup-2026-match-preview"),
        ("direct_slug_away_home", f"{base}/{away}-vs-{home}-prediction-world-cup-2026-match-preview"),
    ]


def team_terms(team: str) -> list[str]:
    aliases = {
        "United States": ["united states", "usa", "usmnt", "stars and stripes"],
        "Bosnia-Herzegovina": ["bosnia-herzegovina", "bosnia and herzegovina", "bosnia"],
        "Bosnia and Herzegovina": ["bosnia-herzegovina", "bosnia and herzegovina", "bosnia"],
        "Ivory Coast": ["ivory coast", "côte d'ivoire", "cote d'ivoire"],
        "South Korea": ["south korea", "korea republic"],
        "DR Congo": ["dr congo", "congo dr"],
        "Curacao": ["curacao", "curaçao"],
    }
    return aliases.get(str(team), [str(team).strip().lower()])


def article_matches_fixture(fixture, article: dict) -> bool:
    text = " ".join(
        [
            str(article.get("title", "")),
            str(article.get("meta_description", "")),
            str(article.get("article_body", ""))[:3000],
        ]
    ).casefold()
    return any(term.casefold() in text for term in team_terms(fixture.home_team)) and any(term.casefold() in text for term in team_terms(fixture.away_team))


def row_from_article(fixture, url: str, article: dict, intel: dict) -> dict:
    confidence = min(0.99, float(intel["confidence"]) + 0.15)
    return {
        "match_key": match_key(fixture),
        "date": fixture.date,
        "home_team": fixture.home_team,
        "away_team": fixture.away_team,
        "source_name": SOURCE_NAME,
        "source_url": url,
        "source_status": "article_body_parsed",
        "title": article.get("title", "unknown") or "unknown",
        "body_char_count": article.get("body_char_count", 0),
        "injuries_home": intel["injuries"],
        "injuries_away": intel["injuries"],
        "suspensions_home": intel["suspensions"],
        "suspensions_away": intel["suspensions"],
        "expected_lineup_home": intel["expected_lineup"],
        "expected_lineup_away": intel["expected_lineup"],
        "coach_comments_home": intel["coach_comments"],
        "coach_comments_away": intel["coach_comments"],
        "intel_has_content": bool(intel["intel_has_content"]),
        "confidence": confidence,
        "fetched_at": utc_now(),
        "article_source_count": 1,
        "article_content_count": 1 if intel["intel_has_content"] else 0,
        "cross_source_agreement": False,
        "conflict_detected": False,
        "final_article_confidence": confidence,
        "quality_reason": "single-source The Analyst article intel" if intel["intel_has_content"] else "The Analyst article found but no usable intel extracted",
    }


def not_found_row(fixture, error_message: str) -> dict:
    return {
        "match_key": match_key(fixture),
        "date": fixture.date,
        "home_team": fixture.home_team,
        "away_team": fixture.away_team,
        "source_name": SOURCE_NAME,
        "source_url": "unknown",
        "source_status": "theanalyst_not_found",
        "title": "unknown",
        "body_char_count": 0,
        "injuries_home": "unknown",
        "injuries_away": "unknown",
        "suspensions_home": "unknown",
        "suspensions_away": "unknown",
        "expected_lineup_home": "unknown",
        "expected_lineup_away": "unknown",
        "coach_comments_home": "unknown",
        "coach_comments_away": "unknown",
        "intel_has_content": False,
        "confidence": 0,
        "fetched_at": utc_now(),
        "article_source_count": 1,
        "article_content_count": 0,
        "cross_source_agreement": False,
        "conflict_detected": False,
        "final_article_confidence": 0,
        "quality_reason": error_message or "The Analyst preview not found",
    }


def fetch_theanalyst_for_match(fixture) -> tuple[dict | None, list[dict]]:
    parser = ArticleIntelParser()
    debug_rows = []
    for query_or_slug, url in theanalyst_urls(fixture.home_team, fixture.away_team):
        article = parser.fetch(url)
        intel = parser.extract_intel(article)
        error_message = article.get("failure_reason", "")
        if article.get("fetch_success") and not article_matches_fixture(fixture, article):
            intel["intel_has_content"] = False
            intel["injuries"] = "unknown"
            intel["suspensions"] = "unknown"
            intel["expected_lineup"] = "unknown"
            intel["coach_comments"] = "unknown"
            error_message = "not_match_relevant"
        row = row_from_article(fixture, url, article, intel)
        debug_rows.append(
            {
                "match_key": match_key(fixture),
                "query_or_slug": query_or_slug,
                "url": url,
                "status_code": article.get("status_code", ""),
                "fetch_success": bool(article.get("fetch_success")),
                "parser_success": bool(article.get("parser_success")),
                "intel_has_content": bool(intel["intel_has_content"]),
                "confidence": row["confidence"],
                "error_message": error_message,
            }
        )
        if row["intel_has_content"]:
            return row, debug_rows
    return not_found_row(fixture, "theanalyst_not_found"), debug_rows or [
        {
            "match_key": match_key(fixture),
            "query_or_slug": "direct_slug",
            "url": "",
            "status_code": "",
            "fetch_success": False,
            "parser_success": False,
            "intel_has_content": False,
            "confidence": 0,
            "error_message": "theanalyst_not_found",
        }
    ]


def merge_existing(rows: list[dict]) -> pd.DataFrame:
    new_output = pd.DataFrame(rows, columns=OUTPUT_COLUMNS)
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
    if combined.empty:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)
    combined = combined.drop_duplicates(["match_key"], keep="last")
    return combined[OUTPUT_COLUMNS].sort_values(["date", "home_team", "away_team"]).reset_index(drop=True)


def write_debug(debug_rows: list[dict]) -> None:
    DEBUG_PATH.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(debug_rows, columns=DEBUG_COLUMNS).to_csv(DEBUG_PATH, index=False, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixtures", type=Path, default=None)
    parser.add_argument("--date", help="Match date in YYYY-MM-DD format.")
    parser.add_argument("--match-key", help='Single match key in "date|home_team|away_team" format.')
    parser.add_argument("--max-matches", type=int, default=3)
    parser.add_argument("--force-refresh", action="store_true")
    args = parser.parse_args()

    fixtures = load_fixtures(args.fixtures or latest_fixtures_path(), args.date, args.match_key, args.max_matches)
    cache = load_cache()
    rows = []
    debug_rows = []
    for fixture in fixtures.itertuples(index=False):
        key = match_key(fixture)
        cached = cache.get(key)
        if cached and cache_is_fresh(cached) and not args.force_refresh:
            cached_row = cached.get("row")
            cached_debug = cached.get("debug_rows", [])
            if cached_row:
                cached_row["source_status"] = "cached_theanalyst_intel"
                rows.append(cached_row)
            for debug in cached_debug:
                debug = dict(debug)
                debug["error_message"] = "cached_theanalyst_intel" if cached_row else "cached_theanalyst_not_found"
                debug_rows.append(debug)
            continue
        row, debug = fetch_theanalyst_for_match(fixture)
        rows.append(row)
        debug_rows.extend(debug)
        cache[key] = {"fetched_at": utc_now(), "row": row, "debug_rows": debug}
        save_cache(cache)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    merge_existing(rows).to_csv(OUTPUT_PATH, index=False, encoding="utf-8")
    write_debug(debug_rows)
    print(f"The Analyst intel written: {OUTPUT_PATH}")
    content_count = sum(str(row.get("intel_has_content", "")).lower() in {"true", "1"} for row in rows)
    print(f"Rows with content: {content_count}")
    print(f"Debug report: {DEBUG_PATH}")


if __name__ == "__main__":
    main()
