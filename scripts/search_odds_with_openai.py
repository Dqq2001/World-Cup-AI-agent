import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.openai_search_client import run_web_search


FIXTURES_CANDIDATES = [
    PROJECT_ROOT / "data" / "processed" / "worldcup_fixtures_resolved.csv",
    PROJECT_ROOT / "data" / "processed" / "worldcup_fixtures.csv",
]
ODDS_OUTPUT = PROJECT_ROOT / "data" / "processed" / "worldcup_openai_odds.csv"
MARKET_OUTPUT = PROJECT_ROOT / "data" / "processed" / "worldcup_openai_market_predictions.csv"
PRIMARY_CONSENSUS_OUTPUT = PROJECT_ROOT / "data" / "processed" / "worldcup_consensus_odds.csv"
PRIMARY_MARKET_OUTPUT = PROJECT_ROOT / "data" / "processed" / "worldcup_market_predictions.csv"
DEBUG_OUTPUT = PROJECT_ROOT / "reports" / "openai_odds_debug.csv"

KEY_COLUMNS = ["date", "group", "home_team", "away_team"]
ODDS_COLUMNS = KEY_COLUMNS + [
    "home_odds",
    "draw_odds",
    "away_odds",
    "odds_status",
    "odds_source",
    "source_urls",
    "confidence",
    "fetched_at",
]
MARKET_COLUMNS = KEY_COLUMNS + ["market_H", "market_D", "market_A"]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def latest_fixtures_path() -> Path:
    for path in FIXTURES_CANDIDATES:
        if path.exists():
            return path
    raise FileNotFoundError("Missing World Cup fixtures CSV.")


def normalize_date(value: str) -> str:
    return pd.to_datetime(value, errors="raise").strftime("%Y-%m-%d")


def load_fixtures(match_date: str, match_key: str | None) -> pd.DataFrame:
    fixtures = pd.read_csv(latest_fixtures_path(), encoding="utf-8")
    for column in KEY_COLUMNS:
        if column not in fixtures.columns:
            fixtures[column] = "" if column == "group" else pd.NA
    fixtures = fixtures.copy()
    fixtures["date"] = pd.to_datetime(fixtures["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    fixtures["home_team"] = fixtures["home_team"].astype(str).str.strip()
    fixtures["away_team"] = fixtures["away_team"].astype(str).str.strip()
    fixtures = fixtures[(fixtures["date"] == match_date) & (fixtures["home_team"] != "TBD") & (fixtures["away_team"] != "TBD")]
    if match_key:
        parts = match_key.split("|")
        if len(parts) != 3:
            raise ValueError('--match-key must use "date|home|away"')
        key_date, home, away = parts
        key_date = normalize_date(key_date)
        fixtures = fixtures[
            (fixtures["date"] == key_date)
            & (fixtures["home_team"].str.casefold() == home.strip().casefold())
            & (fixtures["away_team"].str.casefold() == away.strip().casefold())
        ]
    return fixtures[KEY_COLUMNS].drop_duplicates().reset_index(drop=True)


def has_existing_complete_odds(row) -> bool:
    if not PRIMARY_CONSENSUS_OUTPUT.exists():
        return False
    try:
        odds = pd.read_csv(PRIMARY_CONSENSUS_OUTPUT, encoding="utf-8")
    except pd.errors.EmptyDataError:
        return False
    required = ["date", "home_team", "away_team", "home_odds", "draw_odds", "away_odds"]
    if odds.empty or not all(column in odds.columns for column in required):
        return False
    odds = odds.copy()
    odds["date"] = pd.to_datetime(odds["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    match = odds[
        (odds["date"] == row.date)
        & (odds["home_team"].astype(str).str.casefold() == str(row.home_team).casefold())
        & (odds["away_team"].astype(str).str.casefold() == str(row.away_team).casefold())
    ]
    if match.empty:
        return False
    values = pd.to_numeric(match.iloc[-1][["home_odds", "draw_odds", "away_odds"]], errors="coerce")
    return bool(values.notna().all() and (values > 1).all())


def prompt_for_match(row) -> str:
    return f"""
Search the public web for 1X2 betting odds for this World Cup 2026 match:
date: {row.date}
home_team: {row.home_team}
away_team: {row.away_team}

Use these queries:
- "{row.home_team} vs {row.away_team} odds World Cup 2026"
- "{row.home_team} {row.away_team} betting odds"
- "{row.home_team} vs {row.away_team} 1X2 odds"

Return ONLY strict valid JSON:
{{
  "date": "{row.date}",
  "home_team": "{row.home_team}",
  "away_team": "{row.away_team}",
  "home_odds": null,
  "draw_odds": null,
  "away_odds": null,
  "odds_source": "...",
  "source_urls": ["..."],
  "confidence": 0.0,
  "fetched_at": "{utc_now()}"
}}

Rules:
- Accept decimal odds only, e.g. 1.75, 3.40, 5.20.
- If a source provides American odds, convert them to decimal odds.
- Prefer Pinnacle, Bet365, DraftKings, FanDuel, Oddschecker, BetMGM.
- If home/draw/away are not all supported by source URLs, return null for missing odds.
- If no public source URL supports the odds, source_urls must be [].
- Do not invent odds.
- Do not include markdown or commentary outside the JSON object.
""".strip()


def parse_json_text(text: str) -> dict:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`").removeprefix("json").strip()
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start >= 0 and end >= start:
        cleaned = cleaned[start : end + 1]
    return json.loads(cleaned)


def decimal_from_value(value) -> float | None:
    if value is None or pd.isna(value):
        return None
    text = str(value).strip().replace(",", "")
    if not text or text.lower() in {"null", "none", "unknown", "nan"}:
        return None
    if text.startswith(("+", "-")):
        try:
            american = float(text)
        except ValueError:
            return None
        if american > 0:
            return round(1 + american / 100, 4)
        if american < 0:
            return round(1 + 100 / abs(american), 4)
        return None
    try:
        number = float(text)
    except ValueError:
        return None
    if number > 1:
        return round(number, 4)
    return None


def odds_are_complete(row: dict) -> bool:
    return all(row.get(column) is not None and row.get(column) > 1 for column in ["home_odds", "draw_odds", "away_odds"])


def source_urls_text(urls) -> str:
    if not isinstance(urls, list):
        return ""
    clean = []
    for url in urls:
        text = str(url).strip()
        if text.startswith(("http://", "https://")) and text not in clean:
            clean.append(text)
    return "; ".join(clean)


def empty_odds_row(row, status: str, fetched_at: str | None = None) -> dict:
    return {
        "date": row.date,
        "group": getattr(row, "group", ""),
        "home_team": row.home_team,
        "away_team": row.away_team,
        "home_odds": pd.NA,
        "draw_odds": pd.NA,
        "away_odds": pd.NA,
        "odds_status": status,
        "odds_source": "unknown",
        "source_urls": "",
        "confidence": 0.0,
        "fetched_at": fetched_at or utc_now(),
    }


def odds_row_from_openai(row, result: dict) -> tuple[dict, dict]:
    query = prompt_for_match(row)
    debug = {
        "match_key": f"{row.date}|{row.home_team}|{row.away_team}",
        "query": query,
        "openai_status": "failed",
        "json_parse_success": False,
        "home_odds": "",
        "draw_odds": "",
        "away_odds": "",
        "source_urls_count": 0,
        "accepted": False,
        "reject_reason": "",
    }
    if not result.get("success"):
        debug["openai_status"] = str(result.get("error_message") or result.get("status_code") or "openai_failed")
        debug["reject_reason"] = "openai_failed"
        return empty_odds_row(row, "missing"), debug

    try:
        parsed = parse_json_text(str(result.get("final_text", "")))
        debug["json_parse_success"] = True
    except (json.JSONDecodeError, TypeError) as exc:
        debug["openai_status"] = "openai_parse_failed"
        debug["reject_reason"] = f"json_parse_failed: {exc}"
        return empty_odds_row(row, "missing"), debug

    urls = source_urls_text(parsed.get("source_urls", []))
    source_count = len([url for url in urls.split(";") if url.strip()])
    home_odds = decimal_from_value(parsed.get("home_odds"))
    draw_odds = decimal_from_value(parsed.get("draw_odds"))
    away_odds = decimal_from_value(parsed.get("away_odds"))
    debug.update(
        {
            "openai_status": "ok",
            "home_odds": home_odds or "",
            "draw_odds": draw_odds or "",
            "away_odds": away_odds or "",
            "source_urls_count": source_count,
        }
    )
    accepted = bool(source_count and home_odds and draw_odds and away_odds)
    if not accepted:
        debug["reject_reason"] = "missing_source_urls" if not source_count else "incomplete_odds"
        return empty_odds_row(row, "missing", parsed.get("fetched_at") or utc_now()), debug

    try:
        confidence = float(parsed.get("confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0
    odds_row = {
        "date": row.date,
        "group": getattr(row, "group", ""),
        "home_team": row.home_team,
        "away_team": row.away_team,
        "home_odds": home_odds,
        "draw_odds": draw_odds,
        "away_odds": away_odds,
        "odds_status": "openai",
        "odds_source": str(parsed.get("odds_source", "OpenAI web_search")).strip() or "OpenAI web_search",
        "source_urls": urls,
        "confidence": max(0.0, min(1.0, confidence)),
        "fetched_at": parsed.get("fetched_at") or utc_now(),
    }
    debug["accepted"] = True
    return odds_row, debug


def build_market(odds: pd.DataFrame) -> pd.DataFrame:
    complete = odds[odds["odds_status"].eq("openai")].copy()
    if complete.empty:
        return pd.DataFrame(columns=MARKET_COLUMNS)
    raw_h = 1 / complete["home_odds"].astype(float)
    raw_d = 1 / complete["draw_odds"].astype(float)
    raw_a = 1 / complete["away_odds"].astype(float)
    overround = raw_h + raw_d + raw_a
    market = complete[KEY_COLUMNS].copy()
    market["market_H"] = raw_h / overround
    market["market_D"] = raw_d / overround
    market["market_A"] = raw_a / overround
    return market[MARKET_COLUMNS]


def merge_existing(path: Path, incoming: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    if path.exists():
        try:
            existing = pd.read_csv(path, encoding="utf-8")
        except pd.errors.EmptyDataError:
            existing = pd.DataFrame(columns=columns)
    else:
        existing = pd.DataFrame(columns=columns)
    for column in columns:
        if column not in existing.columns:
            existing[column] = pd.NA
    combined = pd.concat([existing[columns], incoming[columns]], ignore_index=True)
    combined["date"] = pd.to_datetime(combined["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    return combined.drop_duplicates(["date", "home_team", "away_team"], keep="last").sort_values(["date", "home_team", "away_team"]).reset_index(drop=True)


def merge_fill_missing(path: Path, incoming: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    if path.exists():
        try:
            existing = pd.read_csv(path, encoding="utf-8")
        except pd.errors.EmptyDataError:
            existing = pd.DataFrame(columns=columns)
    else:
        existing = pd.DataFrame(columns=columns)
    for column in columns:
        if column not in existing.columns:
            existing[column] = pd.NA
        if column not in incoming.columns:
            incoming[column] = pd.NA
    combined = pd.concat([existing[columns], incoming[columns]], ignore_index=True)
    combined["date"] = pd.to_datetime(combined["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    return combined.drop_duplicates(["date", "home_team", "away_team"], keep="first").sort_values(["date", "home_team", "away_team"]).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", required=True)
    parser.add_argument("--match-key")
    args = parser.parse_args()

    match_date = normalize_date(args.date)
    fixtures = load_fixtures(match_date, args.match_key)
    odds_rows = []
    debug_rows = []
    for fixture in fixtures.itertuples(index=False):
        if has_existing_complete_odds(fixture):
            debug_rows.append(
                {
                    "match_key": f"{fixture.date}|{fixture.home_team}|{fixture.away_team}",
                    "query": "",
                    "openai_status": "skipped_existing_odds",
                    "json_parse_success": False,
                    "home_odds": "",
                    "draw_odds": "",
                    "away_odds": "",
                    "source_urls_count": 0,
                    "accepted": False,
                    "reject_reason": "existing_complete_odds",
                }
            )
            continue
        result = run_web_search(prompt_for_match(fixture))
        odds_row, debug_row = odds_row_from_openai(fixture, result)
        odds_rows.append(odds_row)
        debug_rows.append(debug_row)

    odds = pd.DataFrame(odds_rows, columns=ODDS_COLUMNS) if odds_rows else pd.DataFrame(columns=ODDS_COLUMNS)
    for column in ["home_odds", "draw_odds", "away_odds", "confidence"]:
        odds[column] = pd.to_numeric(odds[column], errors="coerce")
    market = build_market(odds)
    consensus = odds[odds["odds_status"].eq("openai")][KEY_COLUMNS + ["home_odds", "draw_odds", "away_odds"]].copy()

    ODDS_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    DEBUG_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    merge_existing(ODDS_OUTPUT, odds, ODDS_COLUMNS).to_csv(ODDS_OUTPUT, index=False, encoding="utf-8")
    merge_existing(MARKET_OUTPUT, market, MARKET_COLUMNS).to_csv(MARKET_OUTPUT, index=False, encoding="utf-8")
    merge_fill_missing(PRIMARY_CONSENSUS_OUTPUT, consensus, KEY_COLUMNS + ["home_odds", "draw_odds", "away_odds"]).to_csv(PRIMARY_CONSENSUS_OUTPUT, index=False, encoding="utf-8")
    merge_fill_missing(PRIMARY_MARKET_OUTPUT, market, MARKET_COLUMNS).to_csv(PRIMARY_MARKET_OUTPUT, index=False, encoding="utf-8")
    pd.DataFrame(debug_rows).to_csv(DEBUG_OUTPUT, index=False, encoding="utf-8")

    accepted = int(sum(bool(row.get("accepted")) for row in debug_rows))
    print(f"OpenAI odds written: {ODDS_OUTPUT}")
    print(f"OpenAI market probabilities written: {MARKET_OUTPUT}")
    print(f"Primary consensus odds filled: {PRIMARY_CONSENSUS_OUTPUT}")
    print(f"Primary market probabilities filled: {PRIMARY_MARKET_OUTPUT}")
    print(f"Debug report written: {DEBUG_OUTPUT}")
    print(f"matches={len(fixtures)} accepted={accepted}")


if __name__ == "__main__":
    main()
