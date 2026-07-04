import argparse
import json
import re
import ssl
import sys
from datetime import date, datetime, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import quote_plus
from urllib.request import Request, urlopen

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agents.structured_football_intel_agent import StructuredFootballIntelAgent
from src.paths import PROCESSED_DATA_DIR


FIXTURES_PATH = PROCESSED_DATA_DIR / "worldcup_fixtures.csv"
RESOLVED_FIXTURES_PATH = PROCESSED_DATA_DIR / "worldcup_fixtures_resolved.csv"
OUTPUT_PATH = PROCESSED_DATA_DIR / "worldcup_structured_intel.csv"
DEBUG_PATH = PROJECT_ROOT / "reports" / "structured_intel_fetch_debug.csv"
USER_AGENT = "worldcup-ai-agent/1.0"


def create_ssl_context() -> ssl.SSLContext:
    try:
        import certifi

        return ssl.create_default_context(cafile=certifi.where())
    except Exception:
        return ssl.create_default_context()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def latest_fixtures_path() -> Path:
    if RESOLVED_FIXTURES_PATH.exists():
        return RESOLVED_FIXTURES_PATH
    return FIXTURES_PATH


def fetch_url(url: str) -> tuple[int, str, str]:
    request = Request(url, headers={"User-Agent": USER_AGENT, "Accept": "application/json,text/html"})
    with urlopen(request, timeout=25, context=create_ssl_context()) as response:
        body = response.read().decode("utf-8", errors="replace")
        content_type = response.headers.get("content-type", "")
        return int(response.status), body, content_type


def load_fixtures(path: Path, match_date: str | None, match_key: str | None, max_matches: int | None) -> pd.DataFrame:
    fixtures = pd.read_csv(path, encoding="utf-8")
    fixtures["date"] = pd.to_datetime(fixtures["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    fixtures["home_team"] = fixtures["home_team"].fillna("").astype(str).str.strip()
    fixtures["away_team"] = fixtures["away_team"].fillna("").astype(str).str.strip()
    if match_key:
        parts = [part.strip() for part in match_key.split("|")]
        if len(parts) != 3:
            raise ValueError('--match-key must use "date|home_team|away_team"')
        match_date, home_team, away_team = parts
        selected = fixtures[
            (fixtures["date"] == pd.to_datetime(match_date, errors="raise").strftime("%Y-%m-%d"))
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


def parse_json_or_embedded(text: str, content_type: str) -> list[object]:
    payloads = []
    if "json" in content_type:
        try:
            payloads.append(json.loads(text))
            return payloads
        except json.JSONDecodeError:
            pass
    for pattern in [
        r'<script[^>]+id="__NEXT_DATA__"[^>]*>(.*?)</script>',
        r'<script[^>]+type="application/ld\+json"[^>]*>(.*?)</script>',
        r'<script[^>]*>\s*window\.__INITIAL_STATE__\s*=\s*(\{.*?\})\s*</script>',
    ]:
        for match in re.finditer(pattern, text, flags=re.DOTALL | re.IGNORECASE):
            try:
                payloads.append(json.loads(match.group(1)))
            except json.JSONDecodeError:
                continue
    return payloads


def iter_values(payload):
    if isinstance(payload, dict):
        for value in payload.values():
            yield from iter_values(value)
    elif isinstance(payload, list):
        for item in payload:
            yield from iter_values(item)
    else:
        yield payload


def text_from_payload(payloads: list[object]) -> str:
    strings = []
    for payload in payloads:
        for value in iter_values(payload):
            if isinstance(value, str) and len(value.strip()) > 2:
                strings.append(value.strip())
    return " | ".join(strings[:200])


def extract_fields(text: str) -> dict:
    lower = text.lower()
    return {
        "injuries": text if any(term in lower for term in ["injury", "injured", "out", "doubtful", "missing players"]) else "unknown",
        "suspensions": text if any(term in lower for term in ["suspended", "suspension", "red card", "yellow cards"]) else "unknown",
        "lineup": text if any(term in lower for term in ["lineup", "starting xi", "formation", "4-3-3", "4-2-3-1"]) else "unknown",
        "comments": text if any(term in lower for term in ["coach", "manager", "press conference", "said"]) else "unknown",
    }


def row_from_text(agent: StructuredFootballIntelAgent, fixture, source_name: str, source_url: str, text: str) -> dict:
    fields = extract_fields(text)
    return agent.normalize_row(
        {
            "date": fixture.date,
            "home_team": fixture.home_team,
            "away_team": fixture.away_team,
            "source_name": source_name,
            "source_url": source_url,
            "injuries_home": fields["injuries"],
            "injuries_away": fields["injuries"],
            "suspensions_home": fields["suspensions"],
            "suspensions_away": fields["suspensions"],
            "expected_lineup_home": fields["lineup"],
            "expected_lineup_away": fields["lineup"],
            "coach_comments_home": fields["comments"],
            "coach_comments_away": fields["comments"],
            "fetched_at": utc_now(),
        }
    )


def source_urls(fixture) -> list[tuple[str, str]]:
    query = quote_plus(f"{fixture.home_team} {fixture.away_team}")
    return [
        ("sofascore", f"https://www.sofascore.com/api/v1/search/all?q={query}"),
        ("fotmob", f"https://www.fotmob.com/api/searchData?term={query}"),
        ("flashscore", f"https://www.flashscore.com/search/?q={query}"),
    ]


def fetch_source(agent: StructuredFootballIntelAgent, fixture, source_name: str, url: str) -> tuple[dict | None, dict]:
    debug = {
        "match_key": f"{fixture.date}|{fixture.home_team}|{fixture.away_team}",
        "source": source_name,
        "http_status": "",
        "success": False,
        "has_content": False,
        "error_message": "",
    }
    try:
        status, body, content_type = fetch_url(url)
        debug["http_status"] = status
    except HTTPError as exc:
        debug["http_status"] = exc.code
        debug["error_message"] = "source_blocked" if exc.code in {403, 429} else str(exc)
        return None, debug
    except (URLError, TimeoutError) as exc:
        debug["error_message"] = str(exc)
        return None, debug

    if "cloudflare" in body.lower() or "access denied" in body.lower():
        debug["error_message"] = "source_blocked"
        return None, debug

    payloads = parse_json_or_embedded(body, content_type)
    text = text_from_payload(payloads) if payloads else body[:5000]
    row = row_from_text(agent, fixture, source_name, url, text)
    debug["success"] = True
    debug["has_content"] = bool(row["intel_has_content"])
    if not row["intel_has_content"]:
        debug["error_message"] = "source reachable but no injury/suspension/lineup content extracted"
    return row, debug


def fetch_structured_intel(fixtures: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    agent = StructuredFootballIntelAgent()
    output_rows = []
    debug_rows = []
    for fixture in fixtures.itertuples(index=False):
        source_rows = []
        for source_name, url in source_urls(fixture):
            row, debug = fetch_source(agent, fixture, source_name, url)
            debug_rows.append(debug)
            if row:
                source_rows.append(row)
            if row and row["intel_has_content"]:
                break
        if source_rows:
            output_rows.append(agent.best_row(source_rows, fixture.date, fixture.home_team, fixture.away_team))
    return pd.DataFrame(output_rows, columns=agent.OUTPUT_COLUMNS), pd.DataFrame(debug_rows)


def merge_existing(output: pd.DataFrame) -> pd.DataFrame:
    if OUTPUT_PATH.exists():
        try:
            existing = pd.read_csv(OUTPUT_PATH, encoding="utf-8")
        except pd.errors.EmptyDataError:
            existing = pd.DataFrame(columns=output.columns)
    else:
        existing = pd.DataFrame(columns=output.columns)
    combined = pd.concat([existing, output], ignore_index=True)
    if "match_key" in combined.columns:
        combined = combined[combined["match_key"].astype(str).str.contains("|", regex=False)]
    if "source_name" in combined.columns:
        combined = combined[combined["source_name"].astype(str).str.lower().ne("unknown")]
    combined = combined.drop_duplicates("match_key", keep="last")
    return combined.sort_values(["date", "home_team", "away_team"]).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixtures", type=Path, default=None)
    parser.add_argument("--date")
    parser.add_argument("--match-key")
    parser.add_argument("--max-matches", type=int, default=5)
    args = parser.parse_args()

    fixtures = load_fixtures(args.fixtures or latest_fixtures_path(), args.date, args.match_key, args.max_matches)
    output, debug = fetch_structured_intel(fixtures)
    output = merge_existing(output)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    DEBUG_PATH.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(OUTPUT_PATH, index=False, encoding="utf-8")
    debug.to_csv(DEBUG_PATH, index=False, encoding="utf-8")
    print(f"Structured football intel written: {OUTPUT_PATH}")
    print(f"Rows: {len(output)}")
    print(f"Debug report: {DEBUG_PATH}")


if __name__ == "__main__":
    main()
