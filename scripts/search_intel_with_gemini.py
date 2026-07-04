import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.paths import PROCESSED_DATA_DIR


OUTPUT_PATH = PROCESSED_DATA_DIR / "worldcup_gemini_intel.csv"
CACHE_PATH = PROJECT_ROOT / "data" / "cache" / "worldcup_gemini_intel_cache.json"
CACHE_TTL_HOURS = 6
DEFAULT_MODEL = "gemini-1.5-flash"
FALLBACK_MODELS = ["gemini-3.0-flash", "gemini-2.5-flash"]
OUTPUT_COLUMNS = [
    "date",
    "home_team",
    "away_team",
    "injuries_home",
    "injuries_away",
    "suspensions_home",
    "suspensions_away",
    "expected_lineup_home",
    "expected_lineup_away",
    "coach_comments_home",
    "coach_comments_away",
    "source_urls",
    "confidence",
    "fetched_at",
]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_env() -> None:
    env_path = PROJECT_ROOT / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


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


def cache_key(date: str, home_team: str, away_team: str) -> str:
    return f"{date}|{home_team}|{away_team}"


def cache_is_fresh(entry: dict | None) -> bool:
    if not entry:
        return False
    fetched_at = pd.to_datetime(entry.get("fetched_at"), utc=True, errors="coerce")
    if pd.isna(fetched_at):
        return False
    return (datetime.now(timezone.utc) - fetched_at.to_pydatetime()).total_seconds() < CACHE_TTL_HOURS * 3600


def prompt_for_match(date: str, home_team: str, away_team: str) -> str:
    return f"""
Use web search grounding to find pre-match football intelligence for this World Cup 2026 match:
date: {date}
home_team: {home_team}
away_team: {away_team}

Search for:
- {home_team} injury news World Cup 2026
- {away_team} injury news World Cup 2026
- {home_team} predicted lineup
- {away_team} predicted lineup
- {home_team} suspension news
- {away_team} suspension news

Return ONLY valid JSON with these keys:
injuries_home, injuries_away, suspensions_home, suspensions_away,
expected_lineup_home, expected_lineup_away, coach_comments_home, coach_comments_away,
source_urls, confidence.

Rules:
- source_urls must be a list of public URLs used as evidence.
- If a fact is not clearly supported by the sources, use "unknown".
- Do not guess player names, injuries, suspensions, lineups, or quotes.
- confidence is a number from 0 to 1 based on source quality and specificity.
""".strip()


def gemini_payload(date: str, home_team: str, away_team: str, model: str) -> dict:
    # Gemini 1.5 models use google_search_retrieval; newer models may use google_search.
    tool = {"google_search_retrieval": {}} if "1.5" in model else {"google_search": {}}
    return {
        "contents": [{"role": "user", "parts": [{"text": prompt_for_match(date, home_team, away_team)}]}],
        "tools": [tool],
        "generationConfig": {
            "temperature": 0.1,
            "response_mime_type": "application/json",
        },
    }


def call_gemini_model(date: str, home_team: str, away_team: str, model: str, api_key: str) -> dict:
    query = urlencode({"key": api_key})
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?{query}"
    request = Request(
        url,
        data=json.dumps(gemini_payload(date, home_team, away_team, model)).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(request, timeout=120) as response:
        payload = json.loads(response.read().decode("utf-8"))
        payload["_gemini_model_used"] = model
        return payload


def is_model_not_found(exc: HTTPError, body: str) -> bool:
    lowered = body.lower()
    return exc.code == 404 or "not found" in lowered or "not supported" in lowered


def call_gemini(date: str, home_team: str, away_team: str) -> dict:
    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("Missing GEMINI_API_KEY. Add it to .env or environment variables.")
    preferred = os.environ.get("GEMINI_INTEL_MODEL", DEFAULT_MODEL).strip() or DEFAULT_MODEL
    models = [preferred, *[model for model in FALLBACK_MODELS if model != preferred]]
    errors = []
    for model in models:
        try:
            return call_gemini_model(date, home_team, away_team, model, api_key)
        except HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            errors.append(f"{model}: HTTP {exc.code} {body[:300]}")
            if is_model_not_found(exc, body):
                continue
            if "google_search" in body or "google_search_retrieval" in body or "tool" in body.lower():
                raise RuntimeError(f"Gemini web search grounding unavailable for {model}: HTTP {exc.code} {body[:500]}")
            raise RuntimeError(f"Gemini API failed for {model}: HTTP {exc.code} {body[:500]}")
    raise RuntimeError("Gemini model not found or unavailable after fallbacks: " + " | ".join(errors))


def response_text(payload: dict) -> str:
    candidates = payload.get("candidates", [])
    if not candidates:
        return ""
    parts = candidates[0].get("content", {}).get("parts", [])
    return "\n".join(str(part.get("text", "")) for part in parts if part.get("text")).strip()


def grounding_urls(payload: dict) -> list[str]:
    urls = []
    for candidate in payload.get("candidates", []):
        metadata = candidate.get("groundingMetadata", {}) or candidate.get("grounding_metadata", {})
        for chunk in metadata.get("groundingChunks", []) or metadata.get("grounding_chunks", []):
            web = chunk.get("web", {})
            uri = web.get("uri")
            if uri:
                urls.append(str(uri))
    return list(dict.fromkeys(urls))


def parse_json_text(text: str) -> dict:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        cleaned = cleaned.removeprefix("json").strip()
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start >= 0 and end >= start:
        cleaned = cleaned[start : end + 1]
    return json.loads(cleaned)


def normalize_result(date: str, home_team: str, away_team: str, data: dict, grounded_urls: list[str]) -> dict:
    row = {"date": date, "home_team": home_team, "away_team": away_team}
    source_urls = data.get("source_urls", [])
    if isinstance(source_urls, str):
        source_urls = [source_urls] if source_urls.strip() and source_urls.strip().lower() != "unknown" else []
    source_urls = list(dict.fromkeys([str(url).strip() for url in [*source_urls, *grounded_urls] if str(url).strip()]))
    for column in OUTPUT_COLUMNS:
        if column in {"date", "home_team", "away_team", "fetched_at"}:
            continue
        if column == "source_urls":
            row[column] = "; ".join(source_urls) if source_urls else "unknown"
        elif column == "confidence":
            try:
                row[column] = float(data.get(column, 0.0))
            except (TypeError, ValueError):
                row[column] = 0.0
        else:
            value = str(data.get(column, "unknown") or "unknown").strip()
            row[column] = value if value else "unknown"
    if row["source_urls"] == "unknown":
        for column in [
            "injuries_home",
            "injuries_away",
            "suspensions_home",
            "suspensions_away",
            "expected_lineup_home",
            "expected_lineup_away",
            "coach_comments_home",
            "coach_comments_away",
        ]:
            row[column] = "unknown"
        row["confidence"] = 0.0
    row["fetched_at"] = utc_now()
    return row


def merge_existing(row: dict) -> pd.DataFrame:
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
    combined = pd.concat([existing[OUTPUT_COLUMNS], pd.DataFrame([row], columns=OUTPUT_COLUMNS)], ignore_index=True)
    combined["date"] = pd.to_datetime(combined["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    return combined.drop_duplicates(["date", "home_team", "away_team"], keep="last").sort_values(["date", "home_team", "away_team"]).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", required=True)
    parser.add_argument("--home-team", required=True)
    parser.add_argument("--away-team", required=True)
    parser.add_argument("--force-refresh", action="store_true")
    args = parser.parse_args()

    load_env()
    match_date = pd.to_datetime(args.date, errors="raise").strftime("%Y-%m-%d")
    home_team = args.home_team.strip()
    away_team = args.away_team.strip()
    key = cache_key(match_date, home_team, away_team)
    cache = load_cache()

    if cache_is_fresh(cache.get(key)) and not args.force_refresh:
        row = cache[key]["row"]
        row["fetched_at"] = cache[key]["fetched_at"]
        print("Using cached Gemini intel.")
    else:
        try:
            payload = call_gemini(match_date, home_team, away_team)
            data = parse_json_text(response_text(payload))
            row = normalize_result(match_date, home_team, away_team, data, grounding_urls(payload))
        except (URLError, TimeoutError, json.JSONDecodeError, RuntimeError) as exc:
            raise SystemExit(f"Gemini intel search failed: {exc}")
        cache[key] = {"fetched_at": row["fetched_at"], "row": row}
        save_cache(cache)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    merge_existing(row).to_csv(OUTPUT_PATH, index=False, encoding="utf-8")
    print(f"Gemini intel written: {OUTPUT_PATH}")
    print(f"match: {match_date} {home_team} vs {away_team}")
    print(f"source_urls: {row.get('source_urls', 'unknown')}")


if __name__ == "__main__":
    main()
