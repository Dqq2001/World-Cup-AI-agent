import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.paths import PROCESSED_DATA_DIR


OUTPUT_PATH = PROCESSED_DATA_DIR / "worldcup_openai_intel.csv"
DAILY_INTEL_PATH = PROJECT_ROOT / "reports" / "worldcup_daily_intel.csv"
CACHE_PATH = PROJECT_ROOT / "data" / "cache" / "worldcup_openai_intel_cache.json"
MISSING_REPORT_PATH = PROJECT_ROOT / "reports" / "openai_intel_missing_report.csv"
DEBUG_PATH = PROJECT_ROOT / "reports" / "openai_intel_debug.csv"
OVERWRITE_DEBUG_PATH = PROJECT_ROOT / "reports" / "intel_refresh_overwrite_debug.csv"
CACHE_TTL_HOURS = 6
DEFAULT_MODEL = "gpt-4.1-mini"
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
    "source_type",
    "source_status",
    "request_success",
    "status_code",
    "error_type",
    "error_message",
    "web_search_success",
    "final_text_success",
    "intel_has_content",
    "source_url",
    "source_urls",
    "intel_text",
    "confidence",
    "fetched_at",
]
JSON_INTEL_FIELDS = [
    "injuries_home",
    "injuries_away",
    "suspensions_home",
    "suspensions_away",
    "expected_lineup_home",
    "expected_lineup_away",
    "coach_comments_home",
    "coach_comments_away",
]
MISSING_REPORT_COLUMNS = ["base_url", "model", "status_code", "error_type", "error_message"]
INTEL_FIELDS = [
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


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_env() -> None:
    env_path = PROJECT_ROOT / ".env"
    if not env_path.exists():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def mask_key(value: str) -> str:
    if not value:
        return "<missing>"
    if len(value) < 10:
        return "<present>"
    return f"{value[:6]}...{value[-4:]}"


def base_url() -> str:
    return os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1").strip().rstrip("/")


def responses_url(base_url_value: str) -> str:
    base = base_url_value.strip().rstrip("/")
    if base.endswith("/v1"):
        return f"{base}/responses"
    return f"{base}/v1/responses"


def write_missing_report(base_url_value: str, model: str, status_code, error_type: str, error_message: str) -> None:
    MISSING_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "base_url": base_url_value,
                "model": model,
                "status_code": status_code,
                "error_type": error_type,
                "error_message": error_message,
            }
        ],
        columns=MISSING_REPORT_COLUMNS,
    ).to_csv(MISSING_REPORT_PATH, index=False, encoding="utf-8")


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
Search the public web for pre-match football intelligence for this World Cup 2026 match:
date: {date}
home_team: {home_team}
away_team: {away_team}

Return ONLY strict valid JSON with exactly these keys:
{{
  "injuries_home": "...",
  "injuries_away": "...",
  "suspensions_home": "...",
  "suspensions_away": "...",
  "expected_lineup_home": "...",
  "expected_lineup_away": "...",
  "coach_comments_home": "...",
  "coach_comments_away": "...",
  "source_urls": ["..."],
  "confidence": 0.0
}}

Rules:
- source_urls must be a list of public URLs used as evidence.
- If a fact is not clearly supported by the sources, use "unknown".
- Do not guess player names, injuries, suspensions, lineups, or quotes.
- confidence is a number from 0 to 1 based on source quality and specificity.
- Do not include markdown, prose, comments, or extra keys outside the JSON object.
""".strip()


def classify_error(status_code, text: str) -> str:
    lowered = (text or "").lower()
    if status_code == 403 and "1010" in lowered:
        return "provider_blocked"
    if status_code in {401, 403} and any(term in lowered for term in ["invalid api key", "incorrect api key", "unauthorized", "authentication"]):
        return "auth_failed"
    if status_code in {400, 404} and any(term in lowered for term in ["model", "not found", "does not exist"]):
        return "model_not_found"
    if status_code == 404 or any(term in lowered for term in ["responses api", "unsupported", "unknown url", "not supported"]):
        return "responses_api_not_supported"
    return "unknown_error"


def extract_output_text(payload: dict) -> str:
    if isinstance(payload.get("output_text"), str):
        return payload["output_text"].strip()
    parts = []
    for item in payload.get("output", []) or []:
        if item.get("type") != "message":
            continue
        for content in item.get("content", []) or []:
            if isinstance(content, dict) and content.get("text"):
                parts.append(str(content["text"]))
    return "".join(parts).strip()


def response_status_flags(payload: dict) -> tuple[bool, bool, bool, str]:
    output = payload.get("output", []) or []
    web_search_items = [item for item in output if item.get("type") == "web_search_call"]
    message_items = [item for item in output if item.get("type") == "message"]
    web_search_success = any(item.get("status") == "completed" for item in web_search_items)
    web_search_status = ";".join(str(item.get("status", "unknown")) for item in web_search_items) or "missing"
    final_text = extract_output_text(payload)
    return web_search_success, bool(message_items), bool(final_text), web_search_status


def parse_json_text(text: str) -> dict:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`").removeprefix("json").strip()
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start >= 0 and end >= start:
        cleaned = cleaned[start : end + 1]
    return json.loads(cleaned)


def empty_row(date: str, home_team: str, away_team: str, source_status: str, **extra) -> dict:
    row = {
        "date": date,
        "home_team": home_team,
        "away_team": away_team,
        "source_type": "openai",
        "source_status": source_status,
        "request_success": False,
        "status_code": "",
        "error_type": "",
        "error_message": "",
        "web_search_success": False,
        "final_text_success": False,
        "intel_has_content": False,
        "source_url": "unknown",
        "source_urls": "unknown",
        "intel_text": "",
        "confidence": 0.0,
        "fetched_at": utc_now(),
    }
    for field in INTEL_FIELDS:
        row[field] = "unknown"
    row.update(extra)
    return row


def normalize_result(date: str, home_team: str, away_team: str, data: dict, intel_text: str, status_meta: dict) -> dict:
    row = empty_row(date, home_team, away_team, "ok")
    row.update(status_meta)
    row["intel_text"] = intel_text
    source_urls = data.get("source_urls", [])
    if isinstance(source_urls, list):
        urls = [str(item).strip() for item in source_urls if str(item).strip()]
        row["source_urls"] = "; ".join(urls) if urls else "unknown"
    else:
        row["source_urls"] = str(source_urls).strip() or "unknown"
    row["source_url"] = row["source_urls"]
    for field in JSON_INTEL_FIELDS:
        value = data.get(field, "unknown")
        value = str(value).strip() if value is not None else "unknown"
        row[field] = value or "unknown"
    row["team_news_home"] = "unknown"
    row["team_news_away"] = "unknown"
    try:
        row["confidence"] = float(data.get("confidence", 0.0))
    except (TypeError, ValueError):
        row["confidence"] = 0.0
    has_content = any(str(row[field]).strip().lower() not in {"", "unknown", "nan", "none", "<na>"} for field in JSON_INTEL_FIELDS)
    if row["source_urls"] == "unknown":
        row["source_status"] = "openai_failed_or_no_sources"
        row["intel_has_content"] = False
        row["confidence"] = 0.0
        for field in INTEL_FIELDS:
            row[field] = "unknown"
    else:
        row["intel_has_content"] = bool(has_content)
    return row


def call_openai(date: str, home_team: str, away_team: str) -> dict:
    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        write_missing_report(base_url(), os.environ.get("OPENAI_INTEL_MODEL", DEFAULT_MODEL).strip() or DEFAULT_MODEL, "", "missing_env", "OpenAI/Pixvyn API key missing")
        return empty_row(date, home_team, away_team, "openai_failed_or_no_sources", error_type="missing_env", error_message="OpenAI/Pixvyn API key missing")
    base_url_value = base_url()
    model = os.environ.get("OPENAI_INTEL_MODEL", DEFAULT_MODEL).strip() or DEFAULT_MODEL
    endpoint = responses_url(base_url_value)
    payload = {
        "model": model,
        "input": prompt_for_match(date, home_team, away_team),
        "tools": [{"type": "web_search"}],
        "store": False,
    }
    try:
        response = requests.post(
            endpoint,
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json=payload,
            timeout=300,
        )
    except requests.RequestException as exc:
        message = f"{type(exc).__name__}: {exc}"
        write_missing_report(base_url_value, model, "", "request_failed", message)
        return empty_row(date, home_team, away_team, "openai_failed_or_no_sources", error_type="request_failed", error_message=message)

    if not response.ok:
        error_type = classify_error(response.status_code, response.text)
        write_missing_report(base_url_value, model, response.status_code, error_type, response.text[:1000])
        return empty_row(
            date,
            home_team,
            away_team,
            "openai_failed_or_no_sources",
            request_success=False,
            status_code=response.status_code,
            error_type=error_type,
            error_message=response.text[:1000],
        )

    print(f"OpenAI/Pixvyn status_code: {response.status_code}")
    status_meta = {
        "request_success": True,
        "status_code": response.status_code,
        "error_type": "",
        "error_message": "",
    }
    try:
        payload = response.json()
    except ValueError:
        return empty_row(date, home_team, away_team, "openai_failed_or_no_sources", **status_meta, error_type="invalid_json_response", error_message=response.text[:1000])

    print(f"response.status: {payload.get('status', '')}")
    web_search_success, has_final_message, final_text_success, web_search_status = response_status_flags(payload)
    status_meta.update(
        {
            "web_search_success": web_search_success,
            "final_text_success": final_text_success,
            "error_message": f"web_search_status={web_search_status}",
        }
    )
    intel_text = extract_output_text(payload)
    if payload.get("status") == "completed" and web_search_success and not final_text_success:
        return empty_row(date, home_team, away_team, "openai_failed_or_no_sources", **status_meta, error_type="missing_final_text", intel_text="")
    if not final_text_success:
        return empty_row(date, home_team, away_team, "openai_failed_or_no_sources", **status_meta, error_type="missing_final_text", error_message=f"web_search_status={web_search_status}")
    try:
        data = parse_json_text(intel_text)
    except (json.JSONDecodeError, TypeError) as exc:
        return empty_row(date, home_team, away_team, "openai_parse_failed", **status_meta, error_type="openai_parse_failed", error_message=str(exc), intel_text=intel_text)
    return normalize_result(date, home_team, away_team, data, intel_text, status_meta)


def write_debug(row: dict) -> None:
    source_urls = str(row.get("source_urls", "unknown")).strip()
    urls_count = 0 if source_urls.lower() in {"", "unknown", "nan", "none", "<na>"} else len([url for url in source_urls.split(";") if url.strip()])
    debug_row = {
        "match_key": f"{row.get('date')}|{row.get('home_team')}|{row.get('away_team')}",
        "openai_status": row.get("source_status", "unknown"),
        "source_urls_count": urls_count,
        "json_parse_success": row.get("source_status") == "ok",
        "intel_has_content": row.get("intel_has_content", False),
        "included_in_daily_brief": False,
        "exclude_reason": "",
    }
    if DEBUG_PATH.exists():
        try:
            existing = pd.read_csv(DEBUG_PATH, encoding="utf-8")
        except pd.errors.EmptyDataError:
            existing = pd.DataFrame()
    else:
        existing = pd.DataFrame()
    output = pd.concat([existing, pd.DataFrame([debug_row])], ignore_index=True)
    output = output.drop_duplicates("match_key", keep="last")
    DEBUG_PATH.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(DEBUG_PATH, index=False, encoding="utf-8")


def urls_count(row: dict | pd.Series) -> int:
    source_urls = str(row.get("source_urls", "unknown")).strip()
    if source_urls.lower() in {"", "unknown", "nan", "none", "<na>"}:
        return 0
    return len([url for url in source_urls.split(";") if url.strip()])


def is_valid_intel(row: dict | pd.Series) -> bool:
    has_content = str(row.get("intel_has_content", False)).strip().lower() in {"true", "1", "yes"}
    return has_content and urls_count(row) > 0


def write_overwrite_debug(row: dict, old_row: pd.Series | None, action: str, reason: str) -> None:
    old_has_content = False if old_row is None else is_valid_intel(old_row)
    old_source_type = "" if old_row is None else str(old_row.get("source_type", "unknown"))
    debug_row = {
        "match_key": f"{row.get('date')}|{row.get('home_team')}|{row.get('away_team')}",
        "old_source_type": old_source_type,
        "old_intel_has_content": old_has_content,
        "new_openai_status": row.get("source_status", "unknown"),
        "new_source_urls_count": urls_count(row),
        "action": action,
        "reason": reason,
    }
    if OVERWRITE_DEBUG_PATH.exists():
        try:
            existing = pd.read_csv(OVERWRITE_DEBUG_PATH, encoding="utf-8")
        except pd.errors.EmptyDataError:
            existing = pd.DataFrame()
    else:
        existing = pd.DataFrame()
    output = pd.concat([existing, pd.DataFrame([debug_row])], ignore_index=True)
    output = output.drop_duplicates("match_key", keep="last")
    OVERWRITE_DEBUG_PATH.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(OVERWRITE_DEBUG_PATH, index=False, encoding="utf-8")


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
    existing = existing[OUTPUT_COLUMNS].copy()
    if DAILY_INTEL_PATH.exists():
        try:
            daily = pd.read_csv(DAILY_INTEL_PATH, encoding="utf-8")
        except pd.errors.EmptyDataError:
            daily = pd.DataFrame(columns=OUTPUT_COLUMNS)
        for column in OUTPUT_COLUMNS:
            if column not in daily.columns:
                daily[column] = pd.NA
        existing = pd.concat([existing, daily[OUTPUT_COLUMNS]], ignore_index=True)
    existing["date"] = pd.to_datetime(existing["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    match_mask = (
        (existing["date"].astype(str) == str(row["date"]))
        & (existing["home_team"].astype(str) == str(row["home_team"]))
        & (existing["away_team"].astype(str) == str(row["away_team"]))
    )
    old_match = existing.loc[match_mask]
    old_valid = old_match[old_match.apply(is_valid_intel, axis=1)]
    old_row = old_match.iloc[-1] if not old_match.empty else None

    if is_valid_intel(row):
        output_row = row
        action = "replace_with_new"
        reason = "new_openai_has_sources_and_content"
    elif not old_valid.empty:
        output_row = old_valid.iloc[-1].to_dict()
        output_row["source_status"] = "cached_previous_openai"
        action = "keep_old"
        reason = "new_openai_failed_or_no_sources"
    else:
        output_row = row
        action = "write_unknown"
        reason = "no_previous_valid_intel"

    write_overwrite_debug(row, old_row, action, reason)
    existing = existing.loc[~match_mask]
    combined = pd.concat([existing, pd.DataFrame([output_row], columns=OUTPUT_COLUMNS)], ignore_index=True)
    combined["date"] = pd.to_datetime(combined["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    return combined.sort_values(["date", "home_team", "away_team"]).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", required=True)
    parser.add_argument("--home-team", required=True)
    parser.add_argument("--away-team", required=True)
    parser.add_argument("--force-refresh", action="store_true")
    args = parser.parse_args()

    load_env()
    print(f"ROOT={PROJECT_ROOT}")
    print(f"OPENAI_API_KEY={mask_key(os.environ.get('OPENAI_API_KEY', '').strip())}")
    print(f"OPENAI_BASE_URL={base_url()}")
    print(f"OPENAI_FINAL_URL={responses_url(base_url())}")
    print(f"OPENAI_INTEL_MODEL={os.environ.get('OPENAI_INTEL_MODEL', DEFAULT_MODEL).strip() or DEFAULT_MODEL}")
    match_date = pd.to_datetime(args.date, errors="raise").strftime("%Y-%m-%d")
    home_team = args.home_team.strip()
    away_team = args.away_team.strip()
    key = cache_key(match_date, home_team, away_team)
    cache = load_cache()

    if cache_is_fresh(cache.get(key)) and not args.force_refresh:
        row = cache[key]["row"]
        row["fetched_at"] = cache[key]["fetched_at"]
        print("Using cached OpenAI intel.")
    else:
        row = call_openai(match_date, home_team, away_team)
        cache[key] = {"fetched_at": row["fetched_at"], "row": row}
        save_cache(cache)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    merge_existing(row).to_csv(OUTPUT_PATH, index=False, encoding="utf-8")
    write_debug(row)
    print(f"OpenAI intel written: {OUTPUT_PATH}")
    print(f"match: {match_date} {home_team} vs {away_team}")
    print(f"status_code: {row.get('status_code', '')}")
    print(f"source_status: {row.get('source_status', 'unknown')}")
    print(f"intel_has_content: {row.get('intel_has_content', False)}")
    print(f"response.status: {row.get('source_status', 'unknown')}")
    print(f"source_urls_count: {urls_count(row)}")
    print(f"source_urls: {row.get('source_urls', 'unknown')}")


if __name__ == "__main__":
    main()
