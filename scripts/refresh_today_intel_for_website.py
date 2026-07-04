from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path("C:/Users/Administrator/Desktop/worldcup-ai-agent")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.paths import PROCESSED_DATA_DIR


FIXTURES_PATH = PROCESSED_DATA_DIR / "worldcup_fixtures.csv"
RESOLVED_FIXTURES_PATH = PROCESSED_DATA_DIR / "worldcup_fixtures_resolved.csv"
OPENAI_INTEL_PATH = PROCESSED_DATA_DIR / "worldcup_openai_intel.csv"
DAILY_INTEL_CSV = PROJECT_ROOT / "reports" / "worldcup_daily_intel.csv"
WEBSITE_INTEL_JSON = PROJECT_ROOT / "website" / "public" / "data" / "worldcup_daily_intel.json"
REFRESH_STATUS_PATH = PROJECT_ROOT / "data" / "cache" / "website_refresh_status.json"
DEBUG_PATH = PROJECT_ROOT / "reports" / "website_refresh_debug.csv"

OPENAI_COLUMNS = [
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
    "source_name",
    "source_status",
    "intel_has_content",
    "intel_confidence_level",
    "confidence",
    "fetched_at",
    "source_url",
    "source_urls",
]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def latest_fixtures_path() -> Path:
    if RESOLVED_FIXTURES_PATH.exists():
        return RESOLVED_FIXTURES_PATH
    return FIXTURES_PATH


def today_text() -> str:
    return pd.Timestamp.today().strftime("%Y-%m-%d")


def load_today_fixtures() -> pd.DataFrame:
    path = latest_fixtures_path()
    if not path.exists():
        raise FileNotFoundError(f"Missing World Cup fixtures: {path}")
    fixtures = pd.read_csv(path, encoding="utf-8")
    for column in ["date", "home_team", "away_team"]:
        if column not in fixtures.columns:
            raise ValueError(f"fixtures missing required column: {column}")
    fixtures = fixtures.copy()
    fixtures["date"] = pd.to_datetime(fixtures["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    fixtures["home_team"] = fixtures["home_team"].fillna("").astype(str).str.strip()
    fixtures["away_team"] = fixtures["away_team"].fillna("").astype(str).str.strip()
    today = today_text()
    selected = fixtures[(fixtures["date"] == today) & (fixtures["home_team"] != "TBD") & (fixtures["away_team"] != "TBD")]
    return selected.reset_index(drop=True)


def match_key(row) -> str:
    return f"{row.date}|{row.home_team}|{row.away_team}"


def openai_row_for_fixture(fixture) -> dict:
    if not OPENAI_INTEL_PATH.exists():
        return {}
    try:
        data = pd.read_csv(OPENAI_INTEL_PATH, encoding="utf-8")
    except pd.errors.EmptyDataError:
        return {}
    if data.empty:
        return {}
    data["date"] = pd.to_datetime(data["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    row = data[
        (data["date"] == fixture.date)
        & (data["home_team"].astype(str) == str(fixture.home_team))
        & (data["away_team"].astype(str) == str(fixture.away_team))
    ]
    return row.iloc[-1].to_dict() if not row.empty else {}


def run_command(command: list[str]) -> tuple[bool, str]:
    result = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=300,
        check=False,
    )
    output = f"{result.stdout}\n{result.stderr}".strip()
    return result.returncode == 0, output


def export_daily_intel_json() -> int:
    if not DAILY_INTEL_CSV.exists():
        data = pd.DataFrame()
    else:
        try:
            data = pd.read_csv(DAILY_INTEL_CSV, encoding="utf-8")
        except pd.errors.EmptyDataError:
            data = pd.DataFrame()
    WEBSITE_INTEL_JSON.parent.mkdir(parents=True, exist_ok=True)
    payload = data.fillna("").to_dict(orient="records")
    WEBSITE_INTEL_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return len(payload)


def write_status(status: str, matches_count: int, openai_success_count: int, openai_failed_count: int, fallback_count: int, error_message: str = "") -> None:
    REFRESH_STATUS_PATH.parent.mkdir(parents=True, exist_ok=True)
    REFRESH_STATUS_PATH.write_text(
        json.dumps(
            {
                "last_refresh_at": utc_now(),
                "status": status,
                "matches_count": matches_count,
                "openai_success_count": openai_success_count,
                "openai_failed_count": openai_failed_count,
                "fallback_count": fallback_count,
                "error_message": error_message,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def write_debug(rows: list[dict]) -> None:
    DEBUG_PATH.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "match_key",
        "home_team",
        "away_team",
        "provider_used",
        "source_status",
        "intel_has_content",
        "source_urls_count",
        "error_message",
    ]
    pd.DataFrame(rows, columns=columns).to_csv(DEBUG_PATH, index=False, encoding="utf-8")


def main() -> int:
    fixtures = load_today_fixtures()
    debug_rows: list[dict] = []
    openai_success_count = 0
    openai_failed_count = 0

    for fixture in fixtures.itertuples(index=False):
        key = match_key(fixture)
        ok, output = run_command(
            [
                sys.executable,
                "scripts/search_intel_with_openai.py",
                "--date",
                str(fixture.date),
                "--home-team",
                str(fixture.home_team),
                "--away-team",
                str(fixture.away_team),
                "--force-refresh",
            ]
        )
        row = openai_row_for_fixture(fixture)
        source_status = str(row.get("source_status", "openai_failed_or_no_sources"))
        source_urls = str(row.get("source_urls", "unknown"))
        source_urls_count = 0 if source_urls.lower() in {"", "unknown", "nan", "none", "<na>"} else len([url for url in source_urls.split(";") if url.strip()])
        has_content = str(row.get("intel_has_content", "False")).lower() in {"true", "1", "yes"}
        if ok and source_status in {"ok", "cached_previous_openai"} and has_content and source_urls_count > 0:
            openai_success_count += 1
            debug_rows.append(
                {
                    "match_key": key,
                    "home_team": fixture.home_team,
                    "away_team": fixture.away_team,
                    "provider_used": "openai",
                    "source_status": "ok",
                    "intel_has_content": True,
                    "source_urls_count": source_urls_count,
                    "error_message": "",
                }
            )
        else:
            openai_failed_count += 1
            debug_rows.append(
                {
                    "match_key": key,
                    "home_team": fixture.home_team,
                    "away_team": fixture.away_team,
                    "provider_used": "unknown",
                    "source_status": source_status,
                    "intel_has_content": False,
                    "source_urls_count": source_urls_count,
                    "error_message": output[-500:],
                }
            )

    today = today_text()
    daily_ok, daily_output = run_command(
        [sys.executable, "scripts/run_daily_worldcup_intel.py", "--openai-intel", "--as-of-date", today, "--days-ahead", "0"]
    )
    exported_count = export_daily_intel_json()

    fallback_count = openai_failed_count
    status = "success" if daily_ok else "partial_failure"
    error_message = ""
    if not daily_ok:
        error_message += f"Daily intel failed: {daily_output[-800:]}"
    write_debug(debug_rows)
    write_status(status, len(fixtures), openai_success_count, openai_failed_count, fallback_count, error_message.strip())

    print(f"matches_count={len(fixtures)}")
    print(f"openai_success_count={openai_success_count}")
    print(f"openai_failed_count={openai_failed_count}")
    print(f"fallback_count={fallback_count}")
    print(f"daily_intel_rows_exported={exported_count}")
    print(f"daily_intel_csv={DAILY_INTEL_CSV}")
    print(f"website_json={WEBSITE_INTEL_JSON}")
    print(f"debug_report={DEBUG_PATH}")
    print(f"status_file={REFRESH_STATUS_PATH}")
    if error_message:
        print(error_message.strip())
    return 0 if daily_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
