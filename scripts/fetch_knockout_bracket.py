import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.fetch_worldcup_fixtures import KNOCKOUT_SLOTS

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
REPORTS_DIR = PROJECT_ROOT / "reports"
BASE_FIXTURES_PATH = PROCESSED_DIR / "worldcup_fixtures.csv"
RESOLVED_FIXTURES_PATH = PROCESSED_DIR / "worldcup_fixtures_resolved.csv"
DEBUG_PATH = REPORTS_DIR / "knockout_bracket_fetch_debug.csv"
RESULTS_PATH = PROCESSED_DIR / "worldcup_results.csv"
ESPN_SCOREBOARD_URL = (
    "https://site.api.espn.com/apis/site/v2/sports/soccer/fifa.world/"
    "scoreboard?dates={start}-{end}&limit=1000"
)

ROUND_SLUGS = {
    "round-of-32": "Round of 32",
    "round-of-16": "Round of 16",
    "quarterfinal": "Quarter Final",
    "quarterfinals": "Quarter Final",
    "quarter-final": "Quarter Final",
    "quarter-finals": "Quarter Final",
    "semifinal": "Semi Final",
    "semifinals": "Semi Final",
    "semi-final": "Semi Final",
    "semi-finals": "Semi Final",
    "final": "Final",
}

TEAM_ALIASES = {
    "Czechia": "Czech Republic",
    "USA": "United States",
    "U.S.": "United States",
    "United States of America": "United States",
    "South Korea": "Korea Republic",
    "Côte d'Ivoire": "Ivory Coast",
    "Cote d'Ivoire": "Ivory Coast",
    "Curaçao": "Curacao",
    "DR Congo": "Congo DR",
    "Bosnia & Herzegovina": "Bosnia and Herzegovina",
}


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def clean_team_name(value: object) -> str:
    text = re.sub(r"\s+", " ", str(value).replace("\xa0", " ")).strip()
    return TEAM_ALIASES.get(text, text)


def is_actual_knockout_team(value: object) -> bool:
    text = str(value).strip().lower()
    if not text or text == "nan":
        return False
    fallback_tokens = [
        "1st group",
        "2nd group",
        "3rd group",
        "winner",
        "loser",
        "tbd",
        "to be determined",
        "round of",
        "quarterfinal",
        "semifinal",
    ]
    return not any(token in text for token in fallback_tokens)


def slot_frame() -> pd.DataFrame:
    rows = []
    for date, stage, round_name, match_id, home_slot, away_slot in KNOCKOUT_SLOTS:
        rows.append(
            {
                "date": date,
                "group": "",
                "stage": stage,
                "round": round_name,
                "match_id": match_id,
                "home_team": "TBD",
                "away_team": "TBD",
                "home_slot": home_slot,
                "away_slot": away_slot,
                "neutral_venue": True,
                "status": "waiting_for_teams",
                "source": "static_slot_fallback",
                "source_url": "",
                "fetched_at": now_utc(),
            }
        )
    return pd.DataFrame(rows)


def status_from_espn(event: dict) -> str:
    status_type = event.get("status", {}).get("type", {})
    name = str(status_type.get("name", "")).lower()
    state = str(status_type.get("state", "")).lower()
    completed = bool(status_type.get("completed"))
    if completed or "final" in name or state == "post":
        return "completed"
    if "in" in state:
        return "in_progress"
    return "scheduled"


def event_url(event: dict) -> str:
    for link in event.get("links", []):
        href = link.get("href")
        if href:
            return href
    return ""


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
        return clean_team_name((home.get("team") or {}).get("displayName", ""))
    if away.get("winner") is True:
        return clean_team_name((away.get("team") or {}).get("displayName", ""))
    if home_goals > away_goals:
        return clean_team_name((home.get("team") or {}).get("displayName", ""))
    if away_goals > home_goals:
        return clean_team_name((away.get("team") or {}).get("displayName", ""))
    return ""


def parse_espn_event(event: dict, source_url: str) -> dict | None:
    slug = str(event.get("season", {}).get("slug", "")).lower()
    round_name = ROUND_SLUGS.get(slug)
    if not round_name:
        return None

    competition = (event.get("competitions") or [{}])[0]
    competitors = competition.get("competitors", [])
    home = next((item for item in competitors if item.get("homeAway") == "home"), None)
    away = next((item for item in competitors if item.get("homeAway") == "away"), None)
    if not home or not away:
        return None

    home_team = clean_team_name(home.get("team", {}).get("displayName", ""))
    away_team = clean_team_name(away.get("team", {}).get("displayName", ""))
    if not is_actual_knockout_team(home_team) or not is_actual_knockout_team(away_team):
        return None

    event_time = pd.to_datetime(event.get("date"), errors="coerce", utc=True)
    if pd.isna(event_time):
        return None

    parsed = {
        "source_event_id": event.get("id", ""),
        "source_event_date": event_time,
        "date": event_time.strftime("%Y-%m-%d"),
        "stage": round_name,
        "round": round_name,
        "home_team": home_team,
        "away_team": away_team,
        "status": status_from_espn(event),
        "source": "espn",
        "source_url": event_url(event) or source_url,
        "fetched_at": now_utc(),
    }

    if parsed["status"] == "completed":
        try:
            home_goals = int(home.get("score"))
            away_goals = int(away.get("score"))
        except (TypeError, ValueError):
            return parsed
        status_type = (competition.get("status") or event.get("status") or {}).get("type") or {}
        status_text = f"{status_type.get('name', '')} {status_type.get('description', '')}".lower()
        parsed.update(
            {
                "home_goals": home_goals,
                "away_goals": away_goals,
                "aet": "extra" in status_text or "aet" in status_text,
                "penalties_home": read_optional_int(home, ["shootoutScore", "penaltyScore", "penalties"]),
                "penalties_away": read_optional_int(away, ["shootoutScore", "penaltyScore", "penalties"]),
                "winner": winner_from_competitors(home, away, home_goals, away_goals),
            }
        )

    return parsed


def fetch_espn_knockout(start: str, end: str) -> tuple[list[dict], dict]:
    url = ESPN_SCOREBOARD_URL.format(start=start.replace("-", ""), end=end.replace("-", ""))
    debug = {
        "source": "espn",
        "http_status": "",
        "matches_found": 0,
        "actual_team_matches": 0,
        "fallback_slot_matches": 0,
        "error_message": "",
    }
    try:
        request = Request(url, headers={"User-Agent": "worldcup-ai-agent/1.0"})
        with urlopen(request, timeout=30) as response:
            debug["http_status"] = response.status
            payload = json.loads(response.read().decode("utf-8", errors="replace"))
    except HTTPError as exc:
        debug["http_status"] = exc.code
        debug["error_message"] = str(exc)
        return [], debug
    except (URLError, TimeoutError, json.JSONDecodeError) as exc:
        debug["error_message"] = str(exc)
        return [], debug

    events = payload.get("events", [])
    matches = []
    for event in events:
        parsed = parse_espn_event(event, url)
        if parsed:
            matches.append(parsed)

    debug["matches_found"] = len(events)
    debug["actual_team_matches"] = len(matches)
    return matches, debug


def assign_match_ids(matches: list[dict]) -> pd.DataFrame:
    slots = slot_frame()
    if not matches:
        return slots

    events = pd.DataFrame(matches)
    events = events.sort_values(["source_event_date", "home_team", "away_team"]).reset_index(drop=True)
    resolved_rows = []

    for round_name, round_events in events.groupby("round", sort=False):
        round_slots = slots[slots["round"] == round_name].sort_values(["date", "match_id"]).reset_index(drop=True)
        round_events = round_events.sort_values(["source_event_date", "home_team", "away_team"]).reset_index(drop=True)
        for index, event in round_events.iterrows():
            if index >= len(round_slots):
                continue
            slot = round_slots.iloc[index].to_dict()
            for column in [
                "home_team",
                "away_team",
                "status",
                "source",
                "source_url",
                "fetched_at",
                "home_goals",
                "away_goals",
                "aet",
                "penalties_home",
                "penalties_away",
                "winner",
                "source_event_date",
                "source_event_id",
            ]:
                slot[column] = event.get(column, pd.NA)
            slot["date"] = event["date"]
            resolved_rows.append(slot)

    if not resolved_rows:
        return slots

    resolved = pd.DataFrame(resolved_rows)
    resolved_ids = set(resolved["match_id"])
    fallback = slots[~slots["match_id"].isin(resolved_ids)].copy()
    return pd.concat([resolved, fallback], ignore_index=True).sort_values(["date", "match_id"]).reset_index(drop=True)


def save_completed_results(knockout: pd.DataFrame) -> int:
    result_columns = [
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
    if knockout.empty or not {"home_goals", "away_goals"}.issubset(knockout.columns):
        return 0

    completed = knockout[
        knockout["status"].astype(str).str.lower().eq("completed")
        & knockout["home_goals"].notna()
        & knockout["away_goals"].notna()
    ].copy()
    if completed.empty:
        return 0

    completed["group"] = ""
    completed["matched_date_source"] = "espn_knockout_bracket"
    for column in result_columns:
        if column not in completed.columns:
            completed[column] = pd.NA
    completed = completed[result_columns]

    if RESULTS_PATH.exists():
        try:
            existing = pd.read_csv(RESULTS_PATH, encoding="utf-8")
        except pd.errors.EmptyDataError:
            existing = pd.DataFrame(columns=result_columns)
    else:
        existing = pd.DataFrame(columns=result_columns)
    for column in result_columns:
        if column not in existing.columns:
            existing[column] = pd.NA
    existing = existing[result_columns]

    combined = pd.concat([existing, completed], ignore_index=True)
    combined["date"] = pd.to_datetime(combined["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    combined["home_goals"] = pd.to_numeric(combined["home_goals"], errors="coerce")
    combined["away_goals"] = pd.to_numeric(combined["away_goals"], errors="coerce")
    combined = combined.dropna(subset=["date", "home_team", "away_team", "home_goals", "away_goals"])
    combined["_rank"] = combined["status"].astype(str).str.lower().eq("completed").astype(int)
    combined = combined.sort_values(["date", "home_team", "away_team", "_rank", "fetched_at"])
    combined = combined.drop_duplicates(["date", "home_team", "away_team"], keep="last").drop(columns=["_rank"])
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(RESULTS_PATH, index=False, encoding="utf-8")
    return len(completed)


def local_results_knockout() -> tuple[pd.DataFrame, dict]:
    debug = {
        "source": "local_results",
        "http_status": "",
        "matches_found": 0,
        "actual_team_matches": 0,
        "fallback_slot_matches": 0,
        "error_message": "",
    }
    if not RESULTS_PATH.exists():
        debug["error_message"] = f"missing {RESULTS_PATH}"
        return pd.DataFrame(), debug
    try:
        results = pd.read_csv(RESULTS_PATH, encoding="utf-8")
    except pd.errors.EmptyDataError:
        debug["error_message"] = "empty results csv"
        return pd.DataFrame(), debug

    required = ["date", "home_team", "away_team", "status"]
    missing = [column for column in required if column not in results.columns]
    if missing:
        debug["error_message"] = f"results csv missing columns: {missing}"
        return pd.DataFrame(), debug

    results = results.copy()
    results["date"] = pd.to_datetime(results["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    if "group" not in results.columns:
        results["group"] = ""
    knockout = results[
        results["group"].fillna("").astype(str).str.strip().eq("")
        & results["status"].fillna("").astype(str).str.lower().isin(["completed", "final"])
    ].copy()
    knockout = knockout[knockout["home_team"].map(is_actual_knockout_team) & knockout["away_team"].map(is_actual_knockout_team)]
    debug["matches_found"] = len(knockout)

    if knockout.empty:
        return pd.DataFrame(), debug

    slots = slot_frame()
    rows = []
    for match_date, date_results in knockout.sort_values(["date", "home_team", "away_team"]).groupby("date", sort=True):
        date_slots = slots[slots["date"] == match_date].sort_values("match_id").reset_index(drop=True)
        date_results = date_results.reset_index(drop=True)
        for index, result in date_results.iterrows():
            if index >= len(date_slots):
                continue
            slot = date_slots.iloc[index].to_dict()
            slot["home_team"] = clean_team_name(result["home_team"])
            slot["away_team"] = clean_team_name(result["away_team"])
            slot["status"] = "completed"
            slot["source"] = "local_results"
            slot["source_url"] = result.get("source_url", "")
            slot["fetched_at"] = now_utc()
            rows.append(slot)

    output = pd.DataFrame(rows)
    debug["actual_team_matches"] = len(output)
    return output, debug


def merge_with_base_fixtures(knockout: pd.DataFrame) -> pd.DataFrame:
    if BASE_FIXTURES_PATH.exists():
        base = pd.read_csv(BASE_FIXTURES_PATH, encoding="utf-8")
    else:
        base = pd.DataFrame()

    if base.empty:
        return knockout

    if "stage" not in base.columns:
        base["stage"] = ""
    group_rows = base[base["stage"].fillna("").astype(str).str.lower().eq("group stage")].copy()
    for column in knockout.columns:
        if column not in group_rows.columns:
            group_rows[column] = ""
    for column in group_rows.columns:
        if column not in knockout.columns:
            knockout[column] = ""
    combined = pd.concat([group_rows[group_rows.columns], knockout[group_rows.columns]], ignore_index=True)
    return combined.sort_values(["date", "stage", "round", "match_id", "group", "home_team", "away_team"]).reset_index(drop=True)


def actual_knockout_count(data: pd.DataFrame) -> int:
    if data.empty or not {"stage", "home_team", "away_team"}.issubset(data.columns):
        return 0
    knockout = data[data["stage"].fillna("").astype(str).str.lower().ne("group stage")]
    if knockout.empty:
        return 0
    return int(knockout.apply(lambda row: is_actual_knockout_team(row["home_team"]) and is_actual_knockout_team(row["away_team"]), axis=1).sum())


def preserve_existing_if_better(output: pd.DataFrame, debug_rows: list[dict]) -> pd.DataFrame:
    if not RESOLVED_FIXTURES_PATH.exists():
        return output
    try:
        existing = pd.read_csv(RESOLVED_FIXTURES_PATH, encoding="utf-8")
    except pd.errors.EmptyDataError:
        return output
    existing_count = actual_knockout_count(existing)
    output_count = actual_knockout_count(output)
    if existing_count > output_count:
        debug_rows.append(
            {
                "source": "preserve_existing",
                "http_status": "",
                "matches_found": len(existing),
                "actual_team_matches": existing_count,
                "fallback_slot_matches": max(len(output) - output_count, 0),
                "error_message": f"kept existing resolved fixtures because new actual count {output_count} < existing {existing_count}",
            }
        )
        return existing
    return output


def write_debug(rows: list[dict]) -> None:
    DEBUG_PATH.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(DEBUG_PATH, index=False, encoding="utf-8")


def fetch_knockout_bracket(start: str, end: str) -> pd.DataFrame:
    debug_rows = []
    matches, espn_debug = fetch_espn_knockout(start, end)
    debug_rows.append(espn_debug)

    knockout = assign_match_ids(matches)
    initial_actual_count = int(knockout.apply(lambda row: is_actual_knockout_team(row["home_team"]) and is_actual_knockout_team(row["away_team"]), axis=1).sum())
    local_knockout, local_debug = local_results_knockout()
    if initial_actual_count == 0 and not local_knockout.empty:
        resolved_ids = set(local_knockout["match_id"])
        knockout = pd.concat(
            [local_knockout, knockout[~knockout["match_id"].isin(resolved_ids)]],
            ignore_index=True,
        ).sort_values(["date", "match_id"]).reset_index(drop=True)
    elif initial_actual_count > 0:
        local_debug["error_message"] = "skipped because ESPN returned actual knockout teams"
    debug_rows.append(local_debug)

    actual_count = int(knockout.apply(lambda row: is_actual_knockout_team(row["home_team"]) and is_actual_knockout_team(row["away_team"]), axis=1).sum())
    fallback_count = int(len(knockout) - actual_count)
    debug_rows[0]["actual_team_matches"] = actual_count
    debug_rows[0]["fallback_slot_matches"] = fallback_count

    if actual_count == 0:
        debug_rows.extend(
            [
                {
                    "source": "fifa",
                    "http_status": "",
                    "matches_found": 0,
                    "actual_team_matches": 0,
                    "fallback_slot_matches": fallback_count,
                    "error_message": "not attempted after ESPN returned no actual knockout teams; add source-specific parser if needed",
                },
                {
                    "source": "sofascore",
                    "http_status": "",
                    "matches_found": 0,
                    "actual_team_matches": 0,
                    "fallback_slot_matches": fallback_count,
                    "error_message": "not attempted after ESPN returned no actual knockout teams; add source-specific parser if needed",
                },
            ]
        )

    output = merge_with_base_fixtures(knockout)
    output = preserve_existing_if_better(output, debug_rows)
    saved_results = save_completed_results(knockout)
    if debug_rows:
        debug_rows[0]["error_message"] = (debug_rows[0].get("error_message", "") + f"; saved_completed_results={saved_results}").strip("; ")
    RESOLVED_FIXTURES_PATH.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(RESOLVED_FIXTURES_PATH, index=False, encoding="utf-8")
    write_debug(debug_rows)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default="2026-06-28")
    parser.add_argument("--end", default="2026-07-19")
    args = parser.parse_args()

    output = fetch_knockout_bracket(args.start, args.end)
    knockout = output[output.get("stage", pd.Series("", index=output.index)).fillna("").astype(str).str.lower().ne("group stage")]
    actual_count = int(knockout.apply(lambda row: is_actual_knockout_team(row["home_team"]) and is_actual_knockout_team(row["away_team"]), axis=1).sum())
    print(f"Resolved knockout fixtures written: {RESOLVED_FIXTURES_PATH}")
    print(f"Knockout rows: {len(knockout)}")
    print(f"Actual-team knockout rows: {actual_count}")
    print(f"Debug report: {DEBUG_PATH}")


if __name__ == "__main__":
    main()
