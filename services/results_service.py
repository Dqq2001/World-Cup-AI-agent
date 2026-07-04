from __future__ import annotations

from datetime import timedelta

import pandas as pd

from data_access.csv_store import read_csv_safe, write_csv_atomic
from data_access.paths import MANUAL_DIR, PROCESSED_DIR, REPORTS_DIR
from services.script_runner import python_command, run_command


PREDICTION_CANDIDATES = [
    REPORTS_DIR / "worldcup_betting_predictions.csv",
    REPORTS_DIR / "worldcup_model_only_predictions.csv",
]
DAILY_INTEL_PATH = REPORTS_DIR / "worldcup_daily_intel.csv"
FIXTURES_PATH = PROCESSED_DIR / "worldcup_fixtures.csv"
RESOLVED_FIXTURES_PATH = PROCESSED_DIR / "worldcup_fixtures_resolved.csv"
WORLDCUP_RESULTS_PATH = PROCESSED_DIR / "worldcup_results.csv"
MANUAL_RESULTS_PATH = MANUAL_DIR / "worldcup_results_manual.csv"
UNIFIED_MATCH_VIEW_DEBUG_PATH = REPORTS_DIR / "unified_match_view_debug.csv"
MATCHES_MERGE_DEBUG_PATH = REPORTS_DIR / "matches_results_merge_debug.csv"
KNOCKOUT_RESULTS_DISPLAY_DEBUG_PATH = REPORTS_DIR / "knockout_results_display_debug.csv"
RESULTS_SERVICE_DEBUG_PATH = REPORTS_DIR / "results_service_debug.csv"

TEAM_ALIASES = {
    "BRA": "Brazil",
    "MAR": "Morocco",
    "HTI": "Haiti",
    "USA": "United States",
    "US": "United States",
    "USMNT": "United States",
    "BIH": "Bosnia and Herzegovina",
    "KOR": "South Korea",
    "Korea Republic": "South Korea",
    "Czechia": "Czech Republic",
    "Côte d'Ivoire": "Ivory Coast",
    "Cote d'Ivoire": "Ivory Coast",
    "Curaçao": "Curacao",
    "DR Congo": "DR Congo",
    "Congo DR": "DR Congo",
    "Democratic Republic of Congo": "DR Congo",
    "Bosnia & Herzegovina": "Bosnia and Herzegovina",
    "Bosnia-Herzegovina": "Bosnia and Herzegovina",
}


def refresh_results(force: bool = False):
    command = python_command("scripts/fetch_worldcup_results.py", "--all-completed")
    if force:
        command.append("--force-refresh")
    return run_command(command)


def _safe_value(row: pd.Series | dict, column: str, default: str = "unknown"):
    if column not in row:
        return default
    value = row[column]
    if pd.isna(value):
        return default
    return value


def normalize_result_columns(data: pd.DataFrame) -> pd.DataFrame:
    data = data.copy()
    rename_map = {
        "home_score": "home_goals",
        "away_score": "away_goals",
        "home_goal": "home_goals",
        "away_goal": "away_goals",
        "home_goals_ft": "home_goals",
        "away_goals_ft": "away_goals",
    }
    return data.rename(columns={old: new for old, new in rename_map.items() if old in data.columns})


def normalize_keys(data: pd.DataFrame) -> pd.DataFrame:
    data = data.copy()
    if "date" in data.columns:
        data["date"] = pd.to_datetime(data["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    for column in ["group", "stage", "round", "match_id", "home_team", "away_team", "status"]:
        if column in data.columns:
            data[column] = data[column].astype(str).str.strip()
    return data


def normalize_team_name(value) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip()
    text = TEAM_ALIASES.get(text, text)
    text = text.replace("&", "and")
    text = " ".join(text.split())
    return text.casefold()


def canonical_team_name(value) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip()
    text = TEAM_ALIASES.get(text, text)
    text = text.replace("&", "and")
    return " ".join(text.split())


def add_match_merge_keys(data: pd.DataFrame) -> pd.DataFrame:
    data = data.copy()
    data["merge_date"] = pd.to_datetime(data.get("date"), errors="coerce").dt.strftime("%Y-%m-%d")
    data["merge_home_team"] = data.get("home_team", pd.Series(index=data.index, dtype=str)).map(normalize_team_name)
    data["merge_away_team"] = data.get("away_team", pd.Series(index=data.index, dtype=str)).map(normalize_team_name)
    return data


def add_dashboard_merge_key(data: pd.DataFrame) -> pd.DataFrame:
    data = data.copy()
    match_id = data.get("match_id", pd.Series("", index=data.index)).fillna("").astype(str).str.strip()
    fallback = (
        pd.to_datetime(data.get("date"), errors="coerce").dt.strftime("%Y-%m-%d")
        + "|"
        + data.get("home_team", pd.Series(index=data.index, dtype=str)).map(normalize_team_name)
        + "|"
        + data.get("away_team", pd.Series(index=data.index, dtype=str)).map(normalize_team_name)
    )
    data["_dashboard_merge_key"] = match_id.where(match_id.ne(""), fallback)
    return data


def load_predictions() -> pd.DataFrame:
    for path in PREDICTION_CANDIDATES:
        data = normalize_keys(read_csv_safe(path))
        if not data.empty:
            return data
    return pd.DataFrame()


def load_latest_fixtures() -> pd.DataFrame:
    if RESOLVED_FIXTURES_PATH.exists():
        return normalize_keys(read_csv_safe(RESOLVED_FIXTURES_PATH))
    return normalize_keys(read_csv_safe(FIXTURES_PATH))


def is_waiting_for_teams(row: pd.Series | dict) -> bool:
    return (
        str(_safe_value(row, "home_team", "")).strip().upper() == "TBD"
        or str(_safe_value(row, "away_team", "")).strip().upper() == "TBD"
        or str(_safe_value(row, "status", "")).strip().lower() == "waiting_for_teams"
    )


def is_knockout_row(row: pd.Series | dict) -> bool:
    label = f"{_safe_value(row, 'stage', '')} {_safe_value(row, 'round', '')}".lower()
    return any(term in label for term in ["round of", "quarter", "semi", "final"])


def combined_match_data() -> pd.DataFrame:
    predictions = load_predictions()
    fixtures = load_latest_fixtures()
    intel = normalize_keys(read_csv_safe(DAILY_INTEL_PATH))
    data = fixtures.copy() if not fixtures.empty else predictions.copy()
    if data.empty:
        _write_source_debug("combined_match_data", data, [])
        return data

    if not predictions.empty:
        data = add_dashboard_merge_key(data)
        predictions = add_dashboard_merge_key(predictions)
        for column in ["home_team", "away_team", "status"]:
            if column in predictions.columns and column in data.columns:
                predictions[f"{column}_prediction"] = predictions[column]
        prediction_columns = [
            column
            for column in predictions.columns
            if column not in data.columns
            or column == "_dashboard_merge_key"
            or column in {"home_team_prediction", "away_team_prediction", "status_prediction"}
        ]
        predictions = predictions[prediction_columns].drop_duplicates("_dashboard_merge_key", keep="first")
        data = data.merge(predictions, on="_dashboard_merge_key", how="left").drop(columns=["_dashboard_merge_key"])
        for column in ["home_team", "away_team"]:
            prediction_column = f"{column}_prediction"
            if prediction_column in data.columns:
                current = data[column].fillna("").astype(str).str.strip().str.upper()
                predicted = data[prediction_column].fillna("").astype(str).str.strip()
                data.loc[current.eq("TBD") & predicted.ne("") & predicted.str.upper().ne("TBD"), column] = predicted
                data = data.drop(columns=[prediction_column])
        if "status_prediction" in data.columns:
            current_status = data["status"].fillna("").astype(str).str.lower()
            predicted_status = data["status_prediction"].fillna("").astype(str).str.strip()
            data.loc[current_status.isin(["", "waiting_for_teams", "tbd"]) & predicted_status.ne(""), "status"] = predicted_status
            data = data.drop(columns=["status_prediction"])

    if not intel.empty:
        merge_keys = ["date", "home_team", "away_team"]
        intel_columns = [
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
            "source_name",
            "source_url",
            "source_type",
            "source_status",
            "intel_has_content",
            "intel_confidence_level",
            "confidence",
            "intel_risk",
            "fetched_at",
            "article_source_count",
            "article_content_count",
            "cross_source_agreement",
            "conflict_detected",
            "final_article_confidence",
            "quality_reason",
        ]
        intel_columns = [column for column in intel_columns if column in intel.columns]
        if all(column in intel.columns for column in merge_keys):
            intel_subset = intel[intel_columns].drop_duplicates(merge_keys, keep="first")
            overlapping = [column for column in intel_columns if column not in merge_keys and column in data.columns]
            if overlapping:
                data = data.drop(columns=overlapping)
            data = data.merge(intel_subset, on=merge_keys, how="left")

    data = normalize_keys(data)
    waiting_mask = data.apply(is_waiting_for_teams, axis=1)
    if "recommended_action" not in data.columns:
        data["recommended_action"] = pd.NA
    data.loc[waiting_mask, "recommended_action"] = data.loc[waiting_mask, "recommended_action"].fillna("WATCH")
    _write_source_debug("combined_match_data", data, [])
    return data


def worldcup_fixture_keys() -> pd.DataFrame:
    fixtures = load_latest_fixtures()
    required = ["date", "home_team", "away_team"]
    if fixtures.empty or not all(column in fixtures.columns for column in required):
        return pd.DataFrame(columns=required)
    return fixtures[required].drop_duplicates()


def load_worldcup_results(today: pd.Timestamp | None = None) -> tuple[pd.DataFrame, str]:
    today = today or pd.Timestamp.today().normalize()
    candidates = [MANUAL_RESULTS_PATH, WORLDCUP_RESULTS_PATH]
    fixture_keys = worldcup_fixture_keys()
    required_scores = ["home_goals", "away_goals"]

    for path in candidates:
        data = normalize_result_columns(normalize_keys(read_csv_safe(path)))
        if data.empty or not all(column in data.columns for column in required_scores):
            continue
        required_keys = ["date", "home_team", "away_team"]
        if not all(column in data.columns for column in required_keys):
            continue
        results = data.copy()
        if not fixture_keys.empty:
            results = results.merge(fixture_keys, on=required_keys, how="inner")
        results["date_dt"] = pd.to_datetime(results["date"], errors="coerce")
        results["home_goals"] = pd.to_numeric(results["home_goals"], errors="coerce")
        results["away_goals"] = pd.to_numeric(results["away_goals"], errors="coerce")
        if "status" in results.columns:
            results["status"] = results["status"].astype(str).str.strip().str.lower()
            results = results[results["status"] == "completed"]
        for column in ["penalties_home", "penalties_away"]:
            if column in results.columns:
                results[column] = pd.to_numeric(results[column], errors="coerce")
        if "aet" not in results.columns:
            results["aet"] = False
        if "winner" not in results.columns:
            results["winner"] = ""
        results = results[(results["date_dt"] <= today) & results["home_goals"].notna() & results["away_goals"].notna()]
        if not results.empty:
            results["competition"] = "WORLD CUP 2026"
            _write_source_debug("load_worldcup_results", results, [])
            return results.sort_values("date_dt", ascending=False), str(path)

    return pd.DataFrame(), "no completed World Cup 2026 results source with score columns"


def build_matches_search_data(data: pd.DataFrame, today: pd.Timestamp | None = None) -> pd.DataFrame:
    keys = ["date", "home_team", "away_team"]
    merge_keys = ["merge_date", "merge_home_team", "merge_away_team"]
    fixtures = load_latest_fixtures()
    base = fixtures.copy() if not fixtures.empty else data.copy()
    if base.empty:
        return pd.DataFrame()

    if not data.empty and all(column in data.columns for column in keys):
        base = add_dashboard_merge_key(base)
        prediction_data = add_dashboard_merge_key(data)
        prediction_cols = [column for column in prediction_data.columns if column not in base.columns or column == "_dashboard_merge_key"]
        prediction_data = prediction_data[prediction_cols].drop_duplicates("_dashboard_merge_key", keep="first")
        base = base.merge(prediction_data, on="_dashboard_merge_key", how="left").drop(columns=["_dashboard_merge_key"])

    base = add_match_merge_keys(base)
    results, _ = load_worldcup_results(today=today)
    result_cols = ["date", "home_team", "away_team", "home_goals", "away_goals", "status"]
    optional_cols = [column for column in ["aet", "penalties_home", "penalties_away", "winner"] if column in results.columns]
    if not results.empty and all(column in results.columns for column in result_cols):
        completed = results[result_cols + optional_cols].copy()
        completed["status"] = completed["status"].astype(str).str.lower()
        completed = completed[completed["status"] == "completed"]
        completed = add_match_merge_keys(completed)
        completed["matched_result_key"] = completed["date"].astype(str) + "|" + completed["home_team"].astype(str) + "|" + completed["away_team"].astype(str)
        completed = completed[merge_keys + ["matched_result_key", "home_goals", "away_goals", "status"] + optional_cols].rename(
            columns={
                "home_goals": "home_goals_result",
                "away_goals": "away_goals_result",
                "status": "status_result",
                "aet": "aet_result",
                "penalties_home": "penalties_home_result",
                "penalties_away": "penalties_away_result",
                "winner": "winner_result",
            }
        )
        base = base.merge(completed, on=merge_keys, how="left")
        if "status_result" in base.columns:
            base["status"] = base["status_result"].combine_first(base.get("status", pd.Series(index=base.index, dtype=str)))
            base = base.drop(columns=["status_result"])
        for column in ["home_goals", "away_goals", "aet", "penalties_home", "penalties_away", "winner"]:
            result_column = f"{column}_result"
            if result_column in base.columns:
                base[column] = base[result_column]
                base = base.drop(columns=[result_column])

    if "matched_result_key" not in base.columns:
        base["matched_result_key"] = pd.NA
    for column in ["home_goals", "away_goals"]:
        if column not in base.columns:
            base[column] = pd.NA
    base["status"] = base.get("status", pd.Series(index=base.index, dtype=str)).fillna("scheduled")
    base["status"] = base["status"].replace("", "scheduled").astype(str).str.lower()
    base.loc[base["home_goals"].notna() & base["away_goals"].notna(), "status"] = "completed"
    base["date_dt"] = pd.to_datetime(base["date"], errors="coerce")
    write_matches_merge_debug(base)
    sort_columns = [column for column in ["date_dt", "stage", "round", "group", "home_team"] if column in base.columns]
    return base.sort_values(sort_columns, na_position="last").reset_index(drop=True)


def has_match_score(row: pd.Series | dict) -> bool:
    try:
        home = float(_safe_value(row, "home_goals", None))
        away = float(_safe_value(row, "away_goals", None))
    except (TypeError, ValueError):
        return False
    return pd.notna(home) and pd.notna(away)


def display_mode_for_match(row: pd.Series | dict) -> str:
    status = str(_safe_value(row, "status", "")).strip().lower()
    if status in {"completed", "complete", "final"} or has_match_score(row):
        return "result"
    return "prediction"


def unified_sort_rank(row: pd.Series | dict) -> int:
    status = str(_safe_value(row, "status", "")).strip().lower()
    if status in {"live", "in_progress", "in progress"}:
        return 0
    if display_mode_for_match(row) == "prediction":
        return 1
    return 2


def unified_match_view(data: pd.DataFrame, today: pd.Timestamp | None = None, window_start: pd.Timestamp | None = None) -> pd.DataFrame:
    today = today or pd.Timestamp.today().normalize()
    window_start = window_start or (today - timedelta(days=1))
    matches = build_matches_search_data(data, today=today)
    prediction_source = load_predictions()
    if not prediction_source.empty:
        prediction_source = prediction_source.copy()
        prediction_source["date_dt"] = pd.to_datetime(prediction_source["date"], errors="coerce")
        prediction_source = prediction_source[(prediction_source["date_dt"] >= window_start) & (prediction_source["date_dt"] <= today)].drop(columns=["date_dt"], errors="ignore")
        if not prediction_source.empty:
            for column in matches.columns:
                if column not in prediction_source.columns:
                    prediction_source[column] = pd.NA
            for column in prediction_source.columns:
                if column not in matches.columns:
                    matches[column] = pd.NA
            matches = pd.concat([matches, prediction_source[matches.columns]], ignore_index=True)
    if matches.empty:
        write_unified_match_view_debug(matches)
        return matches
    matches["date_dt"] = pd.to_datetime(matches["date"], errors="coerce")
    matches = matches[(matches["date_dt"] >= window_start) & (matches["date_dt"] <= today)].copy()
    if matches.empty:
        write_unified_match_view_debug(matches)
        return matches
    stale_static = matches.get("source", pd.Series("", index=matches.index)).fillna("").astype(str).eq("static_slot_fallback")
    has_actual_names = matches["home_team"].fillna("").astype(str).str.upper().ne("TBD") & matches["away_team"].fillna("").astype(str).str.upper().ne("TBD")
    matches = matches[~(stale_static & has_actual_names)].copy()
    if matches.empty:
        write_unified_match_view_debug(matches)
        return matches
    matches["_has_score"] = matches.apply(has_match_score, axis=1)
    matches["_actual_team_count"] = matches["home_team"].fillna("").astype(str).str.upper().ne("TBD").astype(int) + matches["away_team"].fillna("").astype(str).str.upper().ne("TBD").astype(int)
    matches["_key"] = matches["date"].astype(str) + "|" + matches["home_team"].astype(str).map(normalize_team_name) + "|" + matches["away_team"].astype(str).map(normalize_team_name)
    matches = matches.sort_values(["_key", "_has_score", "_actual_team_count"], ascending=[True, False, False])
    matches = matches.drop_duplicates("_key", keep="first")
    matches["display_mode"] = matches.apply(display_mode_for_match, axis=1)
    matches["_sort_rank"] = matches.apply(unified_sort_rank, axis=1)
    matches = matches.sort_values(["_sort_rank", "date_dt", "match_id", "home_team", "away_team"], ascending=[True, True, True, True, True])
    write_unified_match_view_debug(matches)
    return matches.drop(columns=["_sort_rank", "_has_score", "_actual_team_count", "_key"], errors="ignore")


def standardize_recent_source(data: pd.DataFrame, competition_default: str = "") -> pd.DataFrame:
    required = ["date", "home_team", "away_team", "home_goals", "away_goals"]
    if data.empty or not all(column in data.columns for column in required):
        return pd.DataFrame(columns=required + ["competition", "neutral"])
    result = normalize_result_columns(data).copy()
    result["date"] = pd.to_datetime(result["date"], errors="coerce")
    result["home_goals"] = pd.to_numeric(result["home_goals"], errors="coerce")
    result["away_goals"] = pd.to_numeric(result["away_goals"], errors="coerce")
    result["home_team"] = result["home_team"].map(canonical_team_name)
    result["away_team"] = result["away_team"].map(canonical_team_name)
    if "competition" not in result.columns:
        result["competition"] = competition_default
    result["competition"] = result["competition"].fillna(competition_default).replace("", competition_default)
    if "neutral" not in result.columns:
        result["neutral"] = pd.NA
    if "status" in result.columns:
        result = result[result["status"].astype(str).str.lower().eq("completed")]
    return result[
        result["date"].notna()
        & result["home_goals"].notna()
        & result["away_goals"].notna()
        & result["home_team"].ne("")
        & result["away_team"].ne("")
    ][["date", "competition", "home_team", "away_team", "home_goals", "away_goals", "neutral"]]


def combined_recent_results(history: pd.DataFrame) -> pd.DataFrame:
    historical = standardize_recent_source(history)
    worldcup = standardize_recent_source(read_csv_safe(WORLDCUP_RESULTS_PATH), "WORLD CUP 2026")
    combined = pd.concat([historical, worldcup], ignore_index=True)
    if combined.empty:
        return combined
    combined["_dedupe_key"] = (
        combined["date"].dt.strftime("%Y-%m-%d")
        + "|"
        + combined["home_team"].map(normalize_team_name)
        + "|"
        + combined["away_team"].map(normalize_team_name)
        + "|"
        + combined["home_goals"].astype("Int64").astype(str)
        + "-"
        + combined["away_goals"].astype("Int64").astype(str)
    )
    combined = combined.drop_duplicates("_dedupe_key", keep="last").drop(columns=["_dedupe_key"])
    return combined.sort_values("date", ascending=False).reset_index(drop=True)


def recent_matches(history: pd.DataFrame, team: str, limit: int = 5) -> pd.DataFrame:
    data = combined_recent_results(history)
    if data.empty:
        return pd.DataFrame()
    normalized_team = normalize_team_name(team)
    team_rows = data[
        (data["home_team"].map(normalize_team_name) == normalized_team)
        | (data["away_team"].map(normalize_team_name) == normalized_team)
    ].sort_values("date", ascending=False).head(limit)
    rows = []
    for row in team_rows.itertuples(index=False):
        is_home = normalize_team_name(row.home_team) == normalized_team
        goals_for = row.home_goals if is_home else row.away_goals
        goals_against = row.away_goals if is_home else row.home_goals
        team_result = "W" if goals_for > goals_against else "D" if goals_for == goals_against else "L"
        rows.append(
            {
                "date": row.date.strftime("%Y-%m-%d"),
                "competition": row.competition,
                "home_team": row.home_team,
                "away_team": row.away_team,
                "score": f"{int(row.home_goals)}-{int(row.away_goals)}",
                "result": team_result,
                "goals_for": goals_for,
                "goals_against": goals_against,
            }
        )
    return pd.DataFrame(rows)


def recent_summary(recent: pd.DataFrame) -> dict:
    if recent.empty:
        return {"record": "unknown", "points": "unknown", "goals_for": "unknown", "goals_against": "unknown", "clean_sheets": "unknown"}
    wins = int((recent["result"] == "W").sum())
    draws = int((recent["result"] == "D").sum())
    losses = int((recent["result"] == "L").sum())
    return {
        "record": f"{wins}W-{draws}D-{losses}L",
        "points": wins * 3 + draws,
        "goals_for": int(recent["goals_for"].sum()),
        "goals_against": int(recent["goals_against"].sum()),
        "clean_sheets": int((recent["goals_against"] == 0).sum()),
    }


def result_display_text(row: pd.Series | dict) -> str:
    home_goals = _safe_value(row, "home_goals", "")
    away_goals = _safe_value(row, "away_goals", "")
    return f"{home_goals} - {away_goals}"


def write_matches_merge_debug(matches: pd.DataFrame) -> None:
    if matches.empty:
        write_csv_atomic(pd.DataFrame(), MATCHES_MERGE_DEBUG_PATH)
        return
    debug = matches.copy()
    debug["result_found"] = debug["matched_result_key"].notna()
    has_score = debug["home_goals"].notna() & debug["away_goals"].notna()
    debug["reason_if_not_matched"] = ""
    debug.loc[~debug["result_found"], "reason_if_not_matched"] = "missing_result_for_key"
    debug.loc[debug["result_found"] & ~has_score, "reason_if_not_matched"] = "matched_result_not_completed_or_missing_score"
    columns = ["date", "home_team", "away_team", "result_found", "matched_result_key", "status", "home_goals", "away_goals", "reason_if_not_matched"]
    for column in columns:
        if column not in debug.columns:
            debug[column] = pd.NA
    write_csv_atomic(debug[columns], MATCHES_MERGE_DEBUG_PATH)


def write_unified_match_view_debug(matches: pd.DataFrame) -> None:
    columns = ["event_id", "date", "home_team", "away_team", "espn_status", "has_score", "display_mode", "source", "reason"]
    rows = []
    for _, row in matches.iterrows():
        display_mode = display_mode_for_match(row)
        if display_mode == "result":
            reason = "score_or_completed_status"
        elif is_waiting_for_teams(row):
            reason = "waiting_for_teams"
        else:
            reason = "scheduled_without_final_score"
        rows.append(
            {
                "event_id": _safe_value(row, "source_event_id", _safe_value(row, "match_id", "")),
                "date": _safe_value(row, "date", ""),
                "home_team": _safe_value(row, "home_team", ""),
                "away_team": _safe_value(row, "away_team", ""),
                "espn_status": _safe_value(row, "status", "unknown"),
                "has_score": has_match_score(row),
                "display_mode": display_mode,
                "source": _safe_value(row, "source", "unknown"),
                "reason": reason,
            }
        )
    write_csv_atomic(pd.DataFrame(rows, columns=columns), UNIFIED_MATCH_VIEW_DEBUG_PATH)


def write_knockout_results_display_debug(all_results: pd.DataFrame, visible_results: pd.DataFrame) -> None:
    if all_results.empty:
        return
    knockout = all_results[all_results.apply(is_knockout_row, axis=1)].copy()
    if knockout.empty:
        return
    visible_keys = set()
    if not visible_results.empty:
        visible_keys = set((visible_results["date"].astype(str) + "|" + visible_results["home_team"].astype(str) + "|" + visible_results["away_team"].astype(str)).tolist())
    knockout["_display_key"] = knockout["date"].astype(str) + "|" + knockout["home_team"].astype(str) + "|" + knockout["away_team"].astype(str)
    knockout["display_text"] = knockout.apply(result_display_text, axis=1)
    knockout["shown_in_dashboard"] = knockout["_display_key"].isin(visible_keys)
    knockout["reason_if_not_shown"] = ""
    knockout.loc[~knockout["shown_in_dashboard"], "reason_if_not_shown"] = "outside_today_yesterday_results_panel"
    columns = ["date", "home_team", "away_team", "home_goals", "away_goals", "penalties_home", "penalties_away", "winner", "display_text", "shown_in_dashboard", "reason_if_not_shown"]
    for column in columns:
        if column not in knockout.columns:
            knockout[column] = pd.NA
    write_csv_atomic(knockout[columns], KNOCKOUT_RESULTS_DISPLAY_DEBUG_PATH)


def yesterday_or_recent_worldcup_results(today: pd.Timestamp | None = None) -> tuple[pd.DataFrame, str]:
    today = today or pd.Timestamp.today().normalize()
    results, source = load_worldcup_results(today=today)
    if results.empty:
        return results, source
    yesterday = today - timedelta(days=1)
    today_yesterday = results[results["date_dt"].isin([today, yesterday])]
    if not today_yesterday.empty:
        visible = today_yesterday.sort_values("date_dt", ascending=False)
        write_knockout_results_display_debug(results, visible)
        return visible, source
    start = today - timedelta(days=3)
    visible = results[(results["date_dt"] >= start) & (results["date_dt"] <= today)].sort_values("date_dt", ascending=False)
    write_knockout_results_display_debug(results, visible)
    return visible, source


def _write_source_debug(source_dataframe: str, data: pd.DataFrame, missing_columns: list[str]) -> None:
    row = {
        "source_dataframe": source_dataframe,
        "rows": len(data),
        "columns": ",".join(data.columns.astype(str)) if not data.empty else "",
        "missing_columns": ",".join(missing_columns),
    }
    write_csv_atomic(pd.DataFrame([row]), RESULTS_SERVICE_DEBUG_PATH)
