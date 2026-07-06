import argparse
import json
import sys
from datetime import date
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agents.intelligence_agent import IntelligenceAgent
from src.paths import PROCESSED_DATA_DIR


DEFAULT_FIXTURES = PROCESSED_DATA_DIR / "worldcup_fixtures.csv"
RESOLVED_FIXTURES = PROCESSED_DATA_DIR / "worldcup_fixtures_resolved.csv"
DEFAULT_MANUAL_INTEL = PROJECT_ROOT / "data" / "manual" / "worldcup_intel_manual.csv"
DEFAULT_OPENAI_INTEL = PROCESSED_DATA_DIR / "worldcup_openai_intel.csv"
DEFAULT_ARTICLE_INTEL = PROCESSED_DATA_DIR / "worldcup_article_intel.csv"
DEFAULT_ODDS = PROCESSED_DATA_DIR / "worldcup_odds.csv"
CONSENSUS_ODDS = PROCESSED_DATA_DIR / "worldcup_consensus_odds.csv"
OPENAI_ODDS = PROCESSED_DATA_DIR / "worldcup_openai_odds.csv"
DEFAULT_POISSON = PROCESSED_DATA_DIR / "worldcup_poisson_predictions.csv"
DEFAULT_MODEL_PREDICTIONS = PROCESSED_DATA_DIR / "worldcup_model_predictions.csv"
DEFAULT_MODEL_ONLY = PROJECT_ROOT / "reports" / "worldcup_model_only_predictions.csv"
DEFAULT_BETTING = PROJECT_ROOT / "reports" / "worldcup_betting_predictions.csv"
INTEL_OUTPUT = PROJECT_ROOT / "reports" / "worldcup_daily_intel.csv"
MISSING_OUTPUT = PROJECT_ROOT / "reports" / "worldcup_daily_intel_missing_report.csv"
BRIEF_OUTPUT = PROJECT_ROOT / "reports" / "worldcup_daily_betting_brief.md"
INTEL_MERGE_DEBUG_OUTPUT = PROJECT_ROOT / "reports" / "intel_merge_debug.csv"
DAILY_BRIEF_DEBUG_OUTPUT = PROJECT_ROOT / "reports" / "daily_brief_today_debug.csv"
OPENAI_INTEL_DEBUG_OUTPUT = PROJECT_ROOT / "reports" / "openai_intel_debug.csv"
WORLDCUP_FEATURES_OUTPUT = PROCESSED_DATA_DIR / "worldcup_features.csv"
FEATURES_SOURCE_DEBUG_OUTPUT = PROJECT_ROOT / "reports" / "worldcup_features_source_debug.csv"
POISSON_MERGE_DEBUG_OUTPUT = PROJECT_ROOT / "reports" / "poisson_merge_debug.csv"
INTEL_RISK_DEBUG_OUTPUT = PROJECT_ROOT / "reports" / "intel_risk_debug.csv"
MODEL_MISSING_REPORT_OUTPUT = PROJECT_ROOT / "reports" / "worldcup_model_missing_report.csv"


def choose_predictions_path() -> Path | None:
    if DEFAULT_BETTING.exists():
        return DEFAULT_BETTING
    if DEFAULT_MODEL_ONLY.exists():
        return DEFAULT_MODEL_ONLY
    return None


def choose_fixtures_path() -> Path:
    if RESOLVED_FIXTURES.exists():
        return RESOLVED_FIXTURES
    return DEFAULT_FIXTURES


def format_optional(value) -> str:
    if pd.isna(value):
        return "unknown"
    try:
        return f"{float(value):.3f}"
    except (TypeError, ValueError):
        return str(value)


def format_text(value) -> str:
    if pd.isna(value):
        return "unknown"
    text = str(value).strip()
    if text.lower() in {"", "nan", "none", "<na>"}:
        return "unknown"
    return text


def format_odds(value) -> str:
    if pd.isna(value):
        return "unknown"
    try:
        return f"{float(value):.2f}"
    except (TypeError, ValueError):
        return "unknown"


def implied_market_probabilities(row) -> tuple[str, str, str]:
    market_values = [getattr(row, column, pd.NA) for column in ["market_H", "market_D", "market_A"]]
    if all(pd.notna(value) for value in market_values):
        return tuple(format_optional(value) for value in market_values)

    odds_values = [getattr(row, column, pd.NA) for column in ["home_odds", "draw_odds", "away_odds"]]
    try:
        odds = pd.Series(odds_values, dtype="float64")
    except (TypeError, ValueError):
        return ("unknown", "unknown", "unknown")
    if odds.isna().any() or (odds <= 1).any():
        return ("unknown", "unknown", "unknown")
    probabilities = (1 / odds) / (1 / odds).sum()
    return tuple(f"{value:.3f}" for value in probabilities)


def split_source_urls(value) -> list[str]:
    text = str(value).strip()
    if text.lower() in {"", "unknown", "nan", "none", "<na>"}:
        return []
    urls = []
    for part in text.replace(",", ";").split(";"):
        url = part.strip()
        if url.startswith(("http://", "https://")) and url not in urls:
            urls.append(url)
    return urls


def format_source_links(value, limit: int = 2) -> str:
    urls = split_source_urls(value)
    if not urls:
        return "unknown"
    links = [f"[Source {index + 1}]({url})" for index, url in enumerate(urls[:limit])]
    remaining = len(urls) - limit
    if remaining > 0:
        links.append(f"+ {remaining} more sources")
    return " · ".join(links)


def format_poisson_scores(value) -> str:
    if pd.isna(value):
        return "unknown"
    text = str(value).strip()
    if text.lower() in {"", "unknown", "nan", "none", "<na>"}:
        return "unknown"
    try:
        parsed = json.loads(text)
    except (TypeError, json.JSONDecodeError):
        return text
    if not isinstance(parsed, list):
        return text
    rows = []
    for item in parsed[:5]:
        if not isinstance(item, dict):
            continue
        scoreline = str(item.get("scoreline", "")).strip()
        try:
            probability = float(item.get("probability", 0))
        except (TypeError, ValueError):
            continue
        if scoreline:
            rows.append(f"{scoreline}:{probability:.3f}")
    return "; ".join(rows) if rows else "unknown"


def odds_values_available(row) -> bool:
    try:
        odds = [float(getattr(row, column, pd.NA)) for column in ["home_odds", "draw_odds", "away_odds"]]
    except (TypeError, ValueError):
        return False
    return all(value > 1 for value in odds)


def refresh_reason_after_odds(row) -> str:
    reason = format_text(getattr(row, "reason", ""))
    if not odds_values_available(row):
        return reason
    blocked = "No odds available; only WATCH / PASS is allowed."
    reason = reason.replace(blocked, "").strip()
    return reason or "Odds and pre-match intel are available."


def ensure_daily_intel_alias_columns(data: pd.DataFrame) -> pd.DataFrame:
    output = data.copy()
    aliases = {
        "home_injuries": "injuries_home",
        "away_injuries": "injuries_away",
        "home_suspensions": "suspensions_home",
        "away_suspensions": "suspensions_away",
        "home_expected_lineup": "expected_lineup_home",
        "away_expected_lineup": "expected_lineup_away",
        "home_coach_comments": "coach_comments_home",
        "away_coach_comments": "coach_comments_away",
        "intel_updated_at": "fetched_at",
    }
    for target, source in aliases.items():
        if target not in output.columns:
            output[target] = output[source] if source in output.columns else ""
        elif source in output.columns:
            output[target] = output[target].fillna(output[source])
    if "source_urls" not in output.columns:
        output["source_urls"] = output["source_url"] if "source_url" in output.columns else ""
    if "source_url" in output.columns:
        empty_sources = output["source_urls"].isna() | output["source_urls"].astype(str).str.strip().isin(["", "unknown", "nan", "None", "<NA>"])
        output.loc[empty_sources, "source_urls"] = output.loc[empty_sources, "source_url"]
    return output


def today_only(data: pd.DataFrame, as_of_date: str) -> pd.DataFrame:
    if data.empty or "date" not in data.columns:
        return data
    output = data.copy()
    output["date"] = pd.to_datetime(output["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    return output[output["date"] == as_of_date].reset_index(drop=True)


def normalize_team(value) -> str:
    text = "" if pd.isna(value) else str(value).strip()
    aliases = {
        "Bosnia-Herzegovina": "Bosnia and Herzegovina",
        "Bosnia Herzegovina": "Bosnia and Herzegovina",
        "BIH": "Bosnia and Herzegovina",
        "DR Congo": "Congo DR",
        "Democratic Republic of Congo": "Congo DR",
        "Côte d'Ivoire": "Ivory Coast",
        "Cote d'Ivoire": "Ivory Coast",
        "USA": "United States",
        "USMNT": "United States",
    }
    return aliases.get(text, text).casefold()


def add_merge_keys(data: pd.DataFrame) -> pd.DataFrame:
    output = data.copy()
    output["date"] = pd.to_datetime(output["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    output["_home_key"] = output["home_team"].map(normalize_team)
    output["_away_key"] = output["away_team"].map(normalize_team)
    output["_match_key"] = output["date"].astype(str) + "|" + output["_home_key"] + "|" + output["_away_key"]
    return output


def load_odds_sources(primary_path: Path | None) -> pd.DataFrame:
    paths = []
    for path in [primary_path, CONSENSUS_ODDS, OPENAI_ODDS]:
        if path and path not in paths:
            paths.append(path)

    frames = []
    for priority, path in enumerate(paths):
        if not path.exists():
            continue
        try:
            data = pd.read_csv(path, encoding="utf-8")
        except pd.errors.EmptyDataError:
            continue
        required = ["date", "home_team", "away_team", "home_odds", "draw_odds", "away_odds"]
        if data.empty or not all(column in data.columns for column in required):
            continue
        data = data.copy()
        data["date"] = pd.to_datetime(data["date"], errors="coerce").dt.strftime("%Y-%m-%d")
        data["_odds_priority"] = priority
        if "odds_status" not in data.columns:
            data["odds_status"] = "available"
        if "odds_source" not in data.columns:
            data["odds_source"] = path.stem
        frames.append(data[["date", "home_team", "away_team", "home_odds", "draw_odds", "away_odds", "odds_status", "odds_source", "_odds_priority"]])

    if not frames:
        return pd.DataFrame(columns=["date", "home_team", "away_team", "home_odds", "draw_odds", "away_odds", "odds_status", "odds_source"])

    odds = pd.concat(frames, ignore_index=True)
    for column in ["home_odds", "draw_odds", "away_odds"]:
        odds[column] = pd.to_numeric(odds[column], errors="coerce")
    odds["_complete"] = odds[["home_odds", "draw_odds", "away_odds"]].notna().all(axis=1)
    odds = odds.sort_values(["date", "home_team", "away_team", "_complete", "_odds_priority"], ascending=[True, True, True, False, True])
    return odds.drop_duplicates(["date", "home_team", "away_team"], keep="first").drop(columns=["_complete", "_odds_priority"])


def add_odds_columns(data: pd.DataFrame, odds_path: Path | None) -> pd.DataFrame:
    odds = load_odds_sources(odds_path)
    if data.empty or odds.empty:
        return data
    output = data.merge(odds, on=["date", "home_team", "away_team"], how="left", suffixes=("", "_odds"))
    if "odds_status_odds" in output.columns:
        output["odds_status"] = output["odds_status_odds"].combine_first(output.get("odds_status"))
        output = output.drop(columns=["odds_status_odds"])
    return output


def add_poisson_columns(data: pd.DataFrame, poisson_path: Path = DEFAULT_POISSON) -> pd.DataFrame:
    output = data.copy()
    debug_rows = []
    if output.empty or not poisson_path.exists():
        reason = "daily_empty" if output.empty else "poisson_file_missing"
        for row in output.itertuples(index=False):
            debug_rows.append(
                {
                    "match_key": f"{getattr(row, 'date', '')}|{getattr(row, 'home_team', '')}|{getattr(row, 'away_team', '')}",
                    "exists_in_daily": True,
                    "exists_in_poisson": False,
                    "daily_home": getattr(row, "home_team", ""),
                    "daily_away": getattr(row, "away_team", ""),
                    "poisson_home": "",
                    "poisson_away": "",
                    "poisson_top_scores": "",
                    "merge_success": False,
                    "reason": reason,
                }
            )
        write_poisson_merge_debug(debug_rows)
        return output

    poisson = pd.read_csv(poisson_path, encoding="utf-8")
    required = ["date", "home_team", "away_team", "poisson_top_scores"]
    if poisson.empty or not all(column in poisson.columns for column in required):
        for row in output.itertuples(index=False):
            debug_rows.append(
                {
                    "match_key": f"{getattr(row, 'date', '')}|{getattr(row, 'home_team', '')}|{getattr(row, 'away_team', '')}",
                    "exists_in_daily": True,
                    "exists_in_poisson": False,
                    "daily_home": getattr(row, "home_team", ""),
                    "daily_away": getattr(row, "away_team", ""),
                    "poisson_home": "",
                    "poisson_away": "",
                    "poisson_top_scores": "",
                    "merge_success": False,
                    "reason": "poisson_missing_required_columns",
                }
            )
        write_poisson_merge_debug(debug_rows)
        return output

    for column in ["poisson_home_xg", "poisson_away_xg"]:
        if column not in poisson.columns:
            poisson[column] = pd.NA

    daily_keyed = add_merge_keys(output)
    poisson_keyed = add_merge_keys(poisson)
    poisson_keyed = poisson_keyed.drop_duplicates("_match_key", keep="last")
    poisson_subset = poisson_keyed[
        ["_match_key", "home_team", "away_team", "poisson_home_xg", "poisson_away_xg", "poisson_top_scores"]
    ].rename(
        columns={
            "home_team": "_poisson_home",
            "away_team": "_poisson_away",
            "poisson_home_xg": "_poisson_home_xg",
            "poisson_away_xg": "_poisson_away_xg",
            "poisson_top_scores": "_poisson_top_scores",
        }
    )
    merged = daily_keyed.merge(poisson_subset, on="_match_key", how="left")

    for column, source_column in {
        "poisson_home_xg": "_poisson_home_xg",
        "poisson_away_xg": "_poisson_away_xg",
        "poisson_top_scores": "_poisson_top_scores",
    }.items():
        if column not in merged.columns:
            merged[column] = pd.NA
        current_empty = merged[column].isna() | merged[column].astype(str).str.strip().isin(["", "nan", "None", "<NA>"])
        merged[column] = merged[column].where(~current_empty, merged[source_column])

    for _, row in merged.iterrows():
        exists = pd.notna(row.get("_poisson_top_scores"))
        debug_rows.append(
            {
                "match_key": row.get("_match_key", ""),
                "exists_in_daily": True,
                "exists_in_poisson": bool(exists),
                "daily_home": row.get("home_team", ""),
                "daily_away": row.get("away_team", ""),
                "poisson_home": row.get("_poisson_home", ""),
                "poisson_away": row.get("_poisson_away", ""),
                "poisson_top_scores": row.get("_poisson_top_scores", ""),
                "merge_success": bool(exists),
                "reason": "" if exists else "no_matching_poisson_row",
            }
        )

    write_poisson_merge_debug(debug_rows)
    drop_columns = [column for column in merged.columns if column.startswith("_")]
    return merged.drop(columns=drop_columns)


def add_model_columns(data: pd.DataFrame, model_path: Path = DEFAULT_MODEL_PREDICTIONS) -> pd.DataFrame:
    output = data.copy()
    if output.empty:
        write_model_missing_report(output.iloc[0:0])
        return output
    if not model_path.exists():
        write_model_missing_report(output)
        print("MODEL_PREDICTIONS_ROWS=0")
        print("MODEL_PREDICTIONS_COLUMNS=")
        print(f"MODEL_MISSING_ROWS={len(output)}")
        return output
    model = pd.read_csv(model_path, encoding="utf-8")
    print(f"MODEL_PREDICTIONS_ROWS={len(model)}")
    print(f"MODEL_PREDICTIONS_COLUMNS={','.join(model.columns)}")
    required = ["date", "home_team", "away_team", "model_H", "model_D", "model_A"]
    if model.empty or not all(column in model.columns for column in required):
        write_model_missing_report(output)
        print(f"MODEL_MISSING_ROWS={len(output)}")
        return output
    keyed = add_merge_keys(output)
    model_keyed = add_merge_keys(model)
    model_subset = model_keyed[["_match_key", "model_H", "model_D", "model_A"]].drop_duplicates("_match_key", keep="last")
    merged = keyed.merge(model_subset, on="_match_key", how="left", suffixes=("", "_model_file"))
    for column in ["model_H", "model_D", "model_A"]:
        source = f"{column}_model_file"
        if source in merged.columns:
            current_missing = merged[column].isna() if column in merged.columns else pd.Series(True, index=merged.index)
            if column not in merged.columns:
                merged[column] = pd.NA
            merged.loc[current_missing, column] = merged.loc[current_missing, source]
    missing_mask = (
        merged["home_team"].fillna("").astype(str).str.strip().str.upper().ne("TBD")
        & merged["away_team"].fillna("").astype(str).str.strip().str.upper().ne("TBD")
        & merged[["model_H", "model_D", "model_A"]].isna().any(axis=1)
    )
    missing = merged.loc[missing_mask].copy()
    write_model_missing_report(missing)
    print(f"FEATURE_ROWS_AFTER_MODEL_MERGE={len(merged)}")
    print(f"MODEL_MISSING_ROWS={len(missing)}")
    example = merged[
        (merged["home_team"].astype(str).str.casefold() == "portugal")
        & (merged["away_team"].astype(str).str.casefold() == "spain")
    ]
    if not example.empty:
        print(f"MODEL_EXAMPLE_PORTUGAL_VS_SPAIN={example.iloc[-1][['date', 'home_team', 'away_team', 'model_H', 'model_D', 'model_A']].to_dict()}")
    drop_columns = [column for column in merged.columns if column.startswith("_") or column.endswith("_model_file")]
    return merged.drop(columns=drop_columns)


def write_model_missing_report(rows: pd.DataFrame) -> None:
    MODEL_MISSING_REPORT_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    columns = ["date", "group", "home_team", "away_team", "status", "model_H", "model_D", "model_A"]
    output = rows.copy()
    for column in columns:
        if column not in output.columns:
            output[column] = pd.NA
    output[columns].to_csv(MODEL_MISSING_REPORT_OUTPUT, index=False, encoding="utf-8")


def write_poisson_merge_debug(rows: list[dict]) -> None:
    POISSON_MERGE_DEBUG_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        rows,
        columns=[
            "match_key",
            "exists_in_daily",
            "exists_in_poisson",
            "daily_home",
            "daily_away",
            "poisson_home",
            "poisson_away",
            "poisson_top_scores",
            "merge_success",
            "reason",
        ],
    ).to_csv(POISSON_MERGE_DEBUG_OUTPUT, index=False, encoding="utf-8")


def save_worldcup_features(data: pd.DataFrame) -> None:
    features = data.copy()
    if features.empty:
        missing_columns = []
    else:
        if "poisson_diff" not in features.columns and {"poisson_home_xg", "poisson_away_xg"}.issubset(features.columns):
            features["poisson_diff"] = pd.to_numeric(features["poisson_away_xg"], errors="coerce") - pd.to_numeric(features["poisson_home_xg"], errors="coerce")
        for column in ["home_odds", "draw_odds", "away_odds"]:
            if column in features.columns:
                features[column] = pd.to_numeric(features[column], errors="coerce")
        odds_complete = features[["home_odds", "draw_odds", "away_odds"]].notna().all(axis=1) if all(column in features.columns for column in ["home_odds", "draw_odds", "away_odds"]) else False
        if "odds_status" not in features.columns:
            features["odds_status"] = pd.Series(odds_complete).map({True: "available", False: "missing"})
        else:
            features["odds_status"] = features["odds_status"].fillna(pd.Series(odds_complete).map({True: "available", False: "missing"}))
        for column in ["market_H", "market_D", "market_A"]:
            if column not in features.columns:
                features[column] = pd.NA
        odds_ready = all(column in features.columns for column in ["home_odds", "draw_odds", "away_odds"])
        if odds_ready:
            raw_h = 1 / pd.to_numeric(features["home_odds"], errors="coerce")
            raw_d = 1 / pd.to_numeric(features["draw_odds"], errors="coerce")
            raw_a = 1 / pd.to_numeric(features["away_odds"], errors="coerce")
            total = raw_h + raw_d + raw_a
            missing_market = features[["market_H", "market_D", "market_A"]].isna().any(axis=1)
            valid_total = total.notna() & (total > 0)
            features.loc[missing_market & valid_total, "market_H"] = raw_h[missing_market & valid_total] / total[missing_market & valid_total]
            features.loc[missing_market & valid_total, "market_D"] = raw_d[missing_market & valid_total] / total[missing_market & valid_total]
            features.loc[missing_market & valid_total, "market_A"] = raw_a[missing_market & valid_total] / total[missing_market & valid_total]
        if "home_recent_form_score" in features.columns and "recent_form_home" not in features.columns:
            features["recent_form_home"] = features["home_recent_form_score"]
        if "away_recent_form_score" in features.columns and "recent_form_away" not in features.columns:
            features["recent_form_away"] = features["away_recent_form_score"]
        for column, default in {
            "recent_form_home": pd.NA,
            "recent_form_away": pd.NA,
            "weather_risk_score": 0,
            "poisson_home_xg": pd.NA,
            "poisson_away_xg": pd.NA,
            "rest_days_home": pd.NA,
            "rest_days_away": pd.NA,
            "points_home_before": 0,
            "points_away_before": 0,
            "goal_diff_home_before": 0,
            "goal_diff_away_before": 0,
            "must_win_home": False,
            "must_win_away": False,
            "already_qualified_home": False,
            "already_qualified_away": False,
        }.items():
            if column not in features.columns:
                features[column] = default

    required = [
        "date",
        "home_team",
        "away_team",
        "model_H",
        "model_D",
        "model_A",
        "poisson_home_xg",
        "poisson_away_xg",
        "home_odds",
        "draw_odds",
        "away_odds",
        "odds_status",
        "intel_risk",
        "recent_form_home",
        "recent_form_away",
        "rest_days_home",
        "rest_days_away",
        "weather_risk_score",
    ]
    missing_columns = [column for column in required if column not in features.columns]
    WORLDCUP_FEATURES_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(WORLDCUP_FEATURES_OUTPUT, index=False, encoding="utf-8")
    FEATURES_SOURCE_DEBUG_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "source_dataframe": "run_daily_worldcup_intel.result_data",
                "rows": len(features),
                "columns": len(features.columns),
                "saved_to_worldcup_features": True,
                "missing_columns": ";".join(missing_columns),
            }
        ]
    ).to_csv(FEATURES_SOURCE_DEBUG_OUTPUT, index=False, encoding="utf-8")


def write_daily_brief_debug(all_data: pd.DataFrame, brief_data: pd.DataFrame, as_of_date: str) -> None:
    rows = []
    included_keys = set()
    if not brief_data.empty:
        included_keys = set((brief_data["date"].astype(str) + "|" + brief_data["home_team"].astype(str) + "|" + brief_data["away_team"].astype(str)).tolist())
    if all_data.empty:
        rows.append(
            {
                "match_key": "",
                "openai_status": "no_today_matches",
                "source_urls_count": 0,
                "json_parse_success": False,
                "intel_has_content": False,
                "included_in_daily_brief": False,
                "exclude_reason": "no_today_matches",
            }
        )
    for row in all_data.itertuples(index=False):
        match_key = f"{getattr(row, 'date', '')}|{getattr(row, 'home_team', '')}|{getattr(row, 'away_team', '')}"
        source_urls = str(getattr(row, "source_urls", getattr(row, "source_url", "unknown"))).strip()
        source_urls_count = 0 if source_urls.lower() in {"", "unknown", "nan", "none", "<na>"} else len([url for url in source_urls.split(";") if url.strip()])
        included = match_key in included_keys
        rows.append(
            {
                "match_key": match_key,
                "openai_status": getattr(row, "source_status", "unknown"),
                "source_urls_count": source_urls_count,
                "json_parse_success": getattr(row, "source_status", "") == "ok",
                "intel_has_content": getattr(row, "intel_has_content", False),
                "included_in_daily_brief": included,
                "exclude_reason": "" if included else ("date_not_today" if str(getattr(row, "date", "")) != as_of_date else "not_in_brief_output"),
            }
        )
    DAILY_BRIEF_DEBUG_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    debug = pd.DataFrame(rows)
    debug.to_csv(DAILY_BRIEF_DEBUG_OUTPUT, index=False, encoding="utf-8")
    debug.to_csv(OPENAI_INTEL_DEBUG_OUTPUT, index=False, encoding="utf-8")


def write_intel_risk_debug(data: pd.DataFrame) -> None:
    columns = [
        "match_key",
        "intel_has_content",
        "source_status",
        "injuries_home",
        "injuries_away",
        "suspensions_home",
        "suspensions_away",
        "risk_score",
        "final_intel_risk",
        "reason",
    ]
    rows = []
    for row in data.itertuples(index=False):
        rows.append(
            {
                "match_key": f"{getattr(row, 'date', '')}|{getattr(row, 'home_team', '')}|{getattr(row, 'away_team', '')}",
                "intel_has_content": getattr(row, "intel_has_content", False),
                "source_status": getattr(row, "source_status", ""),
                "injuries_home": getattr(row, "injuries_home", ""),
                "injuries_away": getattr(row, "injuries_away", ""),
                "suspensions_home": getattr(row, "suspensions_home", ""),
                "suspensions_away": getattr(row, "suspensions_away", ""),
                "risk_score": getattr(row, "intel_risk_score", 0),
                "final_intel_risk": getattr(row, "intel_risk", "UNKNOWN"),
                "reason": getattr(row, "intel_risk_reason", ""),
            }
        )
    INTEL_RISK_DEBUG_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=columns).to_csv(INTEL_RISK_DEBUG_OUTPUT, index=False, encoding="utf-8")


def write_brief(data: pd.DataFrame, output_path: Path, as_of_date: str) -> None:
    lines = [
        "# World Cup Daily Betting Brief",
        "",
        f"As of: {as_of_date}",
        "",
        "Note: This is pre-match intelligence support only. It does not auto-place bets.",
        "",
    ]
    if data.empty:
        lines.append("No matches found for today.")
    for row in data.itertuples(index=False):
        kickoff = getattr(row, "kickoff_time", getattr(row, "time", "unknown"))
        source_links = format_source_links(getattr(row, "source_urls", getattr(row, "source_url", "unknown")))
        market_h, market_d, market_a = implied_market_probabilities(row)
        lines.extend(
            [
                f"## {row.date} {row.home_team} vs {row.away_team}",
                "",
                f"- Kickoff time: {kickoff}",
                f"- Action: {row.recommended_action}",
                f"- Odds H/D/A: {format_odds(getattr(row, 'home_odds', pd.NA))} / {format_odds(getattr(row, 'draw_odds', pd.NA))} / {format_odds(getattr(row, 'away_odds', pd.NA))}",
                f"- Odds status: {format_text(getattr(row, 'odds_status', 'missing'))} ({format_text(getattr(row, 'odds_source', 'unknown'))})",
                f"- Market H/D/A: {market_h} / {market_d} / {market_a}",
                f"- Value side / edge: {format_text(getattr(row, 'value_side', 'unknown'))} / {format_optional(getattr(row, 'edge', pd.NA))}",
                f"- Model H/D/A: {format_optional(row.model_H)} / {format_optional(row.model_D)} / {format_optional(row.model_A)}",
                f"- Poisson top scores: {format_poisson_scores(getattr(row, 'poisson_top_scores', pd.NA))}",
                f"- Intel risk: {format_text(getattr(row, 'intel_risk', 'unknown'))}",
                f"- Home injuries: {format_text(getattr(row, 'home_injuries', getattr(row, 'injuries_home', 'unknown')))}",
                f"- Away injuries: {format_text(getattr(row, 'away_injuries', getattr(row, 'injuries_away', 'unknown')))}",
                f"- Home suspensions: {format_text(getattr(row, 'home_suspensions', getattr(row, 'suspensions_home', 'unknown')))}",
                f"- Away suspensions: {format_text(getattr(row, 'away_suspensions', getattr(row, 'suspensions_away', 'unknown')))}",
                f"- Home expected lineup: {format_text(getattr(row, 'home_expected_lineup', getattr(row, 'expected_lineup_home', 'unknown')))}",
                f"- Away expected lineup: {format_text(getattr(row, 'away_expected_lineup', getattr(row, 'expected_lineup_away', 'unknown')))}",
                f"- Home coach comments: {format_text(getattr(row, 'home_coach_comments', getattr(row, 'coach_comments_home', 'unknown')))}",
                f"- Away coach comments: {format_text(getattr(row, 'away_coach_comments', getattr(row, 'coach_comments_away', 'unknown')))}",
                f"- Sources: {source_links}",
                f"- Reason: {refresh_reason_after_odds(row)}",
                "",
            ]
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")


def source_keys(path: Path | None) -> set[str]:
    if path is None or not path.exists() or path.is_dir():
        return set()
    try:
        data = pd.read_csv(path, encoding="utf-8")
    except pd.errors.EmptyDataError:
        return set()
    if data.empty or not all(column in data.columns for column in ["date", "home_team", "away_team"]):
        return set()
    data = data.copy()
    data["date"] = pd.to_datetime(data["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    return set((data["date"].astype(str) + "|" + data["home_team"].astype(str) + "|" + data["away_team"].astype(str)).tolist())


def write_intel_merge_debug(data: pd.DataFrame, openai_intel_path: Path | None, output_path: Path) -> None:
    if data.empty:
        pd.DataFrame(columns=["match_key", "exists_in_openai_csv", "exists_in_daily_csv", "merge_success", "source_status", "last_fetched", "reason"]).to_csv(
            output_path,
            index=False,
            encoding="utf-8",
        )
        return
    openai_keys = source_keys(openai_intel_path)

    rows = []
    for row in data.itertuples(index=False):
        match_key = f"{row.date}|{row.home_team}|{row.away_team}"
        source_status = getattr(row, "source_status", "unknown")
        fetched_at = getattr(row, "fetched_at", "unknown")
        exists_in_openai = match_key in openai_keys
        merge_success = exists_in_openai and str(source_status).strip().lower() not in {"unknown", "not_run"}
        if not exists_in_openai:
            reason = "missing_from_openai_csv"
        elif not merge_success:
            reason = "intel_row_exists_but_status_not_merged"
        elif str(fetched_at).strip().lower() in {"", "unknown", "nan", "<na>"}:
            reason = "fetched_at_missing_after_merge"
        else:
            reason = ""
        rows.append(
            {
                "match_key": match_key,
                "exists_in_openai_csv": exists_in_openai,
                "exists_in_daily_csv": True,
                "merge_success": merge_success,
                "source_status": source_status,
                "last_fetched": fetched_at,
                "reason": reason,
            }
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_path, index=False, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixtures", type=Path, default=None)
    parser.add_argument("--manual-intel", type=Path, default=DEFAULT_MANUAL_INTEL)
    parser.add_argument("--openai-intel", type=Path, nargs="?", const=DEFAULT_OPENAI_INTEL, default=DEFAULT_OPENAI_INTEL)
    parser.add_argument("--article-intel", type=Path, default=None)
    parser.add_argument("--structured-intel", type=Path, default=None)
    parser.add_argument("--news-intel", type=Path, default=None)
    parser.add_argument("--odds", type=Path, default=DEFAULT_ODDS)
    parser.add_argument("--predictions", type=Path, default=None)
    parser.add_argument("--as-of-date", default=date.today().strftime("%Y-%m-%d"))
    parser.add_argument("--days-ahead", type=int, default=0)
    args = parser.parse_args()

    predictions_path = args.predictions or choose_predictions_path()
    result = IntelligenceAgent().run(
        fixtures_path=args.fixtures or choose_fixtures_path(),
        manual_intel_path=args.manual_intel,
        openai_intel_path=args.openai_intel,
        article_intel_path=args.article_intel,
        structured_intel_path=args.structured_intel,
        news_intel_path=args.news_intel,
        predictions_path=predictions_path,
        odds_path=args.odds,
        as_of_date=args.as_of_date,
        days_ahead=args.days_ahead,
    )

    INTEL_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    all_data = result["data"].copy()
    result["data"] = today_only(all_data, args.as_of_date)
    result["data"] = add_odds_columns(result["data"], args.odds)
    result["data"] = add_poisson_columns(result["data"])
    result["data"] = add_model_columns(result["data"])
    if not result["data"].empty:
        odds_available = result["data"].apply(odds_values_available, axis=1)
        result["data"].loc[odds_available, "odds_status"] = result["data"].loc[odds_available, "odds_status"].replace({"missing": "available"})
        result["data"].loc[odds_available, "reason"] = result["data"].loc[odds_available].apply(refresh_reason_after_odds, axis=1)
    result["data"] = ensure_daily_intel_alias_columns(result["data"])
    save_worldcup_features(result["data"])
    result["data"].to_csv(INTEL_OUTPUT, index=False, encoding="utf-8")
    result["missing_report"].to_csv(MISSING_OUTPUT, index=False, encoding="utf-8")
    write_brief(result["data"], BRIEF_OUTPUT, args.as_of_date)
    write_daily_brief_debug(all_data, result["data"], args.as_of_date)
    write_intel_risk_debug(result["data"])
    write_intel_merge_debug(result["data"], args.openai_intel, INTEL_MERGE_DEBUG_OUTPUT)

    print(f"已輸出 daily intel: {INTEL_OUTPUT}")
    print(f"已輸出 missing report: {MISSING_OUTPUT}")
    print(f"已輸出 betting brief: {BRIEF_OUTPUT}")
    print(f"已輸出 intel merge debug: {INTEL_MERGE_DEBUG_OUTPUT}")
    print(f"upcoming matches: {len(result['data'])}")
    print("不自動下注；此輸出只輔助賽前決策。")


if __name__ == "__main__":
    main()

