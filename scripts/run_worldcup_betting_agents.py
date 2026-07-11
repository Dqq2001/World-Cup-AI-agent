import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agents.context_agent import ContextAgent
from agents.data_agent import DataAgent
from agents.draw_risk_agent import DrawRiskAgent
from agents.final_betting_agent import FinalBettingAgent
from agents.league_reference_agent import LeagueReferenceAgent
from agents.market_agent import MarketAgent
from agents.poisson_agent import PoissonAgent
from agents.prediction_adjustment_agent import PredictionAdjustmentAgent
from agents.report_agent import ReportAgent
from agents.risk_agent import RiskAgent
from agents.value_agent import ValueAgent
from scripts.build_worldcup_features import build_group_context
from src.paths import PROCESSED_DATA_DIR, WORLDCUP_FEATURES_PATH


DEFAULT_OUTPUT_CSV = PROJECT_ROOT / "reports" / "worldcup_betting_predictions.csv"
DEFAULT_REPORT_DIR = PROJECT_ROOT / "reports" / "worldcup_betting_reports"
MODEL_ONLY_OUTPUT_CSV = PROJECT_ROOT / "reports" / "worldcup_model_only_predictions.csv"
DRAW_RISK_SUMMARY_CSV = PROJECT_ROOT / "reports" / "worldcup_draw_risk_summary.csv"
HIGH_DRAW_RISK_CSV = PROJECT_ROOT / "reports" / "worldcup_high_draw_risk_matches.csv"
ADJUSTMENT_DEBUG_CSV = PROJECT_ROOT / "reports" / "prediction_adjustment_debug.csv"
PREDICTION_SNAPSHOTS_CSV = PROCESSED_DATA_DIR / "worldcup_prediction_snapshots.csv"
NO_ODDS_REASON = "缺少 odds，所以目前不能計算 value / edge；此模式只允許 WATCH 或 PASS。"


def latest_fixtures_path() -> Path:
    resolved = PROCESSED_DATA_DIR / "worldcup_fixtures_resolved.csv"
    if resolved.exists():
        return resolved
    return PROCESSED_DATA_DIR / "worldcup_fixtures.csv"


def build_prediction(row, agents: dict) -> dict:
    market = agents["market"].run(row)
    poisson = agents["poisson"].run(row)
    context = agents["context"].run(row)
    risk = agents["risk"].run(row, market, poisson, context)
    draw_risk = agents["draw_risk"].run(row)
    league_reference = agents["league_reference"].run(row, market, poisson, draw_risk)
    adjustment = agents["adjustment"].run(row)
    adjusted_row = row.copy()
    adjusted_row["adjusted_H"] = adjustment["adjusted_H"]
    adjusted_row["adjusted_D"] = adjustment["adjusted_D"]
    adjusted_row["adjusted_A"] = adjustment["adjusted_A"]
    value = agents["value"].run(adjusted_row, market)
    final = agents["final"].run(value, risk, draw_risk, league_reference)

    return {
        "date": row["date"],
        "group": row.get("group", ""),
        "stage": row.get("stage", ""),
        "round": row.get("round", ""),
        "match_id": row.get("match_id", ""),
        "home_team": row["home_team"],
        "away_team": row["away_team"],
        "home_slot": row.get("home_slot", ""),
        "away_slot": row.get("away_slot", ""),
        "status": row.get("status", ""),
        "market_type": row.get("market_type", "1x2"),
        "market_H": market["market_H"],
        "market_D": market["market_D"],
        "market_A": market["market_A"],
        "home_odds": row.get("home_odds", pd.NA),
        "draw_odds": row.get("draw_odds", pd.NA),
        "away_odds": row.get("away_odds", pd.NA),
        "odds_status": row.get("odds_status", ""),
        "odds_source": row.get("odds_source", ""),
        "model_H": value["model_H"],
        "model_D": value["model_D"],
        "model_A": value["model_A"],
        "adjusted_H": adjustment["adjusted_H"],
        "adjusted_D": adjustment["adjusted_D"],
        "adjusted_A": adjustment["adjusted_A"],
        "decision_H": value["decision_H"],
        "decision_D": value["decision_D"],
        "decision_A": value["decision_A"],
        "adjustment_score_home": adjustment["adjustment_score_home"],
        "adjustment_score_away": adjustment["adjustment_score_away"],
        "adjustment_reasons": adjustment["adjustment_reasons"],
        "final_prediction_source": value["probability_source"],
        "poisson_home_xg": poisson["poisson_home_xg"],
        "poisson_away_xg": poisson["poisson_away_xg"],
        "poisson_top_scores": "; ".join(f"{item['scoreline']}:{item['probability']:.3f}" for item in poisson["poisson_top_scores"]),
        "upset_risk": risk["upset_risk"],
        "intel_risk": row.get("intel_risk", ""),
        "intel_risk_score": row.get("intel_risk_score", ""),
        "intel_risk_reason": row.get("intel_risk_reason", ""),
        "draw_risk_level": draw_risk["draw_risk_level"],
        "draw_risk_score": draw_risk["draw_risk_score"],
        "draw_risk_reasons": "; ".join(draw_risk["draw_risk_reasons"]),
        "league_reference_available": league_reference["league_reference_available"],
        "league_risk_score": league_reference["league_risk_score"],
        "league_reference_level": league_reference["league_reference_level"],
        "league_reference_reasons": "; ".join(league_reference["league_reference_reasons"]),
        "value_side": value["value_side"],
        "edge": value["edge"],
        "recommended_action": final["recommended_action"],
        "recommended_stake": final["recommended_stake"],
        "reason": final["reason"],
    }


def normalize_keys(data: pd.DataFrame) -> pd.DataFrame:
    data = data.copy()
    data["date"] = pd.to_datetime(data["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    for column in ["group", "stage", "round", "match_id", "home_team", "away_team", "status"]:
        if column not in data.columns:
            data[column] = ""
        data[column] = data[column].fillna("").astype(str).str.strip()
    return data


def load_required_csv(path: Path, required_columns: list[str], label: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"缺少 {label}: {path}")
    data = pd.read_csv(path, encoding="utf-8")
    missing = [column for column in required_columns if column not in data.columns]
    if missing:
        raise ValueError(f"{label} 缺少欄位: {missing}")
    optional_columns = [column for column in ["group", "stage", "round", "match_id", "home_slot", "away_slot", "status", "neutral_venue"] if column in data.columns]
    columns = list(dict.fromkeys(required_columns + optional_columns))
    return normalize_keys(data[columns])


def is_waiting_for_teams(row: pd.Series) -> bool:
    return (
        str(row.get("home_team", "")).strip().upper() == "TBD"
        or str(row.get("away_team", "")).strip().upper() == "TBD"
        or str(row.get("status", "")).strip().lower() == "waiting_for_teams"
    )


def choose_fixture_merge_keys(*frames: pd.DataFrame) -> list[str]:
    if all("match_id" in frame.columns for frame in frames):
        if all(frame["match_id"].fillna("").astype(str).str.strip().ne("").all() for frame in frames):
            return ["match_id"]
    return ["date", "home_team", "away_team"]


def parse_poisson_scores(value) -> str:
    if pd.isna(value):
        return ""
    text = str(value)
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return text
    if isinstance(parsed, list):
        return "; ".join(f"{item.get('scoreline')}:{float(item.get('probability', 0)):.3f}" for item in parsed)
    return text


def odds_available(row: pd.Series) -> bool:
    values = [row.get(column, pd.NA) for column in ["home_odds", "draw_odds", "away_odds"]]
    try:
        odds = [float(value) for value in values]
    except (TypeError, ValueError):
        return False
    return all(value > 1 for value in odds)


def model_only_reason(row: pd.Series) -> str:
    if odds_available(row):
        return "Odds are available; value edge remains disabled in model-only fallback because complete feature generation was unavailable."
    return NO_ODDS_REASON


def load_optional_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path, encoding="utf-8")
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def merge_optional_match_source(data: pd.DataFrame, source: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    if data.empty or source.empty:
        return data
    keys = ["date", "home_team", "away_team"]
    if not all(column in source.columns for column in keys):
        return data
    source = normalize_keys(source.copy())
    keep = keys + [column for column in columns if column in source.columns]
    if len(keep) == len(keys):
        return data
    source = source[keep].drop_duplicates(keys, keep="last")
    overlap = [column for column in keep if column not in keys and column in data.columns]
    if overlap:
        data = data.drop(columns=overlap)
    return data.merge(source, on=keys, how="left")


def add_odds_and_intel(data: pd.DataFrame) -> pd.DataFrame:
    odds_frames = []
    for priority, path in enumerate([PROCESSED_DATA_DIR / "worldcup_consensus_odds.csv", PROCESSED_DATA_DIR / "worldcup_openai_odds.csv"]):
        odds = load_optional_csv(path)
        required = ["date", "home_team", "away_team", "home_odds", "draw_odds", "away_odds"]
        if odds.empty or not all(column in odds.columns for column in required):
            continue
        odds = normalize_keys(odds.copy())
        if "odds_source" not in odds.columns:
            odds["odds_source"] = path.stem
        odds["_priority"] = priority
        odds_frames.append(odds[required + ["odds_source", "_priority"]])
    if odds_frames:
        odds = pd.concat(odds_frames, ignore_index=True)
        for column in ["home_odds", "draw_odds", "away_odds"]:
            odds[column] = pd.to_numeric(odds[column], errors="coerce")
        odds["_complete"] = odds[["home_odds", "draw_odds", "away_odds"]].notna().all(axis=1)
        odds = odds.sort_values(["date", "home_team", "away_team", "_complete", "_priority"], ascending=[True, True, True, False, True])
        odds = odds.drop_duplicates(["date", "home_team", "away_team"], keep="first").drop(columns=["_complete", "_priority"])
        data = merge_optional_match_source(data, odds, ["home_odds", "draw_odds", "away_odds", "odds_source"])
    intel = load_optional_csv(PROJECT_ROOT / "reports" / "worldcup_daily_intel.csv")
    data = merge_optional_match_source(
        data,
        intel,
        [
            "home_injuries",
            "away_injuries",
            "home_suspensions",
            "away_suspensions",
            "home_expected_lineup",
            "away_expected_lineup",
            "home_coach_comments",
            "away_coach_comments",
            "injuries_home",
            "injuries_away",
            "suspensions_home",
            "suspensions_away",
            "expected_lineup_home",
            "expected_lineup_away",
            "coach_comments_home",
            "coach_comments_away",
            "rest_days_home",
            "rest_days_away",
            "intel_risk",
            "intel_updated_at",
            "confidence",
            "source_urls",
            "source_url",
        ],
    )
    return data


def standard_result_frame() -> pd.DataFrame:
    results = load_optional_csv(PROCESSED_DATA_DIR / "worldcup_results.csv")
    history = load_optional_csv(PROCESSED_DATA_DIR / "international_training_data.csv")
    frames = []
    if not history.empty:
        history = history.rename(columns={"competition": "stage"})
        frames.append(history)
    if not results.empty:
        frames.append(results)
    if not frames:
        return pd.DataFrame()
    data = pd.concat(frames, ignore_index=True, sort=False)
    required = ["date", "home_team", "away_team", "home_goals", "away_goals"]
    if not all(column in data.columns for column in required):
        return pd.DataFrame()
    data["date"] = pd.to_datetime(data["date"], errors="coerce")
    data["home_goals"] = pd.to_numeric(data["home_goals"], errors="coerce")
    data["away_goals"] = pd.to_numeric(data["away_goals"], errors="coerce")
    return data.dropna(subset=required)


def recent_score(results: pd.DataFrame, team: str, match_date: str) -> float:
    if results.empty:
        return 0.5
    match_dt = pd.to_datetime(match_date, errors="coerce")
    if pd.isna(match_dt):
        return 0.5
    team_results = results[
        (results["date"] < match_dt)
        & ((results["home_team"].astype(str) == str(team)) | (results["away_team"].astype(str) == str(team)))
    ].sort_values("date", ascending=False).head(5)
    if team_results.empty:
        return 0.5
    points = 0
    for row in team_results.itertuples(index=False):
        is_home = str(row.home_team) == str(team)
        gf = row.home_goals if is_home else row.away_goals
        ga = row.away_goals if is_home else row.home_goals
        points += 3 if gf > ga else 1 if gf == ga else 0
    return float(points) / float(len(team_results) * 3)


def add_recent_form(data: pd.DataFrame) -> pd.DataFrame:
    results = standard_result_frame()
    data = data.copy()
    data["home_recent_form_score"] = data.apply(lambda row: recent_score(results, row["home_team"], row["date"]), axis=1)
    data["away_recent_form_score"] = data.apply(lambda row: recent_score(results, row["away_team"], row["date"]), axis=1)
    return data


def write_adjustment_debug(output: pd.DataFrame) -> None:
    columns = [
        "match_key",
        "base_H",
        "base_D",
        "base_A",
        "odds_weight_used",
        "form_adjustment",
        "injury_adjustment",
        "fatigue_adjustment",
        "final_adjusted_H",
        "final_adjusted_D",
        "final_adjusted_A",
        "reasons",
    ]
    rows = []
    for row in output.itertuples(index=False):
        rows.append(
            {
                "match_key": f"{row.date}|{row.home_team}|{row.away_team}",
                "base_H": getattr(row, "model_H", pd.NA),
                "base_D": getattr(row, "model_D", pd.NA),
                "base_A": getattr(row, "model_A", pd.NA),
                "odds_weight_used": 0.35 if pd.notna(getattr(row, "home_odds", pd.NA)) else 0.0,
                "form_adjustment": "home_recent_form_score/away_recent_form_score",
                "injury_adjustment": getattr(row, "adjustment_score_home", 0) + getattr(row, "adjustment_score_away", 0),
                "fatigue_adjustment": "included_in_reasons",
                "final_adjusted_H": getattr(row, "adjusted_H", pd.NA),
                "final_adjusted_D": getattr(row, "adjusted_D", pd.NA),
                "final_adjusted_A": getattr(row, "adjusted_A", pd.NA),
                "reasons": getattr(row, "adjustment_reasons", ""),
            }
        )
    ADJUSTMENT_DEBUG_CSV.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=columns).to_csv(ADJUSTMENT_DEBUG_CSV, index=False, encoding="utf-8")


def model_only_risk(row: pd.Series) -> str:
    model_confidence = max(float(row["model_H"]), float(row["model_D"]), float(row["model_A"]))
    poisson_diff = abs(float(row["poisson_away_xg"]) - float(row["poisson_home_xg"]))
    if model_confidence < 0.42 or poisson_diff < 0.25:
        return "HIGH"
    if model_confidence < 0.50 or poisson_diff < 0.55:
        return "MEDIUM"
    return "LOW"


def model_only_action(row: pd.Series, upset_risk: str, draw_risk_level: str) -> str:
    model_confidence = max(float(row["model_H"]), float(row["model_D"]), float(row["model_A"]))
    if upset_risk == "HIGH" or draw_risk_level == "HIGH" or model_confidence < 0.40:
        return "PASS"
    return "WATCH"


def build_model_only_data() -> pd.DataFrame:
    fixtures = load_required_csv(
        latest_fixtures_path(),
        ["date", "home_team", "away_team"],
        "World Cup fixtures",
    )
    model = load_required_csv(
        PROCESSED_DATA_DIR / "worldcup_model_predictions.csv",
        ["date", "home_team", "away_team", "model_H", "model_D", "model_A"],
        "World Cup model predictions",
    )
    poisson = load_required_csv(
        PROCESSED_DATA_DIR / "worldcup_poisson_predictions.csv",
        ["date", "home_team", "away_team", "poisson_home_xg", "poisson_away_xg", "poisson_top_scores"],
        "World Cup poisson predictions",
    )
    merge_keys = choose_fixture_merge_keys(fixtures, model, poisson)
    data = fixtures.merge(model, on=merge_keys, how="left", suffixes=("", "_model"))
    data = data.merge(poisson, on=merge_keys, how="left", suffixes=("", "_poisson"))
    required_values = ["model_H", "model_D", "model_A", "poisson_home_xg", "poisson_away_xg"]
    waiting_mask = data.apply(is_waiting_for_teams, axis=1)
    if data.loc[~waiting_mask, required_values].isna().any(axis=1).any():
        missing_count = int(data.loc[~waiting_mask, required_values].isna().any(axis=1).sum())
        raise ValueError(f"model-only 資料仍有 {missing_count} 場缺少 model 或 poisson 預測。")
    context_columns = [column for column in ["date", "group", "stage", "home_team", "away_team"] if column in data.columns]
    context = build_group_context(data[context_columns].copy())
    data = pd.concat([data.reset_index(drop=True), context.reset_index(drop=True)], axis=1)
    data["neutral_venue"] = True
    data["poisson_diff"] = data["poisson_away_xg"] - data["poisson_home_xg"]
    data = add_odds_and_intel(data)
    data = add_recent_form(data)
    return data


def write_draw_risk_reports(output: pd.DataFrame) -> None:
    DRAW_RISK_SUMMARY_CSV.parent.mkdir(parents=True, exist_ok=True)
    if output.empty:
        pd.DataFrame(columns=["draw_risk_level", "count", "avg_model_D", "avg_draw_risk_score"]).to_csv(DRAW_RISK_SUMMARY_CSV, index=False, encoding="utf-8")
        pd.DataFrame().to_csv(HIGH_DRAW_RISK_CSV, index=False, encoding="utf-8")
        return
    summary = (
        output.groupby("draw_risk_level", as_index=False)
        .agg(count=("draw_risk_level", "size"), avg_model_D=("model_D", "mean"), avg_draw_risk_score=("draw_risk_score", "mean"))
        .sort_values("draw_risk_level")
    )
    summary.to_csv(DRAW_RISK_SUMMARY_CSV, index=False, encoding="utf-8")
    output[output["draw_risk_level"] == "HIGH"].copy().to_csv(HIGH_DRAW_RISK_CSV, index=False, encoding="utf-8")


def normalize_team_key(value) -> str:
    return "" if pd.isna(value) else str(value).strip().casefold()


def snapshot_probability(row: pd.Series, side: str):
    adjusted = pd.to_numeric(row.get(f"adjusted_{side}"), errors="coerce")
    if pd.notna(adjusted):
        return adjusted
    return pd.to_numeric(row.get(f"model_{side}"), errors="coerce")


def snapshot_prediction_side(row: pd.Series) -> str:
    values = {side: snapshot_probability(row, side) for side in ["H", "D", "A"]}
    values = {side: (float(value) if pd.notna(value) else -1.0) for side, value in values.items()}
    return max(values, key=values.get)


def snapshot_match_key(row: pd.Series) -> str:
    event_id = str(row.get("event_id", row.get("match_id", ""))).strip()
    if event_id:
        return event_id
    return f"{row.get('date', '')}|{normalize_team_key(row.get('home_team', ''))}|{normalize_team_key(row.get('away_team', ''))}"


def append_prediction_snapshots(output: pd.DataFrame, snapshot_path: Path = PREDICTION_SNAPSHOTS_CSV) -> None:
    if output.empty:
        return
    today = pd.Timestamp.today().normalize()
    created_at = datetime.now(timezone.utc).isoformat()
    rows = []
    for _, row in output.iterrows():
        match_date = pd.to_datetime(row.get("date"), errors="coerce")
        if pd.isna(match_date) or match_date.normalize() < today:
            continue
        if is_waiting_for_teams(row):
            continue
        event_id = str(row.get("event_id", row.get("match_id", ""))).strip()
        probs = {side: snapshot_probability(row, side) for side in ["H", "D", "A"]}
        valid_probs = [float(value) for value in probs.values() if pd.notna(value)]
        if not valid_probs:
            continue
        rows.append(
            {
                "event_id": event_id,
                "match_key": snapshot_match_key(row),
                "prediction_date": today.strftime("%Y-%m-%d"),
                "match_date": match_date.strftime("%Y-%m-%d"),
                "home_team": row.get("home_team", ""),
                "away_team": row.get("away_team", ""),
                "predicted_H": probs["H"],
                "predicted_D": probs["D"],
                "predicted_A": probs["A"],
                "predicted_result": snapshot_prediction_side(row),
                "confidence": max(valid_probs),
                "action": row.get("recommended_action", ""),
                "odds_status": row.get("odds_status", ""),
                "intel_risk": row.get("intel_risk", ""),
                "draw_risk": row.get("draw_risk_level", ""),
                "upset_risk": row.get("upset_risk", ""),
                "created_at": created_at,
            }
        )
    if not rows:
        return
    snapshots = pd.DataFrame(rows)
    snapshot_path.parent.mkdir(parents=True, exist_ok=True)
    existing = load_optional_csv(snapshot_path)
    combined = pd.concat([existing, snapshots], ignore_index=True, sort=False) if not existing.empty else snapshots
    combined["match_date"] = pd.to_datetime(combined["match_date"], errors="coerce").dt.strftime("%Y-%m-%d")
    combined["prediction_date"] = pd.to_datetime(combined["prediction_date"], errors="coerce").dt.strftime("%Y-%m-%d")
    combined["_event_id"] = combined["event_id"].fillna("").astype(str).str.strip()
    combined["_match_key"] = combined["match_key"].fillna("").astype(str).str.strip()
    combined["_dedup_key"] = combined["_event_id"].where(combined["_event_id"].ne(""), combined["_match_key"])
    combined = combined.sort_values(["match_date", "prediction_date", "created_at"]).drop_duplicates("_dedup_key", keep="last")
    combined.drop(columns=["_event_id", "_match_key", "_dedup_key"], errors="ignore").to_csv(snapshot_path, index=False, encoding="utf-8")
    print(f"prediction snapshots appended: {snapshot_path} rows={len(snapshots)} total={len(combined)}")


def run_model_only_agents(output_csv: Path = MODEL_ONLY_OUTPUT_CSV) -> pd.DataFrame | None:
    try:
        data = build_model_only_data()
    except Exception as exc:
        print(f"無法產生 no-odds model-only predictions: {exc}")
        return None

    draw_agent = DrawRiskAgent()
    league_agent = LeagueReferenceAgent(PROJECT_ROOT / "reports")
    adjustment_agent = PredictionAdjustmentAgent()
    rows = []
    for _, row in data.iterrows():
        common = {
            "date": row["date"],
            "group": row.get("group", ""),
            "stage": row.get("stage", ""),
            "round": row.get("round", ""),
            "match_id": row.get("match_id", ""),
            "home_team": row["home_team"],
            "away_team": row["away_team"],
            "home_slot": row.get("home_slot", ""),
            "away_slot": row.get("away_slot", ""),
            "status": row.get("status", ""),
            "market_type": row.get("market_type", "1x2"),
        }
        if is_waiting_for_teams(row):
            rows.append(
                {
                    **common,
                    "market_H": pd.NA,
                    "market_D": pd.NA,
                    "market_A": pd.NA,
                    "home_odds": row.get("home_odds", pd.NA),
                    "draw_odds": row.get("draw_odds", pd.NA),
                    "away_odds": row.get("away_odds", pd.NA),
                    "odds_source": row.get("odds_source", ""),
                    "model_H": pd.NA,
                    "model_D": pd.NA,
                    "model_A": pd.NA,
                    "adjusted_H": pd.NA,
                    "adjusted_D": pd.NA,
                    "adjusted_A": pd.NA,
                    "adjustment_score_home": 0.0,
                    "adjustment_score_away": 0.0,
                    "adjustment_reasons": "Waiting for teams",
                    "final_prediction_source": "weighted_adjustment",
                    "poisson_home_xg": pd.NA,
                    "poisson_away_xg": pd.NA,
                    "poisson_top_scores": "",
                    "upset_risk": "UNKNOWN",
                    "draw_risk_level": "UNKNOWN",
                    "draw_risk_score": 0,
                    "draw_risk_reasons": "Waiting for teams",
                    "league_reference_available": False,
                    "league_risk_score": 0,
                    "league_reference_level": "UNKNOWN",
                    "league_reference_reasons": "Waiting for teams",
                    "value_side": pd.NA,
                    "edge": pd.NA,
                    "recommended_action": "WATCH",
                    "recommended_stake": 0.0,
                    "reason": "Waiting for teams; no model prediction is generated until both knockout teams are confirmed.",
                }
            )
            continue
        model_probs = {"H": float(row["model_H"]), "D": float(row["model_D"]), "A": float(row["model_A"])}
        model_pick = max(model_probs, key=model_probs.get)
        adjustment = adjustment_agent.run(row)
        upset_risk = model_only_risk(row)
        draw_risk = draw_agent.run(row)
        league_reference = league_agent.run(row, market=None, poisson=None, draw_risk=draw_risk)
        action = model_only_action(row, upset_risk, draw_risk["draw_risk_level"])
        rows.append(
            {
                **common,
                "market_H": pd.NA,
                "market_D": pd.NA,
                "market_A": pd.NA,
                "home_odds": row.get("home_odds", pd.NA),
                "draw_odds": row.get("draw_odds", pd.NA),
                "away_odds": row.get("away_odds", pd.NA),
                "odds_source": row.get("odds_source", ""),
                "model_H": row["model_H"],
                "model_D": row["model_D"],
                "model_A": row["model_A"],
                "adjusted_H": adjustment["adjusted_H"],
                "adjusted_D": adjustment["adjusted_D"],
                "adjusted_A": adjustment["adjusted_A"],
                "adjustment_score_home": adjustment["adjustment_score_home"],
                "adjustment_score_away": adjustment["adjustment_score_away"],
                "adjustment_reasons": adjustment["adjustment_reasons"],
                "final_prediction_source": "weighted_adjustment",
                "poisson_home_xg": row["poisson_home_xg"],
                "poisson_away_xg": row["poisson_away_xg"],
                "poisson_top_scores": parse_poisson_scores(row["poisson_top_scores"]),
                "upset_risk": upset_risk,
                "draw_risk_level": draw_risk["draw_risk_level"],
                "draw_risk_score": draw_risk["draw_risk_score"],
                "draw_risk_reasons": "; ".join(draw_risk["draw_risk_reasons"]),
                "league_reference_available": league_reference["league_reference_available"],
                "league_risk_score": league_reference["league_risk_score"],
                "league_reference_level": league_reference["league_reference_level"],
                "league_reference_reasons": "; ".join(league_reference["league_reference_reasons"]),
                "value_side": model_pick,
                "edge": pd.NA,
                "recommended_action": action,
                "recommended_stake": 0.0,
                "reason": model_only_reason(row),
            }
        )
    output = pd.DataFrame(rows)
    invalid_actions = set(output["recommended_action"]) - {"WATCH", "PASS"}
    if invalid_actions:
        raise ValueError(f"no-odds mode 不允許投注動作: {invalid_actions}")
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_csv, index=False, encoding="utf-8")
    if output_csv != MODEL_ONLY_OUTPUT_CSV:
        MODEL_ONLY_OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
        output.to_csv(MODEL_ONLY_OUTPUT_CSV, index=False, encoding="utf-8")
    write_adjustment_debug(output)
    write_draw_risk_reports(output)
    append_prediction_snapshots(output)
    print(f"已輸出 no-odds model-only predictions: {output_csv}")
    print(f"輸出筆數: {len(output)}")
    print(f"已輸出 draw risk summary: {DRAW_RISK_SUMMARY_CSV}")
    print(f"已輸出 high draw risk matches: {HIGH_DRAW_RISK_CSV}")
    print(NO_ODDS_REASON)
    return output


def run_agents(features_path: Path, output_csv: Path, report_dir: Path):
    agents = {
        "data": DataAgent(),
        "market": MarketAgent(),
        "poisson": PoissonAgent(),
        "context": ContextAgent(),
        "risk": RiskAgent(),
        "draw_risk": DrawRiskAgent(),
        "league_reference": LeagueReferenceAgent(PROJECT_ROOT / "reports"),
        "value": ValueAgent(),
        "final": FinalBettingAgent(),
        "adjustment": PredictionAdjustmentAgent(),
        "report": ReportAgent(),
    }
    data_result = agents["data"].run(features_path)
    if not data_result["ok"]:
        for error in data_result["errors"]:
            print(error)
        print("缺少 odds 或完整 World Cup features，改用 no-odds model-only mode。")
        return run_model_only_agents(output_csv)
    data = add_recent_form(add_odds_and_intel(data_result["data"]))
    predictions = [build_prediction(row, agents) for _, row in data.iterrows()]
    output = agents["report"].run(predictions, output_csv, report_dir)
    write_adjustment_debug(output)
    write_draw_risk_reports(output)
    append_prediction_snapshots(output)
    print(f"已輸出 World Cup betting predictions: {output_csv}")
    print(f"已輸出逐場報告資料夾: {report_dir}")
    print(f"已輸出 draw risk summary: {DRAW_RISK_SUMMARY_CSV}")
    print(f"已輸出 high draw risk matches: {HIGH_DRAW_RISK_CSV}")
    print(f"輸出筆數: {len(output)}")
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=Path, default=WORLDCUP_FEATURES_PATH)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    parser.add_argument("--model-only", action="store_true")
    args = parser.parse_args()
    if args.model_only:
        run_model_only_agents()
    else:
        run_agents(args.features, args.output_csv, args.report_dir)


if __name__ == "__main__":
    main()
