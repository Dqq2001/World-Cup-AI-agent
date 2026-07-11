import argparse
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SNAPSHOTS_PATH = PROJECT_ROOT / "data" / "processed" / "worldcup_prediction_snapshots.csv"
PREDICTIONS_PATH = PROJECT_ROOT / "reports" / "worldcup_betting_predictions.csv"
RESULTS_PATH = PROJECT_ROOT / "data" / "processed" / "worldcup_results.csv"
DAILY_OUTPUT = PROJECT_ROOT / "reports" / "daily_prediction_vs_result.csv"
SUMMARY_OUTPUT = PROJECT_ROOT / "reports" / "daily_prediction_summary.csv"
FEEDBACK_OUTPUT = PROJECT_ROOT / "data" / "processed" / "worldcup_prediction_feedback.csv"
ANALYSIS_OUTPUT = PROJECT_ROOT / "reports" / "daily_prediction_error_analysis.csv"
PREDICTION_ACCURACY_OUTPUT = PROJECT_ROOT / "reports" / "error_pattern_analysis.csv"
RISK_SIGNAL_OUTPUT = PROJECT_ROOT / "reports" / "risk_signal_accuracy.csv"
REVIEW_DATE_DEBUG_OUTPUT = PROJECT_ROOT / "reports" / "review_date_debug.csv"
MATCH_DEBUG_OUTPUT = PROJECT_ROOT / "reports" / "prediction_review_match_debug.csv"
UNMATCHED_OUTPUT = PROJECT_ROOT / "reports" / "prediction_review_unmatched.csv"
COVERAGE_DEBUG_OUTPUT = PROJECT_ROOT / "reports" / "prediction_snapshot_coverage_debug.csv"

TEAM_ALIASES = {
    "Bosnia-Herzegovina": "Bosnia and Herzegovina",
    "Bosnia & Herzegovina": "Bosnia and Herzegovina",
    "BIH": "Bosnia and Herzegovina",
    "DR Congo": "Congo DR",
    "Democratic Republic of Congo": "Congo DR",
    "Côte d'Ivoire": "Ivory Coast",
    "Cote d'Ivoire": "Ivory Coast",
    "USA": "United States",
    "USMNT": "United States",
}


def normalize_team(value) -> str:
    text = "" if pd.isna(value) else str(value).strip()
    return TEAM_ALIASES.get(text, text).casefold()


def normalize_date(value) -> str:
    return pd.to_datetime(value, errors="coerce").strftime("%Y-%m-%d")


def load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path, encoding="utf-8")
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def add_keys(data: pd.DataFrame) -> pd.DataFrame:
    data = data.copy()
    if "date" not in data.columns and "match_date" in data.columns:
        data["date"] = data["match_date"]
    data["date"] = pd.to_datetime(data["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    data["_team_key"] = data["date"].astype(str) + "|" + data["home_team"].map(normalize_team) + "|" + data["away_team"].map(normalize_team)
    if "event_id" in data.columns:
        data["_event_key"] = data["event_id"].fillna("").astype(str).str.strip()
    elif "match_id" in data.columns:
        data["_event_key"] = data["match_id"].fillna("").astype(str).str.strip()
    else:
        data["_event_key"] = ""
    data["_match_key"] = data["_event_key"].where(data["_event_key"].ne(""), data["_team_key"])
    return data


def resolved_match_mask(data: pd.DataFrame) -> pd.Series:
    if data.empty:
        return pd.Series(dtype=bool)
    home = data.get("home_team", pd.Series("", index=data.index)).fillna("").astype(str).str.strip().str.upper()
    away = data.get("away_team", pd.Series("", index=data.index)).fillna("").astype(str).str.strip().str.upper()
    status = data.get("status", pd.Series("", index=data.index)).fillna("").astype(str).str.strip().str.lower()
    return home.ne("TBD") & away.ne("TBD") & status.ne("waiting_for_teams")


def load_prediction_source(path: Path | None = None) -> tuple[pd.DataFrame, str]:
    if path is not None:
        data = load_csv(path)
        is_snapshot = path.resolve() == SNAPSHOTS_PATH.resolve()
        return normalize_snapshot_predictions(data) if is_snapshot else data, str(path)
    snapshots = load_csv(SNAPSHOTS_PATH)
    if not snapshots.empty:
        return normalize_snapshot_predictions(snapshots), str(SNAPSHOTS_PATH)
    return load_csv(PREDICTIONS_PATH), str(PREDICTIONS_PATH)


def normalize_snapshot_predictions(data: pd.DataFrame) -> pd.DataFrame:
    if data.empty:
        return data
    output = data.copy()
    if "match_date" in output.columns:
        output["date"] = output["match_date"]
    if "predicted_H" in output.columns:
        output["model_H"] = output["predicted_H"]
        output["adjusted_H"] = output["predicted_H"]
    if "predicted_D" in output.columns:
        output["model_D"] = output["predicted_D"]
        output["adjusted_D"] = output["predicted_D"]
    if "predicted_A" in output.columns:
        output["model_A"] = output["predicted_A"]
        output["adjusted_A"] = output["predicted_A"]
    if "action" in output.columns:
        output["recommended_action"] = output["action"]
    if "draw_risk" in output.columns:
        output["draw_risk_level"] = output["draw_risk"]
    if "event_id" in output.columns and "match_id" not in output.columns:
        output["match_id"] = output["event_id"]
    return output


def probability(row: pd.Series, adjusted: bool, side: str) -> float:
    prefix = "adjusted" if adjusted else "model"
    value = pd.to_numeric(row.get(f"{prefix}_{side}"), errors="coerce")
    if pd.isna(value):
        value = pd.to_numeric(row.get(f"predicted_{side}"), errors="coerce")
    if pd.isna(value) and adjusted:
        value = pd.to_numeric(row.get(f"decision_{side}"), errors="coerce")
    if pd.isna(value):
        value = pd.to_numeric(row.get(f"model_{side}"), errors="coerce")
    return float(value) if pd.notna(value) else 0.0


def actual_result(row: pd.Series) -> str:
    home = pd.to_numeric(row.get("home_goals"), errors="coerce")
    away = pd.to_numeric(row.get("away_goals"), errors="coerce")
    if pd.isna(home) or pd.isna(away):
        return ""
    if home > away:
        return "H"
    if home < away:
        return "A"
    return "D"


def predicted_result(row: pd.Series) -> str:
    probs = {side: probability(row, adjusted=True, side=side) for side in ["H", "D", "A"]}
    return max(probs, key=probs.get)


def confidence(row: pd.Series) -> float:
    return max(probability(row, adjusted=True, side=side) for side in ["H", "D", "A"])


def selected_odds(row: pd.Series, side: str) -> float | None:
    columns = {"H": "home_odds", "D": "draw_odds", "A": "away_odds"}
    value = pd.to_numeric(row.get(columns[side]), errors="coerce")
    return float(value) if pd.notna(value) and value > 1 else None


def profit_if_bet(row: pd.Series, pred: str, actual: str) -> float:
    action = str(row.get("recommended_action", "")).upper()
    if action not in {"BET", "SMALL_BET"}:
        return 0.0
    stake = 1.0 if action == "BET" else 0.5
    odds = selected_odds(row, pred)
    if odds is None:
        return 0.0
    return round((odds - 1) * stake, 4) if pred == actual else round(-stake, 4)


def error_type(row: pd.Series, pred: str, actual: str, correct: bool) -> str:
    action = str(row.get("recommended_action", "")).upper()
    conf = confidence(row)
    market_values = {
        "H": pd.to_numeric(row.get("market_H"), errors="coerce"),
        "D": pd.to_numeric(row.get("market_D"), errors="coerce"),
        "A": pd.to_numeric(row.get("market_A"), errors="coerce"),
    }
    favorite = max(market_values, key=lambda side: -1 if pd.isna(market_values[side]) else market_values[side])

    if pred == "D" and actual == "D":
        return "draw_correct"
    if pred != "D" and actual == "D":
        return "draw_missed"
    if actual != favorite:
        return "underdog_hit" if pred == actual else "favorite_upset"
    if correct and pred == favorite:
        return "favorite_correct"
    if conf >= 0.6 and not correct:
        return "high_confidence_wrong"
    if action == "PASS" and not correct:
        return "pass_correct"
    if action == "PASS" and correct:
        return "pass_missed_value"
    return "correct" if correct else "wrong"


def market_favorite(row: pd.Series) -> str:
    market_values = {
        "H": pd.to_numeric(row.get("market_H"), errors="coerce"),
        "D": pd.to_numeric(row.get("market_D"), errors="coerce"),
        "A": pd.to_numeric(row.get("market_A"), errors="coerce"),
    }
    return max(market_values, key=lambda side: -1 if pd.isna(market_values[side]) else market_values[side])


def model_notes(row: pd.Series, err: str) -> str:
    pieces = [
        f"source={row.get('final_prediction_source', 'unknown')}",
        f"draw_risk={row.get('draw_risk_level', 'unknown')}",
        f"intel_risk={row.get('intel_risk', 'unknown')}",
        f"odds_status={row.get('odds_status', 'unknown')}",
        f"error_type={err}",
    ]
    return "; ".join(pieces)


def evaluate(predictions: pd.DataFrame, results: pd.DataFrame, as_of_date: str | None = None) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if predictions.empty or results.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    predictions = add_keys(predictions)
    results = add_keys(results)
    predictions = predictions[resolved_match_mask(predictions)].copy()
    results = results[resolved_match_mask(results)].copy()
    results = results[results.get("status", "").astype(str).str.lower().isin(["completed", "complete", "final"])]
    if as_of_date:
        predictions = predictions[predictions["date"] <= as_of_date]
        results = results[results["date"] <= as_of_date]

    event_predictions = predictions[predictions["_event_key"].ne("")].drop_duplicates("_event_key", keep="last")
    team_predictions = predictions.drop_duplicates("_team_key", keep="last")
    rows = []
    debug_rows = []
    unmatched_rows = []
    for _, result in results.iterrows():
        pred = pd.Series(dtype=object)
        matched_by = ""
        if result["_event_key"] and result["_event_key"] in set(event_predictions["_event_key"]):
            pred = event_predictions[event_predictions["_event_key"] == result["_event_key"]].iloc[0]
            matched_by = "event_id"
        else:
            matched = team_predictions[team_predictions["_team_key"] == result["_team_key"]]
            if not matched.empty:
                pred = matched.iloc[0]
                matched_by = "match_date_team"
        if pred.empty:
            unmatched = result.drop(labels=[col for col in result.index if col.startswith("_")], errors="ignore").to_dict()
            unmatched["reason"] = "prediction_snapshot_missing"
            unmatched_rows.append(unmatched)
            debug_rows.append(
                {
                    "match_date": result.get("date", ""),
                    "home_team": result.get("home_team", ""),
                    "away_team": result.get("away_team", ""),
                    "event_id": result.get("_event_key", ""),
                    "prediction_snapshot_found": False,
                    "result_found": True,
                    "review_found": False,
                    "matched_by": "",
                    "reason": "prediction_snapshot_missing",
                }
            )
            continue

        actual = actual_result(result)
        if not actual:
            continue
        pred_side = predicted_result(pred)
        is_correct = pred_side == actual
        err = error_type(pred, pred_side, actual, is_correct)
        favorite = market_favorite(pred)
        draw_risk_level = str(pred.get("draw_risk_level", "")).upper()
        upset_risk = str(pred.get("upset_risk", "")).upper()
        rows.append(
            {
                "date": pred["date"],
                "home_team": pred["home_team"],
                "away_team": pred["away_team"],
                "predicted_H": probability(pred, adjusted=False, side="H"),
                "predicted_D": probability(pred, adjusted=False, side="D"),
                "predicted_A": probability(pred, adjusted=False, side="A"),
                "adjusted_H": probability(pred, adjusted=True, side="H"),
                "adjusted_D": probability(pred, adjusted=True, side="D"),
                "adjusted_A": probability(pred, adjusted=True, side="A"),
                "predicted_result": pred_side,
                "actual_result": actual,
                "home_score": int(float(result["home_goals"])),
                "away_score": int(float(result["away_goals"])),
                "correct": is_correct,
                "confidence": confidence(pred),
                "action": pred.get("recommended_action", ""),
                "odds_status": pred.get("odds_status", ""),
                "intel_risk": pred.get("intel_risk", ""),
                "draw_risk_level": pred.get("draw_risk_level", ""),
                "upset_risk": pred.get("upset_risk", ""),
                "draw_risk_hit": draw_risk_level in {"HIGH", "MEDIUM"} and actual == "D",
                "upset_risk_hit": upset_risk in {"HIGH", "MEDIUM"} and actual != favorite,
                "profit_if_bet": profit_if_bet(pred, pred_side, actual),
                "error_type": err,
                "model_notes": model_notes(pred, err),
                "_match_key": result["_match_key"],
                "_features_snapshot": json.dumps(pred.drop(labels=[col for col in pred.index if col.startswith("_")]).to_dict(), ensure_ascii=False, default=str),
            }
        )
        debug_rows.append(
            {
                "match_date": result.get("date", ""),
                "home_team": result.get("home_team", ""),
                "away_team": result.get("away_team", ""),
                "event_id": result.get("_event_key", ""),
                "prediction_snapshot_found": True,
                "result_found": True,
                "review_found": True,
                "matched_by": matched_by,
                "reason": "",
            }
        )
    return pd.DataFrame(rows), pd.DataFrame(debug_rows), pd.DataFrame(unmatched_rows)


def summarize(evaluated: pd.DataFrame) -> pd.DataFrame:
    if evaluated.empty:
        return pd.DataFrame(
            columns=[
                "date",
                "matches_evaluated",
                "accuracy",
                "bet_accuracy",
                "watch_accuracy",
                "pass_accuracy",
                "avg_confidence",
                "high_confidence_errors",
                "draw_miss_count",
                "upset_miss_count",
                "roi_if_bet",
            ]
        )
    rows = []
    for date_value, group in evaluated.groupby("date"):
        bet_group = group[group["action"].isin(["BET", "SMALL_BET"])]
        watch_group = group[group["action"] == "WATCH"]
        pass_group = group[group["action"] == "PASS"]
        stake = group["action"].map({"BET": 1.0, "SMALL_BET": 0.5}).fillna(0).sum()
        rows.append(
            {
                "date": date_value,
                "matches_evaluated": len(group),
                "accuracy": group["correct"].mean(),
                "bet_accuracy": bet_group["correct"].mean() if not bet_group.empty else pd.NA,
                "watch_accuracy": watch_group["correct"].mean() if not watch_group.empty else pd.NA,
                "pass_accuracy": pass_group["correct"].mean() if not pass_group.empty else pd.NA,
                "avg_confidence": group["confidence"].mean(),
                "high_confidence_errors": int(((group["confidence"] >= 0.6) & ~group["correct"]).sum()),
                "draw_miss_count": int((group["error_type"] == "draw_missed").sum()),
                "upset_miss_count": int((group["error_type"] == "favorite_upset").sum()),
                "roi_if_bet": group["profit_if_bet"].sum() / stake if stake > 0 else pd.NA,
            }
        )
    return pd.DataFrame(rows)


def write_feedback(evaluated: pd.DataFrame) -> None:
    FEEDBACK_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    if evaluated.empty:
        return
    existing = load_csv(FEEDBACK_OUTPUT)
    existing_keys = set(existing["match_key"].astype(str)) if not existing.empty and "match_key" in existing.columns else set()
    created_at = datetime.now(timezone.utc).isoformat()
    rows = []
    for _, row in evaluated.iterrows():
        match_key = row["_match_key"]
        if match_key in existing_keys:
            continue
        rows.append(
            {
                "match_key": match_key,
                "features_snapshot": row["_features_snapshot"],
                "prediction_snapshot": json.dumps(
                    {
                        "predicted_result": row["predicted_result"],
                        "predicted_H": row["predicted_H"],
                        "predicted_D": row["predicted_D"],
                        "predicted_A": row["predicted_A"],
                        "adjusted_H": row["adjusted_H"],
                        "adjusted_D": row["adjusted_D"],
                        "adjusted_A": row["adjusted_A"],
                        "confidence": row["confidence"],
                        "action": row["action"],
                    },
                    ensure_ascii=False,
                ),
                "actual_result": row["actual_result"],
                "error_type": row["error_type"],
                "created_at": created_at,
            }
        )
    if not rows:
        return
    output = pd.concat([existing, pd.DataFrame(rows)], ignore_index=True) if not existing.empty else pd.DataFrame(rows)
    output.drop_duplicates("match_key", keep="last").to_csv(FEEDBACK_OUTPUT, index=False, encoding="utf-8")


def write_error_analysis(evaluated: pd.DataFrame) -> None:
    prediction_rows = []
    risk_rows = []
    if not evaluated.empty:
        for column in ["odds_status", "intel_risk", "action", "draw_risk_level"]:
            output_type = "prediction_accuracy_by_draw_risk" if column == "draw_risk_level" else column
            for key, group in evaluated.groupby(column, dropna=False):
                prediction_rows.append(
                    {
                        "analysis_type": output_type,
                        "bucket": key,
                        "matches": len(group),
                        "accuracy": group["correct"].mean(),
                        "errors": int((~group["correct"]).sum()),
                    }
                )
        for column, hit_column in [("draw_risk_level", "draw_risk_hit"), ("upset_risk", "upset_risk_hit")]:
            if column not in evaluated.columns or hit_column not in evaluated.columns:
                continue
            for key, group in evaluated.groupby(column, dropna=False):
                risk_rows.append(
                    {
                        "analysis_type": column,
                        "bucket": key,
                        "matches": len(group),
                        "hit_rate": group[hit_column].mean(),
                        "hits": int(group[hit_column].sum()),
                    }
                )
    prediction_analysis = pd.DataFrame(prediction_rows)
    risk_analysis = pd.DataFrame(risk_rows)
    prediction_analysis.to_csv(PREDICTION_ACCURACY_OUTPUT, index=False, encoding="utf-8")
    risk_analysis.to_csv(RISK_SIGNAL_OUTPUT, index=False, encoding="utf-8")
    prediction_analysis.to_csv(ANALYSIS_OUTPUT, index=False, encoding="utf-8")


def max_date(data: pd.DataFrame) -> str:
    if data.empty or "date" not in data.columns:
        return ""
    values = pd.to_datetime(data["date"], errors="coerce").dropna()
    return values.max().strftime("%Y-%m-%d") if not values.empty else ""


def write_review_date_debug(
    system_date: str,
    requested_as_of_date: str,
    results: pd.DataFrame,
    evaluated: pd.DataFrame,
    summary: pd.DataFrame,
) -> None:
    as_of = pd.to_datetime(requested_as_of_date, errors="coerce")
    filter_end_date = as_of.strftime("%Y-%m-%d") if pd.notna(as_of) else requested_as_of_date
    filter_start_date = (as_of - timedelta(days=1)).strftime("%Y-%m-%d") if pd.notna(as_of) else ""
    REVIEW_DATE_DEBUG_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "system_date": system_date,
                "requested_as_of_date": requested_as_of_date,
                "results_max_date": max_date(results),
                "review_max_date": max_date(evaluated),
                "summary_max_date": max_date(summary),
                "filter_start_date": filter_start_date,
                "filter_end_date": filter_end_date,
            }
        ]
    ).to_csv(REVIEW_DATE_DEBUG_OUTPUT, index=False, encoding="utf-8")


def review_key(data: pd.DataFrame) -> pd.Series:
    if data.empty:
        return pd.Series(dtype=str)
    return (
        pd.to_datetime(data["date"], errors="coerce").dt.strftime("%Y-%m-%d")
        + "|"
        + data["home_team"].map(normalize_team)
        + "|"
        + data["away_team"].map(normalize_team)
    )


def append_new_reviews(existing: pd.DataFrame, evaluated: pd.DataFrame) -> pd.DataFrame:
    output = evaluated.drop(columns=["_match_key", "_features_snapshot"], errors="ignore")
    if not existing.empty:
        existing = existing[resolved_match_mask(existing)].copy()
    if not output.empty:
        output = output[resolved_match_mask(output)].copy()
    if existing.empty:
        return output
    if output.empty:
        return existing
    existing_keys = set(review_key(existing))
    new_rows = output[~review_key(output).isin(existing_keys)]
    if new_rows.empty:
        return existing
    return pd.concat([existing, new_rows], ignore_index=True, sort=False)


def write_review_debug(match_debug: pd.DataFrame, unmatched: pd.DataFrame, final_review: pd.DataFrame) -> None:
    debug_columns = [
        "match_date",
        "home_team",
        "away_team",
        "event_id",
        "prediction_snapshot_found",
        "result_found",
        "review_found",
        "matched_by",
        "reason",
    ]
    if not match_debug.empty and not final_review.empty:
        review_keys = set(review_key(final_review))
        debug_keys = (
            pd.to_datetime(match_debug["match_date"], errors="coerce").dt.strftime("%Y-%m-%d")
            + "|"
            + match_debug["home_team"].map(normalize_team)
            + "|"
            + match_debug["away_team"].map(normalize_team)
        )
        match_debug = match_debug.copy()
        match_debug["review_found"] = debug_keys.isin(review_keys)
    MATCH_DEBUG_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    match_debug.reindex(columns=debug_columns).to_csv(MATCH_DEBUG_OUTPUT, index=False, encoding="utf-8")
    match_debug.reindex(columns=debug_columns).to_csv(COVERAGE_DEBUG_OUTPUT, index=False, encoding="utf-8")
    unmatched.to_csv(UNMATCHED_OUTPUT, index=False, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, default=None)
    parser.add_argument("--results", type=Path, default=RESULTS_PATH)
    parser.add_argument("--as-of-date", default=pd.Timestamp.today().strftime("%Y-%m-%d"))
    args = parser.parse_args()

    parsed_as_of = pd.to_datetime(args.as_of_date, errors="coerce")
    as_of_date = parsed_as_of.strftime("%Y-%m-%d") if pd.notna(parsed_as_of) else pd.Timestamp.today().strftime("%Y-%m-%d")
    predictions, prediction_source = load_prediction_source(args.predictions)
    results = load_csv(args.results)
    evaluated, match_debug, unmatched = evaluate(predictions, results, as_of_date=as_of_date)

    DAILY_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    existing_review = load_csv(DAILY_OUTPUT)
    output = append_new_reviews(existing_review, evaluated)
    summary = summarize(output)
    output.to_csv(DAILY_OUTPUT, index=False, encoding="utf-8")
    summary.to_csv(SUMMARY_OUTPUT, index=False, encoding="utf-8")
    write_feedback(evaluated)
    write_error_analysis(output)
    write_review_date_debug(pd.Timestamp.today().strftime("%Y-%m-%d"), as_of_date, results, evaluated, summary)
    write_review_debug(match_debug, unmatched, output)

    print(f"as_of_date: {as_of_date}")
    print(f"prediction_source: {prediction_source}")
    print(f"evaluated matches: {len(evaluated)}")
    print(f"review rows: {len(output)}")
    print(f"unmatched completed results: {len(unmatched)}")
    print(f"output: {DAILY_OUTPUT}")
    print(f"summary: {SUMMARY_OUTPUT}")
    print(f"feedback: {FEEDBACK_OUTPUT}")
    print(f"debug: {REVIEW_DATE_DEBUG_OUTPUT}")
    print(f"coverage_debug: {COVERAGE_DEBUG_OUTPUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
