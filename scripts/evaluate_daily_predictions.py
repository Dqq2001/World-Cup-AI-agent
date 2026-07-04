import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PREDICTIONS_PATH = PROJECT_ROOT / "reports" / "worldcup_betting_predictions.csv"
RESULTS_PATH = PROJECT_ROOT / "data" / "processed" / "worldcup_results.csv"
DAILY_OUTPUT = PROJECT_ROOT / "reports" / "daily_prediction_vs_result.csv"
SUMMARY_OUTPUT = PROJECT_ROOT / "reports" / "daily_prediction_summary.csv"
FEEDBACK_OUTPUT = PROJECT_ROOT / "data" / "processed" / "worldcup_prediction_feedback.csv"
ANALYSIS_OUTPUT = PROJECT_ROOT / "reports" / "daily_prediction_error_analysis.csv"
PREDICTION_ACCURACY_OUTPUT = PROJECT_ROOT / "reports" / "error_pattern_analysis.csv"
RISK_SIGNAL_OUTPUT = PROJECT_ROOT / "reports" / "risk_signal_accuracy.csv"

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


def probability(row: pd.Series, adjusted: bool, side: str) -> float:
    prefix = "adjusted" if adjusted else "model"
    value = pd.to_numeric(row.get(f"{prefix}_{side}"), errors="coerce")
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


def evaluate(predictions: pd.DataFrame, results: pd.DataFrame) -> pd.DataFrame:
    if predictions.empty or results.empty:
        return pd.DataFrame()
    predictions = add_keys(predictions)
    results = add_keys(results)
    results = results[results.get("status", "").astype(str).str.lower().isin(["completed", "complete", "final"])]

    event_results = results[results["_event_key"].ne("")].drop_duplicates("_event_key", keep="last")
    team_results = results.drop_duplicates("_team_key", keep="last")

    rows = []
    for _, pred in predictions.iterrows():
        result = pd.Series(dtype=object)
        if pred["_event_key"] and pred["_event_key"] in set(event_results["_event_key"]):
            result = event_results[event_results["_event_key"] == pred["_event_key"]].iloc[0]
        else:
            matched = team_results[team_results["_team_key"] == pred["_team_key"]]
            if not matched.empty:
                result = matched.iloc[0]
        if result.empty:
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
                "_match_key": pred["_match_key"],
                "_features_snapshot": json.dumps(pred.drop(labels=[col for col in pred.index if col.startswith("_")]).to_dict(), ensure_ascii=False, default=str),
            }
        )
    return pd.DataFrame(rows)


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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, default=PREDICTIONS_PATH)
    parser.add_argument("--results", type=Path, default=RESULTS_PATH)
    args = parser.parse_args()

    predictions = load_csv(args.predictions)
    results = load_csv(args.results)
    evaluated = evaluate(predictions, results)

    DAILY_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    output = evaluated.drop(columns=["_match_key", "_features_snapshot"], errors="ignore")
    output.to_csv(DAILY_OUTPUT, index=False, encoding="utf-8")
    summarize(evaluated).to_csv(SUMMARY_OUTPUT, index=False, encoding="utf-8")
    write_feedback(evaluated)
    write_error_analysis(evaluated)

    print(f"evaluated matches: {len(evaluated)}")
    print(f"output: {DAILY_OUTPUT}")
    print(f"summary: {SUMMARY_OUTPUT}")
    print(f"feedback: {FEEDBACK_OUTPUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
