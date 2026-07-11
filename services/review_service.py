from __future__ import annotations

from datetime import timedelta
from pathlib import Path

import pandas as pd

from data_access.csv_store import read_csv_safe, write_csv_atomic
from data_access.paths import REPORTS_DIR
from services.script_runner import python_command, run_command


DAILY_PREDICTION_REVIEW_PATH = REPORTS_DIR / "daily_prediction_vs_result.csv"
REVIEW_SERVICE_DEBUG_PATH = REPORTS_DIR / "review_service_debug.csv"
REVIEW_DATE_DEBUG_PATH = REPORTS_DIR / "review_date_debug.csv"
REVIEW_UI_CONSISTENCY_DEBUG_PATH = REPORTS_DIR / "review_ui_consistency_debug.csv"


def refresh_prediction_review(as_of_date: str | None = None):
    requested_date = as_of_date or pd.Timestamp.today().strftime("%Y-%m-%d")
    results = run_command(
        python_command("scripts/fetch_worldcup_results.py", "--all-completed", "--force-refresh"),
        timeout=600,
    )
    if not results.ok:
        return results
    return run_command(
        python_command("scripts/evaluate_daily_predictions.py", "--as-of-date", requested_date),
        timeout=300,
    )


def file_updated_at(path: Path) -> str:
    if not path.exists():
        return ""
    return pd.Timestamp.fromtimestamp(path.stat().st_mtime).isoformat()


def max_date(data: pd.DataFrame) -> str:
    if data.empty or "date" not in data.columns:
        return ""
    values = pd.to_datetime(data["date"], errors="coerce").dropna()
    return values.max().strftime("%Y-%m-%d") if not values.empty else ""


def choose_display_review(review: pd.DataFrame, today: pd.Timestamp) -> pd.DataFrame:
    if review.empty or "date" not in review.columns:
        return pd.DataFrame()
    output = review.copy()
    output["date_dt"] = pd.to_datetime(output["date"], errors="coerce")
    start_date = today - timedelta(days=1)
    display = output[(output["date_dt"] >= start_date) & (output["date_dt"] <= today)].copy()
    if not display.empty:
        return display
    dated = output.dropna(subset=["date_dt"]).copy()
    if dated.empty:
        return output.iloc[0:0].copy()
    return dated.sort_values("date_dt", ascending=False).head(10).sort_values("date_dt").copy()


def remove_unresolved_matches(review: pd.DataFrame) -> pd.DataFrame:
    if review.empty:
        return review
    output = review.copy()
    home = output.get("home_team", pd.Series("", index=output.index)).fillna("").astype(str).str.strip().str.upper()
    away = output.get("away_team", pd.Series("", index=output.index)).fillna("").astype(str).str.strip().str.upper()
    status = output.get("status", pd.Series("", index=output.index)).fillna("").astype(str).str.strip().str.lower()
    return output[home.ne("TBD") & away.ne("TBD") & status.ne("waiting_for_teams")].copy()


def summarize_display_review(display_review: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "matches_evaluated": len(display_review),
                "accuracy": display_review["correct"].mean() if not display_review.empty and "correct" in display_review.columns else pd.NA,
            }
        ]
    )


def build_accuracy_patterns(display_review: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if not display_review.empty and "correct" in display_review.columns:
        for column in ["odds_status", "intel_risk", "action", "draw_risk_level"]:
            if column not in display_review.columns:
                continue
            output_type = "prediction_accuracy_by_draw_risk" if column == "draw_risk_level" else column
            for key, group in display_review.groupby(column, dropna=False):
                rows.append(
                    {
                        "analysis_type": output_type,
                        "bucket": key,
                        "matches": len(group),
                        "accuracy": group["correct"].mean(),
                        "errors": int((~group["correct"]).sum()),
                    }
                )
    return pd.DataFrame(rows, columns=["analysis_type", "bucket", "matches", "accuracy", "errors"])


def build_risk_signal_analysis(display_review: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for column, hit_column in [("draw_risk_level", "draw_risk_hit"), ("upset_risk", "upset_risk_hit")]:
        if display_review.empty or column not in display_review.columns or hit_column not in display_review.columns:
            continue
        for key, group in display_review.groupby(column, dropna=False):
            rows.append(
                {
                    "analysis_type": column,
                    "bucket": key,
                    "matches": len(group),
                    "hit_rate": group[hit_column].mean(),
                    "hits": int(group[hit_column].sum()),
                }
            )
    return pd.DataFrame(rows, columns=["analysis_type", "bucket", "matches", "hit_rate", "hits"])


def write_ui_consistency_debug(review: pd.DataFrame, display_review: pd.DataFrame, summary: pd.DataFrame, patterns: pd.DataFrame) -> None:
    summary_matches = int(summary["matches_evaluated"].iloc[0]) if not summary.empty and "matches_evaluated" in summary.columns else 0
    patterns_input_rows = len(display_review)
    if not display_review.empty and "date_dt" in display_review.columns:
        dates = display_review["date_dt"].dropna()
        display_min_date = dates.min().strftime("%Y-%m-%d") if not dates.empty else ""
        display_max_date = dates.max().strftime("%Y-%m-%d") if not dates.empty else ""
    else:
        display_min_date = ""
        display_max_date = ""
    debug = pd.DataFrame(
        [
            {
                "full_review_rows": len(review),
                "display_review_rows": len(display_review),
                "display_min_date": display_min_date,
                "display_max_date": display_max_date,
                "summary_matches_evaluated": summary_matches,
                "patterns_input_rows": patterns_input_rows,
                "completed_table_rows": len(display_review),
                "all_counts_match": summary_matches == patterns_input_rows == len(display_review),
            }
        ]
    )
    write_csv_atomic(debug, REVIEW_UI_CONSISTENCY_DEBUG_PATH)


def load_prediction_review(today: pd.Timestamp | None = None) -> dict[str, pd.DataFrame]:
    today = pd.to_datetime(today or pd.Timestamp.today()).normalize()
    review = read_csv_safe(DAILY_PREDICTION_REVIEW_PATH)

    if not review.empty and "date" in review.columns:
        review = review.copy()
        review["date_dt"] = pd.to_datetime(review["date"], errors="coerce")
        review = remove_unresolved_matches(review)
    display_review = choose_display_review(review, today)
    summary = summarize_display_review(display_review)
    accuracy_analysis = build_accuracy_patterns(display_review)
    risk_signal_accuracy = build_risk_signal_analysis(display_review)
    write_ui_consistency_debug(review, display_review, summary, accuracy_analysis)

    debug = pd.DataFrame(
        [
            {
                "source_dataframe": "prediction_review",
                "system_date": pd.Timestamp.today().strftime("%Y-%m-%d"),
                "requested_as_of_date": today.strftime("%Y-%m-%d"),
                "review_rows": len(review),
                "recent_rows": len(display_review),
                "summary_rows": len(summary),
                "summary_recent_rows": len(summary),
                "accuracy_rows": len(accuracy_analysis),
                "risk_signal_rows": len(risk_signal_accuracy),
                "review_max_date": max_date(review),
                "summary_max_date": max_date(summary),
            }
        ]
    )
    write_csv_atomic(debug, REVIEW_SERVICE_DEBUG_PATH)
    return {
        "review": review,
        "display_review": display_review,
        "recent": display_review,
        "summary": summary,
        "accuracy_analysis": accuracy_analysis,
        "risk_signal_accuracy": risk_signal_accuracy,
        "last_updated": file_updated_at(DAILY_PREDICTION_REVIEW_PATH),
        "date_debug": read_csv_safe(REVIEW_DATE_DEBUG_PATH),
    }
