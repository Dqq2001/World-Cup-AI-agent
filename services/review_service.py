from __future__ import annotations

from datetime import timedelta
from pathlib import Path

import pandas as pd

from data_access.csv_store import read_csv_safe, write_csv_atomic
from data_access.paths import REPORTS_DIR
from services.script_runner import python_command, run_command


DAILY_PREDICTION_REVIEW_PATH = REPORTS_DIR / "daily_prediction_vs_result.csv"
DAILY_PREDICTION_SUMMARY_PATH = REPORTS_DIR / "daily_prediction_summary.csv"
PREDICTION_ACCURACY_ANALYSIS_PATH = REPORTS_DIR / "error_pattern_analysis.csv"
RISK_SIGNAL_ACCURACY_PATH = REPORTS_DIR / "risk_signal_accuracy.csv"
REVIEW_SERVICE_DEBUG_PATH = REPORTS_DIR / "review_service_debug.csv"
REVIEW_DATE_DEBUG_PATH = REPORTS_DIR / "review_date_debug.csv"


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


def load_prediction_review(today: pd.Timestamp | None = None) -> dict[str, pd.DataFrame]:
    today = today or pd.Timestamp.today().normalize()
    today = pd.to_datetime(today).normalize()
    review = read_csv_safe(DAILY_PREDICTION_REVIEW_PATH)
    summary = read_csv_safe(DAILY_PREDICTION_SUMMARY_PATH)
    accuracy_analysis = read_csv_safe(PREDICTION_ACCURACY_ANALYSIS_PATH)
    risk_signal_accuracy = read_csv_safe(RISK_SIGNAL_ACCURACY_PATH)

    recent = pd.DataFrame()
    if not review.empty and "date" in review.columns:
        review = review.copy()
        review["date_dt"] = pd.to_datetime(review["date"], errors="coerce")
        start_date = today - timedelta(days=1)
        recent = review[(review["date_dt"] >= start_date) & (review["date_dt"] <= today)]

    debug = pd.DataFrame(
        [
            {
                "source_dataframe": "prediction_review",
                "system_date": pd.Timestamp.today().strftime("%Y-%m-%d"),
                "requested_as_of_date": today.strftime("%Y-%m-%d"),
                "review_rows": len(review),
                "recent_rows": len(recent),
                "summary_rows": len(summary),
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
        "recent": recent,
        "summary": summary,
        "accuracy_analysis": accuracy_analysis,
        "risk_signal_accuracy": risk_signal_accuracy,
        "last_updated": file_updated_at(DAILY_PREDICTION_REVIEW_PATH),
        "date_debug": read_csv_safe(REVIEW_DATE_DEBUG_PATH),
    }
