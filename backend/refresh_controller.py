from __future__ import annotations

from datetime import date

import pandas as pd

from data_access.csv_store import write_csv_atomic
from data_access.paths import REPORTS_DIR
from services import intel_service, odds_service, prediction_service, results_service, review_service
from services.script_runner import ScriptResult, python_command, run_command


DEBUG_PATH = REPORTS_DIR / "backend_refresh_debug.csv"


def _result_to_row(result: ScriptResult, workflow: str) -> dict:
    return {
        "workflow": workflow,
        "command": " ".join(result.command),
        "returncode": result.returncode,
        "ok": result.ok,
        "rate_limited": result.rate_limited,
        "output_preview": result.output[-1000:],
    }


def _combine_results(results: list[ScriptResult], workflow: str) -> dict:
    rows = [_result_to_row(result, workflow) for result in results]
    write_csv_atomic(pd.DataFrame(rows), DEBUG_PATH)
    errors = [row["output_preview"] for row in rows if not row["ok"]]
    status = "success"
    if any(row["rate_limited"] for row in rows):
        status = "partial_failed"
    elif errors and len(errors) == len(rows):
        status = "failed"
    elif errors:
        status = "partial_failed"
    return {
        "status": status,
        "updated_files": [],
        "errors": errors,
        "counts": {"commands": len(rows), "failed": len(errors)},
        "rate_limited": any(row["rate_limited"] for row in rows),
        "message": "\n".join(row["output_preview"] for row in rows if row["output_preview"])[-2000:],
    }


def run_script(script: str, *args: str, timeout: int = 1800) -> dict:
    result = run_command(python_command(script, *args), timeout=timeout)
    return _combine_results([result], workflow=script)


def refresh_results(force: bool = False) -> dict:
    return _combine_results([results_service.refresh_results(force=force)], workflow="results")


def refresh_intel() -> dict:
    return _combine_results([intel_service.refresh_daily_intel()], workflow="intel")


def refresh_match_intel(date_text: str, home_team: str, away_team: str) -> dict:
    return _combine_results(
        intel_service.refresh_match_intel(date_text, home_team, away_team),
        workflow="match_intel",
    )


def refresh_odds(date_text: str | None = None) -> dict:
    return _combine_results(odds_service.refresh_odds(date_text or date.today().isoformat()), workflow="odds")


def refresh_predictions() -> dict:
    return _combine_results(prediction_service.refresh_predictions(), workflow="predictions")


def refresh_review() -> dict:
    return _combine_results([review_service.refresh_prediction_review()], workflow="review")


def refresh_all(force: bool = False, date_text: str | None = None) -> dict:
    today_text = date_text or date.today().isoformat()
    results: list[ScriptResult] = [
        run_command(
            python_command(
                "scripts/fetch_worldcup_fixtures.py",
                "--include-knockout",
                "--force-refresh",
                "--skip-downstream",
            ),
            timeout=300,
        ),
        run_command(python_command("scripts/fetch_knockout_bracket.py"), timeout=300),
        results_service.refresh_results(force=force),
    ]
    results.extend(prediction_service.refresh_model_predictions())
    results.extend(odds_service.refresh_odds(today_text))
    results.extend(prediction_service.refresh_betting_predictions())
    results.append(intel_service.refresh_daily_intel())
    return _combine_results(results, workflow="refresh_all")
