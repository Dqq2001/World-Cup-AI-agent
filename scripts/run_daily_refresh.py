import json
import hashlib
import subprocess
import sys
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORTS_DIR = PROJECT_ROOT / "reports"
CACHE_DIR = PROJECT_ROOT / "data" / "cache"
LOG_PATH = REPORTS_DIR / "scheduler_log.txt"
STATUS_PATH = CACHE_DIR / "scheduler_status.json"
TRACKED_OUTPUTS = [
    PROJECT_ROOT / "data" / "processed" / "worldcup_fixtures.csv",
    PROJECT_ROOT / "data" / "processed" / "worldcup_fixtures_resolved.csv",
    PROJECT_ROOT / "data" / "processed" / "worldcup_results.csv",
    PROJECT_ROOT / "data" / "processed" / "worldcup_model_predictions.csv",
    PROJECT_ROOT / "data" / "processed" / "worldcup_poisson_predictions.csv",
    PROJECT_ROOT / "data" / "processed" / "worldcup_openai_odds.csv",
    PROJECT_ROOT / "data" / "processed" / "worldcup_openai_market_predictions.csv",
    PROJECT_ROOT / "data" / "processed" / "worldcup_openai_intel.csv",
    PROJECT_ROOT / "reports" / "worldcup_model_only_predictions.csv",
    PROJECT_ROOT / "reports" / "worldcup_daily_intel.csv",
    PROJECT_ROOT / "reports" / "worldcup_daily_betting_brief.md",
    PROJECT_ROOT / "reports" / "daily_prediction_vs_result.csv",
    PROJECT_ROOT / "reports" / "daily_prediction_summary.csv",
    PROJECT_ROOT / "data" / "processed" / "worldcup_prediction_feedback.csv",
]
KNOCKOUT_DEBUG_PATH = PROJECT_ROOT / "reports" / "knockout_bracket_fetch_debug.csv"


COMMANDS = [
    [sys.executable, "scripts/fetch_worldcup_fixtures.py", "--include-knockout", "--force-refresh", "--skip-downstream"],
    [sys.executable, "scripts/fetch_knockout_bracket.py"],
    [sys.executable, "scripts/fetch_worldcup_results.py", "--all-completed", "--force-refresh"],
    [sys.executable, "scripts/export_worldcup_model_predictions.py"],
    [sys.executable, "scripts/export_worldcup_poisson_predictions.py"],
    [sys.executable, "scripts/import_manual_worldcup_odds.py"],
    [sys.executable, "scripts/search_odds_with_openai.py", "--date", datetime.now().strftime("%Y-%m-%d")],
    [sys.executable, "scripts/build_worldcup_features.py"],
    [sys.executable, "scripts/run_worldcup_betting_agents.py"],
    [sys.executable, "scripts/evaluate_daily_predictions.py"],
    [sys.executable, "scripts/refresh_today_intel_for_website.py"],
]


def now_text() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def append_log(message: str) -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    with LOG_PATH.open("a", encoding="utf-8") as file:
        file.write(f"[{now_text()}] {message}\n")


def read_status() -> dict:
    if not STATUS_PATH.exists():
        return {}
    try:
        return json.loads(STATUS_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def write_status(job_name: str, status: str, message: str) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    payload = read_status()
    payload[job_name] = {
        "last_run": now_text(),
        "last_status": status,
        "last_message": message[-2000:],
    }
    STATUS_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def file_hash(path: Path) -> str | None:
    if not path.exists():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()


def snapshot_outputs() -> dict[str, str | None]:
    return {str(path): file_hash(path) for path in TRACKED_OUTPUTS}


def changed_outputs(before: dict[str, str | None], after: dict[str, str | None]) -> list[str]:
    changed = []
    for path, before_hash in before.items():
        if after.get(path) != before_hash:
            changed.append(path)
    return changed


def unresolved_knockout_count() -> int:
    if not KNOCKOUT_DEBUG_PATH.exists():
        return 0
    try:
        import pandas as pd

        debug = pd.read_csv(KNOCKOUT_DEBUG_PATH, encoding="utf-8")
    except Exception:
        return 0
    if "fallback_slot_matches" in debug.columns:
        return int(pd.to_numeric(debug["fallback_slot_matches"], errors="coerce").fillna(0).sum())
    if "used_fallback" in debug.columns:
        return int(debug["used_fallback"].astype(str).str.lower().isin(["true", "1"]).sum())
    return 0


def run_command(command: list[str]) -> tuple[bool, str]:
    append_log(f"daily command start: {' '.join(command)}")
    try:
        result = subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=900,
            check=False,
        )
    except Exception as exc:
        append_log(f"daily command exception: {exc}")
        return False, str(exc)

    output = f"{result.stdout}\n{result.stderr}".strip()
    if output:
        append_log(output[-4000:])
    if result.returncode != 0:
        append_log(f"daily command failed with exit code {result.returncode}: {' '.join(command)}")
        return False, output
    return True, output


def main() -> int:
    append_log("daily refresh started")
    before = snapshot_outputs()
    failures = []
    messages = []
    for command in COMMANDS:
        ok, output = run_command(command)
        messages.append(output)
        if not ok:
            failures.append(" ".join(command))
            # Continue so cached/partial data can still be refreshed by later steps.
            continue

    after = snapshot_outputs()
    changed = changed_outputs(before, after)
    fallback_count = unresolved_knockout_count()

    if failures:
        message = "failed commands: " + " | ".join(failures)
        write_status("daily", "error", message)
        append_log(f"daily refresh finished with errors: {message}")
        return 1

    if not changed:
        message = "refresh completed; no tracked fixtures/results/predictions/intel outputs changed"
        write_status("daily", "success", message)
        append_log(f"daily refresh finished without actual update: {message}")
        return 0

    message = "\n".join(part for part in messages if part).strip()
    if fallback_count:
        message = f"{message}\nchanged outputs: {len(changed)}; unresolved knockout fallback rows: {fallback_count}"
        write_status("daily", "success", message)
        append_log(f"daily refresh finished successfully with warning: unresolved knockout fallback rows={fallback_count}")
        return 0

    message = f"{message}\nchanged outputs: {len(changed)}"
    write_status("daily", "success", message)
    append_log(f"daily refresh finished successfully; changed outputs={len(changed)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
