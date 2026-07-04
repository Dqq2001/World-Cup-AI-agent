import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

try:
    from apscheduler.schedulers.blocking import BlockingScheduler
    from apscheduler.triggers.cron import CronTrigger
    from apscheduler.triggers.interval import IntervalTrigger
except ImportError as exc:
    raise SystemExit("Missing dependency: install APScheduler or run `pip install -r requirements.txt`.") from exc


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORTS_DIR = PROJECT_ROOT / "reports"
CACHE_DIR = PROJECT_ROOT / "data" / "cache"
LOG_PATH = REPORTS_DIR / "scheduler_log.txt"
STATUS_PATH = CACHE_DIR / "scheduler_status.json"


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


def write_scheduler_status(scheduler: BlockingScheduler) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    payload = read_status()
    for job_name in ["daily", "hourly"]:
        job = scheduler.get_job(job_name)
        if job and job.next_run_time:
            payload.setdefault(job_name, {})
            payload[job_name]["next_run"] = job.next_run_time.strftime("%Y-%m-%d %H:%M:%S")
    STATUS_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def run_script(label: str, script: str) -> None:
    append_log(f"{label} scheduled job started")
    try:
        result = subprocess.run(
            [sys.executable, script],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=1800,
            check=False,
        )
    except Exception as exc:
        append_log(f"{label} scheduled job exception: {exc}")
        return

    output = f"{result.stdout}\n{result.stderr}".strip()
    if output:
        append_log(output[-4000:])
    if "429" in output or "rate limited" in output.lower():
        append_log(f"{label} scheduled job rate limited; using cache")
        return
    if result.returncode != 0:
        append_log(f"{label} scheduled job failed with exit code {result.returncode}")
        return
    append_log(f"{label} scheduled job finished successfully")


def main() -> None:
    scheduler = BlockingScheduler()
    scheduler.add_job(
        lambda: run_script("daily", "scripts/run_daily_refresh.py"),
        CronTrigger(hour=12, minute=0),
        id="daily",
        replace_existing=True,
        max_instances=1,
        coalesce=True,
    )
    scheduler.add_job(
        lambda: run_script("hourly", "scripts/run_hourly_intel_refresh.py"),
        IntervalTrigger(hours=1),
        id="hourly",
        replace_existing=True,
        max_instances=1,
        coalesce=True,
    )
    scheduler.add_job(
        lambda: write_scheduler_status(scheduler),
        IntervalTrigger(minutes=1),
        id="status",
        replace_existing=True,
        max_instances=1,
        coalesce=True,
    )
    write_scheduler_status(scheduler)
    append_log("scheduler started; daily=12:00 local time, hourly=every 1 hour")
    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        append_log("scheduler stopped")


if __name__ == "__main__":
    main()
