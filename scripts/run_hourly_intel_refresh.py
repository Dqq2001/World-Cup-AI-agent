import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORTS_DIR = PROJECT_ROOT / "reports"
CACHE_DIR = PROJECT_ROOT / "data" / "cache"
LOG_PATH = REPORTS_DIR / "scheduler_log.txt"
STATUS_PATH = CACHE_DIR / "scheduler_status.json"


COMMANDS = [
    [sys.executable, "scripts/refresh_today_intel_for_website.py"],
    [sys.executable, "scripts/run_worldcup_betting_agents.py", "--model-only"],
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


def run_command(command: list[str]) -> tuple[bool, str]:
    append_log(f"hourly command start: {' '.join(command)}")
    try:
        result = subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=600,
            check=False,
        )
    except Exception as exc:
        append_log(f"hourly command exception: {exc}")
        return False, str(exc)

    output = f"{result.stdout}\n{result.stderr}".strip()
    if output:
        append_log(output[-4000:])
    if "429" in output or "rate limited" in output.lower():
        append_log("hourly refresh was rate limited; using cached data")
        return True, "Rate limited. Using cached data."
    if result.returncode != 0:
        append_log(f"hourly command failed with exit code {result.returncode}: {' '.join(command)}")
        return False, output
    return True, output


def main() -> int:
    append_log("hourly intel refresh started")
    failures = []
    messages = []
    for command in COMMANDS:
        ok, output = run_command(command)
        messages.append(output)
        if not ok:
            failures.append(" ".join(command))
            continue

    if failures:
        message = "failed commands: " + " | ".join(failures)
        write_status("hourly", "error", message)
        append_log(f"hourly intel refresh finished with errors: {message}")
        return 1

    message = "\n".join(part for part in messages if part).strip()
    write_status("hourly", "success", message)
    append_log("hourly intel refresh finished successfully")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
