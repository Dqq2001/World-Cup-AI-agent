from __future__ import annotations

from data_access.paths import REPORTS_DIR
from services.script_runner import python_command, run_command


DAILY_BRIEF_PATH = REPORTS_DIR / "worldcup_daily_betting_brief.md"


def refresh_daily_brief():
    return run_command(python_command("scripts/run_daily_worldcup_intel.py"), timeout=300)


def load_daily_brief_text() -> tuple[str, str]:
    if not DAILY_BRIEF_PATH.exists():
        return "", "Daily brief not found."
    return DAILY_BRIEF_PATH.read_text(encoding="utf-8", errors="replace"), ""
