from __future__ import annotations

from services.script_runner import python_command, run_command


def refresh_odds(date_text: str):
    manual = run_command(python_command("scripts/import_manual_worldcup_odds.py", "--skip-downstream"))
    openai = run_command(python_command("scripts/search_odds_with_openai.py", "--date", date_text), timeout=300)
    return [manual, openai]

