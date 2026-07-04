from __future__ import annotations

from services.script_runner import python_command, run_command


def refresh_daily_intel():
    return run_command(python_command("scripts/refresh_today_intel_for_website.py"), timeout=300)


def refresh_match_intel(date_text: str, home_team: str, away_team: str):
    openai = run_command(
        python_command(
            "scripts/search_intel_with_openai.py",
            "--date",
            date_text,
            "--home-team",
            home_team,
            "--away-team",
            away_team,
        ),
        timeout=180,
    )
    merge = run_command(python_command("scripts/run_daily_worldcup_intel.py"), timeout=180)
    return [openai, merge]

