from __future__ import annotations

import os
from pathlib import Path

from backend import refresh_controller


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_project_env() -> None:
    env_path = PROJECT_ROOT / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8", errors="replace").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


load_project_env()


def admin_credentials_configured() -> bool:
    return bool(os.getenv("REFRESH_ADMIN_USERNAME") and os.getenv("REFRESH_ADMIN_PASSWORD"))


def validate_refresh_admin(username: str | None, password: str | None) -> bool:
    expected_username = os.getenv("REFRESH_ADMIN_USERNAME")
    expected_password = os.getenv("REFRESH_ADMIN_PASSWORD")
    if not expected_username or not expected_password:
        return True
    return username == expected_username and password == expected_password


def forbidden_response() -> dict:
    return {
        "status": "failed",
        "http_status": 403,
        "updated_files": [],
        "errors": ["Invalid admin credentials"],
        "counts": {},
        "message": "Invalid admin credentials",
    }


def refresh_all(username: str | None = None, password: str | None = None, force: bool = False, date_text: str | None = None) -> dict:
    if not validate_refresh_admin(username, password):
        return forbidden_response()
    return refresh_controller.refresh_all(force=force, date_text=date_text)


def refresh_intel(username: str | None = None, password: str | None = None) -> dict:
    if not validate_refresh_admin(username, password):
        return forbidden_response()
    return refresh_controller.refresh_intel()


def refresh_match_intel(
    username: str | None,
    password: str | None,
    date_text: str,
    home_team: str,
    away_team: str,
) -> dict:
    if not validate_refresh_admin(username, password):
        return forbidden_response()
    return refresh_controller.refresh_match_intel(date_text, home_team, away_team)


def run_script(username: str | None, password: str | None, script: str, *args: str) -> dict:
    if not validate_refresh_admin(username, password):
        return forbidden_response()
    return refresh_controller.run_script(script, *args)
