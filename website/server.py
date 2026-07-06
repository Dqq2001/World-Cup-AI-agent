from __future__ import annotations

import json
import base64
import binascii
import subprocess
import sys
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from backend import api as backend_api


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PUBLIC_DIR = PROJECT_ROOT / "website" / "public"
STATUS_PATH = PROJECT_ROOT / "data" / "cache" / "website_refresh_status.json"
DAILY_INTEL_PATH = PROJECT_ROOT / "reports" / "worldcup_daily_intel.csv"
OPENAI_MISSING_PATH = PROJECT_ROOT / "reports" / "openai_intel_missing_report.csv"


class WebsiteHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(PUBLIC_DIR), **kwargs)

    def do_POST(self) -> None:
        if self.path.rstrip("/") != "/api/worldcup/refresh-intel":
            self.send_json({"success": False, "error": "not_found"}, status=404)
            return
        username, password = self.read_credentials()
        if not backend_api.validate_refresh_admin(username, password):
            self.send_json({"success": False, "error": "Invalid admin credentials"}, status=403)
            return

        result = subprocess.run(
            [sys.executable, "scripts/refresh_today_intel_for_website.py"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=900,
            check=False,
        )
        status_payload = {}
        if STATUS_PATH.exists():
            try:
                status_payload = json.loads(STATUS_PATH.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                status_payload = {}
        success = result.returncode == 0
        self.send_json(
            {
                "success": success,
                "status": status_payload,
                "stdout": result.stdout[-4000:],
                "stderr": result.stderr[-4000:],
            },
            status=200 if success else 500,
        )

    def do_GET(self) -> None:
        if self.path.rstrip("/") != "/api/debug-intel":
            super().do_GET()
            return
        self.send_json(debug_intel_payload())

    def read_credentials(self) -> tuple[str | None, str | None]:
        auth_header = self.headers.get("Authorization", "")
        if auth_header.startswith("Basic "):
            try:
                decoded = base64.b64decode(auth_header[6:]).decode("utf-8")
                username, password = decoded.split(":", 1)
                return username, password
            except (ValueError, UnicodeDecodeError, binascii.Error):
                return None, None

        length = int(self.headers.get("Content-Length", "0") or 0)
        if length <= 0:
            return None, None
        try:
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            return None, None
        return payload.get("username"), payload.get("password")

    def send_json(self, payload: dict, status: int = 200) -> None:
        body = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def debug_intel_payload() -> dict:
    status_payload = {}
    if STATUS_PATH.exists():
        try:
            status_payload = json.loads(STATUS_PATH.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            status_payload = {}
    intel_row_count = 0
    if DAILY_INTEL_PATH.exists():
        try:
            import pandas as pd

            intel_row_count = len(pd.read_csv(DAILY_INTEL_PATH, encoding="utf-8"))
        except Exception:
            intel_row_count = 0
    latest_error = ""
    if OPENAI_MISSING_PATH.exists():
        try:
            import pandas as pd

            report = pd.read_csv(OPENAI_MISSING_PATH, encoding="utf-8")
            if not report.empty:
                latest_error = str(report.iloc[-1].get("error_message", ""))
        except Exception as exc:
            latest_error = str(exc)
    return {
        "env": {
            "OPENAI_API_KEY": bool(__import__("os").environ.get("OPENAI_API_KEY", "").strip()),
            "OPENAI_BASE_URL": bool(__import__("os").environ.get("OPENAI_BASE_URL", "").strip()),
            "OPENAI_INTEL_MODEL": bool(__import__("os").environ.get("OPENAI_INTEL_MODEL", "").strip()),
        },
        "last_intel_refresh_time": status_payload.get("last_refresh_at", ""),
        "intel_csv_exists": DAILY_INTEL_PATH.exists(),
        "intel_row_count": intel_row_count,
        "latest_openai_pixvyn_error": latest_error,
    }


def main() -> None:
    PUBLIC_DIR.mkdir(parents=True, exist_ok=True)
    server = ThreadingHTTPServer(("127.0.0.1", 8765), WebsiteHandler)
    print("Website server: http://127.0.0.1:8765")
    print("Refresh endpoint: POST http://127.0.0.1:8765/api/worldcup/refresh-intel")
    server.serve_forever()


if __name__ == "__main__":
    main()
