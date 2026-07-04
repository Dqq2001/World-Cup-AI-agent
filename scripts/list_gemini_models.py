import json
import os
import sys
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_env() -> None:
    env_path = PROJECT_ROOT / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def mask_key(value: str) -> str:
    if not value:
        return "<missing>"
    if len(value) < 8:
        return "<present>"
    return f"{value[:4]}...{value[-4:]}"


def main() -> int:
    load_env()
    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    model = os.environ.get("GEMINI_INTEL_MODEL", "").strip()
    print(f"GEMINI_API_KEY={mask_key(api_key)}")
    print(f"GEMINI_INTEL_MODEL={'<present> (' + model + ')' if model else '<missing>'}")
    if not api_key:
        print("Missing GEMINI_API_KEY.")
        return 1

    url = "https://generativelanguage.googleapis.com/v1beta/models?" + urlencode({"key": api_key})
    request = Request(url, headers={"Accept": "application/json"})
    try:
        with urlopen(request, timeout=30) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        print(f"Gemini models request failed: HTTP {exc.code} {body[:500]}")
        return 1
    except (URLError, TimeoutError, json.JSONDecodeError) as exc:
        print(f"Gemini models request failed: {exc}")
        return 1

    models = payload.get("models", [])
    if not models:
        print("No models returned.")
        return 1
    for item in models:
        name = str(item.get("name", "")).removeprefix("models/")
        methods = ", ".join(item.get("supportedGenerationMethods", []))
        display = item.get("displayName", "")
        print(f"{name} | {display} | {methods}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
