r"""
PowerShell:

cd C:\Users\Administrator\Desktop\worldcup-ai-agent
$env:OPENAI_BASE_URL="https://api.pixvyn.com/v1"
$env:OPENAI_INTEL_MODEL="gpt-5.5"
$env:OPENAI_API_KEY="YOUR_KEY_HERE"

python src/openai_search_client.py

Custom query:

python src/openai_search_client.py --query "Search the web for Portugal vs Croatia injury and suspension news today. Include source URLs."
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

import requests


PROJECT_ROOT = Path("C:/Users/Administrator/Desktop/worldcup-ai-agent")
DEBUG_DIR = PROJECT_ROOT / "reports" / "debug"
LAST_JSON_RESPONSE = DEBUG_DIR / "openai_search_client_last_response.json"
LAST_TEXT_RESPONSE = DEBUG_DIR / "openai_search_client_last_response.txt"
LAST_REQUEST = DEBUG_DIR / "openai_search_client_last_request.json"
TOOL_TYPE = "web_search"
DEFAULT_QUERY = (
    "Search the web for today's Croatia national football team injury or suspension news before the Portugal match. "
    "Return a concise report with source names and URLs if available."
)

for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, "reconfigure"):
        stream.reconfigure(encoding="utf-8")


def load_env() -> None:
    env_path = PROJECT_ROOT / ".env"
    if not env_path.exists():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def configured_base_url() -> str:
    base_url = os.environ.get("OPENAI_BASE_URL", "https://api.pixvyn.com/v1").strip().rstrip("/")
    if base_url == "https://api.pixvyn.com":
        return "https://api.pixvyn.com/v1"
    return base_url


def extract_final_text(response_json: dict[str, Any]) -> str:
    output_text = response_json.get("output_text")
    if isinstance(output_text, str) and output_text.strip():
        return output_text.strip()

    parts: list[str] = []
    for item in response_json.get("output", []) or []:
        if not isinstance(item, dict) or item.get("type") != "message":
            continue
        for content in item.get("content", []) or []:
            if not isinstance(content, dict):
                continue
            if content.get("type") in {"output_text", "text"} and isinstance(content.get("text"), str):
                parts.append(content["text"].strip())
            elif isinstance(content.get("text"), str):
                parts.append(content["text"].strip())
    return "\n".join(part for part in parts if part).strip()


def extract_source_urls(obj: Any) -> list[str]:
    urls: list[str] = []

    def visit(value: Any) -> None:
        if isinstance(value, dict):
            for nested in value.values():
                visit(nested)
        elif isinstance(value, list):
            for nested in value:
                visit(nested)
        elif isinstance(value, str):
            urls.extend(re.findall(r"https?://[^\s\"'<>),]+", value))

    visit(obj)
    return list(dict.fromkeys(urls))


def save_debug_request(base_url: str, endpoint: str, model: str, query: str) -> None:
    DEBUG_DIR.mkdir(parents=True, exist_ok=True)
    LAST_REQUEST.write_text(
        json.dumps(
            {
                "base_url": base_url,
                "endpoint": endpoint,
                "model": model,
                "tool_type": TOOL_TYPE,
                "query": query,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def save_debug_response(response_json: dict[str, Any] | None, raw_text: str) -> None:
    DEBUG_DIR.mkdir(parents=True, exist_ok=True)
    if response_json is not None:
        LAST_JSON_RESPONSE.write_text(json.dumps(response_json, ensure_ascii=False, indent=2), encoding="utf-8")
    else:
        LAST_TEXT_RESPONSE.write_text(raw_text, encoding="utf-8")


def run_web_search(query: str) -> dict[str, Any]:
    load_env()
    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    base_url = configured_base_url()
    model = os.environ.get("OPENAI_INTEL_MODEL", "gpt-5.5").strip() or "gpt-5.5"
    endpoint = f"{base_url}/responses"
    save_debug_request(base_url, endpoint, model, query)

    if not api_key:
        return {
            "success": False,
            "status_code": None,
            "response_status": "",
            "final_text": "",
            "source_urls": [],
            "source_urls_count": 0,
            "model": model,
            "endpoint": endpoint,
            "tool_type": TOOL_TYPE,
            "error_message": "Missing OPENAI_API_KEY",
        }

    payload = {
        "model": model,
        "tools": [{"type": TOOL_TYPE}],
        "input": query,
    }

    response_json: dict[str, Any] | None = None
    raw_text = ""
    status_code = None
    error_message = ""
    try:
        response = requests.post(
            endpoint,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=(15, 120),
        )
        status_code = response.status_code
        raw_text = response.text
        try:
            response_json = response.json()
        except ValueError:
            error_message = raw_text[:1000]
    except requests.RequestException as exc:
        raw_text = f"{type(exc).__name__}: {exc}"
        error_message = raw_text

    save_debug_response(response_json, raw_text)

    response_status = response_json.get("status", "") if response_json is not None else ""
    final_text = extract_final_text(response_json) if response_json is not None else ""
    source_urls = extract_source_urls(response_json) if response_json is not None else []
    success = status_code == 200 and bool(final_text) and len(source_urls) > 0
    return {
        "success": success,
        "status_code": status_code,
        "response_status": response_status,
        "final_text": final_text,
        "source_urls": source_urls,
        "source_urls_count": len(source_urls),
        "model": model,
        "endpoint": endpoint,
        "tool_type": TOOL_TYPE,
        "error_message": "" if success else error_message,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", default=DEFAULT_QUERY)
    args = parser.parse_args()

    result = run_web_search(args.query)
    print(f"SUCCESS={result['success']}")
    print(f"STATUS_CODE={result['status_code'] if result['status_code'] is not None else ''}")
    print(f"RESPONSE_STATUS={result['response_status']}")
    print(f"SOURCE_URLS_COUNT={result['source_urls_count']}")
    print(f"SOURCE_URLS={'; '.join(result['source_urls'])}")
    print(f"FINAL_TEXT={result['final_text']}")
    if result["error_message"]:
        print(f"ERROR_MESSAGE={result['error_message']}")
    return 0 if result["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
