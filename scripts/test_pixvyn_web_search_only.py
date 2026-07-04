r"""
PowerShell:

cd C:\Users\Administrator\Desktop\worldcup-ai-agent
$env:OPENAI_BASE_URL="https://api.pixvyn.com/v1"
$env:OPENAI_INTEL_MODEL="gpt-5.5"
$env:OPENAI_API_KEY="YOUR_KEY_HERE"
python scripts/test_pixvyn_web_search_only.py
"""

from __future__ import annotations

import csv
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests


ROOT = Path("C:/Users/Administrator/Desktop/worldcup-ai-agent")
DEBUG_DIR = ROOT / "reports" / "debug"
JSON_OUTPUT = DEBUG_DIR / "pixvyn_web_search_only_response.json"
TEXT_OUTPUT = DEBUG_DIR / "pixvyn_web_search_only_response.txt"
SUMMARY_OUTPUT = DEBUG_DIR / "pixvyn_web_search_only_summary.csv"
TOOL_TYPE = "web_search"

for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, "reconfigure"):
        stream.reconfigure(encoding="utf-8")


def load_env() -> None:
    env_path = ROOT / ".env"
    if not env_path.exists():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def base_url() -> str:
    value = os.getenv("OPENAI_BASE_URL", "https://api.pixvyn.com/v1").strip().rstrip("/")
    if value == "https://api.pixvyn.com":
        return "https://api.pixvyn.com/v1"
    return value


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
                parts.append(content["text"])
            elif isinstance(content.get("text"), str):
                parts.append(content["text"])
    return "\n".join(part.strip() for part in parts if part.strip()).strip()


def has_web_search_call(response_json: dict[str, Any]) -> bool:
    return any(
        isinstance(item, dict) and item.get("type") == "web_search_call"
        for item in response_json.get("output", []) or []
    )


def extract_source_urls(response_json: dict[str, Any]) -> list[str]:
    urls: list[str] = []

    def visit(value: Any, key: str = "") -> None:
        if isinstance(value, dict):
            lowered_key = key.lower()
            for candidate_key in ("url", "cited_url", "source_url"):
                candidate = value.get(candidate_key)
                if isinstance(candidate, str) and candidate.startswith(("http://", "https://")):
                    urls.append(candidate)
            if lowered_key in {"annotations", "url_citation"}:
                for nested in value.values():
                    visit(nested)
            for nested_key, nested_value in value.items():
                visit(nested_value, str(nested_key))
        elif isinstance(value, list):
            for nested in value:
                visit(nested)
        elif isinstance(value, str):
            urls.extend(re.findall(r"https?://[^\s\"'<>),]+", value))

    visit(response_json)
    return list(dict.fromkeys(urls))


def classify(status_code: int | None, raw_text: str, json_ok: bool, final_text: str, web_search_found: bool) -> str:
    lowered = raw_text.lower()
    if status_code == 200 and final_text and web_search_found:
        return "WEB_SEARCH_SUCCESS"
    if status_code == 200 and final_text and not web_search_found:
        return "TEXT_RETURNED_BUT_NO_REAL_WEB_SEARCH"
    if status_code == 200 and not final_text:
        return "NO_FINAL_TEXT"
    if status_code == 400 and any(term in lowered for term in ["unknown tool", "unsupported tool", "invalid tool"]):
        return "WEB_SEARCH_TOOL_UNSUPPORTED"
    if status_code == 403 or "1010" in raw_text:
        return "PROXY_BLOCKED_OR_PROVIDER_FORBIDDEN"
    if status_code == 524 or "524" in raw_text or "A timeout occurred" in raw_text:
        return "PROVIDER_UPSTREAM_TIMEOUT_524"
    if status_code in {502, 503, 504}:
        return "PROVIDER_UNAVAILABLE_OR_TIMEOUT"
    return "FAILED_OTHER"


def write_summary(row: dict[str, Any]) -> None:
    DEBUG_DIR.mkdir(parents=True, exist_ok=True)
    columns = [
        "timestamp",
        "base_url",
        "model",
        "endpoint",
        "tool_type",
        "status_code",
        "json_ok",
        "response_status",
        "classification",
        "has_web_search_call",
        "has_final_text",
        "source_urls_count",
        "output_path",
        "final_text_preview",
    ]
    with SUMMARY_OUTPUT.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerow({column: row.get(column, "") for column in columns})


def main() -> int:
    load_env()
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    selected_base_url = base_url()
    model = os.getenv("OPENAI_INTEL_MODEL", "gpt-5.5").strip() or "gpt-5.5"
    endpoint = f"{selected_base_url}/responses"
    payload = {
        "model": model,
        "tools": [{"type": TOOL_TYPE}],
        "input": "Search the web for today's Croatia national football team injury or suspension news before the Portugal match. Return a concise report with source names and URLs if available.",
    }

    print(f"BASE_URL={selected_base_url}")
    print(f"MODEL={model}")
    print(f"ENDPOINT={endpoint}")
    print(f"TOOL_TYPE={TOOL_TYPE}")

    status_code: int | None = None
    raw_text = ""
    response_json: dict[str, Any] | None = None
    json_ok = False
    output_path = TEXT_OUTPUT

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
            json_ok = True
        except ValueError:
            response_json = None
    except requests.RequestException as exc:
        raw_text = f"{type(exc).__name__}: {exc}"

    DEBUG_DIR.mkdir(parents=True, exist_ok=True)
    if response_json is not None:
        JSON_OUTPUT.write_text(json.dumps(response_json, ensure_ascii=False, indent=2), encoding="utf-8")
        output_path = JSON_OUTPUT
    else:
        TEXT_OUTPUT.write_text(raw_text, encoding="utf-8")

    final_text = extract_final_text(response_json) if response_json is not None else ""
    web_search_found = has_web_search_call(response_json) if response_json is not None else False
    source_urls = extract_source_urls(response_json) if response_json is not None else []
    response_status = response_json.get("status", "") if response_json is not None else ""
    classification = classify(status_code, raw_text, json_ok, final_text, web_search_found)

    print(f"STATUS_CODE={status_code if status_code is not None else ''}")
    print(f"JSON_OK={json_ok}")
    print(f"TOP_LEVEL_KEYS={','.join(response_json.keys()) if response_json is not None else ''}")
    print(f"RESPONSE_STATUS={response_status}")
    print(f"HAS_WEB_SEARCH_CALL={web_search_found}")
    print(f"HAS_FINAL_TEXT={bool(final_text)}")
    print(f"SOURCE_URLS_COUNT={len(source_urls)}")
    print(f"SOURCE_URLS={'; '.join(source_urls)}")
    if final_text:
        print(f"FINAL_TEXT={final_text}")
    print(f"CLASSIFICATION={classification}")

    write_summary(
        {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "base_url": selected_base_url,
            "model": model,
            "endpoint": endpoint,
            "tool_type": TOOL_TYPE,
            "status_code": status_code if status_code is not None else "",
            "json_ok": json_ok,
            "response_status": response_status,
            "classification": classification,
            "has_web_search_call": web_search_found,
            "has_final_text": bool(final_text),
            "source_urls_count": len(source_urls),
            "output_path": output_path,
            "final_text_preview": final_text[:500],
        }
    )

    if classification == "WEB_SEARCH_SUCCESS":
        print("FINAL_CONCLUSION=WEB_SEARCH_WORKS")
        return 0
    print("FINAL_CONCLUSION=WEB_SEARCH_FAILED")
    print(f"FAILURE_CLASSIFICATION={classification}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
