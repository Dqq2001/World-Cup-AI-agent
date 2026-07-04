import csv
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import requests


ROOT = Path("C:/Users/Administrator/Desktop/worldcup-ai-agent")
REPORTS_DIR = ROOT / "reports"
RESPONSE_OUTPUT = REPORTS_DIR / "debug_croatia_injuries_web_response.json"
CSV_OUTPUT = REPORTS_DIR / "croatia_injury_intel_test.csv"

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


def responses_url(base_url: str) -> str:
    base = base_url.strip().rstrip("/")
    if base.endswith("/v1"):
        return f"{base}/responses"
    return f"{base}/v1/responses"


def output_item_types(payload: dict) -> str:
    return ",".join(str(item.get("type", "unknown")) for item in payload.get("output", []) or [])


def walk_json(value, texts: list[str], urls: list[str]) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            lowered = str(key).lower()
            if lowered in {"output_text", "text", "content", "message", "summary", "annotations", "title"}:
                if isinstance(item, str) and item.strip():
                    texts.append(item.strip())
            if lowered == "url" and isinstance(item, str) and item.strip():
                urls.append(item.strip())
            walk_json(item, texts, urls)
    elif isinstance(value, list):
        for item in value:
            walk_json(item, texts, urls)


def extract_final_text(payload: dict) -> str:
    if isinstance(payload.get("output_text"), str) and payload["output_text"].strip():
        return payload["output_text"].strip()
    parts = []
    for item in payload.get("output", []) or []:
        if item.get("type") != "message":
            continue
        for content in item.get("content", []) or []:
            if isinstance(content, dict) and isinstance(content.get("text"), str) and content["text"].strip():
                parts.append(content["text"].strip())
            elif isinstance(content, str) and content.strip():
                parts.append(content.strip())
    if parts:
        return "\n".join(parts).strip()
    texts: list[str] = []
    urls: list[str] = []
    walk_json(payload, texts, urls)
    return "\n".join(dict.fromkeys(texts)).strip()


def extract_urls(payload: dict) -> list[str]:
    texts: list[str] = []
    urls: list[str] = []
    walk_json(payload, texts, urls)
    return list(dict.fromkeys(urls))


def web_search_status(payload: dict) -> tuple[bool, str]:
    calls = [item for item in payload.get("output", []) or [] if item.get("type") == "web_search_call"]
    statuses = [str(item.get("status", "unknown")) for item in calls]
    return bool(calls), ";".join(statuses) if statuses else "missing"


def verified_injury_news_found(text: str) -> bool:
    lowered = text.lower()
    negative_markers = [
        "no verified croatia injury news found",
        "沒有找到經證實",
        "未找到經證實",
        "沒有已核實",
        "未有可靠",
    ]
    if any(marker in lowered for marker in negative_markers):
        return False
    evidence_markers = [
        "injury",
        "injured",
        "unavailable",
        "doubtful",
        "suspended",
        "停賽",
        "傷",
        "缺陣",
        "出戰成疑",
    ]
    return any(marker in lowered for marker in evidence_markers)


def write_csv(row: dict) -> None:
    CSV_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "match",
        "team",
        "source_type",
        "source_status",
        "request_success",
        "web_search_success",
        "final_text_success",
        "intel_has_content",
        "verified_injury_news_found",
        "source_urls",
        "intel_text",
        "created_at",
    ]
    with CSV_OUTPUT.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerow({column: row.get(column, "") for column in columns})


def main() -> int:
    load_env()
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    base_url = os.getenv("OPENAI_BASE_URL", "").strip()
    model = os.getenv("OPENAI_INTEL_MODEL", "").strip()
    url = responses_url(base_url)

    print(f"ROOT={ROOT}")
    print(f"url={url}")
    print(f"model={model or '<missing>'}")
    print(f"api_key_prefix={api_key[:6] if api_key else '<missing>'}")
    print(f"api_key_suffix={api_key[-4:] if api_key else '<missing>'}")

    body = {
        "model": model,
        "input": (
            "Search the web for today's Croatia national team injury, suspension, availability, and lineup news before Portugal vs Croatia. "
            "Focus only on Croatia. Return Traditional Chinese. You must include: 1) injury_news_summary, 2) unavailable_players, "
            "3) doubtful_players, 4) key_players_available, 5) source_urls, 6) confidence_level. If no verified injury news is found, "
            "say clearly: no verified Croatia injury news found. Do not invent names. Include source URLs."
        ),
        "tools": [{"type": "web_search_preview"}],
        "store": False,
    }

    try:
        response = requests.post(
            url,
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json=body,
            timeout=180,
        )
    except requests.RequestException as exc:
        print("status_code=")
        print("response.status=")
        print("output_text=")
        print("has_final_message=false")
        print("final_text_success=false")
        print("has_web_search_call=false")
        print("web_search_status=request_failed")
        print("output item types=")
        print("request_success=false")
        print("source_status=request_failed")
        print("intel_has_content=false")
        print(f"error_type={type(exc).__name__}")
        print(f"error_message={exc}")
        return 1

    print(f"status_code={response.status_code}")
    if response.status_code == 403 or "1010" in response.text:
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        RESPONSE_OUTPUT.write_text(
            json.dumps(
                {"status_code": response.status_code, "raw_response_text": response.text},
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        print("response.status=")
        print("output_text=")
        print("has_final_message=false")
        print("final_text_success=false")
        print("has_web_search_call=false")
        print("web_search_status=proxy_blocked")
        print("output item types=")
        print("request_success=false")
        print("source_status=proxy_blocked")
        print("error_type=provider_blocked")
        print("intel_has_content=false")
        print(f"error_message={response.text[:1000]}")
        return 1

    try:
        payload = response.json()
    except ValueError:
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        RESPONSE_OUTPUT.write_text(
            json.dumps(
                {"status_code": response.status_code, "raw_response_text": response.text},
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        print("response.status=")
        print("output_text=")
        print("has_final_message=false")
        print("final_text_success=false")
        print("has_web_search_call=false")
        print("web_search_status=invalid_json_response")
        print("output item types=")
        print(f"request_success={str(response.ok).lower()}")
        print("source_status=request_failed")
        print("intel_has_content=false")
        print(f"error_message={response.text[:1000]}")
        print(f"response_json={RESPONSE_OUTPUT}")
        return 1

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    RESPONSE_OUTPUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    final_text = extract_final_text(payload)
    urls = extract_urls(payload)
    has_web_search_call, search_status = web_search_status(payload)
    has_final_message = any(item.get("type") == "message" for item in payload.get("output", []) or [])
    final_text_success = bool(final_text)
    web_search_success = "completed" in search_status
    if response.status_code == 200 and final_text_success:
        source_status = "ok"
    elif response.status_code == 200 and web_search_success:
        source_status = "search_completed_no_final_text"
    else:
        source_status = "request_failed"
    intel_has_content = source_status == "ok"
    verified_found = verified_injury_news_found(final_text) if final_text_success else False

    print(f"response.status={payload.get('status', '<missing>')}")
    print(f"output_text={final_text}")
    print(f"has_final_message={str(has_final_message).lower()}")
    print(f"final_text_success={str(final_text_success).lower()}")
    print(f"has_web_search_call={str(has_web_search_call).lower()}")
    print(f"web_search_status={search_status}")
    print(f"output item types={output_item_types(payload)}")
    print(f"request_success={str(response.status_code == 200).lower()}")
    print(f"source_status={source_status}")
    print(f"intel_has_content={str(intel_has_content).lower()}")
    print(f"verified_injury_news_found={str(verified_found).lower()}")
    print(f"source_urls={'; '.join(urls) if urls else 'unknown'}")

    if final_text_success:
        write_csv(
            {
                "match": "Portugal vs Croatia",
                "team": "Croatia",
                "source_type": "pixvyn_web_search_preview",
                "source_status": source_status,
                "request_success": response.status_code == 200,
                "web_search_success": web_search_success,
                "final_text_success": final_text_success,
                "intel_has_content": intel_has_content,
                "verified_injury_news_found": verified_found,
                "source_urls": "; ".join(urls) if urls else "unknown",
                "intel_text": final_text,
                "created_at": datetime.now(timezone.utc).isoformat(),
            }
        )
        print(f"csv_output={CSV_OUTPUT}")
    print(f"response_json={RESPONSE_OUTPUT}")
    return 0 if source_status == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
