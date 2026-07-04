import json
import os
from pathlib import Path

import requests


ROOT = Path("C:/Users/Administrator/Desktop/worldcup-ai-agent")


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


def extract_output_text(payload: dict) -> str:
    if isinstance(payload.get("output_text"), str):
        return payload["output_text"].strip()
    parts = []
    for item in payload.get("output", []) or []:
        if item.get("type") != "message":
            continue
        for content in item.get("content", []) or []:
            if isinstance(content, dict) and content.get("text"):
                parts.append(str(content["text"]))
    return "".join(parts).strip()


def output_item_types(payload: dict) -> str:
    return ",".join(str(item.get("type", "unknown")) for item in payload.get("output", []) or [])


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
            "You are a football match intelligence analyst. Analyze tonight's Portugal vs Croatia match. "
            "Focus on team form, tactical matchup, key risks, likely game script, betting risk, and give a cautious final view. "
            "Do not claim live or latest news unless provided. Return the answer in Traditional Chinese."
        ),
        "store": False,
    }

    try:
        response = requests.post(
            url,
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json=body,
            timeout=120,
        )
    except requests.RequestException as exc:
        print("status_code=")
        print("response.status=")
        print("output_text=")
        print("has_final_message=false")
        print("output item types=")
        print("request_success=false")
        print("final_text_success=false")
        print("proxy_text_api_usable=false")
        print("source_status=request_failed")
        print(f"error_type={type(exc).__name__}")
        print(f"error_message={exc}")
        return 1

    print(f"status_code={response.status_code}")
    if response.status_code == 403 or "1010" in response.text:
        print("response.status=")
        print("output_text=")
        print("has_final_message=false")
        print("output item types=")
        print("request_success=false")
        print("final_text_success=false")
        print("proxy_text_api_usable=false")
        print("source_status=proxy_blocked")
        print("error_type=provider_blocked")
        print(f"error_message={response.text[:1000]}")
        return 1

    try:
        payload = response.json()
    except ValueError:
        print("response.status=")
        print("output_text=")
        print("has_final_message=false")
        print("output item types=")
        print(f"request_success={str(response.ok).lower()}")
        print("final_text_success=false")
        print("proxy_text_api_usable=false")
        print("source_status=request_failed")
        print("error_type=invalid_json_response")
        print(f"error_message={response.text[:1000]}")
        return 1

    output_text = extract_output_text(payload)
    has_final_message = any(item.get("type") == "message" for item in payload.get("output", []) or [])
    final_text_success = response.status_code == 200 and bool(output_text)
    print(f"response.status={payload.get('status', '<missing>')}")
    print(f"output_text={output_text}")
    print(f"has_final_message={str(has_final_message).lower()}")
    print(f"output item types={output_item_types(payload)}")
    print(f"request_success={str(response.status_code == 200).lower()}")
    print(f"final_text_success={str(final_text_success).lower()}")
    if final_text_success:
        print("proxy_text_api_usable=true")
        print("source_status=ok")
        return 0
    print("proxy_text_api_usable=false")
    print("source_status=completed_no_final_text" if response.status_code == 200 else "source_status=request_failed")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
