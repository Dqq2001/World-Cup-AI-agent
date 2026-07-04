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


def mask_key(value: str) -> str:
    if not value:
        return "<missing>"
    if len(value) < 10:
        return "<present>"
    return f"{value[:6]}...{value[-4:]}"


def responses_url(base_url: str) -> str:
    base = base_url.strip().rstrip("/")
    if base.endswith("/v1"):
        return f"{base}/responses"
    return f"{base}/v1/responses"


def extract_output_text(payload: dict) -> str:
    if isinstance(payload.get("output_text"), str):
        return payload["output_text"]
    parts = []
    for item in payload.get("output", []) or []:
        if item.get("type") != "message":
            continue
        for content in item.get("content", []) or []:
            if isinstance(content, dict) and content.get("text"):
                parts.append(str(content["text"]))
    return "".join(parts)


def main() -> int:
    load_env()
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    base_url = os.getenv("OPENAI_BASE_URL", "").strip()
    model = os.getenv("OPENAI_INTEL_MODEL", "").strip()
    url = responses_url(base_url)

    print(f"ROOT={ROOT}")
    print(f"url={url}")
    print(f"model={model or '<missing>'}")
    print(f"OPENAI_API_KEY={mask_key(api_key)}")

    payload = {
        "model": model,
        "input": "Reply with exactly: API_OK",
        "store": False,
    }
    try:
        response = requests.post(
            url,
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json=payload,
            timeout=60,
        )
    except requests.RequestException as exc:
        print("status_code=")
        print(f"response_text={type(exc).__name__}: {exc}")
        return 1

    print(f"status_code={response.status_code}")
    print("response_text=" + response.text[:2000])
    if not response.ok:
        return 1

    data = response.json()
    output = data.get("output", []) or []
    has_message = any(item.get("type") == "message" for item in output)
    has_web_search_call = any(item.get("type") == "web_search_call" for item in output)
    output_text = extract_output_text(data)
    print(f"response.status={data.get('status', '<missing>')}")
    print(f"output_text={output_text}")
    print(f"has_message={str(has_message).lower()}")
    print(f"has_web_search_call={str(has_web_search_call).lower()}")
    print(f"proxy_basic_api_usable={str('API_OK' in output_text).lower()}")
    return 0 if "API_OK" in output_text else 1


if __name__ == "__main__":
    raise SystemExit(main())
