import json
import os
import sys
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import NO_PAID_API_MODE

ENV_PATH = PROJECT_ROOT / ".env"
API_FOOTBALL_BASE_URL = "https://v3.football.api-sports.io"


def load_dotenv(path: Path = ENV_PATH) -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def mask_key(value: str) -> str:
    if len(value) <= 8:
        return "<present>"
    return f"{value[:4]}...{value[-4:]}"


def api_get(path: str, api_key: str, params: dict | None = None) -> tuple[dict, dict, int]:
    query = f"?{urlencode(params)}" if params else ""
    request = Request(f"{API_FOOTBALL_BASE_URL}{path}{query}", headers={"x-apisports-key": api_key})
    try:
        with urlopen(request, timeout=30) as response:
            payload = json.loads(response.read().decode("utf-8"))
            return payload, dict(response.headers.items()), response.status
    except HTTPError as exc:
        body = exc.read().decode("utf-8", errors="ignore")
        try:
            payload = json.loads(body)
        except json.JSONDecodeError:
            payload = {"raw_error": body}
        return payload, dict(exc.headers.items()), exc.code


def print_status_payload(payload: dict, http_status: int) -> None:
    print(f"API HTTP status: {http_status}")
    errors = payload.get("errors")
    if errors:
        print(f"API errors: {errors}")
    response = payload.get("response", {})
    if response:
        account = response.get("account", {})
        subscription = response.get("subscription", {})
        requests = response.get("requests", {})
        print(f"Account: {account.get('firstname', '')} {account.get('lastname', '')}".strip())
        print(f"Plan: {subscription.get('plan', '<unknown>')}")
        print(f"Subscription active: {subscription.get('active', '<unknown>')}")
        print(f"Requests current: {requests.get('current', '<unknown>')}")
        print(f"Requests limit day: {requests.get('limit_day', '<unknown>')}")
    else:
        print("API status response 沒有 account/subscription 資訊。")


def find_related_leagues(api_key: str) -> list[dict]:
    rows = []
    seen = set()
    for query in ["world cup", "fifa", "international"]:
        payload, _, http_status = api_get("/leagues", api_key, {"search": query})
        print(f"League search '{query}' HTTP status: {http_status}")
        if payload.get("errors"):
            print(f"League search '{query}' errors: {payload['errors']}")
            continue
        for item in payload.get("response", []):
            league = item.get("league", {})
            country = item.get("country", {})
            league_id = league.get("id")
            if league_id in seen:
                continue
            seen.add(league_id)
            rows.append(
                {
                    "id": league_id,
                    "name": league.get("name"),
                    "type": league.get("type"),
                    "country": country.get("name"),
                }
            )
    return rows


def main() -> None:
    if NO_PAID_API_MODE:
        print("NO_PAID_API_MODE=True: skipping API-Football odds API test.")
        return

    load_dotenv()
    provider = os.environ.get("ODDS_PROVIDER", "").strip()
    api_key = os.environ.get("ODDS_API_KEY", "").strip()

    print(f"ODDS_PROVIDER: {provider or '<missing>'}")
    print(f"ODDS_API_KEY: {mask_key(api_key) if api_key else '<missing>'}")
    if provider != "api_football":
        raise SystemExit("錯誤: ODDS_PROVIDER 不是 api_football。")
    if not api_key:
        raise SystemExit("錯誤: 缺少 ODDS_API_KEY。")

    try:
        status_payload, _, http_status = api_get("/status", api_key)
    except URLError as exc:
        raise SystemExit(f"錯誤: 無法連線到 API-Football: {exc}") from exc

    print_status_payload(status_payload, http_status)
    if http_status != 200 or status_payload.get("errors"):
        raise SystemExit("錯誤: API key 測試未通過。")

    leagues = find_related_leagues(api_key)
    if not leagues:
        raise SystemExit("錯誤: 找不到 World Cup / FIFA / international 相關 league id。")

    print("找到相關 league candidates:")
    for league in leagues[:30]:
        print(f"- id={league['id']} name={league['name']} type={league['type']} country={league['country']}")
    print("測試通過。")


if __name__ == "__main__":
    main()
