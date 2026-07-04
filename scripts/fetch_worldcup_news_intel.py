import argparse
import json
import sys
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import quote_plus
from urllib.request import Request, urlopen

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.paths import PROCESSED_DATA_DIR


FIXTURES_PATH = PROCESSED_DATA_DIR / "worldcup_fixtures.csv"
OUTPUT_PATH = PROCESSED_DATA_DIR / "worldcup_news_intel.csv"
MISSING_REPORT = PROJECT_ROOT / "reports" / "worldcup_news_intel_missing_report.csv"
OUTPUT_COLUMNS = [
    "date",
    "home_team",
    "away_team",
    "team_news_home",
    "team_news_away",
    "injuries_home",
    "injuries_away",
    "suspensions_home",
    "suspensions_away",
    "expected_lineup_home",
    "expected_lineup_away",
    "coach_comments_home",
    "coach_comments_away",
    "source_url",
    "confidence",
]


def load_fixtures(path: Path, as_of_date: str, days_ahead: int) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing fixtures: {path}")
    fixtures = pd.read_csv(path, encoding="utf-8")
    required = ["date", "home_team", "away_team"]
    missing = [column for column in required if column not in fixtures.columns]
    if missing:
        raise ValueError(f"fixtures missing columns: {missing}")
    fixtures = fixtures.copy()
    fixtures["date"] = pd.to_datetime(fixtures["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    as_of = pd.to_datetime(as_of_date)
    end_date = as_of + pd.Timedelta(days=days_ahead)
    fixture_dates = pd.to_datetime(fixtures["date"], errors="coerce")
    return fixtures[(fixture_dates >= as_of) & (fixture_dates <= end_date)].reset_index(drop=True)


def gdelt_search(query: str, max_records: int = 5) -> list[dict]:
    encoded_query = quote_plus(query)
    url = (
        "https://api.gdeltproject.org/api/v2/doc/doc"
        f"?query={encoded_query}&mode=ArtList&format=json&maxrecords={max_records}&sort=HybridRel"
    )
    request = Request(url, headers={"User-Agent": "worldcup-ai-agent/1.0"})
    with urlopen(request, timeout=30) as response:
        payload = json.loads(response.read().decode("utf-8", errors="ignore"))
    return payload.get("articles", [])


def summarize_articles(team: str, articles: list[dict]) -> dict:
    if not articles:
        return {
            "team_news": "unknown",
            "injuries": "unknown",
            "suspensions": "unknown",
            "expected_lineup": "unknown",
            "coach_comments": "unknown",
            "source_url": "unknown",
            "confidence": "low",
        }

    titles = [article.get("title", "") for article in articles[:3] if article.get("title")]
    urls = [article.get("url", "") for article in articles[:3] if article.get("url")]
    combined = " | ".join(titles)
    lower = combined.lower()
    injury_terms = ["injury", "injured", "doubt", "fitness", "out"]
    suspension_terms = ["suspension", "suspended", "ban"]
    lineup_terms = ["lineup", "starting xi", "squad"]
    coach_terms = ["coach", "manager", "press conference", "said"]

    return {
        "team_news": combined or "unknown",
        "injuries": combined if any(term in lower for term in injury_terms) else "unknown",
        "suspensions": combined if any(term in lower for term in suspension_terms) else "unknown",
        "expected_lineup": combined if any(term in lower for term in lineup_terms) else "unknown",
        "coach_comments": combined if any(term in lower for term in coach_terms) else "unknown",
        "source_url": "; ".join(urls) if urls else "unknown",
        "confidence": "medium" if combined else "low",
    }


def collect_for_team(team: str) -> dict:
    query = f'"{team}" football team news injuries suspensions lineup coach World Cup'
    articles = gdelt_search(query)
    return summarize_articles(team, articles)


def write_missing_report(rows: list[dict]) -> None:
    MISSING_REPORT.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(MISSING_REPORT, index=False, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixtures", type=Path, default=FIXTURES_PATH)
    parser.add_argument("--as-of-date", default=pd.Timestamp.today().strftime("%Y-%m-%d"))
    parser.add_argument("--days-ahead", type=int, default=7)
    parser.add_argument("--max-matches", type=int, default=None)
    args = parser.parse_args()

    try:
        fixtures = load_fixtures(args.fixtures, args.as_of_date, args.days_ahead)
    except Exception as exc:
        write_missing_report([{"data_type": "fixtures", "message": str(exc)}])
        raise SystemExit(f"錯誤: {exc}") from exc

    if args.max_matches is not None:
        fixtures = fixtures.head(args.max_matches)

    rows = []
    missing_rows = []
    for fixture in fixtures.itertuples(index=False):
        try:
            home = collect_for_team(fixture.home_team)
            away = collect_for_team(fixture.away_team)
        except (HTTPError, URLError, TimeoutError, json.JSONDecodeError) as exc:
            missing_rows.append(
                {
                    "date": fixture.date,
                    "home_team": fixture.home_team,
                    "away_team": fixture.away_team,
                    "message": f"web/API collector failed: {exc}",
                }
            )
            continue

        rows.append(
            {
                "date": fixture.date,
                "home_team": fixture.home_team,
                "away_team": fixture.away_team,
                "team_news_home": home["team_news"],
                "team_news_away": away["team_news"],
                "injuries_home": home["injuries"],
                "injuries_away": away["injuries"],
                "suspensions_home": home["suspensions"],
                "suspensions_away": away["suspensions"],
                "expected_lineup_home": home["expected_lineup"],
                "expected_lineup_away": away["expected_lineup"],
                "coach_comments_home": home["coach_comments"],
                "coach_comments_away": away["coach_comments"],
                "source_url": "; ".join(url for url in [home["source_url"], away["source_url"]] if url != "unknown") or "unknown",
                "confidence": "medium" if home["confidence"] == "medium" or away["confidence"] == "medium" else "low",
            }
        )

    output = pd.DataFrame(rows, columns=OUTPUT_COLUMNS)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(OUTPUT_PATH, index=False, encoding="utf-8")
    if missing_rows or output.empty:
        if output.empty and not missing_rows:
            missing_rows.append({"data_type": "news_intel", "message": "No upcoming matches or no articles found."})
        write_missing_report(missing_rows)
    else:
        write_missing_report([])

    print(f"已輸出 news intel: {OUTPUT_PATH} ({len(output)} rows)")
    print(f"missing report: {MISSING_REPORT}")


if __name__ == "__main__":
    main()
