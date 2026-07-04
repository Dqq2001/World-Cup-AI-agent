import argparse
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import pandas as pd


RAW_OUTPUT = Path("data/raw/international_results.csv")
MISSING_REPORT = Path("reports/international_data_missing_report.csv")
SOURCE_URLS = [
    "https://raw.githubusercontent.com/martj42/international_results/master/results.csv",
]
REQUIRED_COLUMNS = [
    "date",
    "home_team",
    "away_team",
    "home_score",
    "away_score",
    "tournament",
    "neutral",
]
OUTPUT_COLUMNS = [
    "date",
    "competition",
    "home_team",
    "away_team",
    "home_goals",
    "away_goals",
    "neutral",
]


def write_missing_report(rows: list[dict]) -> None:
    MISSING_REPORT.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(MISSING_REPORT, index=False, encoding="utf-8")


def validate_source(data: pd.DataFrame, source: str) -> None:
    missing = [column for column in REQUIRED_COLUMNS if column not in data.columns]
    if missing:
        raise ValueError(f"{source} missing required columns: {missing}")


def normalize_results(data: pd.DataFrame) -> pd.DataFrame:
    output = pd.DataFrame(
        {
            "date": pd.to_datetime(data["date"], errors="coerce").dt.strftime("%Y-%m-%d"),
            "competition": data["tournament"],
            "home_team": data["home_team"],
            "away_team": data["away_team"],
            "home_goals": pd.to_numeric(data["home_score"], errors="coerce"),
            "away_goals": pd.to_numeric(data["away_score"], errors="coerce"),
            "neutral": data["neutral"].astype(bool),
        }
    )
    output = output.dropna(subset=["date", "home_team", "away_team", "home_goals", "away_goals"])
    output = output.sort_values(["date", "competition", "home_team", "away_team"]).reset_index(drop=True)
    return output[OUTPUT_COLUMNS]


def fetch_csv(url: str) -> pd.DataFrame:
    request = Request(url, headers={"User-Agent": "worldcup-ai-agent/1.0"})
    with urlopen(request, timeout=60) as response:
        return pd.read_csv(response, encoding="utf-8")


def fetch_international_results(output_path: Path = RAW_OUTPUT) -> pd.DataFrame | None:
    if output_path.exists():
        data = pd.read_csv(output_path, encoding="utf-8")
        missing = [column for column in OUTPUT_COLUMNS if column not in data.columns]
        if missing:
            raise ValueError(f"Existing {output_path} missing columns: {missing}")
        print(f"已找到本機國際賽資料: {output_path} ({len(data)} rows)")
        return data

    errors = []
    for url in SOURCE_URLS:
        try:
            source = fetch_csv(url)
            validate_source(source, url)
            output = normalize_results(source)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output.to_csv(output_path, index=False, encoding="utf-8")
            print(f"已下載國際賽資料: {output_path} ({len(output)} rows)")
            return output
        except (HTTPError, URLError, TimeoutError, ValueError, pd.errors.ParserError) as exc:
            errors.append({"source": url, "message": str(exc)})

    write_missing_report(
        [
            {
                "data_type": "international_results",
                "missing": str(output_path),
                "message": "無法從公開來源下載國際賽資料。",
                "details": " | ".join(f"{error['source']}: {error['message']}" for error in errors),
            }
        ]
    )
    print(f"無法取得國際賽資料，已輸出缺資料報告: {MISSING_REPORT}")
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=RAW_OUTPUT)
    args = parser.parse_args()
    fetch_international_results(args.output)


if __name__ == "__main__":
    main()
