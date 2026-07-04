import argparse
from pathlib import Path

import pandas as pd


DEFAULT_META_PATH = Path("data/processed/meta_features.csv")
DEFAULT_MATCHES_PATH = Path("data/processed/matches_features.csv")
DEFAULT_OUTPUT_PATH = Path("data/processed/meta_features_with_odds.csv")
MATCH_KEYS = ["date", "league", "home_team", "away_team"]
STANDARD_ODDS_COLUMNS = {
    "odds_home": "home_odds",
    "odds_draw": "draw_odds",
    "odds_away": "away_odds",
}
RAW_ODDS_CANDIDATES = [
    ("B365H", "B365D", "B365A"),
    ("PSH", "PSD", "PSA"),
    ("AvgH", "AvgD", "AvgA"),
    ("MaxH", "MaxD", "MaxA"),
]


def read_csv(path: Path, name: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"{name} CSV not found: {path}")
    return pd.read_csv(path, encoding="utf-8")


def choose_merge_keys(meta: pd.DataFrame, odds: pd.DataFrame) -> list[str]:
    if "match_id" in meta.columns and "match_id" in odds.columns:
        return ["match_id"]

    missing_meta = [column for column in MATCH_KEYS if column not in meta.columns]
    missing_odds = [column for column in MATCH_KEYS if column not in odds.columns]
    if missing_meta or missing_odds:
        raise ValueError(
            "Cannot merge odds. Missing merge keys: "
            f"meta={missing_meta}, odds_source={missing_odds}"
        )
    return MATCH_KEYS


def normalize_keys(data: pd.DataFrame, merge_keys: list[str]) -> pd.DataFrame:
    data = data.copy()
    if "date" in merge_keys:
        data["date"] = pd.to_datetime(data["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    for column in ["league", "home_team", "away_team"]:
        if column in merge_keys:
            data[column] = data[column].astype(str).str.strip()
    return data


def find_odds_columns(odds: pd.DataFrame) -> tuple[str, str, str]:
    if all(column in odds.columns for column in STANDARD_ODDS_COLUMNS):
        return ("odds_home", "odds_draw", "odds_away")

    if all(column in odds.columns for column in ["home_odds", "draw_odds", "away_odds"]):
        return ("home_odds", "draw_odds", "away_odds")

    for columns in RAW_ODDS_CANDIDATES:
        if all(column in odds.columns for column in columns):
            return columns

    raise ValueError(
        "No odds columns found. Expected one of: "
        "home_odds/draw_odds/away_odds, odds_home/odds_draw/odds_away, "
        "B365H/B365D/B365A, PSH/PSD/PSA, AvgH/AvgD/AvgA, MaxH/MaxD/MaxA."
    )


def prepare_odds_source(odds: pd.DataFrame, merge_keys: list[str]) -> pd.DataFrame:
    home_col, draw_col, away_col = find_odds_columns(odds)
    odds = normalize_keys(odds, merge_keys)
    output = odds[merge_keys + [home_col, draw_col, away_col]].copy()
    output = output.rename(
        columns={
            home_col: "home_odds",
            draw_col: "draw_odds",
            away_col: "away_odds",
        }
    )

    for column in ["home_odds", "draw_odds", "away_odds"]:
        output[column] = pd.to_numeric(output[column], errors="coerce")

    output = output.dropna(subset=["home_odds", "draw_odds", "away_odds"])
    output = output.drop_duplicates(subset=merge_keys, keep="first")
    return output


def add_odds_to_meta_features(
    meta_path: Path = DEFAULT_META_PATH,
    odds_source_path: Path = DEFAULT_MATCHES_PATH,
    output_path: Path = DEFAULT_OUTPUT_PATH,
) -> pd.DataFrame:
    meta = read_csv(meta_path, "meta features")
    odds_source = read_csv(odds_source_path, "odds source")
    merge_keys = choose_merge_keys(meta, odds_source)

    meta = normalize_keys(meta, merge_keys)
    odds = prepare_odds_source(odds_source, merge_keys)
    merged = meta.merge(odds, on=merge_keys, how="left", validate="many_to_one")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output_path, index=False, encoding="utf-8")
    return merged


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--meta-csv", type=Path, default=DEFAULT_META_PATH)
    parser.add_argument("--odds-source", type=Path, default=DEFAULT_MATCHES_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    args = parser.parse_args()

    merged = add_odds_to_meta_features(
        meta_path=args.meta_csv,
        odds_source_path=args.odds_source,
        output_path=args.output,
    )
    print(f"已輸出含賠率 meta features: {args.output}")
    print(f"輸出筆數: {len(merged)}")
    missing_count = int(merged[["home_odds", "draw_odds", "away_odds"]].isna().any(axis=1).sum())
    print(f"缺少賠率筆數: {missing_count}")


if __name__ == "__main__":
    main()
