import argparse
from pathlib import Path

import pandas as pd


DEFAULT_OUTPUT_PATH = Path("data/processed/meta_features.csv")
MATCH_KEYS = ["date", "league", "home_team", "away_team"]
PROBABILITY_TOLERANCE = 0.02

MARKET_COLUMNS = ["market_H", "market_D", "market_A"]
NOMARKET_COLUMNS = ["nomarket_H", "nomarket_D", "nomarket_A"]
POISSON_COLUMNS = ["poisson_home_xg", "poisson_away_xg"]
OUTPUT_COLUMNS = [
    "date",
    "league",
    "home_team",
    "away_team",
    "market_H",
    "market_D",
    "market_A",
    "nomarket_H",
    "nomarket_D",
    "nomarket_A",
    "poisson_home_xg",
    "poisson_away_xg",
    "poisson_diff",
    "market_pick",
    "nomarket_pick",
    "market_confidence",
    "nomarket_confidence",
    "models_agree",
    "market_nomarket_disagree",
    "actual_result",
]


def read_csv(path: Path, name: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"{name} CSV not found: {path}")
    return pd.read_csv(path, encoding="utf-8")


def require_columns(data: pd.DataFrame, columns: list[str], name: str) -> None:
    missing = [column for column in columns if column not in data.columns]
    if missing:
        raise ValueError(f"{name} CSV is missing required columns: {missing}")


def choose_merge_keys(frames: dict[str, pd.DataFrame]) -> list[str]:
    if all("match_id" in frame.columns for frame in frames.values()):
        return ["match_id"]

    for name, frame in frames.items():
        require_columns(frame, MATCH_KEYS, name)
    return MATCH_KEYS


def normalize_keys(data: pd.DataFrame, merge_keys: list[str]) -> pd.DataFrame:
    data = data.copy()
    if "date" in merge_keys:
        data["date"] = pd.to_datetime(data["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    for column in ["league", "home_team", "away_team"]:
        if column in merge_keys:
            data[column] = data[column].astype(str).str.strip()
    return data


def ensure_unique_keys(data: pd.DataFrame, merge_keys: list[str], name: str) -> None:
    duplicates = data[data.duplicated(merge_keys, keep=False)]
    if not duplicates.empty:
        preview = duplicates[merge_keys].head(10).to_dict("records")
        raise ValueError(f"{name} CSV has duplicate merge keys. First duplicates: {preview}")


def validate_probability_columns(data: pd.DataFrame, columns: list[str], name: str) -> None:
    for column in columns:
        data[column] = pd.to_numeric(data[column], errors="coerce")

    if data[columns].isna().any().any():
        bad_columns = data[columns].columns[data[columns].isna().any()].tolist()
        raise ValueError(f"{name} probability columns contain missing or invalid values: {bad_columns}")

    probability_sum = data[columns].sum(axis=1)
    invalid_rows = data.index[(probability_sum - 1).abs() > PROBABILITY_TOLERANCE].tolist()
    if invalid_rows:
        raise ValueError(
            f"{name} H/D/A probabilities must sum close to 1. "
            f"Tolerance={PROBABILITY_TOLERANCE}. Bad rows: {invalid_rows[:10]}"
        )


def prepare_inputs(
    market_path: Path,
    nomarket_path: Path,
    poisson_path: Path,
    actual_path: Path,
) -> tuple[dict[str, pd.DataFrame], list[str]]:
    frames = {
        "market": read_csv(market_path, "market"),
        "nomarket": read_csv(nomarket_path, "no-market"),
        "poisson": read_csv(poisson_path, "poisson"),
        "actual": read_csv(actual_path, "actual"),
    }

    merge_keys = choose_merge_keys(frames)
    required_identity = ["match_id"] if merge_keys == ["match_id"] else MATCH_KEYS

    require_columns(frames["market"], required_identity + MARKET_COLUMNS, "market")
    require_columns(frames["nomarket"], required_identity + NOMARKET_COLUMNS, "no-market")
    require_columns(frames["poisson"], required_identity + POISSON_COLUMNS, "poisson")
    require_columns(frames["actual"], required_identity + ["actual_result"], "actual")

    if merge_keys == ["match_id"]:
        require_columns(frames["actual"], MATCH_KEYS, "actual")

    for name, frame in frames.items():
        frames[name] = normalize_keys(frame, merge_keys)
        ensure_unique_keys(frames[name], merge_keys, name)

    validate_probability_columns(frames["market"], MARKET_COLUMNS, "market")
    validate_probability_columns(frames["nomarket"], NOMARKET_COLUMNS, "no-market")

    for column in POISSON_COLUMNS:
        frames["poisson"][column] = pd.to_numeric(frames["poisson"][column], errors="coerce")
    if frames["poisson"][POISSON_COLUMNS].isna().any().any():
        raise ValueError("poisson CSV contains missing or invalid expected-goals values.")

    frames["actual"]["actual_result"] = frames["actual"]["actual_result"].astype(str).str.upper()
    invalid_results = sorted(set(frames["actual"]["actual_result"]) - {"H", "D", "A"})
    if invalid_results:
        raise ValueError(f"actual_result contains invalid classes: {invalid_results}")

    return frames, merge_keys


def merge_meta_features(frames: dict[str, pd.DataFrame], merge_keys: list[str]) -> pd.DataFrame:
    # Keep match identity from actual results, then attach model outputs by pre-match keys.
    actual_columns = merge_keys + MATCH_KEYS + ["actual_result"] if merge_keys == ["match_id"] else merge_keys + ["actual_result"]
    meta = frames["actual"][actual_columns].copy()

    meta = meta.merge(frames["market"][merge_keys + MARKET_COLUMNS], on=merge_keys, how="inner", validate="one_to_one")
    meta = meta.merge(
        frames["nomarket"][merge_keys + NOMARKET_COLUMNS],
        on=merge_keys,
        how="inner",
        validate="one_to_one",
    )
    meta = meta.merge(frames["poisson"][merge_keys + POISSON_COLUMNS], on=merge_keys, how="inner", validate="one_to_one")

    if merge_keys == ["match_id"]:
        meta = meta.drop(columns=["match_id"])

    return meta


def add_derived_features(meta: pd.DataFrame) -> pd.DataFrame:
    meta = meta.copy()
    meta["poisson_diff"] = meta["poisson_away_xg"] - meta["poisson_home_xg"]
    meta["market_pick"] = meta[MARKET_COLUMNS].idxmax(axis=1).str.replace("market_", "", regex=False)
    meta["nomarket_pick"] = meta[NOMARKET_COLUMNS].idxmax(axis=1).str.replace("nomarket_", "", regex=False)
    meta["market_confidence"] = meta[MARKET_COLUMNS].max(axis=1)
    meta["nomarket_confidence"] = meta[NOMARKET_COLUMNS].max(axis=1)
    meta["models_agree"] = meta["market_pick"] == meta["nomarket_pick"]
    meta["market_nomarket_disagree"] = meta["market_pick"] != meta["nomarket_pick"]
    return meta


def validate_output(meta: pd.DataFrame) -> None:
    missing_values = meta[OUTPUT_COLUMNS].isna().sum()
    missing_values = missing_values[missing_values > 0]
    if not missing_values.empty:
        raise ValueError(f"Output contains missing values: {missing_values.to_dict()}")


def build_meta_features(
    market_path: Path,
    nomarket_path: Path,
    poisson_path: Path,
    actual_path: Path,
    output_path: Path = DEFAULT_OUTPUT_PATH,
) -> pd.DataFrame:
    frames, merge_keys = prepare_inputs(market_path, nomarket_path, poisson_path, actual_path)
    meta = merge_meta_features(frames, merge_keys)
    meta = add_derived_features(meta)
    meta = meta[OUTPUT_COLUMNS].sort_values("date").reset_index(drop=True)
    validate_output(meta)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    meta.to_csv(output_path, index=False, encoding="utf-8")
    return meta


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--market-csv", type=Path, required=True)
    parser.add_argument("--nomarket-csv", type=Path, required=True)
    parser.add_argument("--poisson-csv", type=Path, required=True)
    parser.add_argument("--actual-csv", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    args = parser.parse_args()

    meta = build_meta_features(
        market_path=args.market_csv,
        nomarket_path=args.nomarket_csv,
        poisson_path=args.poisson_csv,
        actual_path=args.actual_csv,
        output_path=args.output,
    )
    print(f"已建立 meta features: {args.output}")
    print(f"輸出筆數: {len(meta)}")


if __name__ == "__main__":
    main()
