import argparse
import math
from pathlib import Path

import pandas as pd


DATA_DIRS = [Path("data/processed"), Path("data/raw")]
DEFAULT_OUTPUT_PATH = Path("data/processed/worldcup_features.csv")
DEFAULT_MISSING_REPORT_PATH = Path("reports/worldcup_features_missing_data_report.csv")
MODEL_MISSING_REPORT_PATH = Path("reports/worldcup_model_missing_report.csv")
ODDS_MERGE_DEBUG_PATH = Path("reports/odds_merge_debug.csv")
MATCH_KEYS = ["date", "home_team", "away_team"]
MATCH_ID_KEY = ["match_id"]

FILE_CANDIDATES = {
    "schedule": [
        "worldcup_fixtures_resolved.csv",
        "worldcup_schedule.csv",
        "worldcup_fixtures.csv",
        "world_cup_schedule.csv",
        "world_cup_fixtures.csv",
    ],
    "results": [
        "worldcup_results.csv",
        "world_cup_results.csv",
    ],
    "market": [
        "worldcup_market_predictions.csv",
        "world_cup_market_predictions.csv",
        "worldcup_market.csv",
        "worldcup_openai_market_predictions.csv",
    ],
    "model": [
        "worldcup_model_predictions.csv",
        "world_cup_model_predictions.csv",
        "worldcup_wdl_predictions.csv",
    ],
    "poisson": [
        "worldcup_poisson_predictions.csv",
        "world_cup_poisson_predictions.csv",
        "worldcup_poisson.csv",
    ],
    "odds": [
        "worldcup_consensus_odds.csv",
        "worldcup_odds.csv",
        "world_cup_odds.csv",
        "worldcup_openai_odds.csv",
    ],
}

REQUIRED_COLUMNS = {
    "schedule": ["date", "home_team", "away_team"],
    "market": ["market_H", "market_D", "market_A"],
    "model": ["model_H", "model_D", "model_A"],
    "poisson": ["poisson_home_xg", "poisson_away_xg"],
    "odds": ["home_odds", "draw_odds", "away_odds"],
}

OUTPUT_COLUMNS = [
    "date",
    "group",
    "stage",
    "round",
    "match_id",
    "home_team",
    "away_team",
    "home_slot",
    "away_slot",
    "status",
    "neutral_venue",
    "market_type",
    "market_H",
    "market_D",
    "market_A",
    "model_H",
    "model_D",
    "model_A",
    "poisson_home_xg",
    "poisson_away_xg",
    "poisson_diff",
    "poisson_top_scores",
    "points_home_before",
    "points_away_before",
    "goal_diff_home_before",
    "goal_diff_away_before",
    "group_matchday",
    "must_win_home",
    "must_win_away",
    "already_qualified_home",
    "already_qualified_away",
    "home_odds",
    "draw_odds",
    "away_odds",
]


def find_candidate(filename_options: list[str]) -> Path | None:
    for directory in DATA_DIRS:
        for filename in filename_options:
            path = directory / filename
            if path.exists():
                return path
    return None


def resolve_path(explicit_path: Path | None, kind: str) -> Path | None:
    if explicit_path:
        return explicit_path
    return find_candidate(FILE_CANDIDATES[kind])


def candidate_paths(kind: str) -> list[Path]:
    paths = []
    for directory in DATA_DIRS:
        for filename in FILE_CANDIDATES[kind]:
            path = directory / filename
            if path.exists() and path not in paths:
                paths.append(path)
    return paths


def missing_report_row(kind: str, path: Path | None, missing: list[str], message: str) -> dict:
    return {
        "data_type": kind,
        "path": "" if path is None else str(path),
        "missing_columns": ", ".join(missing),
        "message": message,
    }


def write_missing_report(rows: list[dict], report_path: Path = DEFAULT_MISSING_REPORT_PATH) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(report_path, index=False, encoding="utf-8")


def read_required_csv(kind: str, path: Path | None, report_rows: list[dict]) -> pd.DataFrame:
    if path is None:
        report_rows.append(
            missing_report_row(
                kind,
                None,
                REQUIRED_COLUMNS.get(kind, []),
                f"找不到 World Cup {kind} CSV。",
            )
        )
        return pd.DataFrame()

    if not path.exists():
        report_rows.append(
            missing_report_row(kind, path, REQUIRED_COLUMNS.get(kind, []), f"檔案不存在: {path}")
        )
        return pd.DataFrame()

    data = pd.read_csv(path, encoding="utf-8")
    required = REQUIRED_COLUMNS.get(kind, [])
    merge_keys = MATCH_ID_KEY if "match_id" in data.columns else MATCH_KEYS
    missing = [column for column in merge_keys + required if column not in data.columns]
    if missing:
        report_rows.append(
            missing_report_row(kind, path, missing, f"World Cup {kind} CSV 缺少必要欄位。")
        )
    return data


def read_optional_csv(path: Path, required: list[str]) -> pd.DataFrame:
    try:
        data = pd.read_csv(path, encoding="utf-8")
    except pd.errors.EmptyDataError:
        return pd.DataFrame()
    missing = [column for column in required if column not in data.columns]
    if missing:
        return pd.DataFrame()
    return normalize_keys(data)


def normalize_keys(data: pd.DataFrame) -> pd.DataFrame:
    data = data.copy()
    if "date" in data.columns:
        data["date"] = pd.to_datetime(data["date"], errors="coerce").dt.date.astype(str)
    for column in ["group", "stage", "round", "match_id", "home_team", "away_team"]:
        if column in data.columns:
            data[column] = data[column].astype(str).str.strip()
    return data


def resolved_non_tbd_mask(data: pd.DataFrame) -> pd.Series:
    home = data.get("home_team", pd.Series("", index=data.index)).fillna("").astype(str).str.strip().str.upper()
    away = data.get("away_team", pd.Series("", index=data.index)).fillna("").astype(str).str.strip().str.upper()
    status = data.get("status", pd.Series("", index=data.index)).fillna("").astype(str).str.strip().str.lower()
    return home.ne("TBD") & away.ne("TBD") & status.ne("waiting_for_teams")


def write_model_missing_report(rows: pd.DataFrame) -> None:
    MODEL_MISSING_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    columns = ["date", "group", "home_team", "away_team", "status", "model_H", "model_D", "model_A"]
    output = rows.copy()
    for column in columns:
        if column not in output.columns:
            output[column] = pd.NA
    output[columns].to_csv(MODEL_MISSING_REPORT_PATH, index=False, encoding="utf-8")


def merge_model_predictions(base: pd.DataFrame, model: pd.DataFrame) -> pd.DataFrame:
    base = normalize_keys(base)
    model = normalize_keys(model)
    required = MATCH_KEYS + ["model_H", "model_D", "model_A"]
    model = model[required].drop_duplicates(MATCH_KEYS, keep="first")
    merged = base.merge(model, on=MATCH_KEYS, how="left", validate="many_to_one")
    missing_mask = resolved_non_tbd_mask(merged) & merged[["model_H", "model_D", "model_A"]].isna().any(axis=1)
    missing = merged.loc[missing_mask].copy()
    write_model_missing_report(missing)
    print(f"FEATURE_ROWS_AFTER_MODEL_MERGE={len(merged)}")
    print(f"MODEL_MISSING_ROWS={len(missing)}")
    example = merged[
        (merged["home_team"].astype(str).str.casefold() == "portugal")
        & (merged["away_team"].astype(str).str.casefold() == "spain")
    ]
    if not example.empty:
        print(f"MODEL_EXAMPLE_PORTUGAL_VS_SPAIN={example.iloc[-1][['date', 'home_team', 'away_team', 'model_H', 'model_D', 'model_A']].to_dict()}")
    if not missing.empty:
        raise ValueError(f"model 合併後仍有 {len(missing)} 筆 resolved non-TBD fixtures 缺少 model_H/D/A。")
    return merged


def choose_merge_keys(base: pd.DataFrame, other: pd.DataFrame) -> list[str]:
    if "match_id" in base.columns and "match_id" in other.columns:
        base_ids = base["match_id"].fillna("").astype(str).str.strip()
        other_ids = other["match_id"].fillna("").astype(str).str.strip()
        if base_ids.ne("").all() and other_ids.ne("").all():
            return MATCH_ID_KEY
    return MATCH_KEYS


def merge_required(base: pd.DataFrame, other: pd.DataFrame, kind: str, columns: list[str]) -> pd.DataFrame:
    merge_keys = choose_merge_keys(base, other)
    other = normalize_keys(other)
    required = merge_keys + columns
    other = other[required].drop_duplicates(merge_keys, keep="first")
    merged = base.merge(other, on=merge_keys, how="left", validate="many_to_one")
    missing_rows = int(merged[columns].isna().any(axis=1).sum())
    if missing_rows:
        raise ValueError(f"{kind} 合併後仍有 {missing_rows} 筆缺失。")
    return merged


def merge_priority_sources(paths: list[Path], required_columns: list[str]) -> pd.DataFrame:
    frames = []
    required = MATCH_KEYS + required_columns
    for priority, path in enumerate(paths):
        data = read_optional_csv(path, required)
        if data.empty:
            continue
        data = data[required].copy()
        for column in required_columns:
            data[column] = pd.to_numeric(data[column], errors="coerce")
        data = data.dropna(subset=required_columns)
        if data.empty:
            continue
        data["_priority"] = priority
        data["_source_file"] = str(path)
        frames.append(data)
    if not frames:
        return pd.DataFrame(columns=required)
    combined = pd.concat(frames, ignore_index=True)
    combined = combined.sort_values("_priority").drop_duplicates(MATCH_KEYS, keep="first")
    return combined.drop(columns=["_priority", "_source_file"])


def complete_row(data: pd.DataFrame, row: pd.Series, columns: list[str]) -> pd.Series | None:
    if data.empty:
        return None
    match = data[
        (data["date"].astype(str) == str(row["date"]))
        & (data["home_team"].astype(str).str.casefold() == str(row["home_team"]).casefold())
        & (data["away_team"].astype(str).str.casefold() == str(row["away_team"]).casefold())
    ]
    if match.empty:
        return None
    values = pd.to_numeric(match.iloc[0][columns], errors="coerce")
    if values.notna().all():
        return match.iloc[0]
    return None


def write_odds_merge_debug(schedule: pd.DataFrame, odds_paths: list[Path], final_odds: pd.DataFrame) -> None:
    manual_api_paths = [path for path in odds_paths if "openai" not in path.name.lower()]
    openai_paths = [path for path in odds_paths if "openai" in path.name.lower()]
    manual_api_odds = merge_priority_sources(manual_api_paths, ["home_odds", "draw_odds", "away_odds"])
    openai_odds = merge_priority_sources(openai_paths, ["home_odds", "draw_odds", "away_odds"])
    rows = []
    for _, row in schedule.iterrows():
        manual_row = complete_row(manual_api_odds, row, ["home_odds", "draw_odds", "away_odds"])
        openai_row = complete_row(openai_odds, row, ["home_odds", "draw_odds", "away_odds"])
        final_row = complete_row(final_odds, row, ["home_odds", "draw_odds", "away_odds"])
        final_status = "missing"
        reason = "no_complete_odds"
        if final_row is not None:
            final_status = "manual_or_api" if manual_row is not None else "openai" if openai_row is not None else "unknown"
            reason = "complete_odds_found"
        rows.append(
            {
                "match_key": f"{row['date']}|{row['home_team']}|{row['away_team']}",
                "manual_odds_found": manual_row is not None,
                "api_odds_found": False,
                "openai_odds_found": openai_row is not None,
                "home_odds": "" if final_row is None else final_row["home_odds"],
                "draw_odds": "" if final_row is None else final_row["draw_odds"],
                "away_odds": "" if final_row is None else final_row["away_odds"],
                "final_odds_status": final_status,
                "reason": reason,
            }
        )
    ODDS_MERGE_DEBUG_PATH.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(ODDS_MERGE_DEBUG_PATH, index=False, encoding="utf-8")


def merge_results(schedule: pd.DataFrame, results: pd.DataFrame) -> pd.DataFrame:
    if results.empty:
        return schedule

    result_columns = [column for column in ["home_goals", "away_goals", "actual_result", "result"] if column in results.columns]
    if not result_columns:
        return schedule

    merge_keys = choose_merge_keys(schedule, results)
    schedule = schedule.copy()
    if "date" in merge_keys:
        schedule["date"] = pd.to_datetime(schedule["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    results = normalize_keys(results)
    results = results[merge_keys + result_columns].drop_duplicates(merge_keys, keep="first")
    merged = schedule.merge(results, on=merge_keys, how="left", validate="many_to_one")
    if "date" in merged.columns:
        merged["date"] = pd.to_datetime(merged["date"], errors="coerce")
    return merged


def match_points(home_goals: float, away_goals: float) -> tuple[int, int]:
    if home_goals > away_goals:
        return 3, 0
    if home_goals < away_goals:
        return 0, 3
    return 1, 1


def rest_days(current_date: pd.Timestamp, previous_date: pd.Timestamp | None) -> float:
    if previous_date is None or pd.isna(previous_date):
        return pd.NA
    return float((current_date - previous_date).days)


def update_group_table(table: dict, team: str, goals_for: float, goals_against: float, points: int) -> None:
    row = table.setdefault(team, {"points": 0, "goal_diff": 0, "played": 0})
    row["points"] += points
    row["goal_diff"] += goals_for - goals_against
    row["played"] += 1


def already_qualified(points: int, matchday: int) -> bool:
    return matchday >= 3 and points >= 6


def must_win(points: int, matchday: int, qualified: bool) -> bool:
    return matchday >= 3 and points <= 3 and not qualified


def build_group_context(schedule: pd.DataFrame) -> pd.DataFrame:
    if "group" not in schedule.columns:
        schedule["group"] = ""
    if "stage" not in schedule.columns:
        schedule["stage"] = schedule["group"].map(lambda value: "Group Stage" if str(value).strip() else "")
    schedule = schedule.sort_values(["date", "group", "home_team", "away_team"]).reset_index(drop=True)
    group_tables: dict[str, dict] = {}
    last_played: dict[str, pd.Timestamp] = {}
    group_match_counts: dict[str, int] = {}
    rows = []

    for row in schedule.itertuples(index=False):
        group = getattr(row, "group")
        home_team = row.home_team
        away_team = row.away_team
        match_date = pd.to_datetime(row.date)
        stage = str(getattr(row, "stage", "")).strip().lower()
        if stage and stage != "group stage":
            rows.append(
                {
                    "points_home_before": 0,
                    "points_away_before": 0,
                    "goal_diff_home_before": 0,
                    "goal_diff_away_before": 0,
                    "group_matchday": 0,
                    "must_win_home": False,
                    "must_win_away": False,
                    "already_qualified_home": False,
                    "already_qualified_away": False,
                    "rest_days_home": rest_days(match_date, last_played.get(home_team)),
                    "rest_days_away": rest_days(match_date, last_played.get(away_team)),
                }
            )
            continue
        table = group_tables.setdefault(group, {})
        group_match_counts[group] = group_match_counts.get(group, 0) + 1
        matchday = int(((group_match_counts[group] - 1) // 2) + 1)

        home_state = table.get(home_team, {"points": 0, "goal_diff": 0})
        away_state = table.get(away_team, {"points": 0, "goal_diff": 0})
        home_qualified = already_qualified(home_state["points"], matchday)
        away_qualified = already_qualified(away_state["points"], matchday)

        rows.append(
            {
                "points_home_before": home_state["points"],
                "points_away_before": away_state["points"],
                "goal_diff_home_before": home_state["goal_diff"],
                "goal_diff_away_before": away_state["goal_diff"],
                "group_matchday": matchday,
                "must_win_home": must_win(home_state["points"], matchday, home_qualified),
                "must_win_away": must_win(away_state["points"], matchday, away_qualified),
                "already_qualified_home": home_qualified,
                "already_qualified_away": away_qualified,
                "rest_days_home": rest_days(match_date, last_played.get(home_team)),
                "rest_days_away": rest_days(match_date, last_played.get(away_team)),
            }
        )

        if "home_goals" in schedule.columns and "away_goals" in schedule.columns:
            home_goals = getattr(row, "home_goals", pd.NA)
            away_goals = getattr(row, "away_goals", pd.NA)
            if pd.notna(home_goals) and pd.notna(away_goals):
                home_points, away_points = match_points(float(home_goals), float(away_goals))
                update_group_table(table, home_team, float(home_goals), float(away_goals), home_points)
                update_group_table(table, away_team, float(away_goals), float(home_goals), away_points)
                last_played[home_team] = match_date
                last_played[away_team] = match_date

    return pd.DataFrame(rows)


def poisson_scorelines(home_xg: float, away_xg: float, max_goals: int = 5) -> str:
    if pd.isna(home_xg) or pd.isna(away_xg):
        return ""
    rows = []
    for home_goals in range(max_goals + 1):
        for away_goals in range(max_goals + 1):
            home_prob = math.exp(-home_xg) * home_xg**home_goals / math.factorial(home_goals)
            away_prob = math.exp(-away_xg) * away_xg**away_goals / math.factorial(away_goals)
            rows.append((f"{home_goals}-{away_goals}", home_prob * away_prob))
    total = sum(probability for _, probability in rows)
    rows = [(score, probability / total) for score, probability in rows]
    return "; ".join(f"{score}:{probability:.3f}" for score, probability in sorted(rows, key=lambda item: item[1], reverse=True)[:5])


def validate_probabilities(data: pd.DataFrame, columns: list[str], name: str, report_rows: list[dict]) -> None:
    sums = data[columns].sum(axis=1)
    bad_count = int(((sums - 1).abs() > 0.02).sum())
    if bad_count:
        report_rows.append(
            missing_report_row(name, None, columns, f"{bad_count} 筆 H/D/A 機率加總不接近 1。")
        )


def build_worldcup_features(
    schedule_path: Path | None = None,
    results_path: Path | None = None,
    market_path: Path | None = None,
    model_path: Path | None = None,
    poisson_path: Path | None = None,
    odds_path: Path | None = None,
    output_path: Path = DEFAULT_OUTPUT_PATH,
    missing_report_path: Path = DEFAULT_MISSING_REPORT_PATH,
) -> pd.DataFrame | None:
    paths = {
        "schedule": resolve_path(schedule_path, "schedule"),
        "results": resolve_path(results_path, "results"),
        "market": resolve_path(market_path, "market"),
        "model": resolve_path(model_path, "model"),
        "poisson": resolve_path(poisson_path, "poisson"),
        "odds": resolve_path(odds_path, "odds"),
    }
    report_rows = []

    schedule = read_required_csv("schedule", paths["schedule"], report_rows)
    if market_path:
        market = read_required_csv("market", paths["market"], report_rows)
    else:
        market = merge_priority_sources(candidate_paths("market"), ["market_H", "market_D", "market_A"])
        if market.empty:
            report_rows.append(missing_report_row("market", None, REQUIRED_COLUMNS["market"], "No complete market probabilities found."))
    model = read_required_csv("model", paths["model"], report_rows)
    print(f"MODEL_PREDICTIONS_ROWS={len(model)}")
    print(f"MODEL_PREDICTIONS_COLUMNS={','.join(model.columns) if not model.empty else ''}")
    if not model.empty and {"home_team", "away_team"}.issubset(model.columns):
        example_model = model[
            (model["home_team"].astype(str).str.casefold() == "portugal")
            & (model["away_team"].astype(str).str.casefold() == "spain")
        ]
        if not example_model.empty:
            print(f"MODEL_EXAMPLE_PORTUGAL_VS_SPAIN={example_model.iloc[-1].to_dict()}")
    poisson = read_required_csv("poisson", paths["poisson"], report_rows)
    if odds_path:
        odds = read_required_csv("odds", paths["odds"], report_rows)
    else:
        odds = merge_priority_sources(candidate_paths("odds"), ["home_odds", "draw_odds", "away_odds"])
        if odds.empty:
            report_rows.append(missing_report_row("odds", None, REQUIRED_COLUMNS["odds"], "No complete odds found."))
    results = pd.read_csv(paths["results"], encoding="utf-8") if paths["results"] and paths["results"].exists() else pd.DataFrame()

    if report_rows:
        if not schedule.empty and all(column in schedule.columns for column in MATCH_KEYS):
            write_odds_merge_debug(normalize_keys(schedule.copy()), candidate_paths("odds"), odds)
        write_missing_report(report_rows, missing_report_path)
        print(f"缺少必要 World Cup 資料，已輸出缺資料報告: {missing_report_path}")
        return None

    schedule = normalize_keys(schedule)
    schedule["date"] = pd.to_datetime(schedule["date"], errors="coerce")
    if schedule["date"].isna().any():
        report_rows.append(missing_report_row("schedule", paths["schedule"], ["date"], "schedule 含無效日期。"))
        write_missing_report(report_rows, missing_report_path)
        return None

    if "neutral_venue" not in schedule.columns:
        schedule["neutral_venue"] = True
    if "group" not in schedule.columns:
        schedule["group"] = ""
    for column in ["stage", "round", "match_id", "home_slot", "away_slot", "status"]:
        if column not in schedule.columns:
            schedule[column] = ""
    if "market_type" not in schedule.columns:
        schedule["market_type"] = "1x2"

    write_odds_merge_debug(schedule.assign(date=schedule["date"].dt.strftime("%Y-%m-%d")), candidate_paths("odds"), odds)

    schedule = merge_results(schedule, results)
    context = build_group_context(schedule)
    features = pd.concat([schedule.reset_index(drop=True), context], axis=1)
    features["date"] = features["date"].dt.strftime("%Y-%m-%d")

    features = merge_model_predictions(features, model)
    try:
        features = merge_required(features, market, "market", ["market_H", "market_D", "market_A"])
    except ValueError as exc:
        report_rows.append(missing_report_row("market", paths["market"], ["market_H", "market_D", "market_A"], str(exc)))
        write_missing_report(report_rows, missing_report_path)
        print(f"World Cup market data is incomplete; falling back to no-odds mode. {missing_report_path}")
        return None
    features = merge_required(features, poisson, "poisson", ["poisson_home_xg", "poisson_away_xg"])
    try:
        features = merge_required(features, odds, "odds", ["home_odds", "draw_odds", "away_odds"])
    except ValueError as exc:
        report_rows.append(missing_report_row("odds", paths["odds"], ["home_odds", "draw_odds", "away_odds"], str(exc)))
        write_missing_report(report_rows, missing_report_path)
        print(f"World Cup odds data is incomplete; falling back to no-odds mode. {missing_report_path}")
        return None

    validate_probabilities(features, ["market_H", "market_D", "market_A"], "market", report_rows)
    validate_probabilities(features, ["model_H", "model_D", "model_A"], "model", report_rows)
    if report_rows:
        write_missing_report(report_rows, missing_report_path)
        print(f"World Cup features 檢查未通過，已輸出缺資料報告: {missing_report_path}")
        return None

    features["poisson_diff"] = features["poisson_away_xg"] - features["poisson_home_xg"]
    features["poisson_top_scores"] = features.apply(
        lambda row: poisson_scorelines(row["poisson_home_xg"], row["poisson_away_xg"]),
        axis=1,
    )

    for column in OUTPUT_COLUMNS:
        if column not in features.columns:
            features[column] = pd.NA
    output = features[OUTPUT_COLUMNS].copy()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_path, index=False, encoding="utf-8")
    write_missing_report([], missing_report_path)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--schedule-csv", type=Path)
    parser.add_argument("--results-csv", type=Path)
    parser.add_argument("--market-csv", type=Path)
    parser.add_argument("--model-csv", type=Path)
    parser.add_argument("--poisson-csv", type=Path)
    parser.add_argument("--odds-csv", type=Path)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--missing-report", type=Path, default=DEFAULT_MISSING_REPORT_PATH)
    args = parser.parse_args()

    features = build_worldcup_features(
        schedule_path=args.schedule_csv,
        results_path=args.results_csv,
        market_path=args.market_csv,
        model_path=args.model_csv,
        poisson_path=args.poisson_csv,
        odds_path=args.odds_csv,
        output_path=args.output,
        missing_report_path=args.missing_report,
    )
    if features is not None:
        print(f"已建立 World Cup features: {args.output}")
        print(f"輸出筆數: {len(features)}")


if __name__ == "__main__":
    main()
