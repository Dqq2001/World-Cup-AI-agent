import argparse
from io import StringIO
import re
import subprocess
import sys
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
RAW_DIR = PROJECT_ROOT / "data" / "raw"
OUTPUT_PATH = PROCESSED_DIR / "worldcup_fixtures.csv"
RESOLVED_OUTPUT_PATH = PROCESSED_DIR / "worldcup_fixtures_resolved.csv"
REPORT_PATH = PROJECT_ROOT / "reports" / "worldcup_fixtures_missing_report.csv"
KNOCKOUT_DEBUG_PATH = PROJECT_ROOT / "reports" / "knockout_fixture_debug.csv"
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
    "neutral_venue",
    "status",
]
REQUIRED_COLUMNS = ["date", "home_team", "away_team", "neutral_venue"]
TOURNAMENT_START = pd.Timestamp("2026-06-11")
TOURNAMENT_END = pd.Timestamp("2026-07-19")
EXPECTED_GROUP_STAGE_MATCHES = 72

LOCAL_CANDIDATES = [
    "worldcup_fixtures.csv",
    "worldcup_schedule.csv",
    "world_cup_fixtures.csv",
    "world_cup_schedule.csv",
]

GROUP_PAGES = {
    "A": "https://en.wikipedia.org/wiki/2026_FIFA_World_Cup_Group_A",
    "B": "https://en.wikipedia.org/wiki/2026_FIFA_World_Cup_Group_B",
    "C": "https://en.wikipedia.org/wiki/2026_FIFA_World_Cup_Group_C",
    "D": "https://en.wikipedia.org/wiki/2026_FIFA_World_Cup_Group_D",
    "E": "https://en.wikipedia.org/wiki/2026_FIFA_World_Cup_Group_E",
    "F": "https://en.wikipedia.org/wiki/2026_FIFA_World_Cup_Group_F",
    "G": "https://en.wikipedia.org/wiki/2026_FIFA_World_Cup_Group_G",
    "H": "https://en.wikipedia.org/wiki/2026_FIFA_World_Cup_Group_H",
    "I": "https://en.wikipedia.org/wiki/2026_FIFA_World_Cup_Group_I",
    "J": "https://en.wikipedia.org/wiki/2026_FIFA_World_Cup_Group_J",
    "K": "https://en.wikipedia.org/wiki/2026_FIFA_World_Cup_Group_K",
    "L": "https://en.wikipedia.org/wiki/2026_FIFA_World_Cup_Group_L",
}

KNOCKOUT_SLOTS = [
    ("2026-06-29", "Round of 32", "Round of 32", "R32_01", "2nd Group A", "2nd Group B"),
    ("2026-06-29", "Round of 32", "Round of 32", "R32_02", "1st Group C", "3rd Group D/E/F"),
    ("2026-06-30", "Round of 32", "Round of 32", "R32_03", "1st Group E", "3rd Group A/B/C/D"),
    ("2026-06-30", "Round of 32", "Round of 32", "R32_04", "1st Group I", "3rd Group C/D/F/G/H"),
    ("2026-07-01", "Round of 32", "Round of 32", "R32_05", "1st Group A", "3rd Group C/E/F/H/I"),
    ("2026-07-01", "Round of 32", "Round of 32", "R32_06", "2nd Group E", "2nd Group I"),
    ("2026-07-02", "Round of 32", "Round of 32", "R32_07", "1st Group G", "3rd Group A/E/H/I/J"),
    ("2026-07-02", "Round of 32", "Round of 32", "R32_08", "2nd Group C", "2nd Group F"),
    ("2026-07-03", "Round of 32", "Round of 32", "R32_09", "1st Group B", "3rd Group E/F/G/I/J"),
    ("2026-07-03", "Round of 32", "Round of 32", "R32_10", "1st Group F", "2nd Group L"),
    ("2026-07-04", "Round of 32", "Round of 32", "R32_11", "1st Group D", "2nd Group K"),
    ("2026-07-04", "Round of 32", "Round of 32", "R32_12", "1st Group H", "2nd Group J"),
    ("2026-07-05", "Round of 32", "Round of 32", "R32_13", "1st Group J", "2nd Group H"),
    ("2026-07-05", "Round of 32", "Round of 32", "R32_14", "1st Group K", "3rd Group B/C/D/F/G"),
    ("2026-07-06", "Round of 32", "Round of 32", "R32_15", "1st Group L", "3rd Group E/H/I/J/K"),
    ("2026-07-06", "Round of 32", "Round of 32", "R32_16", "2nd Group D", "2nd Group G"),
    ("2026-07-07", "Round of 16", "Round of 16", "R16_01", "Winner R32_01", "Winner R32_02"),
    ("2026-07-07", "Round of 16", "Round of 16", "R16_02", "Winner R32_03", "Winner R32_04"),
    ("2026-07-08", "Round of 16", "Round of 16", "R16_03", "Winner R32_05", "Winner R32_06"),
    ("2026-07-08", "Round of 16", "Round of 16", "R16_04", "Winner R32_07", "Winner R32_08"),
    ("2026-07-09", "Round of 16", "Round of 16", "R16_05", "Winner R32_09", "Winner R32_10"),
    ("2026-07-09", "Round of 16", "Round of 16", "R16_06", "Winner R32_11", "Winner R32_12"),
    ("2026-07-10", "Round of 16", "Round of 16", "R16_07", "Winner R32_13", "Winner R32_14"),
    ("2026-07-10", "Round of 16", "Round of 16", "R16_08", "Winner R32_15", "Winner R32_16"),
    ("2026-07-11", "Quarter Final", "Quarter Final", "QF_01", "Winner R16_01", "Winner R16_02"),
    ("2026-07-11", "Quarter Final", "Quarter Final", "QF_02", "Winner R16_03", "Winner R16_04"),
    ("2026-07-12", "Quarter Final", "Quarter Final", "QF_03", "Winner R16_05", "Winner R16_06"),
    ("2026-07-12", "Quarter Final", "Quarter Final", "QF_04", "Winner R16_07", "Winner R16_08"),
    ("2026-07-14", "Semi Final", "Semi Final", "SF_01", "Winner QF_01", "Winner QF_02"),
    ("2026-07-15", "Semi Final", "Semi Final", "SF_02", "Winner QF_03", "Winner QF_04"),
    ("2026-07-19", "Final", "Final", "FINAL", "Winner SF_01", "Winner SF_02"),
]

PLACEHOLDER_PATTERNS = [
    "tbd",
    "to be determined",
    "winner",
    "runner-up",
    "play-off",
    "playoff",
    "qualified team",
    "team 1",
    "team 2",
]


def write_missing_report(rows: list[dict]) -> None:
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(REPORT_PATH, index=False, encoding="utf-8")


def is_actual_team_name(value: object) -> bool:
    text = str(value).strip().lower()
    if not text or text == "nan":
        return False
    fallback_tokens = ["1st group", "2nd group", "3rd group", "winner", "tbd", "to be determined"]
    return not any(token in text for token in fallback_tokens)


def write_knockout_debug(fixtures: pd.DataFrame, source: str) -> None:
    rows = []
    data = fixtures.copy()
    if "stage" not in data.columns:
        data["stage"] = ""
    knockout = data[
        data["stage"].fillna("").astype(str).str.lower().ne("group stage")
        & data["stage"].fillna("").astype(str).str.strip().ne("")
    ]
    for row in knockout.itertuples(index=False):
        home_team = getattr(row, "home_team", "")
        away_team = getattr(row, "away_team", "")
        home_slot = getattr(row, "home_slot", "")
        away_slot = getattr(row, "away_slot", "")
        is_actual = is_actual_team_name(home_team) and is_actual_team_name(away_team)
        used_fallback = not is_actual
        if is_actual:
            reason = "actual knockout teams available in source"
        else:
            reason = "source did not provide actual knockout teams; using static bracket slot fallback"
        rows.append(
            {
                "match_id": getattr(row, "match_id", ""),
                "date": getattr(row, "date", ""),
                "round": getattr(row, "round", ""),
                "home_team": home_team,
                "away_team": away_team,
                "home_slot": home_slot,
                "away_slot": away_slot,
                "status": getattr(row, "status", ""),
                "source": source,
                "is_actual_team": is_actual,
                "used_fallback": used_fallback,
                "reason": reason,
            }
        )
    KNOCKOUT_DEBUG_PATH.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(KNOCKOUT_DEBUG_PATH, index=False, encoding="utf-8")


def clean_team_name(value: object) -> str:
    text = re.sub(r"\[[^\]]+\]", "", str(value))
    text = re.sub(r"\([^)]*\)", "", text)
    text = text.replace("\xa0", " ")
    text = re.sub(r"^toc-", "", text, flags=re.IGNORECASE)
    return " ".join(text.split()).strip()


def is_real_team(value: str) -> bool:
    normalized = value.lower().strip()
    if not normalized or normalized == "nan":
        return False
    return not any(pattern in normalized for pattern in PLACEHOLDER_PATTERNS)


def is_tbd_team(value: str) -> bool:
    return str(value).strip().lower() in {"tbd", "to be determined"}


def knockout_slot_rows() -> pd.DataFrame:
    rows = []
    for date, stage, round_name, match_id, home_slot, away_slot in KNOCKOUT_SLOTS:
        rows.append(
            {
                "date": date,
                "group": "",
                "stage": stage,
                "round": round_name,
                "match_id": match_id,
                "home_team": "TBD",
                "away_team": "TBD",
                "home_slot": home_slot,
                "away_slot": away_slot,
                "neutral_venue": True,
                "status": "waiting_for_teams",
            }
        )
    return pd.DataFrame(rows)


def normalize_fixtures(data: pd.DataFrame, source: str) -> pd.DataFrame:
    data = data.copy()
    missing = [column for column in ["date", "home_team", "away_team"] if column not in data.columns]
    if missing:
        raise ValueError(f"{source} 缺少欄位: {missing}")

    data["date"] = pd.to_datetime(data["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    if "group" not in data.columns:
        data["group"] = ""
    data["group"] = data["group"].fillna("").astype(str).str.strip()
    data["home_team"] = data["home_team"].map(clean_team_name)
    data["away_team"] = data["away_team"].map(clean_team_name)
    if "neutral_venue" not in data.columns:
        data["neutral_venue"] = True
    data["neutral_venue"] = data["neutral_venue"].fillna(True).astype(bool)
    if "stage" not in data.columns:
        data["stage"] = data["group"].map(lambda value: "Group Stage" if str(value).strip() else "")
    if "round" not in data.columns:
        data["round"] = data["stage"]
    if "match_id" not in data.columns:
        data["match_id"] = ""
    if "home_slot" not in data.columns:
        data["home_slot"] = ""
    if "away_slot" not in data.columns:
        data["away_slot"] = ""
    if "status" not in data.columns:
        data["status"] = "scheduled"
    for column in ["stage", "round", "match_id", "home_slot", "away_slot", "status"]:
        data[column] = data[column].fillna("").astype(str).str.strip()

    data = data.dropna(subset=["date"])
    date_values = pd.to_datetime(data["date"], errors="coerce")
    data = data[(date_values >= TOURNAMENT_START) & (date_values <= TOURNAMENT_END)]
    valid_teams = (
        (data["home_team"].map(is_real_team) & data["away_team"].map(is_real_team))
        | (data["home_team"].map(is_tbd_team) & data["away_team"].map(is_tbd_team))
    )
    data = data[valid_teams]
    data.loc[data["home_team"].map(is_tbd_team) | data["away_team"].map(is_tbd_team), "status"] = "waiting_for_teams"
    for column in OUTPUT_COLUMNS:
        if column not in data.columns:
            data[column] = ""
    data = data[OUTPUT_COLUMNS].drop_duplicates(["date", "stage", "round", "match_id", "home_team", "away_team"])
    return data.sort_values(["date", "stage", "round", "match_id", "group", "home_team", "away_team"]).reset_index(drop=True)


def find_local_fixtures() -> Path | None:
    for directory in [PROCESSED_DIR, RAW_DIR]:
        for filename in LOCAL_CANDIDATES:
            path = directory / filename
            if path.exists():
                return path
    return None


def load_local_fixtures(path: Path) -> pd.DataFrame:
    return normalize_fixtures(pd.read_csv(path, encoding="utf-8"), str(path))


def flatten_columns(table: pd.DataFrame) -> pd.DataFrame:
    table = table.copy()
    table.columns = [
        "_".join(str(part) for part in column if str(part) != "nan").lower().replace(" ", "_")
        if isinstance(column, tuple)
        else str(column).lower().replace(" ", "_")
        for column in table.columns
    ]
    return table


def parse_match_tables(group: str, html: str) -> list[dict]:
    rows = []
    for table in pd.read_html(StringIO(html)):
        table = flatten_columns(table)
        date_col = next((column for column in table.columns if "date" in column), None)
        team1_col = next((column for column in table.columns if "team_1" in column or column == "team1"), None)
        team2_col = next((column for column in table.columns if "team_2" in column or column == "team2"), None)
        if not date_col or not team1_col or not team2_col:
            continue

        for _, row in table.iterrows():
            date = pd.to_datetime(row.get(date_col), errors="coerce")
            home_team = clean_team_name(row.get(team1_col))
            away_team = clean_team_name(row.get(team2_col))
            if pd.isna(date) or not is_real_team(home_team) or not is_real_team(away_team):
                continue
            rows.append(
                {
                    "date": date.strftime("%Y-%m-%d"),
                    "group": group,
                    "home_team": home_team,
                    "away_team": away_team,
                    "neutral_venue": True,
                }
            )
    return rows


def strip_tags(value: str) -> str:
    value = re.sub(r"<[^>]+>", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def parse_match_sections(group: str, html: str) -> list[dict]:
    rows = []
    heading_matches = list(re.finditer(r'<h3[^>]+id="([^"]+_vs_[^"]+)"[^>]*>', html))
    for index, heading_match in enumerate(heading_matches):
        section_start = heading_match.end()
        section_end = heading_matches[index + 1].start() if index + 1 < len(heading_matches) else len(html)
        section = html[section_start:section_end]
        heading = heading_match.group(1).replace("_", " ")
        if " vs " not in heading:
            continue
        home_team, away_team = [clean_team_name(part) for part in heading.split(" vs ", 1)]
        if not is_real_team(home_team) or not is_real_team(away_team):
            continue

        date_match = re.search(r"2026-\d{2}-\d{2}", section)
        if not date_match:
            continue
        date = pd.to_datetime(date_match.group(0), errors="coerce")
        if pd.isna(date) or date < TOURNAMENT_START or date > TOURNAMENT_END:
            continue

        rows.append(
            {
                "date": date.strftime("%Y-%m-%d"),
                "group": group,
                "home_team": home_team,
                "away_team": away_team,
                "neutral_venue": True,
            }
        )
    return rows


def fetch_from_wikipedia() -> pd.DataFrame:
    rows = []
    errors = []
    for group, url in GROUP_PAGES.items():
        try:
            request = Request(url, headers={"User-Agent": "worldcup-ai-agent/1.0"})
            with urlopen(request, timeout=30) as response:
                html = response.read().decode("utf-8", errors="ignore")
            rows.extend(parse_match_tables(group, html))
            rows.extend(parse_match_sections(group, html))
        except (HTTPError, URLError, ValueError) as exc:
            errors.append({"source": url, "message": str(exc)})

    if not rows:
        detail = "; ".join(f"{item['source']}: {item['message']}" for item in errors[:5])
        raise RuntimeError(f"Wikipedia 沒有可用 fixtures。{detail}")
    return normalize_fixtures(pd.DataFrame(rows), "Wikipedia group pages")


def save_fixtures(fixtures: pd.DataFrame, output_path: Path = OUTPUT_PATH, source: str = "unknown") -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fixtures.to_csv(output_path, index=False, encoding="utf-8")
    write_knockout_debug(fixtures, source)
    knockout = fixtures[
        fixtures.get("stage", pd.Series("", index=fixtures.index)).fillna("").astype(str).str.lower().ne("group stage")
        & fixtures.get("stage", pd.Series("", index=fixtures.index)).fillna("").astype(str).str.strip().ne("")
    ]
    if not knockout.empty:
        actual_mask = knockout.apply(lambda row: is_actual_team_name(row["home_team"]) and is_actual_team_name(row["away_team"]), axis=1)
        if actual_mask.any():
            RESOLVED_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
            fixtures.to_csv(RESOLVED_OUTPUT_PATH, index=False, encoding="utf-8")
        elif RESOLVED_OUTPUT_PATH.exists():
            RESOLVED_OUTPUT_PATH.unlink()


def with_knockout_slots(fixtures: pd.DataFrame) -> pd.DataFrame:
    fixtures = fixtures.copy()
    existing_match_ids = set()
    if "match_id" in fixtures.columns:
        known_rows = fixtures[
            ~fixtures["home_team"].astype(str).str.upper().eq("TBD")
            & ~fixtures["away_team"].astype(str).str.upper().eq("TBD")
        ]
        existing_match_ids = set(known_rows["match_id"].fillna("").astype(str).str.strip())
    if "stage" in fixtures.columns:
        is_knockout = fixtures["stage"].fillna("").astype(str).str.lower().ne("group stage") & fixtures["stage"].fillna("").astype(str).str.strip().ne("")
        is_tbd_knockout = is_knockout & fixtures["home_team"].astype(str).str.upper().eq("TBD") & fixtures["away_team"].astype(str).str.upper().eq("TBD")
        fixtures = fixtures[~is_tbd_knockout]
    knockout_slots = knockout_slot_rows()
    if existing_match_ids:
        knockout_slots = knockout_slots[~knockout_slots["match_id"].isin(existing_match_ids)]
    combined = pd.concat([fixtures, knockout_slots], ignore_index=True)
    return normalize_fixtures(combined, "fixtures plus knockout slots")


def fetch_fixtures(include_knockout: bool = True, force_refresh: bool = False) -> pd.DataFrame:
    local_path = find_local_fixtures()
    if local_path and not force_refresh:
        fixtures = with_knockout_slots(load_local_fixtures(local_path))
        if not include_knockout:
            fixtures = fixtures[fixtures["stage"].astype(str).str.lower().eq("group stage")]
        if len(fixtures) >= EXPECTED_GROUP_STAGE_MATCHES:
            save_fixtures(fixtures, source="local_file")
            print(f"已使用本機 World Cup fixtures: {local_path} ({len(fixtures)} rows)")
            return fixtures
        print(f"本機 fixtures 不完整: {local_path} ({len(fixtures)} rows)，改從公開資料源嘗試。")

    try:
        fixtures = fetch_from_wikipedia()
    except Exception:
        if local_path:
            fixtures = with_knockout_slots(load_local_fixtures(local_path))
            if not include_knockout:
                fixtures = fixtures[fixtures["stage"].astype(str).str.lower().eq("group stage")]
            save_fixtures(fixtures, source="cached_local_after_public_refresh_failed")
            print(f"Public fixtures refresh failed; using cached local fixtures: {local_path} ({len(fixtures)} rows)")
            return fixtures
        raise
    if len(fixtures) < EXPECTED_GROUP_STAGE_MATCHES:
        raise RuntimeError(f"公開資料源只取得 {len(fixtures)} 筆 fixtures，少於預期 {EXPECTED_GROUP_STAGE_MATCHES} 筆。")
    if include_knockout:
        fixtures = with_knockout_slots(fixtures)
    source = "wikipedia_group_pages_plus_static_knockout_slots" if include_knockout else "wikipedia_group_pages"
    save_fixtures(fixtures, source=source)
    print(f"已從公開資料源取得 World Cup fixtures: {OUTPUT_PATH} ({len(fixtures)} rows)")
    return fixtures


def run_command(command: list[str]) -> bool:
    print(f"執行: {' '.join(command)}")
    completed = subprocess.run(command, cwd=PROJECT_ROOT, check=False)
    if completed.returncode != 0:
        print(f"命令失敗，exit code={completed.returncode}: {' '.join(command)}")
        return False
    return True


def run_downstream_pipeline() -> None:
    commands = [
        [sys.executable, "scripts/export_worldcup_model_predictions.py"],
        [sys.executable, "scripts/export_worldcup_poisson_predictions.py"],
        [sys.executable, "scripts/build_worldcup_features.py"],
        [sys.executable, "scripts/run_worldcup_betting_agents.py"],
    ]
    for command in commands:
        if not run_command(command):
            break


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-downstream", action="store_true")
    parser.add_argument("--include-knockout", action="store_true", default=True)
    parser.add_argument("--force-refresh", action="store_true")
    args = parser.parse_args()

    try:
        fetch_fixtures(include_knockout=args.include_knockout, force_refresh=args.force_refresh)
    except Exception as exc:
        write_missing_report(
            [
                {
                    "data_type": "worldcup_fixtures",
                    "output_path": str(OUTPUT_PATH),
                    "missing_columns": ", ".join(REQUIRED_COLUMNS),
                    "message": f"無法取得 World Cup fixtures: {exc}",
                }
            ]
        )
        raise SystemExit(f"錯誤: 無法取得 World Cup fixtures，已輸出 missing report: {REPORT_PATH}") from exc

    if not args.skip_downstream:
        run_downstream_pipeline()


if __name__ == "__main__":
    main()
