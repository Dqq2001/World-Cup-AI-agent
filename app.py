from __future__ import annotations

import json
import sys
from datetime import timedelta
from difflib import SequenceMatcher
from pathlib import Path
from urllib.parse import quote, unquote

import pandas as pd
import streamlit as st

from backend import api as backend_api
from backend import refresh_controller
from services import daily_brief_service, results_service, review_service
from src.config import NO_PAID_API_MODE


st.set_page_config(
    page_title="World Cup 2026",
    layout="wide",
    initial_sidebar_state="expanded",
)

for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, "reconfigure"):
        stream.reconfigure(encoding="utf-8")


PROJECT_ROOT = Path(__file__).resolve().parent
REPORTS_DIR = PROJECT_ROOT / "reports"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MANUAL_DIR = PROJECT_ROOT / "data" / "manual"
CACHE_DIR = PROJECT_ROOT / "data" / "cache"
DASHBOARD_REFRESH_PATH = CACHE_DIR / "dashboard_last_refresh.json"
SCHEDULER_STATUS_PATH = CACHE_DIR / "scheduler_status.json"
REFRESH_COOLDOWN_MINUTES = 15
NO_ODDS_BANNER_TEXT = (
    "NO-ODDS MODE (Safe Mode) · "
    "No odds available. Recommendations are based on model and risk analysis only."
)

PREDICTION_CANDIDATES = [
    REPORTS_DIR / "worldcup_betting_predictions.csv",
    REPORTS_DIR / "worldcup_model_only_predictions.csv",
]
DAILY_INTEL_PATH = REPORTS_DIR / "worldcup_daily_intel.csv"
DAILY_BRIEF_PATH = REPORTS_DIR / "worldcup_daily_betting_brief.md"
DAILY_PREDICTION_REVIEW_PATH = REPORTS_DIR / "daily_prediction_vs_result.csv"
DAILY_PREDICTION_SUMMARY_PATH = REPORTS_DIR / "daily_prediction_summary.csv"
DAILY_PREDICTION_ERROR_ANALYSIS_PATH = REPORTS_DIR / "daily_prediction_error_analysis.csv"
PREDICTION_ACCURACY_ANALYSIS_PATH = REPORTS_DIR / "error_pattern_analysis.csv"
RISK_SIGNAL_ACCURACY_PATH = REPORTS_DIR / "risk_signal_accuracy.csv"
HIGH_DRAW_RISK_PATH = REPORTS_DIR / "worldcup_high_draw_risk_matches.csv"
OPENAI_INTEL_MISSING_REPORT_PATH = REPORTS_DIR / "openai_intel_missing_report.csv"
UNIFIED_MATCH_VIEW_DEBUG_PATH = REPORTS_DIR / "unified_match_view_debug.csv"
FIXTURES_PATH = PROCESSED_DIR / "worldcup_fixtures.csv"
RESOLVED_FIXTURES_PATH = PROCESSED_DIR / "worldcup_fixtures_resolved.csv"
WORLDCUP_RESULTS_PATH = PROCESSED_DIR / "worldcup_results.csv"
HISTORY_PATH = PROCESSED_DIR / "international_training_data.csv"
MANUAL_ODDS_PATH = MANUAL_DIR / "worldcup_odds_manual.csv"
MANUAL_RESULTS_PATH = MANUAL_DIR / "worldcup_results_manual.csv"
MANUAL_RESULTS_TEMPLATE_PATH = MANUAL_DIR / "worldcup_results_manual_template.csv"
MANUAL_INTEL_PATH = MANUAL_DIR / "worldcup_intel_manual.csv"
MANUAL_INTEL_TEMPLATE_PATH = MANUAL_DIR / "worldcup_intel_manual_template.csv"

TODAY = pd.Timestamp.today().normalize()
TODAY_WINDOW_START = TODAY - timedelta(days=1)

TEAM_TO_COUNTRY_CODE = {
    "Algeria": "dz",
    "Argentina": "ar",
    "Australia": "au",
    "Austria": "at",
    "Belgium": "be",
    "Bosnia and Herzegovina": "ba",
    "Brazil": "br",
    "Canada": "ca",
    "Cape Verde": "cv",
    "Colombia": "co",
    "Croatia": "hr",
    "Curacao": "cw",
    "Curaçao": "cw",
    "Czech Republic": "cz",
    "DR Congo": "cd",
    "Ecuador": "ec",
    "Egypt": "eg",
    "England": "gb-eng",
    "France": "fr",
    "Germany": "de",
    "Ghana": "gh",
    "Haiti": "ht",
    "Iran": "ir",
    "Iraq": "iq",
    "Ivory Coast": "ci",
    "Japan": "jp",
    "Jordan": "jo",
    "Mexico": "mx",
    "Morocco": "ma",
    "Netherlands": "nl",
    "New Zealand": "nz",
    "Norway": "no",
    "Panama": "pa",
    "Paraguay": "py",
    "Portugal": "pt",
    "Qatar": "qa",
    "Saudi Arabia": "sa",
    "Scotland": "gb-sct",
    "Senegal": "sn",
    "South Africa": "za",
    "South Korea": "kr",
    "Spain": "es",
    "Sweden": "se",
    "Switzerland": "ch",
    "Tunisia": "tn",
    "Turkey": "tr",
    "United States": "us",
    "Uruguay": "uy",
    "Uzbekistan": "uz",
}

TEAM_ALIASES = {
    "BRA": "Brazil",
    "MAR": "Morocco",
    "HTI": "Haiti",
    "USA": "United States",
    "US": "United States",
    "USMNT": "United States",
    "BIH": "Bosnia and Herzegovina",
    "KOR": "South Korea",
    "Korea Republic": "South Korea",
    "Czechia": "Czech Republic",
    "Côte d'Ivoire": "Ivory Coast",
    "Cote d'Ivoire": "Ivory Coast",
    "Curaçao": "Curacao",
    "DR Congo": "DR Congo",
    "Congo DR": "DR Congo",
    "Democratic Republic of Congo": "DR Congo",
    "Bosnia & Herzegovina": "Bosnia and Herzegovina",
    "Bosnia-Herzegovina": "Bosnia and Herzegovina",
}


def now_local() -> pd.Timestamp:
    return pd.Timestamp.now().floor("s")


def clean_utf8_text(text: str) -> str:
    return text.encode("utf-8", "ignore").decode("utf-8")


def read_refresh_state() -> dict:
    if not DASHBOARD_REFRESH_PATH.exists():
        return {}
    try:
        return json.loads(DASHBOARD_REFRESH_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def write_refresh_state(status: str, message: str = "") -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "last_refresh": now_local().isoformat(),
        "status": status,
        "message": message,
    }
    DASHBOARD_REFRESH_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def refresh_minutes_remaining(state: dict) -> int:
    last_refresh = pd.to_datetime(state.get("last_refresh"), errors="coerce")
    if pd.isna(last_refresh):
        return 0
    elapsed_minutes = (now_local() - last_refresh).total_seconds() / 60
    return max(0, int(REFRESH_COOLDOWN_MINUTES - elapsed_minutes))


def refresh_status_text(state: dict) -> tuple[str, str]:
    last_refresh = pd.to_datetime(state.get("last_refresh"), errors="coerce")
    if pd.isna(last_refresh):
        return "Never", "0 minutes"
    return last_refresh.strftime("%Y-%m-%d %H:%M"), f"{refresh_minutes_remaining(state)} minutes"


def read_scheduler_status() -> dict:
    if not SCHEDULER_STATUS_PATH.exists():
        return {}
    try:
        return json.loads(SCHEDULER_STATUS_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def job_status_text(status: dict, job_name: str) -> tuple[str, str, str]:
    job = status.get(job_name, {})
    return (
        job.get("last_run", "Never"),
        job.get("next_run", "Scheduler not running"),
        job.get("last_status", "unknown"),
    )


def run_refresh_script(script: str, username: str | None = None, password: str | None = None) -> bool:
    response = backend_api.run_script(username, password, script)
    message = response.get("message", "")
    if response.get("http_status") == 403:
        st.session_state["refresh_warning"] = "Invalid admin credentials"
        return False
    if response.get("rate_limited"):
        st.session_state["refresh_warning"] = "Rate limited. Using cached data."
        st.cache_data.clear()
        return True
    if response.get("status") != "success":
        st.session_state["refresh_warning"] = f"Refresh failed. Using cached data.\n{message[-800:]}"
        st.cache_data.clear()
        return False
    st.session_state.pop("refresh_warning", None)
    st.cache_data.clear()
    return True


def query_param_value(name: str) -> str | None:
    try:
        value = st.query_params.get(name)
    except Exception:
        return None
    if isinstance(value, list):
        return value[0] if value else None
    return value


def apply_query_params() -> None:
    page = query_param_value("page")
    match_key_value = query_param_value("match_key")
    if page:
        st.session_state["page"] = page
    if match_key_value:
        st.session_state["selected_match_key"] = unquote(match_key_value)


def run_dashboard_refresh(force: bool = False, username: str | None = None, password: str | None = None) -> bool:
    state = read_refresh_state()
    if not force and refresh_minutes_remaining(state) > 0:
        return False

    response = backend_api.refresh_all(username, password, force=force, date_text=TODAY.strftime("%Y-%m-%d"))
    message = response.get("message", "")
    if response.get("http_status") == 403:
        st.session_state["refresh_warning"] = "Invalid admin credentials"
        return False
    status = response.get("status", "failed")
    state_status = "rate_limited" if response.get("rate_limited") else status
    write_refresh_state(state_status, message[-2000:])
    st.cache_data.clear()

    if response.get("rate_limited"):
        st.session_state["refresh_warning"] = "Rate limited. Using cached data."
    elif status != "success":
        st.session_state["refresh_warning"] = "Refresh pipeline failed. Using cached data."
    else:
        st.session_state.pop("refresh_warning", None)
    return True


def maybe_auto_refresh_dashboard() -> None:
    if backend_api.admin_credentials_configured():
        return
    if st.session_state.get("dashboard_auto_refresh_checked"):
        return
    st.session_state["dashboard_auto_refresh_checked"] = True
    run_dashboard_refresh(force=False)


def inject_style() -> None:
    st.markdown(
        """
        <style>
        :root {
            --page: #fffbf7;
            --cream: #fff4e5;
            --panel: #ffffff;
            --line: #eedcc2;
            --orange: #ff8a00;
            --ink: #17202a;
            --muted: #6b5a49;
        }
        .stApp {
            background: var(--page);
            color: var(--ink);
        }
        section[data-testid="stSidebar"] {
            width: 260px !important;
            min-width: 260px !important;
            background: linear-gradient(180deg, #fff4e5 0%, #fffbf7 70%, #fff4e5 100%);
            border-right: 1px solid var(--line);
            box-shadow: 8px 0 24px rgba(105, 70, 32, 0.05);
        }
        section[data-testid="stSidebar"] > div {
            padding: 1.25rem 1rem;
        }
        .brand-row {
            display: flex;
            align-items: center;
            gap: 10px;
            white-space: nowrap;
            margin-bottom: 0;
        }
        .brand-icon {
            font-size: 28px;
            line-height: 1;
        }
        .brand-title {
            font-size: 22px;
            font-weight: 800;
            white-space: nowrap;
            line-height: 1.1;
            color: var(--ink);
        }
        .brand-subtitle {
            margin-top: 8px;
            margin-bottom: 22px;
            font-size: 14px;
            color: #777;
        }
        .main .block-container {
            padding-top: 1rem !important;
            padding-left: 2rem !important;
            padding-right: 3rem !important;
            max-width: 100% !important;
        }
        h1, h2, h3 {
            line-height: 1.25 !important;
            overflow: visible !important;
        }
        .section-header {
            height: 56px;
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 1rem;
            margin: 0 0 16px;
        }
        .two-col-section {
            height: 760px;
            display: flex;
            flex-direction: column;
        }
        .section-body {
            flex: 1;
            overflow-y: auto;
        }
        .section-title {
            display: flex;
            align-items: center;
            gap: 0.7rem;
            font-size: 1.35rem;
            font-weight: 900;
            color: var(--ink);
        }
        .section-action {
            border: 1px solid var(--line);
            background: #ffffff;
            border-radius: 0.75rem;
            padding: 0.55rem 0.9rem;
            font-weight: 800;
            box-shadow: 0 6px 18px rgba(105, 70, 32, 0.06);
        }
        .cards-scroll,
        .left-column,
        .right-column {
            display: flex;
            flex-direction: column;
            gap: 0.85rem;
            height: 700px;
            overflow-y: auto;
            overflow-x: hidden;
            padding: 0.25rem 0.35rem 0.8rem 0;
        }
        .card-base,
        .match-card {
            width: 100%;
            box-sizing: border-box;
            background: rgba(255, 255, 255, 0.92);
            border: 1px solid var(--line);
            border-radius: 20px;
            padding: 20px;
            margin-bottom: 18px;
            box-shadow: 0 10px 28px rgba(105, 70, 32, 0.08);
            min-height: 250px;
            height: 250px;
            display: flex;
            flex-direction: column;
            justify-content: space-between;
        }
        .card-top {
            display: flex;
            justify-content: space-between;
            color: #a64b00;
            font-size: 0.76rem;
            font-weight: 900;
            text-transform: uppercase;
        }
        .prediction-row, .result-row {
            display: grid;
            gap: 0.7rem;
            align-items: center;
        }
        .prediction-row {
            grid-template-columns: 1fr 38px 1fr 145px;
            margin-top: 0.9rem;
        }
        .result-row {
            grid-template-columns: 1fr 84px 1fr 150px;
            margin-top: 0.9rem;
        }
        .team-block {
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            gap: 0.55rem;
            min-width: 140px;
            text-align: center;
        }
        .flag-img {
            width: 56px;
            border-radius: 0.35rem;
            box-shadow: 0 4px 12px rgba(60, 35, 12, 0.12);
        }
        .flag-fallback {
            width: 56px;
            border: 1px solid var(--line);
            border-radius: 0.35rem;
            padding: 0.45rem 0;
            background: #fff8ed;
            color: var(--muted);
            font-weight: 900;
        }
        .team-name {
            font-weight: 850;
            line-height: 1.2;
            font-size: 0.92rem;
        }
        .vs {
            text-align: center;
            font-weight: 950;
        }
        .action-badge {
            display: inline-flex;
            justify-content: center;
            min-width: 82px;
            border-radius: 0.55rem;
            padding: 0.42rem 0.8rem;
            font-weight: 950;
        }
        .action-watch { background: #ffe08b; color: #3f2b00; }
        .action-pass { background: #d9dde2; color: #1f2933; }
        .action-bet { background: #0aa85a; color: #ffffff; }
        .action-small { background: #bce7a8; color: #1e4b23; }
        .risk-lines {
            line-height: 1.85;
            font-size: 0.88rem;
        }
        .risk-low { color: #079044; font-weight: 900; }
        .risk-medium { color: #ff6b00; font-weight: 900; }
        .risk-high { color: #dd1d2d; font-weight: 900; }
        .score-box {
            display: block;
            width: 140px;
            box-sizing: border-box;
            text-align: center;
            border: 1px solid #9fd2ad;
            border-radius: 0.55rem;
            background: #edf9ef;
            color: #08753d;
            padding: 0.35rem 0.5rem;
            font-size: 28px;
            font-weight: 800;
        }
        .result-card {
            display: flex;
            flex-direction: column;
            width: 100%;
            box-sizing: border-box;
            padding: 12px 16px;
            margin-bottom: 18px;
            border: 1px solid var(--line);
            border-radius: 20px;
            background: rgba(255, 255, 255, 0.92);
            box-shadow: 0 10px 28px rgba(105, 70, 32, 0.08);
        }
        .result-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            width: 100%;
            color: #a64b00;
            font-size: 0.76rem;
            font-weight: 900;
            text-transform: uppercase;
            margin-bottom: 8px;
        }
        .result-card .flag-img {
            width: 44px;
        }
        .result-card .flag-fallback {
            width: 44px;
            padding: 0.35rem 0;
        }
        .result-card .team-block {
            gap: 0.35rem;
        }
        .result-card .team-name {
            font-size: 0.86rem;
        }
        .result-body {
            display: grid;
            grid-template-columns: 1fr 180px 1fr;
            align-items: center;
            justify-items: center;
            width: 100%;
            min-width: 460px;
        }
        .score-block {
            width: 180px;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
        }
        .score-note {
            margin-top: 8px;
            max-width: 220px;
            text-align: center;
            color: #6f4a23;
            font-size: 0.78rem;
            font-weight: 750;
            line-height: 1.35;
        }
        .result-footer {
            display: grid;
            grid-template-columns: 1fr 180px 1fr;
            justify-items: center;
            width: 100%;
            min-width: 460px;
            margin-top: 8px;
        }
        .view-analysis-button {
            grid-column: 2;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            width: 180px;
            height: 38px;
            border: 1px solid #d0d5dd;
            border-radius: 14px;
            background: #ffffff;
            color: var(--ink);
            font-weight: 850;
            text-decoration: none;
        }
        .analysis-link {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            border: 1px solid var(--line);
            border-radius: 0.7rem;
            background: #ffffff;
            padding: 0.55rem 0.9rem;
            font-weight: 850;
            box-shadow: 0 6px 18px rgba(105, 70, 32, 0.05);
        }
        .view-btn,
        div[data-testid="stButton"] > button {
            min-width: 180px;
            height: 44px;
            border-radius: 14px;
            font-weight: 850;
        }
        .list-panel {
            border: 1px solid var(--line);
            border-radius: 1rem;
            background: rgba(255, 255, 255, 0.58);
            padding: 1rem 1.1rem;
            box-shadow: 0 10px 28px rgba(105, 70, 32, 0.06);
        }
        .view-all {
            display: block;
            width: min(220px, 100%);
            margin: 0.5rem auto 0;
            border: 1px solid var(--line);
            border-radius: 0.75rem;
            background: #ffffff;
            color: var(--orange);
            padding: 0.55rem 1rem;
            text-align: center;
            font-weight: 900;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data
def load_csv_cached(path_text: str, modified_time: float) -> pd.DataFrame:
    path = Path(path_text)
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path, encoding="utf-8")
    except UnicodeDecodeError:
        try:
            return pd.read_csv(path, encoding="utf-8-sig")
        except (pd.errors.EmptyDataError, UnicodeDecodeError):
            return pd.DataFrame()
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def load_csv(path: Path) -> pd.DataFrame:
    modified_time = path.stat().st_mtime if path.exists() else 0.0
    return load_csv_cached(str(path), modified_time)


def openai_proxy_warning() -> str:
    report = load_csv(OPENAI_INTEL_MISSING_REPORT_PATH)
    if report.empty or "error_type" not in report.columns:
        return ""
    last = report.iloc[-1]
    error_type = str(last.get("error_type", "")).strip()
    error_message = str(last.get("error_message", "")).strip()
    status_code = str(last.get("status_code", "")).strip()
    if error_type == "proxy_blocked_or_provider_forbidden" or (status_code == "403" and "1010" in error_message):
        return "OpenAI proxy unavailable: provider returned 403 / 1010. Using manual/unknown fallback."
    return ""


def normalize_keys(data: pd.DataFrame) -> pd.DataFrame:
    data = data.copy()
    if "date" in data.columns:
        data["date"] = pd.to_datetime(data["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    for column in ["group", "stage", "round", "match_id", "home_team", "away_team", "status"]:
        if column in data.columns:
            data[column] = data[column].astype(str).str.strip()
    return data


def is_waiting_for_teams(row: pd.Series | dict) -> bool:
    return (
        str(safe_value(row, "home_team", "")).strip().upper() == "TBD"
        or str(safe_value(row, "away_team", "")).strip().upper() == "TBD"
        or str(safe_value(row, "status", "")).strip().lower() == "waiting_for_teams"
    )


def stage_label(row: pd.Series | dict) -> str:
    stage = str(safe_value(row, "stage", "")).strip()
    round_name = str(safe_value(row, "round", "")).strip()
    group = str(safe_value(row, "group", "")).strip()
    if stage and stage.lower() not in {"", "unknown", "group stage"}:
        if round_name and round_name.lower() != "unknown":
            return round_name
        return stage
    if group and group.lower() not in {"", "unknown", "nan", "none"}:
        return f"Group {group}"
    if stage and stage.lower() not in {"", "unknown", "nan", "none"}:
        return stage
    return "World Cup 2026"


def is_knockout_row(row: pd.Series | dict) -> bool:
    label = f"{safe_value(row, 'stage', '')} {safe_value(row, 'round', '')}".lower()
    return any(term in label for term in ["round of", "quarter", "semi", "final"])


def normalize_team_name(value) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip()
    text = TEAM_ALIASES.get(text, text)
    text = text.replace("&", "and")
    text = " ".join(text.split())
    return text.casefold()


def canonical_team_name(value) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip()
    text = TEAM_ALIASES.get(text, text)
    text = text.replace("&", "and")
    return " ".join(text.split())


def add_match_merge_keys(data: pd.DataFrame) -> pd.DataFrame:
    data = data.copy()
    data["merge_date"] = pd.to_datetime(data.get("date"), errors="coerce").dt.strftime("%Y-%m-%d")
    data["merge_home_team"] = data.get("home_team", pd.Series(index=data.index, dtype=str)).map(normalize_team_name)
    data["merge_away_team"] = data.get("away_team", pd.Series(index=data.index, dtype=str)).map(normalize_team_name)
    return data


def add_dashboard_merge_key(data: pd.DataFrame) -> pd.DataFrame:
    data = data.copy()
    match_id = data.get("match_id", pd.Series("", index=data.index)).fillna("").astype(str).str.strip()
    fallback = (
        pd.to_datetime(data.get("date"), errors="coerce").dt.strftime("%Y-%m-%d")
        + "|"
        + data.get("home_team", pd.Series(index=data.index, dtype=str)).map(normalize_team_name)
        + "|"
        + data.get("away_team", pd.Series(index=data.index, dtype=str)).map(normalize_team_name)
    )
    data["_dashboard_merge_key"] = match_id.where(match_id.ne(""), fallback)
    return data


def load_predictions() -> pd.DataFrame:
    return results_service.load_predictions()


def load_latest_fixtures() -> pd.DataFrame:
    return results_service.load_latest_fixtures()


def combined_match_data() -> pd.DataFrame:
    return results_service.combined_match_data()


def safe_value(row: pd.Series | dict, column: str, default: str = "unknown"):
    if column not in row:
        return default
    value = row[column]
    if pd.isna(value):
        return default
    return value


def safe_float(row: pd.Series, column: str) -> float | None:
    value = safe_value(row, column, None)
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def safe_bool(row: pd.Series | dict, column: str) -> bool:
    value = safe_value(row, column, False)
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes"}


def format_prob(value) -> str:
    if value is None or pd.isna(value):
        return "unknown"
    return f"{float(value):.1%}"


def format_num(value, digits: int = 2) -> str:
    if value is None or pd.isna(value):
        return "unknown"
    return f"{float(value):.{digits}f}"


def flag_icon(team: str) -> str:
    code = TEAM_TO_COUNTRY_CODE.get(str(team))
    if not code:
        return '<span class="flag-fallback">⚑</span>'
    return f'<img class="flag-img" src="https://flagcdn.com/w80/{code}.png" alt="{team} flag">'


def team_block(team: str, extra_class: str = "") -> str:
    class_name = f"team-block {extra_class}".strip()
    return f'<div class="{class_name}">{flag_icon(team)}<div class="team-name">{team}</div></div>'


def display_team_name(row: pd.Series | dict, side: str) -> str:
    team_column = f"{side}_team"
    slot_column = f"{side}_slot"
    team = str(safe_value(row, team_column, "TBD")).strip()
    slot = str(safe_value(row, slot_column, "")).strip()
    if team.upper() == "TBD" and slot and slot.lower() != "unknown":
        return slot
    return team


def risk_class(level: str) -> str:
    return {"LOW": "risk-low", "MEDIUM": "risk-medium", "HIGH": "risk-high", "UNKNOWN": "risk-medium"}.get(str(level).upper(), "")


def action_class(action: str) -> str:
    normalized = str(action).upper().replace(" ", "_")
    return {
        "BET": "action-bet",
        "SMALL_BET": "action-small",
        "WATCH": "action-watch",
        "PASS": "action-pass",
    }.get(normalized, "action-watch")


def odds_missing(data: pd.DataFrame) -> bool:
    odds_cols = ["market_H", "market_D", "market_A"]
    if not all(column in data.columns for column in odds_cols):
        return True
    return data[odds_cols].isna().all(axis=None)


def today_data(data: pd.DataFrame) -> pd.DataFrame:
    if data.empty:
        return data
    output = data.copy()
    output["date_dt"] = pd.to_datetime(output["date"], errors="coerce")
    return output[(output["date_dt"] >= TODAY_WINDOW_START) & (output["date_dt"] <= TODAY)].drop(columns=["date_dt"]).copy()


def has_match_score(row: pd.Series | dict) -> bool:
    return safe_float(row, "home_goals") is not None and safe_float(row, "away_goals") is not None


def display_mode_for_match(row: pd.Series | dict) -> str:
    status = str(safe_value(row, "status", "")).strip().lower()
    if status in {"completed", "complete", "final"} or has_match_score(row):
        return "result"
    return "prediction"


def espn_status_label(row: pd.Series | dict) -> str:
    status = str(safe_value(row, "status", "scheduled")).strip().lower()
    if display_mode_for_match(row) == "result":
        return "Complete"
    if status in {"live", "in_progress", "in progress"}:
        return "Live"
    if is_waiting_for_teams(row):
        return "Waiting for teams"
    return "Scheduled"


def unified_sort_rank(row: pd.Series | dict) -> int:
    status = str(safe_value(row, "status", "")).strip().lower()
    if status in {"live", "in_progress", "in progress"}:
        return 0
    if display_mode_for_match(row) == "prediction":
        return 1
    return 2


def unified_match_view(data: pd.DataFrame) -> pd.DataFrame:
    matches = results_service.unified_match_view(data, today=TODAY, window_start=TODAY_WINDOW_START)
    if not matches.empty:
        matches["status_label"] = matches.apply(espn_status_label, axis=1)
    return matches


def write_unified_match_view_debug(matches: pd.DataFrame) -> None:
    columns = [
        "event_id",
        "date",
        "home_team",
        "away_team",
        "espn_status",
        "has_score",
        "display_mode",
        "source",
        "reason",
    ]
    rows = []
    for _, row in matches.iterrows():
        has_score = has_match_score(row)
        display_mode = display_mode_for_match(row)
        status = str(safe_value(row, "status", "unknown"))
        if display_mode == "result":
            reason = "score_or_completed_status"
        elif is_waiting_for_teams(row):
            reason = "waiting_for_teams"
        else:
            reason = "scheduled_without_final_score"
        rows.append(
            {
                "event_id": safe_value(row, "source_event_id", safe_value(row, "match_id", "")),
                "date": safe_value(row, "date", ""),
                "home_team": safe_value(row, "home_team", ""),
                "away_team": safe_value(row, "away_team", ""),
                "espn_status": status,
                "has_score": has_score,
                "display_mode": display_mode,
                "source": safe_value(row, "source", "unknown"),
                "reason": reason,
            }
        )
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=columns).to_csv(UNIFIED_MATCH_VIEW_DEBUG_PATH, index=False, encoding="utf-8")


def normalize_result_columns(data: pd.DataFrame) -> pd.DataFrame:
    data = data.copy()
    rename_map = {
        "home_score": "home_goals",
        "away_score": "away_goals",
        "home_goal": "home_goals",
        "away_goal": "away_goals",
        "home_goals_ft": "home_goals",
        "away_goals_ft": "away_goals",
    }
    data = data.rename(columns={old: new for old, new in rename_map.items() if old in data.columns})
    return data


def worldcup_fixture_keys() -> pd.DataFrame:
    return results_service.worldcup_fixture_keys()


def load_worldcup_results() -> tuple[pd.DataFrame, str]:
    return results_service.load_worldcup_results(today=TODAY)


def write_knockout_results_display_debug(all_results: pd.DataFrame, visible_results: pd.DataFrame) -> None:
    if all_results.empty:
        return
    knockout = all_results[all_results.apply(is_knockout_row, axis=1)].copy()
    if knockout.empty:
        return
    visible_keys = set()
    if not visible_results.empty:
        visible_keys = set(
            (
                visible_results["date"].astype(str)
                + "|"
                + visible_results["home_team"].astype(str)
                + "|"
                + visible_results["away_team"].astype(str)
            ).tolist()
        )
    knockout["_display_key"] = (
        knockout["date"].astype(str)
        + "|"
        + knockout["home_team"].astype(str)
        + "|"
        + knockout["away_team"].astype(str)
    )
    knockout["display_text"] = knockout.apply(result_display_text, axis=1)
    knockout["shown_in_dashboard"] = knockout["_display_key"].isin(visible_keys)
    knockout["reason_if_not_shown"] = ""
    knockout.loc[~knockout["shown_in_dashboard"], "reason_if_not_shown"] = "outside_today_yesterday_results_panel"
    columns = [
        "date",
        "home_team",
        "away_team",
        "home_goals",
        "away_goals",
        "penalties_home",
        "penalties_away",
        "winner",
        "display_text",
        "shown_in_dashboard",
        "reason_if_not_shown",
    ]
    for column in columns:
        if column not in knockout.columns:
            knockout[column] = pd.NA
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    knockout[columns].to_csv(REPORTS_DIR / "knockout_results_display_debug.csv", index=False, encoding="utf-8")


def yesterday_or_recent_worldcup_results() -> tuple[pd.DataFrame, str]:
    return results_service.yesterday_or_recent_worldcup_results(today=TODAY)


def predicted_result(row: pd.Series) -> tuple[str, float | None]:
    if all(safe_float(row, column) is not None for column in ["adjusted_H", "adjusted_D", "adjusted_A"]):
        probs = {"H": safe_float(row, "adjusted_H"), "D": safe_float(row, "adjusted_D"), "A": safe_float(row, "adjusted_A")}
    else:
        probs = {"H": safe_float(row, "model_H"), "D": safe_float(row, "model_D"), "A": safe_float(row, "model_A")}
    known = {key: value for key, value in probs.items() if value is not None}
    if not known:
        return "unknown", None
    pick = max(known, key=known.get)
    return pick, known[pick]


def prediction_source_label(row: pd.Series | dict) -> str:
    source = str(safe_value(row, "final_prediction_source", "")).strip().lower()
    if source == "weighted_adjustment":
        return "Adjusted"
    if source:
        return source.replace("_", " ").title()
    if all(safe_float(row, column) is not None for column in ["adjusted_H", "adjusted_D", "adjusted_A"]):
        return "Adjusted"
    return "Base"


def actual_result_label(row: pd.Series) -> str:
    home_goals = safe_float(row, "home_goals")
    away_goals = safe_float(row, "away_goals")
    if home_goals is None or away_goals is None:
        return "unknown"
    if home_goals > away_goals:
        return "H"
    if home_goals < away_goals:
        return "A"
    return "D"


def write_matches_merge_debug(matches: pd.DataFrame) -> None:
    debug = matches.copy()
    result_found = debug["matched_result_key"].notna()
    has_score = debug["home_goals"].notna() & debug["away_goals"].notna()
    debug["result_found"] = result_found
    debug["reason_if_not_matched"] = ""
    debug.loc[~result_found, "reason_if_not_matched"] = "missing_result_for_key"
    debug.loc[result_found & ~has_score, "reason_if_not_matched"] = "matched_result_not_completed_or_missing_score"
    columns = [
        "date",
        "home_team",
        "away_team",
        "result_found",
        "matched_result_key",
        "status",
        "home_goals",
        "away_goals",
        "reason_if_not_matched",
    ]
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    debug[columns].to_csv(REPORTS_DIR / "matches_results_merge_debug.csv", index=False, encoding="utf-8")


def build_matches_search_data(data: pd.DataFrame) -> pd.DataFrame:
    return results_service.build_matches_search_data(data, today=TODAY)


def match_label(row: pd.Series) -> str:
    return f"{safe_value(row, 'date')} | {stage_label(row)} | {safe_value(row, 'home_team')} vs {safe_value(row, 'away_team')}"


def match_key(row: pd.Series | dict) -> str:
    match_id = str(safe_value(row, "match_id", "")).strip()
    if match_id and match_id.lower() != "unknown":
        return match_id
    return f"{safe_value(row, 'date')}|{safe_value(row, 'home_team')}|{safe_value(row, 'away_team')}"


def widget_match_key(row: pd.Series | dict, page: str, idx: int | None = None) -> str:
    match_id = str(safe_value(row, "match_id", "")).strip()
    suffix = match_id if match_id and match_id.lower() != "unknown" else f"idx_{idx}"
    return f"analysis_{page}_{suffix}"


def go_to_page(page: str) -> None:
    st.session_state["page"] = page
    st.rerun()


def go_to_analysis(row: pd.Series | dict) -> None:
    st.session_state["selected_match_key"] = match_key(row)
    go_to_page("Analysis")


def analysis_href(row: pd.Series | dict) -> str:
    return f"?page=Analysis&match_key={quote(match_key(row), safe='')}"


def explicit_search_match_key(row: pd.Series | dict) -> str:
    return f"{safe_value(row, 'date')}|{safe_value(row, 'home_team')}|{safe_value(row, 'away_team')}"


def run_intelligence_search_for_match(row: pd.Series | dict) -> tuple[bool, str]:
    response = refresh_controller.refresh_match_intel(
        str(safe_value(row, "date")),
        str(safe_value(row, "home_team")),
        str(safe_value(row, "away_team")),
    )
    st.cache_data.clear()
    return response.get("status") != "failed", str(response.get("message", ""))[-3000:]


def standardize_recent_source(data: pd.DataFrame, competition_default: str = "") -> pd.DataFrame:
    return results_service.standardize_recent_source(data, competition_default)


def combined_recent_results(history: pd.DataFrame) -> pd.DataFrame:
    return results_service.combined_recent_results(history)


def recent_matches(history: pd.DataFrame, team: str, limit: int = 5) -> pd.DataFrame:
    return results_service.recent_matches(history, team, limit)


def recent_summary(recent: pd.DataFrame) -> dict:
    return results_service.recent_summary(recent)


def summary_cards(data: pd.DataFrame) -> None:
    today = today_data(data)
    action_counts = today.get("recommended_action", pd.Series(dtype=str)).fillna("unknown").value_counts()
    high_draw = int((today.get("draw_risk_level", pd.Series(dtype=str)) == "HIGH").sum())
    high_intel = int((today.get("intel_risk", pd.Series(dtype=str)) == "HIGH").sum())
    high_league = int((today.get("league_reference_level", pd.Series(dtype=str)) == "HIGH").sum())
    cards = [
        ("📅", "Window Matches", len(today)),
        ("🎯", "BET", int(action_counts.get("BET", 0))),
        ("📈", "SMALL BET", int(action_counts.get("SMALL_BET", 0) + action_counts.get("SMALL BET", 0))),
        ("👁", "WATCH", int(action_counts.get("WATCH", 0))),
        ("−", "PASS", int(action_counts.get("PASS", 0))),
        ("⚠", "High Draw Risk", high_draw),
        ("🛡", "High Intel Risk", high_intel),
        ("🏟", "High League Ref", high_league),
    ]
    columns = st.columns(len(cards))
    for column, (icon, label, value) in zip(columns, cards):
        with column:
            st.metric(f"{icon} {label}", value)


def render_match_card(row: pd.Series, key_prefix: str, idx: int | None = None) -> None:
    waiting = is_waiting_for_teams(row)
    action = "WATCH" if waiting else str(safe_value(row, "recommended_action")).replace("_", " ")
    home = display_team_name(row, "home")
    away = display_team_name(row, "away")
    draw_level = "UNKNOWN" if waiting else str(safe_value(row, "draw_risk_level"))
    intel_level = "UNKNOWN" if waiting else str(safe_value(row, "intel_risk"))
    league_level = "UNKNOWN" if waiting else str(safe_value(row, "league_reference_level", "UNKNOWN"))
    with st.container(height=250, border=True):
        header_left, header_right = st.columns([1, 1])
        header_left.markdown(f'<div class="card-top"><span>{stage_label(row)}</span></div>', unsafe_allow_html=True)
        header_right.markdown(
            f'<div class="card-top" style="justify-content:flex-end;"><span>{safe_value(row, "kickoff_time", safe_value(row, "status", "TBD"))}</span></div>',
            unsafe_allow_html=True,
        )

        home_col, vs_col, away_col = st.columns([1, 0.34, 1])
        home_col.markdown(team_block(home), unsafe_allow_html=True)
        vs_col.markdown('<div class="vs" style="padding-top:34px;">VS</div>', unsafe_allow_html=True)
        away_col.markdown(team_block(away), unsafe_allow_html=True)

        risk_col, button_col = st.columns([1.25, 1])
        if waiting:
            risk_html = '<div class="risk-lines"><strong>Waiting for teams</strong><div>No model prediction until both teams are confirmed.</div></div>'
        else:
            knockout_note = "<div>Knockout note: 90-min draw can imply extra-time risk.</div>" if is_knockout_row(row) else ""
            risk_html = (
                f'<div class="action-badge {action_class(action)}">{action}</div>'
                f'<div class="risk-lines">'
                f'<div>Draw Risk: <span class="{risk_class(draw_level)}">{draw_level}</span></div>'
                f'<div>Intel Risk: <span class="{risk_class(intel_level)}">{intel_level}</span></div>'
                f'<div>League Ref: <span class="{risk_class(league_level)}">{league_level}</span></div>'
                f'{knockout_note}'
                f'</div>'
            )
        risk_col.markdown(risk_html, unsafe_allow_html=True)
        with button_col:
            st.write("")
            disabled = waiting
            if st.button("View Analysis", key=widget_match_key(row, key_prefix, idx), disabled=disabled):
                go_to_analysis(row)


def render_match_cards(matches: pd.DataFrame, key_prefix: str = "match") -> None:
    if matches.empty:
        st.info("No matches scheduled for today.")
        return
    sort_columns = [column for column in ["stage", "round", "group", "home_team"] if column in matches.columns]
    for idx, (_, row) in enumerate(matches.sort_values(sort_columns).iterrows()):
        render_match_card(row, key_prefix, idx)


def render_unified_match_card(row: pd.Series, key_prefix: str, idx: int | None = None) -> None:
    if display_mode_for_match(row) == "result":
        st.markdown(result_card_html(row, "View Result Analysis"), unsafe_allow_html=True)
        return

    home = display_team_name(row, "home")
    away = display_team_name(row, "away")
    status_label = espn_status_label(row)
    waiting = is_waiting_for_teams(row)
    with st.container(height=260 if not waiting else 210, border=True):
        header_left, header_right = st.columns([1, 1])
        header_left.markdown(
            f'<div class="card-top"><span>{safe_value(row, "date")} · {stage_label(row)}</span></div>',
            unsafe_allow_html=True,
        )
        header_right.markdown(
            f'<div class="card-top" style="justify-content:flex-end;"><span>{status_label}</span></div>',
            unsafe_allow_html=True,
        )

        home_col, vs_col, away_col = st.columns([1, 0.34, 1])
        home_col.markdown(team_block(home), unsafe_allow_html=True)
        vs_col.markdown('<div class="vs" style="padding-top:34px;">VS</div>', unsafe_allow_html=True)
        away_col.markdown(team_block(away), unsafe_allow_html=True)

        info_col, button_col = st.columns([1.35, 1])
        if waiting:
            info_col.markdown(
                '<div class="risk-lines"><strong>Waiting for teams</strong><div>No model prediction until both teams are confirmed.</div></div>',
                unsafe_allow_html=True,
            )
            disabled = True
        else:
            pick, confidence = predicted_result(row)
            source_label = prediction_source_label(row)
            action = str(safe_value(row, "recommended_action", "WATCH")).replace("_", " ")
            draw_level = str(safe_value(row, "draw_risk_level", "UNKNOWN"))
            intel_level = str(safe_value(row, "intel_risk", "UNKNOWN"))
            league_level = str(safe_value(row, "league_reference_level", "UNKNOWN"))
            info_col.markdown(
                f'<div class="risk-lines">'
                f'<div>Action: <span class="action-badge {action_class(action)}">{action}</span></div>'
                f'<div>Final ({source_label}): <strong>{pick}</strong> ({format_prob(confidence)})</div>'
                f'<div>Draw Risk: <span class="{risk_class(draw_level)}">{draw_level}</span></div>'
                f'<div>Intel Risk: <span class="{risk_class(intel_level)}">{intel_level}</span></div>'
                f'<div>League Ref: <span class="{risk_class(league_level)}">{league_level}</span></div>'
                f'</div>',
                unsafe_allow_html=True,
            )
            disabled = False
        with button_col:
            st.write("")
            if st.button("View Prediction Analysis", key=widget_match_key(row, key_prefix, idx), disabled=disabled):
                go_to_analysis(row)


def render_unified_match_cards(matches: pd.DataFrame, key_prefix: str = "unified_match") -> None:
    if matches.empty:
        st.info("No World Cup matches found for this window.")
        return
    for idx, (_, row) in enumerate(matches.iterrows()):
        render_unified_match_card(row, key_prefix, idx)


def result_to_match_row(row: pd.Series) -> dict:
    return {
        "date": safe_value(row, "date"),
        "home_team": safe_value(row, "home_team"),
        "away_team": safe_value(row, "away_team"),
    }


def completed_result_for_match(row: pd.Series | dict) -> pd.Series | None:
    results, _ = load_worldcup_results()
    if results.empty:
        return None
    mask = (
        (results["date"].astype(str) == str(safe_value(row, "date")))
        & (results["home_team"].astype(str) == str(safe_value(row, "home_team")))
        & (results["away_team"].astype(str) == str(safe_value(row, "away_team")))
    )
    if not mask.any():
        return None
    return results[mask].iloc[0]


def has_penalties(row: pd.Series | dict) -> bool:
    home_penalties = safe_float(row, "penalties_home")
    away_penalties = safe_float(row, "penalties_away")
    return home_penalties is not None and away_penalties is not None


def result_display_note(row: pd.Series | dict) -> str:
    if has_penalties(row):
        winner = str(safe_value(row, "winner", "")).strip()
        home = str(safe_value(row, "home_team", "")).strip()
        away = str(safe_value(row, "away_team", "")).strip()
        home_penalties = int(safe_float(row, "penalties_home") or 0)
        away_penalties = int(safe_float(row, "penalties_away") or 0)
        if winner:
            if winner == away:
                return f"{winner} wins {away_penalties} - {home_penalties} on penalties"
            if winner == home:
                return f"{winner} wins {home_penalties} - {away_penalties} on penalties"
            return f"{winner} wins on penalties ({home_penalties} - {away_penalties})"
        return f"Penalties: {home_penalties} - {away_penalties}"
    aet_value = str(safe_value(row, "aet", "")).strip().lower()
    if aet_value in {"true", "1", "yes"}:
        return "After extra time"
    return ""


def result_display_text(row: pd.Series | dict) -> str:
    home = safe_value(row, "home_team")
    away = safe_value(row, "away_team")
    score = f'{int(safe_value(row, "home_goals"))} - {int(safe_value(row, "away_goals"))}'
    note = result_display_note(row)
    if note:
        return f"{home} {score} {away}; {note}"
    return f"{home} {score} {away}"


def result_body_html(home: str, away: str, score: str, note: str = "") -> str:
    note_html = f'<div class="score-note">{note}</div>' if note else ""
    return (
        '<div class="result-body">'
        f'{team_block(home, "home-team")}'
        f'<div class="score-block"><div class="score-box">{score}</div>{note_html}</div>'
        f'{team_block(away, "away-team")}'
        '</div>'
    )


def result_card_html(row: pd.Series | dict, button_label: str = "View Analysis") -> str:
    home = safe_value(row, "home_team")
    away = safe_value(row, "away_team")
    score = f'{int(safe_value(row, "home_goals"))} - {int(safe_value(row, "away_goals"))}'
    note = result_display_note(row)
    return (
        '<div class="result-card">'
        '<div class="result-header">'
        f'<span>{safe_value(row, "date")}</span>'
        '<span>WORLD CUP 2026</span>'
        '</div>'
        f'{result_body_html(home, away, score, note)}'
        '<div class="result-footer">'
        f'<a class="view-analysis-button" href="{analysis_href(row)}">{button_label}</a>'
        '</div>'
        '</div>'
    )


def render_result_card(row: pd.Series, key_prefix: str) -> None:
    st.markdown(result_card_html(row, "View Analysis"), unsafe_allow_html=True)


def render_result_cards(results: pd.DataFrame, key_prefix: str = "result") -> None:
    if results.empty:
        st.info("No completed World Cup results available.")
        return
    for _, row in results.iterrows():
        render_result_card(row, key_prefix)


def render_search_match_card(row: pd.Series, key_prefix: str, idx: int | None = None) -> None:
    home = display_team_name(row, "home")
    away = display_team_name(row, "away")
    status = str(safe_value(row, "status", "scheduled")).lower()
    is_completed = status == "completed"

    if is_completed:
        st.markdown(result_card_html(row, "View Result Analysis"), unsafe_allow_html=True)
        return

    if is_waiting_for_teams(row):
        with st.container(height=180, border=True):
            st.markdown(f'<div class="card-top"><span>{safe_value(row, "date")} · {stage_label(row)}</span></div>', unsafe_allow_html=True)
            home_col, middle_col, away_col = st.columns([1, 1.05, 1])
            home_col.markdown(team_block(home), unsafe_allow_html=True)
            middle_col.markdown('<div class="vs" style="padding-top:34px;">VS</div><div style="text-align:center;">Waiting for teams</div>', unsafe_allow_html=True)
            away_col.markdown(team_block(away), unsafe_allow_html=True)
        return

    card_height = 180 if is_completed else 245
    with st.container(height=card_height, border=True):
        header_left, header_right = st.columns([1, 1])
        header_left.markdown(
            f'<div class="card-top"><span>{safe_value(row, "date")} · {stage_label(row)}</span></div>',
            unsafe_allow_html=True,
        )
        header_right.markdown(
            f'<div class="card-top" style="justify-content:flex-end;"><span>{status.title()}</span></div>',
            unsafe_allow_html=True,
        )

        home_col, middle_col, away_col = st.columns([1, 1.05, 1])
        home_col.markdown(team_block(home), unsafe_allow_html=True)
        middle_col.markdown('<div class="vs" style="padding-top:34px;">VS</div>', unsafe_allow_html=True)
        away_col.markdown(team_block(away), unsafe_allow_html=True)

        info_col, button_col = st.columns([1.35, 1])
        pick, confidence = predicted_result(row)
        source_label = prediction_source_label(row)
        action = str(safe_value(row, "recommended_action")).replace("_", " ")
        draw_level = str(safe_value(row, "draw_risk_level"))
        intel_level = str(safe_value(row, "intel_risk"))
        league_level = str(safe_value(row, "league_reference_level", "UNKNOWN"))
        info_col.markdown(
            f'<div class="risk-lines">'
            f'<div>Action: <span class="action-badge {action_class(action)}">{action}</span></div>'
            f'<div>Final ({source_label}): <strong>{pick}</strong> ({format_prob(confidence)})</div>'
            f'<div>Draw Risk: <span class="{risk_class(draw_level)}">{draw_level}</span> · '
            f'Intel Risk: <span class="{risk_class(intel_level)}">{intel_level}</span></div>'
            f'<div>League Reference: <span class="{risk_class(league_level)}">{league_level}</span></div>'
            f'</div>',
            unsafe_allow_html=True,
        )
        with button_col:
            st.write("")
            if st.button("View Prediction Analysis", key=widget_match_key(row, key_prefix, idx)):
                go_to_analysis(row)


def render_search_match_cards(matches: pd.DataFrame) -> None:
    if matches.empty:
        st.info("No matches found.")
        return
    for idx, (_, row) in enumerate(matches.iterrows()):
        render_search_match_card(row, "matches_analysis", idx)


def fuzzy_match_query(row: pd.Series, query: str) -> bool:
    if len(query) < 3:
        return False
    fields = [
        "home_team",
        "away_team",
        "stage",
        "round",
        "match_id",
        "home_slot",
        "away_slot",
    ]
    values = []
    for field in fields:
        value = str(safe_value(row, field, "")).strip().lower()
        if value:
            values.append(value)
            values.extend(part for part in value.replace("-", " ").split() if len(part) >= 3)
    return any(SequenceMatcher(None, query, value).ratio() >= 0.82 for value in values)


def section_header(title: str, icon: str, action: str | None = None) -> None:
    action_html = f'<div class="section-action">{action} &rsaquo;</div>' if action else ""
    header_html = (
        f'<div class="section-header">'
        f'<div class="section-title"><span>{icon}</span><span>{title}</span></div>'
        f'{action_html}'
        f'</div>'
    )
    st.markdown(header_html, unsafe_allow_html=True)


def page_today(data: pd.DataFrame) -> None:
    today_text = f"{TODAY_WINDOW_START.strftime('%b %d, %Y')} - {TODAY.strftime('%b %d, %Y')}"
    st.title("📅 Today")
    st.caption(today_text)

    if odds_missing(data):
        st.warning(clean_utf8_text(NO_ODDS_BANNER_TEXT))

    summary_cards(data)
    matches = unified_match_view(data)

    with st.container(height=820, border=True):
        section_header("World Cup Matches", "🏆")
        with st.container(height=760):
            render_unified_match_cards(matches, key_prefix="today_unified")

    action_left, action_right = st.columns(2)
    with action_left:
        if st.button("View All Matches", key="view_all_matches"):
            go_to_page("Matches")
    with action_right:
        if st.button("View All Results", key="view_all_results"):
            go_to_page("Results")

    if st.toggle("Show raw table", value=False, key="today_raw"):
        st.caption(f"Debug: {UNIFIED_MATCH_VIEW_DEBUG_PATH.relative_to(PROJECT_ROOT)}")
        st.dataframe(matches.drop(columns=["date_dt"], errors="ignore"), use_container_width=True)


def page_matches(data: pd.DataFrame) -> None:
    st.title("Matches")
    matches = build_matches_search_data(data)
    if matches.empty:
        st.warning("No match data found.")
        return

    group_options = ["All"] + sorted(matches["group"].dropna().astype(str).unique())
    with st.form("matches_search_form"):
        query_text = st.text_input(
            "Search team / country / group / date",
            placeholder="Brazil, Group D, 2026-06-25",
        )
        filter_col1, filter_col2 = st.columns(2)
        status_filter = filter_col1.selectbox("Status", ["All", "Scheduled", "Completed"])
        group_filter = filter_col2.selectbox("Group", group_options)
        submitted = st.form_submit_button("Search")

    if submitted:
        st.session_state["matches_search_query"] = query_text
        st.session_state["matches_status_filter"] = status_filter
        st.session_state["matches_group_filter"] = group_filter

    query = st.session_state.get("matches_search_query", "")
    status_filter = st.session_state.get("matches_status_filter", "All")
    group_filter = st.session_state.get("matches_group_filter", "All")

    has_search = bool(query.strip()) or status_filter != "All" or group_filter != "All"
    if not has_search:
        st.info("Search for a team, country, group, or date to view matches.")
        if st.toggle("Show raw table", value=False, key="matches_raw"):
            st.dataframe(matches.drop(columns=["date_dt"], errors="ignore"), use_container_width=True)
        return

    view = matches.copy()
    if status_filter != "All":
        view = view[view["status"].str.lower() == status_filter.lower()]
    if group_filter != "All":
        view = view[view["group"].astype(str) == group_filter]
    if query.strip():
        search_text = (
            view["date"].astype(str)
            + " group "
            + view["group"].astype(str)
            + " "
            + view.get("stage", pd.Series("", index=view.index)).astype(str)
            + " "
            + view.get("round", pd.Series("", index=view.index)).astype(str)
            + " "
            + view.get("match_id", pd.Series("", index=view.index)).astype(str)
            + " "
            + view["home_team"].astype(str)
            + " "
            + view["away_team"].astype(str)
            + " "
            + view.get("home_slot", pd.Series("", index=view.index)).astype(str)
            + " "
            + view.get("away_slot", pd.Series("", index=view.index)).astype(str)
            + " "
            + view["status"].astype(str)
        ).str.lower()
        query_text = query.strip().lower()
        fuzzy_mask = view.apply(lambda row: fuzzy_match_query(row, query_text), axis=1)
        view = view[search_text.str.contains(query_text, na=False) | fuzzy_mask]

    st.caption(f"{len(view)} match(es) found")
    render_search_match_cards(view)
    if st.toggle("Show raw table", value=False, key="matches_raw"):
        st.dataframe(view, use_container_width=True)


def page_results() -> None:
    st.title("Results")
    results, results_source = load_worldcup_results()
    st.caption(f"Results source: {results_source}")
    render_result_cards(results, key_prefix="results")
    if st.toggle("Show raw table", value=False, key="results_raw"):
        st.dataframe(results.drop(columns=["date_dt"], errors="ignore"), use_container_width=True)


def page_analysis(data: pd.DataFrame) -> None:
    st.title("Recent 5 Matches")
    if data.empty:
        st.warning("No match data found.")
        return
    labels = [match_label(row) for _, row in data.iterrows()]
    keys = [match_key(row) for _, row in data.iterrows()]
    selected_key = st.session_state.get("selected_match_key")
    selected_index = keys.index(selected_key) if selected_key in keys else 0
    selected = st.selectbox("Select match", labels, index=selected_index)
    row = data.iloc[labels.index(selected)]
    st.session_state["selected_match_key"] = match_key(row)
    history = load_csv(HISTORY_PATH)

    home = display_team_name(row, "home")
    away = display_team_name(row, "away")
    st.caption(f"{home} vs {away}")

    st.subheader("Risk Reference")
    r1, r2, r3 = st.columns(3)
    r1.metric("Draw Risk", str(safe_value(row, "draw_risk_level", "UNKNOWN")))
    r2.metric("Intel Risk", str(safe_value(row, "intel_risk", "UNKNOWN")))
    r3.metric("League Reference", str(safe_value(row, "league_reference_level", "UNKNOWN")))
    league_reasons = str(safe_value(row, "league_reference_reasons", "unknown"))
    st.caption(f"League reference reasons: {league_reasons}")

    home_recent = recent_matches(history, home)
    away_recent = recent_matches(history, away)
    home_summary = recent_summary(home_recent)
    away_summary = recent_summary(away_recent)

    table_columns = ["date", "competition", "home_team", "away_team", "score", "result"]
    c1, c2 = st.columns(2)
    with c1:
        st.subheader(f"{home} summary")
        st.json(home_summary)
        st.dataframe(home_recent.reindex(columns=table_columns), use_container_width=True, hide_index=True)
    with c2:
        st.subheader(f"{away} summary")
        st.json(away_summary)
        st.dataframe(away_recent.reindex(columns=table_columns), use_container_width=True, hide_index=True)


def page_daily_brief() -> None:
    st.title("Daily Brief")
    brief_text, error = daily_brief_service.load_daily_brief_text()
    if brief_text:
        st.markdown(brief_text)
    else:
        st.warning(error or "Daily brief not found.")


def page_prediction_review() -> None:
    st.title("Prediction Review")
    if st.button("Run Prediction Review Now", key="run_prediction_review_now"):
        with st.spinner("Evaluating predictions against completed results..."):
            run_refresh_script("scripts/evaluate_daily_predictions.py")
        st.rerun()

    review_payload = review_service.load_prediction_review(today=TODAY)
    review = review_payload["review"]
    recent = review_payload["recent"]
    summary = review_payload["summary"]
    accuracy_analysis = review_payload["accuracy_analysis"]
    risk_signal_accuracy = review_payload["risk_signal_accuracy"]

    if review.empty:
        st.info("No completed prediction review is available yet.")
        return

    if not summary.empty:
        st.subheader("Summary")
        st.dataframe(summary, use_container_width=True, hide_index=True)

    st.subheader("Today / Yesterday Completed Matches")
    display_columns = [
        "date",
        "home_team",
        "away_team",
        "home_score",
        "away_score",
        "predicted_result",
        "actual_result",
        "correct",
        "confidence",
        "action",
        "odds_status",
        "intel_risk",
        "draw_risk_level",
        "error_type",
        "model_notes",
    ]
    st.dataframe(recent.reindex(columns=display_columns), use_container_width=True, hide_index=True)

    if not accuracy_analysis.empty:
        st.subheader("Prediction Accuracy Patterns")
        st.dataframe(accuracy_analysis, use_container_width=True, hide_index=True)

    if not risk_signal_accuracy.empty:
        st.subheader("Risk Signal Accuracy")
        st.dataframe(risk_signal_accuracy, use_container_width=True, hide_index=True)

    if st.toggle("Show raw review table", value=False, key="prediction_review_raw"):
        st.dataframe(review.drop(columns=["date_dt"], errors="ignore"), use_container_width=True)


def page_high_risk(data: pd.DataFrame) -> None:
    st.title("High Risk")
    if data.empty:
        st.warning("No match data found.")
        return
    high = data[
        (data.get("draw_risk_level", pd.Series(index=data.index, dtype=str)) == "HIGH")
        | (data.get("league_reference_level", pd.Series(index=data.index, dtype=str)) == "HIGH")
        | (data.get("intel_risk", pd.Series(index=data.index, dtype=str)) == "HIGH")
        | (data.get("upset_risk", pd.Series(index=data.index, dtype=str)) == "HIGH")
    ]
    render_match_cards(high)
    if st.toggle("Show raw table", value=False, key="risk_raw"):
        st.dataframe(high, use_container_width=True)
    league_columns = ["date", "home_team", "away_team", "league_reference_level", "league_risk_score", "league_reference_reasons"]
    league_view = data.reindex(columns=league_columns)
    league_view = league_view[league_view["league_reference_level"].fillna("UNKNOWN") != "UNKNOWN"]
    if not league_view.empty and st.toggle("Show league reference risk", value=True):
        st.subheader("League Reference Risk")
        st.dataframe(league_view, use_container_width=True, hide_index=True)
    draw_high = load_csv(HIGH_DRAW_RISK_PATH)
    if not draw_high.empty and st.toggle("Show high draw risk report", value=False):
        st.dataframe(draw_high, use_container_width=True)


def page_settings() -> None:
    st.title("Settings")
    st.subheader("Manual Odds")
    st.code(str(MANUAL_ODDS_PATH.relative_to(PROJECT_ROOT)))
    if MANUAL_ODDS_PATH.exists():
        st.dataframe(load_csv(MANUAL_ODDS_PATH), use_container_width=True)
    else:
        st.info("Manual odds file is missing.")

    st.subheader("Manual Intel")
    st.code(str(MANUAL_INTEL_PATH.relative_to(PROJECT_ROOT)))
    if MANUAL_INTEL_PATH.exists():
        st.dataframe(load_csv(MANUAL_INTEL_PATH), use_container_width=True)
    else:
        st.info("Manual intel file is missing. Use the template below.")
    if MANUAL_INTEL_TEMPLATE_PATH.exists():
        st.dataframe(load_csv(MANUAL_INTEL_TEMPLATE_PATH), use_container_width=True)

    st.subheader("Manual Results")
    st.code(str(MANUAL_RESULTS_PATH.relative_to(PROJECT_ROOT)))
    if MANUAL_RESULTS_PATH.exists():
        st.dataframe(load_csv(MANUAL_RESULTS_PATH), use_container_width=True)
    else:
        st.info("Manual results file is missing. Use the template below.")
    if MANUAL_RESULTS_TEMPLATE_PATH.exists():
        st.dataframe(load_csv(MANUAL_RESULTS_TEMPLATE_PATH), use_container_width=True)


def render_sidebar(data: pd.DataFrame) -> str:
    pages = ["Today", "Matches", "Results", "Analysis", "Daily Brief", "Prediction Review", "High Risk", "Settings", "About"]
    if "page" not in st.session_state:
        st.session_state["page"] = "Today"
    if st.session_state["page"] not in pages:
        st.session_state["page"] = "Today"

    with st.sidebar:
        st.markdown(
            """
            <div class="brand-row">
              <span class="brand-icon">🏆</span>
              <span class="brand-title">WORLD CUP 2026</span>
            </div>
            <div class="brand-subtitle">Betting Assistant</div>
            """,
            unsafe_allow_html=True,
        )
        page = st.radio(
            "Navigation",
            pages,
            index=pages.index(st.session_state["page"]),
        )
        if page != st.session_state["page"]:
            st.session_state["page"] = page
        st.divider()
        scheduler_status = read_scheduler_status()
        last_daily, next_daily, daily_status = job_status_text(scheduler_status, "daily")
        last_hourly, next_hourly, hourly_status = job_status_text(scheduler_status, "hourly")
        st.caption(f"Last daily refresh: {last_daily} ({daily_status})")
        st.caption(f"Last hourly intel refresh: {last_hourly} ({hourly_status})")
        st.caption(f"Next daily refresh: {next_daily}")
        st.caption(f"Next hourly refresh: {next_hourly}")
        with st.form("admin_refresh_form"):
            st.caption("Admin refresh")
            admin_username = st.text_input("Username", key="refresh_admin_username")
            admin_password = st.text_input("Password", type="password", key="refresh_admin_password")
            run_daily = st.form_submit_button("Run Daily Refresh Now")
            run_hourly = st.form_submit_button("Run Hourly Intel Refresh Now")
        if run_daily:
            with st.spinner("Running daily refresh..."):
                run_refresh_script("scripts/run_daily_refresh.py", admin_username, admin_password)
            st.rerun()
        if run_hourly:
            with st.spinner("Running hourly intel refresh..."):
                run_refresh_script("scripts/run_hourly_intel_refresh.py", admin_username, admin_password)
            st.rerun()
        if st.session_state.get("refresh_warning"):
            st.warning(st.session_state["refresh_warning"])
        st.divider()
        st.markdown("**Mode**")
        if NO_PAID_API_MODE:
            st.info("No paid API mode\n\nPaid odds APIs are disabled. Odds can only come from manual import or future OCR/manual workflows.")
        if odds_missing(data):
            st.info("🛡 NO-ODDS MODE (Safe Mode)\n\nNo odds available. Recommendations are based on model and risk analysis only.")
        else:
            st.success("Odds mode enabled.")
    return st.session_state["page"]


def main() -> None:
    inject_style()
    apply_query_params()
    data = combined_match_data()
    page = render_sidebar(data)

    if page == "Today":
        page_today(data)
    elif page == "Matches":
        page_matches(data)
    elif page == "Results":
        page_results()
    elif page == "Analysis":
        page_analysis(data)
    elif page == "Daily Brief":
        page_daily_brief()
    elif page == "Prediction Review":
        page_prediction_review()
    elif page == "High Risk":
        page_high_risk(data)
    elif page == "Settings":
        page_settings()
    else:
        st.title("About")
        st.write("World Cup 2026 betting assistant dashboard for model, risk, and intelligence review.")


if __name__ == "__main__":
    main()

