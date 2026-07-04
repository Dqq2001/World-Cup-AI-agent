from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MANUAL_DIR = DATA_DIR / "manual"
CACHE_DIR = DATA_DIR / "cache"
REPORTS_DIR = PROJECT_ROOT / "reports"
WEBSITE_DATA_DIR = PROJECT_ROOT / "website" / "public" / "data"

DASHBOARD_REFRESH_PATH = CACHE_DIR / "dashboard_last_refresh.json"
SCHEDULER_STATUS_PATH = CACHE_DIR / "scheduler_status.json"

