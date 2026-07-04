from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"

MATCHES_CLEAN_PATH = PROCESSED_DATA_DIR / "matches_clean.csv"
FEATURES_PATH = PROCESSED_DATA_DIR / "matches_features.csv"
MODEL_PATH = MODELS_DIR / "wdl_model.pkl"
MARKET_WDL_MODEL_PATH = MODELS_DIR / "market_wdl_model.pkl"
NO_MARKET_WDL_MODEL_PATH = MODELS_DIR / "no_market_wdl_model.pkl"
POISSON_MODEL_PATH = MODELS_DIR / "poisson_model.pkl"
META_FEATURES_PATH = PROCESSED_DATA_DIR / "meta_features.csv"
XGB_META_MODEL_PATH = MODELS_DIR / "xgb_meta_model.json"
WORLDCUP_FEATURES_PATH = PROCESSED_DATA_DIR / "worldcup_features.csv"
INTERNATIONAL_TRAINING_PATH = PROCESSED_DATA_DIR / "international_training_data.csv"
INTERNATIONAL_WDL_MODEL_PATH = MODELS_DIR / "international_wdl_model.json"
INTERNATIONAL_POISSON_HOME_PATH = MODELS_DIR / "international_poisson_home.json"
INTERNATIONAL_POISSON_AWAY_PATH = MODELS_DIR / "international_poisson_away.json"
