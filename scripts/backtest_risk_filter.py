import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.paths import META_FEATURES_PATH, XGB_META_MODEL_PATH
from src.train_xgb_meta import BASE_FEATURE_COLUMNS, ID_TO_CLASS, add_meta_features, chronological_split

try:
    from xgboost import XGBClassifier
except ImportError as exc:
    raise ImportError("Missing dependency: install xgboost with `pip install xgboost`.") from exc


REPORTS_DIR = PROJECT_ROOT / "reports"
MARKET_COLUMNS = ["market_H", "market_D", "market_A"]
RULE_COLUMNS = [
    "risk_abs_poisson_diff_075_100",
    "risk_market_confidence_060_065",
    "risk_market_margin_030_040",
    "risk_xgb_confidence_lt_060",
]


def second_highest(values: pd.Series) -> float:
    return float(values.sort_values(ascending=False).iloc[1])


def load_model(model_path: Path) -> XGBClassifier:
    if not model_path.exists():
        raise FileNotFoundError(f"XGBoost model not found: {model_path}")
    model = XGBClassifier()
    model.load_model(model_path)
    return model


def load_backtest_frame(csv_path: Path, model_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"meta features CSV not found: {csv_path}")

    data = add_meta_features(pd.read_csv(csv_path, encoding="utf-8"))
    _, test_data = chronological_split(data)
    model = load_model(model_path)
    probabilities = model.predict_proba(test_data[BASE_FEATURE_COLUMNS])
    predictions = model.predict(test_data[BASE_FEATURE_COLUMNS])

    frame = test_data.copy()
    frame["market_pred"] = frame[MARKET_COLUMNS].idxmax(axis=1).str.replace("market_", "", regex=False)
    frame["market_confidence"] = frame[MARKET_COLUMNS].max(axis=1)
    frame["market_margin"] = frame["market_confidence"] - frame[MARKET_COLUMNS].apply(second_highest, axis=1)
    frame["xgb_pred"] = [ID_TO_CLASS[int(prediction)] for prediction in predictions]
    frame["xgb_confidence"] = probabilities.max(axis=1)
    frame["xgb_correct"] = frame["xgb_pred"] == frame["actual_result"]
    frame["is_upset"] = frame["actual_result"] != frame["market_pred"]
    frame["abs_poisson_diff"] = (frame["poisson_away_xg"] - frame["poisson_home_xg"]).abs()
    frame["strong_favorite"] = (frame["market_confidence"] >= 0.60) | (frame["xgb_confidence"] >= 0.70)

    frame["risk_abs_poisson_diff_075_100"] = frame["abs_poisson_diff"].between(0.75, 1.00, inclusive="both")
    frame["risk_market_confidence_060_065"] = frame["market_confidence"].between(0.60, 0.65, inclusive="both")
    frame["risk_market_margin_030_040"] = frame["market_margin"].between(0.30, 0.40, inclusive="both")
    frame["risk_xgb_confidence_lt_060"] = frame["xgb_confidence"] < 0.60
    frame["risk_rule_count"] = frame[RULE_COLUMNS].sum(axis=1)
    frame["risk_flag"] = frame[RULE_COLUMNS].any(axis=1)

    frame["final_action"] = "NO_SIGNAL"
    frame.loc[frame["strong_favorite"] & frame["risk_flag"], "final_action"] = "PASS"
    frame.loc[frame["strong_favorite"] & ~frame["risk_flag"], "final_action"] = "ALLOW"
    return frame


def safe_rate(series: pd.Series) -> float:
    if len(series) == 0:
        return np.nan
    return float(series.mean())


def backtest_summary(frame: pd.DataFrame) -> pd.DataFrame:
    strong = frame.loc[frame["strong_favorite"]]
    passed = frame.loc[frame["final_action"] == "PASS"]
    allowed = frame.loc[frame["final_action"] == "ALLOW"]

    original_accuracy = safe_rate(strong["xgb_correct"])
    allowed_accuracy = safe_rate(allowed["xgb_correct"])

    return pd.DataFrame(
        [
            {
                "strong_favorite_count": len(strong),
                "original_strong_favorite_accuracy": original_accuracy,
                "pass_count": len(passed),
                "pass_upset_rate": safe_rate(passed["is_upset"]),
                "allow_count": len(allowed),
                "allow_accuracy": allowed_accuracy,
                "accuracy_lift_after_filter": allowed_accuracy - original_accuracy,
                "coverage_kept_rate": len(allowed) / len(strong) if len(strong) else np.nan,
                "coverage_pass_rate": len(passed) / len(strong) if len(strong) else np.nan,
            }
        ]
    )


def rule_breakdown(frame: pd.DataFrame) -> pd.DataFrame:
    strong = frame.loc[frame["strong_favorite"]]
    rows = []

    for rule in RULE_COLUMNS:
        hit = strong.loc[strong[rule]]
        miss = strong.loc[~strong[rule]]
        rows.append(
            {
                "rule": rule,
                "hit_count": len(hit),
                "hit_upset_rate": safe_rate(hit["is_upset"]),
                "hit_xgb_accuracy": safe_rate(hit["xgb_correct"]),
                "miss_count": len(miss),
                "miss_upset_rate": safe_rate(miss["is_upset"]),
                "miss_xgb_accuracy": safe_rate(miss["xgb_correct"]),
            }
        )

    for count, group in strong.groupby("risk_rule_count"):
        rows.append(
            {
                "rule": f"risk_rule_count == {int(count)}",
                "hit_count": len(group),
                "hit_upset_rate": safe_rate(group["is_upset"]),
                "hit_xgb_accuracy": safe_rate(group["xgb_correct"]),
                "miss_count": np.nan,
                "miss_upset_rate": np.nan,
                "miss_xgb_accuracy": np.nan,
            }
        )

    at_least_two = strong.loc[strong["risk_rule_count"] >= 2]
    rows.append(
        {
            "rule": "risk_rule_count >= 2",
            "hit_count": len(at_least_two),
            "hit_upset_rate": safe_rate(at_least_two["is_upset"]),
            "hit_xgb_accuracy": safe_rate(at_least_two["xgb_correct"]),
            "miss_count": len(strong) - len(at_least_two),
            "miss_upset_rate": safe_rate(strong.loc[strong["risk_rule_count"] < 2, "is_upset"]),
            "miss_xgb_accuracy": safe_rate(strong.loc[strong["risk_rule_count"] < 2, "xgb_correct"]),
        }
    )

    return pd.DataFrame(rows)


def run_backtest(csv_path: Path, model_path: Path, reports_dir: Path = REPORTS_DIR) -> dict[str, pd.DataFrame]:
    frame = load_backtest_frame(csv_path, model_path)
    reports = {
        "summary": backtest_summary(frame),
        "rules": rule_breakdown(frame),
    }

    reports_dir.mkdir(parents=True, exist_ok=True)
    reports["summary"].to_csv(reports_dir / "risk_filter_backtest.csv", index=False, encoding="utf-8")
    reports["rules"].to_csv(reports_dir / "risk_rule_breakdown.csv", index=False, encoding="utf-8")
    return reports


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, default=META_FEATURES_PATH)
    parser.add_argument("--model", type=Path, default=XGB_META_MODEL_PATH)
    args = parser.parse_args()

    reports = run_backtest(args.csv, args.model)
    print("Risk filter backtest complete.")
    print("Summary:")
    print(reports["summary"].to_string(index=False))
    print("\nRule breakdown:")
    print(reports["rules"].to_string(index=False))


if __name__ == "__main__":
    main()
