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
REQUIRED_COLUMNS = [
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
    "actual_result",
]
HIGH_CONFIDENCE_UPSET_COLUMNS = [
    "date",
    "league",
    "home_team",
    "away_team",
    "actual_result",
    "market_pred",
    "market_confidence",
    "xgb_pred",
    "xgb_confidence",
    "market_margin",
    "poisson_home_xg",
    "poisson_away_xg",
    "poisson_total_xg",
    "poisson_draw_signal",
]


def require_columns(data: pd.DataFrame) -> None:
    missing = [column for column in REQUIRED_COLUMNS if column not in data.columns]
    if missing:
        raise ValueError(f"meta features CSV is missing required columns: {missing}")


def load_model(model_path: Path) -> XGBClassifier:
    if not model_path.exists():
        raise FileNotFoundError(f"XGBoost model not found: {model_path}")
    model = XGBClassifier()
    model.load_model(model_path)
    return model


def second_highest(values: pd.Series) -> float:
    return float(values.sort_values(ascending=False).iloc[1])


def load_analysis_frame(csv_path: Path, model_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"meta features CSV not found: {csv_path}")

    raw = pd.read_csv(csv_path, encoding="utf-8")
    require_columns(raw)
    data = add_meta_features(raw)

    # Analyze the same chronological holdout used by XGBoost evaluation.
    _, test_data = chronological_split(data)
    model = load_model(model_path)
    probabilities = model.predict_proba(test_data[BASE_FEATURE_COLUMNS])
    predictions = model.predict(test_data[BASE_FEATURE_COLUMNS])

    analyzed = test_data.copy()
    analyzed["market_pred"] = analyzed[MARKET_COLUMNS].idxmax(axis=1).str.replace("market_", "", regex=False)
    analyzed["market_confidence"] = analyzed[MARKET_COLUMNS].max(axis=1)
    analyzed["market_margin"] = analyzed["market_confidence"] - analyzed[MARKET_COLUMNS].apply(second_highest, axis=1)
    analyzed["xgb_pred"] = [ID_TO_CLASS[int(prediction)] for prediction in predictions]
    analyzed["xgb_H"] = probabilities[:, 0]
    analyzed["xgb_D"] = probabilities[:, 1]
    analyzed["xgb_A"] = probabilities[:, 2]
    analyzed["xgb_confidence"] = probabilities.max(axis=1)
    analyzed["market_correct"] = analyzed["market_pred"] == analyzed["actual_result"]
    analyzed["xgb_correct"] = analyzed["xgb_pred"] == analyzed["actual_result"]
    analyzed["is_upset"] = analyzed["actual_result"] != analyzed["market_pred"]
    analyzed["strong_favorite"] = (analyzed["market_confidence"] >= 0.60) | (analyzed["xgb_confidence"] >= 0.70)
    analyzed["abs_poisson_diff"] = (analyzed["poisson_away_xg"] - analyzed["poisson_home_xg"]).abs()
    analyzed["poisson_total_xg"] = analyzed["poisson_home_xg"] + analyzed["poisson_away_xg"]
    analyzed["poisson_draw_signal"] = analyzed["abs_poisson_diff"]
    return analyzed


def summary(strong: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "segment": "strong_favorite",
                "count": len(strong),
                "market_accuracy": strong["market_correct"].mean(),
                "xgb_accuracy": strong["xgb_correct"].mean(),
                "upset_rate": strong["is_upset"].mean(),
            }
        ]
    )


def bucket_specs() -> dict[str, tuple[list[float], list[str]]]:
    return {
        "market_confidence": (
            [0.0, 0.60, 0.65, 0.70, 0.75, 0.80, np.inf],
            ["<0.60", "0.60-0.65", "0.65-0.70", "0.70-0.75", "0.75-0.80", "0.80+"],
        ),
        "xgb_confidence": (
            [0.0, 0.60, 0.70, 0.80, 0.90, np.inf],
            ["<0.60", "0.60-0.70", "0.70-0.80", "0.80-0.90", "0.90+"],
        ),
        "market_margin": (
            [0.0, 0.10, 0.20, 0.30, 0.40, np.inf],
            ["0.00-0.10", "0.10-0.20", "0.20-0.30", "0.30-0.40", "0.40+"],
        ),
        "abs_poisson_diff": (
            [0.0, 0.25, 0.50, 0.75, 1.00, 1.50, np.inf],
            ["0.00-0.25", "0.25-0.50", "0.50-0.75", "0.75-1.00", "1.00-1.50", "1.50+"],
        ),
        "poisson_total_xg": (
            [0.0, 2.0, 2.5, 3.0, 3.5, np.inf],
            ["<2.0", "2.0-2.5", "2.5-3.0", "3.0-3.5", "3.5+"],
        ),
        "poisson_draw_signal": (
            [0.0, 0.25, 0.50, 0.75, 1.00, 1.50, np.inf],
            ["0.00-0.25", "0.25-0.50", "0.50-0.75", "0.75-1.00", "1.00-1.50", "1.50+"],
        ),
    }


def bucket_analysis(strong: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for feature_name, (bins, labels) in bucket_specs().items():
        bucket_column = f"{feature_name}_bucket"
        bucketed = strong.copy()
        bucketed[bucket_column] = pd.cut(
            bucketed[feature_name],
            bins=bins,
            labels=labels,
            include_lowest=True,
            right=False,
        )

        for bucket, group in bucketed.groupby(bucket_column, observed=False):
            if group.empty:
                continue
            rows.append(
                {
                    "feature": feature_name,
                    "bucket": str(bucket),
                    "count": len(group),
                    "market_accuracy": group["market_correct"].mean(),
                    "xgb_accuracy": group["xgb_correct"].mean(),
                    "upset_rate": group["is_upset"].mean(),
                    "draw_rate": (group["actual_result"] == "D").mean(),
                    "avg_market_confidence": group["market_confidence"].mean(),
                    "avg_xgb_confidence": group["xgb_confidence"].mean(),
                    "avg_poisson_total_xg": group["poisson_total_xg"].mean(),
                    "avg_poisson_draw_signal": group["poisson_draw_signal"].mean(),
                }
            )

    return pd.DataFrame(rows).sort_values(["upset_rate", "count"], ascending=[False, False])


def high_confidence_upsets(strong: pd.DataFrame) -> pd.DataFrame:
    upsets = strong.loc[strong["is_upset"]].copy()
    upsets = upsets.sort_values(["xgb_confidence", "market_confidence"], ascending=False)
    return upsets[HIGH_CONFIDENCE_UPSET_COLUMNS]


def run_analysis(csv_path: Path, model_path: Path, reports_dir: Path = REPORTS_DIR) -> dict[str, pd.DataFrame]:
    analyzed = load_analysis_frame(csv_path, model_path)
    strong = analyzed.loc[analyzed["strong_favorite"]].copy()

    reports = {
        "summary": summary(strong),
        "buckets": bucket_analysis(strong),
        "high_confidence_upsets": high_confidence_upsets(strong),
    }

    reports_dir.mkdir(parents=True, exist_ok=True)
    reports["summary"].to_csv(reports_dir / "upset_risk_summary.csv", index=False, encoding="utf-8")
    reports["buckets"].to_csv(reports_dir / "upset_risk_buckets.csv", index=False, encoding="utf-8")
    reports["high_confidence_upsets"].to_csv(reports_dir / "high_confidence_upsets.csv", index=False, encoding="utf-8")
    return reports


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, default=META_FEATURES_PATH)
    parser.add_argument("--model", type=Path, default=XGB_META_MODEL_PATH)
    args = parser.parse_args()

    reports = run_analysis(args.csv, args.model)
    print("Upset risk analysis complete.")
    print("Summary:")
    print(reports["summary"].to_string(index=False))
    print("\nHighest upset-rate buckets:")
    print(reports["buckets"].head(12).to_string(index=False))
    print("\nHigh-confidence upset samples:")
    print(reports["high_confidence_upsets"].head(10).to_string(index=False))


if __name__ == "__main__":
    main()
