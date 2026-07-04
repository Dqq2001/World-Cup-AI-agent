import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

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
NOMARKET_COLUMNS = ["nomarket_H", "nomarket_D", "nomarket_A"]
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
WORST_ERROR_COLUMNS = [
    "date",
    "league",
    "home_team",
    "away_team",
    "actual_result",
    "market_pred",
    "market_confidence",
    "nomarket_pred",
    "nomarket_confidence",
    "xgb_pred",
    "xgb_confidence",
    "poisson_home_xg",
    "poisson_away_xg",
    "poisson_diff",
]


def require_columns(data: pd.DataFrame) -> None:
    missing = [column for column in REQUIRED_COLUMNS if column not in data.columns]
    if missing:
        raise ValueError(f"meta features CSV is missing required columns: {missing}")


def load_analysis_data(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"meta features CSV not found: {csv_path}")
    data = pd.read_csv(csv_path, encoding="utf-8")
    require_columns(data)
    return add_meta_features(data)


def load_model(model_path: Path) -> XGBClassifier:
    if not model_path.exists():
        raise FileNotFoundError(f"XGBoost model not found: {model_path}")
    model = XGBClassifier()
    model.load_model(model_path)
    return model


def second_highest(values: pd.Series) -> float:
    return float(values.sort_values(ascending=False).iloc[1])


def add_predictions(data: pd.DataFrame, model: XGBClassifier) -> pd.DataFrame:
    data = data.copy()
    probabilities = model.predict_proba(data[BASE_FEATURE_COLUMNS])
    predictions = model.predict(data[BASE_FEATURE_COLUMNS])

    data["xgb_pred"] = [ID_TO_CLASS[int(prediction)] for prediction in predictions]
    data["xgb_H"] = probabilities[:, 0]
    data["xgb_D"] = probabilities[:, 1]
    data["xgb_A"] = probabilities[:, 2]
    data["xgb_confidence"] = probabilities.max(axis=1)

    data["market_pred"] = data[MARKET_COLUMNS].idxmax(axis=1).str.replace("market_", "", regex=False)
    data["nomarket_pred"] = data[NOMARKET_COLUMNS].idxmax(axis=1).str.replace("nomarket_", "", regex=False)
    data["xgb_correct"] = data["xgb_pred"] == data["actual_result"]
    data["market_correct"] = data["market_pred"] == data["actual_result"]
    data["nomarket_correct"] = data["nomarket_pred"] == data["actual_result"]
    data["models_agree"] = data["market_pred"] == data["nomarket_pred"]
    data["market_nomarket_disagree"] = data["market_pred"] != data["nomarket_pred"]
    data["market_confidence"] = data[MARKET_COLUMNS].max(axis=1)
    data["nomarket_confidence"] = data[NOMARKET_COLUMNS].max(axis=1)
    data["poisson_diff"] = data["poisson_away_xg"] - data["poisson_home_xg"]
    data["abs_poisson_diff"] = data["poisson_diff"].abs()
    data["market_margin"] = data["market_confidence"] - data[MARKET_COLUMNS].apply(second_highest, axis=1)
    data["is_fifty_fifty"] = data["market_margin"] < 0.08
    data["is_strong_market_favorite"] = data["market_confidence"] >= 0.55
    data["is_strong_home_market_favorite"] = (data["market_pred"] == "H") & data["is_strong_market_favorite"]
    data["is_upset"] = data["actual_result"] != data["market_pred"]
    return data


def add_buckets(data: pd.DataFrame) -> pd.DataFrame:
    data = data.copy()
    confidence_bins = [0.33, 0.40, 0.45, 0.50, 0.55, 0.60, np.inf]
    confidence_labels = ["0.33-0.40", "0.40-0.45", "0.45-0.50", "0.50-0.55", "0.55-0.60", "0.60+"]
    data["market_confidence_bucket"] = pd.cut(
        data["market_confidence"],
        bins=confidence_bins,
        labels=confidence_labels,
        include_lowest=True,
        right=False,
    )

    margin_bins = [0.0, 0.03, 0.06, 0.08, 0.12, 0.20, np.inf]
    margin_labels = ["0.00-0.03", "0.03-0.06", "0.06-0.08", "0.08-0.12", "0.12-0.20", "0.20+"]
    data["market_margin_bucket"] = pd.cut(
        data["market_margin"],
        bins=margin_bins,
        labels=margin_labels,
        include_lowest=True,
        right=False,
    )
    return data


def _accuracy(series: pd.Series) -> float:
    if len(series) == 0:
        return np.nan
    return float(series.mean())


def overall_summary(data: pd.DataFrame) -> pd.DataFrame:
    rows = [
        {"segment": "overall", "model": "market", "count": len(data), "accuracy": _accuracy(data["market_correct"])},
        {
            "segment": "overall",
            "model": "no_market",
            "count": len(data),
            "accuracy": _accuracy(data["nomarket_correct"]),
        },
        {"segment": "overall", "model": "xgboost", "count": len(data), "accuracy": _accuracy(data["xgb_correct"])},
    ]

    for segment_name, mask in {
        "strong_home_market_favorite": data["is_strong_home_market_favorite"],
        "fifty_fifty": data["is_fifty_fifty"],
        "upset_cases": data["is_upset"],
    }.items():
        segment = data.loc[mask]
        rows.append(
            {
                "segment": segment_name,
                "model": "xgboost",
                "count": len(segment),
                "accuracy": _accuracy(segment["xgb_correct"]),
                "market_confidence_avg": segment["market_confidence"].mean(),
                "poisson_diff_avg": segment["poisson_diff"].mean(),
                "upset_rate": segment["is_upset"].mean() if len(segment) else np.nan,
            }
        )

    return pd.DataFrame(rows)


def grouped_accuracy(data: pd.DataFrame, bucket_column: str) -> pd.DataFrame:
    rows = []
    for bucket, group in data.groupby(bucket_column, observed=False):
        if group.empty:
            continue
        rows.append(
            {
                bucket_column: str(bucket),
                "count": len(group),
                "market_accuracy": _accuracy(group["market_correct"]),
                "nomarket_accuracy": _accuracy(group["nomarket_correct"]),
                "xgb_accuracy": _accuracy(group["xgb_correct"]),
                "draw_rate": float((group["actual_result"] == "D").mean()),
                "upset_rate": float(group["is_upset"].mean()),
            }
        )
    return pd.DataFrame(rows)


def disagreement_summary(data: pd.DataFrame) -> pd.DataFrame:
    group = data.loc[data["market_nomarket_disagree"]]
    return pd.DataFrame(
        [
            {
                "count": len(group),
                "market_accuracy": _accuracy(group["market_correct"]),
                "nomarket_accuracy": _accuracy(group["nomarket_correct"]),
                "xgb_accuracy": _accuracy(group["xgb_correct"]),
                "draw_rate": float((group["actual_result"] == "D").mean()) if len(group) else np.nan,
                "upset_rate": float(group["is_upset"].mean()) if len(group) else np.nan,
            }
        ]
    )


def worst_errors(data: pd.DataFrame) -> pd.DataFrame:
    errors = data.loc[~data["xgb_correct"]].copy()
    errors = errors.sort_values(["xgb_confidence", "market_confidence"], ascending=False)
    return errors[WORST_ERROR_COLUMNS].head(50)


def run_error_analysis(csv_path: Path, model_path: Path, reports_dir: Path = REPORTS_DIR) -> dict[str, pd.DataFrame]:
    data = load_analysis_data(csv_path)
    _, test_data = chronological_split(data)
    model = load_model(model_path)
    analyzed = add_buckets(add_predictions(test_data, model))

    reports = {
        "summary": overall_summary(analyzed),
        "by_confidence": grouped_accuracy(analyzed, "market_confidence_bucket"),
        "by_margin": grouped_accuracy(analyzed, "market_margin_bucket"),
        "disagreement": disagreement_summary(analyzed),
        "worst_errors": worst_errors(analyzed),
    }

    reports_dir.mkdir(parents=True, exist_ok=True)
    reports["summary"].to_csv(reports_dir / "xgb_error_analysis_summary.csv", index=False, encoding="utf-8")
    reports["by_confidence"].to_csv(reports_dir / "xgb_error_analysis_by_confidence.csv", index=False, encoding="utf-8")
    reports["by_margin"].to_csv(reports_dir / "xgb_error_analysis_by_margin.csv", index=False, encoding="utf-8")
    reports["disagreement"].to_csv(reports_dir / "xgb_error_analysis_disagreement.csv", index=False, encoding="utf-8")
    reports["worst_errors"].to_csv(reports_dir / "xgb_worst_errors.csv", index=False, encoding="utf-8")
    return reports


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, default=META_FEATURES_PATH)
    parser.add_argument("--model", type=Path, default=XGB_META_MODEL_PATH)
    args = parser.parse_args()

    reports = run_error_analysis(args.csv, args.model)
    print("XGBoost error analysis complete.")
    print("Overall summary:")
    print(reports["summary"].to_string(index=False))
    print("\nBy market confidence:")
    print(reports["by_confidence"].to_string(index=False))
    print("\nDisagreement:")
    print(reports["disagreement"].to_string(index=False))


if __name__ == "__main__":
    main()
