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
ODDS_COLUMNS = ["home_odds", "draw_odds", "away_odds"]
RESULT_TO_ODDS_COLUMN = {"H": "home_odds", "D": "draw_odds", "A": "away_odds"}
BOOTSTRAP_ITERATIONS = 1000


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
    frame["xgb_pred"] = [ID_TO_CLASS[int(prediction)] for prediction in predictions]
    frame["xgb_H"] = probabilities[:, 0]
    frame["xgb_D"] = probabilities[:, 1]
    frame["xgb_A"] = probabilities[:, 2]
    frame["xgb_confidence"] = probabilities.max(axis=1)

    frame["market_pred"] = frame[MARKET_COLUMNS].idxmax(axis=1).str.replace("market_", "", regex=False)
    frame["market_confidence"] = frame[MARKET_COLUMNS].max(axis=1)
    frame["market_margin"] = frame["market_confidence"] - frame[MARKET_COLUMNS].apply(second_highest, axis=1)
    frame["abs_poisson_diff"] = (frame["poisson_away_xg"] - frame["poisson_home_xg"]).abs()
    frame["strong_favorite"] = (frame["market_confidence"] >= 0.60) | (frame["xgb_confidence"] >= 0.70)

    frame["risk_abs_poisson_diff_075_100"] = frame["abs_poisson_diff"].between(0.75, 1.00, inclusive="both")
    frame["risk_market_confidence_060_065"] = frame["market_confidence"].between(0.60, 0.65, inclusive="both")
    frame["risk_market_margin_030_040"] = frame["market_margin"].between(0.30, 0.40, inclusive="both")
    frame["risk_xgb_confidence_lt_060"] = frame["xgb_confidence"] < 0.60
    risk_columns = [
        "risk_abs_poisson_diff_075_100",
        "risk_market_confidence_060_065",
        "risk_market_margin_030_040",
        "risk_xgb_confidence_lt_060",
    ]
    frame["risk_score"] = frame[risk_columns].sum(axis=1)
    frame["final_action"] = "NO_SIGNAL"
    frame.loc[frame["strong_favorite"] & (frame["risk_score"] == 0), "final_action"] = "BET"
    frame.loc[frame["strong_favorite"] & (frame["risk_score"] >= 1), "final_action"] = "PASS"
    return frame


def has_odds(frame: pd.DataFrame) -> bool:
    return all(column in frame.columns for column in ODDS_COLUMNS)


def max_drawdown(profits: pd.Series) -> float:
    if profits.empty:
        return np.nan
    equity = profits.cumsum()
    running_max = equity.cummax()
    drawdown = equity - running_max
    return float(drawdown.min())


def odds_for_pick(row: pd.Series, pick: str) -> float:
    return float(row[RESULT_TO_ODDS_COLUMN[pick]])


def roi_metrics(group: pd.DataFrame, pick_column: str, segment: str) -> dict:
    group = add_bet_returns(group, pick_column)
    if group.empty:
        return {
            "segment": segment,
            "pick_type": pick_column,
            "total_bets": 0,
            "wins": 0,
            "accuracy": np.nan,
            "total_stake": 0.0,
            "total_profit": np.nan,
            "roi": np.nan,
            "average_odds": np.nan,
            "max_drawdown": np.nan,
        }

    wins = group["bet_win"]
    profits = group["profit"]
    total_stake = float(len(group))
    total_profit = float(profits.sum())

    return {
        "segment": segment,
        "pick_type": pick_column,
        "total_bets": len(group),
        "wins": int(wins.sum()),
        "accuracy": float(wins.mean()),
        "total_stake": total_stake,
        "total_profit": total_profit,
        "roi": total_profit / total_stake if total_stake else np.nan,
        "average_odds": float(group["bet_odds"].mean()),
        "max_drawdown": max_drawdown(profits),
    }


def accuracy_metrics(group: pd.DataFrame, pick_column: str, segment: str) -> dict:
    if group.empty:
        accuracy = np.nan
        wins = 0
    else:
        accuracy = float((group[pick_column] == group["actual_result"]).mean())
        wins = int((group[pick_column] == group["actual_result"]).sum())

    return {
        "segment": segment,
        "pick_type": pick_column,
        "total_bets": len(group),
        "wins": wins,
        "accuracy": accuracy,
        "total_stake": np.nan,
        "total_profit": np.nan,
        "roi": np.nan,
        "average_odds": np.nan,
        "max_drawdown": np.nan,
    }


def build_segments(frame: pd.DataFrame) -> dict[str, pd.DataFrame]:
    return {
        "unfiltered_strong_favorite": frame.loc[frame["strong_favorite"]],
        "filtered_bet_only": frame.loc[frame["final_action"] == "BET"],
        "pass_group": frame.loc[frame["final_action"] == "PASS"],
        "risk_score_0": frame.loc[frame["strong_favorite"] & (frame["risk_score"] == 0)],
        "risk_score_ge_1": frame.loc[frame["strong_favorite"] & (frame["risk_score"] >= 1)],
    }


def add_bet_returns(group: pd.DataFrame, pick_column: str) -> pd.DataFrame:
    group = group.dropna(subset=ODDS_COLUMNS).copy()
    if group.empty:
        return group
    group["pick_type"] = pick_column
    group["bet_pick"] = group[pick_column]
    group["bet_odds"] = group.apply(lambda row: odds_for_pick(row, row[pick_column]), axis=1)
    group["bet_win"] = group[pick_column] == group["actual_result"]
    group["profit"] = np.where(group["bet_win"], group["bet_odds"] - 1.0, -1.0)
    group["year"] = pd.to_datetime(group["date"], errors="coerce").dt.year
    group["odds_bucket"] = pd.cut(
        group["bet_odds"],
        bins=[1.0, 1.25, 1.40, 1.60, 1.80, 2.00, np.inf],
        labels=["1.00-1.25", "1.25-1.40", "1.40-1.60", "1.60-1.80", "1.80-2.00", "2.00+"],
        include_lowest=True,
        right=False,
    )
    return group


def roi_from_profit(group: pd.DataFrame, segment: str) -> dict:
    if group.empty:
        return {
            "segment": segment,
            "total_bets": 0,
            "wins": 0,
            "accuracy": np.nan,
            "total_stake": 0.0,
            "total_profit": np.nan,
            "roi": np.nan,
            "average_odds": np.nan,
            "max_drawdown": np.nan,
        }
    return {
        "segment": segment,
        "total_bets": len(group),
        "wins": int(group["bet_win"].sum()),
        "accuracy": float(group["bet_win"].mean()),
        "total_stake": float(len(group)),
        "total_profit": float(group["profit"].sum()),
        "roi": float(group["profit"].mean()),
        "average_odds": float(group["bet_odds"].mean()),
        "max_drawdown": max_drawdown(group["profit"]),
    }


def grouped_roi(group: pd.DataFrame, by_column: str) -> pd.DataFrame:
    rows = []
    for value, segment in group.groupby(by_column, observed=False):
        if segment.empty:
            continue
        metrics = roi_from_profit(segment, str(value))
        metrics[by_column] = value
        rows.append(metrics)
    return pd.DataFrame(rows)


def cumulative_profit_curve(group: pd.DataFrame) -> pd.DataFrame:
    curve = group.sort_values("date").copy()
    curve["bet_number"] = range(1, len(curve) + 1)
    curve["cumulative_profit"] = curve["profit"].cumsum()
    curve["running_max_profit"] = curve["cumulative_profit"].cummax()
    curve["drawdown"] = curve["cumulative_profit"] - curve["running_max_profit"]
    return curve[
        [
            "bet_number",
            "date",
            "league",
            "home_team",
            "away_team",
            "bet_pick",
            "actual_result",
            "bet_odds",
            "profit",
            "cumulative_profit",
            "drawdown",
        ]
    ]


def bootstrap_roi_ci(group: pd.DataFrame, iterations: int = BOOTSTRAP_ITERATIONS, seed: int = 42) -> pd.DataFrame:
    if group.empty:
        return pd.DataFrame(
            [
                {
                    "iterations": iterations,
                    "bets": 0,
                    "roi_mean": np.nan,
                    "roi_ci_lower_2_5": np.nan,
                    "roi_ci_upper_97_5": np.nan,
                    "observed_roi": np.nan,
                }
            ]
        )

    rng = np.random.default_rng(seed)
    profits = group["profit"].to_numpy()
    sampled_roi = []
    for _ in range(iterations):
        sample = rng.choice(profits, size=len(profits), replace=True)
        sampled_roi.append(sample.mean())

    return pd.DataFrame(
        [
            {
                "iterations": iterations,
                "bets": len(group),
                "roi_mean": float(np.mean(sampled_roi)),
                "roi_ci_lower_2_5": float(np.percentile(sampled_roi, 2.5)),
                "roi_ci_upper_97_5": float(np.percentile(sampled_roi, 97.5)),
                "observed_roi": float(profits.mean()),
                "top_10_wins_contribution": top_contribution(group, largest=True),
                "top_10_losses_contribution": top_contribution(group, largest=False),
            }
        ]
    )


def top_contribution(group: pd.DataFrame, largest: bool) -> float:
    total_profit = group["profit"].sum()
    if total_profit == 0:
        return np.nan
    ordered = group.sort_values("profit", ascending=not largest)
    return float(ordered.head(10)["profit"].sum() / total_profit)


def robustness_reports(frame: pd.DataFrame, pick_column: str = "xgb_pred") -> dict[str, pd.DataFrame]:
    filtered = add_bet_returns(frame.loc[frame["final_action"] == "BET"], pick_column)
    return {
        "by_year": grouped_roi(filtered, "year"),
        "by_league": grouped_roi(filtered, "league"),
        "by_odds_bucket": grouped_roi(filtered, "odds_bucket"),
        "cumulative_profit": cumulative_profit_curve(filtered),
        "bootstrap_ci": bootstrap_roi_ci(filtered),
    }


def backtest(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, bool]:
    odds_available = has_odds(frame)
    metric_fn = roi_metrics if odds_available else accuracy_metrics
    rows = []

    for segment, group in build_segments(frame).items():
        rows.append(metric_fn(group, "market_pred", segment))
        rows.append(metric_fn(group, "xgb_pred", segment))

    summary = pd.DataFrame(rows)

    score_rows = []
    strong = frame.loc[frame["strong_favorite"]]
    for risk_score, group in strong.groupby("risk_score"):
        score_rows.append(metric_fn(group, "market_pred", f"risk_score_{int(risk_score)}"))
        score_rows.append(metric_fn(group, "xgb_pred", f"risk_score_{int(risk_score)}"))

    by_score = pd.DataFrame(score_rows)
    return summary, by_score, odds_available


def run_backtest(csv_path: Path, model_path: Path, reports_dir: Path = REPORTS_DIR) -> tuple[pd.DataFrame, pd.DataFrame, bool]:
    frame = load_backtest_frame(csv_path, model_path)
    summary, by_score, odds_available = backtest(frame)

    reports_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(reports_dir / "roi_risk_filter_backtest.csv", index=False, encoding="utf-8")
    by_score.to_csv(reports_dir / "roi_risk_filter_by_risk_score.csv", index=False, encoding="utf-8")
    if odds_available:
        robust = robustness_reports(frame, pick_column="xgb_pred")
        robust["by_year"].to_csv(reports_dir / "roi_by_year.csv", index=False, encoding="utf-8")
        robust["by_league"].to_csv(reports_dir / "roi_by_league.csv", index=False, encoding="utf-8")
        robust["by_odds_bucket"].to_csv(reports_dir / "roi_by_odds_bucket.csv", index=False, encoding="utf-8")
        robust["bootstrap_ci"].to_csv(reports_dir / "roi_bootstrap_ci.csv", index=False, encoding="utf-8")
        robust["cumulative_profit"].to_csv(reports_dir / "roi_cumulative_profit.csv", index=False, encoding="utf-8")
    return summary, by_score, odds_available


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, default=META_FEATURES_PATH)
    parser.add_argument("--model", type=Path, default=XGB_META_MODEL_PATH)
    args = parser.parse_args()

    summary, by_score, odds_available = run_backtest(args.csv, args.model)
    if not odds_available:
        print("缺少 home_odds, draw_odds, away_odds，暫時只能做 accuracy backtest，不能做 ROI backtest。")

    print("Risk filter backtest:")
    print(summary.to_string(index=False))
    print("\nBy risk score:")
    print(by_score.to_string(index=False))


if __name__ == "__main__":
    main()
