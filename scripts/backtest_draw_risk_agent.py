import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agents.draw_risk_agent import DrawRiskAgent
from src.international_features import ID_TO_RESULT, RESULT_TO_ID, chronological_split, prepare_feature_matrix
from src.paths import (
    INTERNATIONAL_POISSON_AWAY_PATH,
    INTERNATIONAL_POISSON_HOME_PATH,
    INTERNATIONAL_TRAINING_PATH,
    INTERNATIONAL_WDL_MODEL_PATH,
)

try:
    from xgboost import XGBClassifier, XGBRegressor
except ImportError as exc:
    raise ImportError("Missing dependency: install xgboost with `pip install xgboost`.") from exc


SUMMARY_OUTPUT = PROJECT_ROOT / "reports" / "draw_risk_backtest_summary.csv"
SAMPLES_OUTPUT = PROJECT_ROOT / "reports" / "draw_risk_backtest_samples.csv"


def poisson_pmf(goal_count: int, expected_goals: float) -> float:
    expected_goals = max(float(expected_goals), 1e-9)
    return math.exp(-expected_goals) * expected_goals**goal_count / math.factorial(goal_count)


def top_scorelines(home_xg: float, away_xg: float, max_goals: int = 5, top_n: int = 5) -> str:
    rows = []
    for home_goals in range(max_goals + 1):
        home_probability = poisson_pmf(home_goals, home_xg)
        for away_goals in range(max_goals + 1):
            probability = home_probability * poisson_pmf(away_goals, away_xg)
            rows.append({"scoreline": f"{home_goals}-{away_goals}", "probability": round(probability, 6)})
    rows = sorted(rows, key=lambda item: item["probability"], reverse=True)[:top_n]
    return json.dumps(rows, ensure_ascii=False)


def load_test_data(path: Path, test_size: float) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"找不到 international training data: {path}")
    data = pd.read_csv(path, encoding="utf-8")
    required = ["date", "result", "neutral", "home_team", "away_team"]
    missing = [column for column in required if column not in data.columns]
    if missing:
        raise ValueError(f"training data 缺少欄位: {missing}")
    data["date"] = pd.to_datetime(data["date"], errors="coerce")
    data = data.dropna(subset=["date", "result"])
    data = data[data["result"].isin(RESULT_TO_ID)]
    _, test_data = chronological_split(data, test_size=test_size)
    return test_data.reset_index(drop=True)


def add_model_predictions(data: pd.DataFrame) -> pd.DataFrame:
    for path in [INTERNATIONAL_WDL_MODEL_PATH, INTERNATIONAL_POISSON_HOME_PATH, INTERNATIONAL_POISSON_AWAY_PATH]:
        if not path.exists():
            raise FileNotFoundError(f"找不到已訓練模型: {path}")

    X = prepare_feature_matrix(data)
    wdl_model = XGBClassifier()
    wdl_model.load_model(INTERNATIONAL_WDL_MODEL_PATH)
    probabilities = wdl_model.predict_proba(X)
    classes = [int(value) for value in getattr(wdl_model, "classes_", [0, 1, 2])]

    output = data.copy()
    for class_id, values in zip(classes, probabilities.T):
        output[f"model_{ID_TO_RESULT[class_id]}"] = values
    for column in ["model_H", "model_D", "model_A"]:
        if column not in output.columns:
            output[column] = 0.0

    home_model = XGBRegressor()
    away_model = XGBRegressor()
    home_model.load_model(INTERNATIONAL_POISSON_HOME_PATH)
    away_model.load_model(INTERNATIONAL_POISSON_AWAY_PATH)
    output["poisson_home_xg"] = np.clip(home_model.predict(X), 0, None)
    output["poisson_away_xg"] = np.clip(away_model.predict(X), 0, None)
    output["poisson_diff"] = output["poisson_away_xg"] - output["poisson_home_xg"]
    output["poisson_top_scores"] = [
        top_scorelines(home_xg, away_xg)
        for home_xg, away_xg in zip(output["poisson_home_xg"], output["poisson_away_xg"])
    ]
    return output


def add_draw_risk(data: pd.DataFrame) -> pd.DataFrame:
    agent = DrawRiskAgent()
    output = data.copy()
    output["neutral_venue"] = output["neutral"].astype(bool)
    output["group_matchday"] = 0
    output["must_win_home"] = False
    output["must_win_away"] = False

    rows = [agent.run(row) for _, row in output.iterrows()]
    risk_data = pd.DataFrame(rows)
    return pd.concat([output.reset_index(drop=True), risk_data.reset_index(drop=True)], axis=1)


def summarize(data: pd.DataFrame) -> pd.DataFrame:
    data = data.copy()
    data["model_pred"] = data[["model_H", "model_D", "model_A"]].idxmax(axis=1).str[-1]
    data["favorite"] = data["model_pred"]
    data["is_draw"] = data["result"] == "D"
    data["model_correct"] = data["model_pred"] == data["result"]
    data["favorite_win"] = data["favorite"] == data["result"]
    data["upset_or_draw"] = data["result"] != data["favorite"]

    summary = (
        data.groupby("draw_risk_level", as_index=False)
        .agg(
            count=("draw_risk_level", "size"),
            actual_draw_rate=("is_draw", "mean"),
            model_accuracy=("model_correct", "mean"),
            favorite_win_rate=("favorite_win", "mean"),
            upset_or_draw_rate=("upset_or_draw", "mean"),
            avg_draw_risk_score=("draw_risk_score", "mean"),
            avg_model_D=("model_D", "mean"),
        )
        .sort_values("draw_risk_level")
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=INTERNATIONAL_TRAINING_PATH)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--summary-output", type=Path, default=SUMMARY_OUTPUT)
    parser.add_argument("--samples-output", type=Path, default=SAMPLES_OUTPUT)
    args = parser.parse_args()

    test_data = load_test_data(args.input, args.test_size)
    predictions = add_model_predictions(test_data)
    scored = add_draw_risk(predictions)
    summary = summarize(scored)

    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.summary_output, index=False, encoding="utf-8")

    sample_columns = [
        "date",
        "competition",
        "home_team",
        "away_team",
        "result",
        "model_H",
        "model_D",
        "model_A",
        "poisson_home_xg",
        "poisson_away_xg",
        "poisson_top_scores",
        "draw_risk_level",
        "draw_risk_score",
        "draw_risk_reasons",
    ]
    scored[sample_columns].to_csv(args.samples_output, index=False, encoding="utf-8")

    print(summary.to_string(index=False))
    low = summary.loc[summary["draw_risk_level"] == "LOW", "actual_draw_rate"]
    high = summary.loc[summary["draw_risk_level"] == "HIGH", "actual_draw_rate"]
    if not low.empty and not high.empty:
        print(f"HIGH vs LOW actual_draw_rate delta: {float(high.iloc[0] - low.iloc[0]):.4f}")
    print(f"summary: {args.summary_output}")
    print(f"samples: {args.samples_output}")


if __name__ == "__main__":
    main()
