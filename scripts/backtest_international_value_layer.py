import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, log_loss

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.international_features import ID_TO_RESULT, RESULT_TO_ID, prepare_feature_matrix
from src.paths import INTERNATIONAL_WDL_MODEL_PATH

try:
    from xgboost import XGBClassifier
except ImportError as exc:
    raise ImportError("Missing dependency: install xgboost with `pip install xgboost`.") from exc


DEFAULT_INPUT = Path("data/processed/international_training_with_odds.csv")
DEFAULT_OUTPUT = Path("reports/international_value_layer_backtest.csv")
REQUIRED_COLUMNS = ["date", "result", "home_odds", "draw_odds", "away_odds", "market_H", "market_D", "market_A"]
BACKTEST_COLUMNS = [
    "date",
    "competition",
    "home_team",
    "away_team",
    "result",
    "model_pick",
    "market_pick",
    "value_side",
    "edge",
    "risk_score",
    "action",
    "odds",
    "profit",
]
SUMMARY_COLUMNS = [
    "rows_with_odds",
    "model_accuracy",
    "market_accuracy",
    "model_log_loss",
    "market_log_loss",
    "bets",
    "bet_accuracy",
    "profit",
    "roi",
]


def load_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"找不到含 odds 的 training data: {path}")
    data = pd.read_csv(path, encoding="utf-8")
    missing = [column for column in REQUIRED_COLUMNS if column not in data.columns]
    if missing:
        raise ValueError(f"資料缺少欄位: {missing}")
    data = data.dropna(subset=REQUIRED_COLUMNS)
    data = data[data["result"].isin(RESULT_TO_ID)]
    return data.sort_values("date").reset_index(drop=True)


def add_model_probabilities(data: pd.DataFrame, model_path: Path) -> pd.DataFrame:
    if not model_path.exists():
        raise FileNotFoundError(f"找不到國家隊 WDL 模型: {model_path}")
    model = XGBClassifier()
    model.load_model(model_path)
    probabilities = model.predict_proba(prepare_feature_matrix(data))
    classes = [int(value) for value in getattr(model, "classes_", [0, 1, 2])]
    output = data.copy()
    for class_id, values in zip(classes, probabilities.T):
        output[f"model_{ID_TO_RESULT[class_id]}"] = values
    for column in ["model_H", "model_D", "model_A"]:
        if column not in output.columns:
            output[column] = 0.0
    return output


def selected_odds(row: pd.Series, side: str) -> float:
    return float(row[{"H": "home_odds", "D": "draw_odds", "A": "away_odds"}[side]])


def build_backtest_rows(data: pd.DataFrame, edge_threshold: float) -> pd.DataFrame:
    rows = []
    for row in data.itertuples(index=False):
        row_data = pd.Series(row._asdict())
        model_probs = {"H": row_data["model_H"], "D": row_data["model_D"], "A": row_data["model_A"]}
        market_probs = {"H": row_data["market_H"], "D": row_data["market_D"], "A": row_data["market_A"]}
        edges = {side: float(model_probs[side] - market_probs[side]) for side in ["H", "D", "A"]}
        value_side = max(edges, key=edges.get)
        edge = edges[value_side]
        model_pick = max(model_probs, key=model_probs.get)
        market_pick = max(market_probs, key=market_probs.get)
        risk_score = 0
        if model_pick != market_pick:
            risk_score += 1
        if max(model_probs.values()) < 0.45:
            risk_score += 1
        if max(market_probs.values()) < 0.45:
            risk_score += 1
        action = "BET" if edge >= edge_threshold and risk_score <= 1 else "PASS"
        win = value_side == row_data["result"]
        profit = selected_odds(row_data, value_side) - 1 if action == "BET" and win else (-1 if action == "BET" else 0)
        rows.append(
            {
                "date": row_data["date"],
                "competition": row_data.get("competition", ""),
                "home_team": row_data.get("home_team", ""),
                "away_team": row_data.get("away_team", ""),
                "result": row_data["result"],
                "model_pick": model_pick,
                "market_pick": market_pick,
                "value_side": value_side,
                "edge": edge,
                "risk_score": risk_score,
                "action": action,
                "odds": selected_odds(row_data, value_side),
                "profit": profit,
            }
        )
    return pd.DataFrame(rows)


def summarize(data: pd.DataFrame, backtest: pd.DataFrame) -> pd.DataFrame:
    if backtest.empty:
        return pd.DataFrame()
    y_true = data["result"].map(RESULT_TO_ID)
    model_probs = data[["model_H", "model_D", "model_A"]]
    market_probs = data[["market_H", "market_D", "market_A"]]
    model_pred = model_probs.idxmax(axis=1).str[-1]
    market_pred = market_probs.idxmax(axis=1).str[-1]
    bet_rows = backtest[backtest["action"] == "BET"]
    return pd.DataFrame(
        [
            {
                "rows_with_odds": len(data),
                "model_accuracy": accuracy_score(data["result"], model_pred),
                "market_accuracy": accuracy_score(data["result"], market_pred),
                "model_log_loss": log_loss(y_true, model_probs, labels=[0, 1, 2]),
                "market_log_loss": log_loss(y_true, market_probs, labels=[0, 1, 2]),
                "bets": len(bet_rows),
                "bet_accuracy": np.nan if bet_rows.empty else float((bet_rows["value_side"] == bet_rows["result"]).mean()),
                "profit": float(bet_rows["profit"].sum()) if not bet_rows.empty else 0.0,
                "roi": np.nan if bet_rows.empty else float(bet_rows["profit"].sum() / len(bet_rows)),
            }
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--model", type=Path, default=INTERNATIONAL_WDL_MODEL_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--edge-threshold", type=float, default=0.05)
    args = parser.parse_args()

    data = load_data(args.input)
    if data.empty:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(columns=BACKTEST_COLUMNS).to_csv(args.output, index=False, encoding="utf-8")
        pd.DataFrame(columns=SUMMARY_COLUMNS).to_csv(args.output.with_name("international_value_layer_backtest_summary.csv"), index=False, encoding="utf-8")
        print("沒有可回測的 historical odds rows；未假造 odds。")
        return

    data = add_model_probabilities(data, args.model)
    backtest = build_backtest_rows(data, args.edge_threshold)
    summary = summarize(data, backtest)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    backtest.to_csv(args.output, index=False, encoding="utf-8")
    summary.to_csv(args.output.with_name("international_value_layer_backtest_summary.csv"), index=False, encoding="utf-8")
    print(summary.to_string(index=False))
    print(f"backtest output: {args.output}")


if __name__ == "__main__":
    main()
