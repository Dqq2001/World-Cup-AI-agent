import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_worldcup_betting_agents import HIGH_DRAW_RISK_CSV, DRAW_RISK_SUMMARY_CSV, run_model_only_agents


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=PROJECT_ROOT / "reports" / "worldcup_model_only_predictions.csv")
    args = parser.parse_args()

    if not args.input.exists():
        run_model_only_agents()

    data = pd.read_csv(args.input, encoding="utf-8")
    summary = (
        data.groupby("draw_risk_level", as_index=False)
        .agg(
            count=("draw_risk_level", "size"),
            avg_model_D=("model_D", "mean"),
            avg_draw_risk_score=("draw_risk_score", "mean"),
        )
        .sort_values("draw_risk_level")
    )
    summary.to_csv(DRAW_RISK_SUMMARY_CSV, index=False, encoding="utf-8")
    data[data["draw_risk_level"] == "HIGH"].to_csv(HIGH_DRAW_RISK_CSV, index=False, encoding="utf-8")
    print(summary.to_string(index=False))
    print(f"summary: {DRAW_RISK_SUMMARY_CSV}")
    print(f"high risk: {HIGH_DRAW_RISK_CSV}")


if __name__ == "__main__":
    main()
