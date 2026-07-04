from pathlib import Path

import pandas as pd


class DataAgent:
    """Input: World Cup feature CSV path. Output: data frame plus missing-data diagnostics."""

    CORE_COLUMNS = [
        "date",
        "home_team",
        "away_team",
        "model_H",
        "model_D",
        "model_A",
    ]
    OPTIONAL_DEFAULTS = {
        "group": "",
        "market_H": pd.NA,
        "market_D": pd.NA,
        "market_A": pd.NA,
        "poisson_home_xg": pd.NA,
        "poisson_away_xg": pd.NA,
        "poisson_top_scores": "",
        "poisson_diff": pd.NA,
        "points_home_before": 0,
        "points_away_before": 0,
        "goal_diff_home_before": 0,
        "goal_diff_away_before": 0,
        "must_win_home": False,
        "must_win_away": False,
        "already_qualified_home": False,
        "already_qualified_away": False,
        "home_odds": pd.NA,
        "draw_odds": pd.NA,
        "away_odds": pd.NA,
    }

    def run(self, features_path: Path) -> dict:
        if not features_path.exists():
            return {
                "ok": False,
                "data": pd.DataFrame(),
                "errors": [f"缺少 World Cup features 檔案: {features_path}"],
            }

        data = pd.read_csv(features_path, encoding="utf-8")
        missing_columns = [column for column in self.CORE_COLUMNS if column not in data.columns]
        if missing_columns:
            return {
                "ok": False,
                "data": data,
                "errors": [f"World Cup features 缺少必要欄位: {missing_columns}"],
            }

        for column, default in self.OPTIONAL_DEFAULTS.items():
            if column not in data.columns:
                if column == "poisson_diff" and {"poisson_home_xg", "poisson_away_xg"}.issubset(data.columns):
                    data[column] = pd.to_numeric(data["poisson_away_xg"], errors="coerce") - pd.to_numeric(
                        data["poisson_home_xg"], errors="coerce"
                    )
                else:
                    data[column] = default

        row_missing = data[self.CORE_COLUMNS].isna().sum()
        row_missing = row_missing[row_missing > 0]
        if not row_missing.empty:
            return {
                "ok": False,
                "data": data,
                "errors": [f"World Cup features 含缺失值: {row_missing.to_dict()}"],
            }

        data["date"] = pd.to_datetime(data["date"], errors="coerce").dt.strftime("%Y-%m-%d")
        if data["date"].isna().any():
            return {
                "ok": False,
                "data": data,
                "errors": ["World Cup features 含無效日期。"],
            }

        return {"ok": True, "data": data, "errors": []}
