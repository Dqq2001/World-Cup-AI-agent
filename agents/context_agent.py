import pandas as pd


class ContextAgent:
    """Input: one match row. Output: group context and pressure flags."""

    def run(self, row: pd.Series) -> dict:
        return {
            "group": row["group"],
            "points_home_before": int(row["points_home_before"]),
            "points_away_before": int(row["points_away_before"]),
            "goal_diff_home_before": int(row["goal_diff_home_before"]),
            "goal_diff_away_before": int(row["goal_diff_away_before"]),
            "must_win_home": bool(row["must_win_home"]),
            "must_win_away": bool(row["must_win_away"]),
            "already_qualified_home": bool(row["already_qualified_home"]),
            "already_qualified_away": bool(row["already_qualified_away"]),
            "neutral_venue": bool(row.get("neutral_venue", True)),
        }
