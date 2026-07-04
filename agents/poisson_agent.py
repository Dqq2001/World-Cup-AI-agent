import math

import pandas as pd


class PoissonAgent:
    """Input: one match row with expected goals. Output: xG, diff, and top scorelines."""

    def run(self, row: pd.Series, max_goals: int = 5) -> dict:
        home_xg = pd.to_numeric(row.get("poisson_home_xg"), errors="coerce")
        away_xg = pd.to_numeric(row.get("poisson_away_xg"), errors="coerce")
        if pd.isna(home_xg) or pd.isna(away_xg):
            return {
                "poisson_home_xg": None,
                "poisson_away_xg": None,
                "poisson_diff": 999.0,
                "poisson_top_scores": [],
            }

        home_xg = float(home_xg)
        away_xg = float(away_xg)
        scorelines = []

        for home_goals in range(max_goals + 1):
            for away_goals in range(max_goals + 1):
                home_prob = math.exp(-home_xg) * home_xg**home_goals / math.factorial(home_goals)
                away_prob = math.exp(-away_xg) * away_xg**away_goals / math.factorial(away_goals)
                scorelines.append(
                    {
                        "scoreline": f"{home_goals}-{away_goals}",
                        "probability": home_prob * away_prob,
                    }
                )

        total = sum(item["probability"] for item in scorelines)
        if total > 0:
            for item in scorelines:
                item["probability"] = item["probability"] / total

        return {
            "poisson_home_xg": home_xg,
            "poisson_away_xg": away_xg,
            "poisson_diff": away_xg - home_xg,
            "poisson_top_scores": sorted(scorelines, key=lambda item: item["probability"], reverse=True)[:5],
        }
