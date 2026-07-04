import pandas as pd


class RiskAgent:
    """Input: market, model, poisson, and context outputs. Output: risk level and reasons."""

    def run(self, row: pd.Series, market: dict, poisson: dict, context: dict) -> dict:
        risk_points = 0
        reasons = []

        if market["market_confidence"] < 0.45:
            risk_points += 1
            reasons.append("市場信心偏低")

        if market["market_margin"] < 0.08:
            risk_points += 1
            reasons.append("市場機率接近，屬於五五波")

        if abs(poisson["poisson_diff"]) < 0.35:
            risk_points += 1
            reasons.append("Poisson 期望進球差距小")

        if context["must_win_home"] or context["must_win_away"]:
            risk_points += 1
            reasons.append("小組賽形勢有必勝壓力")

        if risk_points >= 2:
            level = "HIGH"
        elif risk_points == 1:
            level = "MEDIUM"
        else:
            level = "LOW"

        return {
            "upset_risk": level,
            "risk_score": risk_points,
            "risk_reasons": reasons or ["主要風險訊號不明顯"],
        }
