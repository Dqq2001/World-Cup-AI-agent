import json

import pandas as pd


class DrawRiskAgent:
    """Input: one match row. Output: rule-based draw risk score, level, and reasons."""

    def run(self, row: pd.Series) -> dict:
        score = 0
        reasons = []

        if pd.isna(row.get("poisson_home_xg")) or pd.isna(row.get("poisson_away_xg")) or pd.isna(row.get("model_D")):
            return {
                "draw_risk_score": 0,
                "draw_risk_level": "UNKNOWN",
                "draw_risk_reasons": ["Waiting for teams or missing model inputs"],
            }

        home_xg = float(row["poisson_home_xg"])
        away_xg = float(row["poisson_away_xg"])
        xg_gap = abs(home_xg - away_xg)
        if xg_gap <= 0.25:
            score += 2
            reasons.append("Poisson expected goals are very close")
        elif xg_gap <= 0.50:
            score += 1
            reasons.append("Poisson expected goals are close")

        model_draw = float(row["model_D"])
        if model_draw >= 0.30:
            score += 2
            reasons.append("Model draw probability is high")
        elif model_draw >= 0.25:
            score += 1
            reasons.append("Model draw probability is elevated")

        top_scores = self._scoreline_text(row.get("poisson_top_scores", ""))
        if "0-0" in top_scores or "1-1" in top_scores:
            score += 1
            reasons.append("Poisson top scorelines include 0-0 or 1-1")

        if bool(row.get("neutral_venue", True)):
            score += 1
            reasons.append("Neutral venue")

        group_matchday = int(row.get("group_matchday", 0))
        must_win_home = bool(row.get("must_win_home", False))
        must_win_away = bool(row.get("must_win_away", False))
        if group_matchday == 3 and not must_win_home and not must_win_away:
            score += 2
            reasons.append("Group matchday 3 with both teams not marked must-win")
        elif group_matchday == 3:
            score += 1
            reasons.append("Group matchday 3")

        if score >= 5:
            level = "HIGH"
        elif score >= 3:
            level = "MEDIUM"
        else:
            level = "LOW"

        if self._is_knockout(row) and level in {"MEDIUM", "HIGH"}:
            score += 1
            reasons.append("Knockout match: draw risk implies extra-time risk")
            if score >= 5:
                level = "HIGH"
            elif score >= 3:
                level = "MEDIUM"

        return {
            "draw_risk_score": score,
            "draw_risk_level": level,
            "draw_risk_reasons": reasons or ["No major draw-risk signal"],
        }

    def _scoreline_text(self, value) -> str:
        if pd.isna(value):
            return ""
        text = str(value)
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            return text
        if isinstance(parsed, list):
            return "; ".join(str(item.get("scoreline", "")) for item in parsed)
        return text

    def _is_knockout(self, row: pd.Series) -> bool:
        stage = str(row.get("stage", "")).strip().lower()
        round_name = str(row.get("round", "")).strip().lower()
        knockout_terms = ["round of", "quarter", "semi", "final"]
        return any(term in stage for term in knockout_terms) or any(term in round_name for term in knockout_terms)
