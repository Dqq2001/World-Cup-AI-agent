class FinalBettingAgent:
    """Input: value, upset risk, draw risk, and league reference outputs."""

    HIGH_DRAW_REFERENCE = "HIGH draw risk historically has higher draw rate and lower favorite accuracy"

    def run(self, value: dict, risk: dict, draw_risk: dict | None = None, league_reference: dict | None = None) -> dict:
        draw_risk = draw_risk or {"draw_risk_level": "LOW", "draw_risk_reasons": []}
        league_reference = league_reference or {
            "league_reference_level": "UNKNOWN",
            "league_reference_reasons": ["League reference unavailable."],
        }
        action, stake = self._base_action(value, risk)
        action, stake, draw_adjustment = self._apply_draw_risk_adjustment(action, stake, draw_risk)
        action, stake, league_adjustment = self._apply_league_reference_adjustment(action, stake, league_reference)

        return {
            "recommended_action": action,
            "recommended_stake": stake,
            "reason": (
                f"value={value['value_level']} edge={value['edge']:.3f}; "
                f"risk={risk['upset_risk']} ({'; '.join(risk['risk_reasons'])}); "
                f"draw_risk={draw_risk['draw_risk_level']} ({'; '.join(draw_risk['draw_risk_reasons'])}); "
                f"league_reference={league_reference['league_reference_level']} "
                f"({'; '.join(league_reference['league_reference_reasons'])})"
                f"{draw_adjustment}{league_adjustment}"
            ),
        }

    def _base_action(self, value: dict, risk: dict) -> tuple[str, float]:
        if risk["upset_risk"] == "HIGH":
            return "PASS", 0.0
        if value["value_level"] == "HIGH" and risk["upset_risk"] == "LOW":
            return "BET", 1.0
        if value["value_level"] == "MEDIUM" and risk["upset_risk"] == "MEDIUM":
            return "SMALL_BET", 0.5
        if value["value_level"] in {"MEDIUM", "HIGH"}:
            return "WATCH", 0.0
        return "PASS", 0.0

    def _apply_draw_risk_adjustment(self, action: str, stake: float, draw_risk: dict) -> tuple[str, float, str]:
        level = draw_risk["draw_risk_level"]
        if level == "HIGH" and action in {"BET", "SMALL_BET"}:
            return "WATCH", 0.0, f"; draw_adjustment={self.HIGH_DRAW_REFERENCE}"
        if level == "MEDIUM" and action == "BET":
            return "SMALL_BET", 0.5, "; draw_adjustment=MEDIUM draw risk downgraded BET to SMALL_BET"
        if level == "MEDIUM" and action == "SMALL_BET":
            return "WATCH", 0.0, "; draw_adjustment=MEDIUM draw risk downgraded SMALL_BET to WATCH"
        return action, stake, ""

    def _apply_league_reference_adjustment(self, action: str, stake: float, league_reference: dict) -> tuple[str, float, str]:
        level = league_reference["league_reference_level"]
        if level == "HIGH" and action == "BET":
            return "SMALL_BET", 0.5, "; league_reference_adjustment=HIGH downgraded BET to SMALL_BET"
        if level == "HIGH" and action == "SMALL_BET":
            return "WATCH", 0.0, "; league_reference_adjustment=HIGH downgraded SMALL_BET to WATCH"
        if level == "MEDIUM":
            return action, stake, "; league_reference_warning=MEDIUM auxiliary risk warning"
        return action, stake, ""
