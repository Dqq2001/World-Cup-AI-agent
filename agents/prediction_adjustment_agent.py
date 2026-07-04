import math
from dataclasses import dataclass

import pandas as pd


@dataclass
class AdjustmentResult:
    adjusted_H: float
    adjusted_D: float
    adjusted_A: float
    adjustment_score_home: float
    adjustment_score_away: float
    adjustment_reasons: str


class PredictionAdjustmentAgent:
    """Weighted pre-match adjustment layer for WDL probabilities."""

    BASE_WEIGHTS = {
        "odds": 0.35,
        "base": 0.30,
        "form": 0.15,
        "injury": 0.10,
        "fatigue": 0.07,
        "intel": 0.03,
    }

    NEGATIVE_TERMS = ["injury", "injured", "doubtful", "out", "ruled out", "miss", "muscle", "suspended", "ban"]
    NO_ISSUE_TERMS = ["no reported", "none", "no injuries", "out: none", "doubtful: none", "no suspension"]

    def run(self, row: pd.Series | dict) -> dict:
        base = self._base_probs(row)
        market = self._market_probs(row)
        weights = self._weights(market)

        form_home = self._float(row, "home_recent_form_score", self._float(row, "recent_form_home", 0.5))
        form_away = self._float(row, "away_recent_form_score", self._float(row, "recent_form_away", 0.5))
        form_probs = self._normalize([0.35 + 0.30 * form_home, 0.25, 0.35 + 0.30 * form_away])

        blended = [
            weights["base"] * base[0] + weights["form"] * form_probs[0],
            weights["base"] * base[1] + weights["form"] * form_probs[1],
            weights["base"] * base[2] + weights["form"] * form_probs[2],
        ]
        if market:
            blended = [
                blended[0] + weights["odds"] * market[0],
                blended[1] + weights["odds"] * market[1],
                blended[2] + weights["odds"] * market[2],
            ]

        home_score = 0.0
        away_score = 0.0
        reasons: list[str] = []

        injury_home, injury_away, injury_reasons = self._injury_adjustment(row)
        home_score += injury_home
        away_score += injury_away
        reasons.extend(injury_reasons)

        fatigue_home, fatigue_away, fatigue_reasons = self._fatigue_adjustment(row)
        home_score += fatigue_home
        away_score += fatigue_away
        reasons.extend(fatigue_reasons)

        intel_reason = self._intel_reason(row)
        if intel_reason:
            reasons.append(intel_reason)

        blended[0] += home_score
        blended[2] += away_score
        if self._xg_close(row):
            blended[1] += 0.015
            reasons.append("Poisson xG is close, slightly supporting draw probability.")

        adjusted = self._normalize(blended)
        if market:
            reasons.append("Odds signal included with 35% target weight.")
        else:
            reasons.append("Odds missing; odds weight redistributed to base model and recent form.")
        reasons.append(f"Recent form scores: home={form_home:.2f}, away={form_away:.2f}.")

        return {
            "adjusted_H": adjusted[0],
            "adjusted_D": adjusted[1],
            "adjusted_A": adjusted[2],
            "adjustment_score_home": home_score,
            "adjustment_score_away": away_score,
            "adjustment_reasons": "; ".join(reasons),
        }

    def _weights(self, market: list[float] | None) -> dict[str, float]:
        weights = self.BASE_WEIGHTS.copy()
        if market is None:
            weights["base"] += 0.25
            weights["form"] += 0.10
            weights["odds"] = 0.0
        return weights

    def _base_probs(self, row) -> list[float]:
        return self._normalize([
            self._float(row, "model_H", 1 / 3),
            self._float(row, "model_D", 1 / 3),
            self._float(row, "model_A", 1 / 3),
        ])

    def _market_probs(self, row) -> list[float] | None:
        market = [self._float(row, "market_H"), self._float(row, "market_D"), self._float(row, "market_A")]
        if all(value is not None and value > 0 for value in market):
            return self._normalize(market)
        odds = [self._float(row, "home_odds"), self._float(row, "draw_odds"), self._float(row, "away_odds")]
        if not all(value is not None and value > 1 for value in odds):
            return None
        implied = [1 / value for value in odds]
        return self._normalize(implied)

    def _injury_adjustment(self, row) -> tuple[float, float, list[str]]:
        if not self._can_use_intel(row):
            return 0.0, 0.0, ["News/injury adjustment skipped because intel confidence or source URLs are insufficient."]
        reasons = []
        home_penalty = self._team_penalty(row, "home")
        away_penalty = self._team_penalty(row, "away")
        risk_level = self._text(row, "intel_risk").upper()
        if risk_level == "MEDIUM":
            home_penalty *= 0.5
            away_penalty *= 0.5
            reasons.append("MEDIUM intel risk; availability impact is applied at half strength.")
        if home_penalty:
            reasons.append(f"Home team availability penalty applied: {home_penalty:.3f}.")
        if away_penalty:
            reasons.append(f"Away team availability penalty applied: {away_penalty:.3f}.")
        return -home_penalty, -away_penalty, reasons

    def _team_penalty(self, row, side: str) -> float:
        injury_text = self._text(row, f"injuries_{side}")
        suspension_text = self._text(row, f"suspensions_{side}")
        penalty = 0.0
        if self._has_negative_availability(injury_text):
            penalty += 0.05 if self._has_key_player_signal(injury_text) else 0.03
        if self._has_negative_availability(suspension_text):
            penalty += 0.03
        return min(penalty, 0.08)

    def _fatigue_adjustment(self, row) -> tuple[float, float, list[str]]:
        home_rest = self._float(row, "rest_days_home")
        away_rest = self._float(row, "rest_days_away")
        if home_rest is None or away_rest is None:
            return 0.0, 0.0, []
        diff = home_rest - away_rest
        if diff >= 2:
            return 0.0, -0.03, [f"Away team has {diff:.0f} fewer rest days; fatigue penalty applied."]
        if diff <= -2:
            return -0.03, 0.0, [f"Home team has {abs(diff):.0f} fewer rest days; fatigue penalty applied."]
        return 0.0, 0.0, []

    def _intel_reason(self, row) -> str:
        confidence = self._float(row, "confidence")
        source_count = self._source_urls_count(row)
        if confidence is None or confidence < 0.5:
            return "Intel confidence is LOW; news impact is capped."
        if source_count == 0:
            return "No source URLs; news impact is disabled."
        return f"Intel confidence considered with {source_count} source URL(s)."

    def _can_use_intel(self, row) -> bool:
        if self._text(row, "intel_risk").upper() == "UNKNOWN":
            return False
        confidence = self._float(row, "confidence")
        return confidence is not None and confidence >= 0.5 and self._source_urls_count(row) > 0

    def _source_urls_count(self, row) -> int:
        value = self._text(row, "source_urls") or self._text(row, "source_url")
        if not value or value.lower() == "unknown":
            return 0
        return len([part for part in value.replace(",", ";").split(";") if part.strip().startswith(("http://", "https://"))])

    def _has_negative_availability(self, text: str) -> bool:
        lower = text.lower()
        if not lower or lower == "unknown":
            return False
        explicit_negative = any(
            term in lower
            for term in [
                "ruled out",
                "listed as doubtful",
                "doubtful with",
                "muscle issue",
                "will miss",
                "is suspended",
                "are suspended",
            ]
        )
        if any(term in lower for term in self.NO_ISSUE_TERMS) and not explicit_negative:
            return False
        return any(term in lower for term in self.NEGATIVE_TERMS)

    def _has_key_player_signal(self, text: str) -> bool:
        return any(term in text.lower() for term in ["key", "captain", "star", "main", "important", "doubtful"])

    def _xg_close(self, row) -> bool:
        home = self._float(row, "poisson_home_xg")
        away = self._float(row, "poisson_away_xg")
        return home is not None and away is not None and abs(home - away) <= 0.25

    def _normalize(self, values: list[float]) -> list[float]:
        cleaned = [max(0.001, float(value)) if value is not None and not math.isnan(float(value)) else 0.001 for value in values]
        total = sum(cleaned)
        return [value / total for value in cleaned]

    def _float(self, row, column: str, default=None):
        value = row.get(column, default) if isinstance(row, dict) else row[column] if column in row else default
        if pd.isna(value):
            return default
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    def _text(self, row, column: str) -> str:
        value = row.get(column, "") if isinstance(row, dict) else row[column] if column in row else ""
        if pd.isna(value):
            return ""
        return str(value).strip()
