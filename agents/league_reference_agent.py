from pathlib import Path

import pandas as pd


class LeagueReferenceAgent:
    """Uses transferable league-model risk patterns as auxiliary warnings only."""

    REPORT_FILES = [
        "roi_risk_filter_backtest.csv",
        "roi_risk_filter_by_risk_score.csv",
        "xgb_error_analysis_by_confidence.csv",
        "xgb_error_analysis_disagreement.csv",
    ]

    def __init__(self, reports_dir: Path | str = "reports") -> None:
        self.reports_dir = Path(reports_dir)
        self.available_reports = self._load_available_reports()

    def run(self, row, market: dict | None = None, poisson: dict | None = None, draw_risk: dict | None = None) -> dict:
        if not self.available_reports:
            return {
                "league_reference_available": False,
                "league_risk_score": 0,
                "league_reference_level": "UNKNOWN",
                "league_reference_reasons": ["League reference reports unavailable; no action impact."],
            }

        market_confidence = self._market_confidence(row, market)
        poisson_diff = self._poisson_diff(row, poisson)
        draw_level = (draw_risk or {}).get("draw_risk_level", row.get("draw_risk_level", "UNKNOWN"))
        score = 0
        reasons = ["Five-league references are auxiliary only due to domain shift."]

        if market_confidence is not None and 0.60 <= market_confidence <= 0.65:
            score += 1
            reasons.append("Transferable pattern: strong favorite with market confidence 0.60-0.65 had higher risk.")
        if poisson_diff is not None and 0.75 <= abs(poisson_diff) <= 1.00:
            score += 1
            reasons.append("Transferable pattern: abs_poisson_diff 0.75-1.00 had elevated upset risk.")
        if draw_level == "HIGH":
            score += 2
            reasons.append("Transferable pattern: high draw risk means avoid favorite 1X2 exposure.")
        if score == 0 and market_confidence is not None and market_confidence >= 0.60:
            reasons.append("Transferable pattern: risk_score=0 favorite was comparatively steadier.")
        if score >= 1:
            reasons.append("risk_score>=1 is warning only; it must not override international model.")

        if score >= 2:
            level = "HIGH"
        elif score == 1:
            level = "MEDIUM"
        else:
            level = "LOW"

        return {
            "league_reference_available": True,
            "league_risk_score": score,
            "league_reference_level": level,
            "league_reference_reasons": reasons,
        }

    def _load_available_reports(self) -> dict[str, pd.DataFrame]:
        reports = {}
        for filename in self.REPORT_FILES:
            path = self.reports_dir / filename
            if not path.exists():
                continue
            try:
                reports[filename] = pd.read_csv(path, encoding="utf-8")
            except pd.errors.EmptyDataError:
                continue
        return reports

    def _market_confidence(self, row, market: dict | None) -> float | None:
        if market and "market_confidence" in market:
            return float(market["market_confidence"])
        values = [row.get("market_H"), row.get("market_D"), row.get("market_A")]
        if any(pd.isna(value) for value in values):
            return None
        return float(max(values))

    def _poisson_diff(self, row, poisson: dict | None) -> float | None:
        if poisson and "poisson_diff" in poisson:
            return float(poisson["poisson_diff"])
        if "poisson_diff" in row and not pd.isna(row["poisson_diff"]):
            return float(row["poisson_diff"])
        if "poisson_home_xg" in row and "poisson_away_xg" in row:
            return float(row["poisson_away_xg"] - row["poisson_home_xg"])
        return None
