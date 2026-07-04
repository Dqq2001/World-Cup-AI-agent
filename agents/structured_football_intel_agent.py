from __future__ import annotations

from datetime import datetime, timezone


class StructuredFootballIntelAgent:
    """Normalizes football-specific structured intelligence from match data sources."""

    CONTENT_COLUMNS = [
        "injuries_home",
        "injuries_away",
        "suspensions_home",
        "suspensions_away",
        "expected_lineup_home",
        "expected_lineup_away",
        "coach_comments_home",
        "coach_comments_away",
    ]
    OUTPUT_COLUMNS = [
        "match_key",
        "date",
        "home_team",
        "away_team",
        "source_name",
        "source_url",
        *CONTENT_COLUMNS,
        "confidence",
        "fetched_at",
        "intel_has_content",
    ]
    SOURCE_CONFIDENCE = {
        "manual": 1.0,
        "sofascore": 0.9,
        "fotmob": 0.85,
        "flashscore": 0.8,
        "news_search": 0.6,
        "cached": 0.5,
        "unknown": 0.2,
    }

    def normalize_row(self, row: dict) -> dict:
        output = {column: row.get(column, "unknown") for column in self.OUTPUT_COLUMNS}
        match_key = str(output.get("match_key", "")).strip()
        if not match_key or match_key.lower() in {"unknown", "nan", "none", "<na>"}:
            output["match_key"] = self.match_key(output)
        output["source_name"] = str(output.get("source_name") or "unknown").lower()
        output["confidence"] = self.confidence(output["source_name"], output.get("confidence"))
        output["fetched_at"] = output.get("fetched_at") or datetime.now(timezone.utc).isoformat()
        output["intel_has_content"] = self.has_content(output)
        return output

    def empty_row(self, date: str, home_team: str, away_team: str, source_name: str = "unknown") -> dict:
        return self.normalize_row(
            {
                "date": date,
                "home_team": home_team,
                "away_team": away_team,
                "source_name": source_name,
                "source_url": "unknown",
                "fetched_at": datetime.now(timezone.utc).isoformat(),
            }
        )

    def best_row(self, rows: list[dict], date: str, home_team: str, away_team: str) -> dict:
        normalized = [self.normalize_row(row) for row in rows]
        with_content = [row for row in normalized if row["intel_has_content"]]
        candidates = with_content or normalized
        if not candidates:
            return self.empty_row(date, home_team, away_team)
        return sorted(candidates, key=lambda row: float(row["confidence"]), reverse=True)[0]

    def has_content(self, row: dict) -> bool:
        for column in self.CONTENT_COLUMNS:
            value = str(row.get(column, "")).strip().lower()
            if value and value not in {"unknown", "nan", "none", "<na>"}:
                return True
        return False

    def confidence(self, source_name: str, explicit_value=None) -> float:
        try:
            if explicit_value not in (None, "", "unknown"):
                return float(explicit_value)
        except (TypeError, ValueError):
            pass
        return self.SOURCE_CONFIDENCE.get(str(source_name).lower(), self.SOURCE_CONFIDENCE["unknown"])

    def confidence_level(self, confidence: float) -> str:
        if confidence >= 0.8:
            return "HIGH"
        if confidence >= 0.5:
            return "MEDIUM"
        return "LOW"

    def match_key(self, row: dict) -> str:
        return f"{row.get('date', '')}|{row.get('home_team', '')}|{row.get('away_team', '')}"
