from __future__ import annotations

from datetime import datetime, timezone

from agents.article_intel_parser import ArticleIntelParser


class TheAnalystIntelAgent:
    """Builds World Cup intelligence rows from The Analyst preview articles."""

    SOURCE_NAME = "The Analyst"

    def __init__(self) -> None:
        self.article_parser = ArticleIntelParser()

    def parse_url(self, fixture, url: str) -> dict:
        article = self.article_parser.fetch(url)
        intel = self.article_parser.extract_intel(article)
        fetched_at = datetime.now(timezone.utc).isoformat()
        return {
            "match_key": f"{fixture.date}|{fixture.home_team}|{fixture.away_team}",
            "date": fixture.date,
            "home_team": fixture.home_team,
            "away_team": fixture.away_team,
            "source_name": self.SOURCE_NAME,
            "source_url": url,
            "title": article.get("title", "unknown") or "unknown",
            "body_char_count": article.get("body_char_count", 0),
            "injuries_home": intel["injuries"],
            "injuries_away": intel["injuries"],
            "suspensions_home": intel["suspensions"],
            "suspensions_away": intel["suspensions"],
            "expected_lineup_home": intel["expected_lineup"],
            "expected_lineup_away": intel["expected_lineup"],
            "coach_comments_home": intel["coach_comments"],
            "coach_comments_away": intel["coach_comments"],
            "intel_has_content": bool(intel["intel_has_content"]),
            "confidence": float(intel["confidence"]),
            "fetched_at": fetched_at,
            "_fetch_success": bool(article.get("fetch_success")),
            "_parser_success": bool(article.get("parser_success")),
            "_failure_reason": article.get("failure_reason", ""),
        }
