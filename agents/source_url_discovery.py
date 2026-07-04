from __future__ import annotations

import re


class SourceUrlDiscovery:
    """Deprecated compatibility wrapper. Article intel now only uses The Analyst."""

    SOURCE_NAME = "The Analyst"

    def discover(self, fixture) -> tuple[list[dict], list[dict]]:
        candidates = []
        debug_rows = []
        for query_or_slug, url in self.theanalyst_direct_urls(str(fixture.home_team), str(fixture.away_team)):
            candidates.append(
                {
                    "source_name": self.SOURCE_NAME,
                    "strategy": "direct_slug",
                    "query": query_or_slug,
                    "candidate_url": url,
                }
            )
            debug_rows.append(
                {
                    "match_key": f"{fixture.date}|{fixture.home_team}|{fixture.away_team}",
                    "source_name": self.SOURCE_NAME,
                    "strategy": "direct_slug",
                    "query": query_or_slug,
                    "url_found": True,
                    "candidate_url": url,
                    "http_status": "",
                    "error_message": "theanalyst_only_mode",
                }
            )
        return candidates, debug_rows

    def theanalyst_direct_urls(self, home_team: str, away_team: str) -> list[tuple[str, str]]:
        home = self.team_slug(home_team)
        away = self.team_slug(away_team)
        base = "https://theanalyst.com/articles"
        return [
            ("direct_slug_home_away", f"{base}/{home}-vs-{away}-prediction-world-cup-2026-match-preview"),
            ("direct_slug_away_home", f"{base}/{away}-vs-{home}-prediction-world-cup-2026-match-preview"),
        ]

    def team_slug(self, team: str) -> str:
        aliases = {
            "United States": "usa",
            "USA": "usa",
            "Bosnia and Herzegovina": "bosnia-herzegovina",
            "Bosnia & Herzegovina": "bosnia-herzegovina",
            "Bosnia-Herzegovina": "bosnia-herzegovina",
            "Czech Republic": "czech-republic",
            "Czechia": "czech-republic",
            "Ivory Coast": "ivory-coast",
            "Côte d'Ivoire": "ivory-coast",
            "DR Congo": "dr-congo",
            "Congo DR": "dr-congo",
            "South Korea": "south-korea",
            "Korea Republic": "south-korea",
            "Curacao": "curacao",
            "Curaçao": "curacao",
        }
        value = aliases.get(team, team)
        return re.sub(r"[^A-Za-z0-9]+", "-", value).strip("-").lower()
