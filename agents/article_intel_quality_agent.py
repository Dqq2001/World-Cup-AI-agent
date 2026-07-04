from __future__ import annotations

from datetime import datetime, timezone
import re

import pandas as pd


class ArticleIntelQualityAgent:
    """Scores article intelligence quality across multiple football article sources."""

    CONTENT_COLUMNS = [
        "injuries_home",
        "injuries_away",
        "suspensions_home",
        "suspensions_away",
        "expected_lineup_home",
        "expected_lineup_away",
    ]
    HIGH_WEIGHT_SOURCES = {"The Analyst", "ESPN Soccer", "BBC Sport Football"}

    def evaluate(self, rows: list[dict], selected_row: dict | None) -> dict:
        source_rows = [row for row in rows if str(row.get("source_name", "")).lower() != "search"]
        content_rows = [row for row in source_rows if self._to_bool(row.get("intel_has_content"))]
        base_confidence = float((selected_row or {}).get("confidence", 0))
        final_confidence = base_confidence
        reasons = []

        if len(content_rows) == 1:
            reasons.append("single-source article intel")
        elif len(content_rows) >= 2:
            reasons.append(f"{len(content_rows)} article sources have content")

        agreement = self._agreement(content_rows)
        conflict = self._conflict(content_rows)
        if agreement:
            final_confidence += 0.05
            reasons.append("cross-source agreement found")
        if conflict:
            final_confidence -= 0.15
            reasons.append("conflicting article evidence found")

        if self._is_stale((selected_row or {}).get("fetched_at")):
            final_confidence -= 0.10
            reasons.append("selected article intel is older than 24 hours")

        selected_source = str((selected_row or {}).get("source_name", "unknown"))
        if selected_source in self.HIGH_WEIGHT_SOURCES:
            reasons.append(f"high-weight source: {selected_source}")

        return {
            "article_source_count": len(source_rows),
            "article_content_count": len(content_rows),
            "cross_source_agreement": agreement,
            "conflict_detected": conflict,
            "final_article_confidence": round(max(0.0, min(0.99, final_confidence)), 3),
            "quality_reason": "; ".join(reasons) if reasons else "no usable article intel",
        }

    def _agreement(self, rows: list[dict]) -> bool:
        for column in self.CONTENT_COLUMNS:
            values = [self._normalize(row.get(column)) for row in rows]
            values = [value for value in values if value]
            if len(values) >= 2 and len(set(values)) < len(values):
                return True
        return False

    def _conflict(self, rows: list[dict]) -> bool:
        if len(rows) < 2:
            return False
        for column in self.CONTENT_COLUMNS:
            values = [self._normalize(row.get(column)) for row in rows]
            unique_values = {value for value in values if value}
            if len(unique_values) >= 2:
                return True
        return False

    def _normalize(self, value) -> str:
        text = str(value or "").strip().lower()
        if text in {"", "unknown", "nan", "none", "<na>"}:
            return ""
        text = re.sub(r"\s+", " ", text)
        return text

    def _is_stale(self, fetched_at) -> bool:
        timestamp = pd.to_datetime(fetched_at, utc=True, errors="coerce")
        if pd.isna(timestamp):
            return False
        return (datetime.now(timezone.utc) - timestamp.to_pydatetime()).total_seconds() > 24 * 3600

    def _to_bool(self, value) -> bool:
        if isinstance(value, bool):
            return value
        return str(value).strip().lower() in {"true", "1", "yes"}
