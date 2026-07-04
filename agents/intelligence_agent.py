from pathlib import Path

import pandas as pd


class IntelligenceAgent:
    """Collects pre-match intelligence from manual CSV, OpenAI web search, and safe no-data fallback."""

    SIDE_COLUMNS = [
        "team_news_home",
        "team_news_away",
        "injuries_home",
        "injuries_away",
        "suspensions_home",
        "suspensions_away",
        "expected_lineup_home",
        "expected_lineup_away",
        "coach_comments_home",
        "coach_comments_away",
    ]
    META_COLUMNS = [
        "source_name",
        "source_url",
        "source_status",
        "intel_has_content",
        "intel_confidence_level",
        "confidence",
        "fetched_at",
        "article_source_count",
        "article_content_count",
        "cross_source_agreement",
        "conflict_detected",
        "final_article_confidence",
        "quality_reason",
    ]
    INTEL_COLUMNS = [*SIDE_COLUMNS, *META_COLUMNS]
    KEY_COLUMNS = ["date", "home_team", "away_team"]

    def run(
        self,
        fixtures_path: Path,
        manual_intel_path: Path,
        openai_intel_path: Path | None,
        article_intel_path: Path | None,
        structured_intel_path: Path | None,
        news_intel_path: Path | None,
        predictions_path: Path | None,
        odds_path: Path | None,
        as_of_date: str,
        days_ahead: int,
    ) -> dict:
        fixtures = self._load_fixtures(fixtures_path)
        upcoming = self._upcoming_matches(fixtures, as_of_date, days_ahead)
        manual = self._load_intel_source(manual_intel_path, "manual")
        openai = self._load_intel_source(openai_intel_path, "openai") if openai_intel_path else self._empty_intel()
        predictions = self._load_predictions(predictions_path)
        odds_status = self._odds_status(odds_path)

        fixture_columns = [column for column in upcoming.columns if column not in self.INTEL_COLUMNS and column != "source_type"]
        output = upcoming[fixture_columns].merge(manual, on=self.KEY_COLUMNS, how="left")
        output = output.merge(openai, on=self.KEY_COLUMNS, how="left", suffixes=("_manual", "_openai"))
        output = self._coalesce_intel(output)
        output = self._normalize_prediction_keys(output)
        predictions = self._normalize_prediction_keys(predictions)
        output = output.merge(predictions, on=["date", "group", "home_team", "away_team"], how="left")
        output = self._add_fixture_context(output)
        output["odds_status"] = odds_status
        output["web_api_status"] = output["source_type"].where(output["source_type"] != "unknown", "no-data fallback")
        output["source_status"] = output["source_status"].where(output["source_status"] != "unknown", "not_run")
        output["intel_has_content"] = output["intel_has_content"].map(self._to_bool)
        output["intel_confidence_level"] = output["intel_confidence_level"].where(output["intel_has_content"], "LOW")
        risk_details = output.apply(self._intel_risk_details, axis=1, result_type="expand")
        output["intel_risk"] = risk_details["intel_risk"]
        output["intel_risk_score"] = risk_details["intel_risk_score"]
        output["intel_risk_reason"] = risk_details["intel_risk_reason"]
        output["recommended_action"] = output.apply(self._recommended_action, axis=1)
        output["reason"] = output.apply(self._reason, axis=1)
        missing_report = self._missing_report(output, manual_intel_path, news_intel_path, odds_path)
        return {"data": output, "missing_report": missing_report}

    def _load_fixtures(self, path: Path) -> pd.DataFrame:
        if not path.exists():
            raise FileNotFoundError(f"Missing World Cup fixtures: {path}")
        fixtures = pd.read_csv(path, encoding="utf-8")
        required = ["date", "group", "home_team", "away_team"]
        missing = [column for column in required if column not in fixtures.columns]
        if missing:
            raise ValueError(f"World Cup fixtures missing columns: {missing}")
        fixtures = fixtures.copy()
        fixtures["date"] = pd.to_datetime(fixtures["date"], errors="coerce").dt.strftime("%Y-%m-%d")
        if "neutral_venue" not in fixtures.columns:
            fixtures["neutral_venue"] = True
        fixtures = self._normalize_prediction_keys(fixtures)
        return fixtures.dropna(subset=["date"])

    def _upcoming_matches(self, fixtures: pd.DataFrame, as_of_date: str, days_ahead: int) -> pd.DataFrame:
        as_of = pd.to_datetime(as_of_date)
        end_date = as_of + pd.Timedelta(days=days_ahead)
        fixture_dates = pd.to_datetime(fixtures["date"], errors="coerce")
        mask = (fixture_dates >= as_of) & (fixture_dates <= end_date)
        return fixtures.loc[mask].sort_values(["date", "group", "home_team", "away_team"]).reset_index(drop=True)

    def _empty_intel(self) -> pd.DataFrame:
        return pd.DataFrame(columns=[*self.KEY_COLUMNS, *self.INTEL_COLUMNS, "source_type"])

    def _load_intel_source(self, path: Path | None, source_type: str) -> pd.DataFrame:
        if path is None or not path.exists():
            return self._empty_intel()
        data = pd.read_csv(path, encoding="utf-8")
        missing = [column for column in self.KEY_COLUMNS if column not in data.columns]
        if missing:
            raise ValueError(f"{source_type} intelligence CSV missing columns: {missing}")
        data = data.copy()
        data["date"] = pd.to_datetime(data["date"], errors="coerce").dt.strftime("%Y-%m-%d")
        for column in self.INTEL_COLUMNS:
            if column not in data.columns:
                data[column] = pd.NA
        data["source_type"] = source_type
        default_status = source_type if source_type == "manual" else "not_run"
        if source_type == "openai":
            default_status = "openai_web_search"
        data["source_status"] = data["source_status"].fillna(default_status)
        if source_type == "openai":
            data["source_name"] = data["source_name"].fillna("OpenAI web search")
            if "source_urls" in data.columns:
                data["source_url"] = data["source_url"].fillna(data["source_urls"])
        content_available = data[self.SIDE_COLUMNS].apply(lambda row: any(str(value).strip().lower() not in {"", "unknown", "nan", "none", "<na>"} for value in row), axis=1)
        data["intel_has_content"] = data["intel_has_content"].fillna(content_available)
        if "confidence" in data.columns:
            numeric_confidence = pd.to_numeric(data["confidence"], errors="coerce")
            data["intel_confidence_level"] = data["intel_confidence_level"].fillna(
                numeric_confidence.apply(lambda value: "HIGH" if pd.notna(value) and value >= 0.8 else "MEDIUM" if pd.notna(value) and value >= 0.5 else "LOW")
            )
        if "source_type" in data.columns and source_type == "search":
            data["source_type"] = data["source_type"].fillna("search")
        return data[[*self.KEY_COLUMNS, *self.INTEL_COLUMNS, "source_type"]].drop_duplicates(self.KEY_COLUMNS, keep="first")

    def _coalesce_intel(self, data: pd.DataFrame) -> pd.DataFrame:
        output = data.copy()
        for column in self.INTEL_COLUMNS:
            manual_column = f"{column}_manual"
            openai_column = f"{column}_openai"
            empty = pd.Series(pd.NA, index=output.index)
            manual_values = output[manual_column] if manual_column in output.columns else empty
            openai_values = output[openai_column] if openai_column in output.columns else empty
            openai_valid = self._valid_openai_mask(output)
            openai_values = openai_values.where(openai_valid)
            output[column] = manual_values.combine_first(openai_values)
            output[column] = output[column].fillna("unknown")

        output["source_type"] = "unknown"
        manual_source = output.get("source_type_manual")
        openai_source = output.get("source_type_openai")
        if manual_source is not None:
            output.loc[manual_source.notna(), "source_type"] = "manual"
        if openai_source is not None:
            output.loc[(output["source_type"] == "unknown") & openai_source.notna() & self._valid_openai_mask(output), "source_type"] = "openai"
        drop_columns = [
            column
            for column in output.columns
            if column.endswith("_manual")
            or column.endswith("_openai")
            or column in {"source_type_manual", "source_type_openai"}
        ]
        return output.drop(columns=drop_columns)

    def _valid_openai_mask(self, data: pd.DataFrame) -> pd.Series:
        status = data.get("source_status_openai", pd.Series("", index=data.index)).astype(str).str.lower()
        has_content = data.get("intel_has_content_openai", pd.Series(False, index=data.index)).map(self._to_bool)
        return (status == "ok") & has_content

    def _load_predictions(self, path: Path | None) -> pd.DataFrame:
        columns = [
            "date",
            "group",
            "home_team",
            "away_team",
            "model_H",
            "model_D",
            "model_A",
            "poisson_top_scores",
            "upset_risk",
            "draw_risk_level",
            "league_reference_level",
            "league_risk_score",
            "league_reference_reasons",
            "market_H",
            "market_D",
            "market_A",
            "value_side",
            "edge",
            "recommended_action",
        ]
        if path is None or not path.exists():
            return pd.DataFrame(columns=columns)
        predictions = pd.read_csv(path, encoding="utf-8")
        for column in columns:
            if column not in predictions.columns:
                predictions[column] = pd.NA
        predictions = self._normalize_prediction_keys(predictions)
        return predictions[columns].drop_duplicates(["date", "group", "home_team", "away_team"], keep="first")

    def _normalize_prediction_keys(self, data: pd.DataFrame) -> pd.DataFrame:
        data = data.copy()
        if "date" in data.columns:
            data["date"] = pd.to_datetime(data["date"], errors="coerce").dt.strftime("%Y-%m-%d")
        for column in ["group", "home_team", "away_team"]:
            if column not in data.columns:
                data[column] = ""
            data[column] = data[column].fillna("").astype(str).str.strip()
        return data

    def _odds_status(self, odds_path: Path | None) -> str:
        if odds_path is None or not odds_path.exists():
            return "missing"
        try:
            odds = pd.read_csv(odds_path, encoding="utf-8")
        except pd.errors.EmptyDataError:
            return "missing"
        if odds.empty:
            return "missing"
        return "manual" if "bookmaker" in odds.columns else "api"

    def _add_fixture_context(self, data: pd.DataFrame) -> pd.DataFrame:
        data = data.copy()
        data["match_date"] = pd.to_datetime(data["date"], errors="coerce")
        data["group_matchday"] = data.groupby("group").cumcount().floordiv(2) + 1
        sorted_data = data.sort_values("match_date").copy()
        last_played: dict[str, pd.Timestamp] = {}
        rest_values: dict[int, tuple[object, object]] = {}
        for index, row in sorted_data.iterrows():
            home_previous = last_played.get(row["home_team"])
            away_previous = last_played.get(row["away_team"])
            rest_values[index] = (
                pd.NA if home_previous is None else (row["match_date"] - home_previous).days,
                pd.NA if away_previous is None else (row["match_date"] - away_previous).days,
            )
            last_played[row["home_team"]] = row["match_date"]
            last_played[row["away_team"]] = row["match_date"]
        data["rest_days_home"] = [rest_values[index][0] for index in data.index]
        data["rest_days_away"] = [rest_values[index][1] for index in data.index]
        return data.drop(columns=["match_date"])

    def _intel_risk_details(self, row: pd.Series) -> dict:
        source_status = str(row.get("source_status", "")).lower()
        if not self._to_bool(row.get("intel_has_content", False)):
            return {
                "intel_risk": "UNKNOWN",
                "intel_risk_score": 0,
                "intel_risk_reason": "no usable intel",
            }
        if source_status in {"rate_limited", "no_search_results", "not_run", "blocked", "source_blocked", "parser_failed"}:
            return {
                "intel_risk": "UNKNOWN",
                "intel_risk_score": 0,
                "intel_risk_reason": "source unavailable",
            }

        risk_score = 0
        reasons = []
        for column in ["injuries_home", "injuries_away", "suspensions_home", "suspensions_away"]:
            points, reason = self._availability_risk_points(row.get(column, ""))
            risk_score += points
            if reason:
                reasons.append(f"{column}: {reason}")

        if risk_score >= 3:
            level = "HIGH"
        elif risk_score >= 1:
            level = "MEDIUM"
        else:
            level = "LOW"

        return {
            "intel_risk": level,
            "intel_risk_score": risk_score,
            "intel_risk_reason": "; ".join(reasons) if reasons else "no confirmed negative team news",
        }

    def _availability_risk_points(self, value) -> tuple[int, str]:
        text = str(value).strip().lower()
        if text in {"", "unknown", "nan", "none", "<na>"}:
            return 0, ""
        no_issue_terms = [
            "no reported",
            "no clear",
            "no suspensions",
            "no suspension",
            "no injuries",
            "out: none",
            "doubtful: none",
            "currently reported injured or suspended",
        ]
        positive_terms = [
            "full squad",
            "available",
            "returned to training",
            "fit to start",
            "basically fine",
        ]
        neutral_phrases = [
            "out: none",
            "doubtful: none",
            "no reported injuries",
            "no reported suspensions",
            "no australia suspensions clearly reported",
            "no further australia injury concerns",
            "currently reported injured or suspended",
        ]
        risk_text = text
        for phrase in neutral_phrases:
            risk_text = risk_text.replace(phrase, "")
        confirmed_terms = [
            "ruled out",
            "will miss",
            "has left the squad",
            "have left the squad",
            "is suspended",
            "are suspended",
            "suspended after",
            "suspended for",
            "hamstring tear",
            "acl",
            "fracture",
            "still out",
            "reported out",
            "out of the round",
            "out of the match",
        ]
        medium_terms = [
            "doubtful",
            "fitness doubt",
            "muscle issue",
            "hamstring fatigue",
            "hamstring strain",
            "injury doubt",
            "uncertain",
            "minor knock",
            "late fitness test",
            "not risk him unless fully fit",
        ]
        if any(term in risk_text for term in confirmed_terms):
            return 2, "confirmed major absence or suspension"
        if any(term in risk_text for term in medium_terms):
            return 1, "availability doubt"
        if any(term in text for term in no_issue_terms) or any(term in text for term in positive_terms):
            return 0, "positive or no confirmed issue"
        return 0, ""

    def _to_bool(self, value) -> bool:
        if isinstance(value, bool):
            return value
        return str(value).strip().lower() in {"true", "1", "yes"}

    def _recommended_action(self, row: pd.Series) -> str:
        action = row.get("recommended_action")
        if pd.isna(action) or action not in {"BET", "SMALL_BET", "WATCH", "PASS"}:
            action = "WATCH"
        if row["odds_status"] == "missing":
            return "PASS" if row["intel_risk"] == "HIGH" else "WATCH"
        if row["intel_risk"] == "HIGH" and action in {"BET", "SMALL_BET"}:
            return "WATCH"
        return action

    def _reason(self, row: pd.Series) -> str:
        parts = []
        if row["odds_status"] == "missing":
            parts.append("No odds available; only WATCH / PASS is allowed.")
        if row["intel_risk"] == "HIGH":
            parts.append("Major injury, suspension, or lineup uncertainty detected; action downgraded.")
        if row["intel_risk"] == "UNKNOWN":
            parts.append("Intel source unavailable or no usable intel; no intel-risk downgrade applied.")
        if row["source_type"] == "unknown":
            parts.append("No manual or OpenAI web_search intel is available; pre-match intel remains unknown.")
        return " ".join(parts) if parts else "Pre-match intel is available; no intel downgrade triggered."

    def _missing_report(self, data: pd.DataFrame, manual_path: Path, news_path: Path | None, odds_path: Path | None) -> pd.DataFrame:
        rows = []
        if not manual_path.exists():
            rows.append(
                {
                    "data_type": "manual_intelligence",
                    "path": str(manual_path),
                    "missing_count": len(data),
                    "message": "Manual intelligence CSV not found; manual fields set to unknown unless search source has data.",
                }
            )
        if odds_path is None or not odds_path.exists():
            rows.append(
                {
                    "data_type": "odds",
                    "path": "" if odds_path is None else str(odds_path),
                    "missing_count": len(data),
                    "message": "Odds file not found; only WATCH / PASS allowed.",
                }
            )
        for column in self.SIDE_COLUMNS:
            unknown_count = int((data[column] == "unknown").sum())
            if unknown_count:
                rows.append(
                    {
                        "data_type": column,
                        "path": str(manual_path),
                        "missing_count": unknown_count,
                        "message": f"{column} unknown for {unknown_count} upcoming matches.",
                    }
                )
        return pd.DataFrame(rows)

