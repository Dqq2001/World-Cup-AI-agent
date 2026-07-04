from pathlib import Path

import pandas as pd


class ReportAgent:
    """Input: prediction rows. Output: CSV and one text report per match."""

    DRAW_RISK_REFERENCE = "Draw risk historical reference: LOW draw rate 20.07%, MEDIUM draw rate 28.89%, HIGH draw rate 30.50%."
    LEAGUE_REFERENCE_NOTE = "League reference uses five-league patterns only as auxiliary risk context; domain shift means it must not override international models."

    def run(self, predictions: list[dict], output_csv: Path, report_dir: Path) -> pd.DataFrame:
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        report_dir.mkdir(parents=True, exist_ok=True)
        data = pd.DataFrame(predictions)
        data.to_csv(output_csv, index=False, encoding="utf-8")
        for index, row in data.iterrows():
            report_path = report_dir / f"{index:03d}_{row['home_team']}_vs_{row['away_team']}.txt"
            report_path.write_text(self.render_report(row), encoding="utf-8")
        return data

    def render_report(self, row: pd.Series) -> str:
        return "\n".join(
            [
                "World Cup Betting Assistant 賽前報告",
                f"日期: {row['date']}",
                f"小組: {row['group']}",
                f"對戰: {row['home_team']} vs {row['away_team']}",
                f"Market H/D/A: {self._format_optional(row.get('market_H'))} / {self._format_optional(row.get('market_D'))} / {self._format_optional(row.get('market_A'))}",
                f"Model H/D/A: {row['model_H']:.4f} / {row['model_D']:.4f} / {row['model_A']:.4f}",
                f"Poisson top scores: {row['poisson_top_scores']}",
                f"Upset risk: {row['upset_risk']}",
                f"Draw risk: {row.get('draw_risk_level', 'NA')} score={row.get('draw_risk_score', 'NA')}",
                f"Draw risk reasons: {row.get('draw_risk_reasons', '')}",
                self.DRAW_RISK_REFERENCE,
                f"League reference: {row.get('league_reference_level', 'NA')} score={row.get('league_risk_score', 'NA')}",
                f"League reference reasons: {row.get('league_reference_reasons', '')}",
                self.LEAGUE_REFERENCE_NOTE,
                f"Value side: {row['value_side']}",
                f"Edge: {self._format_optional(row.get('edge'))}",
                f"Recommended action: {row['recommended_action']}",
                f"Recommended stake: {row['recommended_stake']}",
                f"Reason: {row['reason']}",
            ]
        )

    def _format_optional(self, value) -> str:
        if pd.isna(value):
            return "NA"
        return f"{float(value):.4f}"
