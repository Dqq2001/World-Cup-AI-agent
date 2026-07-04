import pandas as pd


class ValueAgent:
    """Input: one match row plus market output. Output: value side and edge."""

    MODEL_COLUMNS = {"H": "model_H", "D": "model_D", "A": "model_A"}
    ADJUSTED_COLUMNS = {"H": "adjusted_H", "D": "adjusted_D", "A": "adjusted_A"}
    MARKET_COLUMNS = {"H": "market_H", "D": "market_D", "A": "market_A"}

    def run(self, row: pd.Series, market: dict) -> dict:
        probability_columns = self.ADJUSTED_COLUMNS if self._has_adjusted_probabilities(row) else self.MODEL_COLUMNS
        edges = {}
        for side in ["H", "D", "A"]:
            model_probability = float(row[probability_columns[side]])
            market_probability = float(market[self.MARKET_COLUMNS[side]])
            edges[side] = model_probability - market_probability

        value_side = max(edges, key=edges.get)
        edge = edges[value_side]
        if edge >= 0.08:
            value_level = "HIGH"
        elif edge >= 0.04:
            value_level = "MEDIUM"
        else:
            value_level = "LOW"

        return {
            "model_H": float(row["model_H"]),
            "model_D": float(row["model_D"]),
            "model_A": float(row["model_A"]),
            "decision_H": float(row[probability_columns["H"]]),
            "decision_D": float(row[probability_columns["D"]]),
            "decision_A": float(row[probability_columns["A"]]),
            "probability_source": "weighted_adjustment" if probability_columns == self.ADJUSTED_COLUMNS else "base_model",
            "value_side": value_side,
            "edge": edge,
            "value_level": value_level,
            "all_edges": edges,
        }

    def _has_adjusted_probabilities(self, row: pd.Series) -> bool:
        for column in self.ADJUSTED_COLUMNS.values():
            if column not in row or pd.isna(row[column]):
                return False
            try:
                float(row[column])
            except (TypeError, ValueError):
                return False
        return True
