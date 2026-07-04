import pandas as pd


class MarketAgent:
    """Input: one match row. Output: market probabilities, pick, confidence, and margin."""

    MARKET_COLUMNS = ["market_H", "market_D", "market_A"]

    def run(self, row: pd.Series) -> dict:
        probabilities = {
            "H": float(row["market_H"]),
            "D": float(row["market_D"]),
            "A": float(row["market_A"]),
        }
        total = sum(probabilities.values())
        if total <= 0:
            raise ValueError("market probabilities must sum to a positive value.")

        probabilities = {key: value / total for key, value in probabilities.items()}
        ranked = sorted(probabilities.items(), key=lambda item: item[1], reverse=True)
        return {
            "market_H": probabilities["H"],
            "market_D": probabilities["D"],
            "market_A": probabilities["A"],
            "market_pick": ranked[0][0],
            "market_confidence": ranked[0][1],
            "market_margin": ranked[0][1] - ranked[1][1],
        }
