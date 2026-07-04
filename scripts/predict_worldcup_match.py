import argparse
from pathlib import Path

import pandas as pd


DEFAULT_FEATURES_PATH = Path("data/processed/worldcup_features.csv")


def load_features(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"World Cup features not found: {path}")
    data = pd.read_csv(path, encoding="utf-8")
    data["date"] = pd.to_datetime(data["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    return data


def select_match(data: pd.DataFrame, home_team: str, away_team: str, date: str) -> pd.Series:
    matches = data[
        (data["home_team"].str.lower() == home_team.lower())
        & (data["away_team"].str.lower() == away_team.lower())
        & (data["date"] == date)
    ]
    if matches.empty:
        raise ValueError(f"找不到比賽：{date} {home_team} vs {away_team}")
    if len(matches) > 1:
        raise ValueError("找到多筆符合比賽，請檢查 World Cup features 是否重複。")
    return matches.iloc[0]


def normalize_market(row: pd.Series) -> dict[str, float]:
    values = {
        "H": pd.to_numeric(row.get("market_H"), errors="coerce"),
        "D": pd.to_numeric(row.get("market_D"), errors="coerce"),
        "A": pd.to_numeric(row.get("market_A"), errors="coerce"),
    }
    if any(pd.isna(value) for value in values.values()):
        raise ValueError("此比賽缺少 market_H/market_D/market_A，暫時無法輸出 H/D/A probability。")
    total = sum(values.values())
    if total <= 0:
        raise ValueError("H/D/A probability 加總無效。")
    return {key: float(value / total) for key, value in values.items()}


def scoreline_grid(home_xg: float, away_xg: float, max_goals: int = 5) -> pd.DataFrame:
    import math

    rows = []
    for home_goals in range(max_goals + 1):
        for away_goals in range(max_goals + 1):
            home_prob = math.exp(-home_xg) * home_xg**home_goals / math.factorial(home_goals)
            away_prob = math.exp(-away_xg) * away_xg**away_goals / math.factorial(away_goals)
            rows.append({"scoreline": f"{home_goals}-{away_goals}", "probability": home_prob * away_prob})
    grid = pd.DataFrame(rows)
    grid["probability"] = grid["probability"] / grid["probability"].sum()
    return grid.sort_values("probability", ascending=False).reset_index(drop=True)


def confidence_level(max_probability: float) -> str:
    if max_probability >= 0.60:
        return "高"
    if max_probability >= 0.48:
        return "中"
    return "低"


def upset_risk(row: pd.Series, probabilities: dict[str, float]) -> tuple[str, list[str]]:
    reasons = []
    risk_score = pd.to_numeric(row.get("risk_score"), errors="coerce")
    if pd.notna(risk_score) and risk_score >= 2:
        reasons.append("風險分數偏高")

    poisson_diff = pd.to_numeric(row.get("poisson_diff"), errors="coerce")
    if pd.notna(poisson_diff) and abs(poisson_diff) < 0.35:
        reasons.append("Poisson 期望進球差距接近")

    margin = sorted(probabilities.values(), reverse=True)[0] - sorted(probabilities.values(), reverse=True)[1]
    if margin < 0.08:
        reasons.append("勝和負機率接近，屬於五五波")

    if bool(row.get("must_win_home")) or bool(row.get("must_win_away")):
        reasons.append("小組形勢有必勝壓力")

    if len(reasons) >= 2:
        return "高", reasons
    if reasons:
        return "中", reasons
    return "低", ["市場與模型訊號相對一致"]


def build_report(row: pd.Series) -> str:
    probabilities = normalize_market(row)
    best_pick = max(probabilities, key=probabilities.get)
    max_probability = probabilities[best_pick]
    confidence = confidence_level(max_probability)

    home_xg = pd.to_numeric(row.get("poisson_home_xg"), errors="coerce")
    away_xg = pd.to_numeric(row.get("poisson_away_xg"), errors="coerce")
    scorelines = None
    if pd.notna(home_xg) and pd.notna(away_xg):
        scorelines = scoreline_grid(float(home_xg), float(away_xg)).head(5)

    risk, reasons = upset_risk(row, probabilities)

    lines = [
        "World Cup 賽前預測報告",
        f"日期: {row['date']}",
        f"小組: {row['group']}",
        f"對戰: {row['home_team']} vs {row['away_team']}",
        f"中立場地: {row['neutral_venue']}",
        "",
        "勝和負機率",
        f"主勝 H: {probabilities['H']:.4f}",
        f"和局 D: {probabilities['D']:.4f}",
        f"客勝 A: {probabilities['A']:.4f}",
        f"模型傾向: {best_pick}",
        f"信心等級: {confidence}",
        "",
        "小組形勢",
        f"主隊賽前積分/淨勝球: {row['points_home_before']} / {row['goal_diff_home_before']}",
        f"客隊賽前積分/淨勝球: {row['points_away_before']} / {row['goal_diff_away_before']}",
        f"主隊必勝壓力: {row['must_win_home']}",
        f"客隊必勝壓力: {row['must_win_away']}",
        f"主隊已大致晉級: {row['already_qualified_home']}",
        f"客隊已大致晉級: {row['already_qualified_away']}",
        "",
        f"爆冷/平局風險: {risk}",
        "原因: " + "；".join(reasons),
    ]

    if scorelines is not None:
        lines.extend(["", "最可能比分"])
        for scoreline in scorelines.to_dict("records"):
            lines.append(f"{scoreline['scoreline']}: {scoreline['probability']:.4f}")

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--home-team", required=True)
    parser.add_argument("--away-team", required=True)
    parser.add_argument("--date", required=True)
    parser.add_argument("--features", type=Path, default=DEFAULT_FEATURES_PATH)
    args = parser.parse_args()

    data = load_features(args.features)
    row = select_match(data, args.home_team, args.away_team, args.date)
    print(build_report(row))


if __name__ == "__main__":
    main()
