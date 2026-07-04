from pathlib import Path

import pandas as pd

from src.clean import clean_file


def test_clean_file_standardizes_football_data_columns(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    csv_path = raw_dir / "19-20.csv"
    pd.DataFrame(
        {
            "Div": ["E0"],
            "Date": ["09/08/2019"],
            "HomeTeam": ["Liverpool"],
            "AwayTeam": ["Norwich"],
            "FTHG": [4],
            "FTAG": [1],
            "FTR": ["H"],
            "HS": [15],
            "AS": [12],
            "HST": [7],
            "AST": [5],
            "HC": [11],
            "AC": [2],
            "B365H": [1.14],
            "B365D": [10.0],
            "B365A": [19.0],
        }
    ).to_csv(csv_path, index=False, encoding="utf-8")

    cleaned = clean_file(csv_path, raw_dir)

    assert cleaned.loc[0, "league"] == "Premier League"
    assert cleaned.loc[0, "season"] == "19-20"
    assert cleaned.loc[0, "home_team"] == "Liverpool"
    assert cleaned.loc[0, "result"] == "H"
    assert cleaned.loc[0, "source_file"] == "19-20.csv"
