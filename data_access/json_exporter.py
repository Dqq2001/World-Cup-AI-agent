from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def export_dataframe_json(data: pd.DataFrame, path: str | Path) -> Path:
    json_path = Path(path)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    records = data.fillna("").to_dict(orient="records")
    json_path.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    return json_path


def export_csv_to_json(csv_path: str | Path, json_path: str | Path) -> Path:
    data = pd.read_csv(csv_path, encoding="utf-8")
    return export_dataframe_json(data, json_path)

