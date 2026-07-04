from __future__ import annotations

from pathlib import Path

import pandas as pd


def read_csv_safe(path: str | Path, **kwargs) -> pd.DataFrame:
    csv_path = Path(path)
    if not csv_path.exists():
        return pd.DataFrame()
    return pd.read_csv(csv_path, encoding=kwargs.pop("encoding", "utf-8"), **kwargs)


def write_csv_atomic(data: pd.DataFrame, path: str | Path, **kwargs) -> Path:
    csv_path = Path(path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = csv_path.with_suffix(csv_path.suffix + ".tmp")
    data.to_csv(temp_path, index=False, encoding=kwargs.pop("encoding", "utf-8"), **kwargs)
    temp_path.replace(csv_path)
    return csv_path


def append_dedup(data: pd.DataFrame, path: str | Path, key_cols: list[str]) -> pd.DataFrame:
    csv_path = Path(path)
    existing = read_csv_safe(csv_path)
    combined = pd.concat([existing, data], ignore_index=True) if not existing.empty else data.copy()
    if key_cols:
        combined = combined.drop_duplicates(subset=key_cols, keep="last")
    write_csv_atomic(combined, csv_path)
    return combined

