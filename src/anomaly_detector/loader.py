"""Load a tabular dataset from CSV or Excel, uniformly.

The rest of the system works on a pandas DataFrame; this is the single place that turns an
uploaded file (a path, or a stream + filename) into one, picking the reader by extension.
"""
from __future__ import annotations

import pandas as pd

_EXCEL_SUFFIXES = (".xlsx", ".xls", ".xlsm")


def read_table(source, name: str = "") -> pd.DataFrame:
    """Read `source` (path or file-like) into a DataFrame. Excel if the name ends .xlsx/.xls."""
    label = (name or getattr(source, "name", "") or str(source)).lower()
    if label.endswith(_EXCEL_SUFFIXES):
        return pd.read_excel(source)
    return pd.read_csv(source)
