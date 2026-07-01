"""Tests for the table loader — accept CSV and Excel uploads uniformly."""
import io

import numpy as np
import pandas as pd

from anomaly_detector import loader


def _frame():
    return pd.DataFrame({
        "time": pd.date_range("2020-01-01", periods=5, freq="D").astype(str),
        "value": np.arange(5.0),
    })


def test_reads_csv(tmp_path):
    p = tmp_path / "d.csv"
    _frame().to_csv(p, index=False)
    df = loader.read_table(p)
    assert list(df.columns) == ["time", "value"] and len(df) == 5


def test_reads_excel_by_extension(tmp_path):
    p = tmp_path / "d.xlsx"
    _frame().to_excel(p, index=False)
    df = loader.read_table(p)
    assert list(df.columns) == ["time", "value"] and len(df) == 5


def test_reads_excel_from_stream_using_filename():
    buf = io.BytesIO()
    _frame().to_excel(buf, index=False)
    buf.seek(0)
    df = loader.read_table(buf, name="upload.xlsx")
    assert len(df) == 5
