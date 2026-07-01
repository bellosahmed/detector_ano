"""Tests for the dataset profiler — the layer that makes 'any dataset' work.

The profiler inspects an arbitrary dataframe and infers: the time column, the
sampling frequency, the numeric signal columns, and missing-value sentinels.
"""
import numpy as np
import pandas as pd

from anomaly_detector.profile import profile


def _hourly_frame(n=48):
    idx = pd.date_range("2020-01-01", periods=n, freq="h")
    return pd.DataFrame({
        "timestamp": idx.astype(str),          # a date-like string column
        "temperature": np.linspace(0, 10, n),  # a numeric signal
    })


def test_detects_the_time_column():
    df = _hourly_frame()
    result = profile(df)
    assert result.time_col == "timestamp"


def test_detects_numeric_signals_excluding_time():
    df = _hourly_frame()
    df["humidity"] = np.linspace(30, 50, len(df))
    result = profile(df)
    assert set(result.signals) == {"temperature", "humidity"}
    assert "timestamp" not in result.signals


def test_single_signal_is_not_multivariate():
    result = profile(_hourly_frame())
    assert result.is_multivariate is False


def test_multiple_signals_is_multivariate():
    df = _hourly_frame()
    df["pressure"] = np.linspace(1000, 1010, len(df))
    result = profile(df)
    assert result.is_multivariate is True


def test_infers_hourly_frequency():
    result = profile(_hourly_frame())
    assert result.freq is not None and result.freq.upper().startswith("H")


def test_infers_daily_frequency():
    n = 30
    df = pd.DataFrame({
        "date": pd.date_range("2020-01-01", periods=n, freq="D").astype(str),
        "value": np.arange(n, dtype=float),
    })
    result = profile(df)
    assert result.freq is not None and result.freq.upper().startswith("D")


def test_detects_sentinel_value():
    df = _hourly_frame(n=100)
    # inject a classic -9999 missing-marker several times
    df.loc[[10, 20, 30, 40], "temperature"] = -9999.0
    result = profile(df)
    assert -9999.0 in result.sentinels


def test_picks_monotonic_time_column_when_several_date_columns_exist():
    n = 48
    index_time = pd.date_range("2021-06-01", periods=n, freq="h")
    # a second date-like column that is NOT the index (shuffled birthdays)
    rng = np.random.default_rng(0)
    birthdays = pd.to_datetime("1990-01-01") + pd.to_timedelta(
        rng.integers(0, 10000, n), unit="D"
    )
    df = pd.DataFrame({
        "signup_date": birthdays.astype(str),
        "event_time": index_time.astype(str),
        "value": np.linspace(0, 1, n),
    })
    result = profile(df)
    assert result.time_col == "event_time"


def test_infers_frequency_even_when_rows_are_shuffled():
    df = _hourly_frame(n=48).sample(frac=1, random_state=1).reset_index(drop=True)
    result = profile(df)
    assert result.freq is not None and result.freq.upper().startswith("H")


def test_detection_is_by_content_not_column_name():
    # generic, meaningless column names — must still work
    df = _hourly_frame().rename(columns={"timestamp": "a", "temperature": "b"})
    result = profile(df)
    assert result.time_col == "a"
    assert result.signals == ["b"]


def test_time_name_breaks_ties_toward_the_obvious_column():
    n = 60
    # two equally-valid monotonic date columns; the one literally named "timestamp" should win
    a = pd.date_range("2020-01-01", periods=n, freq="h").astype(str)
    b = pd.date_range("2000-01-01", periods=n, freq="D").astype(str)
    df = pd.DataFrame({"created_at_note": a, "timestamp": b, "value": np.arange(n, dtype=float)})
    assert profile(df).time_col == "timestamp"


def test_profile_honours_an_explicit_time_column_override():
    df = _hourly_frame(n=60)
    df["other_time"] = pd.date_range("1990-01-01", periods=60, freq="D").astype(str)
    result = profile(df, time_col="other_time")   # user overrides the auto-detected one
    assert result.time_col == "other_time"
    assert "other_time" not in result.signals
