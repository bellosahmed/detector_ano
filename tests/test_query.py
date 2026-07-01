"""Tests for the query engine — the truth layer.

Every number it returns must equal an independent pandas computation exactly.
This is the guarantee that the system's figures are never invented.
"""
import numpy as np
import pandas as pd

from anomaly_detector import query


def _frame(n=200, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2020-01-01", periods=n, freq="h")
    return pd.DataFrame({
        "time": idx,
        "value": rng.normal(20, 5, n),
    })


def test_maximum_matches_pandas_exactly():
    df = _frame()
    assert query.maximum(df, "value") == df["value"].max()


def test_minimum_matches_pandas_exactly():
    df = _frame()
    assert query.minimum(df, "value") == df["value"].min()


def test_mean_matches_pandas_exactly():
    df = _frame()
    assert query.mean(df, "value") == df["value"].mean()


def test_when_max_returns_timestamp_of_the_peak():
    df = _frame()
    assert query.when_extreme(df, "time", "value", "max") == df.loc[df["value"].idxmax(), "time"]


def test_count_over_threshold_matches_pandas():
    df = _frame()
    assert query.count_over(df, "value", 25.0) == int((df["value"] > 25.0).sum())


def test_count_under_threshold_matches_pandas():
    df = _frame()
    assert query.count_under(df, "value", 15.0) == int((df["value"] < 15.0).sum())


def test_correlation_matches_pandas():
    rng = np.random.default_rng(0)
    a = rng.normal(0, 1, 300)
    df = pd.DataFrame({"a": a, "b": 2 * a + rng.normal(0, 0.1, 300), "c": rng.normal(0, 1, 300)})
    assert abs(query.correlation(df, "a", "b") - df["a"].corr(df["b"])) < 1e-9
    top = query.top_correlations(df, ["a", "b", "c"])
    assert top[0][:2] == ("a", "b")            # a & b are the most correlated pair


def test_missing_count_includes_nan_and_sentinels():
    df = pd.DataFrame({"v": [1.0, np.nan, -9999.0, 4.0, np.nan, -9999.0]})
    assert query.missing_count(df, "v", sentinels=[-9999.0]) == 4   # 2 NaN + 2 sentinels


def test_skewness_matches_pandas():
    df = pd.DataFrame({"v": [1.0, 1, 1, 1, 2, 3, 10]})              # right-skewed
    assert abs(query.skewness(df, "v") - df["v"].skew()) < 1e-9
    assert query.skewness(df, "v") > 0


def test_trend_slope_is_positive_for_rising_series():
    df = pd.DataFrame({"value": np.arange(100.0)})
    assert query.trend(df, "value") > 0


def test_trend_slope_is_negative_for_falling_series():
    df = pd.DataFrame({"value": np.arange(100.0)[::-1]})
    assert query.trend(df, "value") < 0
