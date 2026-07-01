"""Tests for the natural-language interface — ask a dataset questions in plain English.

Intent + which signal are parsed deterministically; every number in the answer comes
from the query engine / core (never invented).
"""
import numpy as np
import pandas as pd

from anomaly_detector import interface


def _frame(n=400, seed=0):
    rng = np.random.default_rng(seed)
    v = 20 + 5 * np.sin(2 * np.pi * np.arange(n) / 24) + rng.normal(0, 0.5, n)
    v[200] = 200.0  # a clear anomaly
    return pd.DataFrame({
        "time": pd.date_range("2020-01-01", periods=n, freq="h"),
        "temperature": v,
    })


def test_highest_question_returns_the_max_and_when():
    df = _frame()
    ans = interface.ask(df, "what was the highest temperature?")
    assert ans.value == df["temperature"].max()
    assert "highest" in ans.text.lower() and "200" in ans.text  # the spike value


def test_lowest_question_returns_the_min():
    df = _frame()
    ans = interface.ask(df, "what is the lowest temperature")
    assert ans.value == df["temperature"].min()


def test_average_question_returns_the_mean():
    df = _frame()
    ans = interface.ask(df, "what's the average temperature?")
    assert abs(ans.value - df["temperature"].mean()) < 1e-9


def test_count_above_question_matches_truth():
    df = _frame()
    ans = interface.ask(df, "how many times did temperature go above 30?")
    assert ans.value == int((df["temperature"] > 30).sum())


def test_anomaly_question_reports_count_and_an_explanation():
    df = _frame()
    ans = interface.ask(df, "were there any anomalies?")
    assert "1" in ans.text or "anomal" in ans.text.lower()
    assert "expected" in ans.text.lower()  # includes reasoning


def _two_year_frame():
    idx = pd.date_range("2019-01-01", periods=730, freq="D")
    v = np.concatenate([np.full(365, 10.0), np.full(365, 20.0)])  # 2019→10, 2020→20
    return pd.DataFrame({"time": idx, "value": v})


def test_question_scoped_to_a_year():
    ans = interface.ask(_two_year_frame(), "what was the average value in 2020?")
    assert abs(ans.value - 20.0) < 1e-6
    assert "2020" in ans.text


def test_comparison_between_two_years():
    ans = interface.ask(_two_year_frame(), "was 2020 higher than 2019?")
    assert "2020" in ans.text and "higher" in ans.text.lower()


def test_trend_detects_a_rising_series():
    ans = interface.ask(_two_year_frame(), "is the value rising over time?")
    assert any(w in ans.text.lower() for w in ("rising", "increas", "upward"))


def _daily_year_frame():
    idx = pd.date_range("2020-01-01", periods=365, freq="D")
    return pd.DataFrame({"time": idx, "value": np.arange(365.0)})


def test_scoped_to_a_named_month():
    df = _daily_year_frame()
    ans = interface.ask(df, "what was the average value in march 2020?")
    march = df[pd.to_datetime(df["time"]).dt.month == 3]["value"].mean()
    assert abs(ans.value - march) < 1e-6
    assert "march" in ans.text.lower()


def test_scoped_to_last_month():
    df = _daily_year_frame()               # ends 2020-12-30 → last month = December
    ans = interface.ask(df, "what was the highest value last month?")
    assert ans.value >= 330                # December sits at the top of the ramp


def test_explain_anomaly_on_a_specific_date():
    n = 400
    idx = pd.date_range("2023-01-01", periods=n, freq="D")
    v = 20 + 5 * np.sin(2 * np.pi * np.arange(n) / 30) + np.random.default_rng(0).normal(0, 0.3, n)
    v[151] = 200.0                                  # the YEAR'S biggest anomaly (2023-06-01)
    v[323] = 60.0                                   # the anomaly on 2023-11-20 the user asks about
    df = pd.DataFrame({"time": idx, "value": v})
    ans = interface.ask(df, "explain the anomaly on 2023-11-20")
    assert "2023-11-20" in ans.text
    assert ans.value == 60.0                        # THAT day's anomaly, not the year's top (200)


def test_month_to_month_comparison():
    idx = pd.date_range("2020-01-01", periods=180, freq="D")   # Jan–Jun 2020
    months = idx.month
    v = np.where(months == 3, 100.0, np.where(months == 4, 50.0, 20.0))  # March > April
    df = pd.DataFrame({"time": idx, "value": v})
    ans = interface.ask(df, "was march higher than april?")
    assert "march" in ans.text.lower() and "higher" in ans.text.lower()


def test_correlation_question_between_two_signals():
    rng = np.random.default_rng(0)
    a = rng.normal(0, 1, 200)
    df = pd.DataFrame({"time": pd.date_range("2020-01-01", periods=200, freq="h"),
                       "power": a, "voltage": 3 * a + rng.normal(0, 0.1, 200)})
    ans = interface.ask(df, "what is the correlation between power and voltage?")
    assert "correlation" in ans.text.lower() and ans.value > 0.9


def test_missing_data_question_reports_counts():
    df = pd.DataFrame({"time": pd.date_range("2020-01-01", periods=6, freq="h"),
                       "value": [1.0, np.nan, 3.0, np.nan, 5.0, 6.0]})
    ans = interface.ask(df, "how much data is missing?")
    assert "missing" in ans.text.lower() and "2" in ans.text


def test_unrecognized_question_gives_helpful_fallback():
    df = _frame()
    ans = interface.ask(df, "what is the meaning of life")
    assert "highest" in ans.text.lower() or "try" in ans.text.lower()
