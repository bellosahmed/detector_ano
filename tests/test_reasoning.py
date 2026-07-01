"""Tests for reasoning — explaining WHY a point is anomalous.

Reasoning never invents numbers: the value and the expected level come from the data;
reasoning only compares them and picks words + a confidence verdict.
"""
import numpy as np
import pandas as pd

from anomaly_detector import reasoning


def test_expected_value_ignores_the_spike_itself():
    s = pd.Series([10.0] * 100)
    s.iloc[50] = 100.0
    expected = reasoning.expected_value(s, 50, period=24)
    assert abs(expected - 10.0) < 1.0  # robust to the spike, not pulled toward 100


def test_describe_reports_direction_below_and_percentage():
    e = reasoning.describe("visits", "2022-03-01", value=100.0, expected=1000.0,
                           co_moving=[], n_signals=1)
    assert e.direction == "below"
    assert e.pct_change is not None and round(e.pct_change) == -90
    assert "below" in e.text.lower()


def test_describe_reports_direction_above():
    e = reasoning.describe("temp", "2021-07-04", value=40.0, expected=20.0,
                           co_moving=[], n_signals=1)
    assert e.direction == "above"
    assert round(e.pct_change) == 100


def test_verdict_real_event_when_multiple_signals_move_together():
    v = reasoning.verdict(n_signals=4, n_co_moving=3)
    assert "real" in v.lower() or "genuine" in v.lower()


def test_verdict_isolated_glitch_when_only_one_signal_moves():
    v = reasoning.verdict(n_signals=4, n_co_moving=0)
    assert "isolat" in v.lower() or "glitch" in v.lower()


def test_temporal_context_names_the_weekday():
    ctx = reasoning.temporal_context("2023-11-20")  # a Monday
    assert "Monday" in ctx


def test_temporal_context_flags_a_weekend():
    ctx = reasoning.temporal_context("2023-11-18")  # a Saturday
    assert "weekend" in ctx.lower()


def test_temporal_context_handles_non_dates_gracefully():
    assert reasoning.temporal_context("not a date") == ""
    assert reasoning.temporal_context(42) == ""


def test_describe_multivariate_names_the_co_moving_signals():
    e = reasoning.describe("power", "2020-01-01", value=5.0, expected=50.0,
                           co_moving=["voltage", "current"], n_signals=3)
    assert set(e.co_moving) == {"voltage", "current"}
    assert "voltage" in e.text and "current" in e.text
