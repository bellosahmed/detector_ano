"""Tests for the core API — the one entry point that ties profiling + detection together.

Both the web app and (later) an MCP server call this, so numbers come from one place.
"""
import numpy as np
import pandas as pd

from anomaly_detector import core


def _frame_with_spike(n=300, spike_at=150, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    value = 20 + 5 * np.sin(2 * np.pi * t / 24) + rng.normal(0, 0.5, n)
    value[spike_at] = 200.0
    return pd.DataFrame({
        "time": pd.date_range("2020-01-01", periods=n, freq="h"),
        "value": value,
    })


def test_analyze_infers_the_profile():
    result = core.analyze(_frame_with_spike())
    assert result.profile.time_col == "time"
    assert result.profile.signals == ["value"]


def test_analyze_flags_the_known_spike_with_its_timestamp():
    df = _frame_with_spike(spike_at=150)
    result = core.analyze(df)
    anomalies = result.anomalies["value"]
    flagged_times = [a.time for a in anomalies]
    assert df.loc[150, "time"] in flagged_times


def test_analyze_attaches_an_explanation_to_each_anomaly():
    df = _frame_with_spike(spike_at=150)
    result = core.analyze(df)
    anomalies = result.anomalies["value"]
    assert anomalies and all(a.explanation for a in anomalies)
    assert any("expected" in a.explanation.lower() for a in anomalies)


def test_analyze_reasons_across_signals():
    """Three signals spiking together → the explanation names the others and calls it a real event."""
    rng = np.random.default_rng(0)
    n = 300
    t = np.arange(n)
    df = pd.DataFrame({"time": pd.date_range("2020-01-01", periods=n, freq="h")})
    for name, base in (("power", 50), ("voltage", 230), ("current", 12)):
        v = base + rng.normal(0, base * 0.02, n)
        v[150] += base  # all three spike at the same instant
        df[name] = v

    result = core.analyze(df)
    power_anoms = [a for a in result.anomalies["power"] if a.time == df.loc[150, "time"]]
    assert power_anoms, "the shared spike should be flagged for 'power'"
    expl = power_anoms[0].explanation
    assert ("voltage" in expl or "current" in expl)
    assert "real event" in expl.lower()


def test_shape_scan_finds_a_pattern_anomaly_the_ensemble_misses():
    import pytest
    pytest.importorskip("stumpy")
    # a periodic signal with one cycle flattened — a SHAPE anomaly (no spike/level change)
    t = np.arange(600)
    x = np.sin(2 * np.pi * t / 20)
    x[300:320] = 0.0
    df = pd.DataFrame({"time": pd.date_range("2020-01-01", periods=600, freq="h"), "value": x})

    default = core.analyze(df)               # diff-ensemble only
    shaped = core.analyze(df, shape=True)    # + Matrix Profile shape scan

    idx = {t: i for i, t in enumerate(df["time"])}

    # the shape scan is opt-in: the default run never produces shape anomalies
    assert all(a.kind == "level" for a in default.anomalies["value"])

    # with the scan on, a SHAPE anomaly is flagged inside the flattened stretch (a discord
    # the ensemble's differencing can't see — the flat interior has no spikes or steps)
    shape_anoms = [a for a in shaped.anomalies["value"] if a.kind == "shape"]
    assert shape_anoms
    assert any(300 <= idx[a.time] <= 320 for a in shape_anoms)


def test_large_series_stays_fast_and_still_detects():
    # above the STL cap the fast O(n) seasonal path must be used, not the slow STL (which
    # took ~30s at this size and made large uploads hang). Guard against that regression.
    import time
    rng = np.random.default_rng(0)
    n = 120_000
    t = np.arange(n)
    v = 20 + 5 * np.sin(2 * np.pi * t / 24) + rng.normal(0, 0.5, n)
    v[60_000] = 200.0
    df = pd.DataFrame({"time": pd.date_range("2000-01-01", periods=n, freq="h"), "value": v})

    t0 = time.time()
    result = core.analyze(df)
    assert time.time() - t0 < 15          # fast path; STL-on-large would blow past this
    assert any(abs(a.value - 200.0) < 1 for a in result.anomalies["value"])


def test_analyze_without_a_time_column():
    rng = np.random.default_rng(0)
    v = rng.normal(0, 1, 300)
    v[150] = 60.0
    df = pd.DataFrame({"value": v})          # no time column at all
    result = core.analyze(df)
    assert result.profile.time_col is None
    assert result.anomalies["value"]         # still finds the spike (indexed by position)


def test_analyze_with_no_numeric_signals_is_graceful():
    df = pd.DataFrame({"name": ["a", "b", "c"], "label": ["x", "y", "z"]})
    result = core.analyze(df)
    assert result.profile.signals == []
    assert result.anomalies == {}


def test_analyze_ignores_sentinels_as_real_anomalies():
    df = _frame_with_spike()
    df.loc[[10, 20, 30, 40], "value"] = -9999.0  # missing markers, not anomalies
    result = core.analyze(df)
    assert -9999.0 in result.profile.sentinels
    flagged_values = [a.value for a in result.anomalies["value"]]
    assert -9999.0 not in flagged_values
