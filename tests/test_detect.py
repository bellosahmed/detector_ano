"""Tests for the anomaly detectors.

Detectors are *judgements*, so unlike the query engine they can be wrong. We verify
them on data with KNOWN injected anomalies and measure a confusion matrix
(TP/TN/FP/FN) → precision/recall/F1. Reported over fixed seeds for stability.
"""
import numpy as np
import pandas as pd

from anomaly_detector import detect


def _clean_signal(n=500, seed=0):
    """A smooth seasonal signal with small noise — no anomalies."""
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    base = 20 + 5 * np.sin(2 * np.pi * t / 24)
    return pd.Series(base + rng.normal(0, 0.5, n))


def test_zscore_flags_an_injected_spike():
    s = _clean_signal()
    s.iloc[100] = 100.0  # obvious spike
    flags = detect.zscore(s, threshold=4.0)
    assert bool(flags.iloc[100]) is True


def test_zscore_leaves_clean_points_unflagged():
    s = _clean_signal()
    s.iloc[100] = 100.0
    flags = detect.zscore(s, threshold=4.0)
    # the point right before the spike is normal
    assert bool(flags.iloc[50]) is False


def test_zscore_returns_a_boolean_mask_aligned_to_input():
    s = _clean_signal(n=120)
    flags = detect.zscore(s, threshold=4.0)
    assert len(flags) == len(s)
    assert flags.dtype == bool


def test_confusion_matrix_counts_are_correct():
    truth = pd.Series([False, False, True, True, False])
    pred = pd.Series([False, True, True, False, False])
    cm = detect.confusion(truth, pred)
    # TP: idx2 ; FP: idx1 ; FN: idx3 ; TN: idx0, idx4
    assert (cm.tp, cm.fp, cm.fn, cm.tn) == (1, 1, 1, 2)


def test_precision_recall_f1_from_confusion():
    cm = detect.Confusion(tp=8, fp=2, fn=2, tn=88)
    assert cm.precision == 0.8
    assert cm.recall == 0.8
    assert round(cm.f1, 3) == 0.8


def test_residual_zscore_detects_a_level_shift():
    """A random walk with a sudden jump — global z misses it, residual z should catch it."""
    rng = np.random.default_rng(3)
    steps = rng.normal(0, 1, 400)
    steps[200] += 25.0  # one big shock → a persistent level shift
    series = pd.Series(100 + np.cumsum(steps))
    flags = detect.residual_zscore(series, period=7, threshold=4.0)
    assert bool(flags.iloc[199] or flags.iloc[200] or flags.iloc[201]) is True


def test_residual_zscore_detects_a_drop_in_a_seasonal_series():
    """Strong weekly seasonality would hide a drop from global z; residual z should see it."""
    rng = np.random.default_rng(4)
    n = 210
    t = np.arange(n)
    v = 5000 + 1500 * np.sin(2 * np.pi * t / 7) + rng.normal(0, 100, n)
    v[100] *= 0.1  # outage drop
    flags = detect.residual_zscore(pd.Series(v), period=7, threshold=4.0)
    assert bool(flags.iloc[100]) is True


def test_residual_zscore_detects_point_spikes_in_a_seasonal_series():
    """Point spikes riding on strong seasonality — the case a plain STL residual misses."""
    rng = np.random.default_rng(5)
    n = 480
    t = np.arange(n)
    v = 20 + 6 * np.sin(2 * np.pi * t / 24) + rng.normal(0, 0.4, n)
    spikes = rng.choice(n, size=10, replace=False)
    v[spikes] += rng.choice([-1, 1], 10) * rng.uniform(20, 30, 10)

    flags = detect.residual_zscore(pd.Series(v), period=24, threshold=4.0)
    truth = pd.Series(False, index=range(n))
    truth.iloc[spikes] = True
    # credit a hit on the spike row or the next (differencing spreads a spike over t, t+1)
    hit = sum(bool(flags.iloc[r]) or (r + 1 < n and bool(flags.iloc[r + 1])) for r in spikes)
    assert hit / len(spikes) >= 0.8


def test_residual_zscore_returns_aligned_boolean_mask():
    s = _clean_signal(n=120)
    flags = detect.residual_zscore(s, period=24, threshold=4.0)
    assert len(flags) == len(s) and flags.dtype == bool


def _seasonal_with_spikes(n=480, seed=6):
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    v = 20 + 6 * np.sin(2 * np.pi * t / 24) + rng.normal(0, 0.4, n)
    spikes = rng.choice(n, size=10, replace=False)
    v[spikes] += rng.choice([-1, 1], 10) * rng.uniform(20, 30, 10)
    return pd.Series(v), spikes


def test_pr_auc_is_one_for_perfect_ranking():
    truth = pd.Series([False, False, True, False, True])
    scores = pd.Series([0.1, 0.2, 0.9, 0.3, 0.8])   # anomalies ranked highest
    assert detect.pr_auc(truth, scores) == 1.0


def test_pr_auc_near_base_rate_for_random_scores():
    rng = np.random.default_rng(0)
    truth = pd.Series(rng.random(1000) < 0.02)       # ~2% positives (imbalanced)
    scores = pd.Series(rng.random(1000))             # random, uninformative
    assert detect.pr_auc(truth, scores) < 0.1        # ~ base rate, not misleadingly high


def test_isolation_forest_scores_flag_an_injected_spike():
    rng = np.random.default_rng(0)
    s = pd.Series(20 + rng.normal(0, 1, 500))
    s.iloc[250] = 100.0
    scores = detect.isolation_forest_scores(s)
    assert len(scores) == len(s)
    assert 250 in set(scores.nlargest(5).index)      # the spike is among the most anomalous


def test_consensus_is_stricter_than_any():
    """Requiring both ensemble views to agree flags a subset of 'any' — fewer false alarms."""
    s, _ = _seasonal_with_spikes()
    any_flags = detect.residual_zscore(s, period=24, threshold=4.0, combine="any")
    all_flags = detect.residual_zscore(s, period=24, threshold=4.0, combine="all")
    assert all_flags.sum() <= any_flags.sum()
    assert int((all_flags & ~any_flags).sum()) == 0  # all-flags are a subset of any-flags


def test_consensus_still_detects_clear_spikes():
    """Consensus must not sacrifice recall on obvious spikes (they show in both views)."""
    s, spikes = _seasonal_with_spikes()
    flags = detect.residual_zscore(s, period=24, threshold=4.0, combine="all")
    n = len(s)
    hit = sum(bool(flags.iloc[r]) or (r + 1 < n and bool(flags.iloc[r + 1])) for r in spikes)
    assert hit / len(spikes) >= 0.8


def test_zscore_recovers_injected_anomalies_with_high_recall():
    """The real benchmark: inject known spikes, measure recall against ground truth."""
    rng = np.random.default_rng(42)
    s = _clean_signal(n=1000, seed=1)
    truth = pd.Series(False, index=s.index)
    spike_positions = rng.choice(s.index, size=20, replace=False)
    s.iloc[spike_positions] += rng.choice([-1, 1], 20) * 30  # clear anomalies
    truth.iloc[spike_positions] = True

    flags = detect.zscore(s, threshold=4.0)
    cm = detect.confusion(truth, flags)
    assert cm.recall >= 0.9  # should catch almost all obvious spikes
