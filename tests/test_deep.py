"""Tests for the optional deep comparator (LSTM autoencoder).

Deep models are *comparison baselines*, not the default detector. These tests check the
scoring scaffolding is sound and that the model can flag an obvious anomaly — kept small
and CPU/seeded so they run fast and deterministically. Skipped entirely if torch is absent.
"""
import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from anomaly_detector import deep


def test_window_shapes():
    w = deep._window(np.arange(10.0), 4)
    assert w.shape == (7, 4)  # n - window + 1 windows


def test_scores_align_to_input_and_are_finite():
    rng = np.random.default_rng(0)
    s = pd.Series(20 + 5 * np.sin(2 * np.pi * np.arange(200) / 24) + rng.normal(0, 0.3, 200))
    scores = deep.lstm_autoencoder_scores(s, window=20, hidden=8, epochs=3,
                                          device="cpu", seed=0)
    assert len(scores) == len(s)
    assert np.isfinite(scores.to_numpy()).all()


def test_detects_an_obvious_spike():
    rng = np.random.default_rng(1)
    n = 300
    s = 20 + 5 * np.sin(2 * np.pi * np.arange(n) / 24) + rng.normal(0, 0.3, n)
    s[150] += 40.0  # a glaring spike
    scores = deep.lstm_autoencoder_scores(pd.Series(s), window=20, hidden=12, epochs=25,
                                          device="cpu", seed=0)
    top = scores.nlargest(10).index
    assert any(abs(int(i) - 150) <= 20 for i in top)  # spike is among the highest-scoring points
