"""Tests for the Matrix Profile shape detector.

Matrix Profile flags *shape* anomalies: a subsequence unlike anything else in the series
(a discord). This targets the subtle/contextual anomalies the diff-based ensemble misses.
Skipped if stumpy is absent (it is an optional dependency).
"""
import numpy as np
import pandas as pd
import pytest

pytest.importorskip("stumpy")

from anomaly_detector import detect


def test_matrix_profile_scores_align_to_input():
    s = pd.Series(np.sin(np.arange(400) * 0.3))
    scores = detect.matrix_profile_scores(s, window=20)
    assert len(scores) == len(s)
    assert np.isfinite(scores.to_numpy()).all()


def test_matrix_profile_flags_a_shape_anomaly():
    # a clean periodic signal with ONE cycle replaced by a flat segment (a shape discord)
    t = np.arange(600)
    x = np.sin(2 * np.pi * t / 20)
    x[300:320] = 0.0  # the anomaly: a flat stretch where a wave should be
    scores = detect.matrix_profile_scores(pd.Series(x), window=20)
    top = scores.nlargest(25).index
    assert any(280 <= int(i) <= 340 for i in top)


def test_multiscale_matrix_profile_detects_anomaly_at_an_unknown_scale():
    # anomaly spans ~60 points; a multi-window scan should still find it without knowing the size
    t = np.arange(800)
    x = np.sin(2 * np.pi * t / 20)
    x[400:460] = 0.0
    scores = detect.matrix_profile_multiscale(pd.Series(x), windows=(20, 50, 100))
    assert len(scores) == len(x)
    top = scores.nlargest(30).index
    assert any(380 <= int(i) <= 480 for i in top)
