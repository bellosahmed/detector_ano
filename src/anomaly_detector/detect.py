"""Anomaly detectors and the metrics to judge them.

Unlike the query engine, detection is a *judgement* and can be wrong — so this module
also provides the confusion matrix (TP/TN/FP/FN) and precision/recall/F1 used to
measure that error rate on data with known labels.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import average_precision_score
from statsmodels.tsa.seasonal import STL

# scale factor making MAD a consistent estimator of the standard deviation
_MAD_TO_STD = 1.4826


def robust_z(series: pd.Series, absolute: bool = True) -> pd.Series:
    """Each point's robust z-score (median/MAD based).

    Using median/MAD instead of mean/std means a few large anomalies don't inflate
    the spread and hide each other. With ``absolute`` (default) the magnitude is returned;
    with ``absolute=False`` the signed z is returned — used for one-sided scores like a
    Matrix-Profile distance, where only high values are anomalous. NaNs propagate as NaN.
    """
    values = pd.to_numeric(series, errors="coerce")
    median = values.median()
    mad = (values - median).abs().median()
    if mad and not np.isnan(mad):
        scale = mad * _MAD_TO_STD
    else:
        scale = values.std(ddof=0)
    if not scale or np.isnan(scale):
        return pd.Series(0.0, index=series.index)
    deviation = values - median
    return (deviation.abs() if absolute else deviation) / scale


def zscore(series: pd.Series, threshold: float = 4.0) -> pd.Series:
    """Flag points whose robust z-score exceeds ``threshold`` (boolean mask)."""
    return (robust_z(series) > threshold).fillna(False).astype(bool)


# STL(robust) scales badly (~24s at 100k points); above this we use an O(n) seasonal estimate
_MAX_STL_POINTS = 50_000


def _seasonal_by_phase(filled: pd.Series, period: int) -> pd.Series:
    """Fast O(n) seasonal estimate: the mean value at each position-in-period (phase)."""
    phase = np.arange(len(filled)) % period
    seasonal = pd.Series(filled.to_numpy()).groupby(phase).transform("mean")
    return pd.Series(seasonal.to_numpy(), index=filled.index)


def _deseasonalize(filled: pd.Series, period: int | None) -> pd.Series:
    """Subtract the seasonal component (leaving trend + remainder).

    Removing only the seasonal part — not the trend — deseasonalises without the trend
    smoother swallowing isolated spikes. Uses a robust STL for normal-size series; for very
    long series (where STL(robust) would take minutes) it falls back to a per-phase mean,
    which is O(n) and keeps large uploads responsive.
    """
    if period and period >= 2 and len(filled) >= 2 * period:
        if len(filled) <= _MAX_STL_POINTS:
            seasonal = STL(filled.to_numpy(), period=period, robust=True).fit().seasonal
            return filled - pd.Series(seasonal, index=filled.index)
        return filled - _seasonal_by_phase(filled, period)
    return filled


def anomaly_scores(series: pd.Series, period: int | None = None,
                   combine: str = "any") -> pd.Series:
    """Per-point robust z-score from an ensemble of two complementary views.

    * differencing the raw signal → catches point spikes and level shifts;
    * differencing the deseasonalised signal → catches seasonal-relative anomalies
      (e.g. a drop that is small in absolute terms but large for that phase).

    ``combine`` decides how the two views are merged:

    * ``"any"`` (max) — flag if *either* view finds the point unusual (best recall);
    * ``"all"`` (min) — flag only if *both* agree (**consensus**: far fewer false alarms,
      since real events show up in both views but noise usually shows in only one).
    """
    filled = pd.to_numeric(series, errors="coerce").interpolate(limit_direction="both")
    z_point = robust_z(filled.diff())
    z_season = robust_z(_deseasonalize(filled, period).diff())
    stacked = pd.concat([z_point, z_season], axis=1)
    return stacked.min(axis=1) if combine == "all" else stacked.max(axis=1)


def residual_zscore(series: pd.Series, period: int | None = None,
                    threshold: float = 4.0, combine: str = "any") -> pd.Series:
    """Detect anomalies (boolean mask) — handles spikes, level shifts and seasonality.

    Missing/sentinel points are never flagged, and a run of consecutive flags is
    collapsed to its first point (differencing spreads one spike across t and t+1).
    """
    z = anomaly_scores(series, period=period, combine=combine)
    flags = (z > threshold).fillna(False).astype(bool)
    flags[pd.to_numeric(series, errors="coerce").isna().to_numpy()] = False

    arr = flags.to_numpy().copy()
    for i in range(1, len(arr)):
        if arr[i] and arr[i - 1]:
            arr[i] = False
    return pd.Series(arr, index=series.index)


def matrix_profile_scores(series: pd.Series, window: int = 100) -> pd.Series:
    """Per-point *shape* anomaly score via the Matrix Profile (higher = more unusual shape).

    For each length-`window` subsequence, the Matrix Profile is the distance to its nearest
    neighbour elsewhere in the series; a large distance means "this shape occurs nowhere else"
    — a discord. This catches subtle/contextual anomalies that differencing misses. Requires
    the optional `stumpy` dependency (imported lazily so the core never needs it).
    """
    import stumpy

    values = pd.to_numeric(series, errors="coerce").interpolate(limit_direction="both").to_numpy(float)
    n = len(values)
    if n < 2 * window:
        return pd.Series(0.0, index=series.index)

    mp = stumpy.stump(values, m=window)[:, 0].astype(float)
    mp[~np.isfinite(mp)] = np.nanmax(mp[np.isfinite(mp)]) if np.isfinite(mp).any() else 0.0

    scores = np.full(n, np.nan)
    centre = window // 2
    scores[centre:centre + len(mp)] = mp
    return pd.Series(scores, index=series.index).interpolate(limit_direction="both").fillna(0.0)


def matrix_profile_multiscale(series: pd.Series,
                              windows: tuple[int, ...] = (50, 100, 200, 400)) -> pd.Series:
    """Matrix Profile at several window lengths, combined — finds discords of unknown size.

    A single window only sees anomalies near that scale; real datasets (e.g. UCR) mix scales.
    Each window's scores are standardised (so longer windows don't dominate) and the per-point
    maximum is taken, so a point is unusual if it stands out at *any* scale. (A lightweight
    stand-in for MERLIN's exhaustive length search.)
    """
    combined = None
    for w in windows:
        if len(series) < 2 * w:
            continue
        s = matrix_profile_scores(series, window=w)
        z = (s - s.mean()) / (s.std(ddof=0) or 1.0)     # standardise this scale
        combined = z if combined is None else np.maximum(combined, z)
    if combined is None:
        return pd.Series(0.0, index=series.index)
    return pd.Series(combined, index=series.index)


@dataclass
class Confusion:
    tp: int
    fp: int
    fn: int
    tn: int

    @property
    def precision(self) -> float:
        denom = self.tp + self.fp
        return self.tp / denom if denom else 0.0

    @property
    def recall(self) -> float:
        denom = self.tp + self.fn
        return self.tp / denom if denom else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) else 0.0


def pr_auc(truth: pd.Series, scores: pd.Series) -> float:
    """Area under the precision-recall curve (average precision).

    Threshold-independent and **robust to class imbalance** (unlike accuracy or ROC-AUC, which
    look flattering when anomalies are rare). `scores`: higher = more anomalous.
    """
    t = np.asarray(truth, dtype=int)
    s = np.asarray(scores, dtype=float)
    if np.isnan(s).any():                      # missing/sentinel points → least anomalous
        fill = np.nanmin(s) if np.isfinite(s).any() else 0.0
        s = np.where(np.isnan(s), fill, s)
    if t.sum() == 0 or t.sum() == len(t):
        return float("nan")
    return float(average_precision_score(t, s))


def isolation_forest_scores(series: pd.Series, seed: int = 0) -> pd.Series:
    """Per-point anomaly score from an Isolation Forest baseline (higher = more anomalous).

    An established off-the-shelf detector, included so results can be compared against it.
    """
    x = pd.to_numeric(series, errors="coerce").interpolate(limit_direction="both").to_numpy()
    model = IsolationForest(random_state=seed, contamination="auto")
    model.fit(x.reshape(-1, 1))
    # score_samples: higher = more normal, so negate to make higher = more anomalous
    return pd.Series(-model.score_samples(x.reshape(-1, 1)), index=series.index)


def confusion(truth: pd.Series, pred: pd.Series) -> Confusion:
    """Confusion matrix of two boolean masks (must share an index/length)."""
    t = np.asarray(truth, dtype=bool)
    p = np.asarray(pred, dtype=bool)
    tp = int(np.sum(t & p))
    fp = int(np.sum(~t & p))
    fn = int(np.sum(t & ~p))
    tn = int(np.sum(~t & ~p))
    return Confusion(tp=tp, fp=fp, fn=fn, tn=tn)
