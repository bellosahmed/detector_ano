"""Query engine — the truth layer.

Every function is a thin, exact wrapper over pandas. The point is not cleverness;
it is that the numbers the user sees come from here (deterministic code) and can be
re-verified against pandas at any time. The LLM never produces these figures.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def maximum(df: pd.DataFrame, signal: str) -> float:
    return df[signal].max()


def minimum(df: pd.DataFrame, signal: str) -> float:
    return df[signal].min()


def mean(df: pd.DataFrame, signal: str) -> float:
    return df[signal].mean()


def when_extreme(df: pd.DataFrame, time_col: str, signal: str, kind: str = "max"):
    idx = df[signal].idxmax() if kind == "max" else df[signal].idxmin()
    return df.loc[idx, time_col]


def count_over(df: pd.DataFrame, signal: str, threshold: float) -> int:
    return int((df[signal] > threshold).sum())


def count_under(df: pd.DataFrame, signal: str, threshold: float) -> int:
    return int((df[signal] < threshold).sum())


def correlation(df: pd.DataFrame, a: str, b: str) -> float:
    """Pearson correlation between two signals (exact, matches pandas)."""
    return float(pd.to_numeric(df[a], errors="coerce").corr(pd.to_numeric(df[b], errors="coerce")))


def top_correlations(df: pd.DataFrame, signals: list[str], k: int = 3) -> list[tuple[str, str, float]]:
    """The k most strongly correlated signal pairs (by absolute correlation)."""
    pairs = []
    for i, a in enumerate(signals):
        for b in signals[i + 1:]:
            c = correlation(df, a, b)
            if not pd.isna(c):
                pairs.append((a, b, c))
    return sorted(pairs, key=lambda p: abs(p[2]), reverse=True)[:k]


def missing_count(df: pd.DataFrame, signal: str, sentinels: list[float] | tuple = ()) -> int:
    """How many values are missing — blanks (NaN) plus sentinel markers like -9999."""
    col = pd.to_numeric(df[signal], errors="coerce")
    n = int(col.isna().sum())
    if len(sentinels):
        n += int(col.isin(list(sentinels)).sum())
    return n


def skewness(df: pd.DataFrame, signal: str) -> float:
    """Distribution skewness of a signal (0 = symmetric, >0 right-tailed). Matches pandas."""
    return float(pd.to_numeric(df[signal], errors="coerce").skew())


def trend(df: pd.DataFrame, signal: str) -> float:
    """Slope of a least-squares line through the signal (per step). Sign = direction."""
    y = pd.to_numeric(df[signal], errors="coerce").to_numpy(float)
    mask = ~np.isnan(y)
    if mask.sum() < 2:
        return 0.0
    x = np.arange(len(y))[mask]
    return float(np.polyfit(x, y[mask], 1)[0])
