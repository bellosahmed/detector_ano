"""Dataset profiler — infer a time-series dataset's shape from an arbitrary dataframe.

This is what lets the system accept *any* dataset: it works out which column is time,
how often it samples, which columns are signals, and what values mean "missing".
Numbers are never invented here — profiling is pure inspection.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import pandas as pd


# heuristics (date parsing, sentinels, frequency) run on at most this many head rows
_SAMPLE_ROWS = 20_000


def _parse_dates(series: pd.Series) -> pd.Series:
    """Coerce to datetimes. We probe arbitrary columns, so the 'could not infer format'
    warning is expected here and suppressed (there is no single format to give)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return pd.to_datetime(series, errors="coerce")


@dataclass
class DatasetProfile:
    time_col: str | None = None
    freq: str | None = None
    signals: list[str] = field(default_factory=list)
    sentinels: list[float] = field(default_factory=list)
    n_rows: int = 0
    is_multivariate: bool = False


# column-name hints — a tie-breaker so an obviously-named time column wins
_TIME_NAMES = ("timestamp", "datetime", "date", "time", "period", "ds")


def _name_bonus(name: object) -> float:
    label = str(name).lower()
    return 0.5 if any(k in label for k in _TIME_NAMES) else 0.0


def _time_score(series: pd.Series, name: object = "") -> float:
    """How much a column looks like the dataset's time index (0 = not time).

    Rewards being parseable as dates and being monotonic (a real index usually increases),
    with a smaller bonus when the column *name* looks time-like — so an incidental date column
    (e.g. birthdays) loses to the index, and an obviously-named "timestamp" wins genuine ties.
    """
    # a purely numeric column is a signal, not a timestamp — pd.to_datetime would otherwise
    # read plain numbers as epoch-nanoseconds and mis-flag it as the time column.
    if pd.api.types.is_numeric_dtype(series):
        return 0.0
    parsed = _parse_dates(series)
    parseable = parsed.notna().mean()
    if parseable < 0.9:
        return 0.0
    ordered = parsed.dropna()
    monotonic = ordered.is_monotonic_increasing or ordered.is_monotonic_decreasing
    return parseable + (1.0 if monotonic else 0.0) + _name_bonus(name)


def _pick_time_col(df: pd.DataFrame) -> str | None:
    scored = [(c, _time_score(df[c], c)) for c in df.columns]
    best = max(scored, key=lambda cs: cs[1], default=(None, 0.0))
    return best[0] if best[1] > 0 else None


def _infer_freq(series: pd.Series) -> str | None:
    """Best-effort sampling frequency of a (possibly unsorted) time column."""
    times = _parse_dates(series).dropna().sort_values()
    if len(times) < 3:
        return None
    return pd.infer_freq(pd.DatetimeIndex(times))


def _find_sentinels(series: pd.Series) -> list[float]:
    """Repeated extreme constants that really mean 'missing' (e.g. -9999)."""
    values = pd.to_numeric(series, errors="coerce").dropna()
    if len(values) < 10:
        return []
    q1, q3 = values.quantile(0.25), values.quantile(0.75)
    iqr = q3 - q1
    if iqr == 0:
        return []
    lo, hi = q1 - 3 * iqr, q3 + 3 * iqr
    counts = values.value_counts()
    sentinels = [
        float(v) for v, n in counts.items()
        if n >= 3 and (v < lo or v > hi)
    ]
    return sentinels


def profile(df: pd.DataFrame, time_col: str | None = None) -> DatasetProfile:
    # detection heuristics (date parsing, freq, sentinels) run on a head sample so profiling
    # a huge file stays fast; dtype checks + n_rows use the full frame.
    sample = df.head(_SAMPLE_ROWS)

    # honour a user-supplied time column; otherwise infer it
    if time_col not in df.columns:
        time_col = _pick_time_col(sample)

    signals = [
        c for c in df.columns
        if c != time_col and pd.api.types.is_numeric_dtype(df[c])
    ]

    freq = _infer_freq(sample[time_col]) if time_col is not None else None

    sentinels: list[float] = []
    for c in signals:
        for s in _find_sentinels(sample[c]):
            if s not in sentinels:
                sentinels.append(s)

    return DatasetProfile(
        time_col=time_col,
        freq=freq,
        signals=signals,
        sentinels=sentinels,
        n_rows=len(df),
        is_multivariate=len(signals) > 1,
    )
