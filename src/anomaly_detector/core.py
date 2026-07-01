"""Core API — the single entry point that profiles a dataset and detects anomalies.

Everything the user sees flows through here, so numbers come from one place (the
profiler + query engine + detectors). The web app and any future adapter call this;
they never re-implement the logic.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import detect, reasoning
from .profile import DatasetProfile, profile


@dataclass
class Anomaly:
    time: object      # the timestamp (whatever the time column holds)
    value: float      # the anomalous reading
    zscore: float     # how far out it is (robust z)
    explanation: str = ""              # plain-English "why", numbers from the data
    verdict: str = ""                  # confidence: real event vs isolated glitch
    co_moving: list[str] = field(default_factory=list)  # other signals that moved too
    kind: str = "level"                # "level" (spike/shift/seasonal) or "shape" (pattern discord)


@dataclass
class AnalysisResult:
    profile: DatasetProfile
    anomalies: dict[str, list[Anomaly]] = field(default_factory=dict)


# seasonal period (in steps) for calendar frequencies that aren't a fixed duration
_CALENDAR_PERIODS = {"W": 52, "M": 12, "Q": 4, "Y": 1, "A": 1}


def _period_from_freq(freq: str | None) -> int | None:
    """Dominant seasonal period, in steps, for an inferred sampling frequency.

    Sub-daily data → one *day* is the cycle (hourly=24, 30-min=48). Daily data →
    one *week* (7). Coarser calendar frequencies map to a yearly cycle.
    """
    if not freq:
        return None
    try:
        rng = pd.date_range("2000-01-01", periods=2, freq=freq)
        delta = rng[1] - rng[0]
    except (ValueError, TypeError):
        delta = None
    if delta is not None and delta > pd.Timedelta(0):
        day = pd.Timedelta("1D")
        if delta < day:
            return int(round(day / delta))        # steps per day
        if delta == day:
            return 7                                # weekly cycle
        if delta == pd.Timedelta("7D"):
            return 52                               # yearly cycle
    key = freq.upper().lstrip("0123456789")[:1]     # "MS" -> "M", "W-SUN" -> "W"
    return _CALENDAR_PERIODS.get(key)


def _mask_sentinels(series: pd.Series, sentinels: list[float]) -> pd.Series:
    """Replace sentinel values (e.g. -9999) with NaN so they aren't seen as anomalies."""
    if not sentinels:
        return series
    return series.mask(series.isin(sentinels))


def _shape_anomalies(clean, level_flags, prof, df, shape_threshold) -> list[Anomaly]:
    """Matrix-Profile 'shape' anomalies for one signal, excluding points already flagged.

    Best-effort: needs the optional `stumpy` dependency; returns [] if it isn't installed.
    Note: Matrix-Profile z-scores sit on a lower scale than the differencing detector, so this
    uses its own `shape_threshold` (~3), calibrated so clean data stays quiet.
    """
    try:
        mp = detect.matrix_profile_multiscale(clean)
    except ImportError:
        return []

    z = detect.robust_z(mp, absolute=False)        # one-sided: only a high Matrix Profile = anomalous
    hits = z[(z > shape_threshold) & ~level_flags].nlargest(10).index  # top few, not double-counted

    out: list[Anomaly] = []
    for idx in hits:
        t = df.loc[idx, prof.time_col] if prof.time_col is not None else idx
        out.append(Anomaly(
            time=t, value=float(clean.loc[idx]), zscore=float(z.loc[idx]), kind="shape",
            explanation=(f"Unusual pattern around {t} — this stretch of "
                         f"{clean.name or 'the signal'} has a shape that occurs nowhere else."),
            verdict="Shape anomaly — pattern differs from the norm (single-signal).",
        ))
    return out


# series longer than this skip the (heavier) shape scan to keep the app responsive
_MAX_SHAPE_POINTS = 200_000


# z=5 is the tuned operating point (see benchmarks/tune.py): on the samples it keeps
# recall while dropping false alarms to ~0%, and cuts NAB false alarms 6.3% -> 4.8%.
def analyze(df: pd.DataFrame, threshold: float = 5.0, combine: str = "any",
            shape: bool = False, shape_threshold: float = 3.0,
            time_col: str | None = None) -> AnalysisResult:
    """Find and explain the anomalies in a table of time-series data.

    In plain terms: work out the shape of the data (which column is time, which are sensors),
    score how unusual each point is for every sensor, then for the flagged points write a plain
    explanation — including which *other* sensors moved at the same moment. Returns an
    ``AnalysisResult`` (the profile + a list of anomalies per signal).

    ``threshold`` = how unusual a point must be to be flagged (higher = stricter). ``shape=True``
    adds a slower Matrix-Profile scan for unusual patterns. ``time_col`` overrides the detected
    time column.
    """
    prof = profile(df, time_col=time_col)
    period = _period_from_freq(prof.freq)
    result = AnalysisResult(profile=prof)

    # first pass: clean each signal + score it (needed before reasoning can cross-check signals)
    scored: dict[str, tuple[pd.Series, pd.Series, pd.Series]] = {}
    for signal in prof.signals:
        clean = _mask_sentinels(pd.to_numeric(df[signal], errors="coerce"), prof.sentinels)
        z = detect.anomaly_scores(clean, period=period, combine=combine)
        flags = detect.residual_zscore(clean, period=period, threshold=threshold, combine=combine)
        scored[signal] = (clean, z, flags)

    # second pass: build anomalies with explanations (why + cross-signal verdict)
    n_signals = len(prof.signals)
    for signal, (clean, z, flags) in scored.items():
        anomalies: list[Anomaly] = []
        for idx in flags[flags].index:
            t = df.loc[idx, prof.time_col] if prof.time_col is not None else idx
            value = float(clean.loc[idx])

            # which OTHER sensors were also unusual at this same moment? (cross-sensor evidence:
            # several sensors moving together points to a real event, one alone to a glitch)
            co_moving = [
                other for other, (_, oz, _) in scored.items()
                if other != signal and float(oz.loc[idx]) > threshold
            ]
            expected = reasoning.expected_value(clean, idx, period)
            expl = reasoning.describe(signal, t, value, expected, co_moving, n_signals)

            anomalies.append(Anomaly(
                time=t, value=value, zscore=float(z.loc[idx]),
                explanation=expl.text, verdict=expl.verdict, co_moving=co_moving,
            ))

        # optional deeper scan: Matrix Profile catches *shape* anomalies (unusual patterns
        # that aren't spikes/level-shifts) the diff-ensemble misses. Opt-in and best-effort.
        if shape and len(clean) <= _MAX_SHAPE_POINTS:
            anomalies += _shape_anomalies(clean, flags, prof, df, shape_threshold)

        result.anomalies[signal] = anomalies

    return result
