"""Reasoning — explain WHY a flagged point is anomalous, in plain words.

The trust rule still holds: the figures (the value, the expected level, the percentage)
are computed from the data; reasoning only *compares* them and chooses wording plus a
confidence verdict. It never invents a number.

Two ingredients:
  * how the point differs from what was expected there (direction + size);
  * whether other signals moved at the same time (cross-signal corroboration), which
    tells us if it looks like a real event or an isolated (single-sensor) glitch.
"""
from __future__ import annotations

import numbers
from dataclasses import dataclass, field

import pandas as pd

from .formatting import fmt as _fmt


@dataclass
class Explanation:
    signal: str
    time: object
    value: float
    expected: float
    direction: str                       # "above" or "below"
    pct_change: float | None
    co_moving: list[str] = field(default_factory=list)
    verdict: str = ""
    text: str = ""


def expected_value(series: pd.Series, idx, period: int | None = None) -> float:
    """The 'normal' level around a point — a robust local median that ignores the point.

    A centered rolling median (window ≈ the seasonal period) tracks the local level and
    is not pulled by the anomaly itself, so 'value vs expected' is a fair comparison.
    """
    values = pd.to_numeric(series, errors="coerce")
    window = max(5, period or 25)
    local = values.rolling(window, center=True, min_periods=1).median()
    return float(local.loc[idx])


def temporal_context(time) -> str:
    """Human-readable timing context (weekday, weekend, date) for an anomaly's timestamp.

    Gives an LLM real footing to hypothesise a cause (holidays, weekends, seasons). Returns
    "" for non-date indices so callers can append it unconditionally.
    """
    if isinstance(time, numbers.Number):
        return ""  # a plain row index, not a date
    try:
        ts = pd.Timestamp(time)
    except (ValueError, TypeError):
        return ""
    if pd.isna(ts):
        return ""
    day = ts.day_name()
    weekend = " (a weekend)" if ts.weekday() >= 5 else ""
    return f"This occurred on a {day}{weekend}, {ts.strftime('%-d %B %Y')}."


def verdict(n_signals: int, n_co_moving: int) -> str:
    """Confidence wording from how many signals moved together (incl. none for univariate)."""
    if n_signals <= 1:
        return "Single signal — cannot cross-check against other signals."
    if n_co_moving >= 2:
        return "Multiple signals moved together — looks like a real event."
    if n_co_moving == 1:
        return "One other signal moved too — possibly a real event."
    return "Isolated to this signal — more likely a sensor glitch than a real event."


def describe(signal: str, time, value: float, expected: float,
             co_moving: list[str] | None = None, n_signals: int = 1) -> Explanation:
    co_moving = co_moving or []
    direction = "above" if value >= expected else "below"
    pct = ((value - expected) / expected * 100.0) if expected not in (0, None) else None

    pct_txt = f"{abs(pct):.0f}% {direction} " if pct is not None else f"{direction} "
    text = (f"{signal} was {_fmt(value)} at {time} — {pct_txt}the expected "
            f"~{_fmt(expected)} for that time.")
    if co_moving:
        text += f" {', '.join(co_moving)} moved at the same time."

    v = verdict(n_signals, len(co_moving))
    return Explanation(
        signal=signal, time=time, value=value, expected=expected,
        direction=direction, pct_change=pct, co_moving=co_moving,
        verdict=v, text=text + " " + v,
    )
