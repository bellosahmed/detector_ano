"""Shared number formatting, so answers read naturally to a person."""
from __future__ import annotations


def fmt(x: float) -> str:
    """Human-friendly number: thousands separators for big values, no scientific notation.

    e.g. 7127.0 → "7,127";  20.5 → "20.5";  0.531 → "0.531".
    """
    if abs(x) >= 100:
        return f"{x:,.0f}"          # big numbers: 7,127 (grouped, no decimals)
    if abs(x) >= 1:
        return f"{x:.2f}".rstrip("0").rstrip(".")   # mid: 20.5, drop trailing zeros
    return f"{x:.3g}"               # small: 0.531 (three significant figures)
