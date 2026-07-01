"""Natural-language interface — ask a dataset questions in plain English.

Deterministic and honest: the intent and which signal are parsed from the question,
then the answer's numbers come from the query engine / core (never invented). This is
the "ask questions about it" half of the tool; an optional LLM can later reword these
answers, but it will never compute them.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import llm, query, reasoning
from .core import analyze
from .profile import DatasetProfile, profile, _parse_dates


@dataclass
class Answer:
    text: str
    value: object = None
    anomaly: object = None   # the top anomaly (when the question is about anomalies)
    recognized: bool = True  # False when keyword parsing fell through to the fallback


def _pick_signal(question: str, prof: DatasetProfile) -> str | None:
    """Match a signal named in the question; else fall back to the first signal."""
    q = question.lower()
    for s in prof.signals:
        if s.lower() in q:
            return s
    return prof.signals[0] if prof.signals else None


def _number(question: str) -> float | None:
    m = re.search(r"-?\d+(?:\.\d+)?", question)
    return float(m.group()) if m else None


def _corr_word(c: float) -> str:
    a = abs(c)
    strength = "strong" if a >= 0.7 else "moderate" if a >= 0.4 else "weak"
    return f"{strength} {'positive' if c >= 0 else 'negative'}"


def _fmt(x: float) -> str:
    if abs(x) >= 100:
        return f"{x:,.0f}"
    return f"{x:.2f}".rstrip("0").rstrip(".")


def ask(df: pd.DataFrame, question: str, prof: DatasetProfile | None = None,
        use_llm: bool = False) -> Answer:
    prof = prof or profile(df)
    ans = _answer(df, question, prof)
    if not (use_llm and llm.available()):
        return ans

    # free phrasing: if the keyword parser didn't recognise the question, let the LLM planner
    # pick the intent — the answer's numbers are then still computed by code.
    if not ans.recognized:
        planned = _from_plan(df, question, prof)
        if planned is not None:
            ans = planned

    if ans.value is None:
        return ans

    if ans.anomaly is not None:
        # explain WHY: the model hypothesises a cause from timing + related signals.
        # Numbers stay code-computed (faithful()); the cause is labelled an AI suggestion.
        a = ans.anomaly
        context = reasoning.temporal_context(a.time)
        if a.co_moving:
            context += f" Related signals that moved at the same time: {', '.join(a.co_moving)}."
        cause = llm.suggest_cause(ans.text, context)
        if cause:
            return Answer(f"{ans.text}\n\nPossible cause (AI suggestion): {cause}",
                          ans.value, ans.anomaly)
        return ans

    # otherwise just reword the computed answer
    return Answer(llm.explain(ans.text), ans.value)


def _years(question: str) -> list[int]:
    return [int(y) for y in re.findall(r"\b(?:19|20)\d{2}\b", question)]


def _scope_year(df: pd.DataFrame, time_col: str | None, year: int) -> pd.DataFrame:
    if time_col is None:
        return df
    years = _parse_dates(df[time_col]).dt.year
    return df[years == year]


_MONTHS = {m: i for i, m in enumerate(
    ["january", "february", "march", "april", "may", "june", "july",
     "august", "september", "october", "november", "december"], start=1)}
_MONTHS.update({m[:3]: i for m, i in list(_MONTHS.items())})


def _months_in(question: str) -> list[tuple[str, int]]:
    """Full month names present in the question, in order, as (name, month-number) pairs."""
    q = question.lower()
    found = [(name, num) for name, num in _MONTHS.items()
             if len(name) > 3 and re.search(rf"\b{name}\b", q)]
    # preserve the order they appear in the question
    return sorted(found, key=lambda nm: q.find(nm[0]))


def _scope_month(df: pd.DataFrame, time_col: str, month_num: int) -> pd.DataFrame:
    return df[_parse_dates(df[time_col]).dt.month == month_num]


def _resolve_period(question: str, df: pd.DataFrame, time_col: str | None):
    """Filter df to a period named in the question. Returns (scoped_df, label).

    Handles: 'last month' / 'last week', a named month (+ optional year), and a bare year.
    """
    if time_col is None:
        return df, ""
    q = question.lower()
    ts = _parse_dates(df[time_col])
    years = _years(question)

    # relative windows, anchored to the most recent timestamp in the data
    end = ts.max()
    if pd.notna(end):
        if "last month" in q:
            m = df[(ts.dt.year == end.year) & (ts.dt.month == end.month)]
            return m, f" (in {end.strftime('%B %Y')})"
        if "last week" in q:
            return df[ts >= end - pd.Timedelta(days=7)], " (last 7 days)"

    # a named month, optionally with a year (else the latest year that has that month)
    for name, num in _MONTHS.items():
        if re.search(rf"\b{name}\b", q):
            in_month = ts.dt.month == num
            year = years[0] if years else int(ts[in_month].dt.year.max())
            label = pd.Timestamp(year=year, month=num, day=1).strftime(" (in %B %Y)")
            return df[in_month & (ts.dt.year == year)], label

    if len(years) == 1:
        return _scope_year(df, time_col, years[0]), f" (in {years[0]})"

    return df, ""


def _specific_date(question: str, default_year: int | None = None) -> pd.Timestamp | None:
    """A single calendar day named in the question: ISO (2023-11-20) or '20 November [2023]'."""
    m = re.search(r"\b(\d{4})-(\d{1,2})-(\d{1,2})\b", question)
    if m:
        try:
            return pd.Timestamp(int(m.group(1)), int(m.group(2)), int(m.group(3)))
        except ValueError:
            return None
    ql = question.lower()
    for name, num in _MONTHS.items():
        mm = re.search(rf"\b(\d{{1,2}})\s+{name}\b", ql) or re.search(rf"\b{name}\s+(\d{{1,2}})\b", ql)
        if mm:
            years = _years(question)
            year = years[0] if years else default_year
            if year is None:
                return None
            try:
                return pd.Timestamp(year, num, int(mm.group(1)))
            except ValueError:
                return None
    return None


def _explain_at_date(df, prof, signal, day: pd.Timestamp) -> Answer:
    """Explain the anomaly on a specific day (or, if none was flagged, what happened that day)."""
    ts = _parse_dates(df[prof.time_col])
    result = analyze(df)
    same_day = [a for a in result.anomalies.get(signal, [])
                if pd.notna(pd.Timestamp(a.time)) and pd.Timestamp(a.time).date() == day.date()]
    if same_day:
        a = max(same_day, key=lambda x: x.zscore)
        return Answer(a.explanation, a.value, anomaly=a)   # ask() adds an AI cause when enabled

    sub = df[ts.dt.date == day.date()]
    if len(sub) == 0:
        return Answer(f"There's no data for {day.date()} in this dataset.")
    val = query.mean(sub, signal)
    return Answer(f"No anomaly was flagged for {signal} on {day.date()}; it averaged "
                  f"{_fmt(val)} that day (within its normal range).", val)


def _answer(df: pd.DataFrame, question: str, prof: DatasetProfile) -> Answer:
    q = question.lower()
    signal = _pick_signal(question, prof)
    if signal is None:
        return Answer("I couldn't find a numeric column to analyse in this dataset.")

    # --- explain a specific day (e.g. "explain the anomaly on 2023-11-20") ---
    if prof.time_col is not None and re.search(
            r"\b(explain|why|anomal\w*|happen\w*|unusual|what)\b", q):
        yrs = _parse_dates(df[prof.time_col]).dt.year.dropna()
        default_year = int(yrs.max()) if len(yrs) else None
        day = _specific_date(question, default_year)
        if day is not None:
            return _explain_at_date(df, prof, signal, day)

    years = _years(question)

    compare = re.search(
        r"\b(higher|lower|hotter|colder|warmer|cooler|more|less|than|compar\w*|versus|vs)\b", q)

    # --- comparison between two years: "was 2020 higher than 2019?" ---
    if len(years) >= 2 and compare:
        a, b = years[0], years[1]
        ma = query.mean(_scope_year(df, prof.time_col, a), signal)
        mb = query.mean(_scope_year(df, prof.time_col, b), signal)
        if not (np.isnan(ma) or np.isnan(mb)):
            hi, lo = (a, b) if ma >= mb else (b, a)
            return Answer(f"{signal} was higher in {hi} (avg {_fmt(max(ma, mb))}) "
                          f"than in {lo} (avg {_fmt(min(ma, mb))}).", hi)

    # --- comparison between two months: "was March higher than April?" ---
    months = _months_in(question)
    if len(months) >= 2 and compare and prof.time_col is not None:
        (na, ia), (nb, ib) = months[0], months[1]
        ma = query.mean(_scope_month(df, prof.time_col, ia), signal)
        mb = query.mean(_scope_month(df, prof.time_col, ib), signal)
        if not (np.isnan(ma) or np.isnan(mb)):
            (hn, hv), (ln, lv) = ((na, ma), (nb, mb)) if ma >= mb else ((nb, mb), (na, ma))
            return Answer(f"{signal} was higher in {hn.title()} (avg {_fmt(hv)}) "
                          f"than in {ln.title()} (avg {_fmt(lv)}).", hn.title())

    # --- trend over time ---
    if re.search(r"\b(trend\w*|rising|falling|increasing|decreasing|declin\w*|growing|"
                 r"over time|over the years)\b", q):
        slope = query.trend(df, signal)
        direction = "rising" if slope > 0 else "falling" if slope < 0 else "flat"
        return Answer(f"{signal} is {direction} over time (slope {_fmt(slope)} per step).", slope)

    # --- scope to a named period (year / month / last month|week), then answer on that slice ---
    scoped, label = _resolve_period(question, df, prof.time_col)
    if label and len(scoped) == 0:
        return Answer(f"There's no data for that period{label.replace(' (in', ' (').rstrip('.')}.")
    if label:
        df = scoped

    ans = _dispatch(df, question, prof, signal)
    if label and ans.value is not None:
        ans.text = ans.text.rstrip(".") + label + "."
    return ans


def _dispatch(df: pd.DataFrame, question: str, prof: DatasetProfile, signal: str) -> Answer:
    q = question.lower()

    # --- anomalies ---
    if re.search(r"\b(anomal|unusual|weird|strange|outlier|odd)\w*", q):
        result = analyze(df)
        anoms = result.anomalies.get(signal, [])
        if not anoms:
            return Answer(f"No anomalies were found in {signal}.", 0)
        top = max(anoms, key=lambda a: a.zscore)
        return Answer(
            f"Found {len(anoms)} anomalies in {signal}. The most extreme: {top.explanation}",
            len(anoms),
            anomaly=top,
        )

    # --- threshold counts ---
    if re.search(r"\b(above|over|greater|more than|exceed)\w*", q):
        thr = _number(question)
        if thr is not None:
            n = query.count_over(df, signal, thr)
            return Answer(f"{signal} went above {_fmt(thr)} {n} times.", n)
    if re.search(r"\b(below|under|less than|fewer)\w*", q):
        thr = _number(question)
        if thr is not None:
            n = query.count_under(df, signal, thr)
            return Answer(f"{signal} went below {_fmt(thr)} {n} times.", n)

    # --- extremes ---
    if re.search(r"\b(highest|max|maximum|peak|hottest|largest|most)\w*", q):
        val = query.maximum(df, signal)
        when = query.when_extreme(df, prof.time_col, signal, "max") if prof.time_col else None
        return Answer(f"The highest {signal} was {_fmt(val)}"
                      + (f" on {when}." if when is not None else "."), val)
    if re.search(r"\b(lowest|min|minimum|coldest|smallest|least)\w*", q):
        val = query.minimum(df, signal)
        when = query.when_extreme(df, prof.time_col, signal, "min") if prof.time_col else None
        return Answer(f"The lowest {signal} was {_fmt(val)}"
                      + (f" on {when}." if when is not None else "."), val)

    # --- average ---
    if re.search(r"\b(average|mean|typical|usual)\b", q):
        val = query.mean(df, signal)
        return Answer(f"The average {signal} was {_fmt(val)}.", val)

    # --- correlation between signals ---
    if re.search(r"\bcorrelat\w*|\brelationship\b|\brelated\b", q):
        named = [s for s in prof.signals if s.lower() in q]
        if len(named) >= 2:
            c = query.correlation(df, named[0], named[1])
            return Answer(f"{named[0]} and {named[1]} have a correlation of {_fmt(c)} "
                          f"({_corr_word(c)}).", c)
        top = query.top_correlations(df, prof.signals, k=3)
        if not top:
            return Answer("There aren't two or more numeric signals to correlate in this dataset.")
        txt = "; ".join(f"{a} & {b}: {_fmt(c)}" for a, b, c in top)
        return Answer(f"Strongest correlations — {txt}.", top[0][2])

    # --- missing data / availability ---
    if re.search(r"\b(missing|gaps?|blanks?|incomplete|availab\w*|coverage)\b", q):
        parts = [f"{s}: {query.missing_count(df, s, prof.sentinels)} "
                 f"({100 * query.missing_count(df, s, prof.sentinels) / max(prof.n_rows, 1):.1f}%)"
                 for s in prof.signals]
        return Answer(f"Missing values — {', '.join(parts)}. They are filled by interpolation "
                      f"(following the local trend) before analysis.")

    # --- fallback ---
    return Answer(
        "I can answer questions like: “what was the highest <signal>?”, "
        "“what's the average?”, “how many times did it go above 30?”, "
        "or “were there any anomalies?”.",
        recognized=False,
    )


def _from_plan(df: pd.DataFrame, question: str, prof: DatasetProfile) -> Answer | None:
    """Answer a free-phrased question via the LLM's intent plan (numbers still from code)."""
    p = llm.plan(question, prof.signals)
    if not p or not prof.signals:
        return None
    signal = p.get("signal") if p.get("signal") in prof.signals else prof.signals[0]
    intent, val = p["intent"], p.get("value")

    if intent == "max":
        return _dispatch(df, "highest", prof, signal)
    if intent == "min":
        return _dispatch(df, "lowest", prof, signal)
    if intent == "mean":
        return _dispatch(df, "average", prof, signal)
    if intent == "anomaly":
        return _dispatch(df, "anomalies", prof, signal)
    if intent == "trend":
        slope = query.trend(df, signal)
        direction = "rising" if slope > 0 else "falling" if slope < 0 else "flat"
        return Answer(f"{signal} is {direction} over time (slope {_fmt(slope)} per step).", slope)
    if intent in ("count_above", "count_below") and isinstance(val, (int, float)):
        n = (query.count_over if intent == "count_above" else query.count_under)(df, signal, val)
        word = "above" if intent == "count_above" else "below"
        return Answer(f"{signal} went {word} {_fmt(val)} {n} times.", n)
    return None
