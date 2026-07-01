"""Optional local-LLM layer (Ollama) — rewords computed answers into friendlier prose.

Hard rule of the whole project: the model never produces a number. It only rephrases text
that already contains code-computed figures. `faithful()` enforces this — if the reworded
text introduces a number that isn't in the computed facts, the rewording is rejected and the
exact computed answer is used instead. If no local model is running, everything falls back
silently to the computed answer, so the system works fully offline with or without an LLM.
"""
from __future__ import annotations

import json
import re
import urllib.error
import urllib.request

DEFAULT_HOST = "http://localhost:11434"
DEFAULT_MODEL = "llama3.2:3b"

_SYSTEM = (
    "Rewrite the finding below in warm, plain language a non-expert would understand. "
    "Use different wording from the original (do not copy it verbatim), keep it to one or two "
    "short sentences, and explain what it means in practice. "
    "Hard rules: keep every number EXACTLY as given; never add a number; never calculate; never "
    "add facts not in the finding. Output only the rewritten text."
)

_CAUSE_SYSTEM = (
    "You interpret an anomaly detected in a dataset. Using the computed facts and the timing/context, "
    "suggest one or two PLAUSIBLE reasons it may have happened — draw on the day of week, the date "
    "(holidays, seasons) and any related signals. These are hypotheses, not certainties: hedge with "
    "words like 'possibly', 'may' or 'could'. Hard rules: keep every number EXACTLY as given; invent "
    "no numbers; state nothing as fact. One or two short sentences."
)


def _numbers(text: str) -> list[float]:
    """Extract numeric values (ignoring thousands separators, dates handled loosely)."""
    out = []
    for token in re.findall(r"-?\d[\d,]*(?:\.\d+)?", text):
        try:
            out.append(float(token.replace(",", "")))
        except ValueError:
            pass
    return out


def faithful(facts: str, worded: str, rel_tol: float = 0.02) -> tuple[bool, list[float]]:
    """True if `worded` invents no number absent from `facts` (rounding tolerated).

    Dropping a number is fine; inventing one is not — that is the hallucination we guard against.
    """
    have = _numbers(facts)

    def known(n: float) -> bool:
        return any(abs(n - h) <= max(rel_tol * abs(h), 0.5) for h in have)

    invented = [n for n in _numbers(worded) if not known(n)]
    return (not invented), invented


def available(host: str = DEFAULT_HOST, timeout: float = 1.0) -> bool:
    """Whether a local Ollama server is reachable. Never raises."""
    try:
        with urllib.request.urlopen(f"{host}/api/tags", timeout=timeout) as resp:
            return resp.status == 200
    except (urllib.error.URLError, OSError, ValueError):
        return False


def _generate(prompt: str, model: str, host: str, timeout: float) -> str | None:
    """Low-level call to Ollama's /api/generate. Returns None on any failure."""
    payload = json.dumps({
        "model": model, "prompt": prompt, "stream": False,
        "options": {"temperature": 0.5, "num_predict": 140},
    }).encode()
    req = urllib.request.Request(f"{host}/api/generate", data=payload,
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            text = json.loads(resp.read()).get("response", "").strip()
    except (urllib.error.URLError, OSError, ValueError, json.JSONDecodeError):
        return None
    return text or None


def reword(facts: str, model: str = DEFAULT_MODEL, host: str = DEFAULT_HOST,
           timeout: float = 30.0) -> str | None:
    """Ask the local model to reword `facts`. Returns None on any failure (caller falls back)."""
    return _generate(f"{_SYSTEM}\n\nFinding: {facts}\n\nRewritten:", model, host, timeout)


_PLAN_INTENTS = {"max", "min", "mean", "count_above", "count_below", "anomaly", "trend"}

_PLAN_SYSTEM = (
    "Map the user's question about a dataset to a compact JSON plan. "
    f"Allowed intent values: {sorted(_PLAN_INTENTS)}. "
    "Fields: intent (required); signal (one of the given signals, or null); "
    "value (a number, only for count_above/count_below, else null). "
    'Output ONLY JSON, e.g. {"intent":"max","signal":null,"value":null}. No prose.'
)


def plan(question: str, signals: list[str], model: str = DEFAULT_MODEL,
         host: str = DEFAULT_HOST, timeout: float = 30.0) -> dict | None:
    """Ask the local model to classify a free-phrased question into a supported intent.

    Returns a validated {intent, signal, value} dict, or None (unavailable / unparseable).
    The model chooses *which* operation; the numbers are still computed by code afterwards.
    """
    if not available(host):
        return None
    prompt = f"{_PLAN_SYSTEM}\nSignals: {signals}\nQuestion: {question}\nJSON:"
    raw = _generate(prompt, model, host, timeout)
    if not raw:
        return None
    match = re.search(r"\{.*\}", raw, re.S)
    if not match:
        return None
    try:
        p = json.loads(match.group())
    except json.JSONDecodeError:
        return None
    return p if p.get("intent") in _PLAN_INTENTS else None


def suggest_cause(facts: str, context: str = "", model: str = DEFAULT_MODEL,
                  host: str = DEFAULT_HOST, timeout: float = 30.0) -> str | None:
    """Hypothesise WHY an anomaly happened, hedged. Returns None if unavailable or unfaithful.

    This is the one place the model adds interpretation beyond the numbers — so the output is
    still number-checked (no invented figures) and callers must present it as an AI hypothesis.
    """
    if not available(host):
        return None
    prompt = f"{_CAUSE_SYSTEM}\n\nFacts: {facts}\nContext: {context}\n\nPossible cause:"
    cause = _generate(prompt, model, host, timeout)
    if cause and faithful(facts + " " + context, cause)[0]:
        return cause
    return None


def explain(facts: str, model: str = DEFAULT_MODEL, host: str = DEFAULT_HOST) -> str:
    """Reword `facts` if a faithful rewording is available; otherwise return `facts` unchanged."""
    if not available(host):
        return facts
    worded = reword(facts, model=model, host=host)
    if worded and faithful(facts, worded)[0]:
        return worded
    return facts
