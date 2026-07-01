"""Tests for the optional local-LLM layer.

The LLM may only *reword* computed answers. The safety net is `faithful()`: a reworded
answer that introduces a number not in the computed facts is rejected. These tests need
no running model — they check the guardrail and the graceful fallback.
"""
from anomaly_detector import interface, llm


def test_faithful_accepts_rewording_that_keeps_the_numbers():
    facts = "The highest visits was 7,127 on 2022-01-10."
    worded = "Visits peaked at 7,127 on 2022-01-10 — the busiest moment in the data."
    ok, invented = llm.faithful(facts, worded)
    assert ok and invented == []


def test_faithful_rejects_an_invented_number():
    facts = "The highest visits was 7,127 on 2022-01-10."
    worded = "Visits peaked at 9,999, about 40% above normal."
    ok, invented = llm.faithful(facts, worded)
    assert not ok
    assert any(abs(x - 9999) < 1 for x in invented)


def test_faithful_allows_dropping_a_number():
    facts = "visits was 544 at 2023-11-20 — 88% below the expected 4,423."
    worded = "Visits fell to 544 on 2023-11-20, far below normal."
    ok, invented = llm.faithful(facts, worded)
    assert ok  # dropping the 88% / 4,423 is fine; only inventing is a problem


def test_available_is_false_when_no_server():
    # nothing is listening here → must report unavailable, never raise
    assert llm.available(host="http://127.0.0.1:1", timeout=0.2) is False


def test_free_phrase_falls_back_to_the_llm_planner(monkeypatch):
    import numpy as np, pandas as pd
    df = pd.DataFrame({"time": pd.date_range("2020-01-01", periods=50, freq="h"),
                       "value": np.arange(50.0)})
    monkeypatch.setattr(llm, "available", lambda *a, **k: True)
    # keyword parsing won't catch this phrasing; the (mocked) planner maps it to "max"
    monkeypatch.setattr(llm, "plan", lambda *a, **k: {"intent": "max", "signal": "value", "value": None})
    ans = interface.ask(df, "gimme the top reading please", use_llm=True)
    assert ans.value == 49.0          # number still computed by code


def test_plan_returns_none_when_unavailable(monkeypatch):
    monkeypatch.setattr(llm, "available", lambda *a, **k: False)
    assert llm.plan("what is the highest?", ["value"]) is None


def test_suggest_cause_returns_none_when_unavailable(monkeypatch):
    monkeypatch.setattr(llm, "available", lambda *a, **k: False)
    assert llm.suggest_cause("visits was 544 on 2023-11-20 — 88% below expected.") is None


def test_ask_with_llm_falls_back_to_computed_answer_when_unavailable(monkeypatch):
    import numpy as np, pandas as pd
    monkeypatch.setattr(llm, "available", lambda *a, **k: False)
    df = pd.DataFrame({"time": pd.date_range("2020-01-01", periods=50, freq="h"),
                       "value": np.arange(50.0)})
    ans = interface.ask(df, "what is the highest value?", use_llm=True)
    assert ans.value == 49.0            # number still exact, from code
    assert "49" in ans.text
