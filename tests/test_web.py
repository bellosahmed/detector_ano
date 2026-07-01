"""Tests for the web app's analyze endpoint.

The endpoint must return the SAME numbers as core.analyze (the web layer holds no
analytics logic of its own) and correctly surface an uploaded file's anomalies.
"""
import io

import numpy as np
import pandas as pd

from anomaly_detector.web.app import create_app


def _csv_with_spike() -> bytes:
    rng = np.random.default_rng(0)
    n = 300
    t = np.arange(n)
    v = 20 + 5 * np.sin(2 * np.pi * t / 24) + rng.normal(0, 0.5, n)
    v[150] = 200.0
    df = pd.DataFrame({
        "time": pd.date_range("2020-01-01", periods=n, freq="h").astype(str),
        "value": v,
    })
    return df.to_csv(index=False).encode()


def _client():
    return create_app().test_client()


def test_index_page_loads():
    resp = _client().get("/")
    assert resp.status_code == 200
    assert b"<html" in resp.data.lower()


def test_analyze_endpoint_returns_profile_and_anomalies():
    resp = _client().post(
        "/api/analyze",
        data={"file": (io.BytesIO(_csv_with_spike()), "data.csv")},
        content_type="multipart/form-data",
    )
    assert resp.status_code == 200
    payload = resp.get_json()

    assert payload["profile"]["time_col"] == "time"
    assert payload["profile"]["signals"] == ["value"]

    anomalies = payload["anomalies"]["value"]
    assert len(anomalies) >= 1
    # the injected spike (row 150) must be reported
    assert any(a["value"] == 200.0 for a in anomalies)


def test_analyze_endpoint_rejects_missing_file():
    resp = _client().post("/api/analyze", data={}, content_type="multipart/form-data")
    assert resp.status_code == 400


def test_analyze_endpoint_honours_time_col_override():
    n = 60
    df = pd.DataFrame({
        "time": pd.date_range("2020-01-01", periods=n, freq="h").astype(str),
        "alt_time": pd.date_range("1990-01-01", periods=n, freq="D").astype(str),
        "value": np.linspace(0, 1, n),
    })
    csv = df.to_csv(index=False).encode()
    resp = _client().post(
        "/api/analyze",
        data={"file": (io.BytesIO(csv), "d.csv"), "time_col": "alt_time"},
        content_type="multipart/form-data",
    )
    assert resp.status_code == 200
    payload = resp.get_json()
    assert payload["profile"]["time_col"] == "alt_time"
    assert "alt_time" in payload["columns"]


def test_ask_endpoint_answers_a_question():
    resp = _client().post(
        "/api/ask",
        data={"file": (io.BytesIO(_csv_with_spike()), "data.csv"),
              "question": "what was the highest value?"},
        content_type="multipart/form-data",
    )
    assert resp.status_code == 200
    answer = resp.get_json()["answer"]
    assert "highest" in answer.lower() and "200" in answer
