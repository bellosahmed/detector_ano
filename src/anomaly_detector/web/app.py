"""Flask web UI: upload any CSV, see the anomalies and ask nothing of the numbers.

The web layer is deliberately thin — it calls anomaly_detector.core.analyze (the single
source of truth) and shapes the result for the browser. No detection logic lives here.

Run:  python3.12 -m anomaly_detector.web.app   →   http://localhost:3020
"""
from __future__ import annotations

import pandas as pd
from flask import Flask, jsonify, render_template, request

from anomaly_detector.core import analyze
from anomaly_detector.interface import ask
from anomaly_detector.loader import read_table

MAX_CHART_POINTS = 1500


def _downsample(times: list, values: list, cap: int = MAX_CHART_POINTS):
    """Thin a long series for plotting (anomalies are sent separately, so none are lost)."""
    n = len(values)
    if n <= cap:
        return times, values
    step = n // cap + 1
    return times[::step], values[::step]


def _payload(df: pd.DataFrame, shape: bool = False, time_col: str | None = None):
    result = analyze(df, shape=shape, time_col=time_col)
    prof = result.profile
    tcol = prof.time_col

    series = {}
    anomalies = {}
    for signal in prof.signals:
        t = [str(x) for x in df[tcol].tolist()] if tcol else list(range(len(df)))
        v = pd.to_numeric(df[signal], errors="coerce").tolist()
        st, sv = _downsample(t, v)
        series[signal] = {"t": st, "v": sv}
        anomalies[signal] = [
            {"time": str(a.time), "value": a.value, "zscore": round(a.zscore, 2),
             "explanation": a.explanation, "verdict": a.verdict, "co_moving": a.co_moving,
             "kind": a.kind}
            for a in result.anomalies[signal]
        ]

    return {
        "profile": {
            "time_col": prof.time_col,
            "freq": prof.freq,
            "signals": prof.signals,
            "sentinels": prof.sentinels,
            "n_rows": prof.n_rows,
        },
        "columns": [str(c) for c in df.columns],   # for the "correct the time column" dropdown
        "series": series,
        "anomalies": anomalies,
    }


def create_app() -> Flask:
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = 200 * 1024 * 1024  # 200 MB uploads

    @app.errorhandler(413)
    def too_large(_e):
        return jsonify({"error": "File too large (200 MB limit). Downsample or split it, "
                                 "or use the command line: python -m anomaly_detector.cli file.csv"}), 413

    @app.get("/")
    def index():
        return render_template("index.html")

    @app.post("/api/analyze")
    def api_analyze():
        file = request.files.get("file")
        if file is None or not file.filename:
            return jsonify({"error": "No file uploaded."}), 400
        try:
            df = read_table(file.stream, name=file.filename)
        except Exception as exc:  # malformed file
            return jsonify({"error": f"Could not read the file: {exc}"}), 400
        if df.empty:
            return jsonify({"error": "The file has no rows."}), 400
        shape = request.form.get("shape") in ("1", "true", "on", "yes")
        time_col = request.form.get("time_col") or None
        try:
            return jsonify(_payload(df, shape=shape, time_col=time_col))
        except Exception as exc:
            return jsonify({"error": f"Could not analyze: {exc}"}), 400

    @app.post("/api/ask")
    def api_ask():
        file = request.files.get("file")
        question = (request.form.get("question") or "").strip()
        if file is None or not file.filename:
            return jsonify({"error": "No file uploaded."}), 400
        if not question:
            return jsonify({"error": "No question asked."}), 400
        use_llm = request.form.get("use_llm") in ("1", "true", "on", "yes")
        try:
            df = read_table(file.stream, name=file.filename)
            return jsonify({"answer": ask(df, question, use_llm=use_llm).text})
        except Exception as exc:
            return jsonify({"error": f"Could not answer: {exc}"}), 400

    return app


def main() -> None:
    import os
    host = os.environ.get("HOST", "127.0.0.1")   # set HOST=0.0.0.0 in Docker
    port = int(os.environ.get("PORT", "3020"))
    create_app().run(host=host, port=port, debug=False, threaded=True)


if __name__ == "__main__":
    main()
