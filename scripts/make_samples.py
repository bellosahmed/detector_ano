"""Generate three *unrelated* sample datasets, each with a committed answer key.

Deterministic (seeded) so the answer keys always match the CSVs. The answer key lets
`verify.py` grade detection automatically (known anomaly positions = ground truth).

Run:  python3.12 scripts/make_samples.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent.parent / "samples"
OUT.mkdir(exist_ok=True)


def _write(name: str, df: pd.DataFrame, time_col: str, signal: str, anomaly_rows, freq: str):
    df.to_csv(OUT / f"{name}.csv", index=False)
    key = {
        "time_col": time_col,
        "signal": signal,
        "freq": freq,
        "anomaly_rows": sorted(int(i) for i in anomaly_rows),
    }
    (OUT / f"{name}.answer.json").write_text(json.dumps(key, indent=2))
    print(f"  {name}.csv  ({len(df)} rows, {len(anomaly_rows)} anomalies)")


def sensor():
    """IoT-style: hourly seasonal temperature + spikes + a -9999 sentinel."""
    rng = np.random.default_rng(1)
    n = 1000
    t = np.arange(n)
    v = 18 + 6 * np.sin(2 * np.pi * t / 24) + rng.normal(0, 0.4, n)
    anoms = rng.choice(n, size=15, replace=False)
    v[anoms] += rng.choice([-1, 1], 15) * rng.uniform(20, 30, 15)
    v[[100, 200, 300]] = -9999.0  # sentinels, NOT anomalies
    df = pd.DataFrame({
        "timestamp": pd.date_range("2021-01-01", periods=n, freq="h").astype(str),
        "temperature": v,
    })
    _write("sensor", df, "timestamp", "temperature", anoms, "h")


def finance():
    """Price-style: daily random walk with a few sharp shocks."""
    rng = np.random.default_rng(2)
    n = 600
    steps = rng.normal(0, 1, n)
    anoms = rng.choice(n, size=8, replace=False)
    steps[anoms] += rng.choice([-1, 1], 8) * rng.uniform(15, 25, 8)
    price = 100 + np.cumsum(steps)
    df = pd.DataFrame({
        "date": pd.date_range("2020-01-01", periods=n, freq="D").astype(str),
        "close": price,
    })
    _write("finance", df, "date", "close", anoms, "D")


def web_traffic():
    """Web-traffic-style: daily visit counts with weekly seasonality + outage drops."""
    rng = np.random.default_rng(3)
    n = 730
    t = np.arange(n)
    base = 5000 + 1500 * np.sin(2 * np.pi * t / 7) + rng.normal(0, 200, n)
    anoms = rng.choice(n, size=10, replace=False)
    base[anoms] *= rng.uniform(0.05, 0.2, 10)  # outages: sudden drops
    df = pd.DataFrame({
        "day": pd.date_range("2022-01-01", periods=n, freq="D").astype(str),
        "visits": base.round().astype(int),
    })
    _write("web_traffic", df, "day", "visits", anoms, "D")


if __name__ == "__main__":
    print("Generating sample datasets + answer keys:")
    sensor()
    finance()
    web_traffic()
    print(f"Done → {OUT}")
