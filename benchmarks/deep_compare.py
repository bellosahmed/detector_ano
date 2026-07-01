"""Detector comparison on labelled synthetic data — the ensemble vs established baselines.

Primary metric: **PR-AUC** (average precision), which is threshold-independent and robust to the
heavy class imbalance of anomaly detection (~1-2% positives), unlike accuracy or ROC-AUC. Compares:
  * the classical robust-z ensemble (this project's default),
  * Isolation Forest (an established off-the-shelf baseline),
  * an LSTM autoencoder (deep baseline).

    python3.12 benchmarks/deep_compare.py

Requires torch (optional) for the LSTM column; skips it if absent.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from anomaly_detector import detect               # noqa: E402
from anomaly_detector.core import _period_from_freq  # noqa: E402
from anomaly_detector.profile import profile         # noqa: E402

SAMPLES = ROOT / "samples"

try:
    from anomaly_detector import deep
    _HAS_DEEP = True
except Exception:
    _HAS_DEEP = False


def _labels(df, key) -> pd.Series:
    truth = pd.Series(False, index=range(len(df)))
    truth.iloc[key["anomaly_rows"]] = True
    return truth


def compare_sample(name: str) -> dict:
    df = pd.read_csv(SAMPLES / f"{name}.csv")
    key = json.loads((SAMPLES / f"{name}.answer.json").read_text())
    prof = profile(df)
    signal = key["signal"]
    clean = pd.to_numeric(df[signal], errors="coerce").mask(lambda x: x.isin(prof.sentinels))
    period = _period_from_freq(prof.freq)
    truth = _labels(df, key)

    out = {
        "ensemble": detect.pr_auc(truth, detect.anomaly_scores(clean, period=period)),
        "iforest": detect.pr_auc(truth, detect.isolation_forest_scores(clean)),
    }
    if _HAS_DEEP:
        out["lstm_ae"] = detect.pr_auc(
            truth, deep.lstm_autoencoder_scores(clean, window=max(16, period or 24),
                                                hidden=16, epochs=30, seed=0))
    return out


def main() -> int:
    cols = ["ensemble", "iforest"] + (["lstm_ae"] if _HAS_DEEP else [])
    print("PR-AUC (average precision) — higher is better; robust to class imbalance\n")
    print(f"{'dataset':14} " + " ".join(f"{c:>10}" for c in cols))
    print("-" * (14 + 11 * len(cols)))
    for name in ("sensor", "finance", "web_traffic"):
        r = compare_sample(name)
        print(f"{name:14} " + " ".join(f"{r[c]:>10.3f}" for c in cols))
    print("\n(ensemble = this project's default; iforest = Isolation Forest baseline"
          + ("; lstm_ae = LSTM autoencoder)" if _HAS_DEEP else "; LSTM skipped — no torch)"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
