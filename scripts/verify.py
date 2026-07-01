"""One-command verification gate. Exits non-zero on any failure.

    python3.12 scripts/verify.py

Checks (this project's own, not inherited):
  1. Query engine == pandas exactly (the truth layer).
  2. Sentinels are never reported as anomalies.
  3. Detection recovers the KNOWN injected anomalies in each sample dataset
     (graded against the committed answer keys) — reports the confusion matrix.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from anomaly_detector import detect, query          # noqa: E402
from anomaly_detector.core import analyze            # noqa: E402

SAMPLES = ROOT / "samples"

_passed = 0
_failed = 0


def check(name: str, ok: bool, detail: str = "") -> None:
    global _passed, _failed
    mark = "✓" if ok else "✗"
    print(f"  {mark} {name}" + (f"  — {detail}" if detail else ""))
    if ok:
        _passed += 1
    else:
        _failed += 1


def verify_query_truth_layer() -> None:
    print("Query engine == pandas:")
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        "time": pd.date_range("2020-01-01", periods=500, freq="h"),
        "value": rng.normal(0, 1, 500),
    })
    check("maximum", query.maximum(df, "value") == df["value"].max())
    check("minimum", query.minimum(df, "value") == df["value"].min())
    check("mean", query.mean(df, "value") == df["value"].mean())
    check("count_over", query.count_over(df, "value", 0.0) == int((df["value"] > 0).sum()))


def verify_sample(name: str) -> None:
    df = pd.read_csv(SAMPLES / f"{name}.csv")
    key = json.loads((SAMPLES / f"{name}.answer.json").read_text())
    signal = key["signal"]

    result = analyze(df)
    prof = result.profile

    # map flagged timestamps back to row positions
    pos_of = {t: i for i, t in enumerate(df[prof.time_col].tolist())}
    flagged_rows = {pos_of[a.time] for a in result.anomalies[signal] if a.time in pos_of}

    truth = pd.Series(False, index=range(len(df)))
    truth.iloc[key["anomaly_rows"]] = True
    pred = pd.Series(False, index=range(len(df)))
    for r in flagged_rows:
        pred.iloc[r] = True

    cm = detect.confusion(truth, pred)

    print(f"\nSample '{name}'  ({len(df)} rows, {len(key['anomaly_rows'])} injected):")
    check("profile finds the time column", prof.time_col == key["time_col"],
          f"{prof.time_col}")
    # sentinels must never be reported as anomalies
    flagged_values = [a.value for a in result.anomalies[signal]]
    check("no sentinel reported as anomaly",
          all(s not in flagged_values for s in prof.sentinels),
          f"sentinels={prof.sentinels}")
    check("recall >= 0.80 on injected anomalies", cm.recall >= 0.80,
          f"P={cm.precision:.2f} R={cm.recall:.2f} F1={cm.f1:.2f} "
          f"(TP={cm.tp} FP={cm.fp} FN={cm.fn} TN={cm.tn})")


def main() -> int:
    print("=" * 64)
    print("detector_ano — verification")
    print("=" * 64)
    verify_query_truth_layer()
    for name in ("sensor", "finance", "web_traffic"):
        verify_sample(name)
    print("\n" + "-" * 64)
    print(f"{_passed} passed, {_failed} failed")
    return 1 if _failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
