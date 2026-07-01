"""Precision tuning sweep: threshold x combine-mode, on synthetic samples + NAB.

Reports recall and false-alarm rate so we can pick a justified operating point
(not an arbitrary threshold). Read-only — changes nothing.

    python3.12 benchmarks/tune.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from anomaly_detector.core import analyze  # noqa: E402

SAMPLES = ROOT / "samples"
NAB_CSV = ROOT / "benchmarks" / "data" / "nyc_taxi.csv"
NAB_LABELS = ROOT / "benchmarks" / "data" / "combined_windows.json"


def _sample_scores(name, threshold, combine):
    df = pd.read_csv(SAMPLES / f"{name}.csv")
    key = json.loads((SAMPLES / f"{name}.answer.json").read_text())
    res = analyze(df, threshold=threshold, combine=combine)
    sig = key["signal"]
    pos = {t: i for i, t in enumerate(df[res.profile.time_col].tolist())}
    flagged = {pos[a.time] for a in res.anomalies[sig] if a.time in pos}
    inj = key["anomaly_rows"]
    tp = sum(1 for r in inj if r in flagged or (r + 1) in flagged)
    recall = tp / len(inj)
    fp_rate = (len(flagged) - tp) / len(df)
    return recall, fp_rate


def _nab_scores(threshold, combine):
    if not NAB_CSV.exists():
        return None
    df = pd.read_csv(NAB_CSV)
    wins = [(pd.Timestamp(a), pd.Timestamp(b))
            for a, b in json.loads(NAB_LABELS.read_text())["realKnownCause/nyc_taxi.csv"]]
    res = analyze(df, threshold=threshold, combine=combine)
    flagged = [pd.Timestamp(a.time) for a in res.anomalies[res.profile.signals[0]]]
    detected = sum(any(a <= t <= b for t in flagged) for a, b in wins)
    in_win = sum(any(a <= t <= b for a, b in wins) for t in flagged)
    return detected / len(wins), (len(flagged) - in_win) / len(df)


def main() -> int:
    print(f"{'combine':7} {'thr':>4} | "
          f"{'sensor R/FP':>14} {'finance R/FP':>14} {'web R/FP':>14} {'NAB R/FP':>14}")
    print("-" * 78)
    for combine in ("any", "all"):
        for thr in (3.0, 4.0, 5.0, 6.0):
            cells = []
            for name in ("sensor", "finance", "web_traffic"):
                r, fp = _sample_scores(name, thr, combine)
                cells.append(f"{r:.2f}/{fp*100:4.1f}%")
            nab = _nab_scores(thr, combine)
            cells.append(f"{nab[0]:.2f}/{nab[1]*100:4.1f}%" if nab else "  (no data)  ")
            print(f"{combine:7} {thr:>4} | "
                  f"{cells[0]:>14} {cells[1]:>14} {cells[2]:>14} {cells[3]:>14}")
        print("-" * 78)
    print("R = recall (higher better) · FP = false-alarm rate (lower better)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
