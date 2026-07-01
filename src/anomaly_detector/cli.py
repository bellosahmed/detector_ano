"""Command-line entry point: point it at ANY CSV and it profiles + detects anomalies.

    python3.12 -m anomaly_detector.cli samples/sensor.csv
"""
from __future__ import annotations

import argparse

from .core import analyze
from .loader import read_table


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Detect anomalies in any time-series CSV or Excel file.")
    ap.add_argument("csv", help="path to a CSV or Excel (.xlsx) file")
    ap.add_argument("--threshold", type=float, default=5.0, help="robust z-score threshold")
    ap.add_argument("--top", type=int, default=10, help="how many anomalies to show per signal")
    ap.add_argument("--shape", action="store_true",
                    help="also run the Matrix Profile shape scan (needs the 'shape' extra)")
    args = ap.parse_args(argv)

    df = read_table(args.csv)
    result = analyze(df, threshold=args.threshold, shape=args.shape)
    p = result.profile

    print(f"\nDataset: {args.csv}")
    print(f"  rows        : {p.n_rows}")
    print(f"  time column : {p.time_col}")
    print(f"  frequency   : {p.freq}")
    print(f"  signals     : {', '.join(p.signals) or '(none)'}")
    if p.sentinels:
        print(f"  sentinels   : {p.sentinels} (treated as missing, not anomalies)")

    for signal, anomalies in result.anomalies.items():
        print(f"\nAnomalies in '{signal}': {len(anomalies)} found")
        for a in sorted(anomalies, key=lambda x: x.zscore, reverse=True)[: args.top]:
            tag = "[shape] " if a.kind == "shape" else ""
            print(f"  {tag}{a.time}   value={a.value:<12.3f} z={a.zscore:.1f}")
            if a.explanation:
                print(f"      ↳ {a.explanation}")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
