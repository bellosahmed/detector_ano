"""Evaluate the detector on NAB's realKnownCause datasets (real, labelled anomalies).

NAB labels anomalies as time *windows* around documented real events. Window-level scoring:
a window counts as detected if the detector flags any point inside it. Flags outside every
window are false alarms (some may be genuine unlabelled anomalies — NAB labels sparsely).

    python3.12 benchmarks/nab_eval.py
    python3.12 benchmarks/nab_eval.py nyc_taxi     # a single dataset

Data is downloaded by benchmarks/fetch_nab.sh (git-ignored, not committed).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from anomaly_detector.core import analyze  # noqa: E402

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
LABELS = DATA / "combined_windows.json"

DATASETS = [
    "ambient_temperature_system_failure",
    "cpu_utilization_asg_misconfiguration",
    "ec2_request_latency_system_failure",
    "machine_temperature_system_failure",
    "nyc_taxi",
    "rogue_agent_key_hold",
    "rogue_agent_key_updown",
]


def evaluate(name: str, labels: dict) -> tuple[int, int, float] | None:
    csv = DATA / f"{name}.csv"
    if not csv.exists():
        return None
    df = pd.read_csv(csv)
    wins = [(pd.Timestamp(a), pd.Timestamp(b)) for a, b in labels[f"realKnownCause/{name}.csv"]]

    result = analyze(df)
    prof = result.profile
    flagged = [pd.Timestamp(a.time) for a in result.anomalies.get(prof.signals[0], [])]

    detected = sum(any(a <= t <= b for t in flagged) for a, b in wins)
    in_window = sum(any(a <= t <= b for a, b in wins) for t in flagged)
    fp_rate = (len(flagged) - in_window) / prof.n_rows if prof.n_rows else 0.0
    return detected, len(wins), fp_rate


def main(argv: list[str]) -> int:
    labels = json.loads(LABELS.read_text())
    names = [argv[0]] if argv else DATASETS

    print("=" * 68)
    print("NAB realKnownCause — real labelled benchmark")
    print("=" * 68)
    print(f"{'dataset':40} {'windows':>9} {'FP rate':>9}")
    print("-" * 68)

    tot_det = tot_win = 0
    fp_rates = []
    for name in names:
        res = evaluate(name, labels)
        if res is None:
            print(f"{name:40} {'(no data)':>9}")
            continue
        det, nwin, fp = res
        tot_det += det
        tot_win += nwin
        fp_rates.append(fp)
        print(f"{name[:40]:40} {f'{det}/{nwin}':>9} {fp*100:>8.1f}%")

    print("-" * 68)
    if tot_win:
        mean_fp = sum(fp_rates) / len(fp_rates)
        print(f"{'TOTAL':40} {f'{tot_det}/{tot_win}':>9} {mean_fp*100:>8.1f}%")
        print(f"\nwindow recall = {tot_det}/{tot_win} = {tot_det/tot_win:.2f} across "
              f"{len(fp_rates)} real datasets; mean false-alarm rate {mean_fp*100:.1f}%.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
