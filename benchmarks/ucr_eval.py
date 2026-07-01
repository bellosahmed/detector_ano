"""Evaluate on the UCR Anomaly Archive (Wu & Keogh, KDD 2021) — a rigorous cross-check.

Each file is one univariate series; the filename encodes the train/test split and the single
labelled anomaly's [start, end] (1-indexed). Protocol: score every point, predict the single
most-anomalous location in the TEST region (the training region is guaranteed normal), and count
it correct if that point falls inside the labelled window (strict) — plus a ±100 tolerance, as
commonly reported. Accuracy = fraction of series localized correctly.

    python3.12 benchmarks/ucr_eval.py            # all 250
    python3.12 benchmarks/ucr_eval.py 40         # first 40 (quick)

Data is fetched/extracted by fetch_ucr.sh (git-ignored, not committed).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from anomaly_detector import detect  # noqa: E402

HERE = Path(__file__).resolve().parent
FULL = (HERE / "data" / "ucr" / "AnomalyDatasets_2021" /
        "UCR_TimeSeriesAnomalyDatasets2021" / "FilesAreInHere" / "UCR_Anomaly_FullData")


def _parse(name: str) -> tuple[int, int, int]:
    p = name[:-4].split("_")
    return int(p[-3]), int(p[-2]), int(p[-1])   # train_split, start, end (1-indexed)


def evaluate(path: Path, tol: int = 100, detector: str = "ensemble") -> tuple[bool, bool]:
    train, start, end = _parse(path.name)
    # UCR files are one-value-per-line OR all values on one whitespace-separated line
    values = np.array(path.read_text().split(), dtype=float)

    if detector == "mp":
        scores = detect.matrix_profile_multiscale(
            pd.Series(values), windows=(50, 100, 200, 400)).to_numpy().copy()
    else:
        scores = detect.anomaly_scores(pd.Series(values), period=None).to_numpy().copy()
    scores[:train] = -np.inf                     # anomaly is in the test region
    pred = int(np.argmax(scores)) + 1            # back to 1-indexed

    strict = start <= pred <= end
    lenient = (start - tol) <= pred <= (end + tol)
    return strict, lenient


def main(argv: list[str]) -> int:
    detector = "mp" if "mp" in argv else "ensemble"
    nums = [a for a in argv if a.isdigit()]
    files = sorted(FULL.glob("*.txt"))
    if nums:
        files = files[: int(nums[0])]
    if not files:
        print("No UCR data found — run: bash benchmarks/fetch_ucr.sh")
        return 1

    s_hits = l_hits = 0
    for f in files:
        try:
            strict, lenient = evaluate(f, detector=detector)
        except Exception as exc:
            print(f"  skip {f.name[:40]}: {type(exc).__name__}")
            continue
        s_hits += strict
        l_hits += lenient

    n = len(files)
    print("=" * 60)
    print(f"UCR Anomaly Archive — {n} series  ·  detector = {detector}")
    print("=" * 60)
    print(f"strict  (pred inside labelled window) : {s_hits}/{n} = {s_hits/n:.2f}")
    print(f"±100 pt tolerance                     : {l_hits}/{n} = {l_hits/n:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
