"""Convenience launcher so you can run without installing:

    python3.12 scripts/analyze.py samples/web_traffic.csv
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from anomaly_detector.cli import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
