#!/usr/bin/env bash
# Download + extract the UCR Anomaly Archive (Wu & Keogh, KDD 2021). ~184 MB.
# Git-ignored; not committed. Usage: bash benchmarks/fetch_ucr.sh
set -euo pipefail

DIR="$(cd "$(dirname "$0")" && pwd)/data"
mkdir -p "$DIR/ucr"
URL="https://www.cs.ucr.edu/~eamonn/time_series_data_2018/UCR_TimeSeriesAnomalyDatasets2021.zip"

echo "downloading UCR archive (~184 MB)…"
curl -sSf -o "$DIR/ucr.zip" "$URL"
unzip -oq "$DIR/ucr.zip" -d "$DIR/ucr"
echo "Done. Now run:  python3.12 benchmarks/ucr_eval.py"
