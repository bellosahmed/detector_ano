#!/usr/bin/env bash
# Run the benchmarks and save their output as committed artifacts in benchmarks/results/.
# NAB/UCR sections need their datasets fetched first (fetch_nab.sh / fetch_ucr.sh).
#   bash scripts/run_benchmarks.sh
set -uo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY="${PY:-python3.12}"
OUT="$ROOT/benchmarks/results"
mkdir -p "$OUT"

run() { echo "→ $1"; "$PY" "$ROOT/benchmarks/$2" > "$OUT/$3" 2>&1 && echo "  wrote results/$3"; }

"$PY" "$ROOT/scripts/verify.py" > "$OUT/verify.txt" 2>&1; echo "  wrote results/verify.txt"
run "threshold tuning"      tune.py         tune.txt
run "classical vs deep"     deep_compare.py deep_compare.txt

if [ -f "$ROOT/benchmarks/data/nyc_taxi.csv" ]; then
  run "NAB (7 datasets)"    nab_eval.py     nab.txt
else
  echo "→ NAB skipped (run: bash benchmarks/fetch_nab.sh)"
fi

if [ -d "$ROOT/benchmarks/data/ucr" ]; then
  echo "→ UCR (ensemble)"; "$PY" "$ROOT/benchmarks/ucr_eval.py" | grep -vE '^  skip' > "$OUT/ucr.txt" 2>&1
  echo "  wrote results/ucr.txt"
else
  echo "→ UCR skipped (run: bash benchmarks/fetch_ucr.sh)"
fi

echo "Done → $OUT"
