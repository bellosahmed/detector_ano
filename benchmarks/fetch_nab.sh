#!/usr/bin/env bash
# Download NAB realKnownCause datasets + labels (git-ignored; not committed).
# Usage:  bash benchmarks/fetch_nab.sh
set -euo pipefail

DIR="$(cd "$(dirname "$0")" && pwd)/data"
mkdir -p "$DIR"
BASE="https://raw.githubusercontent.com/numenta/NAB/master"

DATASETS=(
  ambient_temperature_system_failure
  cpu_utilization_asg_misconfiguration
  ec2_request_latency_system_failure
  machine_temperature_system_failure
  nyc_taxi
  rogue_agent_key_hold
  rogue_agent_key_updown
)

for name in "${DATASETS[@]}"; do
  curl -sSf -o "$DIR/$name.csv" "$BASE/data/realKnownCause/$name.csv"
  echo "downloaded $name"
done
curl -sSf -o "$DIR/combined_windows.json" "$BASE/labels/combined_windows.json"

echo "Done → $DIR"
echo "Now run:  python3.12 benchmarks/nab_eval.py"
