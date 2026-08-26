#!/usr/bin/env bash
# Driver — one JULIA PROCESS per thread count, so each point gets exactly k threads.
# See the header of scaling_threads.jl for why a single session would measure placement
# luck instead of scaling.
#
#   bash run.sh              # k = 1..nproc (capped at 16)
#   bash run.sh 1 2 4 8 16   # explicit list
#
# Appends to results/threads_scaling.csv (removed first, so each run is clean).

set -euo pipefail
cd "$(dirname "$0")"

if [ $# -gt 0 ]; then
  KS=("$@")
else
  MAX=$(nproc)
  [ "$MAX" -gt 16 ] && MAX=16
  KS=($(seq 1 "$MAX"))
fi

mkdir -p results
rm -f results/threads_scaling.csv

echo "Thread-scaling experiment — one process per point: ${KS[*]}"
for k in "${KS[@]}"; do
  julia --threads="$k" --startup-file=no scaling_threads.jl
done

echo
echo "Done -> results/threads_scaling.csv"
echo "Plot it:  python3 plot_threads.py"
