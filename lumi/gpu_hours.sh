#!/bin/bash -l
# Report GPU (GCD) hours for the retraining job.
#   usage: bash lumi/gpu_hours.sh <ARRAY_JOBID> [RETRAINED_DIR]
# Prints two numbers:
#   1) billed GCD-hours from sacct (Elapsed x AllocGPUS, incl. load/queue-inside-job overhead)
#   2) pure training seconds summed from each model's train_metrics.json (compute only)
set -euo pipefail

JOBID="${1:?usage: gpu_hours.sh <ARRAY_JOBID> [RETRAINED_DIR]}"
RETRAINED_DIR="${2:-/scratch/project_465002928/$(whoami)/retrained}"

echo "== billed GCD-hours (sacct, job $JOBID) =="
# Count only the array-task main lines (JobID has no '.') that carry a gres/gpu allocation,
# so per-step sub-lines (.batch/.extern) are not double-counted.
sacct -j "$JOBID" \
      --format=JobID,Elapsed,ElapsedRaw,AllocTRES%60,State \
      --parsable2 --noheader \
| awk -F'|' '
    $1 !~ /\./ && $4 ~ /gres\/gpu=/ {
        match($4, /gres\/gpu=([0-9]+)/, a); g = a[1] + 0;
        total += ($3 + 0) * g; n++;
    }
    END {
        printf "  tasks counted = %d\n", n;
        printf "  total GCD-seconds = %d\n", total;
        printf "  total GCD-hours   = %.2f\n", total / 3600;
    }'

echo "== pure train_runtime sum (train_metrics.json in $RETRAINED_DIR) =="
python3 - "$RETRAINED_DIR" <<'PY'
import glob, json, os, sys
root = sys.argv[1]
files = sorted(glob.glob(os.path.join(root, "*", "train_metrics.json")))
tot = 0.0
for f in files:
    d = json.load(open(f))
    tot += float(d.get("train_runtime_s", 0.0))
print(f"  models = {len(files)}")
print(f"  sum train_runtime = {tot:.0f} s = {tot / 3600:.2f} h")
PY
