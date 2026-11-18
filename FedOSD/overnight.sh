#!/bin/bash
# Overnight chain (server, from FedOSD/, venv active):  nohup bash overnight.sh 2 3 4 5 > overnight.out 2>&1 &
# 1) wait for any running sweep  2) finish the sweep (pat redo)  3) apply the pre-registered rule
# 4) only if the decision is complete: final seeds 31-33 + bundles + audit.
set -u
cd "$(dirname "$0")"
mkdir -p results_soft logs_final
echo "[$(date)] waiting for running sweeps"
while pgrep -u "$USER" -f run_soft_sweep.py >/dev/null; do sleep 60; done
echo "[$(date)] sweep (skips finished jobs)"
python -u run_soft_sweep.py --gpus "$@" --per_gpu 2 > logs_soft_patch4/sweep_overnight.out 2>&1 \
  || { echo "sweep FAILED -> see logs_soft_patch4/sweep_overnight.out"; exit 1; }
echo "[$(date)] analysis"
python analyze_sweep.py > results_soft/analyze.txt 2>&1; rc=$?
cat results_soft/analyze.txt
[ $rc -eq 0 ] || { echo "analysis reported PROBLEMS -> final runs NOT started"; exit 1; }
echo "[$(date)] final runs"
python -u run_final.py --gpus "$@" --per_gpu 2 > logs_final/final.out 2>&1; echo "run_final rc=$?"
echo "[$(date)] done"
