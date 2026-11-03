#!/usr/bin/env python3
"""
run_r1_refinement.py
Runs R1 seeds with Hard-scaled at specified step_scale sequentially (1 at a time, S=0.89) with GPU telemetry.
"""

import os
import sys
import time
import json
import subprocess

DEVICE = 0
HISTORY_FILE = "section_b_r1_refine_history.json"

def get_gpu_stats():
    try:
        cmd = "nvidia-smi --query-gpu=temperature.gpu,clocks.current.graphics,clocks.current.memory --format=csv,noheader,nounits"
        out = subprocess.check_output(cmd, shell=True).decode().strip()
        parts = [p.strip() for p in out.split(',')]
        return {
            'temp_c': float(parts[0]),
            'sm_clock_mhz': int(parts[1]),
            'mem_clock_mhz': int(parts[2])
        }
    except Exception:
        return {'temp_c': -1.0, 'sm_clock_mhz': -1, 'mem_clock_mhz': -1}

def run_job(cmd, desc=""):
    print(f"\n{'='*95}", flush=True)
    print(f"STARTING: {desc}", flush=True)
    print(f"CMD: {cmd}", flush=True)
    gpu_before = get_gpu_stats()
    print(f"[GPU BEFORE] Temp: {gpu_before['temp_c']}°C | SM Clock: {gpu_before['sm_clock_mhz']} MHz | Mem Clock: {gpu_before['mem_clock_mhz']} MHz", flush=True)
    print(f"{'='*95}", flush=True)

    t0 = time.time()
    res = subprocess.run(cmd, shell=True)
    dt = time.time() - t0

    gpu_after = get_gpu_stats()
    print(f"\n[GPU AFTER ] Temp: {gpu_after['temp_c']}°C | SM Clock: {gpu_after['sm_clock_mhz']} MHz | Mem Clock: {gpu_after['mem_clock_mhz']} MHz", flush=True)
    print(f"DURATION: {dt:.1f}s ({dt/60:.2f} min) | Exit Code: {res.returncode}", flush=True)

    if res.returncode != 0:
        print(f"ERROR: Job failed with exit code {res.returncode}", flush=True)
        sys.exit(res.returncode)

    record = {
        'desc': desc,
        'cmd': cmd,
        'duration_sec': dt,
        'gpu_before': gpu_before,
        'gpu_after': gpu_after,
        'returncode': res.returncode
    }
    
    history = []
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, 'r') as f:
                history = json.load(f)
        except Exception:
            history = []
    history.append(record)
    with open(HISTORY_FILE, 'w') as f:
        json.dump(history, f, indent=2)

    return record

def main():
    scale = 0.24
    if len(sys.argv) > 1:
        scale = float(sys.argv[1])
    
    seeds = [106, 107, 108]
    if len(sys.argv) > 2:
        seeds = [int(s) for s in sys.argv[2:]]

    print(f"Running R1 seeds {seeds} with Hard-scaled at step_scale = {scale}", flush=True)
    for s in seeds:
        cmd = (f"python -u run_UnlearningTask.py --seed {s} --device {DEVICE} --module CNN_CIFAR10 "
               f"--algorithm FedOSD --dataloader DataLoader_cifar10_pat --N 10 --NC 2 --balance True "
               f"--B 200 --C 1.0 --R 200 --UR 100 --E 1 --decay 0.999 --step_type bgd --unlearn_cn 1 "
               f"--save_model True --lr 0.0004 --r_lr 1e-06 --proj_mode hard --step_scale {scale} "
               f"--adap_lam 100000.0 --adap_weight_eta 0.0")
        run_job(cmd, f"R1 Hard-scaled {scale} (Seed {s}, r_lr=1e-6)")

if __name__ == '__main__':
    main()
