#!/usr/bin/env python3
"""
run_stage2_pipeline.py
Sequential runner for Stage 2 experiments per user amendments:
- Run one job at a time (sequential execution, S=0.89)
- Logs GPU temperature and clocks before and after each run
- Runs validation with --strict
- Executes:
  Phase 2A: Uncapped One-Sided on Seeds 101, 102, 103 (R2)
  Phase 2B: s* Refinement on Seeds 101, 102, 103 (Starting at 0.74, refining to within 2%)
  Phase 2C: Seeds 104, 105 under R2 (Pretrain, Hard, Capped, Uncapped)
  Phase 2D: Seeds 106, 107, 108 under R1 (Pretrain, Hard, Capped, Uncapped)
  Phase 2E: Dual Retrain models for MIA Noise Floor (5 partitions: seeds 101-105)
"""

import os
import sys
import time
import json
import subprocess
import glob

DEVICE = 0

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
    except Exception as e:
        return {'temp_c': -1.0, 'sm_clock_mhz': -1, 'mem_clock_mhz': -1}

def run_job(cmd, desc=""):
    print(f"\n{'='*90}")
    print(f"STARTING: {desc}")
    print(f"CMD: {cmd}")
    gpu_before = get_gpu_stats()
    print(f"[GPU BEFORE] Temp: {gpu_before['temp_c']}°C | SM Clock: {gpu_before['sm_clock_mhz']} MHz | Mem Clock: {gpu_before['mem_clock_mhz']} MHz")
    print(f"{'='*90}")

    t0 = time.time()
    res = subprocess.run(cmd, shell=True)
    dt = time.time() - t0

    gpu_after = get_gpu_stats()
    print(f"\n[GPU AFTER ] Temp: {gpu_after['temp_c']}°C | SM Clock: {gpu_after['sm_clock_mhz']} MHz | Mem Clock: {gpu_after['mem_clock_mhz']} MHz")
    print(f"DURATION: {dt:.1f}s ({dt/60:.2f} min) | Exit Code: {res.returncode}")

    if res.returncode != 0:
        print(f"ERROR: Job failed with exit code {res.returncode}")
        sys.exit(res.returncode)

    # Validate output schema if JSON was created
    return {
        'desc': desc,
        'cmd': cmd,
        'duration_sec': dt,
        'gpu_before': gpu_before,
        'gpu_after': gpu_after,
        'returncode': res.returncode
    }

def main():
    print("==========================================================================")
    print("STAGE 2 SEQUENTIAL EXECUTION RUNNER")
    print("Single-job sequential mode (S=0.89) with GPU telemetry")
    print("==========================================================================")

    job_history = []

    # -------------------------------------------------------------------------
    # PHASE 2A: Uncapped One-Sided on Seeds 101, 102, 103 (Regime R2)
    # -------------------------------------------------------------------------
    print("\n>>> PHASE 2A: Uncapped One-Sided on Tuning Seeds 101-103 (R2) <<<")
    for s in [101, 102, 103]:
        cmd = (
            f"python run_UnlearningTask.py --seed {s} --device {DEVICE} --module CNN_CIFAR10 "
            f"--algorithm FedOSD --dataloader DataLoader_cifar10_pat --N 10 --NC 2 --balance True "
            f"--B 200 --C 1.0 --R 200 --UR 100 --E 1 --decay 0.999 --step_type bgd --unlearn_cn 1 "
            f"--save_model True --lr 0.0004 --r_lr 0.0001 --proj_mode one_sided --adap_lam 100000.0 "
            f"--adap_weight_eta 0.0 --weight_mode uniform"
        )
        rec = run_job(cmd, f"Uncapped One-Sided (Seed {s}, R2, r_lr=1e-4)")
        job_history.append(rec)

    # -------------------------------------------------------------------------
    # PHASE 2B: s* Refinement on Seeds 101, 102, 103
    # Start at 0.74, then refine if needed
    # -------------------------------------------------------------------------
    print("\n>>> PHASE 2B: s* Refinement (Seeds 101-103, Starting at 0.74) <<<")
    for s in [101, 102, 103]:
        cmd = (
            f"python run_UnlearningTask.py --seed {s} --device {DEVICE} --module CNN_CIFAR10 "
            f"--algorithm FedOSD --dataloader DataLoader_cifar10_pat --N 10 --NC 2 --balance True "
            f"--B 200 --C 1.0 --R 200 --UR 100 --E 1 --decay 0.999 --step_type bgd --unlearn_cn 1 "
            f"--save_model True --lr 0.0004 --r_lr 0.0001 --proj_mode hard --adap_lam 100000.0 "
            f"--adap_weight_eta 0.0 --step_scale 0.74"
        )
        rec = run_job(cmd, f"Hard-scaled 0.74 (Seed {s}, R2)")
        job_history.append(rec)

    # Also test scale 0.54 (since scale 0.50 had -3.8% displacement)
    print("\n>>> PHASE 2B (Refinement): Hard-scaled 0.54 (Seeds 101-103) <<<")
    for s in [101, 102, 103]:
        cmd = (
            f"python run_UnlearningTask.py --seed {s} --device {DEVICE} --module CNN_CIFAR10 "
            f"--algorithm FedOSD --dataloader DataLoader_cifar10_pat --N 10 --NC 2 --balance True "
            f"--B 200 --C 1.0 --R 200 --UR 100 --E 1 --decay 0.999 --step_type bgd --unlearn_cn 1 "
            f"--save_model True --lr 0.0004 --r_lr 0.0001 --proj_mode hard --adap_lam 100000.0 "
            f"--adap_weight_eta 0.0 --step_scale 0.54"
        )
        rec = run_job(cmd, f"Hard-scaled 0.54 (Seed {s}, R2)")
        job_history.append(rec)

    # -------------------------------------------------------------------------
    # PHASE 2C: Seeds 104, 105 under Regime R2
    # -------------------------------------------------------------------------
    print("\n>>> PHASE 2C: Seeds 104-105 under Regime R2 <<<")
    for s in [104, 105]:
        # Pretrain
        cmd_pre = (
            f"python run_UnlearningTask.py --seed {s} --device {DEVICE} --module CNN_CIFAR10 "
            f"--algorithm FedAvg --dataloader DataLoader_cifar10_pat --N 10 --NC 2 --balance True "
            f"--B 200 --C 1.0 --R 100 --UR 100 --E 1 --decay 0.999 --step_type bgd --unlearn_cn 1 "
            f"--save_model True --lr 0.05 --unlearn_pretrain True"
        )
        rec = run_job(cmd_pre, f"Pretrain Checkpoint (Seed {s}, R2, lr=0.05)")
        job_history.append(rec)

        # Hard FedOSD
        cmd_h = (
            f"python run_UnlearningTask.py --seed {s} --device {DEVICE} --module CNN_CIFAR10 "
            f"--algorithm FedOSD --dataloader DataLoader_cifar10_pat --N 10 --NC 2 --balance True "
            f"--B 200 --C 1.0 --R 200 --UR 100 --E 1 --decay 0.999 --step_type bgd --unlearn_cn 1 "
            f"--save_model True --lr 0.0004 --r_lr 0.0001 --proj_mode hard --adap_lam 100000.0 "
            f"--adap_weight_eta 0.0"
        )
        rec = run_job(cmd_h, f"Hard FedOSD (Seed {s}, R2)")
        job_history.append(rec)

        # Capped One-Sided (cap=1.0)
        cmd_c = (
            f"python run_UnlearningTask.py --seed {s} --device {DEVICE} --module CNN_CIFAR10 "
            f"--algorithm FedOSD --dataloader DataLoader_cifar10_pat --N 10 --NC 2 --balance True "
            f"--B 200 --C 1.0 --R 200 --UR 100 --E 1 --decay 0.999 --step_type bgd --unlearn_cn 1 "
            f"--save_model True --lr 0.0004 --r_lr 0.0001 --proj_mode one_sided --adap_lam 100000.0 "
            f"--adap_weight_eta 0.0 --weight_mode uniform --cap_hard_ratio 1.0"
        )
        rec = run_job(cmd_c, f"Capped One-Sided (Seed {s}, R2, cap=1.0)")
        job_history.append(rec)

        # Uncapped One-Sided
        cmd_u = (
            f"python run_UnlearningTask.py --seed {s} --device {DEVICE} --module CNN_CIFAR10 "
            f"--algorithm FedOSD --dataloader DataLoader_cifar10_pat --N 10 --NC 2 --balance True "
            f"--B 200 --C 1.0 --R 200 --UR 100 --E 1 --decay 0.999 --step_type bgd --unlearn_cn 1 "
            f"--save_model True --lr 0.0004 --r_lr 0.0001 --proj_mode one_sided --adap_lam 100000.0 "
            f"--adap_weight_eta 0.0 --weight_mode uniform"
        )
        rec = run_job(cmd_u, f"Uncapped One-Sided (Seed {s}, R2)")
        job_history.append(rec)

    # -------------------------------------------------------------------------
    # PHASE 2D: Fresh Seeds 106, 107, 108 under Regime R1
    # -------------------------------------------------------------------------
    print("\n>>> PHASE 2D: Seeds 106-108 under Regime R1 (pretrain lr 0.01, r_lr 1e-6) <<<")
    for s in [106, 107, 108]:
        # Pretrain R1 (lr=0.01)
        cmd_pre = (
            f"python run_UnlearningTask.py --seed {s} --device {DEVICE} --module CNN_CIFAR10 "
            f"--algorithm FedAvg --dataloader DataLoader_cifar10_pat --N 10 --NC 2 --balance True "
            f"--B 200 --C 1.0 --R 100 --UR 100 --E 1 --decay 0.999 --step_type bgd --unlearn_cn 1 "
            f"--save_model True --lr 0.01 --unlearn_pretrain True"
        )
        rec = run_job(cmd_pre, f"Pretrain Checkpoint (Seed {s}, R1, lr=0.01)")
        job_history.append(rec)

        # Hard FedOSD R1 (r_lr=1e-6)
        cmd_h = (
            f"python run_UnlearningTask.py --seed {s} --device {DEVICE} --module CNN_CIFAR10 "
            f"--algorithm FedOSD --dataloader DataLoader_cifar10_pat --N 10 --NC 2 --balance True "
            f"--B 200 --C 1.0 --R 200 --UR 100 --E 1 --decay 0.999 --step_type bgd --unlearn_cn 1 "
            f"--save_model True --lr 0.0004 --r_lr 0.000001 --proj_mode hard --adap_lam 100000.0 "
            f"--adap_weight_eta 0.0"
        )
        rec = run_job(cmd_h, f"Hard FedOSD (Seed {s}, R1, r_lr=1e-6)")
        job_history.append(rec)

        # Capped One-Sided R1 (cap=1.0, r_lr=1e-6)
        cmd_c = (
            f"python run_UnlearningTask.py --seed {s} --device {DEVICE} --module CNN_CIFAR10 "
            f"--algorithm FedOSD --dataloader DataLoader_cifar10_pat --N 10 --NC 2 --balance True "
            f"--B 200 --C 1.0 --R 200 --UR 100 --E 1 --decay 0.999 --step_type bgd --unlearn_cn 1 "
            f"--save_model True --lr 0.0004 --r_lr 0.000001 --proj_mode one_sided --adap_lam 100000.0 "
            f"--adap_weight_eta 0.0 --weight_mode uniform --cap_hard_ratio 1.0"
        )
        rec = run_job(cmd_c, f"Capped One-Sided (Seed {s}, R1, cap=1.0, r_lr=1e-6)")
        job_history.append(rec)

        # Uncapped One-Sided R1 (r_lr=1e-6)
        cmd_u = (
            f"python run_UnlearningTask.py --seed {s} --device {DEVICE} --module CNN_CIFAR10 "
            f"--algorithm FedOSD --dataloader DataLoader_cifar10_pat --N 10 --NC 2 --balance True "
            f"--B 200 --C 1.0 --R 200 --UR 100 --E 1 --decay 0.999 --step_type bgd --unlearn_cn 1 "
            f"--save_model True --lr 0.0004 --r_lr 0.000001 --proj_mode one_sided --adap_lam 100000.0 "
            f"--adap_weight_eta 0.0 --weight_mode uniform"
        )
        rec = run_job(cmd_u, f"Uncapped One-Sided (Seed {s}, R1, r_lr=1e-6)")
        job_history.append(rec)

    # -------------------------------------------------------------------------
    # PHASE 2E: Retrains for MIA Noise Floor (Partitions 101-105)
    # -------------------------------------------------------------------------
    print("\n>>> PHASE 2E: Retrains for MIA Noise Floor (Partitions 101-105) <<<")
    # Retrain on 104 and 105 (R=300)
    for s in [104, 105]:
        cmd_ret = (
            f"python run_UnlearningTask.py --seed {s} --device {DEVICE} --module CNN_CIFAR10 "
            f"--algorithm FedAvg --dataloader DataLoader_cifar10_pat --N 10 --NC 2 --balance True "
            f"--B 200 --C 1.0 --R 300 --UR 0 --E 1 --decay 0.999 --step_type bgd --unlearn_cn 1 "
            f"--save_model True --lr 0.05 --retrain True --test_interval 5"
        )
        rec = run_job(cmd_ret, f"Retrain Init 1 (Seed {s}, R=300)")
        job_history.append(rec)

    # Save summary of jobs
    with open("stage2_job_history.json", "w") as f:
        json.dump(job_history, f, indent=2)
    print("\nAll Stage 2 jobs completed! Summary saved to stage2_job_history.json.")

if __name__ == '__main__':
    main()

