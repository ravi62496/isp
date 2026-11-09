#!/usr/bin/env python3
"""
run_confirmation.py
Stage 4 Confirmatory Execution Runner (Protocol v5.1).
Sequential runner for seeds 21-30 under Primary Regime R2.

Runs each seed end to end:
  1. Pretrain (FedAvg, R=100, lr=0.05, --unlearn_pretrain True)
  2. Retrain from scratch (FedAvg, R=300, lr=0.05, --test_interval 5)
  3. Arm A1 (Hard FedOSD, step_scale 1.0, R=200, UR=100)
  4. Arm A2 (Hard-scaled FedOSD, step_scale 0.54, R=200, UR=100)
  5. Arm A3 (One-sided, unscaled cap 1.0, R=200, UR=100)
  6. Arm A4 (One-sided rescaled to ||g_u||, R=200, UR=100)

Enforces strict pre-flight checks and safety guards per SUCCESS_CRITERIA_v5.1.
"""

import os
import sys
import time
import json
import re
import hashlib
import argparse
import subprocess
from datetime import datetime
from collections import OrderedDict

CONFIRMATORY_SEEDS = list(range(21, 31))  # [21, 22, 23, 24, 25, 26, 27, 28, 29, 30]
RESULTS_CONFIRM_DIR = "results_confirm"
CRITERIA_FILE = "SUCCESS_CRITERIA_v5.1.md"
EVALUATOR_FILE = "evaluate_confirmation.py"
FROZEN_CONFIGS_FILE = "FROZEN_CONFIGS_v5.md"

GIT_STATUS_CMD = [
    "git", "status", "--porcelain", "--untracked-files=no", "--", ".",
    ":(exclude)UnlearningTask", ":(exclude)figures", ":(exclude)results_confirm", ":(exclude)results_dryrun"
]

BASE_FEDOSD = os.path.join(
    "UnlearningTask", "CNN_CIFAR10",
    "cifar10 pat balanced B200 NC2 N10", "UN1", "E1", "C1.0", "FedOSD"
)
BASE_FEDAVG = os.path.join(
    "UnlearningTask", "CNN_CIFAR10",
    "cifar10 pat balanced B200 NC2 N10", "UN1", "E1", "C1.0", "FedAvg"
)
PRETRAIN_DIR = os.path.join(
    "UnlearningTask", "CNN_CIFAR10",
    "cifar10 pat balanced B200 NC2 N10", "UN1", "E1"
)

def get_gpu_stats():
    """Query NVIDIA GPU telemetry (temperature, SM clock, memory clock)."""
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

def check_evaluator_hash(criteria_path=CRITERIA_FILE, evaluator_path=EVALUATOR_FILE):
    """Verify that the evaluator file's SHA-256 matches the hash in SUCCESS_CRITERIA_v5.1.md."""
    with open(evaluator_path, "rb") as f:
        actual_hash = hashlib.sha256(f.read()).hexdigest().lower()
        
    if not os.path.exists(criteria_path):
        raise FileNotFoundError(f"[PRE-FLIGHT ERROR] Criteria file not found: {criteria_path}")
    with open(criteria_path, "r", encoding="utf-8") as f:
        content = f.read()
    m = re.search(r"evaluate_confirmation\.py\s+<([0-9a-fA-F]{64})>", content) or re.search(r"evaluator SHA-256:\s*([0-9a-fA-F]{64})", content)
    if not m:
        raise ValueError(f"[PRE-FLIGHT ERROR] Could not find 'evaluator SHA-256: <hash>' in {criteria_path}")
    recorded_hash = m.group(1).lower()
    if actual_hash != recorded_hash:
        raise ValueError(f"[PRE-FLIGHT HASH MISMATCH]\nActual ({evaluator_path}): {actual_hash}\nRecorded in {criteria_path}: {recorded_hash}")
    return actual_hash

def preflight_check(require_clean_tree=True, require_empty_results=True, results_dir=RESULTS_CONFIRM_DIR):
    """Execute pre-flight checks required before confirmatory execution."""
    print("=" * 90)
    print("STAGE 4 PROTOCOL v5.1 RUN SCRIPT PRE-FLIGHT INTEGRITY CHECKS")
    print("=" * 90)
    
    # 1. Clean git status check
    print("[PRE-FLIGHT 1/4] Checking git working tree...")
    git_res = subprocess.run(GIT_STATUS_CMD, capture_output=True, text=True)
    if git_res.returncode != 0:
        raise RuntimeError(f"git status command failed: {git_res.stderr}")
    if git_res.stdout.strip():
        print("[FAIL] Git working tree has tracked modifications:")
        print(git_res.stdout.strip())
        if require_clean_tree:
            raise RuntimeError("Stage 4 mandates a clean git working tree before confirmatory runs.")
    else:
        print("  [PASS] Git working tree is clean.")

    # 2. Check evaluator hash matches criteria file
    print("[PRE-FLIGHT 2/4] Verifying evaluator SHA-256 against criteria file...")
    eval_hash = check_evaluator_hash()
    print(f"  [PASS] Evaluator SHA-256 verified: {eval_hash}")

    # 3. Check results directory is empty at start
    print(f"[PRE-FLIGHT 3/4] Verifying {results_dir}/ is empty at start...")
    os.makedirs(results_dir, exist_ok=True)
    files = [f for f in os.listdir(results_dir) if not f.startswith('.')]
    if require_empty_results and len(files) > 0:
        raise RuntimeError(f"{results_dir}/ is not empty at start! Contains: {files}")
    print(f"  [PASS] {results_dir}/ is ready for sealed execution.")

    # 4. Check deterministic settings environment file
    print("[PRE-FLIGHT 4/4] Verifying FROZEN_CONFIGS_v5.md exists...")
    if not os.path.exists(FROZEN_CONFIGS_FILE):
        raise FileNotFoundError(f"Missing {FROZEN_CONFIGS_FILE}")
    print(f"  [PASS] {FROZEN_CONFIGS_FILE} verified.")

    print("=" * 90)
    print("ALL PRE-FLIGHT CHECKS PASSED.")
    print("=" * 90)
    return True

def get_job_spec(seed, device=0):
    """
    Construct the full list of end-to-end commands and their expected output files for a seed.
    Returns an OrderedDict mapping job_id -> dict with 'desc', 'cmd', 'expected_files'.
    """
    common_prefix = (
        f"python -u run_UnlearningTask.py --seed {seed} --device {device} --module CNN_CIFAR10 "
        f"--dataloader DataLoader_cifar10_pat --N 10 --NC 2 --balance True --B 200 --C 1.0 "
        f"--E 1 --decay 0.999 --step_type bgd --unlearn_cn 1 --save_model True"
    )
    
    pretrain_model = os.path.join(PRETRAIN_DIR, f"seed{seed}_unlearn_task_pretrained_model.pth")
    retrain_file = os.path.join(BASE_FEDAVG, f"seed{seed} retrain_from_scratch seed{seed} N10 C1.0 FedAvg CNN_CIFAR10 E1 lr0.05 decay0.999 g_clip-1.0 R300.json")
    a1_unlearn = os.path.join(BASE_FEDOSD, f"seed{seed} N10 C1.0 FedOSD CNN_CIFAR10 E1 lr0.0004 decay0.999 g_clip-1.0 proj_hard lam100000.0 eta0.0 rlr0.0001_unlearn.json")
    a1_recovery = os.path.join(BASE_FEDOSD, f"seed{seed} N10 C1.0 FedOSD CNN_CIFAR10 E1 lr0.0004 decay0.999 g_clip-1.0 proj_hard lam100000.0 eta0.0 rlr0.0001.json")
    a2_unlearn = os.path.join(BASE_FEDOSD, f"seed{seed} N10 C1.0 FedOSD CNN_CIFAR10 E1 lr0.0004 decay0.999 g_clip-1.0 proj_hard lam100000.0 eta0.0 scale0.54 rlr0.0001_unlearn.json")
    a2_recovery = os.path.join(BASE_FEDOSD, f"seed{seed} N10 C1.0 FedOSD CNN_CIFAR10 E1 lr0.0004 decay0.999 g_clip-1.0 proj_hard lam100000.0 eta0.0 scale0.54 rlr0.0001.json")
    a3_unlearn = os.path.join(BASE_FEDOSD, f"seed{seed} N10 C1.0 FedOSD CNN_CIFAR10 E1 lr0.0004 decay0.999 g_clip-1.0 proj_one_sided_cap1.0 wm_uniform lam100000.0 rlr0.0001_unlearn.json")
    a3_recovery = os.path.join(BASE_FEDOSD, f"seed{seed} N10 C1.0 FedOSD CNN_CIFAR10 E1 lr0.0004 decay0.999 g_clip-1.0 proj_one_sided_cap1.0 wm_uniform lam100000.0 rlr0.0001.json")
    a4_unlearn = os.path.join(BASE_FEDOSD, f"seed{seed} N10 C1.0 FedOSD CNN_CIFAR10 E1 lr0.0004 decay0.999 g_clip-1.0 proj_one_sided wm_uniform lam100000.0 eta0.0 rlr0.0001_unlearn.json")
    a4_recovery = os.path.join(BASE_FEDOSD, f"seed{seed} N10 C1.0 FedOSD CNN_CIFAR10 E1 lr0.0004 decay0.999 g_clip-1.0 proj_one_sided wm_uniform lam100000.0 eta0.0 rlr0.0001.json")

    jobs = OrderedDict()
    
    # 1. Pretraining
    jobs['pretrain'] = {
        'desc': f"Phase 1: Pretrain Seed {seed} (FedAvg, R=100, lr=0.05)",
        'cmd': f"{common_prefix} --algorithm FedAvg --R 100 --UR 0 --lr 0.05 --unlearn_pretrain True || exit 1",
        'expected_files': [pretrain_model]
    }
    
    # 2. Retrain from Scratch R=300
    jobs['retrain'] = {
        'desc': f"Reference: Retrain Seed {seed} from Scratch (FedAvg, R=300, lr=0.05, test_interval=5)",
        'cmd': f"{common_prefix} --algorithm FedAvg --retrain True --test_interval 5 --R 300 --UR 0 --lr 0.05 || exit 1",
        'expected_files': [retrain_file]
    }
    
    # 3. Arm A1: Hard FedOSD (scale 1.0)
    jobs['A1'] = {
        'desc': f"Arm A1: Hard FedOSD Seed {seed} (scale 1.0, R=200, UR=100, lr=0.0004, r_lr=0.0001)",
        'cmd': f"{common_prefix} --algorithm FedOSD --R 200 --UR 100 --lr 0.0004 --r_lr 0.0001 --proj_mode hard --step_scale 1.0 --adap_lam 100000.0 --adap_weight_eta 0.0 || exit 1",
        'expected_files': [a1_unlearn, a1_recovery]
    }
    
    # 4. Arm A2: Hard-scaled FedOSD (step_scale 0.54)
    jobs['A2'] = {
        'desc': f"Arm A2: Hard-scaled FedOSD Seed {seed} (step_scale 0.54, R=200, UR=100, lr=0.0004, r_lr=0.0001)",
        'cmd': f"{common_prefix} --algorithm FedOSD --R 200 --UR 100 --lr 0.0004 --r_lr 0.0001 --proj_mode hard --step_scale 0.54 --adap_lam 100000.0 --adap_weight_eta 0.0 || exit 1",
        'expected_files': [a2_unlearn, a2_recovery]
    }
    
    # 5. Arm A3: One-sided, unscaled (cap 1.0) [Method under test]
    jobs['A3'] = {
        'desc': f"Arm A3: One-sided unscaled cap 1.0 Seed {seed} (R=200, UR=100, lr=0.0004, r_lr=0.0001)",
        'cmd': f"{common_prefix} --algorithm FedOSD --R 200 --UR 100 --lr 0.0004 --r_lr 0.0001 --proj_mode one_sided --cap_hard_ratio 1.0 --weight_mode uniform --adap_lam 100000.0 --adap_weight_eta 0.0 || exit 1",
        'expected_files': [a3_unlearn, a3_recovery]
    }
    
    # 6. Arm A4: One-sided rescaled to ||g_u||
    jobs['A4'] = {
        'desc': f"Arm A4: One-sided rescaled Seed {seed} (R=200, UR=100, lr=0.0004, r_lr=0.0001)",
        'cmd': f"{common_prefix} --algorithm FedOSD --R 200 --UR 100 --lr 0.0004 --r_lr 0.0001 --proj_mode one_sided --weight_mode uniform --adap_lam 100000.0 --adap_weight_eta 0.0 || exit 1",
        'expected_files': [a4_unlearn, a4_recovery]
    }
    
    return jobs

def print_seed_commands(seed, device=0):
    """Print the complete command list for a seed without running."""
    jobs = get_job_spec(seed, device=device)
    print(f"\n{'='*100}")
    print(f"FULL COMMAND LIST FOR SEED {seed} (PRIMARY REGIME R2)")
    print(f"{'='*100}")
    for idx, (jid, info) in enumerate(jobs.items(), 1):
        print(f"\n[{idx}/6] {info['desc']}")
        print(f"Command:\n{info['cmd']}")
    print(f"\n{'='*100}\n")

def run_job(cmd, desc=""):
    """Execute a single job while logging GPU telemetry and checking exit status."""
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
    print(f"{'='*95}", flush=True)
    print(f"FINISHED: {desc} (Duration: {dt:.1f}s / {dt/60:.2f}min, Exit Code: {res.returncode})", flush=True)
    print(f"[GPU AFTER ] Temp: {gpu_after['temp_c']}°C | SM Clock: {gpu_after['sm_clock_mhz']} MHz | Mem Clock: {gpu_after['mem_clock_mhz']} MHz", flush=True)
    print(f"{'='*95}\n", flush=True)
    
    if res.returncode != 0:
        raise RuntimeError(f"Job failed with exit code {res.returncode}: {cmd}")
    return dt

class TeeLogger:
    """Tee stdout and stderr to a specified log file with immediate flushing."""
    def __init__(self, filepath, stream):
        self.stream = stream
        self.file = open(filepath, 'a', encoding='utf-8')
    def write(self, data):
        self.stream.write(data)
        self.stream.flush()
        self.file.write(data)
        self.file.flush()
    def flush(self):
        self.stream.flush()
        self.file.flush()

def verify_seed_outputs(seed):
    """
    Verify that all six jobs for the given seed produced their expected output files
    with the exact expected array lengths per protocol v5.1.
    """
    jobs = get_job_spec(seed)
    print(f"\n[INTEGRITY AUDIT] Verifying outputs and array lengths for seed {seed}...")
    
    # 1. Pretrain model checkpoint
    pretrain_file = jobs['pretrain']['expected_files'][0]
    if not os.path.exists(pretrain_file) or os.path.getsize(pretrain_file) == 0:
        raise RuntimeError(f"Missing or empty pretrain model checkpoint for seed {seed}: {pretrain_file}")
    print(f"  [PASS] Pretrain checkpoint verified: {os.path.basename(pretrain_file)}")
    
    # 2. Retrain from scratch baseline
    retrain_file = jobs['retrain']['expected_files'][0]
    if not os.path.exists(retrain_file):
        raise RuntimeError(f"Missing retrain file for seed {seed}: {retrain_file}")
    with open(retrain_file, 'r', encoding='utf-8') as f:
        d = json.load(f)
    ch = d.get('metric_log', {}).get('client_metric_history', [])
    t_acc = ch[0].get('test_accuracy', []) if ch else []
    retrain_len = len(t_acc)
    if retrain_len != 61:
        raise ValueError(f"Retrain array length for seed {seed} is {retrain_len}, expected 61 in {retrain_file}")
    print(f"  [PASS] Retrain output verified (length 61): {os.path.basename(retrain_file)}")

    # 3. Arms A1 - A4
    for arm_id in ['A1', 'A2', 'A3', 'A4']:
        unlearn_file, recovery_file = jobs[arm_id]['expected_files']
        
        # Unlearn check (length 100)
        if not os.path.exists(unlearn_file):
            raise RuntimeError(f"Missing {arm_id} unlearn file for seed {seed}: {unlearn_file}")
        with open(unlearn_file, 'r', encoding='utf-8') as f:
            d_u = json.load(f)
        ch_u = d_u.get('metric_log', {}).get('client_metric_history', [])
        t_acc_u = ch_u[0].get('test_accuracy', []) if ch_u else []
        len_u = len(t_acc_u)
        if len_u != 100:
            raise ValueError(f"{arm_id} unlearn array length for seed {seed} is {len_u}, expected 100 in {unlearn_file}")
        
        # Recovery check (length 201)
        if not os.path.exists(recovery_file):
            raise RuntimeError(f"Missing {arm_id} recovery file for seed {seed}: {recovery_file}")
        with open(recovery_file, 'r', encoding='utf-8') as f:
            d_r = json.load(f)
        ch_r = d_r.get('metric_log', {}).get('client_metric_history', [])
        t_acc_r = ch_r[0].get('test_accuracy', []) if ch_r else []
        len_r = len(t_acc_r)
        if len_r != 201:
            raise ValueError(f"{arm_id} recovery array length for seed {seed} is {len_r}, expected 201 in {recovery_file}")
            
        print(f"  [PASS] Arm {arm_id} outputs verified (unlearn len=100, recovery len=201).")

    print(f"[INTEGRITY AUDIT] All outputs for seed {seed} verified successfully.\n")

def main():
    parser = argparse.ArgumentParser(description="Stage 4 Confirmatory Sequential Runner (Protocol v5.1).")
    parser.add_argument('--seeds', type=int, nargs='+', default=CONFIRMATORY_SEEDS,
                        help="Seeds to execute (default: 21-30).")
    parser.add_argument('--device', type=int, default=0,
                        help="GPU device id (default: 0).")
    parser.add_argument('--print_commands', action='store_true',
                        help="Print the full command list without executing.")
    parser.add_argument('--preflight_only', action='store_true',
                        help="Run pre-flight integrity verification and exit.")
    parser.add_argument('--skip_existing', action='store_true',
                        help="Skip commands whose expected output files already exist.")
    parser.add_argument('--authorize_stage4', action='store_true',
                        help="Explicit authorization flag required to run confirmatory seeds 21-30.")
    parser.add_argument('--log_file', type=str, default=None,
                        help="Optional path to write full runner log output in real time.")
    args = parser.parse_args()

    if args.log_file:
        sys.stdout = TeeLogger(args.log_file, sys.stdout)
        sys.stderr = TeeLogger(args.log_file, sys.stderr)

    is_confirmatory = any(s in CONFIRMATORY_SEEDS for s in args.seeds)

    # Mode 1: Print commands only
    if args.print_commands:
        for s in args.seeds:
            print_seed_commands(s, device=args.device)
        return

    # Mode 2: Preflight checks only
    if args.preflight_only:
        preflight_check(require_clean_tree=is_confirmatory, require_empty_results=is_confirmatory)
        return

    # STRICT SAFETY GUARD: Refuse to execute seeds 21-30 unless explicitly authorized
    if is_confirmatory and not args.authorize_stage4:
        print("\n" + "=" * 95)
        print("[SAFETY GUARD] Stage 4 Confirmatory Seeds (21-30) execution requested.")
        print("Execution of seeds 21-30 is STRICTLY LOCKED and requires explicit user authorization.")
        print("Pass --authorize_stage4 to proceed after formal commit and tagging.")
        print("=" * 95 + "\n")
        sys.exit(1)

    # Preflight check before execution
    preflight_check(require_clean_tree=is_confirmatory, require_empty_results=is_confirmatory)

    # Execute seeds sequentially
    total_t0 = time.time()
    for s in args.seeds:
        print(f"\n{'#'*95}")
        print(f"STARTING PIPELINE EXECUTION FOR SEED {s}")
        print(f"{'#'*95}")
        jobs = get_job_spec(s, device=args.device)
        for jid, info in jobs.items():
            if args.skip_existing:
                all_exist = all(os.path.exists(ef) for ef in info['expected_files'])
                if all_exist:
                    print(f"[SKIP EXISTING] {info['desc']}: outputs exist: {info['expected_files']}")
                    continue
            run_job(info['cmd'], desc=info['desc'])
            
        # Verify that all 6 jobs for seed s produced expected outputs with correct array lengths
        verify_seed_outputs(s)
            
    total_dt = time.time() - total_t0
    print(f"\n[ALL DONE] Completed execution for seeds {args.seeds} in {total_dt:.1f}s ({total_dt/60:.2f}min).")

if __name__ == '__main__':
    main()
