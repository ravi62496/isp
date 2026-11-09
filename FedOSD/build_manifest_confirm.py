#!/usr/bin/env python3
"""
build_manifest_confirm.py
Protocol v5.1 Manifest Builder for Confirmatory Analysis.

Scans the experiment output directories for confirmatory runs, copies the verified
JSON result files into results_confirm/ (or a specified target directory), and generates
RUN_MANIFEST_CONFIRM.csv containing:
  - arm, seed, stage
  - path and absolute_file_path
  - full sha1 (40 characters)
  - git commit hash
  - config hash (full sha256 of canonical params JSON)
  - array length and index used
  - file mtime and cli command

Compliant with SUCCESS_CRITERIA_v5.1 and evaluate_confirmation.py manifest parser.
"""

import os
import sys
import json
import csv
import glob
import shutil
import hashlib
import argparse
from datetime import datetime

CONFIRMATORY_SEEDS = list(range(21, 31))  # [21, 22, 23, 24, 25, 26, 27, 28, 29, 30]
DEFAULT_TARGET_DIR = "results_confirm"
DEFAULT_MANIFEST_OUT = "RUN_MANIFEST_CONFIRM.csv"

BASE_FEDOSD_DEFAULT = os.path.join(
    "UnlearningTask", "CNN_CIFAR10",
    "cifar10 pat balanced B200 NC2 N10", "UN1", "E1", "C1.0", "FedOSD"
)
BASE_FEDAVG_DEFAULT = os.path.join(
    "UnlearningTask", "CNN_CIFAR10",
    "cifar10 pat balanced B200 NC2 N10", "UN1", "E1", "C1.0", "FedAvg"
)

# Candidate filename patterns per arm (priority ordered)
PATTERNS = {
    'A1': [
        'proj_hard lam100000.0 eta0.0 rlr0.0001',
        'proj_hard lam100000.0 eta0.0'
    ],
    'A2': [
        'proj_hard lam100000.0 eta0.0 scale0.54 rlr0.0001',
        'proj_hard lam100000.0 eta0.0 scale0.54'
    ],
    'A3': [
        'proj_one_sided_cap1.0 wm_uniform lam100000.0 rlr0.0001',
        'proj_one_sided_cap1.0 wm_uniform lam100000.0'
    ],
    'A4': [
        'proj_one_sided wm_uniform lam100000.0 eta0.0 rlr0.0001',
        'proj_one_sided wm_uniform lam100000.0 rlr0.0001',
        'proj_one_sided wm_uniform lam100000.0 eta0.0',
        'proj_one_sided wm_uniform lam100000.0'
    ]
}

def locate_candidate_file(base_dir, seed, patterns, suffix):
    """Find matching candidate files in base_dir for specific patterns and suffix."""
    candidates = []
    for pat in patterns:
        fn = f"seed{seed} N10 C1.0 FedOSD CNN_CIFAR10 E1 lr0.0004 decay0.999 g_clip-1.0 {pat}{suffix}"
        fp = os.path.join(base_dir, fn)
        if os.path.exists(fp):
            candidates.append(fp)
    if not candidates:
        for pat in patterns:
            cand = [f for f in glob.glob(os.path.join(base_dir, f"*seed{seed}*{suffix}"))
                    if pat in os.path.basename(f) and not f.endswith('.log') and not f.endswith('.pth')]
            candidates.extend(cand)
    # Deduplicate while preserving order
    seen = set()
    deduped = []
    for c in candidates:
        if c not in seen:
            seen.add(c)
            deduped.append(c)
    return deduped

def locate_retrain_file(base_dir, seed):
    """Find the Retrain R=300 file for a seed."""
    fn_300 = f"seed{seed} retrain_from_scratch seed{seed} N10 C1.0 FedAvg CNN_CIFAR10 E1 lr0.05 decay0.999 g_clip-1.0 R300.json"
    fp_300 = os.path.join(base_dir, fn_300)
    if os.path.exists(fp_300):
        return [fp_300]
    
    # Fallback search
    cand = [f for f in glob.glob(os.path.join(base_dir, f"*seed{seed}*retrain*R300.json")) if not f.endswith('.log')]
    if cand:
        return cand
    cand_any = [f for f in glob.glob(os.path.join(base_dir, f"*seed{seed}*retrain*.json")) if not f.endswith('.log')]
    return cand_any

def inspect_json(fpath, target_round, is_retrain=False):
    """Read JSON and return metadata dict."""
    with open(fpath, 'rb') as f:
        file_bytes = f.read()
    full_sha1 = hashlib.sha1(file_bytes).hexdigest()
    
    with open(fpath, 'r', encoding='utf-8') as f:
        d = json.load(f)
        
    prov = d.get('provenance', {})
    commit = prov.get('git_commit', 'NONE')
    cli_args = prov.get('cli_args', [])
    cli = ' '.join(cli_args) if isinstance(cli_args, list) else str(cli_args)
    
    params = d.get('params', {})
    params_str = json.dumps(params, sort_keys=True)
    config_hash = hashlib.sha256(params_str.encode('utf-8')).hexdigest()
    
    ch = d.get('metric_log', {}).get('client_metric_history', [])
    t_acc = ch[0].get('test_accuracy', []) if ch else []
    arr_len = len(t_acc)
    test_interval = params.get('test_interval', 1)
    
    if is_retrain:
        if target_round == 200:
            idx_used = 40
        elif target_round == 300:
            idx_used = arr_len - 1
        else:
            idx_used = min(arr_len - 1, target_round // test_interval)
    else:
        idx_used = arr_len - 1
        
    mtime = datetime.fromtimestamp(os.path.getmtime(fpath)).strftime('%Y-%m-%d %H:%M:%S')
    
    return {
        'full_sha1': full_sha1,
        'commit': commit,
        'cli': cli,
        'config_hash': config_hash,
        'arr_len': arr_len,
        'idx_used': idx_used,
        'mtime': mtime
    }

def build_manifest(seeds, target_dir=DEFAULT_TARGET_DIR, manifest_out=DEFAULT_MANIFEST_OUT,
                   source_fedosd=BASE_FEDOSD_DEFAULT, source_fedavg=BASE_FEDAVG_DEFAULT,
                   copy_files=True):
    """
    Build RUN_MANIFEST_CONFIRM.csv and optionally copy files to target_dir.
    """
    os.makedirs(target_dir, exist_ok=True)
    manifest_rows = []
    
    print(f"\nScanning outputs for seeds {seeds}...")
    print(f"Source FedOSD: {source_fedosd}")
    print(f"Source FedAvg: {source_fedavg}")
    print(f"Target Directory: {target_dir}")
    print(f"Output Manifest:  {manifest_out}")
    print("=" * 90)

    # 1. Unlearning and recovery arms: A1, A2, A3, A4
    for arm in ['A1', 'A2', 'A3', 'A4']:
        for s in seeds:
            for stg in ['unlearn', 'recovery']:
                suffix = '_unlearn.json' if stg == 'unlearn' else '.json'
                target_round = 100 if stg == 'unlearn' else 200
                
                candidates = locate_candidate_file(source_fedosd, s, PATTERNS[arm], suffix)
                if not candidates:
                    print(f"[MISSING] Arm {arm}, Seed {s}, Stage {stg}")
                    manifest_rows.append({
                        'arm': arm, 'seed': s, 'stage': stg,
                        'path': 'MISSING', 'sha1': 'NONE', 'commit': 'NONE',
                        'config_hash': 'NONE', 'array_length': 0, 'index': -1,
                        'absolute_file_path': 'MISSING', 'file_mtime': 'NONE',
                        'git_commit': 'NONE', 'cli': 'NONE', 'array_index_used': -1,
                        'candidate_files_count': 0
                    })
                    continue
                
                chosen_src = candidates[0]
                dest_fn = os.path.basename(chosen_src)
                dest_path = os.path.join(target_dir, dest_fn)
                
                if copy_files and os.path.abspath(chosen_src) != os.path.abspath(dest_path):
                    shutil.copy2(chosen_src, dest_path)
                    final_path = dest_path
                else:
                    final_path = chosen_src
                    
                meta = inspect_json(final_path, target_round=target_round, is_retrain=False)
                manifest_rows.append({
                    'arm': arm,
                    'seed': s,
                    'stage': stg,
                    'path': final_path,
                    'sha1': meta['full_sha1'],
                    'commit': meta['commit'],
                    'config_hash': meta['config_hash'],
                    'array_length': meta['arr_len'],
                    'index': meta['idx_used'],
                    'absolute_file_path': os.path.abspath(final_path),
                    'file_mtime': meta['mtime'],
                    'git_commit': meta['commit'],
                    'cli': meta['cli'],
                    'array_index_used': meta['idx_used'],
                    'candidate_files_count': len(candidates)
                })
                print(f"[REGISTERED] Arm {arm}, Seed {s}, Stage {stg}: {os.path.basename(final_path)} (sha1: {meta['full_sha1'][:10]}..., arr_len: {meta['arr_len']})")

    # 2. Reference Retrain arms: Retrain at Round 200 and Round 300
    for s in seeds:
        ret_cands = locate_retrain_file(source_fedavg, s)
        if not ret_cands:
            print(f"[MISSING] Retrain Seed {s}")
            for stg, rd in [('unlearn', 200), ('recovery', 300)]:
                manifest_rows.append({
                    'arm': 'Retrain', 'seed': s, 'stage': stg,
                    'path': 'MISSING', 'sha1': 'NONE', 'commit': 'NONE',
                    'config_hash': 'NONE', 'array_length': 0, 'index': -1,
                    'absolute_file_path': 'MISSING', 'file_mtime': 'NONE',
                    'git_commit': 'NONE', 'cli': 'NONE', 'array_index_used': -1,
                    'candidate_files_count': 0
                })
            continue

        chosen_ret = ret_cands[0]
        dest_fn = os.path.basename(chosen_ret)
        dest_path = os.path.join(target_dir, dest_fn)
        
        if copy_files and os.path.abspath(chosen_ret) != os.path.abspath(dest_path):
            shutil.copy2(chosen_ret, dest_path)
            final_ret_path = dest_path
        else:
            final_ret_path = chosen_ret

        for stg, rd in [('unlearn', 200), ('recovery', 300)]:
            meta = inspect_json(final_ret_path, target_round=rd, is_retrain=True)
            manifest_rows.append({
                'arm': 'Retrain',
                'seed': s,
                'stage': stg,
                'path': final_ret_path,
                'sha1': meta['full_sha1'],
                'commit': meta['commit'],
                'config_hash': meta['config_hash'],
                'array_length': meta['arr_len'],
                'index': meta['idx_used'],
                'absolute_file_path': os.path.abspath(final_ret_path),
                'file_mtime': meta['mtime'],
                'git_commit': meta['commit'],
                'cli': meta['cli'],
                'array_index_used': meta['idx_used'],
                'candidate_files_count': len(ret_cands)
            })
            print(f"[REGISTERED] Arm Retrain, Seed {s}, Stage {stg} (Rd {rd}): {os.path.basename(final_ret_path)} (idx: {meta['idx_used']})")

    # Write manifest CSV
    fieldnames = [
        'arm', 'seed', 'stage', 'path', 'sha1', 'commit', 'config_hash',
        'array_length', 'index', 'absolute_file_path', 'file_mtime',
        'git_commit', 'cli', 'array_index_used', 'candidate_files_count'
    ]
    with open(manifest_out, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(manifest_rows)

    print("=" * 90)
    print(f"Manifest written successfully: {manifest_out} ({len(manifest_rows)} cells)")
    return manifest_rows

def main():
    parser = argparse.ArgumentParser(description="Build RUN_MANIFEST_CONFIRM.csv for confirmatory analysis.")
    parser.add_argument('--seeds', type=int, nargs='+', default=CONFIRMATORY_SEEDS,
                        help="Seeds to include in the manifest (default: 21-30).")
    parser.add_argument('--target_dir', type=str, default=DEFAULT_TARGET_DIR,
                        help=f"Target directory to copy result files to (default: {DEFAULT_TARGET_DIR}).")
    parser.add_argument('--manifest_out', type=str, default=DEFAULT_MANIFEST_OUT,
                        help=f"Path to output manifest CSV (default: {DEFAULT_MANIFEST_OUT}).")
    parser.add_argument('--source_fedosd', type=str, default=BASE_FEDOSD_DEFAULT,
                        help="Source directory for FedOSD results.")
    parser.add_argument('--source_fedavg', type=str, default=BASE_FEDAVG_DEFAULT,
                        help="Source directory for FedAvg Retrain results.")
    parser.add_argument('--no_copy', action='store_true',
                        help="Do not copy files; reference them in-place.")
    args = parser.parse_args()

    build_manifest(
        seeds=args.seeds,
        target_dir=args.target_dir,
        manifest_out=args.manifest_out,
        source_fedosd=args.source_fedosd,
        source_fedavg=args.source_fedavg,
        copy_files=(not args.no_copy)
    )

if __name__ == '__main__':
    main()
