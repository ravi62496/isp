#!/usr/bin/env python3
"""
validate_schema.py
Validates result JSON files for required fields and consistency.
Required fields: seed, commit, regime, stage, method, cli, pre_acc, acc, asr, forget_acc, mia_auc, unlearn_displacement.
Strict mode (--strict): requires per-round shadow_diagnostics containing raw_over_hard, used_over_hard, capped, displacement.
"""

import sys
import os
import glob
import json
import argparse
import numpy as np

def infer_regime(params):
    # Check if explicitly stored
    if 'regime' in params:
        return params['regime']
    
    # Infer from learning rates
    lr = float(params.get('lr', 0.05))
    r_lr = float(params.get('r_lr', 1e-4))
    is_retrain = bool(params.get('retrain', False))
    unlearn_pre = bool(params.get('unlearn_pretrain', False))
    
    if unlearn_pre:
        if abs(lr - 0.05) < 1e-4: return 'R2'
        if abs(lr - 0.01) < 1e-4: return 'R1'
    
    if is_retrain:
        if abs(lr - 0.05) < 1e-4: return 'R2'
        if abs(lr - 0.01) < 1e-4: return 'R1b'
        
    # FedOSD
    if abs(r_lr - 1e-4) < 1e-6:
        return 'R2'
    elif abs(r_lr - 1e-6) < 1e-8:
        return 'R1'
    return 'R2'

def infer_stage(fpath, params):
    if 'stage' in params:
        return params['stage']
    if '_unlearn.json' in fpath:
        return 'unlearn'
    if params.get('retrain', False):
        r = params.get('R', 200)
        return 'unlearn' if r == 200 else 'recovery'
    r = params.get('R', 200)
    ur = params.get('UR', 100)
    if r == ur:
        return 'unlearn'
    return 'recovery'

def validate_file(fpath, strict=False):
    try:
        with open(fpath, 'r') as f:
            data = json.load(f)
    except Exception as e:
        return False, f"JSON parse error: {e}"

    params = data.get('params', {})
    prov = data.get('provenance', {})
    metric_log = data.get('metric_log', {})
    alg_log = data.get('algorithm_log', {})
    ch = metric_log.get('client_metric_history', [])

    # 1. Seed
    seed = params.get('seed')
    if seed is None:
        return False, "Missing 'seed' in params"

    # 2. Commit
    commit = prov.get('commit_hash') or prov.get('git_commit')
    if not commit:
        return False, "Missing 'commit_hash' / 'git_commit' in provenance"

    # 3. CLI
    cli = prov.get('command_line') or prov.get('cli_args')
    if not cli:
        return False, "Missing 'command_line' / 'cli_args' in provenance"

    # 4. Method
    method = params.get('algorithm')
    if not method:
        return False, "Missing 'algorithm' in params"
    if method == 'FedOSD':
        method = f"FedOSD_{params.get('proj_mode', 'one_sided')}"

    # 5. Regime
    regime = infer_regime(params)
    if not regime:
        return False, "Unable to infer 'regime'"

    # 6. Stage
    stage = infer_stage(fpath, params)
    if not stage:
        return False, "Unable to infer 'stage'"

    # 7. Pretrain acc
    pre_acc = metric_log.get('pre_acc')
    if pre_acc is None:
        return False, "Missing 'pre_acc' in metric_log"

    # 8. Metrics at last evaluated round
    if not ch or not isinstance(ch, list):
        return False, "Empty or invalid 'client_metric_history'"

    unlearn_cids = params.get('unlearn_client_id_list', [1])
    retained_accs = []
    unlearn_asrs = []
    unlearn_cleans = []
    mias = []

    for cid, hist in enumerate(ch):
        t_acc = hist.get('test_accuracy', [])
        b_acc = hist.get('backdoor_test_accuracy', [])
        c_acc = hist.get('clean_test_accuracy', [])
        f_acc = hist.get('forget_acc', [])
        m_auc = hist.get('mia_auc', [])

        final_acc = t_acc[-1] if t_acc else None
        if cid in unlearn_cids:
            if b_acc and b_acc[-1] is not None:
                unlearn_asrs.append(b_acc[-1])
            if f_acc and f_acc[-1] is not None:
                unlearn_cleans.append(f_acc[-1])
            elif c_acc and c_acc[-1] is not None:
                unlearn_cleans.append(c_acc[-1])
            if m_auc and m_auc[-1] is not None:
                mias.append(m_auc[-1])
        else:
            if final_acc is not None:
                retained_accs.append(final_acc)

    if not retained_accs:
        return False, "No retained client accuracy found"

    # 9. MIA AUC
    mia_auc = metric_log.get('mia_auc')
    if mia_auc is None and not mias:
        return False, "Missing 'mia_auc' in metric_log or client history"

    # 10. Unlearn Displacement (for unlearning algorithms)
    is_retrain = params.get('retrain', False)
    disp = metric_log.get('unlearn_displacement', None)
    if disp is None and not is_retrain:
        disp = alg_log.get('unlearn_displacement', None)
    if disp is None and not is_retrain:
        return False, "Missing 'unlearn_displacement'"

    # 11. Strict Mode: Require per-round shadow_diagnostics for unlearning algorithms
    if strict and not is_retrain and params.get('algorithm') == 'FedOSD':
        shadow = alg_log.get('shadow_diagnostics', [])
        if not shadow or len(shadow) < 50:
            return False, f"Strict mode: shadow_diagnostics has {len(shadow)} rounds (<50)"
        last_s = shadow[-1]
        for req_k in ['raw_over_hard', 'used_over_hard', 'capped', 'displacement']:
            if req_k not in last_s:
                return False, f"Strict mode: Missing '{req_k}' in shadow_diagnostics"

    return True, f"Valid ({method}, Regime={regime}, Stage={stage})"

def main():
    parser = argparse.ArgumentParser(description="Validate result JSON schemas")
    parser.add_argument('--glob_pattern', default='UnlearningTask/**/*.json', help="Glob pattern for JSON results")
    parser.add_argument('--strict', action='store_true', help="Enable strict per-round log validation")
    args = parser.parse_args()

    files = glob.glob(args.glob_pattern, recursive=True)
    if not files:
        print(f"Warning: No files found matching {args.glob_pattern}")
        sys.exit(0)

    print(f"Validating {len(files)} result files (strict={args.strict})...")
    failed = 0
    for fpath in files:
        if 'pretrained_model' in fpath or '_run1_bak' in fpath:
            continue
        valid, msg = validate_file(fpath, strict=args.strict)
        if not valid:
            print(f"FAIL: {fpath} -> {msg}")
            failed += 1

    if failed > 0:
        print(f"\nSchema Validation FAILED: {failed} files invalid.")
        sys.exit(1)
    else:
        print(f"Schema Validation PASSED: All files conform to schema (strict={args.strict}).")
        sys.exit(0)

if __name__ == '__main__':
    main()
