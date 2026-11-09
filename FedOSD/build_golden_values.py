#!/usr/bin/env python3
"""
build_golden_values.py
Builds golden_values.json directly from the raw files registered in RUN_MANIFEST.csv.
Includes: sha1, mean, worst-2, floor, drop, ASR, forget-set, displacement.
For arms A1-A4 x seeds 101-105 at Round 100 (unlearn) and Round 200 (recovery),
plus Retrain at Round 200 and Round 300.
"""

import os
import csv
import json
import hashlib
import numpy as np

UNL = {
    101: 7, 102: 8, 103: 6, 104: 0, 105: 3,
    106: 9, 107: 2, 108: 8
}

def extract_from_file(fpath, target_round, stage, seed):
    with open(fpath, 'r', encoding='utf-8') as f:
        d = json.load(f)
        
    sha1 = hashlib.sha1(open(fpath, 'rb').read()).hexdigest()
    ch = d['metric_log']['client_metric_history']
    unl_cid = UNL[seed]
    pre_acc = d.get('metric_log', {}).get('pre_acc', {})
    is_retrain = ('retrain' in os.path.basename(fpath).lower())
    
    arr_len = len(ch[0]['test_accuracy']) if isinstance(ch, list) else len(ch['test_accuracy'])
    
    if is_retrain:
        assert arr_len == 61, f"Expected 61, got {arr_len}"
        idx = 40 if target_round == 200 else 60
    elif stage == 'unlearn' or target_round == 100:
        assert arr_len == 100, f"Expected 100, got {arr_len}"
        idx = 99
    elif stage == 'recovery' or target_round == 200:
        assert arr_len in [101, 201], f"Expected 101 or 201, got {arr_len}"
        idx = arr_len - 1
    else:
        idx = arr_len - 1
        
    retained_accs, retained_drops = [], []
    for cid in range(10):
        t_acc = ch[cid]['test_accuracy']
        init_acc = pre_acc.get(str(cid), t_acc[0] if t_acc else 0.0)
        final_acc = t_acc[idx]
        if cid == unl_cid:
            b_acc = ch[cid].get('backdoor_test_accuracy', [])
            f_acc = ch[cid].get('forget_acc', [])
            c_acc = ch[cid].get('clean_test_accuracy', [])
            cur_b = b_acc[idx] if b_acc and idx < len(b_acc) else final_acc
            cur_f = f_acc[idx] if f_acc and idx < len(f_acc) else (c_acc[idx] if c_acc and idx < len(c_acc) else final_acc)
            asr = float(cur_b)
            forget = float(cur_f)
        else:
            retained_accs.append(final_acc)
            retained_drops.append(init_acc - final_acc)
            
    sorted_accs = sorted(retained_accs)
    mean_val = float(np.mean(retained_accs))
    worst2_val = float(np.mean(sorted_accs[:2]))
    floor_val = float(sorted_accs[0])
    drop_val = float(np.max(retained_drops)) if not is_retrain else 0.0
    
    dists = d.get('algorithm_log', {}).get('distance_to_init_model', [])
    unl_disp = d.get('unlearn_displacement') or d.get('algorithm_log', {}).get('unlearn_displacement')
    if unl_disp is None and dists and len(dists) >= 100:
        unl_disp = dists[99]

    if (stage == 'recovery' or target_round == 200) and dists:
        disp_val = float(dists[-1]) if not is_retrain else 0.0
    elif target_round is not None and target_round <= 100 and dists and len(dists) >= 100:
        disp_val = float(dists[target_round - 1]) if not is_retrain else 0.0
    else:
        disp_val = float(unl_disp if unl_disp is not None else (dists[-1] if dists else 0.0)) if not is_retrain else 0.0
    
    return {
        'sha1': sha1,
        'mean': mean_val,
        'worst2': worst2_val,
        'floor': floor_val,
        'drop': drop_val,
        'asr': asr,
        'forget': forget,
        'disp': disp_val
    }

def main():
    golden = {}
    with open('RUN_MANIFEST.csv', 'r', newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            arm = row['arm']
            seed = int(row['seed'])
            stage = row['stage']
            fpath = row.get('absolute_file_path') or row.get('path')
            
            if arm == 'Retrain':
                target_round = 200 if stage == 'R200' else 300
                res = extract_from_file(fpath, target_round, stage, seed)
                golden[f"Retrain_{seed}_rd{target_round}"] = res
                golden[f"Retrain_{seed}_{stage}"] = res
            elif stage == 'unlearn':
                res = extract_from_file(fpath, 100, stage, seed)
                golden[f"{arm}_{seed}_rd100"] = res
                golden[f"{arm}_{seed}_unlearn"] = res
            elif stage == 'recovery':
                res = extract_from_file(fpath, 200, stage, seed)
                golden[f"{arm}_{seed}_rd200"] = res
                golden[f"{arm}_{seed}_recovery"] = res

    # Also include Regime R1 manifest (seeds 106-108)
    if os.path.exists('RUN_MANIFEST_R1.csv'):
        with open('RUN_MANIFEST_R1.csv', 'r', newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                arm = row['arm']
                seed = int(row['seed'])
                stage = row['stage']
                fpath = row.get('absolute_file_path') or row.get('path')
                target_round = 100 if stage == 'unlearn' else 200
                res = extract_from_file(fpath, target_round, stage, seed)
                golden[f"R1_{arm}_{seed}_{stage}"] = res
                golden[f"R1_{arm}_{seed}_rd{target_round}"] = res
                golden[f"{arm}_{seed}_{stage}"] = res
                golden[f"{arm}_{seed}_rd{target_round}"] = res
                
    out_path = 'golden_values.json'
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(golden, f, indent=2)
    print(f"Generated {out_path} with {len(golden)} entries.")

if __name__ == '__main__':
    main()

