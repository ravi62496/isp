import os
import glob
import json
import re
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from evaluate_reporting import parse_run_json
from experiment_suite import utility_at, paired_test, holm_bonferroni

OUTPUT_DIR = r"UnlearningTask/CNN_CIFAR10/cifar10 pat balanced B200 NC2 N10/UN1/E1/C1.0/FedOSD"
FIG_DIR = "figures"

def parse_run_with_displacement(fpath):
    with open(fpath, 'r') as f:
        data = json.load(f)
    res = parse_run_json(fpath)
    
    # Extract displacement: check metric_log, algorithm_log, or root
    disp = data.get('unlearn_displacement', None)
    if disp is None:
        disp = data.get('metric_log', {}).get('unlearn_displacement', None)
    if disp is None:
        disp = data.get('algorithm_log', {}).get('unlearn_displacement', None)
    if disp is None:
        dists = data.get('algorithm_log', {}).get('distance_to_init_model', [])
        if dists:
            disp = float(dists[-1])
    if disp is None:
        # Fallback to last recorded distance if present
        disp = float(data.get('algorithm_log', {}).get('distance', 0.0))
        
    res['displacement'] = float(disp) if disp is not None else 0.0
    res['fpath'] = fpath
    res['params'] = data.get('params', {})
    res['shadow_diagnostics'] = data.get('algorithm_log', {}).get('shadow_diagnostics', [])
    return res

def analyze_matched_strength(seeds=[101, 102, 103]):
    os.makedirs(FIG_DIR, exist_ok=True)
    print("=" * 100)
    print(f"MATCHED-STRENGTH CONTROL ANALYSIS (Tuning Seeds {seeds})")
    print("=" * 100)
    
    hard_data = defaultdict(dict)       # seed -> scale -> res
    capped_data = defaultdict(dict)     # seed -> cap -> res
    
    # 1. Collect Hard-scaled runs
    # Scale 1.0 is the baseline Hard run
    for s in seeds:
        # 1.0
        f_base = os.path.join(OUTPUT_DIR, f"seed{s} N10 C1.0 FedOSD CNN_CIFAR10 E1 lr0.0004 decay0.999 g_clip-1.0 proj_hard lam100000.0 eta0.0 rlr0.0001.json")
        if os.path.exists(f_base):
            hard_data[s][1.0] = parse_run_with_displacement(f_base)
            
        for scale in [0.25, 0.5, 0.75]:
            f_scale = os.path.join(OUTPUT_DIR, f"seed{s} N10 C1.0 FedOSD CNN_CIFAR10 E1 lr0.0004 decay0.999 g_clip-1.0 proj_hard lam100000.0 eta0.0 scale{scale} rlr0.0001.json")
            if os.path.exists(f_scale):
                hard_data[s][scale] = parse_run_with_displacement(f_scale)
                
    # 2. Collect Capped One-Sided runs
    for s in seeds:
        for cap in [0.5, 0.75, 1.0, 1.25, 1.5]:
            f_cap = os.path.join(OUTPUT_DIR, f"seed{s} N10 C1.0 FedOSD CNN_CIFAR10 E1 lr0.0004 decay0.999 g_clip-1.0 proj_one_sided_cap{cap} wm_uniform lam100000.0 rlr0.0001.json")
            if os.path.exists(f_cap):
                capped_data[s][cap] = parse_run_with_displacement(f_cap)

    print(f"Hard-scaled runs found per seed: { {s: list(v.keys()) for s, v in hard_data.items()} }")
    print(f"Capped One-Sided runs found per seed: { {s: list(v.keys()) for s, v in capped_data.items()} }")
    
    # Step-Norm Inflation Table: Extract d_ratio from shadow_diagnostics
    print("\n" + "=" * 100)
    print("STEP-NORM RATIO EVIDENCE (||d_used|| / ||d_hard||)")
    print("=" * 100)
    print(f"{'Seed':<6} | {'Cap Ratio':<10} | {'Median Ratio':<14} | {'Max Ratio':<12} | {'Rounds Inflated (>1.0)':<25}")
    print("-" * 100)
    
    for s in sorted(capped_data.keys()):
        for cap in sorted(capped_data[s].keys()):
            diag = capped_data[s][cap].get('shadow_diagnostics', [])
            if diag:
                ratios = [d['d_ratio'] for d in diag if 'd_ratio' in d]
                if ratios:
                    med_r = np.median(ratios)
                    max_r = np.max(ratios)
                    infl_pct = 100.0 * np.sum(np.array(ratios) > 1.0) / len(ratios)
                    print(f"{s:<6} | {cap:<10} | {med_r:12.4f}  | {max_r:10.4f}  | {infl_pct:6.1f}% ({np.sum(np.array(ratios) > 1.0)}/{len(ratios)} rounds)")

    # Compute matched-strength paired differences at cap=1.0 displacement
    print("\n" + "=" * 100)
    print("PAIRED MATCHED-STRENGTH COMPARISON: Capped (cap=1.0) vs Hard-scaled at Matched Displacement")
    print("=" * 100)
    
    diff_means = []
    diff_floors = []
    diff_drops = []
    
    for s in seeds:
        if 1.0 in capped_data[s] and len(hard_data[s]) >= 2:
            c_res = capped_data[s][1.0]
            target_disp = c_res['displacement']
            
            # Build Hard curve: points = [(disp, metric), ...]
            hard_pts_mean = [(v['displacement'], v['mean']) for v in hard_data[s].values()]
            hard_pts_floor = [(v['displacement'], v['floor']) for v in hard_data[s].values()]
            hard_pts_drop = [(v['displacement'], v['max_drop']) for v in hard_data[s].values()]
            
            h_interp_mean = utility_at(hard_pts_mean, target_disp)
            h_interp_floor = utility_at(hard_pts_floor, target_disp)
            h_interp_drop = utility_at(hard_pts_drop, target_disp)
            
            d_mean = c_res['mean'] - h_interp_mean
            d_floor = c_res['floor'] - h_interp_floor
            d_drop = c_res['max_drop'] - h_interp_drop
            
            diff_means.append(d_mean)
            diff_floors.append(d_floor)
            diff_drops.append(d_drop)
            
            print(f"Seed {s}: Target Disp={target_disp:.4f}")
            print(f"  Mean:      Capped={c_res['mean']:.2f}% | Hard-Scaled={h_interp_mean:.2f}% | Diff={d_mean:+.2f}%")
            print(f"  Floor:     Capped={c_res['floor']:.2f}% | Hard-Scaled={h_interp_floor:.2f}% | Diff={d_floor:+.2f}%")
            print(f"  Max Drop:  Capped={c_res['max_drop']:.2f}% | Hard-Scaled={h_interp_drop:.2f}% | Diff={d_drop:+.2f}%")

    if diff_means:
        rep_mean = paired_test(diff_means, [0]*len(diff_means))
        rep_floor = paired_test(diff_floors, [0]*len(diff_floors))
        rep_drop = paired_test(diff_drops, [0]*len(diff_drops))
        
        print("\n" + "-" * 100)
        print("MATCHED-STRENGTH TEST SUMMARY:")
        print(f"  Retained Mean Diff: {rep_mean['mean_diff']:+.2f}% (95% CI: [{rep_mean['ci95'][0]:+.2f}%, {rep_mean['ci95'][1]:+.2f}%], Signs: {rep_mean['signs']})")
        print(f"  Worst-Off Floor Diff: {rep_floor['mean_diff']:+.2f}% (95% CI: [{rep_floor['ci95'][0]:+.2f}%, {rep_floor['ci95'][1]:+.2f}%], Signs: {rep_floor['signs']})")
        print(f"  Max Drop Diff: {rep_drop['mean_diff']:+.2f}% (95% CI: [{rep_drop['ci95'][0]:+.2f}%, {rep_drop['ci95'][1]:+.2f}%], Signs: {rep_drop['signs']})")
        print("-" * 100)

if __name__ == '__main__':
    analyze_matched_strength()

