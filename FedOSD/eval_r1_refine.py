import os
import sys
import json
import numpy as np
from eval_section_a import parse_metrics_with_worst2, BASE_R2

target_capped_disp = 0.2154
scale = sys.argv[1] if len(sys.argv) > 1 else '0.27'
seeds = [106, 107, 108]

print('=' * 105)
print(f'R1 HARD-SCALED AT s={scale} EVALUATION (SEEDS 106-108)')
print('=' * 105)
print(f"{'Seed':<6} | {'Mean Acc':<9} | {'Worst-2':<9} | {'Floor':<9} | {'Drop':<9} | {'ASR':<8} | {'Forget':<9} | {'Disp':<8} | {'Diff vs Capped'}")
print('-' * 105)

disps, means, worst2s, floors, drops, asrs, forgets = [], [], [], [], [], [], []

for s in seeds:
    fname = f'seed{s} N10 C1.0 FedOSD CNN_CIFAR10 E1 lr0.0004 decay0.999 g_clip-1.0 proj_hard lam100000.0 eta0.0 scale{scale}.json'
    fpath = os.path.join(BASE_R2, fname)
    if not os.path.exists(fpath):
        print(f'Seed {s}: File not found: {fpath}')
        continue
    res = parse_metrics_with_worst2(fpath)
    d = res['disp']
    diff_pct = (d - target_capped_disp) / target_capped_disp * 100
    disps.append(d)
    means.append(res['mean'])
    worst2s.append(res['worst2'])
    floors.append(res['floor'])
    drops.append(res['drop'])
    asrs.append(res['asr'])
    forgets.append(res['forget'])
    print(f"{s:<6} | {res['mean']:7.2f}% | {res['worst2']:7.2f}% | {res['floor']:7.2f}% | {res['drop']:7.2f}% | {res['asr']:6.2f}% | {res['forget']:7.2f}% | {d:8.4f} | {diff_pct:+6.2f}%")

print('-' * 105)
if len(disps) > 0:
    mean_disp = np.mean(disps)
    mean_diff_pct = (mean_disp - target_capped_disp) / target_capped_disp * 100
    print(f"{'Mean':<6} | {np.mean(means):7.2f}% | {np.mean(worst2s):7.2f}% | {np.mean(floors):7.2f}% | {np.mean(drops):7.2f}% | {np.mean(asrs):6.2f}% | {np.mean(forgets):7.2f}% | {mean_disp:8.4f} | {mean_diff_pct:+6.2f}%")
    print(f'Target Capped Displacement: {target_capped_disp:.4f}')
    print(f'Achieved Mean Displacement: {mean_disp:.4f}')
    print(f'Absolute Percentage Difference: {abs(mean_diff_pct):.2f}% (Target: within 2.0%)')
