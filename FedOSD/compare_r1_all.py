import os
import glob
import numpy as np
from eval_section_a import parse_metrics_with_worst2, BASE_R2

methods = [
    ('Hard rescaled to ||g_u|| (scale 1.0)', lambda b: 'proj_hard' in b and 'scale' not in b),
    ('Hard-scaled (scale 0.50)', lambda b: 'scale0.5' in b and 'scale0.54' not in b),
    ('Hard-scaled (scale 0.24)', lambda b: 'scale0.24' in b),
    ('Hard-scaled (scale 0.27, refined)', lambda b: 'scale0.27' in b),
    ('One-sided, unscaled (cap 1.0)', lambda b: 'cap_hard_ratio1.0' in b or 'cap1.0' in b),
    ('One-sided rescaled to ||g_u||', lambda b: 'proj_one_sided' in b and 'cap' not in b)
]

seeds = [106, 107, 108]

print("=" * 115)
print("COMPREHENSIVE R1 EVALUATION (SEEDS 106-108)")
print("=" * 115)

for m_name, filter_fn in methods:
    print(f"\n--- Method: {m_name} ---")
    print(f"{'Seed':<6} | {'Mean Acc':<9} | {'Worst-2':<9} | {'Floor':<9} | {'Drop':<9} | {'ASR':<8} | {'Forget':<9} | {'Disp':<8}")
    print("-" * 75)
    m_disps, m_means, m_worst2s, m_floors, m_drops, m_asrs, m_forgets = [], [], [], [], [], [], []
    for s in seeds:
        pattern = os.path.join(BASE_R2, f'seed{s}*.json')
        files = glob.glob(pattern)
        matched = [f for f in files if filter_fn(os.path.basename(f))]
        if not matched:
            print(f"{s:<6} | MISSING")
            continue
        res = parse_metrics_with_worst2(matched[0])
        m_disps.append(res['disp'])
        m_means.append(res['mean'])
        m_worst2s.append(res['worst2'])
        m_floors.append(res['floor'])
        m_drops.append(res['drop'])
        m_asrs.append(res['asr'])
        m_forgets.append(res['forget'])
        print(f"{s:<6} | {res['mean']:7.2f}% | {res['worst2']:7.2f}% | {res['floor']:7.2f}% | {res['drop']:7.2f}% | {res['asr']:6.2f}% | {res['forget']:7.2f}% | {res['disp']:8.4f}")
    if m_disps:
        print("-" * 75)
        print(f"{'Mean':<6} | {np.mean(m_means):7.2f}% | {np.mean(m_worst2s):7.2f}% | {np.mean(m_floors):7.2f}% | {np.mean(m_drops):7.2f}% | {np.mean(m_asrs):6.2f}% | {np.mean(m_forgets):7.2f}% | {np.mean(m_disps):8.4f}")

print("\n" + "=" * 115)

