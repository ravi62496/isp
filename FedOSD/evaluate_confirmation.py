#!/usr/bin/env python3
"""
evaluate_confirmation.py
Pre-registered, frozen evaluation script for Stage 4 confirmatory analysis.
Compliant with SUCCESS_CRITERIA_v5.1 protocol.

Evaluates seeds 21-30 (n=10) on Primary Regime R2.
Confirmatory family: A3 (One-sided, unscaled cap 1.0) vs A2 (Hard-scaled s*=0.54) at Round 100 (m=3, Holm).
Exploratory families: Round 200 post-recovery, A3 vs A4, A3 vs A1, and Sensitivity Control A1 vs A2.
"""

import os
import sys
import glob
import json
import argparse
import numpy as np
import scipy.stats as stats
from collections import OrderedDict

# Protocol Constants
DEFAULT_SEEDS = list(range(21, 31))  # [21, 22, 23, 24, 25, 26, 27, 28, 29, 30]
RESULTS_CONFIRM_DIR = "results_confirm"
BASE_R2_DEFAULT = os.path.join(
    "UnlearningTask", "CNN_CIFAR10",
    "cifar10 pat balanced B200 NC2 N10", "UN1", "E1", "C1.0", "FedOSD"
)
BASE_RETRAIN_DEFAULT = os.path.join(
    "UnlearningTask", "CNN_CIFAR10",
    "cifar10 pat balanced B200 NC2 N10", "UN1", "E1", "C1.0", "FedAvg"
)

# Arms under evaluation
ARMS = {
    'A1': 'Hard FedOSD (rescaled to ||g_u||, scale 1.0)',
    'A2': 'Hard-scaled (step_scale 0.54)',
    'A3': 'One-sided, unscaled (cap 1.0) [Method Under Test]',
    'A4': 'One-sided rescaled to ||g_u||',
    'Retrain': 'Retrain from Scratch (R=300)'
}

# Primary Metrics Specification (SUCCESS_CRITERIA_v5.1)
PRIMARY_METRIC_SPECS = {
    'mean': {
        'name': 'Retained Cohort Mean Acc',
        'practical': 1.0,      # practical threshold >= 1.0 pp
        'equiv_margin': 0.5,   # equivalence margin [-0.5, +0.5] pp
        'higher_better': True  # method minus comparator
    },
    'worst2': {
        'name': 'Worst-2 Clients Mean Acc',
        'practical': 2.0,      # practical threshold >= 2.0 pp
        'equiv_margin': 1.0,   # equivalence margin [-1.0, +1.0] pp
        'higher_better': True  # method minus comparator
    },
    'drop': {
        'name': 'Max Collateral Drop',
        'practical': 2.0,      # practical threshold >= 2.0 pp
        'equiv_margin': 1.0,   # equivalence margin [-1.0, +1.0] pp
        'higher_better': False # comparator minus method (lower drop is better)
    }
}

def parse_args():
    parser = argparse.ArgumentParser(description="Stage 4 Confirmatory Evaluator (Protocol v5.1).")
    parser.add_argument('--seeds', type=int, nargs='+', default=DEFAULT_SEEDS,
                        help="List of seeds to evaluate (default: 21-30).")
    parser.add_argument('--results_dir', type=str, default=RESULTS_CONFIRM_DIR,
                        help=f"Path to sealed results directory (default: {RESULTS_CONFIRM_DIR}).")
    parser.add_argument('--fallback_to_unlearn_dirs', action='store_true', default=True,
                        help="Fallback to default UnlearningTask directories if results_confirm does not have the files.")
    parser.add_argument('--bootstrap_samples', type=int, default=10000,
                        help="Number of bootstrap resamples (protocol specifies 10,000).")
    parser.add_argument('--seed_rng', type=int, default=42,
                        help="Random seed for bootstrap resampling.")
    return parser.parse_args()

def locate_result_file(arm_key, seed, stage='unlearn', results_dir=RESULTS_CONFIRM_DIR, fallback=True):
    """Locate the JSON result file for a given arm, seed, and stage."""
    suffix = '_unlearn.json' if stage == 'unlearn' else '.json'
    rlr_str = "rlr0.0001"
    
    if arm_key == 'A1':
        fname = f"seed{seed} N10 C1.0 FedOSD CNN_CIFAR10 E1 lr0.0004 decay0.999 g_clip-1.0 proj_hard lam100000.0 eta0.0 {rlr_str}" + suffix
    elif arm_key == 'A2':
        fname = f"seed{seed} N10 C1.0 FedOSD CNN_CIFAR10 E1 lr0.0004 decay0.999 g_clip-1.0 proj_hard lam100000.0 eta0.0 scale0.54 {rlr_str}" + suffix
    elif arm_key == 'A3':
        fname = f"seed{seed} N10 C1.0 FedOSD CNN_CIFAR10 E1 lr0.0004 decay0.999 g_clip-1.0 proj_one_sided_cap1.0 wm_uniform lam100000.0 {rlr_str}" + suffix
    elif arm_key == 'A4':
        fname = f"seed{seed} N10 C1.0 FedOSD CNN_CIFAR10 E1 lr0.0004 decay0.999 g_clip-1.0 proj_one_sided wm_uniform lam100000.0 eta0.0 {rlr_str}" + suffix
    elif arm_key == 'Retrain':
        fname = f"seed{seed} retrain_from_scratch seed{seed} N10 C1.0 FedAvg CNN_CIFAR10 E1 lr0.05 decay0.999 g_clip-1.0 R300.json"
    else:
        raise ValueError(f"Unknown arm: {arm_key}")

    # 1. Check results_confirm directory
    p1 = os.path.join(results_dir, fname)
    if os.path.exists(p1):
        return p1
        
    # 2. Fallback check
    if fallback:
        base_dir = BASE_RETRAIN_DEFAULT if arm_key == 'Retrain' else BASE_R2_DEFAULT
        p2 = os.path.join(base_dir, fname)
        if os.path.exists(p2):
            return p2
            
    return None

def parse_metrics_at_round(fpath, target_round=None):
    """Robust parser for both unlearning and retrain files at specified round."""
    if not os.path.exists(fpath):
        return None
    with open(fpath, 'r') as f:
        d = json.load(f)
        
    ch = d.get('metric_log', {}).get('client_metric_history', [])
    unlearn_cids = d.get('params', {}).get('unlearn_client_id_list', [])
    pre_acc = d.get('metric_log', {}).get('pre_acc', {})
    test_interval = d.get('params', {}).get('test_interval', 1)
    
    retained_accs, retained_drops = [], []
    asrs, cleans, mias = [], [], []
    
    for cid, hist in enumerate(ch):
        t_acc = hist.get('test_accuracy', [])
        b_acc = hist.get('backdoor_test_accuracy', [])
        c_acc = hist.get('clean_test_accuracy', [])
        f_acc = hist.get('forget_acc', [])
        m_auc = hist.get('mia_auc', [])
        
        if target_round is not None:
            idx = min(len(t_acc) - 1, target_round // test_interval)
            final_acc = t_acc[idx] if t_acc else 0.0
            cur_b_acc = b_acc[idx] if b_acc else final_acc
            cur_c_acc = c_acc[idx] if c_acc else final_acc
            cur_f_acc = f_acc[idx] if f_acc else cur_c_acc
            cur_m_auc = m_auc[idx] if m_auc else 0.5
        else:
            final_acc = t_acc[-1] if t_acc else 0.0
            cur_b_acc = b_acc[-1] if b_acc else final_acc
            cur_c_acc = c_acc[-1] if c_acc else final_acc
            cur_f_acc = f_acc[-1] if f_acc else cur_c_acc
            cur_m_auc = m_auc[-1] if m_auc else 0.5

        init_acc = pre_acc.get(str(cid), t_acc[0] if t_acc else 0.0)
        if cid in unlearn_cids:
            asrs.append(cur_b_acc)
            cleans.append(cur_f_acc)
            mias.append(cur_m_auc)
        else:
            retained_accs.append(final_acc)
            retained_drops.append(init_acc - final_acc)
            
    sorted_accs = sorted(retained_accs)
    
    disp = d.get('unlearn_displacement') or d.get('algorithm_log', {}).get('unlearn_displacement')
    if disp is None:
        dists = d.get('algorithm_log', {}).get('distance_to_init_model', [])
        if target_round is not None and target_round <= 100 and len(dists) >= 100:
            disp = dists[target_round - 1]
        else:
            disp = dists[-1] if dists else 0.0
            
    solver_times = d.get('algorithm_log', {}).get('solver_time_ms', [])
    median_solver_ms = float(np.median(solver_times)) if solver_times else 0.0
    
    round_times = d.get('metric_log', {}).get('round_duration_sec', [])
    median_round_s = float(np.median(round_times)) if round_times else 0.0

    return {
        'mean': float(np.mean(retained_accs)) if retained_accs else 0.0,
        'worst2': float(np.mean(sorted_accs[:2])) if len(sorted_accs) >= 2 else 0.0,
        'floor': float(sorted_accs[0]) if sorted_accs else 0.0,
        'drop': float(np.max(retained_drops)) if retained_drops else 0.0,
        'asr': float(np.mean(asrs)) if asrs else 0.0,
        'forget': float(np.mean(cleans)) if cleans else 0.0,
        'mia': float(np.mean(mias)) if mias else 0.5,
        'disp': float(disp),
        'solver_time_ms': median_solver_ms,
        'round_time_s': median_round_s
    }

def bootstrap_ci(diffs, n_resamples=10000, alpha=0.05, seed=42):
    """Compute 10,000-resample bootstrap percentile confidence interval."""
    rng = np.random.default_rng(seed)
    n = len(diffs)
    boot_means = np.empty(n_resamples)
    for i in range(n_resamples):
        sample = rng.choice(diffs, size=n, replace=True)
        boot_means[i] = np.mean(sample)
    low = np.percentile(boot_means, 100.0 * (alpha / 2.0))
    high = np.percentile(boot_means, 100.0 * (1.0 - alpha / 2.0))
    return float(low), float(high)

def compute_hypothesis_tests(diffs, practical_thresh, equiv_margin, n_resamples=10000, seed=42):
    """
    Performs paired t-test, bootstrap 95% CI, Wilcoxon signed-rank test,
    and 90% CI for equivalence.
    """
    diffs = np.array(diffs, dtype=np.float64)
    n = len(diffs)
    mean_diff = float(np.mean(diffs))
    se = float(stats.sem(diffs)) if n > 1 else 0.0
    
    # Paired t-test
    if n > 1 and se > 1e-12:
        t_res = stats.ttest_1samp(diffs, 0.0)
        t_stat = float(t_res.statistic)
        p_t = float(t_res.pvalue)
        # 95% two-sided CI
        t_crit_95 = stats.t.ppf(0.975, df=n - 1)
        ci95_t = (mean_diff - t_crit_95 * se, mean_diff + t_crit_95 * se)
        # 90% two-sided CI for equivalence (alpha=0.05 one-sided each)
        t_crit_90 = stats.t.ppf(0.95, df=n - 1)
        ci90_t = (mean_diff - t_crit_90 * se, mean_diff + t_crit_90 * se)
    else:
        t_stat, p_t = 0.0, 1.0
        ci95_t = (mean_diff, mean_diff)
        ci90_t = (mean_diff, mean_diff)

    # Bootstrap 95% CI
    ci95_boot = bootstrap_ci(diffs, n_resamples=n_resamples, alpha=0.05, seed=seed)
    
    # Wilcoxon signed-rank test
    nonzero_diffs = diffs[diffs != 0.0]
    if len(nonzero_diffs) >= 5:
        try:
            w_res = stats.wilcoxon(diffs, alternative='two-sided')
            p_wilcox = float(w_res.pvalue)
        except Exception:
            p_wilcox = 1.0
    else:
        p_wilcox = 1.0

    return {
        'n': n,
        'mean_diff': mean_diff,
        'se': se,
        't_stat': t_stat,
        'p_t': p_t,
        'ci95_t': ci95_t,
        'ci90_t': ci90_t,
        'ci95_boot': ci95_boot,
        'p_wilcox': p_wilcox,
        'practical_thresh': practical_thresh,
        'equiv_margin': equiv_margin
    }

def apply_holm_adjustment(p_values):
    """
    Apply Holm step-down procedure to a list of p-values.
    Returns array of adjusted p-values matching original order.
    """
    m = len(p_values)
    indexed_p = sorted(enumerate(p_values), key=lambda x: x[1])
    adjusted = [0.0] * m
    running_max = 0.0
    for rank, (orig_idx, p_val) in enumerate(indexed_p):
        multiplier = m - rank
        adj_p = p_val * multiplier
        running_max = max(running_max, adj_p)
        adjusted[orig_idx] = min(1.0, running_max)
    return adjusted

def assign_label(test_res, holm_p):
    """
    Protocol Label Precedence:
    Superior: 95% CI > 0, Holm p < 0.05, estimate >= practical
    Inferior: 95% CI < 0, Holm p < 0.05
    Equivalent: 90% CI inside margin [-margin, +margin]
    Better but negligible: 95% CI > 0, Holm p < 0.05, estimate < practical
    Inconclusive: anything else
    """
    ci95_low, ci95_high = test_res['ci95_t']
    ci90_low, ci90_high = test_res['ci90_t']
    est = test_res['mean_diff']
    pract = test_res['practical_thresh']
    margin = test_res['equiv_margin']
    
    is_pos_95 = (ci95_low > 0.0)
    is_neg_95 = (ci95_high < 0.0)
    is_sig = (holm_p < 0.05)
    
    # 1. Superior
    if is_pos_95 and is_sig and est >= pract:
        return 'Superior'
        
    # 2. Inferior
    if is_neg_95 and is_sig:
        return 'Inferior'
        
    # 3. Equivalent (90% CI strictly contained within [-margin, +margin])
    if ci90_low >= -margin and ci90_high <= margin:
        return 'Equivalent'
        
    # 4. Better but negligible
    if is_pos_95 and is_sig and est < pract:
        return 'Better but negligible'
        
    # 5. Inconclusive
    return 'Inconclusive'

def check_strength_matching(a3_runs, a2_runs):
    """
    Strength matched only if, on seed averages at round 100:
    - displacement ratio in [0.95, 1.05]
    - |ASR diff| <= 1.0 pt
    - |forget-set diff| <= 2.0 pt
    """
    disp_a3 = np.mean([r['disp'] for r in a3_runs])
    disp_a2 = np.mean([r['disp'] for r in a2_runs])
    disp_ratio = disp_a3 / (disp_a2 + 1e-12)
    
    asr_a3 = np.mean([r['asr'] for r in a3_runs])
    asr_a2 = np.mean([r['asr'] for r in a2_runs])
    asr_diff = abs(asr_a3 - asr_a2)
    
    fg_a3 = np.mean([r['forget'] for r in a3_runs])
    fg_a2 = np.mean([r['forget'] for r in a2_runs])
    fg_diff = abs(fg_a3 - fg_a2)
    
    cond_disp = (0.95 <= disp_ratio <= 1.05)
    cond_asr = (asr_diff <= 1.0)
    cond_fg = (fg_diff <= 2.0)
    
    is_matched = bool(cond_disp and cond_asr and cond_fg)
    
    return {
        'is_matched': is_matched,
        'disp_ratio': disp_ratio,
        'disp_pass': cond_disp,
        'asr_diff': asr_diff,
        'asr_pass': cond_asr,
        'fg_diff': fg_diff,
        'fg_pass': cond_fg
    }

def evaluate_family(arm_test, arm_comp, stage_name, round_num, retrain_runs, seeds, args, is_confirmatory=False):
    """Evaluate a family of 3 primary hypotheses with Holm adjustment."""
    family_label = "[CONFIRMATORY FAMILY]" if is_confirmatory else "[EXPLORATORY FAMILY]"
    print("\n" + "=" * 110)
    print(f"{family_label} {ARMS[arm_test]} vs {ARMS[arm_comp]} at Round {round_num} ({stage_name})")
    print("=" * 110)
    
    test_runs = []
    comp_runs = []
    active_seeds = []
    
    for s in seeds:
        f_test = locate_result_file(arm_test, s, stage=stage_name, results_dir=args.results_dir, fallback=args.fallback_to_unlearn_dirs)
        f_comp = locate_result_file(arm_comp, s, stage=stage_name, results_dir=args.results_dir, fallback=args.fallback_to_unlearn_dirs)
        
        if f_test and f_comp:
            test_runs.append(parse_metrics_at_round(f_test, target_round=round_num))
            comp_runs.append(parse_metrics_at_round(f_comp, target_round=round_num))
            active_seeds.append(s)
        else:
            print(f"Seed {s}: Incomplete runs (Test: {'Found' if f_test else 'Missing'}, Comp: {'Found' if f_comp else 'Missing'})")
            
    n = len(active_seeds)
    if n == 0:
        print("ERROR: No completed seed runs found for this comparison.")
        return None
        
    print(f"Completed Seeds (n={n}): {active_seeds}")
    
    match_stat = None
    if arm_test == 'A3' and arm_comp == 'A2' and round_num == 100:
        match_stat = check_strength_matching(test_runs, comp_runs)
        print("\n--- Strength Matching Evaluation (Round 100 Seed Averages) ---")
        print(f"Displacement Ratio (A3 / A2): {match_stat['disp_ratio']:.4f} "
              f"[{'PASS [0.95, 1.05]' if match_stat['disp_pass'] else 'FAIL'}]")
        print(f"|ASR Difference|:             {match_stat['asr_diff']:.2f} pt "
              f"[{'PASS <= 1.0 pt' if match_stat['asr_pass'] else 'FAIL'}]")
        print(f"|Forget-Set Acc Difference|:  {match_stat['fg_diff']:.2f} pt "
              f"[{'PASS <= 2.0 pt' if match_stat['fg_pass'] else 'FAIL'}]")
        if match_stat['is_matched']:
            print(">>> STATUS: STRENGTH MATCHED <<<")
        else:
            print(">>> STATUS: STRENGTH NOT MATCHED <<< (Trigger contingency if applicable)")

    diff_dict = {}
    test_res_dict = {}
    p_values = []
    
    metric_keys = ['mean', 'worst2', 'drop']
    for m in metric_keys:
        spec = PRIMARY_METRIC_SPECS[m]
        if spec['higher_better']:
            diffs = [t[m] - c[m] for t, c in zip(test_runs, comp_runs)]
        else:
            diffs = [c[m] - t[m] for t, c in zip(test_runs, comp_runs)]
        diff_dict[m] = diffs
        tres = compute_hypothesis_tests(
            diffs, practical_thresh=spec['practical'],
            equiv_margin=spec['equiv_margin'],
            n_resamples=args.bootstrap_samples, seed=args.seed_rng
        )
        test_res_dict[m] = tres
        p_values.append(tres['p_t'])
        
    holm_p_values = apply_holm_adjustment(p_values)
    
    ret_asr = np.mean([r['asr'] for r in retrain_runs]) if retrain_runs else 0.0
    ret_fg = np.mean([r['forget'] for r in retrain_runs]) if retrain_runs else 0.0
    
    t_asr = np.mean([r['asr'] for r in test_runs])
    c_asr = np.mean([r['asr'] for r in comp_runs])
    t_fg = np.mean([r['forget'] for r in test_runs])
    c_fg = np.mean([r['forget'] for r in comp_runs])

    print(f"\n--- Primary Hypothesis Testing (Family Size m=3, Holm Step-Down) ---")
    print(f"{'Primary Metric':<24} | {'Diff Est':<9} | {'95% CI (t-test)':<22} | {'95% CI (boot)':<22} | {'p (t)':<8} | {'p (Holm)':<8} | {'p (Wilcox)':<10} | {'Decision Label'}")
    print("-" * 130)
    
    labels = {}
    for idx, m in enumerate(metric_keys):
        tres = test_res_dict[m]
        spec = PRIMARY_METRIC_SPECS[m]
        hp = holm_p_values[idx]
        lbl = assign_label(tres, hp)
        labels[m] = lbl
        ci_t_str = f"[{tres['ci95_t'][0]:+6.2f}, {tres['ci95_t'][1]:+6.2f}]"
        ci_b_str = f"[{tres['ci95_boot'][0]:+6.2f}, {tres['ci95_boot'][1]:+6.2f}]"
        print(f"{spec['name']:<24} | {tres['mean_diff']:+7.2f}% | {ci_t_str:<22} | {ci_b_str:<22} | {tres['p_t']:7.4f} | {hp:7.4f} | {tres['p_wilcox']:9.4f} | {lbl}")
        
    print("-" * 130)
    print("Retrain Reference Anchors (evaluated alongside every utility claim):")
    print(f"  Retrain ASR: {ret_asr:.2f}% | Test Dist: {abs(t_asr - ret_asr):.2f} pt | Comp Dist: {abs(c_asr - ret_asr):.2f} pt")
    print(f"  Retrain Forget: {ret_fg:.2f}% | Test Dist: {abs(t_fg - ret_fg):.2f} pt | Comp Dist: {abs(c_fg - ret_fg):.2f} pt")
    
    outcome = None
    if is_confirmatory:
        matched = match_stat['is_matched'] if match_stat else False
        is_sup_mean = (labels['mean'] == 'Superior')
        is_sup_w2_or_drop = (labels['worst2'] == 'Superior' or labels['drop'] == 'Superior')
        has_inferior = any(l == 'Inferior' for l in labels.values())
        all_equivalent = all(l == 'Equivalent' for l in labels.values())
        
        if matched and is_sup_mean and is_sup_w2_or_drop and not has_inferior:
            outcome = 'Outcome A'
            outcome_desc = "Outcome A: Strength matched; Superior on mean and on worst-2 or drop; none Inferior."
        elif matched and all_equivalent:
            outcome = 'Outcome B'
            outcome_desc = "Outcome B: Strength matched; Equivalent on mean, worst-2 and drop."
        else:
            outcome = 'Outcome C'
            reasons = []
            if not matched: reasons.append("strength not matched")
            if has_inferior: reasons.append("inferior metric present")
            if not (is_sup_mean and is_sup_w2_or_drop) and not all_equivalent: reasons.append("neither superiority nor full equivalence established")
            outcome_desc = f"Outcome C: {'; '.join(reasons) if reasons else 'other criteria not met'}."

        print(f"\n==========================================================================================")
        print(f"FINAL CONFIRMATORY OUTCOME: {outcome}")
        print(f"Description: {outcome_desc}")
        print(f"==========================================================================================")

    return {
        'n': n,
        'match_stat': match_stat,
        'test_res_dict': test_res_dict,
        'labels': labels,
        'outcome': outcome
    }

def main():
    args = parse_args()
    print("=" * 110)
    print("STAGE 4 CONFIRMATORY & EXPLORATORY EVALUATOR (PROTOCOL v5.1)")
    print(f"Target Confirmation Seeds: {args.seeds} (Target n={len(args.seeds)})")
    print(f"Results Directory: {args.results_dir}")
    print("=" * 110)

    # 1. Load Retrain reference runs (R=300)
    retrain_unlearn_refs = []
    retrain_recovery_refs = []
    for s in args.seeds:
        f_ret = locate_result_file('Retrain', s, stage='recovery', results_dir=args.results_dir, fallback=args.fallback_to_unlearn_dirs)
        if f_ret:
            retrain_unlearn_refs.append(parse_metrics_at_round(f_ret, target_round=200))
            retrain_recovery_refs.append(parse_metrics_at_round(f_ret, target_round=300))

    # 2. Confirmatory Family: A3 vs A2 at Round 100 (m=3, Holm)
    conf_res = evaluate_family(
        arm_test='A3', arm_comp='A2', stage_name='unlearn', round_num=100,
        retrain_runs=retrain_unlearn_refs, seeds=args.seeds, args=args, is_confirmatory=True
    )

    # 3. Exploratory Families
    # A. Post-recovery: A3 vs A2 at Round 200 (m=3, Holm)
    evaluate_family(
        arm_test='A3', arm_comp='A2', stage_name='recovery', round_num=200,
        retrain_runs=retrain_recovery_refs, seeds=args.seeds, args=args, is_confirmatory=False
    )

    # B. Rescale effect: A3 vs A4 at Round 100 (m=3, Holm)
    evaluate_family(
        arm_test='A3', arm_comp='A4', stage_name='unlearn', round_num=100,
        retrain_runs=retrain_unlearn_refs, seeds=args.seeds, args=args, is_confirmatory=False
    )

    # C. Unscaled projection mode: A3 vs A1 at Round 100 (m=3, Holm)
    evaluate_family(
        arm_test='A3', arm_comp='A1', stage_name='unlearn', round_num=100,
        retrain_runs=retrain_unlearn_refs, seeds=args.seeds, args=args, is_confirmatory=False
    )

    # D. Sensitivity control: A1 vs A2 at Round 100 (m=3, Holm)
    evaluate_family(
        arm_test='A1', arm_comp='A2', stage_name='unlearn', round_num=100,
        retrain_runs=retrain_unlearn_refs, seeds=args.seeds, args=args, is_confirmatory=False
    )

if __name__ == '__main__':
    main()
