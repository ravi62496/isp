#!/usr/bin/env python3
"""
evaluate_confirmation.py
Pre-registered, frozen evaluation script for Stage 4 confirmatory analysis.
Compliant with SUCCESS_CRITERIA_v5.1 protocol.

Evaluates seeds 21-30 (n=10) on Primary Regime R2.
Confirmatory family F1: A3 (One-sided, unscaled cap 1.0) vs A2 (Hard-scaled s*=0.54) at Round 100 (m=3, Holm).
Secondary pre-registered family F2: A3 vs A2 at Round 200 post-recovery (m=3, Holm).
Exploratory families: A3 vs A4, A3 vs A1, and Sensitivity Control A1 vs A2.
Exploratory metrics: worst-2 mean, forget-set acc, distance to Retrain on ASR and forget-set.
Also supports exploratory evaluation of Regime R1 (seeds 106-108).
"""

import os
import sys
import csv
import glob
import json
import re
import hashlib
import argparse
import numpy as np
import scipy.stats as stats
from collections import OrderedDict

# Protocol Constants
GIT_STATUS_CMD = ["git", "status", "--porcelain", "--untracked-files=no", "--", ".",
                  ":(exclude)UnlearningTask", ":(exclude)figures", ":(exclude)results_confirm", ":(exclude)results_dryrun"]
CONFIRMATORY_SEEDS = list(range(21, 31))  # [21, 22, 23, 24, 25, 26, 27, 28, 29, 30]
RESULTS_CONFIRM_DIR = "results_confirm"
RESULTS_DRYRUN_DIR = "results_dryrun"

# Arms under evaluation
ARMS = {
    'A1': 'Hard FedOSD (rescaled to ||g_u||, scale 1.0)',
    'A2': 'Hard-scaled (matched step scale)',
    'A3': 'One-sided, unscaled (cap 1.0) [Method Under Test]',
    'A4': 'One-sided rescaled to ||g_u||',
    'Retrain': 'Retrain from Scratch (R=300)'
}

# Primary Metrics Triad (SUCCESS_CRITERIA_v5.1)
PRIMARY_METRIC_SPECS = {
    'mean': {
        'name': 'Retained Cohort Mean Acc',
        'practical': 1.0,      # practical threshold >= 1.0 pp
        'equiv_margin': 0.5,   # equivalence margin [-0.5, +0.5] pp
        'higher_better': True  # method minus comparator
    },
    'floor': {
        'name': 'Worst-Off Floor (Min Acc)',
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

# Exploratory Metrics Specification
EXPLORATORY_METRIC_SPECS = {
    'worst2': {
        'name': 'Worst-2 Clients Mean Acc',
        'higher_better': True
    },
    'forget': {
        'name': 'Forget-Set Accuracy',
        'higher_better': False
    },
    'dist_retrain_asr': {
        'name': 'Distance to Retrain ASR (|T-R| - |C-R|)',
        'higher_better': False
    },
    'dist_retrain_forget': {
        'name': 'Distance to Retrain Forget (|T-R| - |C-R|)',
        'higher_better': False
    }
}

def check_evaluator_hash(criteria_path="SUCCESS_CRITERIA_v5.1.md", evaluator_path=None):
    """Verify that the evaluator file's SHA-256 matches the hash in SUCCESS_CRITERIA_v5.1.md."""
    if evaluator_path is None:
        evaluator_path = os.path.abspath(__file__)
    with open(evaluator_path, "rb") as f:
        actual_hash = hashlib.sha256(f.read()).hexdigest().lower()
        
    if not os.path.exists(criteria_path):
        raise FileNotFoundError(f"[PRE-FLIGHT ERROR] Criteria file not found: {criteria_path}")
    with open(criteria_path, "r", encoding="utf-8") as f:
        content = f.read()
    m = re.search(r"evaluator SHA-256:\s*([0-9a-fA-F]{64})", content)
    if not m:
        raise ValueError(f"[PRE-FLIGHT ERROR] Could not find 'evaluator SHA-256: <hash>' in {criteria_path}")
    recorded_hash = m.group(1).lower()
    if actual_hash != recorded_hash:
        raise ValueError(f"[PRE-FLIGHT HASH MISMATCH]\nActual ({os.path.basename(evaluator_path)}): {actual_hash}\nRecorded in {criteria_path}:    {recorded_hash}")
    return actual_hash

def run_preflight_checks(criteria_path="SUCCESS_CRITERIA_v5.1.md", evaluator_path=None, results_dir=RESULTS_CONFIRM_DIR, manifest_map=None, confirmatory_seeds=CONFIRMATORY_SEEDS, require_empty_results=False):
    """
    Protocol v5.1 Pre-Flight Verification:
    1. Working tree is clean (git status --porcelain is empty)
    2. Evaluator SHA-256 equals the hash recorded in SUCCESS_CRITERIA_v5.1.md
    3. All cells in manifest share ONE commit and one config hash per arm (for confirmatory seeds)
    4. results_confirm/ is empty at start (if require_empty_results is True)
    """
    import subprocess
    print("=" * 90)
    print("STAGE 4 PROTOCOL v5.1 PRE-FLIGHT INTEGRITY CHECKS")
    print("=" * 90)
    
    # 1. Working tree clean check
    print("[CHECK 1/4] Git working tree status...")
    git_res = subprocess.run(GIT_STATUS_CMD, capture_output=True, text=True)
    if git_res.returncode != 0:
        raise RuntimeError(f"git status failed: {git_res.stderr}")
    if git_res.stdout.strip():
        print("[FAIL] Git working tree is dirty:")
        print(git_res.stdout.strip())
        raise RuntimeError("Stage 4 mandates a completely clean git working tree before confirmatory runs.")
    print("  [PASS] Working tree is clean.")

    # 2. Evaluator hash check
    print("[CHECK 2/4] Evaluator SHA-256 vs criteria file...")
    eval_hash = check_evaluator_hash(criteria_path=criteria_path, evaluator_path=evaluator_path)
    print(f"  [PASS] Evaluator SHA-256 verified: {eval_hash}")

    # 3. Provenance check (ONE commit and one config hash per arm)
    print("[CHECK 3/4] Manifest provenance (single commit and config hash per arm)...")
    if manifest_map:
        conf_commits = set()
        arm_configs = {arm: set() for arm in ['A1', 'A2', 'A3', 'A4', 'Retrain']}
        for (arm, seed, stage), cell in manifest_map.items():
            if seed in confirmatory_seeds:
                conf_commits.add(cell['commit'])
                if cell.get('config_hash'):
                    arm_configs[arm].add(cell['config_hash'])
        if len(conf_commits) > 1:
            raise ValueError(f"Multiple git commits found across confirmatory runs: {conf_commits}")
        for arm, hashes in arm_configs.items():
            if len(hashes) > 1:
                raise ValueError(f"Multiple config hashes found for arm {arm}: {hashes}")
        print("  [PASS] Confirmatory runs share exactly one commit and one config hash per arm.")
    else:
        print("  [INFO] No confirmatory manifest rows evaluated yet (pre-run).")

    # 4. results_confirm/ empty at start
    print(f"[CHECK 4/4] Output directory '{results_dir}' clean check...")
    if require_empty_results and os.path.exists(results_dir):
        files = [f for f in os.listdir(results_dir) if not f.startswith('.')]
        if len(files) > 0:
            raise RuntimeError(f"results_confirm directory is not empty at start! Contains: {files}")
        print("  [PASS] results_confirm/ is empty at start.")
    else:
        print(f"  [PASS] results_confirm/ verified.")
    print("=" * 90)
    print("ALL PRE-FLIGHT CHECKS PASSED.")
    print("=" * 90)
    return True

def parse_args():
    parser = argparse.ArgumentParser(description="Stage 4 Confirmatory Evaluator (Protocol v5.1).")
    parser.add_argument('--seeds', type=int, nargs='+', default=CONFIRMATORY_SEEDS,
                        help="List of seeds to evaluate (default: 21-30).")
    parser.add_argument('--regime', type=str, default='R2', choices=['R2', 'R1'],
                        help="Regime under evaluation: R2 (primary) or R1 (exploratory).")
    parser.add_argument('--r1_scale', type=float, default=0.27,
                        help="Step scale for Hard-scaled in R1 (default: 0.27 matched, or 0.24).")
    parser.add_argument('--results_dir', type=str, default=None,
                        help="Path to results directory (defaults to results_confirm for 21-30, else results_dryrun).")
    parser.add_argument('--manifest', type=str, default="RUN_MANIFEST.csv",
                        help="Path to RUN_MANIFEST.csv (default: RUN_MANIFEST.csv).")
    parser.add_argument('--bootstrap_samples', type=int, default=10000,
                        help="Number of bootstrap resamples (protocol specifies 10,000).")
    parser.add_argument('--seed_rng', type=int, default=42,
                        help="Random seed for bootstrap resampling.")
    parser.add_argument('--preflight', action='store_true',
                        help="Run pre-flight integrity checks (clean tree, criteria hash match, empty results_confirm).")
    return parser.parse_args()

def load_manifest(manifest_path="RUN_MANIFEST.csv"):
    """
    Load RUN_MANIFEST.csv and enforce integrity:
    - Exactly one file per cell (abort on ambiguity)
    - Valid non-empty git commit hash (abort on missing hash)
    - File exists on disk
    """
    if not manifest_path or not os.path.exists(manifest_path):
        return None
    manifest_map = {}
    with open(manifest_path, 'r', newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            arm = row['arm'].strip()
            seed = int(row['seed'].strip())
            stage = row['stage'].strip()
            fpath = row['absolute_file_path'].strip() if 'absolute_file_path' in row else row['path'].strip()
            cand_count = int(row.get('candidate_files_count', 1))
            commit = row.get('git_commit', row.get('commit', '')).strip()
            
            # Abort on ambiguity
            if cand_count > 1:
                raise ValueError(f"[MANIFEST ERROR] Ambiguity detected for arm {arm}, seed {seed}, stage {stage}: {cand_count} candidate files.")
            # Abort on missing commit hash
            if not commit or commit.upper() == 'NONE':
                raise ValueError(f"[MANIFEST ERROR] Missing git commit hash for arm {arm}, seed {seed}, stage {stage}: {fpath}")
            if not os.path.exists(fpath):
                raise FileNotFoundError(f"[MANIFEST ERROR] Manifest file does not exist: {fpath}")
                
            config_hash = row.get('config_hash', '').strip()
            manifest_map[(arm, seed, stage)] = {
                'path': fpath,
                'commit': commit,
                'config_hash': config_hash,
                'sha1': row.get('sha1', ''),
                'array_length': int(row.get('array_length', -1)) if row.get('array_length') else None,
                'index': int(row.get('index', row.get('array_index_used', -1))) if row.get('index') or row.get('array_index_used') else None
            }
    return manifest_map

def locate_result_file(arm_key, seed, stage='unlearn', regime='R2', r1_scale=0.27, results_dir=None, manifest_map=None):
    """
    Locate exactly one JSON result file for a given arm, seed, stage, and regime.
    1. If manifest_map contains the cell, return it directly.
    2. Else if results_dir is provided, look for matching files in results_dir.
       Aborts on ambiguity (>1 matches) or missing git commit hash.
    No fallback to unlearn directories or guessing tagged/untagged legacy files.
    """
    # 1. Manifest lookup
    if manifest_map is not None:
        if (arm_key, seed, stage) in manifest_map:
            fpath = manifest_map[(arm_key, seed, stage)]['path']
            if not os.path.exists(fpath):
                raise FileNotFoundError(f"[EVALUATOR ERROR] Manifest file not found on disk: {fpath}")
            return fpath
            
        if arm_key == 'Retrain':
            stg_order = ['R200', 'recovery', 'R300'] if stage == 'unlearn' else ['R300', 'recovery', 'R200']
            for stg in stg_order:
                if (arm_key, seed, stg) in manifest_map:
                    fpath = manifest_map[(arm_key, seed, stg)]['path']
                    if os.path.exists(fpath):
                        return fpath
        
    # 2. Results directory lookup
    suffix = '_unlearn.json' if stage == 'unlearn' else '.json'
    if arm_key == 'A1':
        fname = f"seed{seed} N10 C1.0 FedOSD CNN_CIFAR10 E1 lr0.0004 decay0.999 g_clip-1.0 proj_hard lam100000.0 eta0.0 rlr0.0001" + suffix
    elif arm_key == 'A2':
        scale_str = "scale0.54" if regime == 'R2' else f"scale{r1_scale}"
        fname = f"seed{seed} N10 C1.0 FedOSD CNN_CIFAR10 E1 lr0.0004 decay0.999 g_clip-1.0 proj_hard lam100000.0 eta0.0 {scale_str} rlr0.0001" + suffix
    elif arm_key == 'A3':
        fname = f"seed{seed} N10 C1.0 FedOSD CNN_CIFAR10 E1 lr0.0004 decay0.999 g_clip-1.0 proj_one_sided_cap1.0 wm_uniform lam100000.0 rlr0.0001" + suffix
    elif arm_key == 'A4':
        fname = f"seed{seed} N10 C1.0 FedOSD CNN_CIFAR10 E1 lr0.0004 decay0.999 g_clip-1.0 proj_one_sided wm_uniform lam100000.0 eta0.0 rlr0.0001" + suffix
    elif arm_key == 'Retrain':
        fname = f"seed{seed} retrain_from_scratch seed{seed} N10 C1.0 FedAvg CNN_CIFAR10 E1 lr0.05 decay0.999 g_clip-1.0 R300.json"
    else:
        raise ValueError(f"Unknown arm: {arm_key}")

    if results_dir and os.path.exists(results_dir):
        p = os.path.join(results_dir, fname)
        if os.path.exists(p):
            with open(p, 'r') as jf:
                jd = json.load(jf)
            commit = jd.get('git_commit')
            if not commit or commit.upper() == 'NONE':
                raise ValueError(f"[EVALUATOR ERROR] Missing git commit hash in result file: {p}")
            return p
            
        # Search for unique file matching seed and stage in results_dir
        cand = [f for f in glob.glob(os.path.join(results_dir, f"*seed{seed}*{suffix}")) if not f.endswith('.log')]
        if arm_key == 'A1':
            cand = [f for f in cand if 'proj_hard' in f and 'scale' not in f]
        elif arm_key == 'A2':
            cand = [f for f in cand if 'scale' in f]
        elif arm_key == 'A3':
            cand = [f for f in cand if 'cap1.0' in f]
        elif arm_key == 'A4':
            cand = [f for f in cand if 'proj_one_sided' in f and 'cap' not in f]
        elif arm_key == 'Retrain':
            cand = [f for f in cand if 'retrain' in f]

        if len(cand) > 1:
            raise ValueError(f"[EVALUATOR ERROR] Ambiguity detected in {results_dir}: multiple files match {arm_key} seed {seed} stage {stage}: {cand}")
        elif len(cand) == 1:
            with open(cand[0], 'r') as jf:
                jd = json.load(jf)
            commit = jd.get('git_commit')
            if not commit or commit.upper() == 'NONE':
                raise ValueError(f"[EVALUATOR ERROR] Missing git commit hash in result file: {cand[0]}")
            return cand[0]
            
    return None

def parse_metrics_at_round(fpath, target_round=None, stage=None):
    """
    Robust parser for unlearning, recovery, and retrain files.
    Asserts expected array length and reads the target element (last element for completed runs).
    """
    if not os.path.exists(fpath):
        return None
    with open(fpath, 'r') as f:
        d = json.load(f)
        
    ch = d.get('metric_log', {}).get('client_metric_history', [])
    unlearn_cids = d.get('params', {}).get('unlearn_client_id_list', [])
    pre_acc = d.get('metric_log', {}).get('pre_acc', {})
    test_interval = d.get('params', {}).get('test_interval', 1)
    
    is_retrain = ('retrain' in os.path.basename(fpath).lower())
    
    retained_accs, retained_drops = [], []
    asrs, cleans, mias = [], [], []
    
    for cid, hist in enumerate(ch):
        t_acc = hist.get('test_accuracy', [])
        b_acc = hist.get('backdoor_test_accuracy', [])
        c_acc = hist.get('clean_test_accuracy', [])
        f_acc = hist.get('forget_acc', [])
        m_auc = hist.get('mia_auc', [])
        
        arr_len = len(t_acc)
        
        # Array length assertions per protocol
        if is_retrain:
            assert arr_len == 61, f"[EVALUATOR ERROR] Expected Retrain array length 61, got {arr_len} in {fpath}"
            if target_round == 200:
                idx = 40  # round 200
            elif target_round == 300 or target_round is None:
                idx = arr_len - 1  # round 300 (last element)
            else:
                idx = min(arr_len - 1, target_round // test_interval)
        elif stage == 'unlearn' or (target_round == 100 and stage != 'recovery' and fpath.endswith('_unlearn.json')):
            assert arr_len == 100, f"[EVALUATOR ERROR] Expected unlearn array length 100, got {arr_len} in {fpath}"
            idx = arr_len - 1  # round 100 (last element, index 99)
        elif stage == 'recovery' or target_round == 200:
            assert arr_len in [101, 201], f"[EVALUATOR ERROR] Expected recovery array length 101 or 201, got {arr_len} in {fpath}"
            idx = arr_len - 1  # round 200 (last element)
        else:
            if target_round is not None:
                idx = min(arr_len - 1, target_round // test_interval)
            else:
                idx = arr_len - 1

        final_acc = t_acc[idx] if t_acc else 0.0
        cur_b_acc = b_acc[idx] if b_acc else final_acc
        cur_c_acc = c_acc[idx] if c_acc else final_acc
        cur_f_acc = f_acc[idx] if f_acc else cur_c_acc
        cur_m_auc = m_auc[idx] if m_auc else 0.5

        init_acc = pre_acc.get(str(cid), t_acc[0] if t_acc else 0.0)
        if cid in unlearn_cids:
            asrs.append(cur_b_acc)
            cleans.append(cur_f_acc)
            mias.append(cur_m_auc)
        else:
            retained_accs.append(final_acc)
            retained_drops.append(init_acc - final_acc)
            
    sorted_accs = sorted(retained_accs)
    
    dists = d.get('algorithm_log', {}).get('distance_to_init_model', [])
    unl_disp = d.get('unlearn_displacement') or d.get('algorithm_log', {}).get('unlearn_displacement')
    if unl_disp is None and dists and len(dists) >= 100:
        unl_disp = dists[99]

    if (stage == 'recovery' or target_round == 200) and dists:
        disp = dists[-1]
    elif target_round is not None and target_round <= 100 and dists and len(dists) >= 100:
        disp = dists[target_round - 1]
    else:
        disp = unl_disp if unl_disp is not None else (dists[-1] if dists else 0.0)
            
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
        'disp_unlearn': float(unl_disp if unl_disp is not None else disp),
        'solver_time_ms': median_solver_ms,
        'round_time_s': median_round_s,
        'array_len': len(ch[0].get('test_accuracy', [])) if ch else 0
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

def compute_hypothesis_tests(diffs, practical_thresh=None, equiv_margin=None, n_resamples=10000, seed=42):
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

def assign_label(test_res, holm_p, strength_matched=True, n=10, allow_equivalence=False):
    """
    Protocol Label Rules (SUCCESS_CRITERIA_v5.1):
    1. If strength is not matched: 'not comparable'
    2. When n < 10: 'descriptive (n<10)'
    3. Confirmatory labels (n >= 10, strength matched):
       - Superior: 95% CI > 0, Holm p < 0.05, estimate >= practical
       - Better but negligible: 95% CI > 0, Holm p < 0.05, estimate < practical
       - Inferior: 95% CI < 0, Holm p < 0.05, estimate <= -practical
       - Worse but negligible: 95% CI < 0, Holm p < 0.05, estimate > -practical
       - Equivalent: 90% CI inside margin [-margin, +margin] (ONLY permitted when allow_equivalence=True, e.g. Round 200)
       - Else: Inconclusive
    Strictly NO 'Equivalent' label at Round 100 (allow_equivalence=False).
    """
    if not strength_matched:
        return 'not comparable'

    if n < 10:
        return 'descriptive (n<10)'

    ci95_low, ci95_high = test_res['ci95_t']
    ci90_low, ci90_high = test_res['ci90_t']
    est = test_res['mean_diff']
    pract = test_res['practical_thresh']
    margin = test_res['equiv_margin']
    
    is_pos_95 = (ci95_low > 0.0)
    is_neg_95 = (ci95_high < 0.0)
    is_sig = (holm_p < 0.05)
    
    # 1. Clear directional findings above the practical threshold
    if is_pos_95 and is_sig and est >= pract:
        return 'Superior'
    if is_neg_95 and is_sig and est <= -pract:
        return 'Inferior'

    # 2. Equivalence takes precedence over 'negligible' labels (permitted only when allow_equivalence=True)
    if allow_equivalence and (margin is not None):
        if ci90_low >= -margin and ci90_high <= margin:
            return 'Equivalent'

    # 3. Statistically detectable but practically negligible differences
    if is_pos_95 and is_sig:
        return 'Better but negligible'
    if is_neg_95 and is_sig:
        return 'Worse but negligible'

    # 4. Inconclusive
    return 'Inconclusive'

def check_strength_matching(test_runs, comp_runs, round_num=100):
    """
    SUCCESS_CRITERIA_v5.1 Protocol:
    Strength matched only if, on seed averages at round 100:
      displacement ratio in [0.95, 1.05] AND |ASR diff| <= 1.0 pt.
    For Round 200 (F2), evaluates displacement measured at round 200,
    and also reports the stored unlearn displacement (round 100).
    """
    disp_test = np.mean([r['disp'] for r in test_runs])
    disp_comp = np.mean([r['disp'] for r in comp_runs])
    disp_ratio = disp_test / (disp_comp + 1e-12)
    
    disp_unl_test = np.mean([r.get('disp_unlearn', r['disp']) for r in test_runs])
    disp_unl_comp = np.mean([r.get('disp_unlearn', r['disp']) for r in comp_runs])
    disp_unl_ratio = disp_unl_test / (disp_unl_comp + 1e-12)

    asr_test = np.mean([r['asr'] for r in test_runs])
    asr_comp = np.mean([r['asr'] for r in comp_runs])
    asr_diff = abs(asr_test - asr_comp)
    
    cond_disp = (0.95 <= disp_ratio <= 1.05)
    cond_asr = (asr_diff <= 1.0)
    
    is_matched = bool(cond_disp and cond_asr)
    
    return {
        'is_matched': is_matched,
        'disp_ratio': disp_ratio,
        'disp_unl_ratio': disp_unl_ratio,
        'disp_pass': cond_disp,
        'asr_diff': asr_diff,
        'asr_pass': cond_asr,
        'round_num': round_num
    }

def evaluate_family(arm_test, arm_comp, stage_name, round_num, retrain_runs, seeds, args,
                    is_confirmatory=False, allow_equivalence=False, family_name="Family"):
    """
    Evaluate a family of 3 primary hypotheses (mean, floor, drop) with Holm adjustment,
    followed by exploratory metrics (worst2, forget, dist to retrain).
    Writes out script-generated CSVs for every reported table.
    """
    is_sensitivity_control = (arm_test == 'A1' and arm_comp == 'A2')
    
    family_label = "[CONFIRMATORY FAMILY F1]" if (is_confirmatory and round_num == 100) else \
                   "[SECONDARY PRE-REGISTERED FAMILY F2]" if (is_confirmatory and round_num == 200) else \
                   "[EXPLORATORY FAMILY]"
    if not is_confirmatory:
        mode_prefix = "NON-CONFIRMATORY DRY RUN" if sorted(seeds) != sorted(CONFIRMATORY_SEEDS) else "EXPLORATORY"
        family_label = f"[{mode_prefix} FAMILY]"

    print("\n" + "=" * 125)
    print(f"{family_label} {ARMS[arm_test]} vs {ARMS[arm_comp]} at Round {round_num} ({stage_name})")
    if is_sensitivity_control:
        print("Note: Sensitivity Control A1 vs A2 has no strength rule (deliberate step-size contrast).")
    print("=" * 125)
    
    test_runs = []
    comp_runs = []
    active_seeds = []
    
    for s in seeds:
        f_test = locate_result_file(arm_test, s, stage=stage_name, regime=args.regime, r1_scale=args.r1_scale, results_dir=args.results_dir, manifest_map=getattr(args, 'manifest_map', None))
        f_comp = locate_result_file(arm_comp, s, stage=stage_name, regime=args.regime, r1_scale=args.r1_scale, results_dir=args.results_dir, manifest_map=getattr(args, 'manifest_map', None))
        
        if f_test and f_comp:
            test_runs.append(parse_metrics_at_round(f_test, target_round=round_num, stage=stage_name))
            comp_runs.append(parse_metrics_at_round(f_comp, target_round=round_num, stage=stage_name))
            active_seeds.append(s)
        else:
            print(f"Seed {s}: Incomplete runs (Test: {'Found' if f_test else 'Missing'}, Comp: {'Found' if f_comp else 'Missing'})")
            
    n = len(active_seeds)
    if n == 0:
        print("ERROR: No completed seed runs found for this comparison.")
        return None
        
    print(f"Completed Seeds (n={n}): {active_seeds}")
    
    # Evaluate strength matching
    match_stat = check_strength_matching(test_runs, comp_runs, round_num=round_num)
    if not is_sensitivity_control:
        print(f"\n--- Strength Matching Evaluation (Round {round_num} Seed Averages) ---")
        if round_num == 200:
            print(f"Measured Round 200 Displacement Ratio (Test / Comp): {match_stat['disp_ratio']:.4f} "
                  f"[{'PASS [0.95, 1.05]' if match_stat['disp_pass'] else 'FAIL (Outside [0.95, 1.05])'}]")
            print(f"Stored Unlearn Round 100 Displacement Ratio:       {match_stat['disp_unl_ratio']:.4f} "
                  f"[{'PASS [0.95, 1.05]' if (0.95 <= match_stat['disp_unl_ratio'] <= 1.05) else 'FAIL'}]")
        else:
            print(f"Displacement Ratio (Test / Comp): {match_stat['disp_ratio']:.4f} "
                  f"[{'PASS [0.95, 1.05]' if match_stat['disp_pass'] else 'FAIL (Outside [0.95, 1.05])'}]")
        print(f"|ASR Difference|:                 {match_stat['asr_diff']:.2f} pt "
              f"[{'PASS <= 1.0 pt' if match_stat['asr_pass'] else 'FAIL'}]")
        if match_stat['is_matched']:
            print(">>> STATUS: STRENGTH MATCHED <<<")
        else:
            print(">>> STATUS: STRENGTH NOT MATCHED <<< (All comparison labels assigned 'not comparable')")
    else:
        print(f"\n--- Strength Matching Evaluation: Exempt (Sensitivity Control, Round {round_num}) ---")
        if round_num == 200:
            print(f"Measured Round 200 Displacement Ratio (A1 / A2):     {match_stat['disp_ratio']:.4f}")
            print(f"Stored Unlearn Round 100 Displacement Ratio:       {match_stat['disp_unl_ratio']:.4f}")
        else:
            print(f"Displacement Ratio (A1 / A2):     {match_stat['disp_ratio']:.4f}")
        print(f"|ASR Difference|:                 {match_stat['asr_diff']:.2f} pt")

    diff_dict = {}
    test_res_dict = {}
    p_values = []
    
    # Primary Metrics Triad: mean | floor | drop
    primary_keys = ['mean', 'floor', 'drop']
    for m in primary_keys:
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
        
    # Apply Holm step-down procedure directly to paired t-test p-values
    holm_p_values = apply_holm_adjustment(p_values)
    
    effective_strength_matched = True if is_sensitivity_control else match_stat['is_matched']

    print(f"\n--- Primary Hypothesis Testing (Triad m=3, Holm Step-Down on t-test p) ---")
    print(f"{'Primary Metric':<24} | {'Diff Est':<9} | {'95% CI (t-test)':<22} | {'95% CI (boot)':<22} | {'p (t)':<8} | {'p (Holm)':<8} | {'p (Wilcox)':<10} | {'Decision Label'}")
    print("-" * 135)
    
    labels = {}
    csv_rows = []
    for idx, m in enumerate(primary_keys):
        tres = test_res_dict[m]
        spec = PRIMARY_METRIC_SPECS[m]
        hp = holm_p_values[idx]
        lbl = assign_label(tres, hp, strength_matched=effective_strength_matched, n=n, allow_equivalence=allow_equivalence)
        labels[m] = lbl
        ci_t_str = f"[{tres['ci95_t'][0]:+6.2f}, {tres['ci95_t'][1]:+6.2f}]"
        ci_b_str = f"[{tres['ci95_boot'][0]:+6.2f}, {tres['ci95_boot'][1]:+6.2f}]"
        print(f"{spec['name']:<24} | {tres['mean_diff']:+7.2f}% | {ci_t_str:<22} | {ci_b_str:<22} | {tres['p_t']:7.4f} | {hp:7.4f} | {tres['p_wilcox']:9.4f} | {lbl}")
        
        test_val_mean = float(np.mean([t[m] for t in test_runs]))
        comp_val_mean = float(np.mean([c[m] for c in comp_runs]))
        csv_rows.append({
            'family': family_name,
            'arm_test': arm_test,
            'arm_comp': arm_comp,
            'stage': stage_name,
            'round': round_num,
            'metric': m,
            'metric_name': spec['name'],
            'test_mean': test_val_mean,
            'comp_mean': comp_val_mean,
            'diff_est': tres['mean_diff'],
            'ci95_t_low': tres['ci95_t'][0],
            'ci95_t_high': tres['ci95_t'][1],
            'ci95_boot_low': tres['ci95_boot'][0],
            'ci95_boot_high': tres['ci95_boot'][1],
            'p_t': tres['p_t'],
            'p_holm': hp,
            'p_wilcox': tres['p_wilcox'],
            'label': lbl
        })
    print("-" * 135)

    # Save family CSV table
    family_csv_path = os.path.join(args.results_dir, f"table_{family_name}.csv")
    with open(family_csv_path, 'w', newline='', encoding='utf-8') as cf:
        fieldnames = ['family', 'arm_test', 'arm_comp', 'stage', 'round', 'metric', 'metric_name',
                      'test_mean', 'comp_mean', 'diff_est', 'ci95_t_low', 'ci95_t_high',
                      'ci95_boot_low', 'ci95_boot_high', 'p_t', 'p_holm', 'p_wilcox', 'label']
        writer = csv.DictWriter(cf, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_rows)

    # Exploratory & Secondary Paired Outcomes
    print("\n--- Exploratory & Secondary Paired Outcomes (Reported with 95% CIs; No Matching Role) ---")
    exploratory_csv_rows = []
    
    # 1. Worst-2 clients mean
    w2_diffs = [t['worst2'] - c['worst2'] for t, c in zip(test_runs, comp_runs)]
    w2_tres = compute_hypothesis_tests(w2_diffs, n_resamples=args.bootstrap_samples, seed=args.seed_rng)
    w2_ci_t = f"[{w2_tres['ci95_t'][0]:+6.2f}, {w2_tres['ci95_t'][1]:+6.2f}]"
    w2_ci_b = f"[{w2_tres['ci95_boot'][0]:+6.2f}, {w2_tres['ci95_boot'][1]:+6.2f}]"
    print(f"{'[EXPLORATORY] Worst-2 Clients Mean':<35} | Diff: {w2_tres['mean_diff']:+6.2f} pt | 95% CI (t): {w2_ci_t:<18} | 95% CI (boot): {w2_ci_b:<18} | p(t): {w2_tres['p_t']:.4f}")
    exploratory_csv_rows.append({
        'family': family_name, 'metric': 'worst2', 'diff_est': w2_tres['mean_diff'],
        'ci95_t_low': w2_tres['ci95_t'][0], 'ci95_t_high': w2_tres['ci95_t'][1],
        'ci95_boot_low': w2_tres['ci95_boot'][0], 'ci95_boot_high': w2_tres['ci95_boot'][1],
        'p_t': w2_tres['p_t'], 'p_wilcox': w2_tres['p_wilcox']
    })

    # 2. Forget-set accuracy
    fg_diffs = [t['forget'] - c['forget'] for t, c in zip(test_runs, comp_runs)]
    fg_tres = compute_hypothesis_tests(fg_diffs, n_resamples=args.bootstrap_samples, seed=args.seed_rng)
    fg_ci_t = f"[{fg_tres['ci95_t'][0]:+6.2f}, {fg_tres['ci95_t'][1]:+6.2f}]"
    fg_ci_b = f"[{fg_tres['ci95_boot'][0]:+6.2f}, {fg_tres['ci95_boot'][1]:+6.2f}]"
    print(f"{'[EXPLORATORY] Forget-Set Acc (T - C)':<35} | Diff: {fg_tres['mean_diff']:+6.2f} pt | 95% CI (t): {fg_ci_t:<18} | 95% CI (boot): {fg_ci_b:<18} | p(t): {fg_tres['p_t']:.4f}")
    exploratory_csv_rows.append({
        'family': family_name, 'metric': 'forget', 'diff_est': fg_tres['mean_diff'],
        'ci95_t_low': fg_tres['ci95_t'][0], 'ci95_t_high': fg_tres['ci95_t'][1],
        'ci95_boot_low': fg_tres['ci95_boot'][0], 'ci95_boot_high': fg_tres['ci95_boot'][1],
        'p_t': fg_tres['p_t'], 'p_wilcox': fg_tres['p_wilcox']
    })

    # 3. Distance to Retrain ASR & Forget-Set
    if retrain_runs and len(retrain_runs) >= n:
        ret_asrs = [r['asr'] for r in retrain_runs[:n]]
        ret_fgs = [r['forget'] for r in retrain_runs[:n]]
        
        dist_asr_diffs = [abs(t['asr'] - r) - abs(c['asr'] - r) for t, c, r in zip(test_runs, comp_runs, ret_asrs)]
        d_asr_tres = compute_hypothesis_tests(dist_asr_diffs, n_resamples=args.bootstrap_samples, seed=args.seed_rng)
        d_asr_ci_t = f"[{d_asr_tres['ci95_t'][0]:+6.2f}, {d_asr_tres['ci95_t'][1]:+6.2f}]"
        d_asr_ci_b = f"[{d_asr_tres['ci95_boot'][0]:+6.2f}, {d_asr_tres['ci95_boot'][1]:+6.2f}]"
        print(f"{'[EXPLORATORY] Dist to Retrain ASR':<35} | Diff: {d_asr_tres['mean_diff']:+6.2f} pt | 95% CI (t): {d_asr_ci_t:<18} | 95% CI (boot): {d_asr_ci_b:<18} | p(t): {d_asr_tres['p_t']:.4f}")
        exploratory_csv_rows.append({
            'family': family_name, 'metric': 'dist_retrain_asr', 'diff_est': d_asr_tres['mean_diff'],
            'ci95_t_low': d_asr_tres['ci95_t'][0], 'ci95_t_high': d_asr_tres['ci95_t'][1],
            'ci95_boot_low': d_asr_tres['ci95_boot'][0], 'ci95_boot_high': d_asr_tres['ci95_boot'][1],
            'p_t': d_asr_tres['p_t'], 'p_wilcox': d_asr_tres['p_wilcox']
        })

        dist_fg_diffs = [abs(t['forget'] - r) - abs(c['forget'] - r) for t, c, r in zip(test_runs, comp_runs, ret_fgs)]
        d_fg_tres = compute_hypothesis_tests(dist_fg_diffs, n_resamples=args.bootstrap_samples, seed=args.seed_rng)
        d_fg_ci_t = f"[{d_fg_tres['ci95_t'][0]:+6.2f}, {d_fg_tres['ci95_t'][1]:+6.2f}]"
        d_fg_ci_b = f"[{d_fg_tres['ci95_boot'][0]:+6.2f}, {d_fg_tres['ci95_boot'][1]:+6.2f}]"
        print(f"{'[EXPLORATORY] Dist to Retrain Forget':<35} | Diff: {d_fg_tres['mean_diff']:+6.2f} pt | 95% CI (t): {d_fg_ci_t:<18} | 95% CI (boot): {d_fg_ci_b:<18} | p(t): {d_fg_tres['p_t']:.4f}")
        exploratory_csv_rows.append({
            'family': family_name, 'metric': 'dist_retrain_forget', 'diff_est': d_fg_tres['mean_diff'],
            'ci95_t_low': d_fg_tres['ci95_t'][0], 'ci95_t_high': d_fg_tres['ci95_t'][1],
            'ci95_boot_low': d_fg_tres['ci95_boot'][0], 'ci95_boot_high': d_fg_tres['ci95_boot'][1],
            'p_t': d_fg_tres['p_t'], 'p_wilcox': d_fg_tres['p_wilcox']
        })

        ret_asr_mean = np.mean(ret_asrs)
        ret_fg_mean = np.mean(ret_fgs)
        t_asr_mean = np.mean([r['asr'] for r in test_runs])
        c_asr_mean = np.mean([r['asr'] for r in comp_runs])
        t_fg_mean = np.mean([r['forget'] for r in test_runs])
        c_fg_mean = np.mean([r['forget'] for r in comp_runs])
        print(f"  Retrain Reference Anchors: Retrain ASR: {ret_asr_mean:.2f}% (T dist: {abs(t_asr_mean - ret_asr_mean):.2f} pt, C dist: {abs(c_asr_mean - ret_asr_mean):.2f} pt) | Retrain Forget: {ret_fg_mean:.2f}% (T dist: {abs(t_fg_mean - ret_fg_mean):.2f} pt, C dist: {abs(c_fg_mean - ret_fg_mean):.2f} pt)")
    else:
        print("  Retrain Reference Anchors: N/A for this family/regime.")

    exploratory_csv_path = os.path.join(args.results_dir, f"table_{family_name}_exploratory.csv")
    with open(exploratory_csv_path, 'w', newline='', encoding='utf-8') as cf:
        fieldnames = ['family', 'metric', 'diff_est', 'ci95_t_low', 'ci95_t_high', 'ci95_boot_low', 'ci95_boot_high', 'p_t', 'p_wilcox']
        writer = csv.DictWriter(cf, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(exploratory_csv_rows)

    return {
        'family_name': family_name,
        'n': n,
        'active_seeds': active_seeds,
        'test_runs': test_runs,
        'comp_runs': comp_runs,
        'match_stat': match_stat,
        'test_res_dict': test_res_dict,
        'labels': labels
    }

def determine_outcome(f1_res, f2_res, n_seeds):
    """Protocol v5.1 outcome rules (pure function; unit-tested)."""
    if n_seeds < 10:
        return "Descriptive (n<10)", f"Sample size n={n_seeds} < 10. Stage 4 protocol decision rules require n >= 10."
    if not (f1_res and f2_res):
        return "Incomplete", "F1 or F2 results are missing."
    f1_labels, f2_labels = f1_res['labels'], f2_res['labels']
    f1_matched = bool(f1_res['match_stat']['is_matched'])
    f2_matched = bool(f2_res['match_stat']['is_matched'])
    num_sup_f1 = sum(1 for l in f1_labels.values() if l == 'Superior')
    has_inf_f1 = any(l == 'Inferior' for l in f1_labels.values())
    all_equiv_f2 = all(l == 'Equivalent' for l in f2_labels.values())

    if f1_matched and num_sup_f1 >= 2 and not has_inf_f1:
        return "Outcome A", "F1 strength matched; Superior on at least two primary metrics, none Inferior."
    if f1_matched and f2_matched and num_sup_f1 == 0 and not has_inf_f1 and all_equiv_f2:
        return "Outcome B", "F1 strength matched with no Superior or Inferior; Equivalent on all three metrics in F2."
    reasons = []
    if not f1_matched:
        reasons.append("F1 strength not matched")
    if has_inf_f1:
        reasons.append("Inferior metric present in F1")
    if not f2_matched:
        reasons.append("F2 strength not matched")
    if num_sup_f1 < 2 and not all_equiv_f2:
        reasons.append("fewer than 2 Superior in F1 and F2 not fully Equivalent")
    return "Outcome C", f"Outcome C: {'; '.join(reasons) if reasons else 'criteria for Outcome A or B not met'}."


def main():
    args = parse_args()
    
    # Determine execution mode and target output directory
    is_confirmatory_seeds = (sorted(args.seeds) == sorted(CONFIRMATORY_SEEDS))
    
    if args.results_dir is None:
        args.results_dir = RESULTS_CONFIRM_DIR if is_confirmatory_seeds else RESULTS_DRYRUN_DIR
        
    os.makedirs(args.results_dir, exist_ok=True)
    
    run_mode_banner = "CONFIRMATORY EXECUTION (SEEDS 21-30)" if is_confirmatory_seeds else "NON-CONFIRMATORY DRY RUN (TUNING SEEDS)"
    
    print("=" * 125)
    print(f"STAGE 4 EVALUATOR (PROTOCOL v5.1) -- {run_mode_banner}")
    print(f"Target Seeds: {args.seeds} (n={len(args.seeds)}) | Regime: {args.regime}")
    print(f"Results Directory: {args.results_dir}")
    print("=" * 125)

    manifest_map = load_manifest(args.manifest) if getattr(args, 'manifest', None) else None
    if manifest_map:
        print(f"Loaded manifest from {args.manifest} ({len(manifest_map)} registered cells).")
    args.manifest_map = manifest_map

    if args.preflight:
        run_preflight_checks(criteria_path="SUCCESS_CRITERIA_v5.1.md", results_dir=args.results_dir, manifest_map=manifest_map, require_empty_results=True)
        sys.exit(0)

    # STRICT SAFETY AND PROVENANCE CHECKS FOR CONFIRMATORY SEEDS (21-30)
    any_confirmatory = any(s in CONFIRMATORY_SEEDS for s in args.seeds)
    if any_confirmatory:
        # 1. Assert evaluator SHA-256 matches criteria file
        check_evaluator_hash("SUCCESS_CRITERIA_v5.1.md")

        # 2. Assert clean git working tree per SUCCESS_CRITERIA_v5.1
        import subprocess
        git_status = subprocess.run(GIT_STATUS_CMD, capture_output=True, text=True)
        if git_status.returncode != 0:
            raise RuntimeError(f"[CONFIRMATORY INTEGRITY ERROR] Failed to run git status: {git_status.stderr}")
        if git_status.stdout.strip():
            print(f"\n[FATAL INTEGRITY ABORT] Dirty working tree detected during confirmatory evaluation (seeds 21-30).")
            print("SUCCESS_CRITERIA_v5.1 protocol mandates all confirmatory runs are evaluated on a clean working tree:")
            print(git_status.stdout.strip())
            sys.exit(1)

        # 2. Assert single git commit and single config hash per arm across confirmation runs
        if manifest_map:
            conf_commits = set()
            arm_configs = {arm: set() for arm in ['A1', 'A2', 'A3', 'A4', 'Retrain']}
            for (arm, s, stg), cell in manifest_map.items():
                if s in CONFIRMATORY_SEEDS:
                    conf_commits.add(cell['commit'])
                    if cell.get('config_hash'):
                        arm_configs[arm].add(cell['config_hash'])
            if len(conf_commits) > 1:
                print(f"\n[FATAL PROVENANCE ABORT] Multiple git commits detected across confirmatory runs: {conf_commits}")
                sys.exit(1)
            for arm, hashes in arm_configs.items():
                if len(hashes) > 1:
                    print(f"\n[FATAL PROVENANCE ABORT] Multiple config hashes detected for arm {arm} across confirmatory runs: {hashes}")
                    sys.exit(1)

        # 3. Completeness check
        print("Conducting pre-flight completeness check for confirmatory seeds (21-30)...")
        missing_runs = []
        for s in args.seeds:
            if s in CONFIRMATORY_SEEDS:
                for arm in ['A1', 'A2', 'A3', 'A4']:
                    for stg in ['unlearn', 'recovery']:
                        f = locate_result_file(arm, s, stage=stg, regime=args.regime, r1_scale=args.r1_scale, results_dir=args.results_dir, manifest_map=args.manifest_map)
                        if f is None:
                            missing_runs.append((arm, s, stg))
                f_ret = locate_result_file('Retrain', s, stage='recovery', regime=args.regime, results_dir=args.results_dir, manifest_map=args.manifest_map)
                if f_ret is None:
                    missing_runs.append(('Retrain', s, 'R300'))
                    
        if missing_runs:
            print(f"\n[FATAL SAFETY ABORT] Refusing execution on seeds 21-30: {len(missing_runs)} required arm runs are incomplete:")
            for arm, s, stg in missing_runs[:10]:
                print(f"  Missing: Arm {arm}, Seed {s}, Stage {stg}")
            if len(missing_runs) > 10:
                print(f"  ... and {len(missing_runs) - 10} more.")
            print("\nStage 4 confirmatory analysis is sealed and cannot proceed until all runs finish.")
            sys.exit(1)
        else:
            print("Pre-flight completeness check passed: all 10 seeds complete for all arms.")

    # 1. Load Retrain reference runs (R=300 for R2, or R1 Retrain runs)
    retrain_unlearn_refs = []
    retrain_recovery_refs = []
    for s in args.seeds:
        f_ret = locate_result_file('Retrain', s, stage='recovery', regime=args.regime, results_dir=args.results_dir, manifest_map=args.manifest_map)
        if f_ret:
            retrain_unlearn_refs.append(parse_metrics_at_round(f_ret, target_round=200 if args.regime == 'R2' else 100, stage='unlearn'))
            retrain_recovery_refs.append(parse_metrics_at_round(f_ret, target_round=300 if args.regime == 'R2' else 100, stage='recovery'))

    # If regime is R1 and no retrain files matched for args.seeds directly (e.g. fresh seeds 106-108):
    if args.regime == 'R1' and len(retrain_unlearn_refs) < len(args.seeds):
        canonical_r1_seeds = [11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
        r1_refs = []
        for cs in canonical_r1_seeds:
            f_cs = locate_result_file('Retrain', cs, stage='recovery', regime='R1', results_dir=args.results_dir, manifest_map=args.manifest_map)
            if f_cs:
                r1_refs.append(parse_metrics_at_round(f_cs, target_round=100))
        if r1_refs:
            retrain_unlearn_refs = r1_refs[:len(args.seeds)]
            retrain_recovery_refs = r1_refs[:len(args.seeds)]

    eval_results = OrderedDict()

    if args.regime == 'R2':
        # 2. Confirmatory Family F1: A3 vs A2 at Round 100 (m=3, Holm)
        eval_results['F1_Confirmatory_A3_vs_A2_Rd100'] = evaluate_family(
            arm_test='A3', arm_comp='A2', stage_name='unlearn', round_num=100,
            retrain_runs=retrain_unlearn_refs, seeds=args.seeds, args=args,
            is_confirmatory=is_confirmatory_seeds, allow_equivalence=False,
            family_name="F1_Confirmatory_A3_vs_A2_Rd100"
        )

        # 3. Secondary Pre-Registered Family F2: A3 vs A2 at Round 200 (m=3, Holm)
        eval_results['F2_Secondary_A3_vs_A2_Rd200'] = evaluate_family(
            arm_test='A3', arm_comp='A2', stage_name='recovery', round_num=200,
            retrain_runs=retrain_recovery_refs, seeds=args.seeds, args=args,
            is_confirmatory=is_confirmatory_seeds, allow_equivalence=True,
            family_name="F2_Secondary_A3_vs_A2_Rd200"
        )

        # 4. Exploratory Families
        # A. Rescale effect: A3 vs A4 at Round 100 (m=3, Holm)
        eval_results['Exploratory_A3_vs_A4_Rd100'] = evaluate_family(
            arm_test='A3', arm_comp='A4', stage_name='unlearn', round_num=100,
            retrain_runs=retrain_unlearn_refs, seeds=args.seeds, args=args,
            is_confirmatory=False, allow_equivalence=False,
            family_name="Exploratory_A3_vs_A4_Rd100"
        )

        # B. Unscaled projection mode: A3 vs A1 at Round 100 (m=3, Holm)
        eval_results['Exploratory_A3_vs_A1_Rd100'] = evaluate_family(
            arm_test='A3', arm_comp='A1', stage_name='unlearn', round_num=100,
            retrain_runs=retrain_unlearn_refs, seeds=args.seeds, args=args,
            is_confirmatory=False, allow_equivalence=False,
            family_name="Exploratory_A3_vs_A1_Rd100"
        )

        # C. Sensitivity control: A1 vs A2 at Round 100 (m=3, Holm)
        eval_results['Sensitivity_Control_A1_vs_A2_Rd100'] = evaluate_family(
            arm_test='A1', arm_comp='A2', stage_name='unlearn', round_num=100,
            retrain_runs=retrain_unlearn_refs, seeds=args.seeds, args=args,
            is_confirmatory=False, allow_equivalence=False,
            family_name="Sensitivity_Control_A1_vs_A2_Rd100"
        )

        # OVERALL CONFIRMATORY OUTCOME DETERMINATION (Protocol v5.1)
        f1_res = eval_results.get('F1_Confirmatory_A3_vs_A2_Rd100')
        f2_res = eval_results.get('F2_Secondary_A3_vs_A2_Rd200')
        
        outcome, outcome_desc = determine_outcome(f1_res, f2_res, len(args.seeds))

        print(f"\n" + "=" * 125)
        print(f"CONFIRMATORY PROTOCOL v5.1 OUTCOME: {outcome}")
        print(f"Description: {outcome_desc}")
        print("=" * 125)

    else:  # Regime R1
        print(f"\n--- REGIME R1 EVALUATION (Seeds {args.seeds}) ---")
        eval_results['F3_Secondary_Regime_R1_A3_vs_A2_Rd100'] = evaluate_family(
            arm_test='A3', arm_comp='A2', stage_name='unlearn', round_num=100,
            retrain_runs=retrain_unlearn_refs, seeds=args.seeds, args=args,
            is_confirmatory=is_confirmatory_seeds, allow_equivalence=False,
            family_name="F3_Secondary_Regime_R1_A3_vs_A2_Rd100"
        )
        eval_results['R1_Exploratory_A3_vs_A1_Rd100'] = evaluate_family(
            arm_test='A3', arm_comp='A1', stage_name='unlearn', round_num=100,
            retrain_runs=retrain_unlearn_refs, seeds=args.seeds, args=args,
            is_confirmatory=False, allow_equivalence=False,
            family_name="R1_Exploratory_A3_vs_A1_Rd100"
        )
        eval_results['R1_Sensitivity_Control_A1_vs_A2_Rd100'] = evaluate_family(
            arm_test='A1', arm_comp='A2', stage_name='unlearn', round_num=100,
            retrain_runs=retrain_unlearn_refs, seeds=args.seeds, args=args,
            is_confirmatory=False, allow_equivalence=False,
            family_name="R1_Sensitivity_Control_A1_vs_A2_Rd100"
        )

    # Save per-seed metrics CSV for all loaded runs
    for stage_eval, round_eval in [('unlearn', 100), ('recovery', 200)]:
        per_seed_rows = []
        for arm in ['A1', 'A2', 'A3', 'A4']:
            for s in args.seeds:
                fpath = locate_result_file(arm, s, stage=stage_eval, regime=args.regime, r1_scale=args.r1_scale, results_dir=args.results_dir, manifest_map=args.manifest_map)
                if fpath:
                    m = parse_metrics_at_round(fpath, target_round=round_eval, stage=stage_eval)
                    per_seed_rows.append({
                        'stage': stage_eval, 'round': round_eval, 'arm': arm, 'seed': s,
                        'mean': m['mean'], 'worst2': m['worst2'], 'floor': m['floor'],
                        'drop': m['drop'], 'asr': m['asr'], 'forget': m['forget'],
                        'mia': m['mia'], 'disp': m['disp'], 'solver_time_ms': m['solver_time_ms']
                    })
        if per_seed_rows:
            ps_csv_path = os.path.join(args.results_dir, f"table_per_seed_metrics_rd{round_eval}.csv")
            with open(ps_csv_path, 'w', newline='', encoding='utf-8') as cf:
                fieldnames = ['stage', 'round', 'arm', 'seed', 'mean', 'worst2', 'floor', 'drop', 'asr', 'forget', 'mia', 'disp', 'solver_time_ms']
                writer = csv.DictWriter(cf, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(per_seed_rows)

    # Save overall summary CSV table
    summary_rows = []
    for fam, res in eval_results.items():
        if res:
            mres = res['match_stat']
            lbls = res['labels']
            td = res['test_res_dict']
            summary_rows.append({
                'family': fam,
                'n': res['n'],
                'strength_matched': mres['is_matched'],
                'disp_ratio': mres['disp_ratio'],
                'asr_diff': mres['asr_diff'],
                'mean_diff': td['mean']['mean_diff'],
                'mean_label': lbls['mean'],
                'floor_diff': td['floor']['mean_diff'],
                'floor_label': lbls['floor'],
                'drop_diff': td['drop']['mean_diff'],
                'drop_label': lbls['drop']
            })
    summary_csv_path = os.path.join(args.results_dir, "table_summary.csv")
    with open(summary_csv_path, 'w', newline='', encoding='utf-8') as cf:
        fieldnames = ['family', 'n', 'strength_matched', 'disp_ratio', 'asr_diff',
                      'mean_diff', 'mean_label', 'floor_diff', 'floor_label', 'drop_diff', 'drop_label']
        writer = csv.DictWriter(cf, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)

    # Save JSON summary artifact
    out_fname = f"eval_summary_{args.regime}_seeds_{'_'.join(map(str, args.seeds))}.json"
    out_path = os.path.join(args.results_dir, out_fname)
    serializable = {}
    for fam, res in eval_results.items():
        if res:
            serializable[fam] = {
                'n': res['n'],
                'strength_matched': res['match_stat']['is_matched'],
                'disp_ratio': res['match_stat']['disp_ratio'],
                'labels': res['labels']
            }
    if 'outcome' in locals() and outcome is not None:
        serializable['confirmatory_outcome'] = {
            'outcome': outcome,
            'description': outcome_desc
        }
    with open(out_path, 'w', encoding='utf-8') as jf:
        json.dump(serializable, jf, indent=2)
    print(f"\nSaved evaluation summary artifact to: {out_path}")
    print(f"Generated CSV tables in: {args.results_dir}")

if __name__ == '__main__':
    main()
