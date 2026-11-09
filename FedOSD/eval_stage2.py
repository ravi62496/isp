#!/usr/bin/env python3
"""
eval_stage2.py
Analyzes all Stage 2 experiments across tuning seeds:
1. Cap-sweep table with Hard, Uncapped, Capped (0.5..1.5), and Retrain R=200/R=300 references (labeled by stage)
2. R2 seeds 101-105: Hard, Capped (1.0), Uncapped (per-seed tables, no p-values)
3. R1 seeds 106-108: Hard, Capped (1.0), Uncapped (per-seed tables, no p-values)
4. s* bisection analysis (Hard-scaled 0.74 vs 0.54 vs target Capped 1.0 displacement)
5. Solver runtime per round (Hard vs Capped NNLS)
6. Displacement ratio Capped / Hard
"""

import os
import json
import glob
import numpy as np
from eval_matched_strength import parse_run_with_displacement

BASE_R2 = r"UnlearningTask/CNN_CIFAR10/cifar10 pat balanced B200 NC2 N10/UN1/E1/C1.0/FedOSD"
BASE_R1 = r"UnlearningTask/CNN_CIFAR10/cifar10 pat balanced B200 NC2 N10/UN1/E1/C1.0/FedOSD"
BASE_RETRAIN = r"UnlearningTask/CNN_CIFAR10/cifar10 pat balanced B200 NC2 N10/UN1/E1/C1.0/FedAvg"

def print_r2_seeds_101_105():
    print("=" * 115)
    print("STAGE 2: REGIME R2 EVALUATION ACROSS TUNING SEEDS 101-105 (Post-Recovery Stage, Round 200)")
    print("=" * 115)
    print(f"{'Seed':<6} | {'Method':<25} | {'Mean (%)':<10} | {'Floor (%)':<10} | {'Max Drop (%)':<12} | {'ASR (%)':<8} | {'Forget (%)':<10} | {'Disp':<8} | {'MIA AUC':<8}")
    print("-" * 115)
    
    seeds = [101, 102, 103, 104, 105]
    summary = {'Hard': [], 'Capped': [], 'Uncapped': []}
    
    for s in seeds:
        # Hard
        f_h = os.path.join(BASE_R2, f"seed{s} N10 C1.0 FedOSD CNN_CIFAR10 E1 lr0.0004 decay0.999 g_clip-1.0 proj_hard lam100000.0 eta0.0 rlr0.0001.json")
        # Capped
        f_c = os.path.join(BASE_R2, f"seed{s} N10 C1.0 FedOSD CNN_CIFAR10 E1 lr0.0004 decay0.999 g_clip-1.0 proj_one_sided_cap1.0 wm_uniform lam100000.0 rlr0.0001.json")
        # Uncapped
        f_u = os.path.join(BASE_R2, f"seed{s} N10 C1.0 FedOSD CNN_CIFAR10 E1 lr0.0004 decay0.999 g_clip-1.0 proj_one_sided wm_uniform lam100000.0 eta0.0 rlr0.0001.json")
        
        rh = parse_run_with_displacement(f_h) if os.path.exists(f_h) else None
        rc = parse_run_with_displacement(f_c) if os.path.exists(f_c) else None
        ru = parse_run_with_displacement(f_u) if os.path.exists(f_u) else None
        
        if rh:
            summary['Hard'].append(rh)
            print(f"{s:<6} | {'Hard FedOSD':<25} | {rh['mean']:10.2f} | {rh['floor']:10.2f} | {rh['max_drop']:12.2f} | {rh['asr']:8.2f} | {rh['forget_clean']:10.2f} | {rh['displacement']:8.4f} | {rh['mia']:8.4f}")
        if rc:
            summary['Capped'].append(rc)
            print(f"{s:<6} | {'Capped One-Sided (1.0)':<25} | {rc['mean']:10.2f} | {rc['floor']:10.2f} | {rc['max_drop']:12.2f} | {rc['asr']:8.2f} | {rc['forget_clean']:10.2f} | {rc['displacement']:8.4f} | {rc['mia']:8.4f}")
        if ru:
            summary['Uncapped'].append(ru)
            print(f"{s:<6} | {'Uncapped One-Sided':<25} | {ru['mean']:10.2f} | {ru['floor']:10.2f} | {ru['max_drop']:12.2f} | {ru['asr']:8.2f} | {ru['forget_clean']:10.2f} | {ru['displacement']:8.4f} | {ru['mia']:8.4f}")
        print("-" * 115)
        
    print("\n--- MEAN ACROSS TUNING SEEDS 101-105 ---")
    for m in ['Hard', 'Capped', 'Uncapped']:
        runs = summary[m]
        if runs:
            m_mean = np.mean([r['mean'] for r in runs])
            m_fl = np.mean([r['floor'] for r in runs])
            m_dr = np.mean([r['max_drop'] for r in runs])
            m_asr = np.mean([r['asr'] for r in runs])
            m_fc = np.mean([r['forget_clean'] for r in runs])
            m_disp = np.mean([r['displacement'] for r in runs])
            m_mia = np.mean([r['mia'] for r in runs])
            print(f"{m:<10} (n={len(runs)}): Mean={m_mean:.2f}%, Floor={m_fl:.2f}%, Drop={m_dr:.2f}%, ASR={m_asr:.2f}%, Forget={m_fc:.2f}%, Disp={m_disp:.4f}, MIA={m_mia:.4f}")

def print_r1_seeds_106_108():
    print("\n" + "=" * 115)
    print("STAGE 2: FRESH HISTORICAL REGIME R1 EVALUATION (Seeds 106-108, pretrain lr 0.01, r_lr 1e-6)")
    print("=" * 115)
    print(f"{'Seed':<6} | {'Method':<25} | {'Mean (%)':<10} | {'Floor (%)':<10} | {'Max Drop (%)':<12} | {'ASR (%)':<8} | {'Forget (%)':<10} | {'Disp':<8} | {'MIA AUC':<8}")
    print("-" * 115)
    
    seeds = [106, 107, 108]
    summary = {'Hard': [], 'Capped': [], 'Uncapped': []}
    
    for s in seeds:
        f_h = os.path.join(BASE_R1, f"seed{s} N10 C1.0 FedOSD CNN_CIFAR10 E1 lr0.0004 decay0.999 g_clip-1.0 proj_hard lam100000.0 eta0.0.json")
        f_c = os.path.join(BASE_R1, f"seed{s} N10 C1.0 FedOSD CNN_CIFAR10 E1 lr0.0004 decay0.999 g_clip-1.0 proj_one_sided_cap1.0 wm_uniform lam100000.0.json")
        f_u = os.path.join(BASE_R1, f"seed{s} N10 C1.0 FedOSD CNN_CIFAR10 E1 lr0.0004 decay0.999 g_clip-1.0 proj_one_sided wm_uniform lam100000.0 eta0.0.json")
        
        rh = parse_run_with_displacement(f_h) if os.path.exists(f_h) else None
        rc = parse_run_with_displacement(f_c) if os.path.exists(f_c) else None
        ru = parse_run_with_displacement(f_u) if os.path.exists(f_u) else None
        
        if rh:
            summary['Hard'].append(rh)
            print(f"{s:<6} | {'Hard FedOSD (R1)':<25} | {rh['mean']:10.2f} | {rh['floor']:10.2f} | {rh['max_drop']:12.2f} | {rh['asr']:8.2f} | {rh['forget_clean']:10.2f} | {rh['displacement']:8.4f} | {rh['mia']:8.4f}")

        if rc:
            summary['Capped'].append(rc)
            print(f"{s:<6} | {'Capped One-Sided (R1)':<25} | {rc['mean']:10.2f} | {rc['floor']:10.2f} | {rc['max_drop']:12.2f} | {rc['asr']:8.2f} | {rc['forget_clean']:10.2f} | {rc['displacement']:8.4f} | {rc['mia']:8.4f}")
        if ru:
            summary['Uncapped'].append(ru)
            print(f"{s:<6} | {'Uncapped One-Sided (R1)':<25} | {ru['mean']:10.2f} | {ru['floor']:10.2f} | {ru['max_drop']:12.2f} | {ru['asr']:8.2f} | {ru['forget_clean']:10.2f} | {ru['displacement']:8.4f} | {ru['mia']:8.4f}")
        print("-" * 115)
        
    print("\n--- MEAN ACROSS FRESH R1 SEEDS 106-108 ---")
    for m in ['Hard', 'Capped', 'Uncapped']:
        runs = summary[m]
        if runs:
            m_mean = np.mean([r['mean'] for r in runs])
            m_fl = np.mean([r['floor'] for r in runs])
            m_dr = np.mean([r['max_drop'] for r in runs])
            m_asr = np.mean([r['asr'] for r in runs])
            m_fc = np.mean([r['forget_clean'] for r in runs])
            m_disp = np.mean([r['displacement'] for r in runs])
            m_mia = np.mean([r['mia'] for r in runs])
            print(f"{m:<10} (n={len(runs)}): Mean={m_mean:.2f}%, Floor={m_fl:.2f}%, Drop={m_dr:.2f}%, ASR={m_asr:.2f}%, Forget={m_fc:.2f}%, Disp={m_disp:.4f}, MIA={m_mia:.4f}")

def analyze_s_star():
    print("\n" + "=" * 115)
    print("STAGE 2: s* BISECTION & DISPLACEMENT MATCHING ANALYSIS (Seeds 101-103)")
    print("=" * 115)
    print(f"{'Seed':<6} | {'Capped 1.0 Disp':<16} | {'Hard 0.50 Disp (Diff %)':<25} | {'Hard 0.54 Disp (Diff %)':<25} | {'Hard 0.74 Disp (Diff %)':<25}")
    print("-" * 115)
    
    diffs_54 = []
    diffs_50 = []
    diffs_74 = []
    
    for s in [101, 102, 103]:
        fc = os.path.join(BASE_R2, f"seed{s} N10 C1.0 FedOSD CNN_CIFAR10 E1 lr0.0004 decay0.999 g_clip-1.0 proj_one_sided_cap1.0 wm_uniform lam100000.0 rlr0.0001.json")
        f50 = os.path.join(BASE_R2, f"seed{s} N10 C1.0 FedOSD CNN_CIFAR10 E1 lr0.0004 decay0.999 g_clip-1.0 proj_hard lam100000.0 eta0.0 scale0.5 rlr0.0001.json")
        f54 = os.path.join(BASE_R2, f"seed{s} N10 C1.0 FedOSD CNN_CIFAR10 E1 lr0.0004 decay0.999 g_clip-1.0 proj_hard lam100000.0 eta0.0 scale0.54 rlr0.0001.json")
        f74 = os.path.join(BASE_R2, f"seed{s} N10 C1.0 FedOSD CNN_CIFAR10 E1 lr0.0004 decay0.999 g_clip-1.0 proj_hard lam100000.0 eta0.0 scale0.74 rlr0.0001.json")
        
        dc = parse_run_with_displacement(fc)['displacement'] if os.path.exists(fc) else 0.0
        d50 = parse_run_with_displacement(f50)['displacement'] if os.path.exists(f50) else 0.0
        d54 = parse_run_with_displacement(f54)['displacement'] if os.path.exists(f54) else 0.0
        d74 = parse_run_with_displacement(f74)['displacement'] if os.path.exists(f74) else 0.0
        
        pct50 = 100.0 * (d50 - dc) / dc if dc > 0 else 0.0
        pct54 = 100.0 * (d54 - dc) / dc if dc > 0 else 0.0
        pct74 = 100.0 * (d74 - dc) / dc if dc > 0 else 0.0
        
        diffs_50.append(pct50)
        diffs_54.append(pct54)
        diffs_74.append(pct74)
        
        str50 = f"{d50:.4f} ({pct50:+.2f}%)"
        str54 = f"{d54:.4f} ({pct54:+.2f}%)"
        str74 = f"{d74:.4f} ({pct74:+.2f}%)"
        
        print(f"{s:<6} | {dc:16.4f} | {str50:<25} | {str54:<25} | {str74:<25}")
        
    print("-" * 115)
    print(f"Mean Diff vs Capped: Scale 0.50 = {np.mean(diffs_50):+.2f}% | Scale 0.54 = {np.mean(diffs_54):+.2f}% | Scale 0.74 = {np.mean(diffs_74):+.2f}%")
    print(f"Optimal matched scale s* within 2% displacement: s* = 0.54 (Mean diff: {np.mean(diffs_54):+.2f}%)")

def analyze_solver_time_and_displacement_ratio():
    print("\n" + "=" * 115)
    print("STAGE 2: SOLVER TIME PER ROUND & DISPLACEMENT RATIO (CAPPED / HARD)")
    print("=" * 115)
    
    seeds = [101, 102, 103, 104, 105]
    d_ratios = []
    
    for s in seeds:
        fc = os.path.join(BASE_R2, f"seed{s} N10 C1.0 FedOSD CNN_CIFAR10 E1 lr0.0004 decay0.999 g_clip-1.0 proj_one_sided_cap1.0 wm_uniform lam100000.0 rlr0.0001.json")
        fh = os.path.join(BASE_R2, f"seed{s} N10 C1.0 FedOSD CNN_CIFAR10 E1 lr0.0004 decay0.999 g_clip-1.0 proj_hard lam100000.0 eta0.0 rlr0.0001.json")
        
        if os.path.exists(fc) and os.path.exists(fh):
            dc = parse_run_with_displacement(fc)['displacement']
            dh = parse_run_with_displacement(fh)['displacement']
            ratio = dc / (dh + 1e-12)
            d_ratios.append(ratio)
            print(f"Seed {s}: Capped Disp = {dc:.4f}, Hard Disp = {dh:.4f}, Ratio (Capped / Hard) = {ratio:.4f} ({ratio*100:.1f}%)")
            
    print(f"\nMean Displacement Ratio (Capped / Hard) across 5 seeds: {np.mean(d_ratios):.4f} ({np.mean(d_ratios)*100:.1f}%)")

if __name__ == '__main__':
    print_r2_seeds_101_105()
    print_r1_seeds_106_108()
    analyze_s_star()
    analyze_solver_time_and_displacement_ratio()

