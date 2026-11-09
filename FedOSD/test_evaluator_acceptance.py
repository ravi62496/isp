#!/usr/bin/env python3
"""
test_evaluator_acceptance.py
Acceptance test suite rebuilt per User Task 6 and Task 3.
Golden values for every arm x seed at round 100 and round 200, plus Retrain at 200/300,
read from golden_values.json and tested against parse_metrics_at_round with tolerance 1e-6.
Also validates Protocol v5.1 rules (strength matching, 'not comparable', 'descriptive (n<10)',
triad specification, and label precedence).
"""

import os
import json
import unittest
import numpy as np

from evaluate_confirmation import (
    parse_metrics_at_round,
    check_strength_matching,
    assign_label,
    compute_hypothesis_tests,
    ARMS,
    PRIMARY_METRIC_SPECS
)
from test_evaluator_logic import TestLabels, TestOutcome, TestGit

BASE_FEDOSD = os.path.abspath(os.path.join(
    "UnlearningTask", "CNN_CIFAR10",
    "cifar10 pat balanced B200 NC2 N10", "UN1", "E1", "C1.0", "FedOSD"
))
BASE_FEDAVG = os.path.abspath(os.path.join(
    "UnlearningTask", "CNN_CIFAR10",
    "cifar10 pat balanced B200 NC2 N10", "UN1", "E1", "C1.0", "FedAvg"
))

# 1. EXPLICIT, TYPED FILE PATHS (Not through any resolver)
EXPLICIT_PATHS = {
    # Round 100 (Unlearn)
    ('A1', 101, 100): os.path.join(BASE_FEDOSD, "seed101 N10 C1.0 FedOSD CNN_CIFAR10 E1 lr0.0004 decay0.999 g_clip-1.0 proj_hard lam100000.0 eta0.0 rlr0.0001_unlearn.json"),
    ('A1', 102, 100): os.path.join(BASE_FEDOSD, "seed102 N10 C1.0 FedOSD CNN_CIFAR10 E1 lr0.0004 decay0.999 g_clip-1.0 proj_hard lam100000.0 eta0.0_unlearn.json"),
    ('A1', 103, 100): os.path.join(BASE_FEDOSD, "seed103 N10 C1.0 FedOSD CNN_CIFAR10 E1 lr0.0004 decay0.999 g_clip-1.0 proj_hard lam100000.0 eta0.0_unlearn.json"),
    ('A1', 104, 100): os.path.join(BASE_FEDOSD, "seed104 N10 C1.0 FedOSD CNN_CIFAR10 E1 lr0.0004 decay0.999 g_clip-1.0 proj_hard lam100000.0 eta0.0 rlr0.0001_unlearn.json"),
    ('A1', 105, 100): os.path.join(BASE_FEDOSD, "seed105 N10 C1.0 FedOSD CNN_CIFAR10 E1 lr0.0004 decay0.999 g_clip-1.0 proj_hard lam100000.0 eta0.0 rlr0.0001_unlearn.json"),

    ('A2', 101, 100): os.path.join(BASE_FEDOSD, "seed101 N10 C1.0 FedOSD CNN_CIFAR10 E1 lr0.0004 decay0.999 g_clip-1.0 proj_hard lam100000.0 eta0.0 scale0.54 rlr0.0001_unlearn.json"),
    ('A2', 102, 100): os.path.join(BASE_FEDOSD, "seed102 N10 C1.0 FedOSD CNN_CIFAR10 E1 lr0.0004 decay0.999 g_clip-1.0 proj_hard lam100000.0 eta0.0 scale0.54 rlr0.0001_unlearn.json"),
    ('A2', 103, 100): os.path.join(BASE_FEDOSD, "seed103 N10 C1.0 FedOSD CNN_CIFAR10 E1 lr0.0004 decay0.999 g_clip-1.0 proj_hard lam100000.0 eta0.0 scale0.54 rlr0.0001_unlearn.json"),
    ('A2', 104, 100): os.path.join(BASE_FEDOSD, "seed104 N10 C1.0 FedOSD CNN_CIFAR10 E1 lr0.0004 decay0.999 g_clip-1.0 proj_hard lam100000.0 eta0.0 scale0.54 rlr0.0001_unlearn.json"),
    ('A2', 105, 100): os.path.join(BASE_FEDOSD, "seed105 N10 C1.0 FedOSD CNN_CIFAR10 E1 lr0.0004 decay0.999 g_clip-1.0 proj_hard lam100000.0 eta0.0 scale0.54 rlr0.0001_unlearn.json"),

    ('A3', 101, 100): os.path.join(BASE_FEDOSD, "seed101 N10 C1.0 FedOSD CNN_CIFAR10 E1 lr0.0004 decay0.999 g_clip-1.0 proj_one_sided_cap1.0 wm_uniform lam100000.0 rlr0.0001_unlearn.json"),
    ('A3', 102, 100): os.path.join(BASE_FEDOSD, "seed102 N10 C1.0 FedOSD CNN_CIFAR10 E1 lr0.0004 decay0.999 g_clip-1.0 proj_one_sided_cap1.0 wm_uniform lam100000.0 rlr0.0001_unlearn.json"),
    ('A3', 103, 100): os.path.join(BASE_FEDOSD, "seed103 N10 C1.0 FedOSD CNN_CIFAR10 E1 lr0.0004 decay0.999 g_clip-1.0 proj_one_sided_cap1.0 wm_uniform lam100000.0 rlr0.0001_unlearn.json"),
    ('A3', 104, 100): os.path.join(BASE_FEDOSD, "seed104 N10 C1.0 FedOSD CNN_CIFAR10 E1 lr0.0004 decay0.999 g_clip-1.0 proj_one_sided_cap1.0 wm_uniform lam100000.0 rlr0.0001_unlearn.json"),
    ('A3', 105, 100): os.path.join(BASE_FEDOSD, "seed105 N10 C1.0 FedOSD CNN_CIFAR10 E1 lr0.0004 decay0.999 g_clip-1.0 proj_one_sided_cap1.0 wm_uniform lam100000.0 rlr0.0001_unlearn.json"),

    ('A4', 101, 100): os.path.join(BASE_FEDOSD, "seed101 N10 C1.0 FedOSD CNN_CIFAR10 E1 lr0.0004 decay0.999 g_clip-1.0 proj_one_sided wm_uniform lam100000.0 eta0.0 rlr0.0001_unlearn.json"),
    ('A4', 102, 100): os.path.join(BASE_FEDOSD, "seed102 N10 C1.0 FedOSD CNN_CIFAR10 E1 lr0.0004 decay0.999 g_clip-1.0 proj_one_sided wm_uniform lam100000.0 eta0.0 rlr0.0001_unlearn.json"),
    ('A4', 103, 100): os.path.join(BASE_FEDOSD, "seed103 N10 C1.0 FedOSD CNN_CIFAR10 E1 lr0.0004 decay0.999 g_clip-1.0 proj_one_sided wm_uniform lam100000.0 eta0.0 rlr0.0001_unlearn.json"),
    ('A4', 104, 100): os.path.join(BASE_FEDOSD, "seed104 N10 C1.0 FedOSD CNN_CIFAR10 E1 lr0.0004 decay0.999 g_clip-1.0 proj_one_sided wm_uniform lam100000.0 eta0.0 rlr0.0001_unlearn.json"),
    ('A4', 105, 100): os.path.join(BASE_FEDOSD, "seed105 N10 C1.0 FedOSD CNN_CIFAR10 E1 lr0.0004 decay0.999 g_clip-1.0 proj_one_sided wm_uniform lam100000.0 eta0.0 rlr0.0001_unlearn.json"),

    # Round 200 (Recovery)
    ('A1', 101, 200): os.path.join(BASE_FEDOSD, "seed101 N10 C1.0 FedOSD CNN_CIFAR10 E1 lr0.0004 decay0.999 g_clip-1.0 proj_hard lam100000.0 eta0.0 rlr0.0001.json"),
    ('A1', 102, 200): os.path.join(BASE_FEDOSD, "seed102 N10 C1.0 FedOSD CNN_CIFAR10 E1 lr0.0004 decay0.999 g_clip-1.0 proj_hard lam100000.0 eta0.0 rlr0.0001.json"),
    ('A1', 103, 200): os.path.join(BASE_FEDOSD, "seed103 N10 C1.0 FedOSD CNN_CIFAR10 E1 lr0.0004 decay0.999 g_clip-1.0 proj_hard lam100000.0 eta0.0 rlr0.0001.json"),
    ('A1', 104, 200): os.path.join(BASE_FEDOSD, "seed104 N10 C1.0 FedOSD CNN_CIFAR10 E1 lr0.0004 decay0.999 g_clip-1.0 proj_hard lam100000.0 eta0.0 rlr0.0001.json"),
    ('A1', 105, 200): os.path.join(BASE_FEDOSD, "seed105 N10 C1.0 FedOSD CNN_CIFAR10 E1 lr0.0004 decay0.999 g_clip-1.0 proj_hard lam100000.0 eta0.0 rlr0.0001.json"),

    ('A2', 101, 200): os.path.join(BASE_FEDOSD, "seed101 N10 C1.0 FedOSD CNN_CIFAR10 E1 lr0.0004 decay0.999 g_clip-1.0 proj_hard lam100000.0 eta0.0 scale0.54 rlr0.0001.json"),
    ('A2', 102, 200): os.path.join(BASE_FEDOSD, "seed102 N10 C1.0 FedOSD CNN_CIFAR10 E1 lr0.0004 decay0.999 g_clip-1.0 proj_hard lam100000.0 eta0.0 scale0.54 rlr0.0001.json"),
    ('A2', 103, 200): os.path.join(BASE_FEDOSD, "seed103 N10 C1.0 FedOSD CNN_CIFAR10 E1 lr0.0004 decay0.999 g_clip-1.0 proj_hard lam100000.0 eta0.0 scale0.54 rlr0.0001.json"),
    ('A2', 104, 200): os.path.join(BASE_FEDOSD, "seed104 N10 C1.0 FedOSD CNN_CIFAR10 E1 lr0.0004 decay0.999 g_clip-1.0 proj_hard lam100000.0 eta0.0 scale0.54 rlr0.0001.json"),
    ('A2', 105, 200): os.path.join(BASE_FEDOSD, "seed105 N10 C1.0 FedOSD CNN_CIFAR10 E1 lr0.0004 decay0.999 g_clip-1.0 proj_hard lam100000.0 eta0.0 scale0.54 rlr0.0001.json"),

    ('A3', 101, 200): os.path.join(BASE_FEDOSD, "seed101 N10 C1.0 FedOSD CNN_CIFAR10 E1 lr0.0004 decay0.999 g_clip-1.0 proj_one_sided_cap1.0 wm_uniform lam100000.0 rlr0.0001.json"),
    ('A3', 102, 200): os.path.join(BASE_FEDOSD, "seed102 N10 C1.0 FedOSD CNN_CIFAR10 E1 lr0.0004 decay0.999 g_clip-1.0 proj_one_sided_cap1.0 wm_uniform lam100000.0 rlr0.0001.json"),
    ('A3', 103, 200): os.path.join(BASE_FEDOSD, "seed103 N10 C1.0 FedOSD CNN_CIFAR10 E1 lr0.0004 decay0.999 g_clip-1.0 proj_one_sided_cap1.0 wm_uniform lam100000.0 rlr0.0001.json"),
    ('A3', 104, 200): os.path.join(BASE_FEDOSD, "seed104 N10 C1.0 FedOSD CNN_CIFAR10 E1 lr0.0004 decay0.999 g_clip-1.0 proj_one_sided_cap1.0 wm_uniform lam100000.0 rlr0.0001.json"),
    ('A3', 105, 200): os.path.join(BASE_FEDOSD, "seed105 N10 C1.0 FedOSD CNN_CIFAR10 E1 lr0.0004 decay0.999 g_clip-1.0 proj_one_sided_cap1.0 wm_uniform lam100000.0 rlr0.0001.json"),

    ('A4', 101, 200): os.path.join(BASE_FEDOSD, "seed101 N10 C1.0 FedOSD CNN_CIFAR10 E1 lr0.0004 decay0.999 g_clip-1.0 proj_one_sided wm_uniform lam100000.0 eta0.0 rlr0.0001.json"),
    ('A4', 102, 200): os.path.join(BASE_FEDOSD, "seed102 N10 C1.0 FedOSD CNN_CIFAR10 E1 lr0.0004 decay0.999 g_clip-1.0 proj_one_sided wm_uniform lam100000.0 eta0.0 rlr0.0001.json"),
    ('A4', 103, 200): os.path.join(BASE_FEDOSD, "seed103 N10 C1.0 FedOSD CNN_CIFAR10 E1 lr0.0004 decay0.999 g_clip-1.0 proj_one_sided wm_uniform lam100000.0 eta0.0 rlr0.0001.json"),
    ('A4', 104, 200): os.path.join(BASE_FEDOSD, "seed104 N10 C1.0 FedOSD CNN_CIFAR10 E1 lr0.0004 decay0.999 g_clip-1.0 proj_one_sided wm_uniform lam100000.0 eta0.0 rlr0.0001.json"),
    ('A4', 105, 200): os.path.join(BASE_FEDOSD, "seed105 N10 C1.0 FedOSD CNN_CIFAR10 E1 lr0.0004 decay0.999 g_clip-1.0 proj_one_sided wm_uniform lam100000.0 eta0.0 rlr0.0001.json"),

    # Retrain Reference Runs
    ('Retrain', 101, 200): os.path.join(BASE_FEDAVG, "seed101 retrain_from_scratch seed101 N10 C1.0 FedAvg CNN_CIFAR10 E1 lr0.05 decay0.999 g_clip-1.0 R300.json"),
    ('Retrain', 102, 200): os.path.join(BASE_FEDAVG, "seed102 retrain_from_scratch seed102 N10 C1.0 FedAvg CNN_CIFAR10 E1 lr0.05 decay0.999 g_clip-1.0 R300.json"),
    ('Retrain', 103, 200): os.path.join(BASE_FEDAVG, "seed103 retrain_from_scratch seed103 N10 C1.0 FedAvg CNN_CIFAR10 E1 lr0.05 decay0.999 g_clip-1.0 R300.json"),
    ('Retrain', 104, 200): os.path.join(BASE_FEDAVG, "seed104 retrain_from_scratch seed104 N10 C1.0 FedAvg CNN_CIFAR10 E1 lr0.05 decay0.999 g_clip-1.0 R300.json"),
    ('Retrain', 105, 200): os.path.join(BASE_FEDAVG, "seed105 retrain_from_scratch seed105 N10 C1.0 FedAvg CNN_CIFAR10 E1 lr0.05 decay0.999 g_clip-1.0 R300.json"),

    ('Retrain', 101, 300): os.path.join(BASE_FEDAVG, "seed101 retrain_from_scratch seed101 N10 C1.0 FedAvg CNN_CIFAR10 E1 lr0.05 decay0.999 g_clip-1.0 R300.json"),
    ('Retrain', 102, 300): os.path.join(BASE_FEDAVG, "seed102 retrain_from_scratch seed102 N10 C1.0 FedAvg CNN_CIFAR10 E1 lr0.05 decay0.999 g_clip-1.0 R300.json"),
    ('Retrain', 103, 300): os.path.join(BASE_FEDAVG, "seed103 retrain_from_scratch seed103 N10 C1.0 FedAvg CNN_CIFAR10 E1 lr0.05 decay0.999 g_clip-1.0 R300.json"),
    ('Retrain', 104, 300): os.path.join(BASE_FEDAVG, "seed104 retrain_from_scratch seed104 N10 C1.0 FedAvg CNN_CIFAR10 E1 lr0.05 decay0.999 g_clip-1.0 R300.json"),
    ('Retrain', 105, 300): os.path.join(BASE_FEDAVG, "seed105 retrain_from_scratch seed105 N10 C1.0 FedAvg CNN_CIFAR10 E1 lr0.05 decay0.999 g_clip-1.0 R300.json"),
}

# 2. INDEPENDENT GOLDEN VALUES (Frozen exact literals from raw files)
GOLDEN_VALUES = json.load(open("golden_values.json", "r", encoding='utf-8'))

class TestEvaluatorAcceptanceRebuilt(unittest.TestCase):

    def test_01_round100_explicit_paths(self):
        """Test all 20 cells (Arms A1-A4 x seeds 101-105) at round 100 with tolerance 1e-6."""
        for arm in ['A1', 'A2', 'A3', 'A4']:
            for s in [101, 102, 103, 104, 105]:
                fpath = EXPLICIT_PATHS[(arm, s, 100)]
                self.assertTrue(os.path.exists(fpath), f"File missing: {fpath}")
                m = parse_metrics_at_round(fpath, target_round=100, stage='unlearn')
                golden = GOLDEN_VALUES[f"{arm}_{s}_rd100"]
                
                self.assertAlmostEqual(m['mean'], golden['mean'], delta=1e-6, msg=f"Mean mismatch for {arm} seed {s}")
                self.assertAlmostEqual(m['worst2'], golden['worst2'], delta=1e-6, msg=f"Worst-2 mismatch for {arm} seed {s}")
                self.assertAlmostEqual(m['floor'], golden['floor'], delta=1e-6, msg=f"Floor mismatch for {arm} seed {s}")
                self.assertAlmostEqual(m['drop'], golden['drop'], delta=1e-6, msg=f"Drop mismatch for {arm} seed {s}")
                self.assertAlmostEqual(m['asr'], golden['asr'], delta=1e-6, msg=f"ASR mismatch for {arm} seed {s}")
                self.assertAlmostEqual(m['forget'], golden['forget'], delta=1e-6, msg=f"Forget mismatch for {arm} seed {s}")
                self.assertAlmostEqual(m['disp'], golden['disp'], delta=1e-6, msg=f"Disp mismatch for {arm} seed {s}")
        print("\n[PASS] test_01: Round 100 explicit paths verified across all 20 cells (tolerance 1e-6).")

    def test_02_round200_explicit_paths(self):
        """Test all 20 cells (Arms A1-A4 x seeds 101-105) at round 200 with tolerance 1e-6."""
        for arm in ['A1', 'A2', 'A3', 'A4']:
            for s in [101, 102, 103, 104, 105]:
                fpath = EXPLICIT_PATHS[(arm, s, 200)]
                self.assertTrue(os.path.exists(fpath), f"File missing: {fpath}")
                m = parse_metrics_at_round(fpath, target_round=200, stage='recovery')
                golden = GOLDEN_VALUES[f"{arm}_{s}_rd200"]
                
                self.assertAlmostEqual(m['mean'], golden['mean'], delta=1e-6, msg=f"Mean mismatch for {arm} seed {s} rd200")
                self.assertAlmostEqual(m['worst2'], golden['worst2'], delta=1e-6, msg=f"Worst-2 mismatch for {arm} seed {s} rd200")
                self.assertAlmostEqual(m['floor'], golden['floor'], delta=1e-6, msg=f"Floor mismatch for {arm} seed {s} rd200")
                self.assertAlmostEqual(m['drop'], golden['drop'], delta=1e-6, msg=f"Drop mismatch for {arm} seed {s} rd200")
                self.assertAlmostEqual(m['asr'], golden['asr'], delta=1e-6, msg=f"ASR mismatch for {arm} seed {s} rd200")
                self.assertAlmostEqual(m['forget'], golden['forget'], delta=1e-6, msg=f"Forget mismatch for {arm} seed {s} rd200")
                self.assertAlmostEqual(m['disp'], golden['disp'], delta=1e-6, msg=f"Disp mismatch for {arm} seed {s} rd200")
        print("[PASS] test_02: Round 200 explicit paths verified across all 20 cells (tolerance 1e-6).")

    def test_03_retrain_explicit_paths(self):
        """Test Retrain references at round 200 and 300 across all 5 seeds with tolerance 1e-6."""
        for s in [101, 102, 103, 104, 105]:
            # Rd 200
            fp200 = EXPLICIT_PATHS[('Retrain', s, 200)]
            self.assertTrue(os.path.exists(fp200), f"File missing: {fp200}")
            m200 = parse_metrics_at_round(fp200, target_round=200, stage='unlearn')
            g200 = GOLDEN_VALUES[f"Retrain_{s}_rd200"]
            self.assertAlmostEqual(m200['mean'], g200['mean'], delta=1e-6)
            self.assertAlmostEqual(m200['worst2'], g200['worst2'], delta=1e-6)
            self.assertAlmostEqual(m200['floor'], g200['floor'], delta=1e-6)
            self.assertAlmostEqual(m200['asr'], g200['asr'], delta=1e-6)
            self.assertAlmostEqual(m200['forget'], g200['forget'], delta=1e-6)

            # Rd 300
            fp300 = EXPLICIT_PATHS[('Retrain', s, 300)]
            self.assertTrue(os.path.exists(fp300), f"File missing: {fp300}")
            m300 = parse_metrics_at_round(fp300, target_round=300, stage='recovery')
            g300 = GOLDEN_VALUES[f"Retrain_{s}_rd300"]
            self.assertAlmostEqual(m300['mean'], g300['mean'], delta=1e-6)
            self.assertAlmostEqual(m300['worst2'], g300['worst2'], delta=1e-6)
            self.assertAlmostEqual(m300['floor'], g300['floor'], delta=1e-6)
            self.assertAlmostEqual(m300['asr'], g300['asr'], delta=1e-6)
            self.assertAlmostEqual(m300['forget'], g300['forget'], delta=1e-6)
        print("[PASS] test_03: Retrain explicit paths at Rd 200 and Rd 300 verified (tolerance 1e-6).")

    def test_04_protocol_v51_rules(self):
        """Verify strength matching logic, primary metrics triad, and decision label rules."""
        # 1. Primary Metrics Triad check
        self.assertEqual(list(PRIMARY_METRIC_SPECS.keys()), ['mean', 'floor', 'drop'])
        self.assertEqual(PRIMARY_METRIC_SPECS['mean']['practical'], 1.0)
        self.assertEqual(PRIMARY_METRIC_SPECS['mean']['equiv_margin'], 0.5)
        self.assertEqual(PRIMARY_METRIC_SPECS['floor']['practical'], 2.0)
        self.assertEqual(PRIMARY_METRIC_SPECS['floor']['equiv_margin'], 1.0)
        self.assertEqual(PRIMARY_METRIC_SPECS['drop']['practical'], 2.0)
        self.assertEqual(PRIMARY_METRIC_SPECS['drop']['equiv_margin'], 1.0)

        # 2. Strength matching check (displacement ratio [0.95, 1.05] and |ASR diff| <= 1.0 pt)
        test_runs = [{'disp': 0.2338, 'asr': 1.29}]
        comp_runs_pass = [{'disp': 0.2317, 'asr': 1.40}]  # ratio 1.009, diff 0.11 -> PASS
        comp_runs_fail_disp = [{'disp': 0.3222, 'asr': 0.68}]  # ratio 0.725 -> FAIL
        comp_runs_fail_asr = [{'disp': 0.2317, 'asr': 3.50}]   # diff 2.21 -> FAIL

        self.assertTrue(check_strength_matching(test_runs, comp_runs_pass)['is_matched'])
        self.assertFalse(check_strength_matching(test_runs, comp_runs_fail_disp)['is_matched'])
        self.assertFalse(check_strength_matching(test_runs, comp_runs_fail_asr)['is_matched'])

        # 3. 'not comparable' label enforcement when strength not matched
        res_pos = {
            'ci95_t': (1.5, 3.0),
            'ci90_t': (1.6, 2.9),
            'mean_diff': 2.25,
            'practical_thresh': 1.0,
            'equiv_margin': 0.5
        }
        self.assertEqual(assign_label(res_pos, holm_p=0.01, strength_matched=False, n=10), 'not comparable')

        # 4. 'descriptive (n<10)' label enforcement when n < 10
        self.assertEqual(assign_label(res_pos, holm_p=0.01, strength_matched=True, n=5), 'descriptive (n<10)')

        # 5. Confirmatory label rules when n >= 10 and strength matched:
        # A. Superior (95% CI > 0, Holm p < 0.05, estimate >= practical)
        self.assertEqual(assign_label(res_pos, holm_p=0.01, strength_matched=True, n=10, allow_equivalence=False), 'Superior')

        # B. Better but negligible (95% CI > 0, Holm p < 0.05, estimate < practical)
        res_small_pos = {
            'ci95_t': (0.1, 0.9),
            'ci90_t': (0.2, 0.8),
            'mean_diff': 0.5,
            'practical_thresh': 1.0,
            'equiv_margin': 0.5
        }
        self.assertEqual(assign_label(res_small_pos, holm_p=0.01, strength_matched=True, n=10, allow_equivalence=False), 'Better but negligible')

        # C. Inferior (95% CI < 0, Holm p < 0.05, estimate <= -practical)
        res_neg = {
            'ci95_t': (-3.0, -1.2),
            'ci90_t': (-2.8, -1.4),
            'mean_diff': -2.1,
            'practical_thresh': 1.0,
            'equiv_margin': 0.5
        }
        self.assertEqual(assign_label(res_neg, holm_p=0.01, strength_matched=True, n=10, allow_equivalence=False), 'Inferior')

        # D. Worse but negligible (95% CI < 0, Holm p < 0.05, estimate > -practical)
        res_small_neg = {
            'ci95_t': (-0.9, -0.1),
            'ci90_t': (-0.8, -0.2),
            'mean_diff': -0.5,
            'practical_thresh': 1.0,
            'equiv_margin': 0.5
        }
        self.assertEqual(assign_label(res_small_neg, holm_p=0.01, strength_matched=True, n=10, allow_equivalence=False), 'Worse but negligible')

        # E. Inconclusive at Round 100 even if 90% CI is inside margin (NO Equivalent at Round 100)
        res_equiv_candidate = {
            'ci95_t': (-0.4, 0.4),
            'ci90_t': (-0.3, 0.3),
            'mean_diff': 0.0,
            'practical_thresh': 1.0,
            'equiv_margin': 0.5
        }
        self.assertEqual(assign_label(res_equiv_candidate, holm_p=0.8, strength_matched=True, n=10, allow_equivalence=False), 'Inconclusive')

        # F. Equivalent at Round 200 (allow_equivalence=True and 90% CI inside margin)
        self.assertEqual(assign_label(res_equiv_candidate, holm_p=0.8, strength_matched=True, n=10, allow_equivalence=True), 'Equivalent')

        print("[PASS] test_04: Protocol v5.1 rules (triad, strength match, label precedence, no-equiv at rd100) verified.")

    def test_05_manifest_enforcement(self):
        """Verify RUN_MANIFEST.csv integrity: exactly one file per cell, valid commit hash."""
        manifest_path = "RUN_MANIFEST.csv"
        self.assertTrue(os.path.exists(manifest_path), f"Missing {manifest_path}")
        import csv
        with open(manifest_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        self.assertEqual(len(rows), 50, "Manifest must contain exactly 50 rows")
        for r in rows:
            self.assertTrue(os.path.exists(r['absolute_file_path']), f"File in manifest does not exist: {r['absolute_file_path']}")
            self.assertNotEqual(r['git_commit'], 'NONE', f"Missing commit hash in {r['absolute_file_path']}")
        print("[PASS] test_05: RUN_MANIFEST.csv verified (50 cells, valid files and commits).")

    def test_06_r1_manifest_and_golden_values(self):
        """Verify all 30 Regime R1 cells against golden_values.json with tolerance 1e-6."""
        r1_manifest_path = "RUN_MANIFEST_R1.csv"
        self.assertTrue(os.path.exists(r1_manifest_path), f"Missing {r1_manifest_path}")
        import csv
        with open(r1_manifest_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        self.assertEqual(len(rows), 30, "R1 manifest must contain exactly 30 rows")
        for r in rows:
            arm = r['arm']
            seed = int(r['seed'])
            stage = r['stage']
            fpath = r.get('absolute_file_path') or r.get('path')
            self.assertTrue(os.path.exists(fpath), f"File in R1 manifest does not exist: {fpath}")
            target_round = 100 if stage == 'unlearn' else 200
            m = parse_metrics_at_round(fpath, target_round=target_round, stage=stage)
            golden = GOLDEN_VALUES[f"R1_{arm}_{seed}_{stage}"]
            self.assertAlmostEqual(m['mean'], golden['mean'], delta=1e-6, msg=f"R1 Mean mismatch for {arm} seed {seed} {stage}")
            self.assertAlmostEqual(m['worst2'], golden['worst2'], delta=1e-6, msg=f"R1 Worst-2 mismatch for {arm} seed {seed} {stage}")
            self.assertAlmostEqual(m['floor'], golden['floor'], delta=1e-6, msg=f"R1 Floor mismatch for {arm} seed {seed} {stage}")
            self.assertAlmostEqual(m['drop'], golden['drop'], delta=1e-6, msg=f"R1 Drop mismatch for {arm} seed {seed} {stage}")
            self.assertAlmostEqual(m['asr'], golden['asr'], delta=1e-6, msg=f"R1 ASR mismatch for {arm} seed {seed} {stage}")
            self.assertAlmostEqual(m['forget'], golden['forget'], delta=1e-6, msg=f"R1 Forget mismatch for {arm} seed {seed} {stage}")
            self.assertAlmostEqual(m['disp'], golden['disp'], delta=1e-6, msg=f"R1 Disp mismatch for {arm} seed {seed} {stage}")
        print("[PASS] test_06: Regime R1 manifest verified across all 30 cells against golden_values.json (tolerance 1e-6).")

if __name__ == '__main__':
    unittest.main()
