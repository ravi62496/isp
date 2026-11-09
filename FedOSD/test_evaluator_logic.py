"""Unit tests for evaluate_confirmation.py decision logic at n=10 (synthetic data, no result files needed).
Run:  python test_evaluator_logic.py            (tests ./evaluate_confirmation.py)
      EVAL_PATH=other.py python test_evaluator_logic.py
"""
import os, sys, unittest, importlib.util
import numpy as np

EVAL_PATH = os.environ.get('EVAL_PATH', 'evaluate_confirmation.py')
sys.argv = [sys.argv[0]]
spec = importlib.util.spec_from_file_location('ev', EVAL_PATH)
ev = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ev)

def make(mean, sd, n=10, seed=0):
    r = np.random.default_rng(seed).standard_normal(n)
    r = (r - r.mean()) / r.std(ddof=1)
    return mean + sd * r

def labels(diffs, matched=True, allow_eq=False, n=10):
    tests, ps = {}, []
    for m in ['mean', 'floor', 'drop']:
        sp = ev.PRIMARY_METRIC_SPECS[m]
        tests[m] = ev.compute_hypothesis_tests(diffs[m], practical_thresh=sp['practical'],
                                               equiv_margin=sp['equiv_margin'], n_resamples=500)
        ps.append(tests[m]['p_t'])
    hp = ev.apply_holm_adjustment(ps)
    return {m: ev.assign_label(tests[m], hp[i], strength_matched=matched, n=n, allow_equivalence=allow_eq)
            for i, m in enumerate(['mean', 'floor', 'drop'])}

def fam(lbls, matched=True):
    return {'labels': lbls, 'match_stat': {'is_matched': matched}}

class TestLabels(unittest.TestCase):
    def test_f1_floor_inferior_when_tuning_effects_hold(self):
        l = labels({'mean': make(0.92, 1.1), 'floor': make(-3.93, 1.6, seed=1), 'drop': make(-2.47, 3.0, seed=2)})
        self.assertEqual(l['floor'], 'Inferior')
        self.assertNotIn('Equivalent', l.values())

    def test_round100_never_equivalent(self):
        l = labels({'mean': make(0.0, 0.21), 'floor': make(0.0, 0.45, seed=1), 'drop': make(0.0, 0.55, seed=2)}, allow_eq=False)
        self.assertTrue(all(v == 'Inconclusive' for v in l.values()), l)

    def test_f2_small_detectable_difference_is_equivalent_not_negligible(self):
        # tuning-seed round-200 effect (+0.20, SD 0.21) at n=10 must not block Outcome B
        l = labels({'mean': make(0.20, 0.21), 'floor': make(0.23, 0.45, seed=1), 'drop': make(0.35, 0.55, seed=2)}, allow_eq=True)
        self.assertEqual(l['mean'], 'Equivalent', l)

    def test_superior_requires_practical_threshold(self):
        l = labels({'mean': make(1.5, 0.3), 'floor': make(2.5, 0.3, seed=1), 'drop': make(2.5, 0.3, seed=2)})
        self.assertTrue(all(v == 'Superior' for v in l.values()), l)
        l2 = labels({'mean': make(0.6, 0.1), 'floor': make(2.5, 0.3, seed=1), 'drop': make(2.5, 0.3, seed=2)})
        self.assertEqual(l2['mean'], 'Better but negligible')

    def test_not_comparable_and_descriptive(self):
        d = {'mean': make(1.5, 0.3), 'floor': make(2.5, 0.3, seed=1), 'drop': make(2.5, 0.3, seed=2)}
        self.assertTrue(all(v == 'not comparable' for v in labels(d, matched=False).values()))
        d9 = {k: make(2.0, 0.3, n=9) for k in d}
        self.assertTrue(all(v == 'descriptive (n<10)' for v in labels(d9, n=9).values()))

    def test_holm_known_values(self):
        adj = ev.apply_holm_adjustment([0.04, 0.001, 0.02])
        self.assertTrue(np.allclose(adj, [0.04, 0.003, 0.04]), adj)

class TestOutcome(unittest.TestCase):
    def setUp(self):
        if not hasattr(ev, 'determine_outcome'):
            self.skipTest('determine_outcome() not present (outcome logic is inline in main)')

    def test_A(self):
        o, _ = ev.determine_outcome(fam({'mean': 'Superior', 'floor': 'Superior', 'drop': 'Inconclusive'}),
                                    fam({'mean': 'Equivalent', 'floor': 'Equivalent', 'drop': 'Equivalent'}), 10)
        self.assertEqual(o, 'Outcome A')

    def test_B(self):
        o, _ = ev.determine_outcome(fam({'mean': 'Inconclusive', 'floor': 'Inconclusive', 'drop': 'Inconclusive'}),
                                    fam({'mean': 'Equivalent', 'floor': 'Equivalent', 'drop': 'Equivalent'}), 10)
        self.assertEqual(o, 'Outcome B')

    def test_B_requires_f1_strength_matched(self):
        o, _ = ev.determine_outcome(fam({'mean': 'not comparable', 'floor': 'not comparable', 'drop': 'not comparable'}, matched=False),
                                    fam({'mean': 'Equivalent', 'floor': 'Equivalent', 'drop': 'Equivalent'}), 10)
        self.assertEqual(o, 'Outcome C')

    def test_C_inferior_floor(self):
        o, d = ev.determine_outcome(fam({'mean': 'Inconclusive', 'floor': 'Inferior', 'drop': 'Inconclusive'}),
                                    fam({'mean': 'Equivalent', 'floor': 'Equivalent', 'drop': 'Equivalent'}), 10)
        self.assertEqual(o, 'Outcome C'); self.assertIn('Inferior', d)

    def test_n_lt_10(self):
        o, _ = ev.determine_outcome(fam({}), fam({}), 9)
        self.assertTrue(o.startswith('Descriptive'))

class TestGit(unittest.TestCase):
    def test_dirty_check_ignores_result_and_data_dirs(self):
        cmd = getattr(ev, 'GIT_STATUS_CMD', None)
        if cmd is None:
            self.skipTest('GIT_STATUS_CMD not defined')
        joined = ' '.join(cmd)
        self.assertIn('--untracked-files=no', joined)
        self.assertIn(':(exclude)UnlearningTask', joined)

if __name__ == '__main__':
    unittest.main(verbosity=2)
