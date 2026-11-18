"""
Apply the pre-registered decision rule (PREREG_SOFT_HETEROGENEITY.md + Amendment A) to the
tuning-seed sweep. Reads VALIDATION accuracy only; the test split is never read.

Metrics per run, at unlearning round 100 (last element of *_unlearn.json, as in Stage 4):
  floor = min retained-client val acc     mean = mean retained-client val acc
  asr   = forget client's acc on its own triggered training images (backdoor label)
Sanity checks per run:
  - forget client in params matches the saved trigger file seed<s>_client_<c>_backdoor.npy
  - backdoor present before unlearning: round-0 ASR (first element of final .json) >= 80%

Usage (from FedOSD/):  python analyze_sweep.py   -> prints tables, writes results_soft/sweep_decision.json
"""
import glob
import json
import os
import sys

import numpy as np

ROOT = 'UnlearningTask_patch4'
SEEDS = [101, 102, 103]
NICK = {'pat': 'cifar10 pat balanced B200 NC2 N10', 'dir0.5': 'cifar10 dir0.5 B200 NC2 N10',
        'dir0.1': 'cifar10 dir0.1 B200 NC2 N10'}
NAME = ('seed{s} N10 C1.0 FedOSD CNN_CIFAR10 E1 lr0.0004 decay0.999 g_clip-1.0 '
        'proj_two_sided wm_{wm} lam{lam} eta{eta} rlr0.0001')
ADAPT_MIN_GAIN = 2.0   # pp, pre-registered
ASR_MAX = 5.0          # %, pre-registered (Amendment A: patch4 ASR)
ROUND0_ASR_MIN = 80.0  # sanity: backdoor must exist before unlearning


def base(split):
    return os.path.join(ROOT, 'CNN_CIFAR10', NICK[split], 'UN1', 'E1')


def load_run(split, seed, eta, lam):
    wm = 'uniform' if eta == 0 else 'deficit'
    stem = os.path.join(base(split), 'C1.0', 'FedOSD', NAME.format(s=seed, wm=wm, lam=lam, eta=eta))
    f_unl, f_fin = stem + '_unlearn.json', stem + '.json'
    if not os.path.exists(f_unl):
        return None, f'missing {os.path.basename(f_unl)}'
    d = json.load(open(f_unl))
    ch = d['metric_log']['client_metric_history']
    forget = d['params']['unlearn_client_id_list']
    if len(forget) != 1:
        return None, 'expected one forget client'
    fc = forget[0]
    problems = []
    if not os.path.exists(os.path.join(base(split), 'backdoors', f'seed{seed}_client_{fc}_backdoor.npy')):
        problems.append(f'no trigger file for client {fc} (pretrain/unlearn client mismatch?)')
    vals = []
    for cid, h in enumerate(ch):
        if cid == fc:
            continue
        v = h.get('val_accuracy', [])
        if not v:
            return None, f'client {cid} has no val_accuracy'
        vals.append(v[-1])
    asr = ch[fc]['test_accuracy'][-1]  # forget client's test data = its triggered training images
    asr0 = None
    if os.path.exists(f_fin):
        asr0 = json.load(open(f_fin))['metric_log']['client_metric_history'][fc]['test_accuracy'][0]
        if asr0 < ROUND0_ASR_MIN:
            problems.append(f'round-0 ASR {asr0:.1f}% < {ROUND0_ASR_MIN}% (backdoor absent?)')
    else:
        problems.append('final .json missing (round-0 check skipped)')
    return {'floor': float(np.min(vals)), 'mean': float(np.mean(vals)), 'asr': float(asr),
            'asr_round0': asr0, 'forget_client': fc, 'problems': problems}, None


def collect(split, eta, lam):
    runs, errs = {}, []
    for s in SEEDS:
        r, e = load_run(split, s, eta, lam)
        if e:
            errs.append(f'{split} eta{eta} lam{lam} s{s}: {e}')
        else:
            runs[s] = r
            errs += [f'{split} eta{eta} lam{lam} s{s}: {p}' for p in r['problems']]
    return runs, errs


def fmt(runs, key):
    v = [runs[s][key] for s in SEEDS if s in runs]
    return f'{np.mean(v):6.2f} +/- {np.std(v):4.2f} (n={len(v)})' if v else '   n/a'


def main():
    out, errors = {'tables': {}}, []
    print('=== Heterogeneity test (lam=20), retained VAL accuracy at unlearning round 100 ===')
    print(f'{"split":7} {"eta":>4}  {"floor":>22}  {"mean":>22}  {"forget ASR":>22}')
    runs = {}
    for split in NICK:
        for eta in (0.0, 0.4, 1.0):
            r, e = collect(split, eta, 20.0)
            runs[(split, eta)] = r
            errors += e
            print(f'{split:7} {eta:4}  {fmt(r, "floor")}  {fmt(r, "mean")}  {fmt(r, "asr")}')
            out['tables'][f'{split}_eta{eta}_lam20'] = r

    def gain(split, eta):
        a, b = runs[(split, eta)], runs[(split, 0.0)]
        common = [s for s in SEEDS if s in a and s in b]
        return float(np.mean([a[s]['floor'] - b[s]['floor'] for s in common])) if common else float('nan')

    g01 = {eta: gain('dir0.1', eta) for eta in (0.4, 1.0)}
    eta_star = max(g01, key=lambda k: g01[k])
    G = {sp: gain(sp, eta_star) for sp in NICK}
    complete = not errors and all(np.isfinite(v) for v in list(G.values()) + list(g01.values()))
    adaptive = G['dir0.1'] >= ADAPT_MIN_GAIN and G['dir0.1'] > G['pat']
    print(f'\neta* (chosen on dir0.1) = {eta_star};  G = floor(eta*) - floor(eta=0):  '
          + ', '.join(f'{k}={v:+.2f} pp' for k, v in G.items()))
    verdict = ('UNDECIDED: missing runs or failed sanity checks (see PROBLEMS)' if not complete else
               'ADAPTIVE is the main method' if adaptive else 'UNIFORM soft FedOSD (eta=0) is the main method')
    print(f'Rule: G(dir0.1) >= {ADAPT_MIN_GAIN} AND G(dir0.1) > G(pat)  ->  {verdict}')

    print('\n=== Lambda selection (pat, eta=0), VAL mean subject to forget ASR <= 5% on ALL seeds ===')
    lam_rows = {}
    for lam in (5.0, 20.0, 100.0):
        r, e = (runs[('pat', 0.0)], []) if lam == 20.0 else collect('pat', 0.0, lam)
        errors += e
        ok = len(r) == len(SEEDS) and all(r[s]['asr'] <= ASR_MAX for s in SEEDS)
        lam_rows[lam] = {'mean': float(np.mean([r[s]['mean'] for s in r])) if r else float('nan'), 'eligible': ok}
        print(f'lam={lam:6}  mean {fmt(r, "mean")}  floor {fmt(r, "floor")}  ASR {fmt(r, "asr")}  '
              f'{"eligible" if ok else "NOT eligible"}')
        out['tables'][f'pat_eta0.0_lam{lam}'] = r
    elig = {k: v for k, v in lam_rows.items() if v['eligible']}
    lam_star = max(elig, key=lambda k: elig[k]['mean']) if elig else None
    print(f'Chosen lambda: {lam_star if lam_star is not None else "NONE eligible -> report and discuss before final runs"}')

    if errors:
        print('\n!!! PROBLEMS (resolve before trusting the decision):')
        for e in errors:
            print('  -', e)
    out.update({'eta_star': eta_star, 'G': G, 'verdict': verdict,
                'adaptive_main': bool(adaptive) if complete else None, 'lambda_star': lam_star,
                'lambda_rows': {str(k): v for k, v in lam_rows.items()}, 'problems': errors,
                'rule': 'PREREG_SOFT_HETEROGENEITY.md (+ Amendment A); validation split only'})
    os.makedirs('results_soft', exist_ok=True)
    json.dump(out, open('results_soft/sweep_decision.json', 'w'), indent=2)
    print('\nwrote results_soft/sweep_decision.json')
    sys.exit(1 if errors else 0)


if __name__ == '__main__':
    main()
