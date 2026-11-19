"""
Calibrated relearning attack (PREREG_RELEARN.md). Run on the server from FedOSD/, venv active:
  nohup python -u run_calib_relearn.py --gpus 2 3 4 > logs_calib/calib.out 2>&1 &
Steps: retrain on tuning seeds 101-103 -> export tuning bundles -> 9-config grid (retrain +
original, ASR only) -> apply the pre-registered rule -> run the chosen attack on final bundles 31-33.
"""
import argparse
import json
import os
import subprocess
import sys

import numpy as np

import run_soft_sweep as rs
import run_final as rf

rs.LOG_DIR = 'logs_calib'
TUNE, FINAL = [101, 102, 103], [31, 32, 33]
GRID = [(n, lr) for n in (8, 16, 32) for lr in (0.001, 0.003, 0.01)]
AUDIT = os.path.join('..', 'audit_code')
CAL = os.path.abspath(os.path.join('..', 'results_calib'))
FIN = os.path.abspath(os.path.join('..', 'results_final'))


def relearn(bundle, out, n, lr, tag, gpu, extra=()):
    cmd = [sys.executable, '-m', 'audit.relearn', '--bundle', os.path.abspath(bundle), '--out', out,
           '--n', str(n), '--lr', str(lr), '--tag', tag, *extra]
    r = subprocess.run(cmd, cwd=AUDIT, capture_output=True, text=True,
                       env={**os.environ, 'CUDA_VISIBLE_DEVICES': str(gpu)})
    print(r.stdout[-1200:], r.stderr[-600:] if r.returncode else '', flush=True)
    if r.returncode:
        sys.exit(f'relearn failed: {cmd}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gpus', type=int, nargs='+', default=[0])
    a = ap.parse_args()
    os.makedirs(rs.LOG_DIR, exist_ok=True)

    jobs = [(f'retrain_s{s}', rf.jobs_B(s, 100.0, 0.0)[0][1]) for s in TUNE]
    print('=== retrain on tuning seeds ===', flush=True)
    if rs.run_phase(jobs, a.gpus, 1):
        sys.exit('retrain failed')

    for s in TUNE:
        b = os.path.join('..', 'bundles', f'tune_seed{s}')
        r = subprocess.run([sys.executable, 'export_bundle.py', '--seed', str(s), '--split', 'pat',
                            '--soft_lam', '100.0', '--soft_eta', '0.0', '--out', b], capture_output=True, text=True)
        print(r.stdout[-400:], r.stderr[-400:] if r.returncode else '', flush=True)
        if r.returncode:
            sys.exit(f'export failed for tuning seed {s}')

    print('=== calibration grid (tuning seeds) ===', flush=True)
    table = {}
    for n, lr in GRID:
        tag = f'_n{n}_lr{lr}'
        asr = {'retrain': [], 'original': []}
        for s in TUNE:
            out = os.path.join(CAL, f'seed{s}')
            relearn(os.path.join('..', 'bundles', f'tune_seed{s}'), out, n, lr, tag, a.gpus[0],
                    ('--models', 'original', 'retrain', '--fast'))
            summ = json.load(open(os.path.join(out, f'relearn_summary{tag}.json')))['per_model']
            for m in asr:
                asr[m].append(summ[m]['asr_end'])
        table[tag] = {'n': n, 'lr': lr, 'retrain_asr100': float(np.mean(asr['retrain'])),
                      'original_asr100': float(np.mean(asr['original'])),
                      'eligible': float(np.mean(asr['retrain'])) <= 20.0}
        print(f'[grid] n={n:2} lr={lr:<6} retrain ASR@100={table[tag]["retrain_asr100"]:6.2f}  '
              f'original={table[tag]["original_asr100"]:6.2f}  {"eligible" if table[tag]["eligible"] else ""}', flush=True)
    elig = [v for v in table.values() if v['eligible']]
    choice = max(elig, key=lambda v: (v['n'] * v['lr'], v['n'])) if elig else None
    json.dump({'rule': 'PREREG_RELEARN.md', 'grid': table, 'choice': choice},
              open(os.path.join(CAL, 'decision.json'), 'w'), indent=2)
    print('[choice]', choice, flush=True)
    if not choice:
        sys.exit('no eligible attack -> report, do not run on final seeds')

    print('=== calibrated attack on final seeds ===', flush=True)
    for s in FINAL:
        relearn(os.path.join('..', 'bundles', f'seed{s}'), os.path.join(FIN, f'seed{s}'),
                choice['n'], choice['lr'], '_cal', a.gpus[0])
    r = subprocess.run([sys.executable, '-m', 'audit.relearn', '--summarize', FIN, '--tag', '_cal'],
                       cwd=AUDIT, capture_output=True, text=True)
    print(r.stdout, r.stderr[-600:], flush=True)
    print('All calibration phases complete.', flush=True)


if __name__ == '__main__':
    main()
