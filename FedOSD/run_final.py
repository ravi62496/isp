"""
Final reporting runs on seeds 31-33 (pat split), with the configuration frozen by the
pre-registered decision (results_soft/sweep_decision.json, written by analyze_sweep.py).
Refuses to start unless that decision is complete (no PROBLEMS, lambda chosen).

Phases (each finishes before the next starts; re-running skips jobs with a .done marker):
  A  pretrain (FedAvg R100, patch4 backdoor)
  G  backdoor gate on validation halves (export_bundle.py --gate); stops on any FAIL
  B  retrain-from-scratch R300 | FedOSD hard (E4) | FedOSD hard + plain post-training (E3)
     | soft FedOSD uniform (lam*) | soft FedOSD adaptive (eta*, lam*) | plain GA baseline
  C  export audit bundles -> ../bundles/seed<s>, run the audit -> ../results_final/seed<s>

All runs of one seed use this machine and its data pool; pool MD5 + git commit are recorded in
results_final/provenance.json.

Usage (on the server, from FedOSD/, venv active):
  nohup python -u run_final.py --gpus 2 3 4 5 --per_gpu 2 > logs_final/final.out 2>&1 &
"""
import argparse
import glob
import hashlib
import json
import os
import subprocess
import sys

import run_soft_sweep as rs

FINAL_SEEDS = [31, 32, 33]
SPLIT = 'pat'
rs.LOG_DIR = 'logs_final'
OUT = os.path.join('..', 'results_final')


def jobs_B(seed, lam, eta):
    p = rs.prefix(seed, SPLIT)
    un = ' --R 200 --UR 100 --lr 0.0004 --r_lr 0.0001'
    hard = un + ' --algorithm FedOSD --proj_mode hard --step_scale 1.0 --adap_lam 100000.0 --adap_weight_eta 0.0'
    j = [(f'retrain_s{seed}', p + ' --algorithm FedAvg --retrain True --test_interval 5 --R 300 --UR 0 --lr 0.05'),
         (f'hard_s{seed}', p + hard),
         (f'hard_recplain_s{seed}', p + hard + ' --recovery_projection False'),
         (f'ga_s{seed}', p + un + ' --algorithm FedAvg_GA'),
         rs.soft_job(seed, SPLIT, 0.0, lam)]
    if eta != 0.0:
        j.append(rs.soft_job(seed, SPLIT, eta, lam))
    return j


def md5(path):
    h = hashlib.md5()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 24), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gpus', type=int, nargs='+', default=[0])
    ap.add_argument('--per_gpu', type=int, default=2)
    a = ap.parse_args()
    os.makedirs(rs.LOG_DIR, exist_ok=True)
    os.makedirs(OUT, exist_ok=True)

    dec = json.load(open('results_soft/sweep_decision.json'))
    if dec.get('problems') or dec.get('lambda_star') is None or dec.get('adaptive_main') is None:
        sys.exit('Decision incomplete (problems / no lambda / undecided) -> not starting final runs.')
    lam = float(dec['lambda_star'])
    eta_adapt = float(dec['eta_star'])
    eta_main = eta_adapt if dec['adaptive_main'] else 0.0
    print(f'[config] lambda*={lam}  eta*={eta_adapt}  main method eta={eta_main}  ({dec["verdict"]})', flush=True)

    pools = glob.glob('tfedplat/pool/CIFAR10_pat_pool_*.npy')
    prov = {'git_commit': subprocess.run(['git', 'rev-parse', 'HEAD'], capture_output=True, text=True).stdout.strip(),
            'host': os.uname().nodename if hasattr(os, 'uname') else '', 'pool_md5': {os.path.basename(p): md5(p) for p in pools},
            'decision': dec, 'lambda': lam, 'eta_main': eta_main, 'seeds': FINAL_SEEDS, 'split': SPLIT}
    json.dump(prov, open(os.path.join(OUT, 'provenance.json'), 'w'), indent=2)
    print('[provenance]', prov['git_commit'], prov['pool_md5'], flush=True)

    pre = [rs.pretrain_job(s, SPLIT) for s in FINAL_SEEDS if not rs.pretrain_exists(s, SPLIT)]
    print(f'\n=== phase A: {len(pre)} pretrain jobs ===', flush=True)
    if rs.run_phase(pre, a.gpus, a.per_gpu):
        sys.exit('pretraining failed')

    print('\n=== phase G: backdoor gate ===', flush=True)
    bad = []
    for s in FINAL_SEEDS:
        r = subprocess.run([sys.executable, 'export_bundle.py', '--gate', '--seed', str(s), '--split', SPLIT],
                           capture_output=True, text=True)
        print(*[l for l in r.stdout.splitlines() if l.startswith('[GATE]')] or [r.stdout[-500:] + r.stderr[-500:]], flush=True)
        if r.returncode != 0:
            bad.append(s)
    if bad:
        sys.exit(f'gate FAILED for seeds {bad}; no unlearning runs started.')

    jb = [j for s in FINAL_SEEDS for j in jobs_B(s, lam, eta_adapt)]
    print(f'\n=== phase B: {len(jb)} jobs ===', flush=True)
    failed = rs.run_phase(jb, a.gpus, a.per_gpu)
    if failed:
        print('FAILED jobs:', failed, flush=True)

    print('\n=== phase C: bundles + audit ===', flush=True)
    for s in FINAL_SEEDS:
        b = os.path.join('..', 'bundles', f'seed{s}')
        r = subprocess.run([sys.executable, 'export_bundle.py', '--seed', str(s), '--split', SPLIT,
                            '--soft_lam', str(lam), '--soft_eta', str(eta_main), '--out', b], capture_output=True, text=True)
        print(r.stdout[-1500:], r.stderr[-1500:], flush=True)
        if r.returncode != 0:
            continue
        models = [m for m in json.load(open(os.path.join(b, 'meta.json')))['models']]
        r = subprocess.run([sys.executable, '-m', 'audit.run_audit', '--bundle', os.path.abspath(b),
                            '--out', os.path.abspath(os.path.join(OUT, f'seed{s}')), '--models', *models],
                           cwd=os.path.join('..', 'audit_code'), capture_output=True, text=True)
        open(os.path.join(rs.LOG_DIR, f'audit_s{s}.log'), 'w').write(r.stdout + r.stderr)
        print(f'[audit] seed {s} rc={r.returncode} -> {rs.LOG_DIR}/audit_s{s}.log', flush=True)
    print('\nAll final phases complete.', flush=True)


if __name__ == '__main__':
    main()
