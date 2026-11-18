"""
Soft FedOSD heterogeneity test + lambda sweep (PREREG_SOFT_HETEROGENEITY.md).
Tuning seeds 101-103 ONLY. Never pass final seeds (31-33) or confirmatory seeds (21-30).

Phases (each phase finishes before the next starts):
  1a  pretrain seed 101 for each split (creates each cached data pool once, no write race)
  1b  pretrain remaining seeds
  2   unlearning runs: 3 splits x eta{0,0.4,1.0} x 3 seeds (lam=20)  +  pat x lam{5,100} x eta0 x 3 seeds

Usage (on the GPU server, from FedOSD/):
  nohup python -u run_soft_sweep.py --gpus 0 1 2 3 4 5 6 7 > logs_soft_patch4/sweep.out 2>&1 &
  python run_soft_sweep.py --dry_run          # print the job list only
Re-running skips jobs whose .done marker exists.
"""
import argparse
import glob
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from queue import Queue

TUNING_SEEDS = [101, 102, 103]
FORBIDDEN_SEEDS = set(range(21, 31)) | {31, 32, 33}
LOG_DIR = 'logs_soft_patch4'
ROOT = 'UnlearningTask_patch4'  # separate from legacy-trigger outputs (UnlearningTask/)

SPLITS = {  # name -> (dataloader args, nickname used in result folders)
    'pat':   ('--dataloader DataLoader_cifar10_pat', 'cifar10 pat balanced B200 NC2 N10'),
    'dir0.5': ('--dataloader DataLoader_cifar10_dir --Diralpha 0.5', 'cifar10 dir0.5 B200 NC2 N10'),
    'dir0.1': ('--dataloader DataLoader_cifar10_dir --Diralpha 0.1', 'cifar10 dir0.1 B200 NC2 N10'),
}


def prefix(seed, split):
    dl, _ = SPLITS[split]
    return (f"python -u run_UnlearningTask.py --seed {seed} --device {{dev}} --module CNN_CIFAR10 "
            f"{dl} --N 10 --NC 2 --balance True --B 200 --C 1.0 "
            f"--E 1 --decay 0.999 --step_type bgd --unlearn_cn 1 --save_model True --backdoor patch4")


def pretrain_exists(seed, split):
    _, nick = SPLITS[split]
    pat = os.path.join(ROOT, 'CNN_CIFAR10', nick, 'UN1', 'E1',
                       f'seed{seed}_unlearn_task_pretrained_model.pth')
    return len(glob.glob(pat)) > 0


def pretrain_job(seed, split):
    return (f'pre_{split}_s{seed}',
            prefix(seed, split) + ' --algorithm FedAvg --R 100 --UR 0 --lr 0.05 --unlearn_pretrain True')


def soft_job(seed, split, eta, lam):
    wm = 'uniform' if eta == 0 else 'deficit'
    return (f'soft_{split}_lam{lam}_eta{eta}_s{seed}',
            prefix(seed, split) + f' --algorithm FedOSD --R 200 --UR 100 --lr 0.0004 --r_lr 0.0001 '
                                  f'--proj_mode two_sided --weight_mode {wm} --adap_lam {lam} --adap_weight_eta {eta}')


def build(seeds):
    p1a = [pretrain_job(seeds[0], sp) for sp in SPLITS if not pretrain_exists(seeds[0], sp)]
    p1b = [pretrain_job(s, sp) for s in seeds[1:] for sp in SPLITS if not pretrain_exists(s, sp)]
    p2 = [soft_job(s, sp, eta, 20.0) for sp in SPLITS for eta in (0.0, 0.4, 1.0) for s in seeds]
    p2 += [soft_job(s, 'pat', 0.0, lam) for lam in (5.0, 100.0) for s in seeds]
    return [('1a', p1a), ('1b', p1b), ('2', p2)]


def build_pools():
    """Phase 0: create every cached data pool BEFORE any training run.
    The forget client is drawn from the global np.random state (UnlearningTask.py:58). A run that
    *creates* a pool consumes random draws first and so picks a different client than runs that
    *read* the cache -> pretrain backdoor client != unlearned client. Building pools here means
    every training run reads the cache and picks the same client for a given seed."""
    import random
    import numpy as np
    import tfedplat as fp
    for split in SPLITS:
        np.random.seed(0)
        random.seed(0)
        params = {'N': 10, 'NC': 2, 'B': 200, 'balance': 'True'}
        if split == 'pat':
            fp.DataLoader_cifar10_pat(params=params, input_require_shape=[3, -1, -1])
        else:
            params['Diralpha'] = float(split[3:])
            fp.DataLoader_cifar10_dir(params=params, input_require_shape=[3, -1, -1])
        print(f'[pool] {split} ready', flush=True)


def check_backdoor_clients():
    """Print which clients carry a backdoor per split. With pools prebuilt, each seed must map to
    exactly one client; a second file appearing for the same seed means a client mismatch."""
    for split, (_, nick) in SPLITS.items():
        files = glob.glob(os.path.join(ROOT, 'CNN_CIFAR10', nick, 'UN1', 'E1', 'backdoors', 'seed*_client_*_backdoor.npy'))
        print(f'[check] {split}: backdoor files {sorted(os.path.basename(f) for f in files)}', flush=True)


def run_phase(jobs, gpus, per_gpu):
    slots = Queue()
    for g in gpus:
        for _ in range(per_gpu):
            slots.put(g)
    failed = []

    def work(job):
        name, cmd = job
        done = os.path.join(LOG_DIR, name + '.done')
        if os.path.exists(done):
            print(f'[skip] {name}', flush=True)
            return
        dev = slots.get()
        try:
            t0 = time.time()
            print(f'[start] {name} on GPU {dev}', flush=True)
            with open(os.path.join(LOG_DIR, name + '.log'), 'w') as f:
                rc = subprocess.run(cmd.format(dev=dev), shell=True, stdout=f, stderr=subprocess.STDOUT).returncode
            dt = (time.time() - t0) / 60
            if rc == 0:
                open(done, 'w').write(f'{dt:.1f} min\n')
                print(f'[ok]    {name} ({dt:.1f} min)', flush=True)
            else:
                failed.append(name)
                print(f'[FAIL]  {name} rc={rc} -> see {LOG_DIR}/{name}.log', flush=True)
        finally:
            slots.put(dev)

    with ThreadPoolExecutor(max_workers=len(gpus) * per_gpu) as ex:
        list(ex.map(work, jobs))
    return failed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gpus', type=int, nargs='+', default=[0])
    ap.add_argument('--per_gpu', type=int, default=2, help='concurrent jobs per GPU (small CNN)')
    ap.add_argument('--seeds', type=int, nargs='+', default=TUNING_SEEDS)
    ap.add_argument('--dry_run', action='store_true')
    a = ap.parse_args()
    bad = FORBIDDEN_SEEDS & set(a.seeds)
    if bad:
        sys.exit(f'Refusing: seeds {sorted(bad)} are reserved (confirmatory/final). Tuning seeds only.')
    os.makedirs(LOG_DIR, exist_ok=True)
    if not a.dry_run:
        print('\n=== phase 0: build data pools ===', flush=True)
        build_pools()
    phases = build(a.seeds)
    for ph, jobs in phases:
        print(f'\n=== phase {ph}: {len(jobs)} jobs ===', flush=True)
        if a.dry_run:
            for n, c in jobs:
                print(f'{n}\n    {c.format(dev="<gpu>")}')
            continue
        failed = run_phase(jobs, a.gpus, a.per_gpu)
        if failed:
            sys.exit(f'Phase {ph} had failures: {failed}. Fix and re-run (finished jobs are skipped).')
        if ph == '1b':
            check_backdoor_clients()
            gate_fail = []
            for s in a.seeds:
                for sp in SPLITS:
                    rc = subprocess.run([sys.executable, 'export_bundle.py', '--gate', '--seed', str(s), '--split', sp],
                                        capture_output=True, text=True)
                    print([l for l in rc.stdout.splitlines() if l.startswith('[GATE]')] or rc.stdout[-500:] + rc.stderr[-500:], flush=True)
                    if rc.returncode != 0:
                        gate_fail.append(f'{sp}/s{s}')
            if gate_fail:
                sys.exit(f'Backdoor gate FAILED for {gate_fail}: the original model did not learn the trigger. '
                         f'No unlearning runs started.')
    print('\nAll phases complete.', flush=True)


if __name__ == '__main__':
    main()
