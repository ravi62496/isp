"""
Relearning test (protocol fixed before any relearning result was seen, 2026-11-19).

Question: after unlearning, is the backdoor gone, or only suppressed and quick to come back?
An attacker fine-tunes each model on a SMALL subset of the forget client's triggered training
images (backdoor labels) and we track how fast the backdoor returns. A model that truly forgot
should relearn no faster than the retrain gold standard.

Protocol (identical for every model and seed):
  data      N_RELEARN=32 images drawn from forget_train_trig with np.random.default_rng(bundle seed)
  optimiser plain SGD, lr 0.01, full batch of the 32 images, cross-entropy, model.train()
  steps     0..100; evaluate at every step (the full curve is the primary result)
  metrics   asr        = triggered retained TEST images predicted as the target (trigger_test)
            forget_acc = forget client's clean training images, true labels (forget_train_clean)
            clean_acc  = retained TEST accuracy (retained_test)
            mia_auc    = audit MIA attack AUC at steps 0, 50, 100 (None elsewhere)
  summary   first step at which asr >= 50% (steps_to_50; None if never), asr at steps 10 and 100

Usage (from audit_code/):
  python -m audit.relearn --bundle ../bundles/seed31 --out ../results_final/seed31
  python -m audit.relearn --summarize ../results_final --seeds 31 32 33
Writes <out>/relearn.json (rows {round, model, asr, forget_acc, clean_acc, mia_auc}, the format
demo/app.py reads) and <out>/relearn_summary.json.
"""
import argparse
import copy
import json
import os

import numpy as np
import torch
import torch.nn.functional as F

from audit import mia
from audit.api import MODEL_NAMES, load_api, pick_device, seed_everything

N_RELEARN, LR, STEPS, MIA_AT = 32, 0.01, 100, (0, 50, 100)
TAG, FAST, ONLY = '', False, None  # set from CLI (calibration: see FedOSD/PREREG_RELEARN.md)


@torch.no_grad()
def acc(net, X, y, device, bs=1000):
    net.eval()
    pred = torch.cat([net(X[i:i + bs].to(device)).argmax(1).cpu() for i in range(0, len(X), bs)])
    return 100.0 * float((pred == y).float().mean())


def relearn_one(net, sets, Xr, yr, device):
    net = copy.deepcopy(net).to(device)
    opt = torch.optim.SGD(net.parameters(), lr=LR)
    rows = []
    for step in range(STEPS + 1):
        if step > 0:
            net.train()
            opt.zero_grad()
            F.cross_entropy(net(Xr), yr).backward()
            opt.step()
        r = {'round': step, 'asr': acc(net, *sets['trigger_test'], device), 'mia_auc': None}
        if FAST:
            rows.append(r)
            continue
        r['forget_acc'] = acc(net, *sets['forget_train_clean'], device)
        r['clean_acc'] = acc(net, *sets['retained_test'], device)
        if step in MIA_AT:
            net.eval()
            r['mia_auc'] = mia.run_mia(net, sets['forget_train_clean'], sets['forget_heldout_clean'],
                                       device)['attack_auc_mean']
        rows.append(r)
    return rows


def summarize_rows(rows):
    out = {}
    for m in dict.fromkeys(r['model'] for r in rows):
        a = {r['round']: r['asr'] for r in rows if r['model'] == m}
        hit = [s for s in sorted(a) if a[s] >= 50.0]
        out[m] = {'steps_to_50': hit[0] if hit else None, 'asr_0': a[0], 'asr_10': a.get(10), 'asr_end': a.get(STEPS)}
    return out


def run(bundle, out, device_name):
    api = load_api(bundle)
    meta = json.load(open(os.path.join(bundle, 'meta.json')))
    seed_everything(int(meta['seed']))
    device = pick_device(device_name)
    sets = {k: api.get_set(k) for k in ('trigger_test', 'forget_train_clean', 'forget_heldout_clean',
                                        'retained_test', 'forget_train_trig')}
    Xt, yt = sets['forget_train_trig']
    idx = np.random.default_rng(int(meta['seed'])).choice(len(Xt), N_RELEARN, replace=False)
    Xr, yr = Xt[idx].to(device), yt[idx].to(device)
    rows = []
    for m in MODEL_NAMES:
        if m not in meta['models'] or (ONLY and m not in ONLY):
            continue
        print(f'[relearn] {m}', flush=True)
        torch.manual_seed(0)
        rows += [{'model': m, **r} for r in relearn_one(api.load_model(m), sets, Xr, yr, device)]
    os.makedirs(out, exist_ok=True)
    json.dump(rows, open(os.path.join(out, f'relearn{TAG}.json'), 'w'), indent=1)
    s = summarize_rows(rows)
    json.dump({'protocol': {'n': N_RELEARN, 'lr': LR, 'steps': STEPS}, 'per_model': s},
              open(os.path.join(out, f'relearn_summary{TAG}.json'), 'w'), indent=2)
    for m, v in s.items():
        print(f'{m:16} ASR step0 {v["asr_0"]:6.2f}  step10 {v["asr_10"]:6.2f}  step{STEPS} {v["asr_end"]:6.2f}  '
              f'steps to 50%: {v["steps_to_50"]}')


def summarize(root, seeds):
    per = {}
    for s in seeds:
        for m, v in json.load(open(os.path.join(root, f'seed{s}', f'relearn_summary{TAG}.json')))['per_model'].items():
            per.setdefault(m, []).append(v)
    print(f'{"model":16} {"ASR@10":>16} {"ASR@100":>16}  steps_to_50 per seed')
    res = {}
    for m, vs in per.items():
        a5, a30 = [v['asr_10'] for v in vs], [v['asr_end'] for v in vs]
        res[m] = {'asr_10': [float(np.mean(a5)), float(np.std(a5))], 'asr_end': [float(np.mean(a30)), float(np.std(a30))],
                  'steps_to_50': [v['steps_to_50'] for v in vs]}
        print(f'{m:16} {np.mean(a5):7.2f} +/- {np.std(a5):5.2f} {np.mean(a30):7.2f} +/- {np.std(a30):5.2f}  {res[m]["steps_to_50"]}')
    json.dump({'seeds': seeds, 'per_model': res}, open(os.path.join(root, f'relearn_overall{TAG}.json'), 'w'), indent=2)


def main():
    global N_RELEARN, LR, TAG, FAST, ONLY
    ap = argparse.ArgumentParser()
    ap.add_argument('--bundle')
    ap.add_argument('--out')
    ap.add_argument('--device', default='auto')
    ap.add_argument('--summarize', help='results root containing seed<s>/relearn_summary.json')
    ap.add_argument('--seeds', type=int, nargs='+', default=[31, 32, 33])
    ap.add_argument('--n', type=int, default=N_RELEARN)
    ap.add_argument('--lr', type=float, default=LR)
    ap.add_argument('--tag', default='', help='suffix for output files, e.g. _n8_lr0.003')
    ap.add_argument('--models', nargs='+', default=None)
    ap.add_argument('--fast', action='store_true', help='ASR only (calibration)')
    a = ap.parse_args()
    N_RELEARN, LR, TAG, FAST, ONLY = a.n, a.lr, a.tag, a.fast, a.models
    if a.summarize:
        summarize(a.summarize, a.seeds)
    else:
        run(a.bundle, a.out, a.device)


if __name__ == '__main__':
    main()
