"""
Final report tables over seeds (mean +/- std). Run ONCE, after the configuration is frozen.

Per model and seed:
  clean_acc        accuracy on retained clients' TEST halves (bundle retained_test)
  asr              triggered retained TEST images predicted as the target (bundle trigger_test)
  forget_trig_acc  forget client's own triggered training images -> backdoor label (memorisation)
  forget_clean_acc forget client's clean training images, true labels
  heldout_acc      forget client's held-out TEST half, true labels
  floor_test       worst retained client's final-round TEST accuracy (from the run's metric log)
plus every numeric column of results_final/seed<s>/audit_table.csv (MIA, distance, Grad-CAM).

Usage (from audit_code/):
  python -m audit.aggregate --seeds 31 32 33 --bundles ../bundles --audits ../results_final --fedosd ../FedOSD
"""
import argparse
import csv
import json
import os
from collections import defaultdict

import numpy as np
import torch

from audit.api import MODEL_LABELS, MODEL_NAMES, load_api, pick_device


@torch.no_grad()
def acc(net, X, y, device, bs=1000):
    pred = torch.cat([net(X[i:i + bs].to(device)).argmax(1).cpu() for i in range(0, len(X), bs)])
    return 100.0 * float((pred == y).float().mean())


def floor_from_log(pth, fedosd, forget):
    js = os.path.join(fedosd, os.path.splitext(pth)[0] + '.json')
    if not os.path.exists(js):
        return float('nan')
    ch = json.load(open(js))['metric_log']['client_metric_history']
    vals = [h['test_accuracy'][-1] for cid, h in enumerate(ch)
            if h.get('test_accuracy') and not (len(ch) == 10 and cid == forget)]
    return float(min(vals)) if vals else float('nan')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', type=int, nargs='+', default=[31, 32, 33])
    ap.add_argument('--bundles', default='../bundles')
    ap.add_argument('--audits', default='../results_final')
    ap.add_argument('--fedosd', default='../FedOSD')
    ap.add_argument('--device', default='auto')
    ap.add_argument('--ablation', default=None, help='"lam:eta" of the adaptive run, e.g. 100.0:1.0')
    a = ap.parse_args()
    device = pick_device(a.device)
    vals = defaultdict(lambda: defaultdict(list))  # model -> metric -> [per seed]
    for s in a.seeds:
        b = os.path.join(a.bundles, f'seed{s}')
        api, meta = load_api(b), json.load(open(os.path.join(b, 'meta.json')))
        sets = {k: api.get_set(k) for k in ('retained_test', 'trigger_test', 'forget_train_trig',
                                            'forget_train_clean', 'forget_heldout_clean')}
        for m in MODEL_NAMES:
            if m not in meta['models']:
                continue
            net = api.load_model(m).to(device).eval()
            v = vals[m]
            v['clean_acc'].append(acc(net, *sets['retained_test'], device))
            v['asr'].append(acc(net, *sets['trigger_test'], device))
            v['forget_trig_acc'].append(acc(net, *sets['forget_train_trig'], device))
            v['forget_clean_acc'].append(acc(net, *sets['forget_train_clean'], device))
            v['heldout_acc'].append(acc(net, *sets['forget_heldout_clean'], device))
            v['floor_test'].append(floor_from_log(meta['models'][m], a.fedosd, meta['forget_client']))
        with open(os.path.join(a.audits, f'seed{s}', 'audit_table.csv')) as f:
            for row in csv.DictReader(f):
                for k, x in row.items():
                    if k == 'model':
                        continue
                    try:
                        vals[row['model']][k].append(float(x))
                    except (TypeError, ValueError):
                        pass

    models = [m for m in MODEL_NAMES if m in vals]
    metrics = list(dict.fromkeys(k for m in models for k in vals[m]))
    ms = lambda xs: (float(np.nanmean(xs)), float(np.nanstd(xs)), len(xs)) if xs else (float('nan'),) * 2 + (0,)
    summary = {m: {k: ms(vals[m][k]) for k in metrics} for m in models}
    json.dump({'seeds': a.seeds, 'summary': summary}, open(os.path.join(a.audits, 'final_summary.json'), 'w'), indent=2)
    with open(os.path.join(a.audits, 'final_summary.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['metric'] + models)
        for k in metrics:
            w.writerow([k] + [f'{summary[m][k][0]:.3f} +/- {summary[m][k][1]:.3f}' for m in models])

    head = ['clean_acc', 'asr', 'forget_trig_acc', 'heldout_acc', 'floor_test']
    lines = [f'Seeds {a.seeds}, mean +/- std. Accuracies in %.', '',
             '| Model | ' + ' | '.join(head) + ' |', '|' + '---|' * (len(head) + 1)]
    for m in models:
        lines.append(f'| {MODEL_LABELS.get(m, m)} | ' + ' | '.join(
            f'{summary[m][k][0]:.2f} +/- {summary[m][k][1]:.2f}' for k in head) + ' |')
    lines += ['', 'All metrics (including MIA / distance / Grad-CAM): final_summary.csv']
    if a.ablation:  # uniform vs adaptive soft FedOSD, from the run logs (TEST halves, final seeds)
        lam, eta = a.ablation.split(':')
        stem = os.path.join(a.fedosd, 'UnlearningTask_patch4', 'CNN_CIFAR10', 'cifar10 pat balanced B200 NC2 N10',
                            'UN1', 'E1', 'C1.0', 'FedOSD', 'seed{s} N10 C1.0 FedOSD CNN_CIFAR10 E1 lr0.0004 '
                            'decay0.999 g_clip-1.0 proj_two_sided wm_{wm} lam{lam} eta{eta} rlr0.0001{suf}.json')
        lines += ['', f'Ablation: soft FedOSD weighting (lambda={lam}), retained TEST accuracy from run logs', '',
                  '| Variant | stage | mean | floor |', '|---|---|---|---|']
        for wm, e in (('uniform', '0.0'), ('deficit', eta)):
            for stage, suf in (('after unlearning', '_unlearn'), ('after post-training', '')):
                mean, flo = [], []
                for s in a.seeds:
                    p = stem.format(s=s, wm=wm, lam=lam, eta=e, suf=suf)
                    if not os.path.exists(p):
                        continue
                    d = json.load(open(p))
                    fc = d['params']['unlearn_client_id_list'][0]
                    v = [h['test_accuracy'][-1] for c, h in enumerate(d['metric_log']['client_metric_history'])
                         if c != fc and h.get('test_accuracy')]
                    mean.append(np.mean(v)); flo.append(np.min(v))
                if mean:
                    lines.append(f'| eta={e} ({wm}) | {stage} | {np.mean(mean):.2f} +/- {np.std(mean):.2f} | '
                                 f'{np.mean(flo):.2f} +/- {np.std(flo):.2f} (n={len(flo)}) |')
    open(os.path.join(a.audits, 'final_table.md'), 'w').write('\n'.join(lines) + '\n')
    print('\n'.join(lines))
    print('\n' + open(os.path.join(a.audits, 'final_summary.csv')).read())


if __name__ == '__main__':
    main()
