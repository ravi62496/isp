"""
Export a self-contained audit bundle for one (seed, split): evaluation sets + model weights.

The task object is constructed exactly as the PRETRAIN run constructs it (same argv, same RNG
path), without training. So the forget client, its val/test split and its trigger pattern are
the ones the original model was actually trained with. Nothing here trains or tunes anything.

Usage (from FedOSD/):
  python export_bundle.py --seed 999 --models_root backup_seed999_reference --out ../bundles/seed999
  python export_bundle.py --seed 31 --soft_lam 20 --out ../bundles/seed31

Sets (all normalised exactly as during training):
  forget_train_clean    forget client's training images, clean labels       (MIA members)
  forget_train_trig     same images, trigger on all, backdoor labels (y+5)%10
  forget_heldout_clean  forget client's held-out TEST half, never trained on (MIA non-members)
  retained_test         retained clients' TEST halves (val halves are left out)
  trigger_test          retained_test with trigger + backdoor labels (ASR)
"""
import argparse
import copy
import glob
import json
import os
import shutil
import sys

import torch

SPLIT_ARGS = {
    'pat': ['--dataloader', 'DataLoader_cifar10_pat'],
    'dir0.5': ['--dataloader', 'DataLoader_cifar10_dir', '--Diralpha', '0.5'],
    'dir0.1': ['--dataloader', 'DataLoader_cifar10_dir', '--Diralpha', '0.1'],
}
FEDOSD = 'seed{s} N10 C1.0 FedOSD CNN_CIFAR10 E1 lr0.0004 decay0.999 g_clip-1.0 '
MODEL_FILES = {  # name -> glob pattern relative to <models_root>/CNN_CIFAR10/<nickname>/UN1/E1/
    'original': 'seed{s}_unlearn_task_pretrained_model.pth',
    'retrain': 'C1.0/FedAvg/seed{s} retrain_from_scratch seed{s} N10 C1.0 FedAvg CNN_CIFAR10 E1 lr0.05 decay0.999 g_clip-1.0 R300.pth',
    'fedosd_hard': 'C1.0/FedOSD/' + FEDOSD + 'proj_hard lam100000.0 eta0.0 rlr0.0001_unlearn.pth',
    'posttrain_proj': 'C1.0/FedOSD/' + FEDOSD + 'proj_hard lam100000.0 eta0.0 rlr0.0001.pth',
    'posttrain_plain': 'C1.0/FedOSD/' + FEDOSD + 'proj_hard lam100000.0 eta0.0 rec_plain rlr0.0001.pth',
    'soft_fedosd': 'C1.0/FedOSD/' + FEDOSD + 'proj_two_sided wm_uniform lam{lam} eta0.0 rlr0.0001_unlearn.pth',
    'plain_ga': 'C1.0/FedAvg_GA/seed{s} *_unlearn.pth',
}


def cat(batches):
    return torch.cat([b[0] for b in batches]).float().cpu(), torch.cat([b[1] for b in batches]).long().cpu()


def triggered(batches, backdoor):
    b = copy.deepcopy(batches)
    backdoor.add_backdoor(b, attack_portion=1.0)
    return cat(b)


def build_task(seed, split, backdoor):
    import tfedplat as fp
    sys.argv = ['export', '--seed', str(seed), '--device', '-1', '--module', 'CNN_CIFAR10',
                *SPLIT_ARGS[split], '--N', '10', '--NC', '2', '--balance', 'True', '--B', '200',
                '--C', '1.0', '--E', '1', '--decay', '0.999', '--step_type', 'bgd', '--unlearn_cn', '1',
                '--save_model', 'False', '--algorithm', 'FedAvg', '--R', '100', '--UR', '0',
                '--lr', '0.05', '--unlearn_pretrain', 'True', '--backdoor', backdoor]
    return fp.UnlearningTask()


@torch.no_grad()
def gate(base, seed, fc, bd, ret_batches, ret_batches_t, min_asr):
    """Exit 0 iff the ORIGINAL (pretrained) model learned the trigger."""
    import tfedplat as fp
    from tfedplat.model.CNN import CNN
    path = os.path.join(base, f'seed{seed}_unlearn_task_pretrained_model.pth')
    net = CNN(False, [3, 32, 32], 10)
    net.load_state_dict(torch.load(path, map_location='cpu'))
    net.eval()

    def acc(X, y):
        return 100.0 * float((torch.cat([net(X[i:i + 1000]).argmax(1) for i in range(0, len(X), 1000)]) == y).float().mean())
    asr_test = acc(*triggered(ret_batches_t, bd))
    asr_forget = acc(*triggered(fc.clean_training_data, bd))
    clean = acc(*cat(ret_batches))
    no_trig = acc(cat(ret_batches_t)[0], torch.full((len(cat(ret_batches_t)[1]),), getattr(bd, 'target_label', -1)))
    # The trigger must CAUSE the target prediction: high ASR, low target rate without the trigger,
    # and a model that still classifies clean images (a collapsed model predicts the target always).
    ok = asr_test >= min_asr and asr_test - no_trig >= 50.0 and no_trig <= 20.0 and clean >= 40.0  # Amendment B
    print(f'[GATE] (validation halves) seed {seed} forget client {fc.id} target {getattr(bd, "target_label", None)}: '
          f'ASR(held-out, triggered)={asr_test:.1f}%  ASR(forget train)={asr_forget:.1f}%  '
          f'target-rate without trigger={no_trig:.1f}%  clean retained acc={clean:.1f}%  -> {"PASS" if ok else "FAIL"}')
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed', type=int, required=True)
    ap.add_argument('--split', default='pat', choices=list(SPLIT_ARGS))
    ap.add_argument('--backdoor', default='patch4', choices=['legacy', 'patch4'])
    ap.add_argument('--models_root', default=None, help='default: UnlearningTask_patch4 (or UnlearningTask for legacy)')
    ap.add_argument('--gate', action='store_true', help='only check that the original model learned the backdoor')
    ap.add_argument('--gate_min_asr', type=float, default=60.0)  # Amendment B (was 90)
    ap.add_argument('--soft_lam', default='20.0')
    ap.add_argument('--out', default=None)
    a = ap.parse_args()
    if a.models_root is None:
        a.models_root = 'UnlearningTask' if a.backdoor == 'legacy' else 'UnlearningTask_patch4'
    seed, split, out = a.seed, a.split, a.out

    task = build_task(seed, split, a.backdoor)
    alg = task.algorithm
    forget = [c for c in alg.client_list if c.unlearn_flag]
    assert len(forget) == 1, 'expected exactly one forget client'
    fc = forget[0]
    bd = fc.backdoor_setting
    retained = [c for c in alg.client_list if not c.unlearn_flag]

    ret_batches = [b for c in retained for b in c.local_test_data]
    if hasattr(bd, 'target_label'):  # standard ASR: exclude images whose true class is the target
        ret_batches_t = [[bx[by != bd.target_label], by[by != bd.target_label]] for bx, by in ret_batches]
    else:
        ret_batches_t = ret_batches
    base = os.path.join(a.models_root, 'CNN_CIFAR10', alg.data_loader.nickname, 'UN1', 'E1')
    if a.gate:  # go/no-go decision -> use VALIDATION halves only, never the test halves
        gv = [b for c in retained for b in c.local_val_data]
        gv_t = ([[bx[by != bd.target_label], by[by != bd.target_label]] for bx, by in gv]
                if hasattr(bd, 'target_label') else gv)
        sys.exit(gate(base, seed, fc, bd, gv, gv_t, a.gate_min_asr))
    if not out:
        sys.exit('--out is required unless --gate')
    os.makedirs(os.path.join(out, 'sets'), exist_ok=True)
    os.makedirs(os.path.join(out, 'models'), exist_ok=True)
    sets = {
        'forget_train_clean': cat(fc.clean_training_data),
        'forget_train_trig': triggered(fc.clean_training_data, bd),
        'forget_heldout_clean': cat(fc.clean_test_data),
        'retained_test': cat(ret_batches),
        'trigger_test': triggered(ret_batches_t, bd),
    }
    for k, (X, y) in sets.items():
        torch.save({'X': X, 'y': y}, os.path.join(out, 'sets', k + '.pt'))

    mask = torch.zeros(bd.image_size[-2:], dtype=torch.bool)
    mask[bd.x_top: bd.x_top + bd.x_len if bd.x_top + bd.x_len != 0 else None,
         bd.y_top: bd.y_top + bd.y_len if bd.y_top + bd.y_len != 0 else None] = True
    torch.save(mask, os.path.join(out, 'trigger_mask.pt'))

    found = {}
    for name, pat in MODEL_FILES.items():
        rel = pat.format(s=seed, lam=a.soft_lam)
        hits = sorted(glob.glob(glob.escape(base + os.sep) + '*'.join(glob.escape(p) for p in rel.split('*'))))
        if hits:
            torch.save(torch.load(hits[0], map_location='cpu'), os.path.join(out, 'models', name + '.pt'))
            found[name] = hits[0]
        else:
            print(f'[missing] {name}: no file matching {pat.format(s=seed, lam=a.soft_lam)}')

    meta = {'seed': seed, 'split': split, 'nickname': alg.data_loader.nickname,
            'forget_client': int(fc.id), 'retained_clients': [int(c.id) for c in retained],
            'set_sizes': {k: int(len(v[1])) for k, v in sets.items()},
            'models': found, 'input_shape': list(alg.data_loader.input_data_shape),
            'norm_mean': [0.4914, 0.4822, 0.4465], 'norm_std': [0.2023, 0.1994, 0.2010],
            'trigger_box': [bd.x_top, bd.y_top, bd.x_len, bd.y_len], 'backdoor': a.backdoor,
            'target_label': getattr(bd, 'target_label', None),
            'note': 'retained_test/forget_heldout use TEST halves only; val halves excluded'}
    json.dump(meta, open(os.path.join(out, 'meta.json'), 'w'), indent=2)
    shutil.copy(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'bundle_api_template.py'),
                os.path.join(out, 'bundle_api.py'))
    print(json.dumps({k: meta[k] for k in ('seed', 'split', 'forget_client', 'set_sizes')}, indent=1))
    print('models:', sorted(found))


if __name__ == '__main__':
    main()
