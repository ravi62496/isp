"""Membership-inference audit: loss/confidence features + a logistic-regression attacker.

Members = client-0 training images, non-members = client-0 held-out images. Both are
subsampled to equal size and equal class counts with a fixed seed. The attacker is
fitted on the attack-train half only and scored on the other half.
"""
from __future__ import annotations

from typing import Iterable

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

FEATURES = ("loss", "max_softmax", "entropy", "true_class_margin")


@torch.no_grad()
def sample_features(model: torch.nn.Module, X: torch.Tensor, y: torch.Tensor,
                    device: torch.device, batch_size: int = 256) -> np.ndarray:
    """Per-sample [CE loss, max softmax, entropy, true-logit minus best other logit]."""
    model.eval()
    out = []
    for i in range(0, len(X), batch_size):
        xb, yb = X[i:i + batch_size].to(device), y[i:i + batch_size].to(device)
        logits = model(xb).float()
        logp = F.log_softmax(logits, dim=1)
        p = logp.exp()
        loss = F.nll_loss(logp, yb, reduction="none")
        entropy = -(p * logp).sum(1)
        true = logits.gather(1, yb[:, None]).squeeze(1)
        other = logits.masked_fill(F.one_hot(yb, logits.shape[1]).bool(), float("-inf")).max(1).values
        out.append(torch.stack([loss, p.max(1).values, entropy, true - other], 1).cpu())
    feats = torch.cat(out).numpy().astype(np.float64)
    if not np.isfinite(feats).all():
        raise ValueError("non-finite MIA features (check the model outputs)")
    return feats


def balanced_indices(y_mem: np.ndarray, y_non: np.ndarray, seed: int = 0,
                     min_per_class: int = 2) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """Equal count per class in both groups (k_c = min of the two counts).

    Classes with fewer than `min_per_class` samples in either group are dropped so the
    stratified split below is possible. Depends only on labels, so every model gets the
    same subsample.
    """
    rng = np.random.RandomState(seed)
    mem, non, used = [], [], []
    for c in np.union1d(np.unique(y_mem), np.unique(y_non)):
        a, b = np.flatnonzero(y_mem == c), np.flatnonzero(y_non == c)
        k = min(len(a), len(b))
        if k < min_per_class:
            continue
        mem.append(rng.choice(a, k, replace=False))
        non.append(rng.choice(b, k, replace=False))
        used.append(int(c))
    if not used:
        raise ValueError("no class has enough samples in both member and non-member sets")
    return np.concatenate(mem), np.concatenate(non), used


def attack_from_features(feats: np.ndarray, is_member: np.ndarray, strat: np.ndarray,
                         seeds: Iterable[int]) -> dict[str, float]:
    """Stratified 50/50 split per seed; scaler + logistic regression fit on train half only."""
    aucs, accs, idx = [], [], np.arange(len(feats))
    for s in seeds:
        tr, te = train_test_split(idx, test_size=0.5, stratify=strat, random_state=s)
        clf = Pipeline([("scale", StandardScaler()),
                        ("lr", LogisticRegression(max_iter=1000))])
        clf.fit(feats[tr], is_member[tr])  # evaluation half is never touched here
        prob = clf.predict_proba(feats[te])[:, 1]
        aucs.append(roc_auc_score(is_member[te], prob))
        accs.append(accuracy_score(is_member[te], prob >= 0.5))
    return {"attack_auc_mean": float(np.mean(aucs)), "attack_auc_std": float(np.std(aucs)),
            "attack_acc_mean": float(np.mean(accs)), "attack_acc_std": float(np.std(accs)),
            "n_splits": len(aucs)}


def run_mia(model: torch.nn.Module, member: tuple[torch.Tensor, torch.Tensor],
            nonmember: tuple[torch.Tensor, torch.Tensor], device: torch.device,
            seeds: Iterable[int] = range(5), subsample_seed: int = 0,
            batch_size: int = 256) -> dict[str, float]:
    """Full MIA report for one model."""
    (Xm, ym), (Xn, yn) = member, nonmember
    im, inn, used = balanced_indices(ym.cpu().numpy(), yn.cpu().numpy(), subsample_seed)
    X = torch.cat([Xm[im], Xn[inn]])
    y = torch.cat([ym[im], yn[inn]])
    feats = sample_features(model, X, y, device, batch_size)
    is_member = np.r_[np.ones(len(im), int), np.zeros(len(inn), int)]
    strat = is_member * (int(y.max()) + 1) + y.cpu().numpy()
    res = attack_from_features(feats, is_member, strat, seeds)
    # Untrained reference: lower loss => "member", AUC over the whole balanced set.
    res["threshold_auc_loss"] = float(roc_auc_score(is_member, -feats[:, 0]))
    res.update(n_members=int(len(im)), n_nonmembers=int(len(inn)), n_classes_used=len(used))
    return res
