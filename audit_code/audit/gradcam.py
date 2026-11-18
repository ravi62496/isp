"""Grad-CAM audit (pytorch-grad-cam): trigger-region energy, similarity to retrain, figure.

Target class: the label stored with each set (true label for clean sets, backdoor
target for triggered sets). It is identical for every model, and so is the layer
(`api.gradcam_layer`). The library min-max normalises each CAM and resizes it
bilinearly to the input size (32x32), so energy fractions refer to normalised maps.
"""
from __future__ import annotations

from pathlib import Path
from types import ModuleType

import numpy as np
import torch
import torch.nn.functional as F

CAM_GROUPS = ("forget_train_trig", "trigger_test", "retained_test", "forget_heldout_clean")
FIGURE_GROUPS = ("forget_train_trig", "retained_test", "forget_heldout_clean")
FIGURE_MODELS = ("original", "soft_fedosd", "fedosd_hard", "retrain")
TITLES = {"original": "Original", "soft_fedosd": "Soft FedOSD",
          "fedosd_hard": "FedOSD hard", "retrain": "Retrain"}
GROUP_SHORT = {"forget_train_trig": "forget+trigger", "retained_test": "retained test",
               "forget_heldout_clean": "forget held-out"}


def compute_cams(api: ModuleType, model: torch.nn.Module, X: torch.Tensor, y: torch.Tensor,
                 device: torch.device, batch_size: int = 64) -> np.ndarray:
    """CAMs [N,32,32] in [0,1] for target class y."""
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

    model.eval()
    cam = GradCAM(model=model, target_layers=[api.gradcam_layer(model)])
    out = []
    try:
        for i in range(0, len(X), batch_size):
            targets = [ClassifierOutputTarget(int(c)) for c in y[i:i + batch_size]]
            out.append(cam(input_tensor=X[i:i + batch_size].to(device), targets=targets))
    finally:
        cam.activations_and_grads.release()
    return np.concatenate(out).astype(np.float32)


def energy_in_mask(cams: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Per-image fraction of CAM mass inside the trigger region."""
    total = cams.sum((1, 2))
    return np.where(total > 1e-12, (cams * mask).sum((1, 2)) / np.maximum(total, 1e-12), np.nan)


def ssim_per_image(a: np.ndarray, b: np.ndarray, data_range: float = 1.0) -> np.ndarray:
    """SSIM, 11x11 Gaussian window (sigma 1.5), valid region, for [N,H,W] arrays."""
    x, y = (torch.from_numpy(v)[:, None].double() for v in (a, b))
    ax = torch.arange(11, dtype=torch.float64) - 5
    g = torch.exp(-(ax ** 2) / (2 * 1.5 ** 2))
    w = ((g / g.sum())[:, None] * (g / g.sum())[None, :])[None, None]
    mx, my = F.conv2d(x, w), F.conv2d(y, w)
    sxx, syy = F.conv2d(x * x, w) - mx ** 2, F.conv2d(y * y, w) - my ** 2
    sxy = F.conv2d(x * y, w) - mx * my
    c1, c2 = (0.01 * data_range) ** 2, (0.03 * data_range) ** 2
    s = ((2 * mx * my + c1) * (2 * sxy + c2)) / ((mx ** 2 + my ** 2 + c1) * (sxx + syy + c2))
    return s.mean((1, 2, 3)).numpy()


def pearson_per_image(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pearson r per image; NaN where either CAM is constant."""
    a = a.reshape(len(a), -1).astype(np.float64)
    b = b.reshape(len(b), -1).astype(np.float64)
    a, b = a - a.mean(1, keepdims=True), b - b.mean(1, keepdims=True)
    den = np.sqrt((a ** 2).sum(1) * (b ** 2).sum(1))
    return np.where(den > 1e-12, (a * b).sum(1) / np.maximum(den, 1e-12), np.nan)


def _ms(v: np.ndarray) -> tuple[float, float]:
    return (float(np.nanmean(v)), float(np.nanstd(v))) if np.isfinite(v).any() else (float("nan"),) * 2


def cam_metrics(cams: dict[str, np.ndarray], mask: np.ndarray, ref: str = "retrain") -> dict:
    """Per model: trigger-region energy, SSIM and Pearson vs the `ref` model's CAMs."""
    out = {}
    for name, c in cams.items():
        e, s, r = energy_in_mask(c, mask), ssim_per_image(c, cams[ref]), pearson_per_image(c, cams[ref])
        em, es = _ms(e)
        sm, ss = _ms(s)
        rm, rs = _ms(r)
        out[name] = {"energy_in_trigger_mean": em, "energy_in_trigger_std": es,
                     "ssim_vs_retrain_mean": sm, "ssim_vs_retrain_std": ss,
                     "pearson_vs_retrain_mean": rm, "pearson_vs_retrain_std": rs,
                     "pearson_undefined_frac": float(np.isnan(r).mean()), "n_images": len(c)}
    return out


def subsample(n_total: int, n: int, seed: int) -> np.ndarray:
    if n >= n_total:
        return np.arange(n_total)
    return np.sort(np.random.RandomState(seed).choice(n_total, n, replace=False))


def pick_indices(labels: np.ndarray, k: int, seed: int = 0) -> np.ndarray:
    """k fixed images, preferring distinct classes first (deterministic)."""
    seen, first, rest = set(), [], []
    for i in np.random.RandomState(seed).permutation(len(labels)):
        (rest if labels[i] in seen else first).append(int(i))
        seen.add(labels[i])
    return np.array((first + rest)[:k])


def to_hwc(api: ModuleType, X: torch.Tensor) -> np.ndarray:
    """api.denormalize (assumed NCHW, like the inputs) -> float [N,H,W,3] in [0,1]."""
    d = torch.as_tensor(api.denormalize(X)).float().clamp(0, 1)
    return d.permute(0, 2, 3, 1).cpu().numpy()


def overlay_rgb(img01: np.ndarray, cam: np.ndarray, size: int = 64, alpha: float = 0.45) -> np.ndarray:
    """Smoothly upsample image and CAM to size x size and blend a jet heatmap."""
    import matplotlib
    img = torch.from_numpy(np.ascontiguousarray(img01)).permute(2, 0, 1)[None].float()
    c = torch.from_numpy(np.ascontiguousarray(cam))[None, None].float()
    img = F.interpolate(img, (size, size), mode="bicubic", align_corners=False).clamp(0, 1)
    c = F.interpolate(c, (size, size), mode="bicubic", align_corners=False).clamp(0, 1)
    heat = matplotlib.colormaps["jet"](c[0, 0].numpy())[..., :3]
    out = (1 - alpha) * img[0].permute(1, 2, 0).numpy() + alpha * heat
    return (out.clip(0, 1) * 255).astype(np.uint8)


def render_figure(api: ModuleType, models: dict[str, torch.nn.Module],
                  sets: dict[str, tuple[torch.Tensor, torch.Tensor]], device: torch.device,
                  path: Path, n_per_group: int = 6, seed: int = 0, dpi: int = 200) -> dict:
    """Rows = fixed images per group; columns = FIGURE_MODELS. Saves a PNG."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    names = [m for m in FIGURE_MODELS if m in models]
    mask = api.trigger_mask().cpu().numpy().astype(bool)
    ys_, xs_ = np.where(mask)
    scale, rows, chosen = 64 / mask.shape[0], [], {}
    for g in FIGURE_GROUPS:
        X, y = sets[g]
        idx = pick_indices(y.cpu().numpy(), n_per_group, seed)
        chosen[g] = idx.tolist()
        Xs, ys = X[idx], y[idx]
        cams = {m: compute_cams(api, models[m], Xs, ys, device) for m in names}
        imgs = to_hwc(api, Xs)
        rows += [(g, int(ys[k]), imgs[k], {m: cams[m][k] for m in names}) for k in range(len(idx))]
    fig, axes = plt.subplots(len(rows), len(names), squeeze=False,
                             figsize=(1.6 * len(names) + 1.4, 1.6 * len(rows) + 0.8))
    for r, (g, label, img, cams) in enumerate(rows):
        for c, m in enumerate(names):
            ax = axes[r, c]
            ax.imshow(overlay_rgb(img, cams[m]))
            ax.set_xticks([])
            ax.set_yticks([])
            if g == "forget_train_trig" and len(xs_):
                ax.add_patch(Rectangle((xs_.min() * scale - .5, ys_.min() * scale - .5),
                                       (xs_.max() - xs_.min() + 1) * scale,
                                       (ys_.max() - ys_.min() + 1) * scale,
                                       fill=False, ec="red", lw=0.8))
            if r == 0:
                ax.set_title(TITLES.get(m, m), fontsize=9)
            if c == 0:
                ax.set_ylabel(f"{GROUP_SHORT[g]}\nlabel {label}", fontsize=7)
    fig.suptitle("Grad-CAM, target = set label (red box = trigger region)", fontsize=10)
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return {"path": Path(path).name, "models": names, "indices": chosen}
