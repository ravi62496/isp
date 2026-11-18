"""Distance of a model to the retrained reference model."""
from __future__ import annotations

import torch
import torch.nn.functional as F


@torch.no_grad()
def l2_param_distance(model: torch.nn.Module, ref: torch.nn.Module) -> float:
    """sqrt(sum of squared differences) over named parameters (buffers excluded)."""
    a, b = dict(model.named_parameters()), dict(ref.named_parameters())
    if a.keys() != b.keys():
        raise ValueError("models have different parameter names")
    total = 0.0
    for k, p in a.items():
        if p.shape != b[k].shape:
            raise ValueError(f"shape mismatch for {k}")
        total += float(((p.double().cpu() - b[k].double().cpu()) ** 2).sum())
    return total ** 0.5


@torch.no_grad()
def mean_kl(ref: torch.nn.Module, model: torch.nn.Module, X: torch.Tensor,
            device: torch.device, batch_size: int = 256) -> float:
    """Mean over samples of KL(softmax(ref) || softmax(model))."""
    ref.eval()
    model.eval()
    total = 0.0
    for i in range(0, len(X), batch_size):
        xb = X[i:i + batch_size].to(device)
        logp_ref = F.log_softmax(ref(xb).double(), dim=1)
        logp_mod = F.log_softmax(model(xb).double(), dim=1)
        total += float((logp_ref.exp() * (logp_ref - logp_mod)).sum().cpu())
    return total / len(X)


def distance_report(model: torch.nn.Module, ref: torch.nn.Module, X_retained: torch.Tensor,
                    device: torch.device) -> dict[str, float]:
    return {"l2_param_to_retrain": l2_param_distance(model, ref),
            "mean_kl_retrain_to_model": mean_kl(ref, model, X_retained, device)}
