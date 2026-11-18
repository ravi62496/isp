"""Mock of bundle_api.py for smoke tests: random tensors + a tiny CNN.

Every number it produces is meaningless. Use it only to check that scripts run.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

MODEL_NAMES = ("original", "retrain", "fedosd_hard", "soft_fedosd", "plain_ga",
               "posttrain_plain", "posttrain_proj")
_SIZES = {"forget_train_clean": 240, "forget_train_trig": 240,
          "forget_heldout_clean": 160, "retained_test": 400, "trigger_test": 120}
_SEEDS = {"forget_train_clean": 1, "forget_train_trig": 1,  # trig = clean + patch
          "forget_heldout_clean": 2, "retained_test": 3, "trigger_test": 4}
_BACKDOOR_TARGET = 0


class TinyCNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 8, 3, padding=1)
        self.conv2 = nn.Conv2d(8, 16, 3, padding=1)
        self.fc = nn.Linear(16, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        return self.fc(x.mean(dim=(2, 3)))


def load_model(name: str) -> nn.Module:
    if name not in MODEL_NAMES:
        raise KeyError(name)
    with torch.random.fork_rng():
        torch.manual_seed(1000 + MODEL_NAMES.index(name))
        model = TinyCNN()
    return model.eval()


def get_set(name: str) -> tuple[torch.Tensor, torch.Tensor]:
    if name not in _SIZES:
        raise KeyError(name)
    g = torch.Generator().manual_seed(_SEEDS[name])
    n = _SIZES[name]
    X = torch.randn(n, 3, 32, 32, generator=g)
    y = torch.randint(0, 10, (n,), generator=g)
    if name in ("forget_train_trig", "trigger_test"):
        X[:, :, 28:, 28:] = 3.0
        y = torch.full((n,), _BACKDOOR_TARGET, dtype=torch.long)
    return X, y


def trigger_mask() -> torch.Tensor:
    m = torch.zeros(32, 32, dtype=torch.bool)
    m[28:, 28:] = True
    return m


def gradcam_layer(model: nn.Module) -> nn.Module:
    return model.conv2


def denormalize(X: torch.Tensor) -> torch.Tensor:
    return (X * 0.25 + 0.5).clamp(0, 1)
