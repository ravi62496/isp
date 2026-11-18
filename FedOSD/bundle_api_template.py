"""bundle_api.py - standalone loader for an audit bundle (copied in by export_bundle.py).

No imports from the training repo: the CNN is redefined here with identical layer names.
"""
from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn as nn

HERE = Path(__file__).resolve().parent
META = json.loads((HERE / 'meta.json').read_text())


class CNN(nn.Module):
    """Same architecture and parameter names as tfedplat.model.CNN.CNN (CNN_CIFAR10)."""

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.ignore_head = False
        self.encoder = nn.Sequential(nn.Conv2d(3, 64, 5), nn.ReLU(), nn.MaxPool2d(2),
                                     nn.Conv2d(64, 64, 5), nn.ReLU(), nn.MaxPool2d(2))
        self.decoder = nn.Sequential(nn.Linear(1600, 384), nn.Dropout(0.2), nn.ReLU(),
                                     nn.Linear(384, 192), nn.Dropout(0.5), nn.ReLU())
        self.predictor = nn.Linear(192, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.predictor(self.decoder(self.encoder(x).flatten(1)))


def available_models() -> list[str]:
    return sorted(p.stem for p in (HERE / 'models').glob('*.pt'))


def load_model(name: str) -> nn.Module:
    path = HERE / 'models' / f'{name}.pt'
    if not path.is_file():
        raise FileNotFoundError(f'model {name!r} not in bundle (have: {available_models()})')
    m = CNN()
    m.load_state_dict(torch.load(path, map_location='cpu'))
    return m.eval()


def get_set(name: str) -> tuple[torch.Tensor, torch.Tensor]:
    d = torch.load(HERE / 'sets' / f'{name}.pt', map_location='cpu')
    return d['X'], d['y']


def trigger_mask() -> torch.Tensor:
    return torch.load(HERE / 'trigger_mask.pt')


def gradcam_layer(model: nn.Module) -> nn.Module:
    return model.encoder[3]  # second conv, 10x10 map


def denormalize(X: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor(META['norm_mean']).view(1, 3, 1, 1)
    std = torch.tensor(META['norm_std']).view(1, 3, 1, 1)
    return (X * std + mean).clamp(0, 1)
