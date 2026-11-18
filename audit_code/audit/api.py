"""Glue between the audit scripts and a bundle (or the mock).

A real bundle folder contains bundle_api.py (written by FedOSD/export_bundle.py) plus its
data files. It is loaded by path, so the audit code never imports the training repo.
"""
from __future__ import annotations

import importlib
import importlib.util
import random
from pathlib import Path
from types import ModuleType

import numpy as np
import torch

MODEL_NAMES = ("original", "retrain", "fedosd_hard", "soft_fedosd", "plain_ga",
               "posttrain_plain", "posttrain_proj")
MODEL_LABELS = {"original": "Original (before forgetting)", "retrain": "Retrain (gold standard)",
                "fedosd_hard": "FedOSD (hard)", "soft_fedosd": "Soft FedOSD (ours)",
                "plain_ga": "Plain gradient ascent", "posttrain_plain": "Post-training, no projection",
                "posttrain_proj": "Post-training, projected"}


def load_api(bundle: str | None, mock: bool = False) -> ModuleType:
    if mock:
        return importlib.import_module("audit.bundle_api_mock")
    if not bundle:
        raise SystemExit("pass --bundle PATH (or --mock)")
    path = Path(bundle) / "bundle_api.py"
    if not path.is_file():
        raise SystemExit(f"{path} not found")
    spec = importlib.util.spec_from_file_location("bundle_api", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def pick_device(name: str = "auto") -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
