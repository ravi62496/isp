"""Run MIA + distance + Grad-CAM audits for every model.

Usage: python -m audit.run_audit --bundle PATH [--out results]   (or --mock)
Writes results/audit.json, results/audit_table.csv, results/audit_gradcam.png.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from audit import distance, gradcam, mia  # noqa: E402
from audit.api import MODEL_NAMES, load_api, pick_device, seed_everything  # noqa: E402

SETS = ("forget_train_clean", "forget_train_trig", "forget_heldout_clean",
        "retained_test", "trigger_test")


def clean(o):
    """Make JSON-safe: numpy -> python, NaN/inf -> null."""
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, (float, np.floating)):
        v = float(o)
        return v if math.isfinite(v) else None
    if isinstance(o, np.integer):
        return int(o)
    return o


def flatten(d: dict, prefix: str = "") -> dict:
    out = {}
    for k, v in d.items():
        out.update(flatten(v, f"{prefix}{k}_") if isinstance(v, dict) else {f"{prefix}{k}": v})
    return out


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bundle", help="bundle folder")
    ap.add_argument("--mock", action="store_true", help="use bundle_api_mock (smoke test)")
    ap.add_argument("--out", default="results", help="output folder (default: ./results)")
    ap.add_argument("--models", nargs="+", default=list(MODEL_NAMES))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-splits", type=int, default=5, help="MIA attack split seeds")
    ap.add_argument("--n-cam", type=int, default=200, help="images per Grad-CAM group")
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--device", default="auto")
    a = ap.parse_args(argv)

    seed_everything(a.seed)
    api, device, out = load_api(a.bundle, a.mock), pick_device(a.device), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    sets = {n: api.get_set(n) for n in SETS}
    models, skipped = {}, {}
    for n in a.models:
        try:
            models[n] = api.load_model(n).to(device).eval()
        except Exception as e:  # keep going; recorded in meta
            skipped[n] = repr(e)
            print(f"[warn] could not load {n}: {e}", file=sys.stderr)
    if "retrain" not in models:
        raise SystemExit("the 'retrain' model is required as the reference")
    ref, mask = models["retrain"], api.trigger_mask().cpu().numpy().astype(bool)
    per = {m: {} for m in models}

    for m, net in models.items():
        print(f"[audit] MIA + distance: {m}", file=sys.stderr)
        per[m]["mia"] = mia.run_mia(net, sets["forget_train_clean"], sets["forget_heldout_clean"],
                                    device, range(a.n_splits), a.seed, a.batch_size)
        per[m]["distance"] = distance.distance_report(net, ref, sets["retained_test"][0], device)
    for g in gradcam.CAM_GROUPS:
        print(f"[audit] Grad-CAM group: {g}", file=sys.stderr)
        X, y = sets[g]
        idx = gradcam.subsample(len(X), a.n_cam, a.seed)
        cams = {m: gradcam.compute_cams(api, net, X[idx], y[idx], device) for m, net in models.items()}
        for m, res in gradcam.cam_metrics(cams, mask).items():
            per[m].setdefault("gradcam", {})[g] = res
    fig = gradcam.render_figure(api, models, sets, device, out / "audit_gradcam.png", seed=a.seed)

    meta = {"bundle": a.bundle, "mock": a.mock, "device": str(device), "seed": a.seed,
            "mia_split_seeds": list(range(a.n_splits)), "n_cam_images_per_group": a.n_cam,
            "reference_model": "retrain", "skipped_models": skipped, "figure": fig,
            "trigger_mask_area_fraction": float(mask.mean()),
            "cam_target": "label stored with each set (same for every model)"}
    (out / "audit.json").write_text(json.dumps(clean({"meta": meta, "models": per}), indent=2))
    rows = [{"model": m, **flatten(v)} for m, v in clean(per).items()]
    cols = list(dict.fromkeys(k for r in rows for k in r))
    with open(out / "audit_table.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    print(f"[audit] wrote {out / 'audit.json'}, {out / 'audit_table.csv'}", file=sys.stderr)


if __name__ == "__main__":
    main()
