"""Gradio demo. Inference only: reads results/*.json, results/*.png and the bundle.

Usage: python -m demo.app --bundle PATH [--results results]   (or --mock)
"""
from __future__ import annotations

import argparse
import json
import sys
import threading
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from audit import gradcam  # noqa: E402
from audit.api import MODEL_LABELS, MODEL_NAMES, load_api, pick_device, seed_everything  # noqa: E402

# Display names assume the standard CIFAR-10 label order.
CLASSES = ("airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck")
GROUPS = {"Forget client + trigger": "forget_train_trig",
          "Retained clients (test)": "retained_test",
          "Forget client held-out (clean)": "forget_heldout_clean",
          "Trigger test": "trigger_test"}
N_POOL, SIZE = 24, 128


def read_json(path: Path, default):
    return json.loads(path.read_text()) if path.is_file() else default


def flatten(d: dict, prefix: str = "") -> dict:
    out = {}
    for k, v in d.items():
        out.update(flatten(v, f"{prefix}{k}.") if isinstance(v, dict) else {f"{prefix}{k}": v})
    return out


def fmt(v) -> str:
    if v is None:
        return "n/a"
    return f"{v:.4f}" if isinstance(v, float) else str(v)


def cname(k: int) -> str:
    return f"{k} ({CLASSES[k]})" if 0 <= k < len(CLASSES) else str(k)


def relearn_figure(rows: list, highlight: str):
    from matplotlib.figure import Figure
    fig = Figure(figsize=(10, 3))
    axes = fig.subplots(1, 3)
    if not rows:
        fig.text(0.5, 0.5, "results/relearn.json not found or empty", ha="center")
        return fig
    for ax, key in zip(axes, ("asr", "forget_acc", "mia_auc")):
        for m in sorted({r["model"] for r in rows}):
            pts = sorted((r["round"], r[key]) for r in rows if r["model"] == m and r.get(key) is not None)
            if pts:
                ax.plot(*zip(*pts), marker="o", ms=3, lw=3 if m == highlight else 1.2,
                        label=MODEL_LABELS.get(m, m))
        ax.set_title(key)
        ax.set_xlabel("round")
    axes[-1].legend(fontsize=7)
    fig.tight_layout()
    return fig


class Demo:
    def __init__(self, api, results: Path, device: torch.device, seed: int = 0) -> None:
        self.api, self.device, self.seed, self.results = api, device, seed, results
        self.lock = threading.Lock()  # Grad-CAM hooks are not thread-safe
        audit = read_json(results / "audit.json", {})
        self.flat = {m: flatten(v) for m, v in audit.get("models", {}).items()}
        self.relearn = read_json(results / "relearn.json", [])
        self.names = list(self.flat) or list(MODEL_NAMES)
        self._models, self._pools = {}, {}

    def model(self, name: str):
        if name not in self._models:
            self._models[name] = self.api.load_model(name).to(self.device).eval()
        return self._models[name]

    def pool(self, key: str):
        if key not in self._pools:
            X, y = self.api.get_set(key)
            idx = gradcam.pick_indices(y.cpu().numpy(), N_POOL, self.seed)
            true = self.api.get_set('forget_train_clean')[1][idx] if key == 'forget_train_trig' else None
            self._pools[key] = (X[idx], y[idx], true)
        return self._pools[key]

    def render(self, model_name: str, group: str, i: float):
        with self.lock:
            X, y, true = self.pool(GROUPS[group])
            i = int(min(i, len(X) - 1))
            x, t = X[i:i + 1], y[i:i + 1]
            img = gradcam.to_hwc(self.api, x)[0]
            shown = ["original", model_name, "retrain"]
            ims, preds = [], []
            for m in shown:
                net = self.model(m)
                with torch.no_grad():
                    p = torch.softmax(net(x.to(self.device)), 1)[0].cpu()
                cam = gradcam.compute_cams(self.api, net, x, t, self.device)[0]
                ims.append(gradcam.overlay_rgb(img, cam, SIZE))
                k = int(p.argmax())
                preds.append([MODEL_LABELS.get(m, m), cname(k), f"{float(p[k]):.3f}"])
        keys = [k for k in dict.fromkeys(k for m in shown for k in self.flat.get(m, {}))]
        table = [[k] + [fmt(self.flat.get(m, {}).get(k)) for m in shown] for k in keys]
        if GROUPS[group] == 'forget_train_trig':
            label = (f"**True class:** {cname(int(true[i]))}  |  **Backdoor label (Grad-CAM target):** "
                     f"{cname(int(t))}. The patch at the bottom right is the trigger.")
        elif GROUPS[group] == 'trigger_test':
            label = (f"**Backdoor label (Grad-CAM target):** {cname(int(t))}. The true class is a different "
                     "class (images of the target class are excluded).")
        else:
            label = f"**True class / Grad-CAM target:** {cname(int(t))}"
        return ims[0], ims[1], ims[2], label, preds, table, relearn_figure(self.relearn, model_name)


def build_ui(demo: Demo):
    import gradio as gr
    default = "original" if "original" in demo.names else demo.names[0]
    forgotten = "soft_fedosd" if "soft_fedosd" in demo.names else default
    with gr.Blocks(title="Federated unlearning audit") as ui:
        gr.Markdown("# Federated unlearning: what did the model forget?")
        with gr.Row():
            model_dd = gr.Dropdown([(MODEL_LABELS.get(m, m), m) for m in demo.names],
                                   value=default, label="Model")
            group_dd = gr.Dropdown(list(GROUPS), value=list(GROUPS)[0], label="Image group")
            slider = gr.Slider(0, N_POOL - 1, value=0, step=1, label="Image")
        forget = gr.Button("Request to be forgotten", variant="primary")
        status = gr.Markdown()
        with gr.Row():
            im_o = gr.Image(label="Original", height=300)
            im_s = gr.Image(label="Selected model", height=300)
            im_r = gr.Image(label="Retrain (reference)", height=300)
        label_md = gr.Markdown()
        preds = gr.Dataframe(headers=["model", "prediction", "confidence"], interactive=False)
        gr.Markdown("### Audit metrics (from results/audit.json)")
        metrics = gr.Dataframe(headers=["metric", "original", "selected", "retrain"],
                               interactive=False, wrap=True)
        gr.Markdown("### Relearning curve (from results/relearn.json)")
        curve = gr.Plot()
        png = demo.results / "audit_gradcam.png"
        if png.is_file():
            with gr.Accordion("Full Grad-CAM audit figure", open=False):
                gr.Image(str(png), label="results/audit_gradcam.png")
        outs = [im_o, im_s, im_r, label_md, preds, metrics, curve]
        for c in (model_dd, group_dd, slider):
            c.change(demo.render, [model_dd, group_dd, slider], outs)
        fc = getattr(demo.api, "META", {}).get("forget_client", "?")
        forget.click(lambda: (forgotten, f"Forget request for client {fc} applied (precomputed unlearned model): "
                              f"showing **{MODEL_LABELS.get(forgotten, forgotten)}**."),
                     None, [model_dd, status])
        ui.load(demo.render, [model_dd, group_dd, slider], outs)
    return ui


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bundle")
    ap.add_argument("--mock", action="store_true")
    ap.add_argument("--results", default="results")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--share", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    seed_everything(a.seed)
    demo = Demo(load_api(a.bundle, a.mock), Path(a.results), pick_device(a.device), a.seed)
    build_ui(demo).launch(server_port=a.port, share=a.share)


if __name__ == "__main__":
    main()
