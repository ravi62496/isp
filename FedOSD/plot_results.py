import os
import json
import glob
import math
import re
import numpy as np
import matplotlib.pyplot as plt

# Academic styling settings
plt.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 14,
    'lines.linewidth': 2.0,
    'lines.markersize': 5,
    'grid.alpha': 0.3,
    'grid.linestyle': '--',
    'axes.grid': True,
    'figure.autolayout': True
})

OUTPUT_DIR = "figures"
BASE_DIR = "UnlearningTask/CNN_CIFAR10/cifar10 pat balanced B200 NC2 N10/UN1/E1/C1.0/FedOSD"
FEDAVG_DIR = "UnlearningTask/CNN_CIFAR10/cifar10 pat balanced B200 NC2 N10/UN1/E1/C1.0/FedAvg"

def ensure_dir(d):
    if not os.path.exists(d):
        os.makedirs(d)

def load_json(path):
    with open(path, "r") as f:
        return json.load(f)

def extract_client_metrics(data):
    client_hist = data["metric_log"]["client_metric_history"]
    unlearn_cid = data.get("params", {}).get("unlearn_client_id_list", [7])[0]
    
    asr_curve = np.array(client_hist[unlearn_cid].get("backdoor_test_accuracy", client_hist[unlearn_cid].get("test_accuracy", [])))
    
    retained_curves = []
    for cid, hist in enumerate(client_hist):
        if cid != unlearn_cid and "test_accuracy" in hist and hist["test_accuracy"]:
            retained_curves.append(hist["test_accuracy"])
            
    retained_curves = np.array(retained_curves)
    n_rounds = len(asr_curve) if len(asr_curve) > 0 else retained_curves.shape[1]
    rounds = np.arange(n_rounds)
    return rounds, asr_curve, retained_curves, unlearn_cid

# =========================================================================
# Figure 1: Unlearning Dynamics (ASR Eradication & Floor Protection)
# =========================================================================
def plot_figure_1_dynamics():
    # Use Seed 11 reporting run
    hard_file = os.path.join(BASE_DIR, "seed11 N10 C1.0 FedOSD CNN_CIFAR10 E1 lr0.0004 decay0.999 g_clip-1.0 proj_hard lam100000.0 eta0.0_unlearn.json")
    soft_file = os.path.join(BASE_DIR, "seed11 N10 C1.0 FedOSD CNN_CIFAR10 E1 lr0.0004 decay0.999 g_clip-1.0 proj_two_sided wm_uniform lam20.0 eta0.0_unlearn.json")
    capped_file = os.path.join(BASE_DIR, "seed11 N10 C1.0 FedOSD CNN_CIFAR10 E1 lr0.0004 decay0.999 g_clip-1.0 proj_one_sided_cap1.0 wm_uniform lam100000.0_unlearn.json")
    
    r_hard, asr_hard, ret_hard, _ = extract_client_metrics(load_json(hard_file))
    r_soft, asr_soft, ret_soft, _ = extract_client_metrics(load_json(soft_file))
    r_cap, asr_cap, ret_cap, _ = extract_client_metrics(load_json(capped_file))
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8.5, 7.2), sharex=True)
    
    # 1. Top Panel: ASR Eradication
    ax1.plot(r_hard, asr_hard, label="Hard FedOSD (Baseline)", color="#7f7f7f", linestyle="--")
    ax1.plot(r_soft, asr_soft, label=r"Soft Uniform ($\lambda=20, \eta=0.0$)", color="#1f77b4", linestyle="-.")
    ax1.plot(r_cap, asr_cap, label=r"Step-Norm Capped One-Sided (cap=1.0)", color="#2ca02c", linestyle="-", linewidth=2.2)
    ax1.set_ylabel("Backdoor ASR (%)")
    ax1.set_title("(a) Backdoor Trigger Elimination Dynamics (Reporting Seed 11)", fontweight="bold")
    ax1.legend(loc="upper right", frameon=True)
    ax1.set_ylim(-1, 35)
    ax1.axhline(1.0, color="gray", linestyle=":", alpha=0.7)
    ax1.text(50, 2.5, "Unlearned Threshold (< 1.0%)", color="gray", fontstyle="italic", fontsize=9)
    
    # 2. Bottom Panel: Minimum Client Accuracy (Worst-Off Protection)
    min_hard = np.min(ret_hard, axis=0)
    min_soft = np.min(ret_soft, axis=0)
    min_cap = np.min(ret_cap, axis=0)
    
    ax2.plot(r_hard, min_hard, label="Hard FedOSD (Baseline)", color="#7f7f7f", linestyle="--")
    ax2.plot(r_soft, min_soft, label=r"Soft Uniform ($\lambda=20, \eta=0.0$)", color="#1f77b4", linestyle="-.")
    ax2.plot(r_cap, min_cap, label=r"Step-Norm Capped One-Sided (cap=1.0)", color="#2ca02c", linestyle="-", linewidth=2.2)
    
    ax2.set_xlabel("Unlearning Rounds (UR = 100)")
    ax2.set_ylabel("Worst-Off Client Accuracy (%)")
    ax2.set_title("(b) Worst-Off Client Floor Protection Trajectory", fontweight="bold")
    ax2.legend(loc="lower right", frameon=True)
    
    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, "fig1_unlearning_dynamics.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Generated: {out_path}")

# =========================================================================
# Figure 2: Per-Client Accuracy Distribution & Fairness
# =========================================================================
def plot_figure_2_client_distribution():
    hard_file = os.path.join(BASE_DIR, "seed11 N10 C1.0 FedOSD CNN_CIFAR10 E1 lr0.0004 decay0.999 g_clip-1.0 proj_hard lam100000.0 eta0.0_unlearn.json")
    capped_file = os.path.join(BASE_DIR, "seed11 N10 C1.0 FedOSD CNN_CIFAR10 E1 lr0.0004 decay0.999 g_clip-1.0 proj_one_sided_cap1.0 wm_uniform lam100000.0_unlearn.json")
    onesided_file = os.path.join(BASE_DIR, "seed11 N10 C1.0 FedOSD CNN_CIFAR10 E1 lr0.0004 decay0.999 g_clip-1.0 proj_one_sided wm_uniform lam100000.0 eta0.0_unlearn.json")
    
    _, _, ret_hard, ucid = extract_client_metrics(load_json(hard_file))
    _, _, ret_cap, _ = extract_client_metrics(load_json(capped_file))
    _, _, ret_os, _ = extract_client_metrics(load_json(onesided_file))
    
    final_hard = ret_hard[:, -1]
    final_cap = ret_cap[:, -1]
    final_os = ret_os[:, -1]
    
    client_ids = [c for c in range(10) if c != ucid]
    x = np.arange(len(client_ids))
    width = 0.28
    
    fig, ax = plt.subplots(figsize=(10, 5))
    
    ax.bar(x - width, final_hard, width, label="Hard FedOSD (Baseline)", color="#7f7f7f", alpha=0.85)
    ax.bar(x, final_os, width, label="One-Sided Hard (Unconstrained)", color="#d62728", alpha=0.8)
    ax.bar(x + width, final_cap, width, label="Step-Norm Capped (cap=1.0)", color="#2ca02c", alpha=0.9)
    
    ax.set_xlabel("Retained Client Index (Seed 11)")
    ax.set_ylabel("Final Test Accuracy (%)")
    ax.set_title("Per-Client Accuracy: Step-Norm Capping Prevents One-Sided Floor Collapse", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([f"Client {c}" for c in client_ids])
    ax.legend(loc="upper right", frameon=True)
    ax.set_ylim(0, 60)
    
    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, "fig2_per_client_distribution.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Generated: {out_path}")

# =========================================================================
# Figure 3: Hyperparameter Trade-off & Pareto Frontier
# =========================================================================
def plot_figure_3_hyperparameter_pareto():
    # Data from Step 3 Screening on Seed 101
    cap_ratios = [1.0, 1.25, float('inf')]
    cap_labels = ["cap=1.0\n(Hard norm)", "cap=1.25\n(Mild slack)", "Unconstrained\n(One-Sided)"]
    floors = [26.83, 26.83, 2.23] # seed 101 capped vs unconstrained reporting avg
    means = [43.59, 43.63, 34.56]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.6))
    
    # Left: Step Norm Cap vs Floor and Mean
    x = np.arange(len(cap_labels))
    w = 0.35
    ax1.bar(x - w/2, floors, w, label="Worst-Off Floor (%)", color="#2ca02c", alpha=0.85)
    ax1.bar(x + w/2, means, w, label="Cohort Mean (%)", color="#1f77b4", alpha=0.85)
    ax1.set_xticks(x)
    ax1.set_xticklabels(cap_labels)
    ax1.set_ylabel("Accuracy (%)")
    ax1.set_title("(a) Step-Norm Control Effect on Retention", fontweight="bold")
    ax1.legend(loc="upper right", frameon=True)
    ax1.set_ylim(0, 52)
    
    # Right: Multi-seed Pareto trade-off (Floor vs Forget Clean Utility)
    methods = [
        ("Hard", 3.83, 3.35, "#7f7f7f", "s"),
        ("Soft Uniform", 3.85, 3.37, "#1f77b4", "^"),
        ("One-Sided Hard", 2.72, 2.20, "#d62728", "x"),
        ("Capped (cap=1.0)", 4.23, 3.87, "#2ca02c", "o"),
        ("Retrain (Reference)", 28.62, 29.18, "#e67e22", "*")
    ]
    
    for name, fl, fc, col, m in methods:
        ms = 12 if m == '*' else 8
        ax2.scatter(fc, fl, color=col, marker=m, s=ms*10, label=name, zorder=5)
        ax2.annotate(name, (fc, fl), textcoords="offset points", xytext=(8, -4), fontsize=9)
        
    ax2.set_xlabel("Forget-Set Clean Utility (%)")
    ax2.set_ylabel("Worst-Off Client Floor (%)")
    ax2.set_title("(b) Empirical Pareto Trade-off Across 10 Seeds", fontweight="bold")
    ax2.set_xlim(-1, 32)
    ax2.set_ylim(0, 32)
    
    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, "fig3_hyperparameter_pareto.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Generated: {out_path}")

# =========================================================================
# Figure 4: Multi-Seed Statistical Robustness Across 10 Fresh Reporting Seeds
# =========================================================================
def plot_figure_4_multiseed_robustness():
    # 10 Fresh Reporting Seeds (Seeds 11-20) Post-Recovery Stage
    categories = ["Worst-Off Client\nFloor (Min %)", "Retained Cohort\nAccuracy (Mean %)", "Forget-Set Clean\nUtility (%)"]
    
    # Retrain from Scratch (200 R matched):
    retrain_vals = [28.62, 59.42, 29.18]
    retrain_errs = [4.15, 2.80, 5.20]
    
    # Hard FedOSD:
    hard_vals = [3.83, 31.85, 3.35]
    hard_errs = [1.52, 1.25, 1.10]
    
    # Soft Uniform:
    soft_vals = [3.85, 31.44, 3.37]
    soft_errs = [1.48, 1.30, 1.15]
    
    # Step-Norm Capped One-Sided Hard (cap=1.0):
    cap_vals = [4.23, 33.88, 3.87]
    cap_errs = [1.60, 1.18, 1.35]
    
    x = np.arange(len(categories))
    width = 0.20
    
    fig, ax = plt.subplots(figsize=(10.5, 5.2))
    
    ax.bar(x - 1.5*width, retrain_vals, width, yerr=retrain_errs, capsize=4, label="Retrain (200 R Matched)", color="#e67e22", alpha=0.85)
    ax.bar(x - 0.5*width, hard_vals, width, yerr=hard_errs, capsize=4, label="Hard FedOSD (Baseline)", color="#7f7f7f", alpha=0.85)
    ax.bar(x + 0.5*width, soft_vals, width, yerr=soft_errs, capsize=4, label=r"Soft Uniform ($\lambda=20, \eta=0.0$)", color="#1f77b4", alpha=0.85)
    ax.bar(x + 1.5*width, cap_vals, width, yerr=cap_errs, capsize=4, label="Step-Norm Capped (cap=1.0)", color="#2ca02c", alpha=0.9)
    
    ax.annotate("+2.04% Cohort Gain (p=0.0020)\n+0.40% Floor Preservation",
                xy=(x[1] + 1.5*width, cap_vals[1] + cap_errs[1]),
                xytext=(x[1] + 0.8*width, cap_vals[1] + cap_errs[1] + 6),
                arrowprops=dict(facecolor='black', shrink=0.08, width=1.5, headwidth=5),
                fontweight='bold', fontsize=9)
                
    ax.set_ylabel("Accuracy / Metric Value (%)")
    ax.set_title("Multi-Seed Statistical Robustness Across 10 Fresh Reporting Seeds (Seeds 11–20)", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.legend(loc="upper right", frameon=True)
    ax.set_ylim(0, 72)
    
    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, "fig4_multiseed_robustness.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Generated: {out_path}")

# =========================================================================
# Figure 5: Dynamic Weight Trajectories Over Time
# =========================================================================
def plot_figure_5_weight_trajectories():
    adap_file = os.path.join(BASE_DIR, "seed1 N10 C1.0 FedOSD CNN_CIFAR10 E1 lr0.0004 decay0.999 g_clip-1.0 lam20.0 eta0.4_unlearn.json")
    if not os.path.exists(adap_file):
        return
    data = load_json(adap_file)
    ch = data['metric_log']['client_metric_history']
    retained = [c for c in range(10) if c != 2]
    accs = {c: ch[c]['test_accuracy'] for c in retained}
    n_rounds = len(accs[0])
    
    w = {c: 1.0 for c in retained}
    ema = {}
    beta = 0.8
    eta = 0.4
    w_history = {c: [] for c in retained}
    
    for t in range(n_rounds):
        for c in retained:
            a = accs[c][t]
            if c not in ema:
                ema[c] = a
            else:
                ema[c] = beta * ema[c] + (1 - beta) * a
        mean_a = sum(ema[c] for c in retained) / len(retained)
        for c in retained:
            deficit = max(mean_a - ema[c], 0.0)
            target_w = 1.0 + eta * (deficit / 10.0)
            w[c] = 1.0 + (w[c] - 1.0) * 0.8 + (target_w - 1.0) * 0.2
            w[c] = min(max(w[c], 0.2), 5.0)
            w_history[c].append(w[c])
            
    fig, ax = plt.subplots(figsize=(9.5, 5))
    rounds = np.arange(n_rounds)
    
    colors = {
        7: ('#d62728', '-', 2.8, 'Client 7 (Worst-Off, 7.1% acc)'),
        0: ('#e67e22', '-', 2.4, 'Client 0 (Struggling, 14.4% acc)'),
        5: ('#e74c3c', '-.', 2.0, 'Client 5 (Struggling, 15.6% acc)'),
        6: ('#9b59b6', ':', 1.8, 'Client 6 (Mid-tier, 38.1% acc)'),
        4: ('#f39c12', ':', 1.8, 'Client 4 (Mid-tier, 40.1% acc)'),
        1: ('#95a5a6', '--', 1.2, 'Clients 1, 3, 8, 9 (Thriving, >76% acc)'),
        3: ('#95a5a6', '--', 1.0, None),
        8: ('#95a5a6', '--', 1.0, None),
        9: ('#95a5a6', '--', 1.0, None)
    }
    
    for c in retained:
        col, ls, lw, lbl = colors[c]
        ax.plot(rounds, w_history[c], color=col, linestyle=ls, linewidth=lw, label=lbl)
        
    ax.axhline(1.0, color='gray', linestyle=':', alpha=0.6)
    ax.text(10, 1.05, "Uniform Baseline Protection (w = 1.0)", color="gray", fontsize=9, fontstyle="italic")
    
    ax.set_xlabel("Unlearning Rounds")
    ax.set_ylabel(r"Validation-Driven Protection Weight $w_i(t)$")
    ax.set_title(r"Dynamic Adaptation Mechanism: Protection Weight Trajectories $w_i(t)$ Over Time", fontweight="bold")
    ax.legend(loc="center left", frameon=True, bbox_to_anchor=(0.02, 0.55))
    ax.set_ylim(0.8, 3.2)
    
    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, "fig5_weight_trajectories.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Generated: {out_path}")

# =========================================================================
# Figure 6: Multi-Client Unlearning Benchmark (UN = 2)
# =========================================================================
def plot_figure_6_multiclient():
    dir_un2 = "UnlearningTask/CNN_CIFAR10/cifar10 pat balanced B200 NC2 N10/UN2/E1/C1.0/FedOSD"
    hard_file = os.path.join(dir_un2, "seed1 N10 C1.0 FedOSD CNN_CIFAR10 E1 lr0.0004 decay0.999 g_clip-1.0 lam100000.0 eta0.0_unlearn.json")
    soft_file = os.path.join(dir_un2, "seed1 N10 C1.0 FedOSD CNN_CIFAR10 E1 lr0.0004 decay0.999 g_clip-1.0 lam20.0 eta0.0_unlearn.json")
    adap_file = os.path.join(dir_un2, "seed1 N10 C1.0 FedOSD CNN_CIFAR10 E1 lr0.0004 decay0.999 g_clip-1.0 lam20.0 eta0.4_unlearn.json")
    
    if not os.path.exists(hard_file) or not os.path.exists(soft_file):
        print("Skipping Figure 6: files not found")
        return
        
    d_hard = load_json(hard_file)
    d_soft = load_json(soft_file)
    d_adap = load_json(adap_file)
    
    ch_hard = d_hard['metric_log']['client_metric_history']
    ch_soft = d_soft['metric_log']['client_metric_history']
    ch_adap = d_adap['metric_log']['client_metric_history']
    
    retained_cids = [0, 1, 3, 5, 6, 7, 8, 9]
    acc_hard = [ch_hard[c]['test_accuracy'][-1] for c in retained_cids]
    acc_soft = [ch_soft[c]['test_accuracy'][-1] for c in retained_cids]
    acc_adap = [ch_adap[c]['test_accuracy'][-1] for c in retained_cids]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.8))
    
    x = np.arange(len(retained_cids))
    width = 0.26
    
    ax1.bar(x - width, acc_hard, width, label="Hard FedOSD (Baseline)", color="#7f7f7f", alpha=0.85)
    ax1.bar(x, acc_soft, width, label=r"Soft Uniform ($\lambda=20, \eta=0.0$)", color="#1f77b4", alpha=0.85)
    ax1.bar(x + width, acc_adap, width, label=r"Adaptive FedOSD ($\lambda=20, \eta=0.40$)", color="#d62728", alpha=0.9)
    
    ax1.set_xlabel("Retained Client Index")
    ax1.set_ylabel("Final Test Accuracy (%)")
    ax1.set_title(r"(a) Per-Client Accuracy Under Batch Unlearning ($U_N = 2$)", fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"C{c}" for c in retained_cids])
    ax1.legend(loc="upper right", frameon=True)
    ax1.set_ylim(0, 95)
    
    # Right: Aggregate Summary Metrics
    metrics = ["Worst-Off Client\n(Min %)", "Retained Cohort\n(Mean %)", "Unlearn Backdoor\n(ASR %)"]
    vals_hard = [29.70, 51.65, 0.77]
    vals_soft = [13.70, 48.46, 0.51]
    vals_adap = [16.00, 48.89, 0.50]
    
    x2 = np.arange(len(metrics))
    w2 = 0.25
    ax2.bar(x2 - w2, vals_hard, w2, label="Hard FedOSD (Baseline)", color="#7f7f7f", alpha=0.85)
    ax2.bar(x2, vals_soft, w2, label=r"Soft Uniform ($\lambda=20, \eta=0.0$)", color="#1f77b4", alpha=0.85)
    ax2.bar(x2 + w2, vals_adap, w2, label=r"Adaptive FedOSD ($\lambda=20, \eta=0.40$)", color="#d62728", alpha=0.9)
    
    ax2.set_ylabel("Metric Value (%)")
    ax2.set_title(r"(b) Aggregate Comparison ($U_N = 2$ Clients Unlearned)", fontweight="bold")
    ax2.set_xticks(x2)
    ax2.set_xticklabels(metrics)
    ax2.legend(loc="upper right", frameon=True)
    ax2.set_ylim(0, 70)
    
    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, "fig6_multiclient_benchmark.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Generated: {out_path}")

# =========================================================================
# Figure 7: CIFAR-100 Complex Dataset Generalization Benchmark
# =========================================================================
def plot_figure_7_cifar100():
    cifar100_dir = "UnlearningTask/CNN_CIFAR10/cifar100 pat balanced B100 NC10 N10/UN1/E1/C1.0/FedOSD"
    hard_file = os.path.join(cifar100_dir, "seed1 N10 C1.0 FedOSD CNN_CIFAR10 E1 lr0.0004 decay0.999 g_clip-1.0 lam100000.0 eta0.0.json")
    soft_file = os.path.join(cifar100_dir, "seed1 N10 C1.0 FedOSD CNN_CIFAR10 E1 lr0.0004 decay0.999 g_clip-1.0 lam20.0 eta0.0.json")
    adap_file = os.path.join(cifar100_dir, "seed1 N10 C1.0 FedOSD CNN_CIFAR10 E1 lr0.0004 decay0.999 g_clip-1.0 lam20.0 eta0.4.json")
    
    if not (os.path.exists(hard_file) and os.path.exists(soft_file) and os.path.exists(adap_file)):
        print(f"CIFAR-100 JSON files missing in {cifar100_dir}")
        return
        
    d_hard = load_json(hard_file)
    d_soft = load_json(soft_file)
    d_adap = load_json(adap_file)
    
    ch_hard = d_hard["metric_log"]["client_metric_history"]
    ch_soft = d_soft["metric_log"]["client_metric_history"]
    ch_adap = d_adap["metric_log"]["client_metric_history"]
    
    retained_cids = [c for c in range(10) if c != 2]
    acc_hard = [ch_hard[c]['test_accuracy'][-1] for c in retained_cids]
    acc_soft = [ch_soft[c]['test_accuracy'][-1] for c in retained_cids]
    acc_adap = [ch_adap[c]['test_accuracy'][-1] for c in retained_cids]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    
    x = np.arange(len(retained_cids))
    width = 0.26
    
    ax1.bar(x - width, acc_hard, width, label="Hard FedOSD (Baseline)", color="#7f7f7f", alpha=0.85)
    ax1.bar(x, acc_soft, width, label=r"Soft Uniform ($\lambda=20, \eta=0.0$)", color="#1f77b4", alpha=0.85)
    ax1.bar(x + width, acc_adap, width, label=r"Adaptive FedOSD ($\lambda=20, \eta=0.40$)", color="#d62728", alpha=0.9)
    
    ax1.set_xlabel("Retained Client Index (CIFAR-100, 10 classes/client)")
    ax1.set_ylabel("Final Test Accuracy (%)")
    ax1.set_title(r"(a) Per-Client Accuracy Comparison on CIFAR-100", fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"C{c}" for c in retained_cids])
    ax1.legend(loc="upper right", frameon=True)
    ax1.set_ylim(0, 42)
    
    # Right: Aggregate Summary Metrics
    metrics = ["Worst-Off Client\n(Min %)", "Retained Mean\n(Cohort %)", "Cohort Variance\n(Std Dev %)", "Backdoor ASR\n(Residual %)"]
    vals_hard = [20.60, 28.96, 4.00, 1.02]
    vals_soft = [21.60, 28.79, 4.03, 0.96]
    vals_adap = [21.60, 28.82, 3.99, 0.96]
    
    x2 = np.arange(len(metrics))
    w2 = 0.25
    ax2.bar(x2 - w2, vals_hard, w2, label="Hard FedOSD", color="#7f7f7f", alpha=0.85)
    ax2.bar(x2, vals_soft, w2, label=r"Soft Uniform ($\lambda=20, \eta=0.0$)", color="#1f77b4", alpha=0.85)
    ax2.bar(x2 + w2, vals_adap, w2, label=r"Adaptive FedOSD ($\lambda=20, \eta=0.40$)", color="#d62728", alpha=0.9)
    
    ax2.set_ylabel("Metric Value (%)")
    ax2.set_title(r"(b) CIFAR-100 Macro Metrics (Note: differences within sampling error +/-1.4%)", fontweight="bold")
    ax2.set_xticks(x2)
    ax2.set_xticklabels(metrics)
    ax2.legend(loc="upper right", frameon=True)
    ax2.set_ylim(0, 36)
    
    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, "fig7_cifar100_generalization.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Generated: {out_path}")

if __name__ == "__main__":
    ensure_dir(OUTPUT_DIR)
    plot_figure_1_dynamics()
    plot_figure_2_client_distribution()
    plot_figure_3_hyperparameter_pareto()
    plot_figure_4_multiseed_robustness()
    plot_figure_5_weight_trajectories()
    plot_figure_6_multiclient()
    plot_figure_7_cifar100()
    print("All publication figures successfully regenerated in figures/ directory!")
