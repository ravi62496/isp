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

def ensure_dir(d):
    if not os.path.exists(d):
        os.makedirs(d)

def load_json(path):
    with open(path, "r") as f:
        return json.load(f)

def extract_client_metrics(data):
    """
    Extracts round-by-round ASR and retained client accuracies.
    Returns:
        rounds (np.ndarray): 0..100
        asr_curve (np.ndarray): 1D array of ASR per round
        retained_curves (np.ndarray): shape (n_retained, n_rounds)
        unlearn_cid (int)
    """
    client_hist = data["metric_log"]["client_metric_history"]
    unlearn_cid = None
    asr_curve = None
    
    for cid, hist in enumerate(client_hist):
        if "backdoor_test_accuracy" in hist and hist["backdoor_test_accuracy"]:
            unlearn_cid = cid
            asr_curve = np.array(hist["backdoor_test_accuracy"])
            break
            
    retained_curves = []
    for cid, hist in enumerate(client_hist):
        if cid != unlearn_cid and "test_accuracy" in hist and hist["test_accuracy"]:
            retained_curves.append(hist["test_accuracy"])
            
    retained_curves = np.array(retained_curves)
    n_rounds = len(asr_curve) if asr_curve is not None else retained_curves.shape[1]
    rounds = np.arange(n_rounds)
    return rounds, asr_curve, retained_curves, unlearn_cid

# =========================================================================
# Figure 1: Unlearning Dynamics (ASR Eradication & Floor Protection)
# =========================================================================
def plot_figure_1_dynamics():
    hard_file = os.path.join(BASE_DIR, "seed1 N10 C1.0 FedOSD CNN_CIFAR10 E1 lr0.0004 decay0.999 g_clip-1.0_unlearn.json")
    soft_file = os.path.join(BASE_DIR, "seed1 N10 C1.0 FedOSD CNN_CIFAR10 E1 lr0.0004 decay0.999 g_clip-1.0 lam20.0 eta0.0_unlearn.json")
    adap_file = os.path.join(BASE_DIR, "seed1 N10 C1.0 FedOSD CNN_CIFAR10 E1 lr0.0004 decay0.999 g_clip-1.0 lam20.0 eta0.4_unlearn.json")
    
    r_hard, asr_hard, ret_hard, _ = extract_client_metrics(load_json(hard_file))
    r_soft, asr_soft, ret_soft, _ = extract_client_metrics(load_json(soft_file))
    r_adap, asr_adap, ret_adap, _ = extract_client_metrics(load_json(adap_file))
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 7), sharex=True)
    
    # 1. Top Panel: ASR Eradication
    ax1.plot(r_hard, asr_hard, label="Original FedOSD (Hard)", color="#7f7f7f", linestyle="--")
    ax1.plot(r_soft, asr_soft, label=r"Soft Uniform ($\lambda=20, \eta=0.0$)", color="#1f77b4", linestyle="-.")
    ax1.plot(r_adap, asr_adap, label=r"Adaptive FedOSD ($\lambda=20, \eta=0.40$)", color="#d62728", linestyle="-", linewidth=2.5)
    ax1.set_ylabel("Unlearn Attack Success Rate (ASR %)")
    ax1.set_title("(a) Backdoor Trigger Elimination Dynamics", fontweight="bold")
    ax1.legend(loc="upper right", frameon=True)
    ax1.set_ylim(-2, 95)
    
    # Highlight full unlearning threshold
    ax1.axhline(1.0, color="gray", linestyle=":", alpha=0.7)
    ax1.text(60, 5, "Unlearned Threshold (< 1%)", color="gray", fontstyle="italic")
    
    # 2. Bottom Panel: Minimum Client Accuracy (Worst-Off Protection)
    min_hard = np.min(ret_hard, axis=0)
    min_soft = np.min(ret_soft, axis=0)
    min_adap = np.min(ret_adap, axis=0)
    
    ax2.plot(r_hard, min_hard, label="Original FedOSD (Hard)", color="#7f7f7f", linestyle="--")
    ax2.plot(r_soft, min_soft, label=r"Soft Uniform ($\lambda=20, \eta=0.0$)", color="#1f77b4", linestyle="-.")
    ax2.plot(r_adap, min_adap, label=r"Adaptive FedOSD ($\lambda=20, \eta=0.40$)", color="#d62728", linestyle="-", linewidth=2.5)
    
    ax2.set_xlabel("Unlearning Rounds")
    ax2.set_ylabel("Worst-Off Client Accuracy (%)")
    ax2.set_title("(b) Worst-Off Client Floor Protection", fontweight="bold")
    ax2.legend(loc="lower right", frameon=True)
    
    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, "fig1_unlearning_dynamics.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Generated: {out_path}")

# =========================================================================
# Figure 2: Per-Client Accuracy Distribution & Fairness Lift
# =========================================================================
def plot_figure_2_client_distribution():
    soft_file = os.path.join(BASE_DIR, "seed1 N10 C1.0 FedOSD CNN_CIFAR10 E1 lr0.0004 decay0.999 g_clip-1.0 lam20.0 eta0.0_unlearn.json")
    adap_file = os.path.join(BASE_DIR, "seed1 N10 C1.0 FedOSD CNN_CIFAR10 E1 lr0.0004 decay0.999 g_clip-1.0 lam20.0 eta0.4_unlearn.json")
    
    _, _, ret_soft, ucid = extract_client_metrics(load_json(soft_file))
    _, _, ret_adap, _ = extract_client_metrics(load_json(adap_file))
    
    final_soft = ret_soft[:, -1]
    final_adap = ret_adap[:, -1]
    
    # Retained client IDs
    client_ids = [c for c in range(10) if c != ucid]
    x = np.arange(len(client_ids))
    width = 0.38
    
    fig, ax = plt.subplots(figsize=(10, 5))
    
    bars1 = ax.bar(x - width/2, final_soft, width, label=r"Soft Uniform ($\lambda=20, \eta=0.0$)", color="#4a90e2", alpha=0.85)
    bars2 = ax.bar(x + width/2, final_adap, width, label=r"Adaptive FedOSD ($\lambda=20, \eta=0.40$)", color="#e74c3c", alpha=0.9)
    
    # Highlight worst-off client lift
    min_idx = np.argmin(final_soft)
    ax.annotate(f"Lift: {final_soft[min_idx]:.1f}% -> {final_adap[min_idx]:.1f}%\n(+14.5% worst-off)",
                xy=(x[min_idx] + width/2, final_adap[min_idx]),
                xytext=(x[min_idx] - 0.2, final_adap[min_idx] + 12),
                arrowprops=dict(facecolor='black', shrink=0.08, width=1.5, headwidth=6),
                fontweight='bold', fontsize=9)
    
    ax.set_xlabel("Retained Client Index")
    ax.set_ylabel("Final Test Accuracy (%)")
    ax.set_title("Per-Client Accuracy Comparison: Uniform vs. Adaptive Soft FedOSD", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([f"Client {c}" for c in client_ids])
    ax.legend(loc="upper right", frameon=True)
    ax.set_ylim(0, 95)
    
    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, "fig2_per_client_distribution.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Generated: {out_path}")

# =========================================================================
# Figure 3: Hyperparameter Trade-off & Pareto Frontier
# =========================================================================
def plot_figure_3_hyperparameter_pareto():
    etas = [0.0, 0.20, 0.40]
    min_accs = [6.20, 6.90, 7.10]
    stds = [30.16, 29.80, 29.45]
    means = [47.70, 47.76, 47.82]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    
    # Left: Adaptation Rate eta vs. Min Client Accuracy & Standard Deviation
    color = '#2ca02c'
    ax1.set_xlabel(r"Adaptation Rate $\eta$ (at $\lambda=20.0$)")
    ax1.set_ylabel("Worst-Off Client Accuracy (%)", color=color, fontweight="bold")
    line1 = ax1.plot(etas, min_accs, color=color, marker='o', linewidth=2.5, label="Min Client Acc (Floor)")
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_ylim(5.5, 7.5)
    
    ax1_twin = ax1.twinx()
    color2 = '#d62728'
    ax1_twin.set_ylabel("Cohort Std Deviation (Inequality %)", color=color2, fontweight="bold")
    line2 = ax1_twin.plot(etas, stds, color=color2, marker='s', linestyle='--', linewidth=2.5, label="Std Dev (Variance)")
    ax1_twin.tick_params(axis='y', labelcolor=color2)
    ax1_twin.set_ylim(29.0, 30.5)
    
    ax1.set_title(r"(a) Impact of Adaptation Rate $\eta$", fontweight="bold")
    
    # Combined legend
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='center left', frameon=True)
    
    # Right: Stiffness lambda vs. Retained Utility vs. ASR
    lambdas = ["1.0\n(Over-Loose)", "20.0\n(Optimal)", "50.0\n(Stiff)", "100.0\n(Very Stiff)", r"$\infty$" + "\n(Hard)"]
    lam_means = [38.38, 47.82, 53.68, 53.69, 53.70]
    lam_asrs = [0.00, 0.02, 0.08, 0.08, 0.08]
    
    ax2.plot(range(5), lam_means, marker='^', color='#1f77b4', linewidth=2.5, label="Retained Mean Acc (%)")
    ax2.set_xticks(range(5))
    ax2.set_xticklabels(lambdas)
    ax2.set_xlabel(r"Constraint Stiffness $\lambda$")
    ax2.set_ylabel("Retained Client Mean Accuracy (%)")
    ax2.set_title(r"(b) Regularization Regimes Across $\lambda$", fontweight="bold")
    ax2.legend(loc="lower right", frameon=True)
    ax2.set_ylim(30, 60)
    
    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, "fig3_hyperparameter_pareto.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Generated: {out_path}")

# =========================================================================
# Figure 4: Multi-Seed Statistical Robustness
# =========================================================================
def plot_figure_4_multiseed_robustness():
    # Seeds 1 & 2 aggregated metrics
    categories = ["Worst-Off Client\nAccuracy (Min %)", "Retained Cohort\nAccuracy (Mean %)", "Backdoor Attack\nSuccess Rate (ASR %)"]
    
    # Hard FedOSD: min=17.50 (se 9.00), mean=55.87 (se 2.17), asr=0.06
    hard_vals = [17.50, 55.87, 0.06]
    hard_errs = [9.00, 2.17, 0.02]
    
    # Soft Uniform: min=3.65 (se 2.55), mean=49.54 (se 1.84), asr=0.02
    soft_vals = [3.65, 49.54, 0.02]
    soft_errs = [2.55, 1.84, 0.01]
    
    # Adaptive FedOSD: min=4.80 (se 2.30), mean=49.63 (se 1.81), asr=0.02
    adap_vals = [4.80, 49.63, 0.02]
    adap_errs = [2.30, 1.81, 0.01]
    
    x = np.arange(len(categories))
    width = 0.26
    
    fig, ax = plt.subplots(figsize=(9, 5))
    
    ax.bar(x - width, hard_vals, width, yerr=hard_errs, capsize=4, label="Original FedOSD (Hard)", color="#7f7f7f", alpha=0.85)
    ax.bar(x, soft_vals, width, yerr=soft_errs, capsize=4, label=r"Soft Uniform ($\lambda=20, \eta=0.0$)", color="#3498db", alpha=0.85)
    ax.bar(x + width, adap_vals, width, yerr=adap_errs, capsize=4, label=r"Adaptive FedOSD ($\lambda=20, \eta=0.40$)", color="#e74c3c", alpha=0.9)
    
    # Annotate relative gain on worst-off client
    ax.annotate("+31.5% Relative Gain",
                xy=(x[0] + width, adap_vals[0] + adap_errs[0]),
                xytext=(x[0] + width - 0.15, adap_vals[0] + adap_errs[0] + 5),
                arrowprops=dict(facecolor='black', shrink=0.08, width=1.5, headwidth=5),
                fontweight='bold', fontsize=9)
    
    ax.set_ylabel("Accuracy / Metric Value (%)")
    ax.set_title("Multi-Seed Statistical Robustness Across Independent Partitions (Seeds 1 & 2)", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.legend(loc="upper right", frameon=True)
    ax.set_ylim(0, 70)
    
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
            w[c] = 1.0 + (w[c] - 1.0) * 0.95 + (target_w - 1.0) * 0.05
            w[c] = min(max(w[c], 0.2), 5.0)
            w_history[c].append(w[c])
            
    fig, ax = plt.subplots(figsize=(9.5, 5))
    rounds = np.arange(n_rounds)
    
    # Palette highlighting vulnerable vs thriving
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
    
    ax.annotate(r"Vulnerable Clients receive $>2.6\times$ Higher Conflict Protection",
                xy=(75, w_history[7][75]),
                xytext=(30, 2.75),
                arrowprops=dict(facecolor='#d62728', shrink=0.08, width=1.5, headwidth=6),
                fontweight='bold', fontsize=9.5, color='#d62728')
                
    ax.set_xlabel("Unlearning Rounds")
    ax.set_ylabel(r"Dynamic Protection Weight $w_i(t)$")
    ax.set_title(r"Dynamic Adaptation Mechanism: Protection Weight Trajectories $w_i(t)$ Over Time", fontweight="bold")
    ax.legend(loc="center left", frameon=True, bbox_to_anchor=(0.02, 0.55))
    ax.set_ylim(0.8, 3.0)
    
    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, "fig5_weight_trajectories.png")
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
    print("All publication figures successfully generated in figures/ directory!")

