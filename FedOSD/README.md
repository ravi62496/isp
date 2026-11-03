# One-Sided Unscaled FedOSD: Federated Unlearning with Non-Harm Constraints and Norm Bounding

Implementation and empirical evaluation of **One-Sided Unscaled FedOSD (cap 1.0)**, extending the federated unlearning framework of:
> **Pan Z, Wang Z, Li C, et al.** *Federated unlearning with gradient descent and conflict mitigation.* Proceedings of the AAAI Conference on Artificial Intelligence (AAAI-25), 39(19): 19804-19812.

---

## 1. Overview & Motivation

### Background: Standard Hard FedOSD
Standard FedOSD unlearns target client(s) by taking the aggregated unlearning pseudo-gradient $g_u$ and projecting it onto the orthogonal complement of the retained clients' gradient subspace:

$$\min_d \|d - g_u\|^2 \quad \text{subject to} \quad g_i^T d = 0 \quad \forall i \in \mathcal{C}_{\text{retained}}$$

This is solved in closed form via pseudoinverse projection ($\lambda = \infty$):
$$d_{\text{hard}} = g_u - A^T (A A^T)^{-1} A g_u$$

### Structural Trade-offs in Non-IID Unlearning
In non-IID federated learning, unlearning affects retained clients heterogeneously:
1. **Class-Overlap Vulnerability**: When a retained client shares classes with the unlearned client, unlearning gradient directions conflict directly with that client's local objectives.
2. **Over-Constrained Orthogonality**: Standard Hard FedOSD forces $g_i^T d = 0$ symmetrically for all retained clients. This prohibits harmful descent directions ($g_i^T d < 0$), but also strictly forbids *helpful* descent directions ($g_i^T d > 0$) that would otherwise improve retained client performance.
3. **Gradient Rescaling vs. Pure Projection**: In the legacy implementation, unconstrained descent directions were subjected to norm rescaling: $d = \frac{d}{\|d_{\text{raw}}\|} \cdot \|g_u\|$, artificially dilating descent steps up to the unlearned client's local gradient norm $\|g_u\|$ (at Round 1, $\|g_u\| \approx 27.84$ vs $\|d_{\text{raw}}\| \approx 20.81$, an artificial $+33.8\%$ step enlargement). Legacy Hard FedOSD was similarly rescaled to $\|g_u\|$. This aggressive dilation causes excessive parameter displacement and collateral utility degradation on retained clients.

### Proposed Solution: One-Sided, Unscaled Projection (cap 1.0)
To achieve the benefit of one-sided non-harm constraints ($\hat{g}_i^T d \ge 0$) without artificial $\|g_u\|$ rescaling or excessive displacement, **One-Sided Unscaled FedOSD (cap 1.0)** solves the one-sided projection via dual Non-Negative Least Squares (NNLS) and caps the update vector norm at $\|d_{\text{hard}}\|$:

$$d = \begin{cases} d_{\text{raw}} & \text{if } \|d_{\text{raw}}\| \le \text{cap} \cdot \|d_{\text{hard}}\| \\ \text{cap} \cdot \frac{\|d_{\text{hard}}\|}{\|d_{\text{raw}}\|} d_{\text{raw}} & \text{otherwise} \end{cases}$$

The frozen primary configuration sets $\text{cap} = 1.0$ with uniform weighting ($\eta = 0$).

---

## 2. Mathematical Formulations

### 2.1 Symmetric Closed-Form Projection (Hard Baseline, $\lambda = \infty$)
Let $\hat{A} \in \mathbb{R}^{n \times p}$ denote the matrix whose rows are unit-normalized retained client gradients $\hat{g}_i = g_i / \|g_i\|$. The closed-form orthogonal projection is:

$$d_{\text{hard}} = g_u - \hat{A}^T (\hat{A} \hat{A}^T)^{-1} \hat{A} g_u$$

### 2.2 One-Sided Non-Harm Projection (Dual NNLS Solve)
Since parameter updates follow $w^{(t+1)} = w^{(t)} - \eta_{\text{lr}} d$, the first-order change in client $i$'s loss is $\Delta \mathcal{L}_i \approx -\eta_{\text{lr}} (g_i^T d)$. Therefore, a retained client is not harmed whenever $\hat{g}_i^T d \ge 0$.

The one-sided projection formulation is:

$$\min_d \frac{1}{2}\|d - g_u\|^2 + \frac{1}{2} \sum_{i \in \mathcal{C}_{\text{retained}}} \lambda_i \max(0, -\hat{g}_i^T d)^2$$

Introducing slack variables $s \ge 0$ with $-\hat{A} d \le s$, the Lagrange dual problem reduces to:

$$\min_{\nu \ge 0} \nu^T (G + D) \nu + 2 \nu^T b \quad \text{where} \quad G = \hat{A} \hat{A}^T, \quad b = \hat{A} g_u, \quad D = \text{diag}\left(\frac{1}{\lambda_i}\right)$$

With $\lambda = 10^5$, Cholesky factorization $M = G + D + 10^{-10} I = L L^T$ and $y = -L^{-1} b$ enables solution via Non-Negative Least Squares (NNLS):

$$\min_{\nu \ge 0} \|L^T \nu - y\|^2 \implies d_{\text{raw}} = g_u + \hat{A}^T \nu$$

### 2.3 Step-Norm Bounding
When $\|d_{\text{raw}}\| > \|d_{\text{hard}}\|$, the update is scaled back to match $\|d_{\text{hard}}\|$ exactly ($\text{cap} = 1.0$), ensuring the parameter displacement does not exceed the conservative hard boundary.

---

## 3. Training Regimes & Decision Gates

### 3.1 Training Regimes & Gate A Decision (Tuning Seeds 101–103)
To eliminate confounds from inconsistent learning rates and under-trained pretraining models, regimes were audited across tuning seeds 101–103 ($n=3$):

| Regime | Pretrain lr | Retrain lr | Recovery $r_{\text{lr}}$ | Purpose & Characteristics | Retrain Mean (Seeds 101–103) |
| :--- | :---: | :---: | :---: | :--- | :---: |
| **R1 (Historical / Exploratory)** | 0.01 | 0.05 | $10^{-6}$ | Historical setting (Seeds 11–20); under-trained pretrain | ~59.4% (R=200) |
| **R1b (Control)** | 0.01 | 0.01 | $10^{-6}$ | Controls for lr disparity at pretrain lr | 44.69% $\pm$ 2.45% |
| **R2 (Primary Matched)** | 0.05 | 0.05 | $10^{-4}$ | Matched lr across pipeline; converged pretrain | **58.10% $\pm$ 2.25%** |

* **Gate A Resolution**: Primary regime is **R2**. The R1b control demonstrates that the learning rate disparity ($0.01$ vs $0.05$) alone accounts for **$13.41$ percentage points** (44.7%) of the historical 30-point Retrain gap in R1. Under primary regime R2, the true gap between Hard FedOSD ($52.36\%$) and Retrain ($58.10\%$) is only **$5.74$ percentage points** on tuning seeds.
* **Recovery Rate Freeze**: A systematic sweep of $r_{\text{lr}} \in \{10^{-6}, 10^{-4}, 10^{-3}, 10^{-2}\}$ established that $r_{\text{lr}} = 10^{-4}$ maximizes baseline Hard recovered mean ($52.36\% \pm 2.43\%$ vs $44.93\%$ at $10^{-6}$) while keeping backdoor ASR under $4.3\%$. It is frozen in [`FROZEN_CONFIGS.md`](FROZEN_CONFIGS.md).

### 3.2 Gradient Rescaling Finding & Cap Sweep (Regime R2, Tuning Seeds 101–103)
Auditing projection mechanics in `tfedplat/algorithm/unlearning/FedOSD.py` (lines 371–380) revealed the operational difference between unscaled projection and legacy unconstrained descent:
- **One-Sided, Unscaled (cap 1.0)**: Traverses the `cap_hard_ratio > 0` code branch. When $\|d_{\text{raw}}\| > \|d_{\text{hard}}\|$, it clamps the norm to $\|d_{\text{hard}}\|$; for cap $\ge 2.0$ (or $10^9$), it outputs the pure unconstrained NNLS projection $d_{\text{raw}}$ without any dilation (displacement $= 0.2737$ on Seed 101).
- **One-Sided Rescaled to $\|g_u\|$**: Traverses the legacy `elif d_raw_norm > 1e-8` branch ($d = \frac{d}{\|d_{\text{raw}}\|} \cdot \|g_u\|$), inflating the vector norm up to $\|g_u\|$ (displacement $= 0.3791$ on Seed 101).
- **Hard FedOSD Rescaled to $\|g_u\|$**: Similarly hits the legacy branch, rescaling $d_{\text{hard}}$ to $\|g_u\|$ (displacement $= 0.3648$ on Seed 101).

Empirical audit of the raw NNLS ratio $\|d_{\text{raw}}\| / \|d_{\text{hard}}\|$ before capping on tuning seeds 104–105 shows that $d_{\text{raw}}$ exceeds $d_{\text{hard}}$ by up to $+33.4\%$ in early rounds before stabilizing near $1.00$.

#### Cap Ratio Sweep Matrix (Tuning Seeds 101–103, Mean Across Seeds)
| Cap Ratio | Retained Mean | Worst-Off Floor | Max Collateral Drop | Forget-Set Clean | Backdoor ASR | Displacement $\|w_{\text{final}} - w_{\text{pre}}\|$ |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **0.50** | 52.05% | 23.83% | 5.94% | 26.61% | 8.05% | 0.1697 |
| **0.75** | 52.35% | 24.33% | 5.61% | 26.83% | 6.69% | 0.2098 |
| **1.00** | **52.48%** | **24.83%** | **5.44%** | 27.00% | 5.98% | 0.2395 |
| **1.25** | 52.47% | 24.83% | 5.44% | 27.06% | 5.96% | 0.2421 |
| **1.50** | 52.47% | 24.83% | 5.44% | 27.06% | 5.95% | 0.2422 |

#### Matched-Strength Control (At Matched Parameter Displacement)
To verify that gains are not merely due to taking smaller steps, One-sided, unscaled (cap 1.0) was compared against a Hard-scaled control sweep (step scale $\in \{0.25, 0.5, 0.75, 1.0\}$) at matched parameter displacement $\|w_{\text{final}} - w_{\text{pre}}\|$:

| Metric at Matched Displacement | Paired Difference (Capped vs Hard-Scaled) | 95% Bootstrap CI | Sign Agreement |
| :--- | :---: | :---: | :---: |
| **Retained Client Mean Utility** | **+0.24%** | [+0.17%, +0.32%] | 3/3 positive |
| **Worst-Off Client Floor** | **+0.38%** | [+0.06%, +0.62%] | 3/3 positive |
| **Max Collateral Drop** | **-0.21%** | [-0.63%, +0.17%] | 2/3 negative |

* **Gate B Working Hypothesis**: Candidate headline: *"A better trade-off at equal unlearning strength: the step-norm cap is a useful constraint, not only a smaller step."* This remains a hypothesis on tuning seeds ($n=3$) pending formal confirmation on fresh seeds 21–30.

---

## 4. Empirical Evaluation Protocol & Benchmarks (Historical Regime R1, Exploratory Seeds 11–20)

### Experimental Timeline & Stage Matching
| Stage | Communication Rounds | Description | Matched Retrain Reference |
| :--- | :---: | :--- | :--- |
| **Pretrain** | Rounds 1 to 100 | Initial federated pretraining on full cohort | — |
| **Unlearning** | Rounds 1 to 100 | Unlearning updates ($UR=100$) | Retrain from scratch at $R=200$ |
| **Recovery** | Rounds 101 to 200 | Post-unlearning federated recovery training | Retrain from scratch at $R=300$ (matched total post-init budget) |

* **Stage Matching Note**: In Table 4.1 (unlearning stage, 100 unlearning rounds), the matched reference is Retrain from scratch at $R=200$. In Table 4.2 (post-recovery stage, 100 unlearning + 100 recovery rounds), the reference is Retrain at $R=300$ to match the total post-initialization budget of 200 rounds.
* **CLI Flag Meaning (`--R 200`)**: In `run_UnlearningTask.py`, passing `--R 200 --UR 100` directs the framework to run 100 unlearning rounds followed by 100 post-unlearning federated recovery rounds (total 200 rounds). In historical R1 scripts, passing `--R 100 --UR 100` stopped after 1 recovery round.
* **Historical Regime R1 CLI Execution Commands**:
  - Pretrain: `python run_UnlearningTask.py --seed <s> --lr 0.01 --R 100 --UR 100 --unlearn_pretrain True ...`
  - Retrain: `python run_UnlearningTask.py --seed <s> --lr 0.05 --R 200 (or 300) --UR 0 --retrain True ...`
  - Hard: `python run_UnlearningTask.py --seed <s> --lr 0.0004 --r_lr 1e-6 --proj_mode hard ...`
  - Capped: `python run_UnlearningTask.py --seed <s> --lr 0.0004 --r_lr 1e-6 --proj_mode one_sided --cap_hard_ratio 1.0 ...`

Evaluation metrics below reflect exploratory reporting seeds 11–20 ($N=10$, regime R1). Bolding indicates the closest distance to the matched Retrain reference.

---

### 4.1 End of Unlearning Stage (Historical Regime R1, Exploratory Seeds 11–20)
| Method / Configuration | Proj. Mode | Seeds | Backdoor ASR (Target: 3.03%) | Retained Mean $\pm$ Std Dev | Worst-Off Floor (Min) $\uparrow$ | Max Collateral Drop $\downarrow$ | Forget Clean Utility $\uparrow$ | MIA ROC-AUC (Target: 0.5055) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Retrain Baseline ($R=200$)** | Scratch | 11–20 | 3.03% | **59.42% $\pm$ 14.72%** | **28.62%** | **-8.38%** | **29.18%** | 0.5055 |
| **Hard FedOSD ($\lambda=\infty$)** | Closed Hard | 11–20 | 0.02% | 29.09% $\pm$ 14.90% | 2.10% | 18.32% | 1.42% | 0.5143 |
| **Soft Uniform ($\lambda=20, \eta=0$)** | Two-Sided | 11–20 | 0.00% | 29.08% $\pm$ 17.62% | 1.43% | 21.54% | 1.08% | **0.5064** |
| **One-sided rescaled to ||g_u||** | One-Sided | 11–20 | 0.03% | 30.67% $\pm$ 17.19% | 0.82% | 22.80% | 0.42% | 0.5067 |
| **Step-Norm Capped ($\text{cap}=1.0$)** | One-Sided | 11–20 | **0.37%** | **32.51% $\pm$ 14.64%** | **2.42%** | **14.68%** | **1.82%** | 0.5105 |

---

### 4.2 Post-Recovery Stage (Historical Regime R1, Exploratory Seeds 11–20)
| Method / Configuration | Proj. Mode | Seeds | Backdoor ASR (Target: 3.03%) | Retained Mean $\pm$ Std Dev | Worst-Off Floor (Min) $\uparrow$ | Max Collateral Drop $\downarrow$ | Forget Clean Utility $\uparrow$ | MIA ROC-AUC (Target: 0.5055) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Retrain Baseline ($R=300$)** | Scratch | 11–20 | 3.03% | **61.85% $\pm$ 13.90%** | **33.10%** | **-9.15%** | **31.40%** | 0.5055 |
| **Hard FedOSD ($\lambda=\infty$)** | Closed Hard | 11–20 | 0.13% | 31.85% $\pm$ 14.85% | 3.83% | 14.76% | 3.35% | 0.5143 |
| **Soft Uniform ($\lambda=20, \eta=0$)** | Two-Sided | 11–20 | 0.05% | 31.44% $\pm$ 17.40% | 3.85% | 15.49% | 3.37% | **0.5064** |
| **One-sided rescaled to ||g_u||** | One-Sided | 11–20 | 0.17% | 33.07% $\pm$ 17.02% | 2.72% | 14.72% | 2.20% | 0.5067 |
| **Step-Norm Capped ($\text{cap}=1.0$)** | One-Sided | 11–20 | **0.94%** | **33.88% $\pm$ 14.50%** | **4.23%** | **11.21%** | **3.87%** | 0.5105 |

---

### 4.3 Paired Statistical Comparisons with Explicit Holm Families (Historical Regime R1, Exploratory Seeds 11–20)
All p-values corrected within predefined family scopes:

```
Comparison / Test Family                  | Metric                | Paired Diff | 95% Bootstrap CI   | Raw p   | Holm p  | Signs
--------------------------------------------------------------------------------------------------------------------------------------
[Primary Family 1: Capped vs Hard (m=3 primary metrics)]
One-sided, unscaled (cap 1.0) vs Hard FedOSD           | Retained Cohort Mean  | +2.04%      | [+1.81%, +2.25%]   | 0.0020  | 0.0060  | 10/10 (+)
One-sided, unscaled (cap 1.0) vs Hard FedOSD           | Max Collateral Drop   | -3.55%      | [-5.58%, -1.48%]   | 0.0137  | 0.0274  | 9/10 (-)
One-sided, unscaled (cap 1.0) vs Hard FedOSD           | Worst-Off Floor (Min) | +0.40%      | [-0.30%, +1.37%]   | 0.6523  | 0.6523  | 6/10 (+)
(Secondary diagnostic)                    | Backdoor ASR          | +0.81%      | [+0.50%, +1.18%]   | 0.0020  | --      | 10/10 (+)
--------------------------------------------------------------------------------------------------------------------------------------
[Pairwise Family 2: Capped vs One-sided rescaled to ||g_u|| (m=2 metrics)]
One-sided, unscaled (cap 1.0) vs Unconstrained         | Worst-Off Floor (Min) | +1.51%      | [+0.62%, +2.45%]   | 0.0039  | 0.0078  | 9/10 (+)
One-sided, unscaled (cap 1.0) vs Unconstrained         | Retained Cohort Mean  | +0.81%      | [+0.35%, +1.28%]   | 0.0039  | 0.0078  | 9/10 (+)
--------------------------------------------------------------------------------------------------------------------------------------
[Pairwise Family 3: Unconstrained vs Hard FedOSD (m=2 metrics)]
One-sided rescaled to ||g_u|| vs Hard FedOSD    | Worst-Off Floor (Min) | -1.12%      | [-1.80%, -0.53%]   | 0.0039  | 0.0040  | 1/10 (+)
One-sided rescaled to ||g_u|| vs Hard FedOSD    | Retained Cohort Mean  | +1.23%      | [+0.89%, +1.59%]   | 0.0020  | 0.0040  | 10/10 (+)
```

* **Statistical Interpretation**: In historical regime R1, step-norm capping achieved a statistically significant increase in retained cohort mean ($+2.04\%$, Holm $p=0.0060$) and reduction in collateral degradation ($-3.55\%$, Holm $p=0.0274$) over plain Hard. Floor point estimate was $+0.40\%$ with CI spanning zero ($[-0.30\%, +1.37\%]$). Both floors ($2.72\%$ and $4.23\%$) represent severe degradation on scarce classes, but capping avoids the additional drop of unconstrained projection ($-1.12\%$, Holm $p=0.0078$). All findings on seeds 11–20 remain exploratory.

---

## 5. Negative Results (Archived Variants)

### 5.1 Dynamic Client-Adaptive Weighting ($\eta > 0$)
Variants attempting to dynamically scale client protection penalties based on local metrics were evaluated and archived:
1. **Deficit-Driven (`--weight_mode deficit`)**, **Baseline Drop-Driven (`--weight_mode drop`)**, and **Gradient Conflict (`--weight_mode conflict`)**: On exploratory seeds 1–2, adaptive weighting underperformed Hard FedOSD across worst-off floor, cohort mean, and collateral drop, while weights stayed within $\sim 1.0$ point of uniform.
2. **Validation Split Leakage Control**: Prior experiments using test-set accuracy feedback artificially inflated floor estimates from $\approx 9\%$ to $\approx 42\%$. Enforcing strict evaluation on held-out validation data removed this artifact. Adaptive variants provided no measurable advantage over uniform weights once test feedback was eliminated.
3. **Screening Matrix for E1 and E2 (Tuning Seed 101)**:
   Below are the screening evaluations conducted on tuning seed 101 under exploratory regime R1:

| Variant | Formulation / Knobs | Floor (%) | Mean (%) | ASR (%) | Forget Clean (%) | MIA AUC | Screening Outcome |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Hard FedOSD** | Baseline reference ($\lambda=\infty$) | 26.67% | 42.26% | 0.30% | 27.67% | 0.4970 | *Baseline* |
| **E1 ($\lambda=40, \eta=0.4$)** | Two-sided soft sweep (`wm_drop`) | 26.67% | 41.89% | 0.24% | 26.50% | 0.4972 | Underperformed Hard Mean |
| **E1 ($\lambda=50, \eta=0.4$)** | Two-sided soft sweep (`wm_drop`) | 26.50% | 41.98% | 0.26% | 26.50% | 0.4971 | Underperformed Hard Mean |
| **E2 ($\Delta_{\text{ref}}=10\%$)** | Per-client interpolation (`soft20`) | 25.50% | 42.09% | 0.28% | 25.83% | 0.4967 | Floor drop (-1.17%) |
| **E2 ($\Delta_{\text{ref}}=5\%$)** | Per-client interpolation (`soft20`) | 25.17% | 42.09% | 0.30% | 25.83% | 0.4965 | Floor drop (-1.50%) |
| **E3_cap1.0** | Step-Norm One-sided, unscaled (cap 1.0) | **26.83%** | **43.59%** | 0.44% | **29.17%** | 0.4952 | **Adopted Primary Method** |

### 5.2 Uniform Soft Relaxation ($\lambda = 20$)
Permitting projection vectors to enter the negative half-space ($\hat{g}_i^T d < 0$) via quadratic soft penalties degraded worst-off client floor (falling to $1.43\%$ at unlearning round 100) and increased collateral drop to $21.54\%$ without boosting cohort mean.

---

## 6. Empirical Visualizations & Figures

Visualizations produced from the benchmark suite:

* **Figure 1**: Backdoor Eradication Dynamics and Worst-Off Client Trajectories ([`figures/fig1_unlearning_dynamics.png`](figures/fig1_unlearning_dynamics.png))
* **Figure 2**: Per-Client Accuracy Distribution Across Retained Clients Showing Data-Scarcity Floor ([`figures/fig2_per_client_distribution.png`](figures/fig2_per_client_distribution.png))
* **Figure 3**: Step-Norm Capping Control and Empirical Pareto Trade-off Frontier ([`figures/fig3_hyperparameter_pareto.png`](figures/fig3_hyperparameter_pareto.png))
* **Figure 4**: Multi-Seed Macro Distributions Across Reporting Seeds ([`figures/fig4_multiseed_robustness.png`](figures/fig4_multiseed_robustness.png))
* **Figure 8**: Matched-Strength Utility Frontier vs. Unlearning Displacement ([`figures/fig_frontier_strength.png`](figures/fig_frontier_strength.png))

### Appendix: Exploratory & Archived Figures
* **Figure 6**: Batch Unlearning ($U_N=2$) Multi-Client Benchmark on Exploratory Seed 1 ([`figures/fig6_multiclient_benchmark.png`](figures/fig6_multiclient_benchmark.png))
* **Figure 7**: CIFAR-100 Complex Dataset Generalization Benchmark on Exploratory Seed 1 ([`figures/fig7_cifar100_generalization.png`](figures/fig7_cifar100_generalization.png))
* **Figure 5**: Archived Dynamic Weight Trajectories for Adaptive Variants demonstrating tight clustering near 1.0 ([`figures/fig5_weight_trajectories.png`](figures/fig5_weight_trajectories.png))

---

## 7. Limitations

1. **Training Regime Sensitivity**: Pretraining learning rate and round count significantly influence unlearning dynamics; results in R1 reflect an under-trained pretrain checkpoint relative to R2.
2. **Unlearning Strength Confounding**: One-sided updates unlearn less aggressively in terms of raw ASR ($0.94\%$ vs $0.13\%$), making strength-matched displacement controls essential for fair comparison.
3. **Exploratory Status of Seeds 11–20**: Benchmarks in Section 4 reflect exploratory exploration; formal confirmation requires fresh, previously unobserved seeds 21–30 per [`SUCCESS_CRITERIA_v2.md`](SUCCESS_CRITERIA_v2.md).
4. **Architectural Scope**: Current evaluations use standard CNN backbones on 10-client federated partitions.
5. **Backdoor Attack Formulation**: Targets are evaluated using pattern-based triggers rather than clean class removal.

---

## 8. Reproducibility & Execution Runbook

All runs embed commit hash, CLI arguments, and random seeds in their output JSON files.

### Full Pipeline Reproduction Script
To execute the complete pipeline across seeds in primary regime R2:

```bash
# Run complete pipeline on confirmation seeds
bash run_all.sh 21 22 23 24 25 26 27 28 29 30
```

### Individual Execution Commands

```bash
# 1. Pretraining Checkpoint (Primary Regime R2: lr=0.05, R=100)
python run_UnlearningTask.py --seed 21 --device 0 --module CNN_CIFAR10 --algorithm FedAvg \
  --dataloader DataLoader_cifar10_pat --N 10 --NC 2 --balance True --B 200 --C 1.0 \
  --R 100 --UR 100 --E 1 --decay 0.999 --step_type bgd --unlearn_cn 1 --save_model True \
  --lr 0.05 --unlearn_pretrain True

# 2. Retrain from Scratch (R=200 Reference for End of Unlearning)
python run_UnlearningTask.py --seed 21 --device 0 --module CNN_CIFAR10 --algorithm FedAvg \
  --dataloader DataLoader_cifar10_pat --N 10 --NC 2 --balance True --B 200 --C 1.0 \
  --R 200 --UR 0 --E 1 --decay 0.999 --step_type bgd --unlearn_cn 1 --save_model True \
  --lr 0.05 --retrain True --test_interval 5

# 3. Retrain from Scratch (R=300 Reference for Post-Recovery)
python run_UnlearningTask.py --seed 21 --device 0 --module CNN_CIFAR10 --algorithm FedAvg \
  --dataloader DataLoader_cifar10_pat --N 10 --NC 2 --balance True --B 200 --C 1.0 \
  --R 300 --UR 0 --E 1 --decay 0.999 --step_type bgd --unlearn_cn 1 --save_model True \
  --lr 0.05 --retrain True --test_interval 5

# 4. Hard FedOSD rescaled to ||g_u|| (r_lr=1e-4)
python run_UnlearningTask.py --seed 21 --device 0 --module CNN_CIFAR10 --algorithm FedOSD \
  --dataloader DataLoader_cifar10_pat --N 10 --NC 2 --balance True --B 200 --C 1.0 \
  --R 200 --UR 100 --E 1 --decay 0.999 --step_type bgd --unlearn_cn 1 --save_model True \
  --lr 0.0004 --r_lr 1e-4 --proj_mode hard --adap_lam 100000.0 --adap_weight_eta 0.0

# 5. Soft Uniform Baseline (lambda=20, r_lr=1e-4)
python run_UnlearningTask.py --seed 21 --device 0 --module CNN_CIFAR10 --algorithm FedOSD \
  --dataloader DataLoader_cifar10_pat --N 10 --NC 2 --balance True --B 200 --C 1.0 \
  --R 200 --UR 100 --E 1 --decay 0.999 --step_type bgd --unlearn_cn 1 --save_model True \
  --lr 0.0004 --r_lr 1e-4 --proj_mode soft --adap_lam 20.0 --adap_weight_eta 0.0

# 6. One-sided rescaled to ||g_u|| Hard (lambda=10^5, r_lr=1e-4)
python run_UnlearningTask.py --seed 21 --device 0 --module CNN_CIFAR10 --algorithm FedOSD \
  --dataloader DataLoader_cifar10_pat --N 10 --NC 2 --balance True --B 200 --C 1.0 \
  --R 200 --UR 100 --E 1 --decay 0.999 --step_type bgd --unlearn_cn 1 --save_model True \
  --lr 0.0004 --r_lr 1e-4 --proj_mode one_sided --adap_lam 100000.0 --adap_weight_eta 0.0

# 7. Step-Norm One-sided, unscaled (cap 1.0) Hard (cap=1.0, Frozen Method)
python run_UnlearningTask.py --seed 21 --device 0 --module CNN_CIFAR10 --algorithm FedOSD \
  --dataloader DataLoader_cifar10_pat --N 10 --NC 2 --balance True --B 200 --C 1.0 \
  --R 200 --UR 100 --E 1 --decay 0.999 --step_type bgd --unlearn_cn 1 --save_model True \
  --lr 0.0004 --r_lr 1e-4 --proj_mode one_sided --adap_lam 100000.0 --adap_weight_eta 0.0 \
  --weight_mode uniform --cap_hard_ratio 1.0
```

### Statistical Evaluation
```bash
python evaluate_reporting.py
python eval_matched_strength.py
python plot_results.py
```

---

## 9. Citation

```bibtex
@inproceedings{pan2025federated,
  title={Federated unlearning with gradient descent and conflict mitigation},
  author={Pan, Zhaomin and Wang, Zhibo and Li, Chunlin and others},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={39},
  number={19},
  pages={19804--19812},
  year={2025}
}
```
