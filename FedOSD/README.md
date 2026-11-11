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
2. **Orthogonal Complement vs. One-Sided Relaxation**: Standard Hard FedOSD forces $g_i^T d = 0$ symmetrically for all retained clients. One-sided FedOSD relaxes this constraint to $\hat{g}_i^T d \ge 0$. While theoretically intended to permit descent directions aligned with retained client gradients, empirical evaluations show that relaxing orthogonality does *not* consistently protect retained or worst-off clients (on multi-seed audits at round 100, the worst-off floor drops relative to hard projection).
3. **Gradient Rescaling vs. Pure Projection**: In the legacy codebase, unconstrained descent directions were subjected to norm rescaling: $d = \frac{d}{\|d_{\text{raw}}\|} \cdot \|g_u\|$, dilating descent steps up to the unlearned client's local gradient norm $\|g_u\|$ (at Round 1, $\|g_u\| \approx 27.84$ vs $\|d_{\text{raw}}\| \approx 20.81$, a $+33.8\%$ step enlargement). Legacy Hard FedOSD was similarly rescaled to $\|g_u\|$. This dilation was associated with excessive parameter displacement and collateral degradation (exploratory evidence) on retained clients.

### Method Under Test: One-Sided, Unscaled Projection (cap 1.0)
To investigate the one-sided relaxation ($\hat{g}_i^T d \ge 0$) without artificial $\|g_u\|$ rescaling, **One-Sided Unscaled FedOSD (cap 1.0)** solves the one-sided projection via dual Non-Negative Least Squares (NNLS) and caps the update vector norm at $\|d_{\text{hard}}\|$:

$$d = \begin{cases} d_{\text{raw}} & \text{if } \|d_{\text{raw}}\| \le \text{cap} \cdot \|d_{\text{hard}}\| \\ \text{cap} \cdot \frac{\|d_{\text{hard}}\|}{\|d_{\text{raw}}\|} d_{\text{raw}} & \text{otherwise} \end{cases}$$

The frozen primary configuration sets $\text{cap} = 1.0$ with uniform weighting ($\eta = 0$).

---

## 2. Mathematical Formulations

### 2.1 Symmetric Closed-Form Projection (Hard Baseline, $\lambda = \infty$)
Let $\hat{A} \in \mathbb{R}^{n \times p}$ denote the matrix whose rows are unit-normalized retained client gradients $\hat{g}_i = g_i / \|g_i\|$. The closed-form orthogonal projection is:

$$d_{\text{hard}} = g_u - \hat{A}^T (\hat{A} \hat{A}^T)^{-1} \hat{A} g_u$$

### 2.2 One-Sided Non-Harm Projection (Dual NNLS Solve)
The one-sided projection formulation is:

$$\min_d \frac{1}{2}\|d - g_u\|^2 + \frac{1}{2} \sum_{i \in \mathcal{C}_{\text{retained}}} \lambda_i \max(0, -\hat{g}_i^T d)^2$$

Introducing slack variables $s \ge 0$ with $-\hat{A} d \le s$, the Lagrange dual problem reduces to:

$$\min_{\nu \ge 0} \nu^T (G + D) \nu + 2 \nu^T b \quad \text{where} \quad G = \hat{A} \hat{A}^T, \quad b = \hat{A} g_u, \quad D = \text{diag}\left(\frac{1}{\lambda_i}\right)$$

With $\lambda = 10^5$, Cholesky factorization $M = G + D + 10^{-10} I = L L^T$ and $y = -L^{-1} b$ enables solution via Non-Negative Least Squares (NNLS):

$$\min_{\nu \ge 0} \|L^T \nu - y\|^2 \implies d_{\text{raw}} = g_u + \hat{A}^T \nu$$

### 2.3 Step-Norm Bounding
When $\|d_{\text{raw}}\| > \|d_{\text{hard}}\|$, the update is scaled back to match $\|d_{\text{hard}}\|$ exactly ($\text{cap} = 1.0$), ensuring parameter displacement does not exceed the hard boundary norm.

---

## 3. Training Regimes & Decision Gates

### 3.1 Training Regimes & Gate A Decision (Tuning Seeds 101–103)
To eliminate confounds from inconsistent learning rates and under-trained pretraining models, regimes were audited across tuning seeds 101–103 ($n=3$):

| Regime | Pretrain lr | Retrain lr | Recovery $r_{\text{lr}}$ | Purpose & Characteristics | Retrain Mean (Regime R2, Seeds 101–103) |
| :--- | :---: | :---: | :---: | :--- | :---: |
| **R1 (Historical / Exploratory)** | 0.01 | 0.05 | $10^{-6}$ | Historical setting (Seeds 11–20); under-trained pretrain | ~59.4% (R=200) |
| **R1b (Control)** | 0.01 | 0.01 | $10^{-6}$ | Controls for lr disparity at pretrain lr | 44.69% $\pm$ 2.45% |
| **R2 (Primary Matched)** | 0.05 | 0.05 | $10^{-4}$ | Matched lr across pipeline; converged pretrain | **58.10% $\pm$ 2.25%** |

* **Gate A Resolution**: Primary regime is **R2**. The R1b control demonstrates that the learning rate disparity ($0.01$ vs $0.05$) alone accounts for **$13.41$ percentage points** (44.7%) of the historical 30-point Retrain gap in R1. Under primary regime R2, the true gap between Hard FedOSD ($52.36\%$) and Retrain ($58.10\%$) is only **$5.74$ percentage points** on tuning seeds.
* **Recovery Rate Freeze**: A systematic sweep of $r_{\text{lr}} \in \{10^{-6}, 10^{-4}, 10^{-3}, 10^{-2}\}$ established that $r_{\text{lr}} = 10^{-4}$ maximizes baseline Hard recovered mean ($52.36\% \pm 2.43\%$ vs $44.93\%$ at $10^{-6}$) while keeping backdoor ASR under $4.3\%$. It is frozen in [`FROZEN_CONFIGS_v5.md`](FROZEN_CONFIGS_v5.md).

### 3.2 Gradient Rescaling Finding & Cap Sweep (Regime R2, Tuning Seeds 101–103)
Auditing projection mechanics in `tfedplat/algorithm/unlearning/FedOSD.py` (lines 371–380) revealed the operational difference between unscaled projection and legacy unconstrained descent:
- **One-Sided, Unscaled (cap 1.0)**: Traverses the `cap_hard_ratio > 0` code branch. When $\|d_{\text{raw}}\| > \|d_{\text{hard}}\|$, it clamps the norm to $\|d_{\text{hard}}\|$; for cap $\ge 2.0$ (or $10^9$), it outputs the pure unconstrained NNLS projection $d_{\text{raw}}$ without any dilation.
- **One-Sided Rescaled to $\|g_u\|$**: Traverses the legacy `elif d_raw_norm > 1e-8` branch ($d = \frac{d}{\|d_{\text{raw}}\|} \cdot \|g_u\|$), inflating the vector norm up to $\|g_u\|$.
- **Hard FedOSD Rescaled to $\|g_u\|$**: Similarly hits the legacy branch, rescaling $d_{\text{hard}}$ to $\|g_u\|$.

#### Table 3.1: Cap Ratio Sweep Matrix (Regime R2, Post-Recovery Round 200, Tuning Seeds 101–103)
| Cap Ratio | Retained Mean | Worst-Off Floor | Max Collateral Drop | Forget-Set Clean | Backdoor ASR | Displacement $\|w_{\text{final}} - w_{\text{pre}}\|$ |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **0.50** | 52.05% | 23.83% | 5.94% | 26.61% | 8.05% | 0.1697 |
| **0.75** | 52.35% | 24.33% | 5.61% | 26.83% | 6.69% | 0.2098 |
| **1.00** | **52.48%** | **24.83%** | **5.44%** | 27.00% | 5.98% | 0.2395 |
| **1.25** | 52.47% | 24.83% | 5.44% | 27.06% | 5.96% | 0.2421 |
| **1.50** | 52.47% | 24.83% | 5.44% | 27.06% | 5.95% | 0.2422 |

#### Table 3.2: Matched-Strength Control Comparison (Regime R2, Post-Recovery Round 200, Tuning Seeds 101–103)
To verify that effects are evaluated at matched displacement rather than differing step sizes, One-sided, unscaled (cap 1.0) was compared against a Hard-scaled control ($s^*=0.54$) at matched parameter displacement $\|w_{\text{final}} - w_{\text{pre}}\|$:

| Metric at Matched Displacement | Paired Difference (Capped vs Hard-Scaled) | 95% Bootstrap CI | Sign Agreement |
| :--- | :---: | :---: | :---: |
| **Retained Client Mean Utility** | **+0.24%** | [+0.17%, +0.32%] | 3/3 positive |
| **Worst-Off Client Floor** | **+0.38%** | [+0.06%, +0.62%] | 3/3 positive |
| **Max Collateral Drop** | **-0.21%** | [-0.63%, +0.17%] | 2/3 negative |

*Note: Table 3.2 is exploratory ($n=3$, round 200, tuning seeds 101–103). The pre-registered confirmatory round-100 result on fresh seeds 21–30 ($n=10$) is reported in the new [Main result](#main-result) section below and points the other way on the worst-off floor (-5.47 points).*

---

## 4. Empirical Evaluation Protocol & Benchmarks (Historical Regime R1, Exploratory Seeds 11–20)

### Experimental Timeline & Stage Matching
| Stage | Communication Rounds | Description | Matched Retrain Reference |
| :--- | :---: | :--- | :--- |
| **Pretrain** | Rounds 1 to 100 | Initial federated pretraining on full cohort | — |
| **Unlearning** | Rounds 1 to 100 | Unlearning updates ($UR=100$) | Retrain from scratch at $R=200$ |
| **Recovery** | Rounds 101 to 200 | Post-unlearning federated recovery training | Retrain from scratch at $R=300$ |

Evaluation metrics below reflect exploratory reporting seeds 11–20 ($N=10$, regime R1).

---

### Table 4.1: End of Unlearning Stage (Regime R1, End of Unlearning Round 100, Exploratory Seeds 11–20)
| Method / Configuration | Proj. Mode | Seeds | Backdoor ASR (Target: 3.03%) | Retained Mean $\pm$ Std Dev | Worst-Off Floor (Min) $\uparrow$ | Max Collateral Drop $\downarrow$ | Forget Clean Utility $\uparrow$ | MIA ROC-AUC (Target: 0.5055) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Retrain Baseline ($R=200$)** | Scratch | 11–20 | 3.03% | **59.42% $\pm$ 14.72%** | **28.62%** | **-8.38%** | **29.18%** | 0.5055 |
| **Hard FedOSD ($\lambda=\infty$)** | Closed Hard | 11–20 | 0.02% | 29.09% $\pm$ 14.90% | 2.10% | 18.32% | 1.42% | 0.5143 |
| **Soft Uniform ($\lambda=20, \eta=0$)** | Two-Sided | 11–20 | 0.00% | 29.08% $\pm$ 17.62% | 1.43% | 21.54% | 1.08% | **0.5064** |
| **One-sided rescaled to ||g_u||** | One-Sided | 11–20 | 0.03% | 30.67% $\pm$ 17.19% | 0.82% | 22.80% | 0.42% | 0.5067 |
| **Step-Norm Capped ($\text{cap}=1.0$)** | One-Sided | 11–20 | **0.37%** | **32.51% $\pm$ 14.64%** | **2.42%** | **14.68%** | **1.82%** | 0.5105 |

---

### Table 4.2: Post-Recovery Stage (Regime R1, Post-Recovery Round 200, Exploratory Seeds 11–20)
| Method / Configuration | Proj. Mode | Seeds | Backdoor ASR (Target: 3.03%) | Retained Mean $\pm$ Std Dev | Worst-Off Floor (Min) $\uparrow$ | Max Collateral Drop $\downarrow$ | Forget Clean Utility $\uparrow$ | MIA ROC-AUC (Target: 0.5055) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Retrain Baseline ($R=300$)** | Scratch | 11–20 | 3.03% | **61.85% $\pm$ 13.90%** | **33.10%** | **-9.15%** | **31.40%** | 0.5055 |
| **Hard FedOSD ($\lambda=\infty$)** | Closed Hard | 11–20 | 0.13% | 31.85% $\pm$ 14.85% | 3.83% | 14.76% | 3.35% | 0.5143 |
| **Soft Uniform ($\lambda=20, \eta=0$)** | Two-Sided | 11–20 | 0.05% | 31.44% $\pm$ 17.40% | 3.85% | 15.49% | 3.37% | **0.5064** |
| **One-sided rescaled to ||g_u||** | One-Sided | 11–20 | 0.17% | 33.07% $\pm$ 17.02% | 2.72% | 14.72% | 2.20% | 0.5067 |
| **Step-Norm Capped ($\text{cap}=1.0$)** | One-Sided | 11–20 | **0.94%** | **33.88% $\pm$ 14.50%** | **4.23%** | **11.21%** | **3.87%** | 0.5105 |

---

### Table 4.3: Paired Statistical Comparisons with Explicit Holm Families (Regime R1, End of Unlearning Round 100, Exploratory Seeds 11–20)
All p-values corrected within predefined family scopes:

```text
Comparison / Test Family                  | Metric                | Paired Diff | 95% Bootstrap CI   | Raw p   | Holm p  | Signs
--------------------------------------------------------------------------------------------------------------------------------------
[Exploratory Family 1: Capped vs Hard (m=3 primary metrics)]
One-sided, unscaled (cap 1.0) vs Hard FedOSD           | Retained Cohort Mean  | +2.04%      | [+1.81%, +2.25%]   | 0.0020  | 0.0060  | 10/10 (+)
One-sided, unscaled (cap 1.0) vs Hard FedOSD           | Max Collateral Drop   | -3.55%      | [-5.58%, -1.48%]   | 0.0137  | 0.0274  | 9/10 (-)
One-sided, unscaled (cap 1.0) vs Hard FedOSD           | Worst-Off Floor (Min) | +0.40%      | [-0.30%, +1.37%]   | 0.6523  | 0.6523  | 6/10 (+)
(Secondary diagnostic)                    | Backdoor ASR          | +0.81%      | [+0.50%, +1.18%]   | 0.0020  | --      | 10/10 (+)
--------------------------------------------------------------------------------------------------------------------------------------
[Exploratory Family 2: Capped vs One-sided rescaled to ||g_u|| (m=2 metrics)]
One-sided, unscaled (cap 1.0) vs Unconstrained         | Worst-Off Floor (Min) | +1.51%      | [+0.62%, +2.45%]   | 0.0039  | 0.0078  | 9/10 (+)
One-sided, unscaled (cap 1.0) vs Unconstrained         | Retained Cohort Mean  | +0.81%      | [+0.35%, +1.28%]   | 0.0039  | 0.0078  | 9/10 (+)
--------------------------------------------------------------------------------------------------------------------------------------
[Exploratory Family 3: Unconstrained vs Hard FedOSD (m=2 metrics)]
One-sided rescaled to ||g_u|| vs Hard FedOSD    | Worst-Off Floor (Min) | -1.12%      | [-1.80%, -0.53%]   | 0.0039  | 0.0040  | 1/10 (+)
One-sided rescaled to ||g_u|| vs Hard FedOSD    | Retained Cohort Mean  | +1.23%      | [+0.89%, +1.59%]   | 0.0020  | 0.0040  | 10/10 (+)
```

---

## 5. Negative Results & Screening Audit

### 5.1 Dynamic Client-Adaptive Weighting ($\eta > 0$)
Variants attempting to dynamically scale client protection penalties based on local metrics were evaluated and archived:
1. **Deficit-Driven (`--weight_mode deficit`)**, **Baseline Drop-Driven (`--weight_mode drop`)**, and **Gradient Conflict (`--weight_mode conflict`)**: On exploratory seeds 1–2, adaptive weighting underperformed Hard FedOSD across worst-off floor, cohort mean, and collateral drop, while weights stayed within $\sim 1.0$ point of uniform.
2. **Validation Split Leakage Control**: Prior experiments using test-set accuracy feedback artificially inflated floor estimates from $\approx 9\%$ to $\approx 42\%$. Enforcing strict evaluation on held-out validation data removed this artifact. Adaptive variants provided no measurable advantage over uniform weights once test feedback was eliminated.

### 5.2 Screening Matrix for Initial Variants (Regime R2, End of Unlearning Round 100, Tuning Seed 101)
Below are the screening evaluations conducted on tuning seed 101 under primary regime R2 (converged pretraining `lr=0.05`, unlearning rounds $UR=100$):

#### Table 5.1: Initial Screening Matrix (Regime R2, End of Unlearning Round 100, Tuning Seed 101)
| Variant | Formulation / Knobs | Floor (%) | Mean (%) | ASR (%) | Forget Clean (%) | MIA AUC | Screening Outcome |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Hard FedOSD** | Baseline reference ($\lambda=\infty$) | 26.67% | 42.26% | 0.30% | 27.67% | 0.4970 | *Baseline Reference* |
| **E1 ($\lambda=40, \eta=0.4$)** | Two-sided soft sweep (`wm_drop`) | 26.67% | 41.89% | 0.24% | 26.50% | 0.4972 | Underperformed Hard Mean |
| **E1 ($\lambda=50, \eta=0.4$)** | Two-sided soft sweep (`wm_drop`) | 26.50% | 41.98% | 0.26% | 26.50% | 0.4971 | Underperformed Hard Mean |
| **E2 ($\Delta_{\text{ref}}=10\%$)** | Per-client interpolation (`soft20`) | 25.50% | 42.09% | 0.28% | 25.83% | 0.4967 | Floor drop (-1.17%) |
| **E2 ($\Delta_{\text{ref}}=5\%$)** | Per-client interpolation (`soft20`) | 25.17% | 42.09% | 0.30% | 25.83% | 0.4965 | Floor drop (-1.50%) |
| **E3_cap1.0** | Step-Norm One-sided, unscaled (cap 1.0) | **26.83%** | **43.59%** | 0.44% | **29.17%** | 0.4952 | Selected for multi-seed tuning |

* **Screening Audit Note**: On seed 101 alone, E3_cap1.0 exhibited a small numerical advantage on floor ($26.83\%$ vs $26.67\%$). However, comprehensive multi-seed evaluation across tuning seeds 101–105 shows that this single-seed floor advantage does *not* generalize: on tuning seeds 101–105, One-sided unscaled (cap 1.0) exhibits a paired floor drop of $-3.93$ pp relative to Hard-scaled at round 100 [exploratory: $n=5$, unadjusted $p=0.005$, Holm $p=0.015$]. One-sided projection does not guarantee protection for worst-off clients.

---

## Main result

Pre-registered outcome: C (trade-off).

At round 100, A3 (one-sided, capped) has higher retained mean accuracy than A2 (hard projection, step scale 0.54, strength-matched) by +1.15 points, but a lower worst-off floor (-5.47) and a larger max collateral drop (-3.37). At round 200 the mean difference is +0.31 (labeled Equivalent). The method does not protect the worst-off client.

### Claim Table

| Claim | Supporting table row | Evaluator label |
|---|---|---|
| mean +1.15 at R100 | F1_Confirmatory_A3_vs_A2_Rd100: Retained Cohort Mean Acc | Superior |
| floor -5.47 | F1_Confirmatory_A3_vs_A2_Rd100: Worst-Off Floor (Min Acc) | Inferior |
| max drop -3.37 | F1_Confirmatory_A3_vs_A2_Rd100: Max Collateral Drop | Inferior |
| mean equivalent at R200 | F2_Secondary_A3_vs_A2_Rd200: Retained Cohort Mean Acc | Equivalent |

## Strength matching and sample size

| Family | n | Strength matched | Displacement ratio | abs ASR diff (pt) |
|---|---|---|---|---|
| F1_Confirmatory_A3_vs_A2_Rd100 | 10 | True | 1.0119 | 0.09 |
| F2_Secondary_A3_vs_A2_Rd200 | 10 | True | 1.0114 | 0.43 |
| Exploratory_A3_vs_A4_Rd100 | 10 | False | 0.7075 | 0.64 |
| Exploratory_A3_vs_A1_Rd100 | 10 | False | 0.7327 | 0.52 |
| Sensitivity_Control_A1_vs_A2_Rd100 | 10 | False | 1.3810 | 0.61 |

## F1 (confirmatory): A3 vs A2 at round 100

| Metric | Method (A3) | Comparator (A2) | Diff (95% CI, t) | p (Holm) | Label |
|---|---|---|---|---|---|
| Retained Cohort Mean Acc | 47.89 | 46.75 | +1.15 [+0.72, +1.58] | 0.0004 | Superior |
| Worst-Off Floor (Min Acc) | 19.93 | 25.40 | -5.47 [-7.10, -3.84] | 0.0001 | Inferior |
| Max Collateral Drop | 14.00 | 10.63 | -3.37 [-6.19, -0.54] | 0.0245 | Inferior |

Source file: `table_F1_Confirmatory_A3_vs_A2_Rd100.csv`

## F2 (secondary, pre-registered): A3 vs A2 at round 200

| Metric | Method (A3) | Comparator (A2) | Diff (95% CI, t) | p (Holm) | Label |
|---|---|---|---|---|---|
| Retained Cohort Mean Acc | 52.39 | 52.08 | +0.31 [+0.17, +0.45] | 0.0015 | Equivalent |
| Worst-Off Floor (Min Acc) | 27.50 | 26.50 | +1.00 [+0.62, +1.38] | 0.0006 | Better but negligible |
| Max Collateral Drop | 5.05 | 5.57 | +0.52 [+0.09, +0.95] | 0.0236 | Equivalent |

Source file: `table_F2_Secondary_A3_vs_A2_Rd200.csv`

## Sensitivity control (exploratory): A1 vs A2 at round 100

| Metric | Method (A1) | Comparator (A2) | Diff (95% CI, t) | p (Holm) | Label |
|---|---|---|---|---|---|
| Retained Cohort Mean Acc | 45.33 | 46.75 | -1.41 [-1.70, -1.13] | 0.0000 | Inferior |
| Worst-Off Floor (Min Acc) | 23.82 | 25.40 | -1.58 [-2.42, -0.75] | 0.0020 | Worse but negligible |
| Max Collateral Drop | 12.88 | 10.63 | -2.25 [-3.21, -1.29] | 0.0010 | Inferior |

Source file: `table_Sensitivity_Control_A1_vs_A2_Rd100.csv`

## Pre-registered outcome

**Outcome C** - Outcome C: Inferior metric present in F1; fewer than 2 Superior in F1 and F2 not fully Equivalent.

Source file: `eval_summary_R2_seeds_21_22_23_24_25_26_27_28_29_30.json`

### Deviations and disclosures

- **Baseline A2**: baseline A2 = hard projection with step_scale 0.54 chosen to match A3 displacement (in `tfedplat/algorithm/unlearning/FedOSD.py` and the A2 command line in `AMENDMENT_1.md`, A2 falls through to the legacy rescaling branch that rescales to the unlearning gradient norm $\|g_u\|$, scaled by step_scale 0.54).
- **Recovery**: recovery is plain FedAvg.
- **Pretraining rounds**: 100 rounds ($R=100$) were run and evaluated in code/README; "paper used 2000" is UNVERIFIED.
- **Pre-registration freeze**: rules frozen in git tag freeze-v5.1.
- **Amendment**: one amendment (`AMENDMENT_1.md`) changed only the config_hash column the evaluator aborted before reading any metric, and I cannot confirm whether I saw any metrics before the amendment.
- **Table script**: `make_readme_tables.py` was edited after the run to ignore `*_exploratory.csv`, changing no numbers.
- **Exploratory comparisons**: exploratory comparisons (A3 vs A4, A3 vs A1, A1 vs A2) are not strength-matched and not confirmatory.

### Not tested

- Other datasets/model types.
- The paper's post-training projection.
- Multiple hardware setups.

---

## 6. Limitations

1. **Training Regime Sensitivity**: Pretraining learning rate and round count significantly influence unlearning dynamics; results in R1 reflect an under-trained pretrain checkpoint relative to R2.
2. **No Claim of Worst-Off Client Protection**: Empirical results demonstrate that one-sided non-harm constraints do not protect the worst-off clients or prevent collateral floor degradation relative to hard projection.
3. **Exploratory Status of Seeds 11–20 and 101–105**: All benchmark findings on seeds 11–20 and tuning seeds 101–105 are strictly exploratory. Confirmatory claims are governed by [`SUCCESS_CRITERIA_v5.1.md`](SUCCESS_CRITERIA_v5.1.md) on fresh seeds 21–30.
4. **Architectural Scope**: Current evaluations use standard CNN backbones on 10-client federated partitions.
5. **Backdoor Attack Formulation**: Targets are evaluated using pattern-based triggers rather than clean class removal.

---

## 7. Execution Runbook

```bash
# Evaluate Stage 4 confirmatory or dry-run protocol:
python evaluate_confirmation.py --regime R2 --seeds 101 102 103 104 105

# Generate README confirmation tables:
python make_readme_tables.py results_confirm README_confirmation_tables.md
```

---

## 8. Citation

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
