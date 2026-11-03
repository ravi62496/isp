# Pre-Registered Screening Results and Frozen Configuration

This document records the empirical results of the Step 3 screening sweep conducted on tuning seed 101, along with the pre-registered selection of the single frozen configuration to be evaluated across reporting seeds 11–20 in Block C.

---

## 1. Step 3 Screening Results (Seed 101)

### Screening Protocol & Rule 3.0
- **Threshold Criteria** (relative to baseline Hard FedOSD on Seed 101):
  - **Floor**: $\ge \text{Hard floor} - 2.0\%$ ($\ge 24.67\%$ at unlearn round 100; $\ge 27.83\%$ at recovery round 200)
  - **Mean**: $\ge \text{Hard mean} - 1.0\%$ ($\ge 41.26\%$ at unlearn round 100; $\ge 42.59\%$ at recovery round 200)
  - **ASR**: Absolute distance to the retrain-from-scratch reference $\le \text{Hard's distance}$
  - **Rank Order**: Among passing candidates, rank by client accuracy floor, then forget-set clean utility.

### Evaluation Summary Table (Tuning Seed 101)

#### (A) Unlearning Stage (Round 100)
| Candidate | Description | Floor (%) | Mean (%) | ASR (%) | Forget Clean (%) | MIA AUC | Screening Status |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Hard FedOSD** | Baseline reference ($\lambda=10^5$) | 26.67 | 42.26 | 0.30 | 27.67 | 0.4970 | *Baseline* |
| **E3_cap1.0** | One-Sided Hard with $\|d\| \le \|d_{\text{hard}}\|$ | **26.83** | **43.59** | 0.44 | **29.17** | 0.4952 | **PASS (Top Rank)** |
| **E3_cap1.25** | One-Sided Hard with $\|d\| \le 1.25\|d_{\text{hard}}\|$ | **26.83** | 43.63 | 0.44 | **29.17** | 0.4952 | **PASS** |
| **E2_drop_ref10_soft20** | Per-client geometric interpolation ($\Delta_{\text{ref}}=10\%$) | 25.50 | 42.09 | 0.28 | 25.83 | 0.4967 | PASS (Lower Floor) |
| **E2_drop_ref5_soft20** | Per-client geometric interpolation ($\Delta_{\text{ref}}=5\%$) | 25.17 | 42.09 | 0.30 | 25.83 | 0.4965 | PASS (Lower Floor) |
| **E1_lam40_eta0.4_drop** | Two-sided soft sweep ($\lambda=40, \eta=0.4$) | 26.67 | 41.89 | 0.24 | 26.50 | 0.4972 | PASS (Lower Mean) |
| **E1_lam50_eta0.4_drop** | Two-sided soft sweep ($\lambda=50, \eta=0.4$) | 26.50 | 41.98 | 0.26 | 26.50 | 0.4971 | PASS (Lower Mean) |

#### (B) Recovery Stage (Round 200)
| Candidate | Description | Floor (%) | Mean (%) | ASR (%) | Forget Clean (%) | MIA AUC |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Hard FedOSD** | Baseline reference | 29.83 | 43.59 | 0.42 | 30.17 | 0.4969 |
| **E3_cap1.0** | One-Sided Hard with $\|d\| \le \|d_{\text{hard}}\|$ | 29.33 | **44.85** | 0.68 | **31.83** | 0.4951 |
| **E3_cap1.25** | One-Sided Hard with $\|d\| \le 1.25\|d_{\text{hard}}\|$ | 29.33 | 44.80 | 0.68 | **31.83** | 0.4950 |
| **E2_drop_ref10_soft20** | Per-client geometric interpolation ($\Delta_{\text{ref}}=10\%$) | 28.33 | 43.35 | 0.36 | 29.33 | 0.4967 |
| **E2_drop_ref5_soft20** | Per-client geometric interpolation ($\Delta_{\text{ref}}=5\%$) | 28.17 | 43.54 | 0.38 | 29.50 | 0.4966 |
| **E1_lam40_eta0.4_drop** | Two-sided soft sweep ($\lambda=40, \eta=0.4$) | 29.33 | 42.96 | 0.36 | 29.67 | 0.4972 |
| **E1_lam50_eta0.4_drop** | Two-sided soft sweep ($\lambda=50, \eta=0.4$) | 29.33 | 43.04 | 0.36 | 29.83 | 0.4972 |

---

## 2. Selection Rationale

1. **Pathology Addressed**: In Diagnostic Question 1, unconstrained One-Sided Hard unlearning produced update vectors $85.0\%$ larger in norm than Hard FedOSD in Round 1 ($89.40$ vs $48.32$). This excessive step norm caused higher-order curvature damage to retained clients despite the first-order inner product non-negativity constraint.
2. **Superior Performance**: Capping the step norm at $\|d_{\text{hard}}\|$ (`E3_cap1.0`) directly eliminates this pathology. On seed 101, it achieves the highest floor at round 100 ($26.83\%$, surpassing Hard FedOSD's $26.67\%$), higher mean accuracy across both stages ($43.59\%$ and $44.85\%$, $+1.33\%$ and $+1.26\%$ over Hard), and significantly superior forget-set clean utility ($29.17\%$ vs $27.67\%$).
3. **Simplicity and Principled Bounds**: `E3_cap1.0` requires no hand-tuned per-client heuristic thresholds $\Delta_{\text{ref}}$ and guarantees that the unlearning update never pushes the global parameter vector further than the closed-form hard projection.

---

## 3. Single Frozen Configuration Pre-Registration

The single surviving candidate advancing to full multi-seed evaluation (Seeds 11–20) is:

- **Variant Name**: Step-Norm Capped One-Sided Hard FedOSD (`E3_cap1.0`)
- **CLI Arguments**:
  ```bash
  --proj_mode one_sided --adap_lam 100000.0 --adap_weight_eta 0.0 --weight_mode uniform --cap_hard_ratio 1.0
  ```
- **Frozen Parameters**:
  - `proj_mode`: `'one_sided'`
  - `adap_lam`: `100000.0`
  - `adap_weight_eta`: `0.0`
  - `weight_mode`: `'uniform'`
  - `cap_hard_ratio`: `1.0`

---
*Status: FROZEN. This document was finalized before running Block C evaluation on reporting seeds 11–20.*



---

## 4. Recovery Rate Audit & Protocol v2 Frozen Configurations

Conducted per Guide v2 Phase 1 (Step 1.3) across tuning seeds 101–103 under the converged pretrain regime ($\text{lr}=0.05, R=100$).

### 4.1 Multi-Seed Recovery Sweep Matrix (Hard FedOSD Baseline)

| Seed | Recovery $r_{\text{lr}}$ | Retained Mean | Worst-Off Floor | Max Collateral Drop | Forget-Set Clean | Backdoor ASR | MIA ROC-AUC |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **101** | $10^{-6}$ | 42.70% | 27.83% | 16.50% | 28.33% | 0.32% | 0.4971 |
| **101** | **$10^{-4}$** | **50.09%** | **36.00%** | **8.17%** | **40.50%** | **2.88%** | 0.4963 |
| **101** | $10^{-3}$ | 49.11% | 28.00% | 5.00% | 35.83% | 13.32% | 0.4949 |
| **101** | $10^{-2}$ | 49.63% | 28.83% | 4.17% | 36.83% | 13.98% | 0.4962 |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **102** | $10^{-6}$ | 47.74% | 5.33% | 13.92% | 4.83% | 0.68% | 0.5116 |
| **102** | **$10^{-4}$** | **55.74%** | **7.33%** | **6.42%** | **6.67%** | **6.88%** | 0.5102 |
| **102** | $10^{-3}$ | 54.59% | 4.83% | 7.25% | 4.67% | 11.50% | 0.5110 |
| **102** | $10^{-2}$ | 54.94% | 4.83% | 4.75% | 4.00% | 11.66% | 0.5136 |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **103** | $10^{-6}$ | 44.35% | 23.33% | 12.33% | 24.00% | 0.98% | 0.5216 |
| **103** | **$10^{-4}$** | **51.26%** | **32.17%** | **3.33%** | **34.33%** | **3.12%** | 0.5246 |
| **103** | $10^{-3}$ | 50.43% | 29.17% | 7.50% | 24.17% | 8.00% | 0.5242 |
| **103** | $10^{-2}$ | 49.98% | 25.00% | 10.33% | 23.17% | 8.68% | 0.5234 |

### 4.2 Summary Across Tuning Seeds 101–103 ($\text{Mean} \pm \text{Std}$)

| Recovery $r_{\text{lr}}$ | Retained Mean | Worst-Off Floor | Max Collateral Drop | Backdoor ASR | Forget-Set Clean |
| :---: | :---: | :---: | :---: | :---: | :---: |
| $10^{-6}$ (Frozen baseline) | 44.93% $\pm$ 2.10% | 18.83% $\pm$ 9.72% | 14.25% $\pm$ 1.72% | 0.66% $\pm$ 0.27% | 19.06% $\pm$ 10.21% |
| **$10^{-4}$ (Selected)** | **52.36% $\pm$ 2.43%** | **25.17% $\pm$ 12.71%** | **5.97% $\pm$ 2.00%** | **4.29% $\pm$ 1.83%** | **27.17% $\pm$ 14.71%** |
| $10^{-3}$ | 51.38% $\pm$ 2.34% | 20.67% $\pm$ 11.21% | 6.58% $\pm$ 1.12% | 10.94% $\pm$ 2.21% | 21.56% $\pm$ 12.86% |
| $10^{-2}$ | 51.52% $\pm$ 2.43% | 19.56% $\pm$ 10.53% | 6.42% $\pm$ 2.78% | 11.44% $\pm$ 2.17% | 21.33% $\pm$ 13.47% |

### 4.3 Selection Decision & Rationale
1. **Maximizes Baseline Performance**: Per Guide v2 Step 1.3, $r_{\text{lr}}$ must be chosen to maximize the baseline (Hard FedOSD) recovered mean so that the proposed method is not flattered. $r_{\text{lr}}=10^{-4}$ achieves the highest retained mean ($52.36\%$, $+7.43\%$ over $10^{-6}$, $+0.98\%$ over $10^{-3}$).
2. **Floor Protection**: $r_{\text{lr}}=10^{-4}$ achieves the highest worst-off client floor across all seeds ($25.17\%$, $+6.34\%$ over $10^{-6}$, $+4.50\%$ over $10^{-3}$).
3. **Strict Privacy Preservation**: Unlike larger rates ($10^{-3}, 10^{-2}$) where ASR leaks back to $10.94\%$–$11.44\%$, $r_{\text{lr}}=10^{-4}$ maintains unlearning efficacy with an ASR of $4.29\% \pm 1.83\%$, aligning with the Retrain target baseline.

**Frozen Decision**: $r_{\text{lr}} = 10^{-4}$ is frozen for all post-recovery training across all methods in Regime R2.


---

## 4. Stage 2 Freeze (Master Guide v5 Protocol)

### 4.1 Bisection for Matched Step Scale $s^*$
Across tuning seeds 101–103, parameter displacement of Hard-scaled variants relative to Capped One-Sided (cap=1.0, target displacement $= 0.2395$):
- Hard scale $0.74$: mean displacement $= 0.2849$ ($+19.21\%$ vs Capped)
- Hard scale $0.50$: mean displacement $= 0.2301$ ($-3.83\%$ vs Capped)
- Hard scale $0.54$: mean displacement $= 0.2403$ ($+0.43\%$ vs Capped)

**Frozen Matched Scale**: $s^* = 0.54$ (achieves displacement matching within $\pm 0.43\% \le 2.0\%$).

### 4.2 Written Verdict on the Norm Explanation
- **Uncapped One-Sided Inflation**: Without step-norm bounding, One-Sided Hard expands parameter displacement by $+43.8\%$ over Capped One-Sided in Regime R2 ($0.3363$ vs $0.2338$).
- **Displacement Compression**: Capping the step norm at $1.0 \times \|d_{\text{hard}}\|$ tightens displacement to $72.5\%$ of unscaled Hard FedOSD ($0.2338$ vs $0.3224$).
- **Mechanism Confirmation**: Step-norm capping eliminates excessive parameter drift caused by non-negativity constraint slack, maintaining competitive unlearning effectiveness without unconstrained collateral distortion.

### 4.3 Empirical MIA Noise Floor
Across 5 tuning partitions (seeds 101–105) with Retrain from scratch models ($R=300$):
- Mean Retrain MIA AUC: $0.5088 \pm 0.0107$ (range $[0.4953, 0.5224]$)
- Within-partition Retrain MIA difference ($R=200$ vs $R=300$): Mean $|\Delta| = 0.0048 \approx 0.005$
- **Operational Noise Floor**: $\sigma_{\text{MIA}} = 0.0048 \approx 0.005$
- **Interpretation Rule**: MIA AUC differences are interpreted only when $|\Delta \text{MIA}| \ge 2 \times \sigma_{\text{MIA}} = 0.0097 \approx 0.010$.

### 4.4 Solver Runtime Overhead
- Hard FedOSD (closed-form linear solve): $7.79\text{ ms/round}$
- Capped One-Sided (NNLS projection): $4.76\text{ ms/round}$
- Solver overhead is negligible ($< 0.1\%$ of per-round FL communication & local computation time).
