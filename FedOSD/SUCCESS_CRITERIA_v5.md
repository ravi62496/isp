# SUCCESS_CRITERIA_v5

> **Protocol Note**: Committed and tagged (`freeze-v5`) prior to running or evaluating any confirmation seeds 21–30.

- **Confirmation Seeds**: 21–30 ($n=10$, sealed, strictly single sequential execution).
- **Primary Regime**: Regime R2 (Pretrain `lr=0.05`, decay=0.999, recovery `r_lr=0.0001`, `g_clip=1.0`).
- **Primary Stage**: End of unlearning (Round 100).
- **Secondary Stage**: Post-recovery (Round 200, separate Holm family).
- **Method Under Test**: Step-Norm Capped One-Sided Hard FedOSD (`cap=1.0`, `wm_uniform`, `lam=100000.0`).
- **Primary Comparator**: Hard-scaled at $s^* = 0.54$ (`proj_hard`, `step_scale=0.54`, `lam=100000.0`).
- **Secondary Comparator**: Closed-form baseline Hard FedOSD (`proj_hard`, `step_scale=1.0`, `lam=100000.0`).

---

## 1. Primary Metrics (Method Minus Comparator)
Oriented such that positive values indicate improvement:

| Primary Metric | Practical Threshold | Equivalence Margin |
| :--- | :---: | :---: |
| **1. Retained Cohort Mean Accuracy** | $\ge +1.0\%$ | $\pm 0.5\%$ |
| **2. Max Collateral Drop** (lower drop is better; $\text{Drop}_{\text{comp}} - \text{Drop}_{\text{method}}$) | $\ge +2.0\%$ | $\pm 1.0\%$ |
| **3. Worst-Off Client Floor** | $\ge +2.0\%$ | $\pm 1.0\%$ |

---

## 2. Secondary Metrics & Interpretation Rules
- **Secondary Metrics**: Mean of worst 2 clients, absolute distance to Retrain on ASR, forget-set clean accuracy, solver runtime per round.
- **Membership Inference Attack (MIA)**:
  - Empirical Retrain noise floor: $\sigma_{\text{MIA}} = 0.0048 \approx 0.005$.
  - Interpretation threshold: Differences $|\Delta \text{MIA}|$ are interpreted **only** when they exceed $2 \times \sigma_{\text{MIA}} \ge 0.010$.
- **Strength Matching Requirement**: Displacement ratio $\frac{\text{Displacement}(\text{Method})}{\text{Displacement}(\text{Comparator})}$ must fall within $[0.95, 1.05]$ on the seed average. If outside this bracket, report *'strength not matched'* and evaluate against the interpolated Hard-scaled sweep.

---

## 3. Statistical Testing & Decision Rules
- **Statistical Tests**: Paired $t$-test 95% confidence interval, 1,000-resample bootstrap 95% confidence interval, paired Wilcoxon signed-rank test $p$-value.
- **Multiple Comparisons**: Holm-Bonferroni correction over the 3 primary metrics within each stage family ($m=3$).
- **Classification Labels**:
  - **Superior**: 95% CI lower bound $> 0$, Holm $p < 0.05$, and point estimate $\ge$ practical threshold.
  - **Inferior**: 95% CI upper bound $< 0$ and Holm $p < 0.05$.
  - **Equivalent**: Two-sided 90% CI completely contained within the equivalence margin $[-\delta, +\delta]$.
  - **Better but Negligible**: 95% CI lower bound $> 0$, but point estimate $<$ practical threshold.
  - **Inconclusive**: Does not meet any above criterion.

---

## 4. Study Outcome Classifications
- **Outcome A**: Superior on retained mean and superior on drop or floor; no metric inferior.
- **Outcome B**: Every primary metric classified as Equivalent.
- **Outcome C**: Any other outcome (e.g., partial wins, negligible differences, or trade-offs).

---

## 5. Protocol Constraints
- Stopping rule: $n=10$ confirmation seeds (21–30). If fewer seeds finish, report the completed subset without modifying any thresholds.
- Confirmation runs must be executed in exact sequential order, results sealed, and analyzed once using the single frozen confirmation evaluator script.
