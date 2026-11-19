# Federated Unlearning Conflict Mitigation (ISP Project Workspace)

## Final project: Federated Unlearning for Vision Models with Auditing & Visualization

**Goal (from the brief):** simulate a client's "right to be forgotten" request in federated
learning, unlearn that client, and show with visual and quantitative evidence what the model
forgot.

**What we did.**
- **FL pipeline:** FedAvg on CIFAR-10 with 10 clients, 2 classes each (non-IID), and a small CNN.
- **Forget request:** one client, chosen at random by the seed, plants a backdoor during
  training: a 4x4 white patch that turns any image into a target class. It then asks to be
  forgotten. The backdoor works as that client's fingerprint: if the client is forgotten, the
  patch must stop working.
- **Unlearning:** plain gradient ascent, FedOSD (Pan et al., AAAI-25) and **our Soft FedOSD**
  (closed-form soft projection with optional adaptive client weights), against retraining from
  scratch as the gold standard. Recovery variants: E3 (plain post-training) and E4 (projected).
- **Audits:** backdoor attack success rate (ASR), membership inference (MIA), Grad-CAM
  (trigger energy, SSIM vs retrain), L2/KL distance to retrain, and a relearning test.
- **Packaging:** Gradio demo ([guide](audit_code/demo/DEMO_GUIDE.md)), results notebook
  ([notebooks/results.ipynb](notebooks/results.ipynb)) and slides ([presentation/](presentation/)).
- **No data leakage:** every choice was made on validation data of tuning seeds 101-103 by a
  pre-registered rule; the final seeds 31-33 were run once.

**Final results (CIFAR-10, seeds 31-33, test split, mean +/- std)**

| Model | Clean acc | Backdoor ASR | Forget-client held-out acc | Worst retained client | MIA AUC |
|---|---|---|---|---|---|
| Original (before forgetting) | 49.3 +/- 0.4 | 84.3 +/- 0.4 | 48.4 +/- 5.9 | n/a | 0.496 |
| Retrain (gold standard) | 59.6 +/- 0.1 | 4.6 +/- 1.4 | 53.2 +/- 2.5 | 45.3 +/- 3.2 | 0.493 |
| FedOSD (hard) | 42.7 +/- 1.0 | 1.8 +/- 1.3 | 39.2 +/- 5.3 | 23.6 +/- 2.8 | 0.504 |
| **Soft FedOSD (ours)** | 42.2 +/- 1.1 | 1.8 +/- 1.3 | 37.8 +/- 5.7 | 23.2 +/- 3.4 | 0.505 |
| Plain gradient ascent | 34.3 +/- 2.1 | 0.3 +/- 0.2 | 2.8 +/- 2.2 | 2.6 +/- 1.9 | 0.513 |
| E3: post-training, plain | 51.6 +/- 0.4 | 6.6 +/- 2.8 | 45.6 +/- 3.6 | 33.4 +/- 2.0 | 0.497 |
| E4: post-training, projected | 51.6 +/- 0.5 | 6.4 +/- 2.8 | 45.6 +/- 3.6 | 33.3 +/- 1.9 | 0.497 |

**Key findings**
1. **Forgetting works.** All unlearning methods cut the backdoor ASR from 84% to under 2%.
   Post-training restores clean accuracy to 51.6% with ASR near retrain's.
2. **Our Soft FedOSD matches hard FedOSD but does not beat it.** At the selected lambda = 100
   the soft solve approaches the hard projection. Adaptive client weights gave +2-3 pp to the
   worst client on the tuning seeds but failed the pre-registered rule and did not replicate on
   the final seeds. This is reported as a null result.
3. **Projection matters during unlearning, not during recovery.** Plain gradient ascent
   over-forgets: forget-client held-out accuracy falls to 2.8% and the worst client to 2.6%.
   E3 and E4 are equal.
4. **MIA cannot certify forgetting here.** Its AUC is about 0.50 even before unlearning,
   because the original model never overfit the forget client.
5. **Grad-CAM:** the original model looks at the trigger patch; retrain looks at the object.
   After unlearning the backdoor prediction is gone, but part of the CAM still falls in the
   patch.
6. **Relearning:** even retrain learns the patch from 1-2 steps of attack. A pre-registered
   calibration on the tuning seeds found no attack strength that a clean model resists, so
   relearning cannot certify forgetting for this trigger.

**Full write-up, with all tables, disclosures and deviations:**
[FedOSD/RESULTS_SOFT_FEDOSD.md](FedOSD/RESULTS_SOFT_FEDOSD.md).
**Pre-registrations:** [heterogeneity test](FedOSD/PREREG_SOFT_HETEROGENEITY.md) and
[relearning calibration](FedOSD/PREREG_RELEARN.md).

**Reproduce** (GPU server, from `FedOSD/`):
```bash
python run_soft_sweep.py --gpus 0 1 2 3   # tuning sweep, seeds 101-103 (validation only)
python analyze_sweep.py                   # pre-registered decision -> results_soft/sweep_decision.json
python run_final.py --gpus 0 1 2 3        # final seeds 31-33: all methods, bundles, audit
cd ../audit_code
python -m audit.relearn --bundle ../bundles/seed31 --out ../results_final/seed31   # per seed
python -m audit.aggregate --ablation 100.0:1.0                                      # final tables
python -m demo.app --bundle ../bundles/seed31 --results ../results_final/seed31    # demo
```

**Deviations from the brief:**
- CIFAR-10 instead of the suggested CelebA or CIFAR-100, and a small CNN instead of ResNet-18,
  both to fit the compute budget.
- The earlier exploratory results further down this README used a trigger that was later
  found never to be learned; see the correction note in [FedOSD/README.md](FedOSD/README.md).

---

### Earlier work in this repository

This repository contains the complete experimental code, statistical evaluators, pre-registration protocols, and publication-grade artifacts for the empirical evaluation of federated unlearning conflict mitigation algorithms (based on Pan et al., AAAI-25).

The primary research codebase, detailed documentation, and experimental results are located in the [`FedOSD/`](FedOSD/) directory.

---

## Quick Navigation

- **Primary Comprehensive Documentation**: [FedOSD/README.md](FedOSD/README.md)
- **Pre-Registration Criteria**: [FedOSD/SUCCESS_CRITERIA_v5.1.md](FedOSD/SUCCESS_CRITERIA_v5.1.md)
- **Frozen Environment Specifications**: [FedOSD/FROZEN_CONFIGS_v5.md](FedOSD/FROZEN_CONFIGS_v5.md)
- **Amendment 1 Disclosure**: [FedOSD/AMENDMENT_1.md](FedOSD/AMENDMENT_1.md)
- **Confirmation Result Tables**: [FedOSD/README_confirmation_tables.md](FedOSD/README_confirmation_tables.md)
- **Publication Figures**: [FedOSD/figures/](FedOSD/figures/)
- **Stage 4 Evaluation Outputs**: [FedOSD/results_confirm/](FedOSD/results_confirm/)

---

## Workspace Structure

- **`FedOSD/`**: The core research codebase containing the federated unlearning platform (`tfedplat`), evaluation scripts, plotting pipelines, and confirmatory experiment results.
- **`seed999_laptop/`**: Independent reference run executed on a laptop environment for cross-platform calibration (Seed 999).
- **`stage4_backup.zip`**: Complete archived backup of all raw JSON and CSV results from the Stage 4 confirmatory runs (Seeds 21–30).

---

## Research Summary & Key Findings

1. **Phase 1 (Exploratory Benchmarks & Goals 1–4)**:
   - Formulated **Adaptive FedOSD** with closed-form Woodbury identity solve and dynamic deficit-driven weighting $w_i(t)$.
   - Built a 300-DPI publication visualization suite (Figures 1–5 in [`FedOSD/figures/`](FedOSD/figures/)).
   - Evaluated multi-client batch unlearning ($U_N=2$, Figure 6) and complex vision generalization on CIFAR-100 (Figure 7).
2. **Phase 2 (Scientific Auditing & Gate A)**:
   - Eliminated test feedback leakage artifact.
   - Identified that $13.41$ percentage points (44.7%) of historical unlearning gaps were an artifact of learning rate disparities between pretraining ($0.01$) and retraining ($0.05$).
   - Established primary matched **Regime R2** (Pretrain $0.05$, Retrain $0.05$, Recovery $10^{-4}$).
   - Discovered $+33.8\%$ gradient step dilation in legacy code and formulated **One-Sided Unscaled FedOSD (cap 1.0)** via dual NNLS and hard-norm bounding.
   - Calibrated a strength-matched baseline comparator (Arm A2, $s^* = 0.54$) to match parameter displacement within $1.1\%$.
3. **Phase 3 (Pre-Registration Protocol)**:
   - Codified strict pre-registration protocols (`SUCCESS_CRITERIA_v5.1.md`) and locked code at git tag `freeze-v5.1`.
   - Documented Amendment 1 in `AMENDMENT_1.md` before viewing metrics.
4. **Phase 4 (Confirmatory Trial on Seeds 21–30 - Final Outcome C)**:
   - Ran 100 runs across 10 unseen seeds on GPU (11.8 hours).
   - **Pre-Registered Confirmatory Family F1 (Round 100, Matched A3 vs A2)**:
     - *Retained Cohort Mean Acc*: $+1.15\%$ ($47.89\%$ vs $46.75\%$, Holm $p = 0.0004$) $\to$ **Superior**.
     - *Worst-Off Client Floor*: $-5.47\%$ ($19.93\%$ vs $25.40\%$, Holm $p = 0.0001$) $\to$ **Inferior**.
     - *Max Collateral Drop*: $-3.37\%$ ($14.00\%$ vs $10.63\%$, Holm $p = 0.0245$) $\to$ **Inferior**.
   - **Final Verdict: OUTCOME C (Trade-Off / Reject Floor Protection)**:
     - Relaxing hard orthogonality provides a modest cohort mean boost ($+1.15$ pp), but **fundamentally fails to protect worst-off clients** in non-IID federated distributions, dropping the worst-off floor by $5.47$ pp. Standard hard projection remains strictly necessary for worst-case client protection.

For full mathematical derivations, statistical tables, and runbooks, please see **[`FedOSD/README.md`](FedOSD/README.md)**.
