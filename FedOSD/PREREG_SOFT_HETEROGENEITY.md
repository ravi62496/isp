# Pre-registration: Soft FedOSD heterogeneity test (H-adapt)

Written 2026-11-18, **before any run of this experiment**. Commit this file before launching jobs.

## Method under test
Soft (two-sided) FedOSD: `--proj_mode two_sided`, closed-form Woodbury solve
(`FedOSD.get_two_sided_soft_d`), `--adap_lam 20`.

## Hypothesis H-adapt
Adaptive client weighting (eta > 0) raises the worst-off retained-client floor *more as
client heterogeneity increases*.

## Design (tuning seeds only)
- Seeds: 101, 102, 103 (tuning seeds; never the final reporting seeds 31-33, never 21-30).
- Heterogeneity: `pat` NC2 (current), Dirichlet alpha=0.5, Dirichlet alpha=0.1
  (`--dataloader DataLoader_cifar10_dir --Diralpha <a>`).
- eta in {0.0 (uniform), 0.4, 1.0}; lambda fixed at 20.
- 3 x 3 x 3 = 27 unlearning runs; everything else as in Regime R2 (FROZEN_CONFIGS_v5.md).
- Metric: worst-off retained-client accuracy at unlearning round 100, on the **validation**
  split only (test split untouched until final seeds).

## Decision rule (fixed now)
Let G(h) = mean over seeds of [floor(eta*) - floor(eta=0)] at heterogeneity h, where eta* is the
better of {0.4, 1.0} chosen on the alpha=0.1 setting.
- **Adaptive becomes the main method** iff BOTH hold:
  (a) G(alpha=0.1) >= 2.0 pp, and (b) G(alpha=0.1) > G(pat).
  (G(alpha=0.5) is reported but is not part of the rule.)
- Otherwise **uniform soft FedOSD (eta=0) is the main method** and adaptive is reported as an
  ablation with its null result.
- lambda for the final runs is chosen on these same tuning seeds (sweep {5, 20, 100}) by highest
  validation retained mean subject to backdoor ASR <= 5%.

## Leakage rules
- No test-split metric is read to make any choice above.
- Code refuses any val->test fallback (FedOSD.py / UnlearningTask.py raise instead).
- Final numbers are reported once, on seeds 31-33, with the configuration frozen here.
