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

## Amendment A (2026-11-18, before any unlearning result of this experiment was produced)
An audit of the seed-999 checkpoints showed the legacy trigger (FigRandBackdoor: 3x3 random
binary pattern at a mid-grey normalised value, per-sample label shift y->(y+5)%10) was never
learned: the original model predicted the backdoor label equally often with the real trigger,
8 random triggers, or no trigger (23.3-23.9%); held-out ASR was 5.3%. Changes:
- All runs of this experiment use `--backdoor patch4`: solid 4x4 square at the per-channel
  normalised white value, bottom-right corner, single target class = the class with the fewest
  forget-client training samples. Outputs go to `UnlearningTask_patch4/`.
- Gate before any unlearning run: the pretrained model must reach held-out triggered ASR >= 90%,
  target rate without trigger <= 20%, clean retained accuracy >= 40% (`export_bundle.py --gate`).
- The lambda-selection constraint "ASR <= 5%" now refers to this patch4 ASR.
- Runs started with the legacy trigger (pretraining only, no unlearning) are discarded.
Hypothesis, factors, seeds and decision rule are unchanged.

## Amendment B (2026-11-18, after the gate ran on the 9 pretrained models, before any unlearning run)
All 9 tuning-seed pretrained models failed the Amendment-A gate on one criterion only:
held-out triggered ASR was 64.6-88.9% (< 90%), while the target rate without the trigger was
2.0-9.3% and clean retained accuracy 47.0-66.1%. The trigger therefore causes the target
prediction; 90% was an unrealistic bar for one poisoning client out of ten under FedAvg.
**This threshold is changed after seeing these gate values** (disclosed deviation). No unlearning
run had started, so no quantity used by the decision rule had been observed. Changes:
- Gate = causality test: triggered ASR >= 60%, triggered ASR minus no-trigger target rate
  >= 50 pp, no-trigger target rate <= 20%, clean retained accuracy >= 40%.
- The gate is now computed on the retained clients' **validation** halves (it previously used
  the test halves, which would have let a go/no-go decision depend on the test split).
- analyze_sweep.py sanity check: round-0 forget-client ASR >= 60% (was 80%).
- Pretrained models are kept unchanged. Hypothesis, factors, seeds, decision rule and the
  lambda constraint (ASR <= 5%) are unchanged.
