# Pre-registration: calibrated relearning attack (written 2026-11-19, before any run below)

## Why
The fixed relearning attack (32 triggered images, SGD lr 0.01, 100 steps) saturates. It
installs the backdoor even in the retrain model, which never saw the trigger, within 1 step.
So it cannot tell real forgetting from suppression. The attack strength is therefore
calibrated on the **tuning seeds 101-103** and then applied once to the final seeds 31-33.
The original (saturated) result stays reported.

## Calibration (tuning seeds only)
- Models: retrain from scratch (R300, same config as the final runs) and the original model,
  pat split, seeds 101-103. Bundles go to `bundles/tune_seed<s>`.
- Grid: n in {8, 16, 32} triggered forget-client images x lr in {0.001, 0.003, 0.01};
  100 full-batch SGD steps; the subset is drawn with `np.random.default_rng(seed)`.
- Metric: backdoor ASR on held-out triggered images after 100 steps.

## Rule (fixed now)
A configuration is **eligible** if retrain's ASR after 100 steps, averaged over seeds 101-103,
is <= 20%. In other words, the attack cannot install the backdoor from scratch in a model that
never saw it. Among the eligible configurations, choose the **strongest** (largest n x lr; on a
tie, larger n). If none is eligible, report that the attack cannot be calibrated and stop.

## Final application (seeds 31-33, once)
Run the chosen configuration on all 7 models of each final bundle.
- Primary result: ASR after 100 steps per model against retrain.
- An unlearned model whose ASR clearly exceeds retrain's (beyond the seed spread) retains
  latent trigger sensitivity.
