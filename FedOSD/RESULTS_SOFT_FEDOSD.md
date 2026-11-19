# Soft FedOSD: final results (CIFAR-10, seeds 31-33)

Configuration frozen on tuning seeds 101-103 by the pre-registered rule
(`PREREG_SOFT_HETEROGENEITY.md`, Amendments A and B), then run **once** on the reporting
seeds 31-33. Every number below that involves the test split comes from those final runs only.

## 1. Setup

| Item | Value |
|---|---|
| Data | CIFAR-10, 10 clients, pathological split (2 classes per client), 50/50 val/test split of each client's held-out data |
| Model | `CNN_CIFAR10` (2 conv + 3 FC). **Deviation:** the brief suggests ResNet-18; kept the repo CNN for compute reasons |
| Forget client | 1 client per seed, carrying a backdoor (below) |
| Backdoor | 4x4 white patch, bottom-right; single target class = the forget client's rarest class; 80% of its images poisoned during pretraining |
| Pretraining | FedAvg, 100 rounds, lr 0.05 |
| Unlearning | 100 rounds, lr 0.0004, UCE loss; then 100 post-training rounds, lr 1e-4 |
| Retrain (gold standard) | FedAvg from scratch on the 9 retained clients, 300 rounds |
| Soft FedOSD | `--proj_mode two_sided`, lambda = 100, eta = 0 (uniform weights), chosen by the rule |

## 2. How the configuration was chosen (validation split only)

**Heterogeneity test (H-adapt).** Does adaptive client weighting (eta > 0) raise the
worst-off retained client more as heterogeneity grows? Retained **validation** accuracy after
100 unlearning rounds, lambda = 20, mean +/- std over seeds 101-103:

| Split | eta | floor | mean | forget ASR |
|---|---|---|---|---|
| pat | 0.0 | 16.33 +/- 7.57 | 41.56 +/- 1.31 | 0.17 |
| pat | 1.0 | 19.00 +/- 6.13 | 41.35 +/- 1.32 | 0.16 |
| dir0.5 | 0.0 | 57.61 +/- 2.28 | 63.07 +/- 1.04 | 0.56 |
| dir0.5 | 1.0 | 57.69 +/- 2.10 | 63.12 +/- 0.93 | 0.55 |
| dir0.1 | 0.0 | 22.67 +/- 5.82 | 51.14 +/- 2.73 | 1.12 |
| dir0.1 | 1.0 | 24.94 +/- 5.22 | 50.99 +/- 3.10 | 1.00 |

(The eta = 0.4 floor lies between the eta = 0 and eta = 1 floors on every split; full table in `results_soft/analyze.txt`.)

eta* = 1.0 (chosen on dir0.1). Floor gain G = floor(eta*) - floor(eta=0):
**pat +2.67 pp, dir0.5 +0.08 pp, dir0.1 +2.28 pp.**
The rule needs G(dir0.1) >= 2.0 (met) **and** G(dir0.1) > G(pat) (not met).
**Verdict: uniform soft FedOSD (eta = 0) is the main method**; adaptive weighting is reported
as an ablation.

*Exploratory reading, not part of the verdict:* by floor accuracy, `pat` (16%) is a harsher
split than dir0.1 (23%), so condition (b) assumed an ordering of heterogeneity that the data
did not follow. Adaptive weighting gave a +2-3 pp floor gain on the two harsh splits at no cost
to the mean, and nothing on the mild one. With n = 3 and seed std of about 6 pp, the gain is not
statistically established.

**Lambda selection** (pat, eta = 0): highest validation mean subject to forget ASR <= 5% on
every seed. All three values qualified; lambda = 5: 37.93, lambda = 20: 41.56,
**lambda = 100: 43.19 -> chosen**.

## 3. Final results (seeds 31-33, test split, mean +/- std)

All 3 final pretrained models passed the backdoor gate: held-out triggered ASR 82.7-85.3%,
backdoor-label rate without the trigger 6.1-8.4%.

| Model | clean acc | backdoor ASR | forget-client triggered acc | forget held-out acc | worst retained client |
|---|---|---|---|---|---|
| Original (before forgetting) | 49.31 +/- 0.36 | 84.30 +/- 0.42 | 93.40 +/- 0.58 | 48.44 +/- 5.86 | n/a |
| Retrain (gold standard) | 59.59 +/- 0.15 | 4.58 +/- 1.42 | 3.54 +/- 2.79 | 53.17 +/- 2.48 | 45.28 +/- 3.25 |
| FedOSD (hard) | 42.72 +/- 1.01 | 1.85 +/- 1.31 | 0.21 +/- 0.15 | 39.17 +/- 5.33 | 23.56 +/- 2.78 |
| **Soft FedOSD (ours)** | 42.17 +/- 1.05 | 1.77 +/- 1.25 | 0.20 +/- 0.15 | 37.78 +/- 5.74 | 23.17 +/- 3.42 |
| Plain gradient ascent | 34.29 +/- 2.14 | 0.31 +/- 0.24 | 0.03 +/- 0.02 | 2.78 +/- 2.18 | 2.56 +/- 1.89 |
| Post-training, no projection (E3) | 51.59 +/- 0.44 | 6.62 +/- 2.81 | 4.13 +/- 1.31 | 45.61 +/- 3.62 | 33.44 +/- 1.98 |
| Post-training, projected (E4) | 51.59 +/- 0.49 | 6.43 +/- 2.85 | 3.85 +/- 1.17 | 45.61 +/- 3.62 | 33.33 +/- 1.91 |

Definitions:
- clean acc: retained clients' test halves.
- backdoor ASR: triggered retained test images (true class not the target) predicted as the target.
- forget-client triggered acc: the forget client's own triggered training images.
- forget held-out acc: the forget client's clean held-out test half.
- worst retained client: lowest final-round test accuracy among the 9 retained clients.

**Audit (same seeds).** The full table is in `results_final/final_summary.csv`.

| Model | MIA AUC | KL(retrain to model) | L2 to retrain | Grad-CAM energy in trigger (forget, triggered) | CAM SSIM vs retrain (trigger test) |
|---|---|---|---|---|---|
| Original | 0.496 | 0.563 | 7.24 | 0.280 | 0.240 |
| Retrain | 0.493 | 0 | 0 | 0.030 | 1 |
| FedOSD (hard) | 0.504 | 0.753 | 7.22 | 0.487 | 0.288 |
| Soft FedOSD (ours) | 0.505 | 0.771 | 7.22 | 0.492 | 0.287 |
| Plain GA | 0.513 | 1.538 | 7.23 | 0.387 | 0.271 |
| E3 post-train, plain | 0.497 | 0.503 | 7.21 | 0.437 | 0.288 |
| E4 post-train, projected | 0.497 | 0.504 | 7.21 | 0.436 | 0.288 |

## 4. Findings

1. **All unlearning methods remove the backdoor's effect.** ASR falls from 84% to under 2%
   (retrain: 4.6%). After post-training it settles at about 6.5%, near retrain.
2. **Soft FedOSD at the chosen lambda is indistinguishable from hard FedOSD** on every metric
   (for example clean accuracy 42.2 vs 42.7%, worst client 23.2 vs 23.6%). The tuning sweep
   preferred the largest lambda, and lambda -> infinity is the hard projection, so in this
   setting the soft relaxation gave no measurable advantage. This is a null result for the
   method, reported as such.
3. **Plain gradient ascent overshoots.** It forgets the forget client's clean classes too
   (held-out accuracy 2.8%) and collapses the worst retained client (2.6%). Projection prevents
   this.
4. **Post-training recovers utility; its projection does not matter here.** E3 and E4 are equal
   within noise on every metric (clean accuracy 51.59 vs 51.59%, ASR 6.6 vs 6.4%).
5. **The MIA cannot certify forgetting in this setting.** Its AUC is about 0.50 even for the
   original model, because that model does not overfit the forget client's clean data
   (training accuracy 47.7% vs held-out 48.4%). A membership signal that is absent before
   unlearning cannot show that unlearning removed it.
6. **L2 parameter distance to retrain is uninformative.** It is about 7.2 for every model,
   because retrain follows a different trajectory. KL on outputs does separate the methods:
   E3/E4 0.50 < original 0.56 < hard/soft 0.75-0.77 < GA 1.54.
7. **Grad-CAM shows the trigger is still attended to after unlearning.** On triggered forget
   images, the CAM toward the target class puts 44-49% of its energy in the 4x4 patch after
   unlearning (original 28%, retrain 3%). So the models stop predicting the backdoor label but
   still respond to the patch. Caveat: these CAMs are for a class the model no longer predicts
   and are partly near zero (the fraction with undefined Pearson rises to 6-13%). The relearning
   test (section 5) is the behavioural check.

## 5. Relearning test

Protocol in `audit_code/audit/relearn.py`, fixed before any relearning result was produced.
An attacker fine-tunes each model with plain SGD (lr 0.01, 100 full-batch steps) on 32 of the
forget client's triggered images, and we track backdoor ASR.

Backdoor ASR on held-out triggered test images during relearning (mean +/- std, seeds 31-33):

| Model | ASR at step 0 | ASR at step 10 | ASR at step 100 | steps to 50% (per seed) |
|---|---|---|---|---|
| Original | 84.3 | 97.57 +/- 0.87 | 99.68 +/- 0.19 | 0, 0, 0 |
| Retrain (gold) | 4.6 | **91.67 +/- 3.07** | **95.83 +/- 2.66** | 1, 1, 1 |
| FedOSD (hard) | 1.8 | 99.12 +/- 0.70 | 100.00 +/- 0.00 | 1, 2, 2 |
| Soft FedOSD (ours) | 1.8 | 99.01 +/- 0.75 | 100.00 +/- 0.00 | 1, 2, 2 |
| Plain GA | 0.3 | 97.09 +/- 1.12 | 99.94 +/- 0.05 | 1, 2, 3 |
| E3 post-train, plain | 6.6 | 99.51 +/- 0.33 | 100.00 +/- 0.00 | 1, 1, 1 |
| E4 post-train, projected | 6.4 | 99.54 +/- 0.31 | 100.00 +/- 0.00 | 1, 1, 1 |

- **The attack saturates.** It installs the backdoor even in retrain, which never saw the
  trigger, within 1 step. So "steps to 50%" cannot separate the methods.
- **A small but consistent residual remains.** On every seed, every unlearned model relearns
  faster and further than retrain (step 10: retrain 87.3 / 93.9 / 93.7% against FedOSD hard
  98.2 / 99.9 / 99.3%). This agrees with the Grad-CAM finding (section 4, item 7): the behaviour
  is removed but some trigger sensitivity is not.
- E3 vs E4 again show no difference.
- A weaker attack (fewer images or a smaller lr) would separate the methods better. It was not
  run, because choosing the attack strength after seeing final-seed results would be tuning on
  the test seeds. It is a candidate for future work, to be fixed on the tuning seeds.

### Ablation: adaptive weighting on the final seeds

Retained test accuracy from the run logs, lambda = 100, seeds 31-33:

| Variant | stage | mean | floor |
|---|---|---|---|
| eta = 0 (uniform, main) | after unlearning | 42.17 +/- 1.05 | 23.17 +/- 3.42 |
| eta = 1 (adaptive) | after unlearning | 42.12 +/- 1.02 | 23.56 +/- 3.35 |
| eta = 0 (uniform, main) | after post-training | 51.51 +/- 0.47 | 33.39 +/- 1.81 |
| eta = 1 (adaptive) | after post-training | 51.51 +/- 0.47 | 33.44 +/- 1.85 |

The +2-3 pp floor gain seen on the tuning seeds (at lambda = 20) does not appear at the final
lambda = 100 (+0.4 pp after unlearning, 0 after post-training). This fits finding 2: at large
lambda the soft solve approaches the hard projection, which leaves little room for the client
weights to matter.

## 6. Deviations and disclosures

- **Amendment A:** the legacy trigger was never learned (seed 999: the same backdoor-label
  rate with and without the trigger; held-out ASR 5.3%). It was replaced by the patch trigger.
  **All earlier "ASR" numbers in this repo's README (Fig. 1, the CIFAR-100 table, the
  multi-client table) measure label-flip memorisation, not a backdoor.**
- **Amendment B:** the gate threshold was lowered from 90% to a causality test (ASR >= 60% and
  at least 50 pp above the no-trigger rate) **after seeing the gate values**, before any
  unlearning run. The gate also moved from the test halves to the validation halves.
- The ASR <= 5% constraint for lambda was applied per seed. The pre-registration did not specify
  per seed or on average; this was fixed before results existed.
- Machines: the tuning sweep's pathological runs were redone on the GPU server so that each
  split used one machine and one data-pool file. The Dirichlet pretrains were made on a laptop
  with byte-identical pools (verified by identical gate outputs). All final runs (seeds 31-33)
  ran on the server; the pool MD5 is in `results_final/provenance.json`.
- Plain GA uses the repo's `FedAvg_GA`: unprojected ascent on the UCE loss, same lr and rounds
  as FedOSD.
- Retrain uses 300 rounds against the original's 100, so its higher clean accuracy (59.6%)
  partly reflects longer training.
- Knowledge distillation was not run (optional per the plan).
