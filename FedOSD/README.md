# FedOSD/: training and unlearning code

This folder holds the federated learning and unlearning code for the project. The project
overview and results summary are in the [top-level README](../README.md); the full write-up is
in **[RESULTS_SOFT_FEDOSD.md](RESULTS_SOFT_FEDOSD.md)**.

Based on: Pan Z, Wang Z, Li C, et al. *Federated unlearning with gradient descent and conflict
mitigation.* AAAI-25, 39(19): 19804-19812.

## Files

| File | Purpose |
|---|---|
| `run_UnlearningTask.py` | Entry point for one run (pretraining, retrain, or an unlearning method) |
| `tfedplat/` | FL library: FedAvg, FedOSD, our Soft FedOSD (`proj_mode two_sided`), plain gradient ascent, CIFAR-10 loaders (pathological and Dirichlet), patch backdoor |
| `run_soft_sweep.py` | Tuning sweep on seeds 101-103 (validation only) |
| `analyze_sweep.py` | Applies the pre-registered decision rule and writes `results_soft/sweep_decision.json` |
| `run_final.py` | Final seeds 31-33: every method, backdoor gate, bundle export and audit |
| `overnight.sh` | Chains sweep → analysis → final runs |
| `run_calib_relearn.py` | Pre-registered calibration of the relearning attack |
| `export_bundle.py` | Exports evaluation sets and models into a standalone audit bundle; `--gate` checks that the backdoor was learned |
| `bundle_api_template.py` | Loader copied into each bundle; used by `audit_code/` |
| `PREREG_SOFT_HETEROGENEITY.md` | Pre-registration of the tuning study, with Amendments A and B |
| `PREREG_RELEARN.md` | Pre-registration of the relearning-attack calibration |
| `FROZEN_CONFIGS_v5.md` | Frozen training configuration (Regime R2) referenced by the pre-registration |

## Key options of `run_UnlearningTask.py`

| Option | Meaning |
|---|---|
| `--backdoor patch4` | 4x4 white patch trigger on the forget client (`legacy` = old trigger, not used) |
| `--proj_mode hard` / `two_sided` | FedOSD hard projection / Soft FedOSD |
| `--adap_lam`, `--adap_weight_eta`, `--weight_mode` | Soft FedOSD lambda, adaptive weighting strength, `uniform` or `deficit` |
| `--recovery_projection False` | E3: post-training without projection |
| `--retrain True` | Retrain from scratch on the retained clients (gold standard) |

## Earlier work

An earlier exploratory and confirmatory study (Stages 1-4, one-sided vs hard projection, seeds
21-30) used a trigger that was later found never to be learned. Its code and results were
removed to keep the repository focused. They remain available in git under the tag
`legacy-stage4` (`git checkout legacy-stage4`).
