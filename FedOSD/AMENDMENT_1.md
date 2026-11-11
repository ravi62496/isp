# AMENDMENT 1: provenance config-hash check (change did not use any metric)

Date and time written: 2026-11-10 18:25:00 +05:30
Pre-registration in effect: tag freeze-v5.1 (commit 7df1ab9). Pinned files are UNCHANGED:
- evaluate_confirmation.py  SHA-256 9a961ace93a8dbf103e049bd31e9d82831120f1ce90ff72dbdaf63a0410a5f3b
- run_confirmation.py       SHA-256 d4d46681523048761c754bbcf97419683420b0a17163435fe40b50d08cb7692d
- build_manifest_confirm.py SHA-256 f30db579225c986105648023a52128a6b5e1fd8bf80f5ed8848a7d9a3aacb5a5

## 1. What happened
After the 10-seed run finished (seeds 21-30, 100 manifest cells, one commit), the pinned evaluator
aborted at its provenance check ("Multiple config hashes detected for arm A1": 10 distinct hashes
for 10 seeds). It aborted before reading any metric.

## 2. Cause
The pre-registered rule is one configuration per arm. build_manifest_confirm.py wrote a
config_hash that depends on the seed, so the rule could never be satisfied by 10 different seeds.
The strict check only runs for seeds 21-30, so the tuning-seed dry runs could not reveal it.

## 3. What was changed, and what was not
- Changed: the config_hash column of the manifest only. amend_manifest_config_hash.py read each
  result file's stored command line, removed "--seed N" and "--device N", and hashed the rest.
  It read no metrics. The original manifest is kept as RUN_MANIFEST_CONFIRM.csv; the amended
  one is RUN_MANIFEST_CONFIRM_amended.csv.
- Not changed: the evaluator, the runner, the builder, the criteria, the result files.
- The script aborts if any arm used more than one command line or more than one commit.

## 4. Verification (paste the script output here)
```
Cells: 100 | distinct commits: ['7df1ab9680b45f007cd393b25c45c7a2ffb417ab']
A1       distinct normalized command lines: 1
    run_UnlearningTask.py --module CNN_CIFAR10 --dataloader DataLoader_cifar10_pat --N 10 --NC 2 --balance True --B 200 --C 1.0 --E 1 --decay 0.999 --step_type bgd --unlearn_cn 1 --save_model True --algorithm FedOSD --R 200 --UR 100 --lr 0.0004 --r_lr 0.0001 --proj_mode hard --step_scale 1.0 --adap_lam 100000.0 --adap_weight_eta 0.0
A2       distinct normalized command lines: 1
    run_UnlearningTask.py --module CNN_CIFAR10 --dataloader DataLoader_cifar10_pat --N 10 --NC 2 --balance True --B 200 --C 1.0 --E 1 --decay 0.999 --step_type bgd --unlearn_cn 1 --save_model True --algorithm FedOSD --R 200 --UR 100 --lr 0.0004 --r_lr 0.0001 --proj_mode hard --step_scale 0.54 --adap_lam 100000.0 --adap_weight_eta 0.0
A3       distinct normalized command lines: 1
    run_UnlearningTask.py --module CNN_CIFAR10 --dataloader DataLoader_cifar10_pat --N 10 --NC 2 --balance True --B 200 --C 1.0 --E 1 --decay 0.999 --step_type bgd --unlearn_cn 1 --save_model True --algorithm FedOSD --R 200 --UR 100 --lr 0.0004 --r_lr 0.0001 --proj_mode one_sided --cap_hard_ratio 1.0 --weight_mode uniform --adap_lam 100000.0 --adap_weight_eta 0.0
A4       distinct normalized command lines: 1
    run_UnlearningTask.py --module CNN_CIFAR10 --dataloader DataLoader_cifar10_pat --N 10 --NC 2 --balance True --B 200 --C 1.0 --E 1 --decay 0.999 --step_type bgd --unlearn_cn 1 --save_model True --algorithm FedOSD --R 200 --UR 100 --lr 0.0004 --r_lr 0.0001 --proj_mode one_sided --weight_mode uniform --adap_lam 100000.0 --adap_weight_eta 0.0
Retrain  distinct normalized command lines: 1
    run_UnlearningTask.py --module CNN_CIFAR10 --dataloader DataLoader_cifar10_pat --N 10 --NC 2 --balance True --B 200 --C 1.0 --E 1 --decay 0.999 --step_type bgd --unlearn_cn 1 --save_model True --algorithm FedAvg --retrain True --test_interval 5 --R 300 --UR 0 --lr 0.05
Wrote RUN_MANIFEST_CONFIRM_amended.csv
sha256 RUN_MANIFEST_CONFIRM.csv: a163c302d84f20eb0c796383db54da80ff428c7d03c15b74d2f2b51ce282ab7a
sha256 RUN_MANIFEST_CONFIRM_amended.csv: 86b2e879c7c8920a17398dc55ec8abff98dbc9689699f8e207ca7dab1f858adf
```

SHA-256 of both manifests:
- `RUN_MANIFEST_CONFIRM.csv`:         `a163c302d84f20eb0c796383db54da80ff428c7d03c15b74d2f2b51ce282ab7a`
- `RUN_MANIFEST_CONFIRM_amended.csv`: `86b2e879c7c8920a17398dc55ec8abff98dbc9689699f8e207ca7dab1f858adf`

The script I ran (amend_manifest_config_hash.py) had SHA-256: ff16da3b0f5c67ee7b5beabf4e979836f5e04bf9fd6349e242c68c552e386827

## 5. What was viewed before this amendment (fill in truthfully)
Section 5 completed 2026-11-11

- I opened stage4_eval.txt. It contained only the header and the abort message ('Multiple config hashes detected for arm A1'). It contained no metrics.
- I am not sure whether I or an AI assistant saw log lines for seed 30 arm A4.
- I watched the run live.
- I did not open stage4_log.txt.
- I don't remember exactly whether I saw any metrics for F1/F2 arms or seeds 21-30 before the abort.
- Results for earlier exploratory and tuning seeds (11–20, 101–105) were evaluated before the run and are documented in README.md. They are separate from the confirmatory seeds 21–30.

## 6. Consequence
The evaluator is run unmodified on the amended manifest. This amendment is reported in the
README. No label, threshold or comparison was altered.
