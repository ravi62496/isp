Seeds [31, 32, 33], mean +/- std. Accuracies in %.

| Model | clean_acc | asr | forget_trig_acc | heldout_acc | floor_test |
|---|---|---|---|---|---|
| Original (before forgetting) | 49.31 +/- 0.36 | 84.30 +/- 0.42 | 93.40 +/- 0.58 | 48.44 +/- 5.86 | nan +/- nan |
| Retrain (gold standard) | 59.59 +/- 0.15 | 4.58 +/- 1.42 | 3.54 +/- 2.79 | 53.17 +/- 2.48 | 45.28 +/- 3.25 |
| FedOSD (hard) | 42.72 +/- 1.01 | 1.85 +/- 1.31 | 0.21 +/- 0.15 | 39.17 +/- 5.33 | 23.56 +/- 2.78 |
| Soft FedOSD (ours) | 42.17 +/- 1.05 | 1.77 +/- 1.25 | 0.20 +/- 0.15 | 37.78 +/- 5.74 | 23.17 +/- 3.42 |
| Plain gradient ascent | 34.29 +/- 2.14 | 0.31 +/- 0.24 | 0.03 +/- 0.02 | 2.78 +/- 2.18 | 2.56 +/- 1.89 |
| Post-training, no projection | 51.59 +/- 0.44 | 6.62 +/- 2.81 | 4.13 +/- 1.31 | 45.61 +/- 3.62 | 33.44 +/- 1.98 |
| Post-training, projected | 51.59 +/- 0.49 | 6.43 +/- 2.85 | 3.85 +/- 1.17 | 45.61 +/- 3.62 | 33.33 +/- 1.91 |

All metrics (including MIA / distance / Grad-CAM): final_summary.csv

Ablation: soft FedOSD weighting (lambda=100.0), retained TEST accuracy from run logs

| Variant | stage | mean | floor |
|---|---|---|---|
| eta=0.0 (uniform) | after unlearning | 42.17 +/- 1.05 | 23.17 +/- 3.42 (n=3) |
| eta=0.0 (uniform) | after post-training | 51.51 +/- 0.47 | 33.39 +/- 1.81 (n=3) |
| eta=1.0 (deficit) | after unlearning | 42.12 +/- 1.02 | 23.56 +/- 3.35 (n=3) |
| eta=1.0 (deficit) | after post-training | 51.51 +/- 0.47 | 33.44 +/- 1.85 (n=3) |
