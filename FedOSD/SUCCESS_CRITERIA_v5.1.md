# SUCCESS_CRITERIA_v5.1 (supersedes v5; commit, then tag freeze-v5.1 before any seed 21-30
# result exists; keep tag freeze-v5)
# Pinned files (SHA-256 of the LF files at the freeze-v5.1 commit):
#   evaluate_confirmation.py   <9a961ace93a8dbf103e049bd31e9d82831120f1ce90ff72dbdaf63a0410a5f3b> (evaluator SHA-256: 9a961ace93a8dbf103e049bd31e9d82831120f1ce90ff72dbdaf63a0410a5f3b)
#   run_confirmation.py        <d4d46681523048761c754bbcf97419683420b0a17163435fe40b50d08cb7692d>
#   build_manifest_confirm.py  <f30db579225c986105648023a52128a6b5e1fd8bf80f5ed8848a7d9a3aacb5a5>
# The code commit is the commit tagged freeze-v5.1 (not recorded in this file).

Seeds 21-30 (n=10), regime R2, one job at a time, results sealed in results_confirm/.
Environment: FROZEN_CONFIGS_v5.md (deterministic mode ON).
Arms per seed: pretrain (R=100); Retrain R=300 (evaluated at rounds 200 and 300);
  A1 Hard (rescaled to ||g_u||, scale 1.0); A2 Hard-scaled (0.54);
  A3 One-sided, unscaled (cap 1.0) [method]; A4 One-sided rescaled to ||g_u||.
Reference mapping: round 100 vs Retrain at round 200; round 200 vs Retrain at round 300.

Data provenance: RUN_MANIFEST_CONFIRM.csv (arm, seed, stage, path, sha1, commit, config hash,
array length, index) is written by build_manifest_confirm.py after all runs finish. All runs
share ONE git commit and one config hash per arm. The evaluator uses exactly one file per
cell, reads the last array element, asserts its length, and aborts on any ambiguity or on
modified tracked code/doc files (UnlearningTask, figures and result directories are
excluded). Raw outputs are not opened until all 10 seeds finish. Labels need n >= 10;
otherwise 'descriptive (n<10)'. Every reported table is a script-generated CSV.

Confirmatory family F1: A3 vs A2 at round 100, Holm m=3 on t-test p:
  retained cohort mean | worst-off floor (min) | max collateral drop
  (method minus comparator, plus = better; drop = comparator minus method)
  practical thresholds: mean 1.0, floor 2.0, drop 2.0 pts.
Secondary pre-registered family F2: A3 vs A2 at round 200, same metrics, own Holm m=3,
  equivalence margins mean 0.5, floor 1.0, drop 1.0 (90% CI inside the margin).
  F2 labels are also gated by strength matching measured at round 200.
Labels, in this precedence: Superior (95% CI>0, Holm p<0.05, estimate>=practical);
  Inferior (95% CI<0, Holm p<0.05, estimate<=-practical); Equivalent (F2 only; 90% CI inside
  the margin); Better but negligible (95% CI>0, Holm p<0.05); Worse but negligible
  (95% CI<0, Holm p<0.05); else Inconclusive. No Equivalent label at round 100 (tuning-seed
  paired SDs: mean 1.1, floor 1.6, drop 3.0; equivalence with margin <= 1.0 is unattainable
  at n=10).
Strength matching (seed averages): displacement ratio in [0.95, 1.05] and |ASR diff| <= 1.0 pt
  (F1 at round 100; F2 at round 200, measured displacement); otherwise every label for that
  comparison is 'not comparable'. If F1 is not matched the outcome is C. Extra runs or
  interpolated comparisons are exploratory and use scripts not pinned here.
Exploratory (labeled so): worst-2 mean; forget-set; distance to Retrain on ASR and forget-set;
  A3 vs A4; A3 vs A1; sensitivity control A1 vs A2 (expected: A1 worse on mean at round 100;
  tuning seeds -1.49), exempt from strength matching.
Optional replication family F3 (only if R1 arms are run on seeds 21-30): regime R1,
  A3' vs A2' (Hard-scaled s=0.27), same metrics and labels as F1, own Holm m=3; reported
  outside the outcome rule below.
Tests: paired t 95% CI, bootstrap 95% CI (10,000), Wilcoxon p; Holm on t p-values.
Outcome A: F1 strength matched; Superior on at least two primary metrics; none Inferior.
Outcome B: F1 and F2 strength matched; no Superior and no Inferior in F1; Equivalent on all
  three metrics in F2.
Outcome C: anything else (for example Inferior on the floor at round 100 with Equivalent
  after recovery).
Stopping: n=10; if fewer seeds finish, report n; no replacement; a crashed run is rerun with
  the identical command.