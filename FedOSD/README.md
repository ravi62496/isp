# Code of the AAAI-25 paper
![image](https://github.com/user-attachments/assets/97b41151-d3ba-4275-94dc-d6cdc87f993b)


# How to run the code
First, run the following command to train a FL global model as the original model for unlearning experiments.
```
/usr/bin/python run_UnlearningTask.py --seed 1 --device 0 --module CNN_CIFAR10 --algorithm FedAvg --dataloader DataLoader_cifar10_pat --N 10 --NC 2 --balance True --B 200 --C 1.0 --R 2000 --E 1 --lr 0.05 --decay 0.999 --step_type bgd --unlearn_cn 1 --unlearn_pretrain True --save_model True
```

Next, execute the following command to run the unlearning algorithm.
```
/usr/bin/python run_UnlearningTask.py --seed 1 --device 0 --module CNN_CIFAR10 --algorithm FedOSD --dataloader DataLoader_cifar10_pat --N 10 --NC 2 --balance True --B 200 --C 1.0 --R 200 --UR 100 --E 1 --decay 0.999 --step_type bgd --unlearn_cn 1 --save_model True --lr 0.0004 --r_lr 1e-6
```

# Background
![image](https://github.com/user-attachments/assets/b94026b9-9a9e-409b-bdae-393b09dad0fd)

![image](https://github.com/user-attachments/assets/3d576e45-45b6-4333-9ba3-32019f29b8c9)

![image](https://github.com/user-attachments/assets/9d4188ed-f3b3-476d-a840-bac05fd2f917)

![image](https://github.com/user-attachments/assets/2319c9a4-c8ad-436f-a2a7-a2da2efc3015)

## Key Challenges to FU
![image](https://github.com/user-attachments/assets/9f554e07-a711-4ae7-8eed-09f9314b4b76)

# The Proposed FedOSD
![image](https://github.com/user-attachments/assets/4e487b64-95a2-4501-aa82-c89f69d59535)

## Unlearning Cross-Entropy (UCE) Loss
![image](https://github.com/user-attachments/assets/0525c0c9-4648-4a55-bcc9-29a6aa2968f4)

## Orthogonal Steepest Descent Direction
![image](https://github.com/user-attachments/assets/70321347-e3d8-4a52-849d-27288340cda1)

![image](https://github.com/user-attachments/assets/b9c92c8c-5de9-4446-8f64-583895819582)

## Model Reverting Issue
![image](https://github.com/user-attachments/assets/a4d41a84-9202-49cb-a84d-a3f2a440e71a)

# Experiments
![image](https://github.com/user-attachments/assets/c2850137-0976-4ee1-900a-f026882db461)

![image](https://github.com/user-attachments/assets/ede468f8-5f1c-48ed-bbf8-72378b5cd917)

![image](https://github.com/user-attachments/assets/8770ebc9-8fd7-47f0-ba3f-6bd36b37aa03)

![image](https://github.com/user-attachments/assets/e2a325a0-98fb-413b-97ac-3ed5f2e25578)

![image](https://github.com/user-attachments/assets/25ab047a-0b2e-4f6d-a593-7a3d605e869b)

# Conmunication
Welcome to our session during the conference time. We will be there on March 1, 12:00 - 14:30.

# Cite
Pan Z, Wang Z, Li C, et al. Federated unlearning with gradient descent and conflict mitigation[C]//Proceedings of the AAAI Conference on Artificial Intelligence. 2025, 39(19): 19804-19812.

---

# Our Extension: Adaptive FedOSD

This fork extends the original FedOSD algorithm with a **client-weighted soft
constraint**, so that retained clients who are more affected by an
unlearning step get more protection than clients who are barely touched by
it. The original FedOSD gives every retained client identical, absolute
protection (`g_i^T d = 0` for all `i`); this extension makes that constraint
*adaptive* instead of uniform.

## Motivation

FedOSD's orthogonal steepest-descent direction treats every retained
client's gradient the same way when computing the unlearning direction `d`.
In practice, retained clients are not affected equally by an unlearning
step — some clients' data overlaps more with the client being forgotten and
loses more accuracy, while others are essentially untouched. A uniform hard
constraint cannot tell these two cases apart.

**Research question:** can adaptive, client-specific protection improve the
forgetting–fairness trade-off of FedOSD, without weakening forgetting or
overall retained utility?

## Method

### From hard orthogonality to a weighted soft constraint

Original FedOSD solves:

```
minimize over d:  ||d - g_u||^2   subject to   g_i^T d = 0   for all i
```

via a pseudoinverse projection onto the null space of the retained-client
gradient matrix.

This extension instead solves:

```
minimize over d:  ||d - g_u||^2 + lambda * sum_i  w_i * (g_i^T d)^2
```

where `w_i` is a per-client protection weight and `lambda` controls how
strongly the constraint is enforced overall. The closed-form solution is
derived via the **Woodbury matrix identity** so the linear solve stays in
the small (`n_retained_clients x n_retained_clients`) space rather than the
model's full parameter space — the same trick that makes the original
FedOSD projection tractable.

Two limits were used to validate the math:

- As `lambda -> infinity`, the soft constraint converges exactly to
  FedOSD's original hard-orthogonality projection. This was confirmed
  empirically: with `lambda = 100000`, the resulting model trajectory
  (measured via distance-from-init, retained accuracy, and ASR) matches
  the unmodified FedOSD baseline almost exactly.
- As `lambda -> 0`, the constraint vanishes and `d -> g_u` (no protection
  at all).

### Adaptive per-client weights

`w_i` starts at `1.0` for every retained client and is updated once per
round:

1. Track each client's local test accuracy as an exponential moving
   average (EMA, `beta = 0.8`) to filter out single-round noise.
2. Compute each client's deficit relative to the group's mean smoothed
   accuracy.
3. Increase `w_i` in proportion to that deficit (`weight_eta` controls the
   update rate).
4. Apply mild multiplicative decay (`0.999`/round) and a hard cap
   (`5.0`) so one volatile round or one client cannot permanently
   dominate the protection budget.
5. Feed the updated weights into the next round's soft-constraint solve.

## New CLI arguments

Two new arguments were added to `UnlearningTask.py`'s `read_params()`
(prefixed `adap_` to avoid a name collision with an existing argument):

| Argument | Default | Meaning |
|---|---|---|
| `--adap_lam` | `1.0` | Soft-constraint strength (`lambda`). Larger = closer to original FedOSD's hard constraint. |
| `--adap_weight_eta` | `0.05` | Per-client weight update rate. Larger = protection weights diverge faster/more aggressively. |

Example run:

```
python run_UnlearningTask.py --seed 1 --device 0 --module CNN_CIFAR10 --algorithm FedOSD --dataloader DataLoader_cifar10_pat --N 10 --NC 2 --balance True --B 200 --C 1.0 --R 200 --UR 100 --E 1 --decay 0.999 --step_type bgd --unlearn_cn 1 --save_model True --lr 0.0004 --r_lr 1e-6 --adap_lam 100.0 --adap_weight_eta 0.35
```

## Files changed

- `tfedplat/algorithm/unlearning/FedOSD.py`
  - Replaced `cal_psedoinverse` / `get_nearest_oth_d` (SVD-based hard
    projection) with `get_weighted_soft_d` (Woodbury-identity linear
    solve for the weighted soft constraint).
  - Added `update_client_weights`: EMA-smoothed, decayed, capped
    per-client weight update, with a debug printout of the weight
    vector every 10 rounds.
  - `train_a_round` now tracks retained clients by id (not just their
    gradients) so weights can be looked up and applied per client.
- `tfedplat/task/unlearning/UnlearningTask.py`
  - Added `--adap_lam` and `--adap_weight_eta` to `read_params()`.

## Implementation status

- [x] Baseline FedOSD reproduced end-to-end on the paper's own CIFAR-10
      configuration (10 clients, pathological non-IID split, backdoored
      target client) and checked against the paper's published numbers
      for the matching client/split configuration.
- [x] Weighted soft-constraint solver implemented and derived via the
      Woodbury identity.
- [x] Solver verified to converge to FedOSD's original hard-constraint
      behavior in the `lambda -> infinity` limit (confirms the math is
      correct, not just plausible).
- [x] Adaptive per-client weighting implemented, with EMA smoothing,
      decay, and capping to avoid reacting to single-round noise.
- [x] Debug instrumentation added (`Running on device: ...`, resolved
      `lambda`/`weight_eta` printout, per-round weight vector printout)
      so future runs are easy to sanity-check.
- [ ] Hyperparameter characterization in progress: mapping how `lambda`
      and `weight_eta` jointly control the trade-off between forgetting
      strength and retained-client protection/fairness. Early sweeps
      found that very small `lambda` and very large `weight_eta` both
      degrade retained accuracy rather than improving fairness, so the
      useful operating range appears to sit between the two tested
      extremes and is still being narrowed down.
- [ ] Final comparison of adaptive vs. uniform-weight soft constraint on
      retained-accuracy fairness (mean/min/std across clients) not yet
      concluded.

## Known pitfalls hit along the way (for future reference)

- The repo's public model/dataloader names don't match `BasicTask.py`'s
  hardcoded defaults (e.g. `DataLoader_cifar10_non_iid` isn't exposed;
  the correct name is `DataLoader_cifar10_pat`). Always check
  `tfedplat/__init__.py` for the actual exposed names before assuming a
  default config works.
- Watch argument-name collisions: a new `--adap_lam`/`--adap_weight_eta`
  pair was needed after `--lam`/`--weight_eta` collided with an
  existing argument elsewhere in the parser chain.
- When adding a new `params[...]` key, double-check the *existence
  check* matches the *key you're reading* — a mismatched `'lam' in
  params` vs. `params['adap_lam']` silently falls back to the default
  every time, with no error raised.
- When deriving a closed-form solve via the Woodbury identity, sanity
  check the algebraic limit (e.g. `lambda -> infinity`) by hand before
  trusting empirical results — an earlier version of the formula had
  the `lambda` terms cancel out asymptotically, causing the "soft"
  constraint to silently saturate at a fixed, `lambda`-independent
  point instead of approaching the hard constraint.
