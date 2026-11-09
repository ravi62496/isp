# FROZEN_CONFIGS_v5

> **Status**: Frozen configuration for Master Guide v5 protocol evaluation.
> **Regime**: Primary Regime R2 (Converged Pretraining, Calibrated Post-Recovery).

---

## 1. Complete Hyperparameter Specification

### 1.1 Dataset and Architecture
- **Task Module**: `CNN_CIFAR10`
- **Data Loader**: `DataLoader_cifar10_pat`
- **Partitioning**: Pathological non-IID (`balance=True`, `NC=2` classes/client, `N=10` clients, `B=200` batch size)
- **Client Participation**: Full participation `C=1.0` (all 10 clients selected per round)
- **Gradient Step Type**: Batch Gradient Descent (`step_type='bgd'`)
- **Momentum**: `0.0`
- **Learning Rate Decay**: `decay = 0.999` per round across all phases
- **Gradient Clipping**: `g_clip = 1.0` across all phases

---

### 1.2 Phase 1: Converged Pretraining (FedAvg)
- **Algorithm**: `FedAvg`
- **Communication Rounds**: `R = 100` (100 rounds converged, verified by timing and checkpoints)
- **Local Epochs**: `E = 1`
- **Initial Learning Rate**: `lr = 0.05`
- **Decay**: `0.999`
- **Gradient Clipping**: `g_clip = 1.0`
- **Unlearn Target Portion**: Client chosen deterministically per partition seed; 80% clean training data poisoned with backdoor trigger during pretrain.
- **Exact Pretrain CLI Command**:
  ```bash
  python -u run_UnlearningTask.py --seed <seed> --device 0 --module CNN_CIFAR10 \
    --dataloader DataLoader_cifar10_pat --N 10 --NC 2 --balance True --B 200 --C 1.0 \
    --E 1 --decay 0.999 --step_type bgd --unlearn_cn 1 --save_model True \
    --algorithm FedAvg --R 100 --UR 0 --lr 0.05 --unlearn_pretrain True
  ```

---

### 1.3 Phase 2: Unlearning Stage (Rounds 1–100)
- **Algorithm**: `FedOSD`
- **Unlearning Rounds**: `UR = 100` (Rounds 1–100)
- **Unlearning Base Learning Rate**: `lr = 0.0004`
- **Decay**: `0.999`
- **Gradient Clipping**: `g_clip = 1.0`
- **Unlearned Clients**: `unlearn_cn = 1` (Client 7 for seed 101, 8 for 102, 6 for 103, 0 for 104, 3 for 105)
- **Projection Parameter**: `adap_lam = 100000.0` ($\lambda = 10^5$, enforcing strict hard/one-sided constraints)
- **Client Adaptation**: `adap_weight_eta = 0.0`, `weight_mode = 'uniform'`
- **Step Scale**: `1.0` (for standard methods) or `s* = 0.54` (for Hard-scaled comparator)

---

### 1.4 Phase 3: Post-Recovery Stage (Rounds 101–200)
- **Recovery Rounds**: Rounds 101–200 (`R = 200` total)
- **Recovery Learning Rate**: `r_lr = 0.0001` ($10^{-4}$, calibrated via multi-seed audit on baseline Hard FedOSD)
- **Decay**: `0.999`
- **Gradient Clipping**: `g_clip = 1.0`
- **Online Clients**: Retained clients only (9 retained clients; unlearned client excluded)

---

### 1.5 Baseline Reference: Retrain from Scratch
- **Algorithm**: `FedAvg` (trained on the 9 retained clients from scratch)
- **Total Rounds**: `R = 300` (evaluates at round 200 for end-of-unlearning reference, and round 300 for post-recovery reference)
- **Learning Rate**: `lr = 0.05`
- **Decay**: `0.999`
- **Gradient Clipping**: `g_clip = 1.0`
- **Test Interval**: `--test_interval 5` (evaluates metrics every 5 communication rounds, producing 61 evaluations)
- **Exact Retrain CLI Command**:
  ```bash
  python -u run_UnlearningTask.py --seed <seed> --device 0 --module CNN_CIFAR10 \
    --dataloader DataLoader_cifar10_pat --N 10 --NC 2 --balance True --B 200 --C 1.0 \
    --E 1 --decay 0.999 --step_type bgd --unlearn_cn 1 --save_model True \
    --algorithm FedAvg --retrain True --test_interval 5 --R 300 --UR 0 --lr 0.05
  ```

---

## 2. Frozen Evaluation Methods & Code Paths

### 2.1 Proposed Method: One-sided, unscaled (cap 1.0)
- **CLI Arguments**:
  ```bash
  --algorithm FedOSD --proj_mode one_sided --cap_hard_ratio 1.0 \
  --adap_lam 100000.0 --adap_weight_eta 0.0 --weight_mode uniform \
  --lr 0.0004 --r_lr 0.0001 --decay 0.999 --g_clip 1.0 --R 200 --UR 100
  ```
- **Code Path**: Follows `cap_hard_ratio > 0` path at line 371 in `tfedplat/algorithm/unlearning/FedOSD.py`. If $\|d_{\text{raw}}\| > 1.0 \cdot \|d_{\text{hard}}\|$, caps norm to $\|d_{\text{hard}}\|$.

### 2.2 Primary Comparator: Hard-scaled FedOSD at Matched Scale ($s^* = 0.54$)
- **CLI Arguments**:
  ```bash
  --algorithm FedOSD --proj_mode hard --step_scale 0.54 \
  --adap_lam 100000.0 --adap_weight_eta 0.0 \
  --lr 0.0004 --r_lr 0.0001 --decay 0.999 --g_clip 1.0 --R 200 --UR 100
  ```
- **Displacement Match**: Matches One-sided, unscaled (cap 1.0) displacement to within $\mathbf{0.81\%}$ across seeds 101–105 ($0.2330$ vs $0.2349$).

### 2.3 Secondary Comparator: Hard FedOSD rescaled to $\|g_u\|$
- **CLI Arguments**:
  ```bash
  --algorithm FedOSD --proj_mode hard --step_scale 1.0 \
  --adap_lam 100000.0 --adap_weight_eta 0.0 \
  --lr 0.0004 --r_lr 0.0001 --decay 0.999 --g_clip 1.0 --R 200 --UR 100
  ```

### 2.4 Baseline Ablation: One-sided rescaled to $\|g_u\|$ (Legacy Uncapped)
- **CLI Arguments**:
  ```bash
  --algorithm FedOSD --proj_mode one_sided \
  --adap_lam 100000.0 --adap_weight_eta 0.0 --weight_mode uniform \
  --lr 0.0004 --r_lr 0.0001 --decay 0.999 --g_clip 1.0 --R 200 --UR 100
  ```

---

### 2.5 Code Path Documentation (`tfedplat/algorithm/unlearning/FedOSD.py` lines 371–380)

```python
# Line 371: Capped / Unscaled code path (cap_hard_ratio > 0)
if self.cap_hard_ratio > 0:
    cap = self.cap_hard_ratio * d_hard_norm
    if d_raw_norm > cap:
        d = d * (cap / (d_raw_norm + 1e-12))
# Line 375: Absolute norm cap branch
elif self.d_norm_cap > 0 and d_raw_norm > self.d_norm_cap:
    d = d / d_raw_norm * self.d_norm_cap
# Line 377: Legacy rescaling branch (cap_hard_ratio <= 0)
elif d_raw_norm > 1e-8:
    d = d / d_raw_norm * g_norm  # rescales d to ||g_u||
else:
    d = gu
```

- **Path A (`self.cap_hard_ratio > 0`, lines 371–374)**:
  - Invoked when `--cap_hard_ratio` is explicitly passed (e.g. `1.0`, `2.0`, `3.0`, `1e9`).
  - For `cap 1.0`: if $\|d_{\text{raw}}\| > \|d_{\text{hard}}\|$, $d$ is clipped down to $\|d_{\text{hard}}\|$.
  - For `cap >= 2.0` (including `1e9`): because $\|d_{\text{raw}}\| \approx 20.8 \ll \text{cap} \cdot \|d_{\text{hard}}\|$, the cap is never exceeded. $d$ remains untouched as the **pure unconstrained NNLS solution** $d_{\text{raw}}$.
- **Path B (Legacy branch `self.cap_hard_ratio <= 0`, lines 377–378)**:
  - Default when `--cap_hard_ratio` is omitted (defaults to `-1.0`).
  - Both legacy Hard FedOSD and legacy Uncapped One-Sided execute this branch.
  - Line 378 rescales $d$ to match the norm of the unlearned client's local gradient $\|g_u\|$. At round 1, $\|g_u\| \approx 27.84$ vs $\|d_{\text{raw}}\| \approx 20.81$ ($+33.8\%$ step enlargement).

---

## 3. Empirical Characteristics from Stage 2 Audit (Seeds 101–105)

1. **Parameter Displacement Matching (Regime R2)**:
   - Baseline Hard FedOSD: Mean displacement $= 0.3225$ (Round 100) / $0.3233$ (Round 200)
   - One-sided, unscaled (cap 1.0): Mean displacement $= 0.2343$ (Round 100) / $0.2349$ (Round 200)
   - Displacement Ratio (Capped / Hard): $0.7265$ ($72.65\%$)
   - Hard-scaled at $s^*=0.54$: Mean displacement $= 0.2323$ (Round 100) / $0.2330$ (Round 200)
   - Displacement Match Difference: $\mathbf{0.85\%}$ at Round 100 and $\mathbf{0.81\%}$ at Round 200 (strictly within the $\pm 2\%$ tolerance).

2. **Trajectory Displacement Scaling Law**:
   - Parameter trajectory displacement accumulates sublinearly with step scale: $D(s) \propto s^\alpha$ where $\alpha \approx 0.473 \approx 0.5$ ($D \propto \sqrt{s}$).
   - In Regime R1 (pretrain lr 0.01, $r_{\text{lr}}=10^{-6}$): Capped/Hard ratio is $0.505$. Hard-scaled at $s=0.50$ produces displacement $0.3055$ (+42.4% above target $0.2145$). Hard-scaled at $s=0.24$ produces $0.2014$ (-6.1%). Refined matching at $s^*_{\text{R1}} = 0.27$ produces displacement $0.2162$ (+0.79% vs $0.2145$), achieving exact matching within the $\pm 2\%$ window.

3. **MIA Empirical Noise Floor**:
   - Pretrained model MIA ROC-AUC across 5 partitions: Mean $= 0.5100 \pm 0.0152$
   - Dual-Retrain from scratch ($R=300$) on identical data partitions with independent model initializations (`model_init_seed` 201–205 vs default 101–105):
     - Seed 101: Init 1 = 0.4953, Init 2 = 0.5017, $|\Delta| = 0.0064$
     - Seed 102: Init 1 = 0.5224, Init 2 = 0.5099, $|\Delta| = 0.0125$
     - Seed 103: Init 1 = 0.5063, Init 2 = 0.5127, $|\Delta| = 0.0064$
     - Seed 104: Init 1 = 0.5199, Init 2 = 0.5082, $|\Delta| = 0.0117$
     - Seed 105: Init 1 = 0.4999, Init 2 = 0.4968, $|\Delta| = 0.0032$
   - **Empirical Between-Initialization Noise Floor**:
     - $\text{Mean } |\Delta| = \mathbf{0.0080}$ ($0.80$ percentage points)
     - $\text{Max } |\Delta| = \mathbf{0.0125}$ ($1.25$ percentage points)
     - Standard Deviation $= 0.0035$
   - **Protocol Threshold**: Any unlearning MIA ROC-AUC difference $< 0.0125$ relative to Retrain is within the stochastic training noise of Retrain.

4. **Solver Computational Overhead**:
   - Hard closed-form solve: **Median $= 2.85\text{ ms/round}$** (mean: $8.71\text{ ms/round}$)
   - Capped NNLS solve: **Median $= 2.01\text{ ms/round}$** (mean: $3.85\text{ ms/round}$)
   - Both account for $< 0.1\%$ of total FL round time.

5. **Exact Applied Path Lengths and Triangle Inequality Verification**:
   - Baseline Hard FedOSD (Seed 101): Exact applied path length $\sum \|d_{\text{applied}}\| = \mathbf{1040.8381}$. Bound $\text{lr} \times 1040.8381 = 0.4163 > \text{displacement } 0.3640$ (ratio $0.8743$, strictly satisfying the triangle inequality).
   - Baseline Hard FedOSD (Seed 999): Exact applied path length $\sum \|d_{\text{applied}}\| = \mathbf{934.4363}$. Bound $\text{lr} \times 934.4363 = 0.3738 > \text{displacement } 0.3243$ (ratio $0.8676$, strictly satisfying the triangle inequality).
   - Hard-scaled $s=0.54$ (Seed 101): Exact applied path length $\sum \|d_{\text{applied}}\| = \mathbf{749.0776}$. Bound $\text{lr} \times 749.0776 = 0.2996 > \text{displacement } 0.2691$ (ratio $0.8981$, strictly satisfying the triangle inequality).
   - Hard-scaled $s=0.54$ (Seed 999): Exact applied path length $\sum \|d_{\text{applied}}\| = \mathbf{681.6123}$. Bound $\text{lr} \times 681.6123 = 0.2726 > \text{displacement } 0.2430$ (ratio $0.8913$, strictly satisfying the triangle inequality).
   - Capped One-Sided cap 1.0 (Seed 999): Exact applied path length $\sum \|d_{\text{applied}}\| = \mathbf{645.1115}$. Bound $\text{lr} \times 645.1115 = 0.2580 > \text{displacement } 0.2400$ (ratio $0.9301$, strictly satisfying the triangle inequality).
   - **Historical Logging Artifact**: Logs recorded before the logging fix recorded unscaled $\|d_{\text{hard}}\|$ for Hard and unscaled $\|g_u\|$ for Hard-scaled. The true physical descent steps applied to the network weights always strictly adhered to the triangle inequality.

---

## 4. Hardware and Execution Environment Specification

To ensure exact reproducibility across machines and executions, all confirmatory experiments are frozen to the following software and hardware environment:
- **Operating System**: Windows 11
- **Python Version**: 3.12.10
- **PyTorch Version**: 2.6.0+cu124 (`torch==2.6.0+cu124`)
- **Torchvision Version**: 0.21.0+cu124 (`torchvision==0.21.0+cu124`)
- **NumPy Version**: 2.5.4 (`numpy==2.5.4`)
- **SciPy Version**: 1.18.1 (`scipy==1.18.1`)
- **CUDA Version**: 12.4
- **cuDNN Version**: 90100 (cuDNN 9.1.0)
- **NVIDIA GPU Driver**: 621.01
- **GPU Architecture**: NVIDIA GeForce RTX 3050 6GB Laptop GPU (Ampere GA107, 6GB GDDR6 VRAM, Compute Capability 8.6)
- **Execution Mode**: Sequential single-job execution ($S=0.89$), logging GPU temperature and graphics/memory clocks per run
- **Deterministic Settings**:
  - `torch.manual_seed(seed)`
  - `torch.cuda.manual_seed_all(seed)`
  - `np.random.seed(seed)`
  - `random.seed(seed)`
  - `torch.backends.cudnn.deterministic = True`
  - `torch.backends.cudnn.benchmark = False`
  - `torch.use_deterministic_algorithms(True, warn_only=True)`
  - `os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"`
