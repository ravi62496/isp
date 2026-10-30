@echo off
REM ============================================================================
REM Seed 2 Unlearning Verification Experiments (Hard vs Uniform vs Adaptive)
REM ============================================================================

set BASE_CMD=python run_UnlearningTask.py --seed 2 --device 0 --module CNN_CIFAR10 --algorithm FedOSD --dataloader DataLoader_cifar10_pat --N 10 --NC 2 --balance True --B 200 --C 1.0 --R 200 --UR 100 --E 1 --decay 0.999 --step_type bgd --unlearn_cn 1 --save_model True --lr 0.0004 --r_lr 1e-6

echo ============================================================================
echo [1/3] Seed 2: Original Hard FedOSD Baseline (lambda=100000.0, eta=0.0)
echo ============================================================================
%BASE_CMD% --adap_lam 100000.0 --adap_weight_eta 0.0

echo ============================================================================
echo [2/3] Seed 2: Uniform Soft Constraint Baseline (lambda=20.0, eta=0.0)
echo ============================================================================
%BASE_CMD% --adap_lam 20.0 --adap_weight_eta 0.0

echo ============================================================================
echo [3/3] Seed 2: Adaptive FedOSD (lambda=20.0, eta=0.4)
echo ============================================================================
%BASE_CMD% --adap_lam 20.0 --adap_weight_eta 0.4

echo ============================================================================
echo Seed 2 Experiments Completed! Running multi-seed statistical aggregation...
echo ============================================================================
python evaluate_multiseed.py

