@echo off
REM ============================================================================
REM FedOSD Adaptive Hyperparameter Sweep & Benchmarking Script (GPU-enabled)
REM ============================================================================

set BASE_CMD=python run_UnlearningTask.py --seed 1 --device 0 --module CNN_CIFAR10 --algorithm FedOSD --dataloader DataLoader_cifar10_pat --N 10 --NC 2 --balance True --B 200 --C 1.0 --R 200 --UR 100 --E 1 --decay 0.999 --step_type bgd --unlearn_cn 1 --save_model True --lr 0.0004 --r_lr 1e-6

echo ============================================================================
echo [1/5] Running Adaptive FedOSD (lambda=100.0, weight_eta=0.05)
echo ============================================================================
%BASE_CMD% --adap_lam 100.0 --adap_weight_eta 0.05

echo ============================================================================
echo [2/5] Running Adaptive FedOSD (lambda=100.0, weight_eta=0.15)
echo ============================================================================
%BASE_CMD% --adap_lam 100.0 --adap_weight_eta 0.15

echo ============================================================================
echo [3/5] Running Adaptive FedOSD (lambda=50.0, weight_eta=0.05)
echo ============================================================================
%BASE_CMD% --adap_lam 50.0 --adap_weight_eta 0.05

echo ============================================================================
echo [4/5] Running Adaptive FedOSD (lambda=50.0, weight_eta=0.15)
echo ============================================================================
%BASE_CMD% --adap_lam 50.0 --adap_weight_eta 0.15

echo ============================================================================
echo [5/5] Running Adaptive FedOSD (lambda=500.0, weight_eta=0.05)
echo ============================================================================
%BASE_CMD% --adap_lam 500.0 --adap_weight_eta 0.05

echo ============================================================================
echo Sweeps completed! Analyzing results...
echo ============================================================================
python evaluate_experiments.py
