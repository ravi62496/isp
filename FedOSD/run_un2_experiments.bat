@echo off
echo ============================================================================
echo [Goal 3] Multi-Client Unlearning Benchmark (UN = 2)
echo ============================================================================

echo.
echo [1/3] Running Hard FedOSD Baseline (UN = 2, Hard Orthogonality)
echo ----------------------------------------------------------------------------
python run_UnlearningTask.py --seed 1 --device 0 --module CNN_CIFAR10 --algorithm FedOSD --dataloader DataLoader_cifar10_pat --N 10 --NC 2 --balance True --B 200 --C 1.0 --R 200 --UR 100 --E 1 --decay 0.999 --step_type bgd --unlearn_cn 2 --save_model True --lr 0.0004 --r_lr 1e-6 --adap_lam 100000.0 --adap_weight_eta 0.0

echo.
echo [2/3] Running Soft Uniform Baseline (UN = 2, lambda = 20.0, eta = 0.0)
echo ----------------------------------------------------------------------------
python run_UnlearningTask.py --seed 1 --device 0 --module CNN_CIFAR10 --algorithm FedOSD --dataloader DataLoader_cifar10_pat --N 10 --NC 2 --balance True --B 200 --C 1.0 --R 200 --UR 100 --E 1 --decay 0.999 --step_type bgd --unlearn_cn 2 --save_model True --lr 0.0004 --r_lr 1e-6 --adap_lam 20.0 --adap_weight_eta 0.0

echo.
echo [3/3] Running Adaptive FedOSD (UN = 2, lambda = 20.0, eta = 0.4)
echo ----------------------------------------------------------------------------
python run_UnlearningTask.py --seed 1 --device 0 --module CNN_CIFAR10 --algorithm FedOSD --dataloader DataLoader_cifar10_pat --N 10 --NC 2 --balance True --B 200 --C 1.0 --R 200 --UR 100 --E 1 --decay 0.999 --step_type bgd --unlearn_cn 2 --save_model True --lr 0.0004 --r_lr 1e-6 --adap_lam 20.0 --adap_weight_eta 0.4

echo.
echo ============================================================================
echo All UN = 2 experiments completed successfully!
echo ============================================================================
