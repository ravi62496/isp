#!/usr/bin/env bash
set -e

echo "============================================================================"
echo "[Goal 4] CIFAR-100 Federated Unlearning Generalization Benchmark"
echo "============================================================================"

echo ""
echo "[1/4] Pre-training Global FL Model on CIFAR-100 (FedAvg, 500 rounds)"
echo "----------------------------------------------------------------------------"
python run_UnlearningTask.py --seed 1 --device 0 --module CNN_CIFAR10 --algorithm FedAvg --dataloader DataLoader_cifar100_pat --N 10 --NC 10 --balance True --B 100 --C 1.0 --R 500 --UR 0 --E 1 --lr 0.05 --decay 0.999 --step_type bgd --unlearn_cn 1 --unlearn_pretrain True --save_model True --test_interval 50

echo ""
echo "[2/4] Running Original FedOSD (Hard Orthogonality) on CIFAR-100"
echo "----------------------------------------------------------------------------"
python run_UnlearningTask.py --seed 1 --device 0 --module CNN_CIFAR10 --algorithm FedOSD --dataloader DataLoader_cifar100_pat --N 10 --NC 10 --balance True --B 100 --C 1.0 --R 200 --UR 100 --E 1 --decay 0.999 --step_type bgd --unlearn_cn 1 --save_model True --lr 0.0004 --r_lr 1e-6 --adap_lam 100000.0 --adap_weight_eta 0.0

echo ""
echo "[3/4] Running Soft Uniform Baseline (lambda=20.0, eta=0.0) on CIFAR-100"
echo "----------------------------------------------------------------------------"
python run_UnlearningTask.py --seed 1 --device 0 --module CNN_CIFAR10 --algorithm FedOSD --dataloader DataLoader_cifar100_pat --N 10 --NC 10 --balance True --B 100 --C 1.0 --R 200 --UR 100 --E 1 --decay 0.999 --step_type bgd --unlearn_cn 1 --save_model True --lr 0.0004 --r_lr 1e-6 --adap_lam 20.0 --adap_weight_eta 0.0

echo ""
echo "[4/4] Running Adaptive FedOSD (lambda=20.0, eta=0.4) on CIFAR-100"
echo "----------------------------------------------------------------------------"
python run_UnlearningTask.py --seed 1 --device 0 --module CNN_CIFAR10 --algorithm FedOSD --dataloader DataLoader_cifar100_pat --N 10 --NC 10 --balance True --B 100 --C 1.0 --R 200 --UR 100 --E 1 --decay 0.999 --step_type bgd --unlearn_cn 1 --save_model True --lr 0.0004 --r_lr 1e-6 --adap_lam 20.0 --adap_weight_eta 0.4

echo ""
echo "============================================================================"
echo "CIFAR-100 Generalization Benchmark Completed!"
echo "============================================================================"
