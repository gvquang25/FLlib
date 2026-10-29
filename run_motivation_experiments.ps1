# Script chay tron goi 4 thuc nghiem Figure 1 ICML 2026
$ErrorActionPreference = "Stop"
$env:PYTHONUNBUFFERED = "1"
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path

Set-Location "$scriptDir\system"
$PYTHON = "$scriptDir\venv\Scripts\python.exe"

Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host " [1/4] Chay FedSCD (Ours)..." -ForegroundColor Green
Write-Host "==========================================================" -ForegroundColor Cyan
& $PYTHON main.py -data Cifar10 -m cnn -algo FedSCD -gr 100 -did 0 -lbs 32 -tbs 512 -lr 0.01 -nc 20 --noniid --alpha_dirich 0.5 -lam 0.1 -sd 42 -log --proj_mode shared --lamda_virt_local 0.0

Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host " [2/4] Chay FedProto..." -ForegroundColor Green
Write-Host "==========================================================" -ForegroundColor Cyan
& $PYTHON main.py -data Cifar10 -m cnn -algo FedProto -gr 100 -did 0 -lbs 32 -tbs 512 -lr 0.01 -nc 20 --noniid --alpha_dirich 0.5 -lam 0.1 -sd 42 -log

Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host " [3/4] Chay FedTGP..." -ForegroundColor Green
Write-Host "==========================================================" -ForegroundColor Cyan
& $PYTHON main.py -data Cifar10 -m cnn -algo FedTGP -gr 100 -did 0 -lbs 32 -tbs 512 -lr 0.01 -nc 20 --noniid --alpha_dirich 0.5 -lam 0.1 -sd 42 -log

Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host " [4/4] Chay CCVR..." -ForegroundColor Green
Write-Host "==========================================================" -ForegroundColor Cyan
& $PYTHON main.py -data Cifar10 -m cnn -algo CCVR -gr 100 -did 0 -lbs 32 -tbs 512 -lr 0.01 -nc 20 --noniid --alpha_dirich 0.5 -lam 0.1 -sd 42 -log

Set-Location $scriptDir
Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host " [HOAN TAT] Xuat bieu do Figure 1 (figures/motivation.pdf)..." -ForegroundColor Yellow
Write-Host "==========================================================" -ForegroundColor Cyan
& $PYTHON plot_motivation.py
