#!/usr/bin/env bash
# Phase-1 models for one seed (run from the repository root):  bash scripts/run_all.sh [seed]
# BCE+Dice and clDice train in parallel; CAPE then fine-tunes the converged BCE+Dice model (protocol of the CAPE authors).
set -e
S=${1:-0}; SUF=$([ "$S" = 0 ] && echo "" || echo "_s$S")
mkdir -p runs
python train.py --loss dice   --seed "$S" --out "runs/dice$SUF"             > "runs/dice$SUF.log"   2>&1 &
python train.py --loss cldice --seed "$S" --out "runs/cldice$SUF" --lam 0.5 > "runs/cldice$SUF.log" 2>&1 &
wait
python train.py --loss cape --seed "$S" --init "runs/dice$SUF/last.pt" --warmup 0 --iters 2000 --lr 3e-4 \
    --out "runs/cape$SUF" > "runs/cape$SUF.log" 2>&1
echo "phase-1 seed $S done"
