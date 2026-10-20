#!/usr/bin/env bash
# Run from the repository root:  bash scripts/queue_master.sh
# Single master queue (one process -> no slot races). At most MAX live GPU jobs. A job is skipped when its
# checkpoint exists, waited for (not restarted) when a live process already writes to its run dir.
export PYTHONIOENCODING=utf-8
MAX=2
running() { powershell -NoProfile -ExecutionPolicy Bypass -File scripts/jobs.ps1 | tr -d ''; }
alive()   { powershell -NoProfile -ExecutionPolicy Bypass -File scripts/jobs.ps1 -Name "$1" | tr -d ''; }
done_()   { [ -f "runs/$1/last.pt" ]; }   # every trainer writes last.pt only when it finishes
need()    { for f in "$@"; do until [ -f "$f" ]; do sleep 30; done; done; }
freemb()   { nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' '; }
# a slot needs both: fewer than MAX live jobs AND >= 3 GB really free (exited processes can keep GPU memory)
slot()    { until [ "$(running)" -lt "$(cat runs/.max 2>/dev/null || echo $MAX)" ] && [ "$(freemb)" -ge 3000 ]; do sleep 30; done; }
go() {  # go NAME script args...   (NAME = run dir; the job must write runs/NAME/{best,last}.pt)
  local n=$1; shift
  done_ "$n" && { echo "skip $n"; return; }
  [ "$(alive "$n")" -gt 0 ] && { echo "$(date +%T) $n already running (counts as a slot)"; return; }
  slot; echo "$(date +%T) start $n"; python "$@" > "runs/$n.log" 2>&1 & sleep 150   # let it allocate before the next memory check
}
A="--iters0 2000 --iters_round 1000 --rounds 2 --k 40"

go sp_unc_div_r12 train_sparse.py --strategy unc_div --nms_r 12 --out runs/sp_unc_div_r12 $A
go sp_unc_div_r48 train_sparse.py --strategy unc_div --nms_r 48 --out runs/sp_unc_div_r48 $A
go sam_drive_tex train_sam.py --domain drive --sup dense --polarity --texture 6 --out runs/sam_drive_tex
for s in 1 2; do
  go dice_s$s   train.py --loss dice   --seed $s --out runs/dice_s$s
  go cldice_s$s train.py --loss cldice --seed $s --lam 0.5 --out runs/cldice_s$s
done
for s in 0 1 2; do
  go unet_roads_clk_s$s train_sparse.py --domain roads --strategy none --clicks --seed $s --out runs/unet_roads_clk_s$s --iters0 2000 --rounds 0 --cape_from 600 --bs 2
done
for s in 1 2; do
  go unet_roads_sp_s$s train_sparse.py --domain roads --strategy none --seed $s --out runs/unet_roads_sp_s$s --iters0 2000 --rounds 0 --cape_from 600 --bs 2
  go sam_roads_scratch_s$s train_sam.py --domain roads --sup sparse --seed $s --out runs/sam_roads_scratch_s$s --iters 2000 --cape_from 600
done
for s in 1 2; do
  need runs/dice_s$s/last.pt
  go cape_s$s train.py --loss cape --init runs/dice_s$s/last.pt --warmup 0 --iters 2000 --lr 3e-4 --seed $s --out runs/cape_s$s
done
need runs/sam_drive_tex/last.pt
for s in 0 1 2; do
  go sam_roads_fs_tex_s$s train_sam.py --domain roads --sup sparse --init runs/sam_drive_tex/best.pt --seed $s --out runs/sam_roads_fs_tex_s$s --iters 2000 --cape_from 600
done
for s in 1 2; do
  for st in none random unc unc_div; do
    go sp_${st}_s$s train_sparse.py --strategy $st --seed $s --out runs/sp_${st}_s$s $A
  done
done
# NMS radius selected on validation (clDice tie -> lowest Betti error): r = 48
for s in 1 2; do
  go sp_unc_div_r48_s$s train_sparse.py --strategy unc_div --nms_r 48 --seed $s --out runs/sp_unc_div_r48_s$s $A
done
for s in 0 1 2; do
  go sp_unc_div_noisy_s$s train_sparse.py --strategy unc_div --nms_r 48 --seed $s --jitter 2 --flip 0.1 --out runs/sp_unc_div_noisy_s$s $A
done
until [ "$(running)" -eq 0 ]; do sleep 30; done
echo "$(date +%T) training done"
[ -f results/tta_select_sam_drive_pol.json ] || python tta_select.py --n 30 --ckpt runs/sam_drive_pol/best.pt > runs/tta_select_pol.log 2>&1
[ -f results/tta_select_sam_drive_tex.json ] || python tta_select.py --n 30 --ckpt runs/sam_drive_tex/best.pt > runs/tta_select_tex.log 2>&1
python eval_seeds.py --group drive > runs/eval_seeds_drive.log 2>&1
python eval_seeds.py --group roads > runs/eval_seeds_roads.log 2>&1
python aggregate_seeds.py > runs/aggregate.log 2>&1
echo MASTER_DONE
