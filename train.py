"""Train a U-Net on DRIVE with one of: dice (BCE+Dice), cldice (+soft clDice), cape (+official CAPE after warm-up).

    python train.py --loss cape --out runs/cape
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from ccape.data import CropSampler, load
from ccape.losses import CAPELoss, bce_dice, soft_cldice
from ccape.metrics import all_metrics
from ccape.model import UNet


@torch.no_grad()
def predict(model, img, dev):
    """(N,3,h,w) any size -> (N,h,w) probabilities."""
    model.eval()
    h, w = img.shape[-2:]
    x = F.pad(torch.from_numpy(img).to(dev), (0, -w % 16, 0, -h % 16))
    with torch.autocast("cuda", torch.bfloat16):
        return np.concatenate([torch.sigmoid(model(x[i:i + 4]).float())[:, :h, :w].cpu().numpy()
                               for i in range(0, len(x), 4)])


def validate(model, d, dev):
    probs = predict(model, d["img"], dev)
    ms = [all_metrics(p, g, f > 0) for p, g, f in zip(probs, d["lab"], d["fov"])]
    return {k: float(np.mean([m[k] for m in ms])) for k in ms[0]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--loss", choices=["dice", "cldice", "cape"], required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--iters", type=int, default=6000)
    ap.add_argument("--warmup", type=int, default=3000, help="cape: pixel-loss iterations before CAPE switches on")
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--crop", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--lam", type=float, default=0.5, help="weight of the topology term")
    ap.add_argument("--eval_every", type=int, default=250)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--init", default=None, help="start from a pixel-loss checkpoint (CAPE fine-tuning protocol)")
    a = ap.parse_args()

    torch.manual_seed(a.seed); np.random.seed(a.seed)
    dev = "cuda"
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    tr, va = load("train"), load("calib")
    sample = CropSampler(tr, a.crop, dev)
    model = UNet().to(dev)
    if a.init:
        model.load_state_dict(torch.load(a.init, map_location=dev))
    opt =torch.optim.AdamW(model.parameters(), a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=a.iters, pct_start=0.05)
    cape = CAPELoss() if a.loss == "cape" else None

    best, log, t0 = -1.0, [], time.time()
    for it in range(1, a.iters + 1):
        model.train()
        x, y = sample(a.bs)
        with torch.autocast("cuda", torch.bfloat16):
            logits = model(x).float()
        loss = bce_dice(logits, y)
        topo = torch.zeros((), device=dev)
        if a.loss == "cldice":
            topo = soft_cldice(torch.sigmoid(logits), y)
        elif a.loss == "cape" and it > a.warmup:
            topo = cape(torch.sigmoid(logits), y)
        (loss + a.lam * topo).backward()
        opt.step(); opt.zero_grad(set_to_none=True); sched.step()

        if it % a.eval_every == 0 or it == a.iters:
            m = validate(model, va, dev)
            score = (m["dice"] + m["cldice"]) / 2
            rec = {"it": it, "loss": float(loss), "topo": float(topo), "sec": round(time.time() - t0), **m}
            log.append(rec)
            # for CAPE only select checkpoints once the topology term is active
            if score > best and (a.loss != "cape" or it > a.warmup):
                best = score
                torch.save(model.state_dict(), out / "best.pt")
            print(json.dumps(rec), flush=True)
            (out / "log.json").write_text(json.dumps(log, indent=1))
    torch.save(model.state_dict(), out / "last.pt")


if __name__ == "__main__":
    main()
