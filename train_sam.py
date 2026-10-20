"""Phase 3: TopoSAM training (SAM ViT-B frozen encoder, trainable decoder + refiner, CAPE topology loss).

    python train_sam.py --domain drive --sup dense  --out runs/sam_drive                 # vessels, dense + CAPE
    python train_sam.py --domain roads --sup sparse --init runs/sam_drive/best.pt --out runs/sam_roads_fs
                                                     # few-shot roads: 40 tiles, scribbles only (phase 2 labels)
Prompts are always phase-2 style clicks sampled from scribbles.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from ccape.data import load, load_roads
from ccape.losses import CAPELoss, bce_dice
from ccape.metrics import all_metrics
from ccape.sam import Domain, TopoSAM
from train_sparse import partial_bce


def validate(model, dom):
    ms = []
    for i in range(dom.N):
        p, _ = dom.predict(model, i)
        ms.append(all_metrics(p, dom.d["lab"][i] > 0, dom.d["fov"][i] > 0))
    return {k: float(np.mean([m[k] for m in ms])) for k in ms[0]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", choices=["drive", "roads"], required=True)
    ap.add_argument("--sup", choices=["dense", "sparse"], required=True)
    ap.add_argument("--init", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--iters", type=int, default=3000)
    ap.add_argument("--cape_from", type=int, default=1000)
    ap.add_argument("--lam", type=float, default=1.0)
    ap.add_argument("--bs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--eval_every", type=int, default=250)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--polarity", action="store_true", help="train on both intensity polarities (domain-invariant prior)")
    ap.add_argument("--texture", type=int, default=0, help="number of RandConv texture variants per image")
    a = ap.parse_args()

    torch.manual_seed(a.seed); rng = np.random.default_rng(a.seed)
    dev, out = "cuda", Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    model = TopoSAM().to(dev)
    if a.init:
        model.load_state_dict(torch.load(a.init, map_location=dev), strict=False)  # encoder weights come from SAM
    ld = load if a.domain == "drive" else load_roads
    tr = Domain(model, ld("train"), dev, seed=a.seed, polarity=a.polarity, texture=a.texture, flips=2 if a.texture else 4)
    va = Domain(model, ld("calib"), dev, seed=a.seed + 1, flips=1)
    groups = [{"params": [p for n, p in model.named_parameters() if p.requires_grad and not n.startswith("refiner")], "lr": a.lr},
              {"params": model.refiner.parameters(), "lr": 10 * a.lr}]
    opt = torch.optim.AdamW(groups, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, [a.lr, 10 * a.lr], total_steps=a.iters, pct_start=0.05)
    cape = CAPELoss()
    best, log, t0 = -1.0, [], time.time()
    for it in range(1, a.iters + 1):
        model.train(); model.sam.vision_encoder.eval()
        idx = rng.choice(tr.N, a.bs, replace=False)
        e, x, pts, lab, y, ps, ng = tr.batch(idx, int(rng.integers(len(tr.emb))), int(rng.integers(5, 21)), rng)
        with torch.autocast("cuda", torch.bfloat16):
            logits = model(e, x, pts, lab, tr.scale, (tr.H, tr.W)).float()
        p = torch.sigmoid(logits)
        if a.sup == "dense":
            loss = bce_dice(logits, y)
            topo = cape(p, y) if it > a.cape_from else torch.zeros((), device=dev)
        else:
            loss = partial_bce(logits, ps, ng)
            topo = cape(p, ps.float()) if it > a.cape_from else torch.zeros((), device=dev)
        (loss + a.lam * topo).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); opt.zero_grad(set_to_none=True); sched.step()
        if it % a.eval_every == 0 or it == a.iters:
            m = validate(model, va) if a.sup == "dense" else {}
            rec = {"it": it, "loss": float(loss), "topo": float(topo), "sec": round(time.time() - t0), **m}
            log.append(rec); print(json.dumps(rec), flush=True)
            (out / "log.json").write_text(json.dumps(log, indent=1))
            score = (m["dice"] + m["cldice"]) / 2 if m else it        # sparse: no dense labels -> keep last
            if score > best and it > a.cape_from:
                best = score
                torch.save({k: v for k, v in model.state_dict().items() if "vision_encoder" not in k}, out / "best.pt")
    torch.save({k: v for k, v in model.state_dict().items() if "vision_encoder" not in k}, out / "last.pt")


if __name__ == "__main__":
    main()
