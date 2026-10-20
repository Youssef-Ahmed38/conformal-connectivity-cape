"""Phase 2: label-efficient CAPE.

Supervision = scribbles (frac of vessel length) + connectivity clicks queried over R rounds.
    python train_sparse.py --strategy unc    --out runs/sp_unc      # phase-1 uncertainty picks the queries
    python train_sparse.py --strategy random --out runs/sp_random   # same budget, random queries
    python train_sparse.py --strategy none   --out runs/sp_none     # scribbles only
No dense label is used for training or checkpoint selection (the final model is saved as best.pt).
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from skimage.morphology import skeletonize

from ccape.data import load, load_roads
from ccape.infer import infer
from ccape.losses import CAPELoss
from ccape.model import UNet
from ccape.sparse import query_pairs, scribbles
from train import validate


def partial_bce(logits, pos, neg):
    l = F.binary_cross_entropy_with_logits(logits, pos.float(), reduction="none")
    return 0.5 * ((l * pos).sum() / pos.sum().clamp_min(1) + (l * neg).sum() / neg.sum().clamp_min(1))


def pad16(x):
    h, w = x.shape[-2:]
    return F.pad(x, (0, -w % 16, 0, -h % 16))


def size_prior(p, fov, lo=0.08, hi=0.16):
    """Constrained-CNN size loss (Kervadec et al., MIDL'19): squared hinge on the predicted foreground fraction
    inside the FOV, outside a loose population band (retinal vessels cover ~12% of the FOV)."""
    frac = (p * fov).sum((1, 2)) / fov.sum((1, 2)).clamp_min(1)
    return (F.relu(lo - frac) ** 2 + F.relu(frac - hi) ** 2).mean()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--strategy", choices=["unc", "unc_div", "random", "none"], required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--frac", type=float, default=0.1, help="scribble length as fraction of vessel skeleton")
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--k", type=int, default=40, help="click-pairs queried per image per round")
    ap.add_argument("--iters0", type=int, default=3000)
    ap.add_argument("--iters_round", type=int, default=1500)
    ap.add_argument("--cape_from", type=int, default=1000)
    ap.add_argument("--lam_cape", type=float, default=1.0)
    ap.add_argument("--lam_pair", type=float, default=1.0)
    ap.add_argument("--lam_size", type=float, default=10.0)
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--pairs_per_img", type=int, default=32)
    ap.add_argument("--T", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--domain", choices=["drive", "roads"], default="drive")
    ap.add_argument("--nms_r", type=float, default=24, help="unc_div: min distance between queried pairs (px)")
    ap.add_argument("--jitter", type=int, default=0, help="noisy annotator: stroke offset (px)")
    ap.add_argument("--flip", type=float, default=0.0, help="noisy annotator: P(wrong yes/no answer)")
    ap.add_argument("--clicks", action="store_true",
                    help="click-prompted U-Net: + 2 heat-map channels of clicks sampled from the scribbles (fair TopoSAM baseline)")
    ap.add_argument("--size_band", nargs=2, type=float, default=None,
                    help="foreground-fraction prior band (default: vessels 0.08-0.16, roads 0.02-0.08)")
    a = ap.parse_args()
    band = a.size_band or ((0.08, 0.16) if a.domain == "drive" else (0.02, 0.08))

    torch.manual_seed(a.seed); rng = np.random.default_rng(a.seed)
    dev, out = "cuda", Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    ld = load if a.domain == "drive" else load_roads
    tr, va = ld("train"), ld("calib")
    N = len(tr["ids"])
    gts, fovs = tr["lab"] > 0, tr["fov"] > 0
    sc = [scribbles(g, f, a.frac, seed=a.seed * 100 + i, jitter=a.jitter) for i, (g, f) in enumerate(zip(gts, fovs))]
    pos = np.stack([s[0] for s in sc]); neg = np.stack([s[1] for s in sc])
    sk_gt = sum(int(skeletonize(g).sum()) for g in gts)
    ann = {"scribble_len_pos": sum(int(skeletonize(q).sum()) for q in pos), "scribble_len_neg": int(neg.sum()),
           "gt_vessel_len": sk_gt, "annotated_px_frac": float((pos | neg).sum() / fovs.sum()),
           "pairs": 0, "pairs_yes": 0, "pairs_no": 0, "pairs_notvessel": 0}
    X = pad16(torch.from_numpy(tr["img"]).to(dev))
    P = pad16(torch.from_numpy(pos).to(dev)); Nn = pad16(torch.from_numpy(neg).to(dev))
    Fv = pad16(torch.from_numpy(fovs).to(dev).float())
    PH, PW = X.shape[-2:]
    pairs = [(np.zeros((0, 2), int), np.zeros((0, 2), int), np.zeros(0, int)) for _ in range(N)]

    model = UNet(cin=5 if a.clicks else 3).to(dev)
    if a.clicks:                                   # validation images get fixed clicks from their own scribbles
        from ccape.clickmap import heatmaps, sample_clicks
        vsc = [scribbles(g > 0, f > 0, a.frac, seed=999 + i) for i, (g, f) in enumerate(zip(va["lab"], va["fov"]))]
        g = torch.Generator(device=dev); g.manual_seed(0)
        vp, vl = sample_clicks(torch.from_numpy(np.stack([q[0] for q in vsc])).to(dev),
                               torch.from_numpy(np.stack([q[1] for q in vsc])).to(dev), 10, g)
        va = dict(va, img=np.concatenate([va["img"], heatmaps(vp, vl, *va["lab"].shape[1:]).cpu().numpy()], 1))
    total = a.iters0 + a.rounds * a.iters_round
    opt = torch.optim.AdamW(model.parameters(), 1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 1e-3, total_steps=total, pct_start=0.05)
    cape = CAPELoss()
    log, it, t0 = [], 0, time.time()

    def train(n_iters):
        nonlocal it
        for _ in range(n_iters):
            it += 1
            model.train()
            idx = rng.choice(N, a.bs, replace=False)
            x, ps, ng, fv = X[idx], P[idx], Nn[idx], Fv[idx]
            fy, fx = rng.random() < 0.5, rng.random() < 0.5
            dims = [d for d, f in ((-2, fy), (-1, fx)) if f]
            if dims:
                x, ps, ng, fv = x.flip(dims), ps.flip(dims), ng.flip(dims), fv.flip(dims)
            x = x * (1 + 0.2 * (torch.rand(a.bs, 1, 1, 1, device=dev) - 0.5))
            if a.clicks:
                cp, cl = sample_clicks(ps > 0, ng > 0, int(rng.integers(5, 21)))
                x = torch.cat([x, heatmaps(cp, cl, PH, PW)], 1)
            with torch.autocast("cuda", torch.bfloat16):
                logits = model(x).float()
            p = torch.sigmoid(logits)
            loss = partial_bce(logits, ps, ng) + a.lam_size * size_prior(p, fv, *band)
            lc = lp = torch.zeros((), device=dev)
            if it > a.cape_from:
                lc = cape(p, ps.float())
                bb, uu, vv, aa = [], [], [], []
                for j, i in enumerate(idx):
                    u, v, an = pairs[i]
                    m = np.nonzero(an >= 0)[0]
                    if len(m) > a.pairs_per_img:
                        m = rng.choice(m, a.pairs_per_img, replace=False)
                    u, v = u[m].copy(), v[m].copy()
                    if fy: u[:, 0], v[:, 0] = PH - 1 - u[:, 0], PH - 1 - v[:, 0]
                    if fx: u[:, 1], v[:, 1] = PW - 1 - u[:, 1], PW - 1 - v[:, 1]
                    bb.append(np.full(len(m), j)); uu.append(u); vv.append(v); aa.append(an[m])
                if sum(map(len, bb)):
                    lp = cape.f.pair_loss(p, np.concatenate(bb), np.concatenate(uu), np.concatenate(vv), np.concatenate(aa))
            (loss + a.lam_cape * lc + a.lam_pair * lp).backward()
            opt.step(); opt.zero_grad(set_to_none=True); sched.step()
            if it % 250 == 0:
                print(json.dumps({"it": it, "bce": float(loss), "cape": float(lc), "pair": float(lp),
                                  "sec": round(time.time() - t0)}), flush=True)

    def record(rnd):
        m = validate(model, va, dev)
        log.append({"round": rnd, "it": it, **ann, **m}); print(json.dumps(log[-1]), flush=True)
        (out / "log.json").write_text(json.dumps(log, indent=1))

    assert not (a.clicks and a.strategy != "none"), "--clicks is a scribble-only baseline"
    train(a.iters0); record(0)
    for r in range(1, a.rounds + 1):
        if a.strategy != "none":
            for i in range(N):
                p_det, p_mc = infer(model, tr["img"][i], a.T, dev)
                u, v, an = query_pairs(p_det, p_mc, gts[i], fovs[i], a.k, a.strategy, rng, a.nms_r, a.flip)
                for q in (u[an < 0], v[an < 0]):          # "not a vessel" -> negative click (3x3)
                    for y, x in q:
                        if not gts[i][max(y - 1, 0):y + 2, max(x - 1, 0):x + 2].any():
                            Nn[i, max(y - 1, 0):y + 2, max(x - 1, 0):x + 2] = True
                pu, pv, pa = pairs[i]
                pairs[i] = (np.concatenate([pu, u]), np.concatenate([pv, v]), np.concatenate([pa, an]))
                ann["pairs"] += len(an); ann["pairs_yes"] += int((an == 1).sum())
                ann["pairs_no"] += int((an == 0).sum()); ann["pairs_notvessel"] += int((an < 0).sum())
        train(a.iters_round); record(r)
    torch.save(model.state_dict(), out / "best.pt")
    torch.save(model.state_dict(), out / "last.pt")   # written only at the very end: marks a finished run


if __name__ == "__main__":
    main()
