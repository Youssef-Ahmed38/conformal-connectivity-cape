"""Label-free test-time canonicalisation for zero-shot transfer (vessels -> roads).

Candidates per tile: polarity {as-is, inverted} x scale {1, s_hessian, 0.5}, where s_hessian comes from the
scale-space ridge estimate (ccape/canon.py). Selection rules compared (no labels used by any of them):
  oracle-fixed : the best single global variant chosen with labels (upper reference, not a method)
  mean-ens     : average of the two polarities at scale 0.5 (previous fix)
  min-entropy  : per tile, the variant whose prediction has the lowest mean binary entropy in the FOV
  min-ent-ens  : per tile, average the two lowest-entropy variants
    python tta_select.py --n 30 --device cpu
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from ccape.canon import estimate, reference
from ccape.data import load, load_roads
from ccape.metrics import all_metrics
from ccape.sam import Domain, TopoSAM
from zeroshot_probe import transform, up


def entropy(p, m):
    p = np.clip(p[m], 1e-6, 1 - 1e-6)
    return float(-(p * np.log(p) + (1 - p) * np.log(1 - p)).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--ckpt", default="runs/sam_drive_pol/best.pt")
    a = ap.parse_args()
    dev = a.device
    d0 = load_roads("test"); d0 = {k: v[:a.n] for k, v in d0.items()}
    gt, fv = d0["lab"] > 0, d0["fov"] > 0
    tr = load("train")
    ref = reference(tr["rgb"], tr["fov"] > 0)
    s_h = float(np.clip(ref[1] / np.median([estimate(x)[1] for x in d0["rgb"]]), 0.25, 1.0))
    scales = sorted({1.0, round(s_h, 2), 0.5}, reverse=True)
    m = TopoSAM().to(dev); m.load_state_dict(torch.load(a.ckpt, map_location=dev), strict=False)
    P = {}
    for inv in (False, True):
        for sc in scales:
            dom = Domain(m, transform(d0, inv, sc), dev, seed=3, flips=1)
            P[(inv, sc)] = [up(dom.predict(m, i)[0], 1024) for i in range(a.n)]
            print("done", inv, sc, flush=True)
    V = list(P)
    H = np.array([[entropy(P[v][i], fv[i]) for v in V] for i in range(a.n)])     # (n, variants)
    score = lambda ps: {k: float(np.mean([all_metrics(p, g, f)[k] for p, g, f in zip(ps, gt, fv)])) for k in ("dice", "cldice", "betti0")}
    res = {"hessian_scale": s_h, "reference": ref}
    for v in V:
        res[f"fixed inv={v[0]} s={v[1]}"] = score(P[v])
    res["mean-ens (both polarities, s=0.5)"] = score([(x + y) / 2 for x, y in zip(P[(False, 0.5)], P[(True, 0.5)])])
    res["min-entropy (per tile)"] = score([P[V[int(np.argmin(H[i]))]][i] for i in range(a.n)])
    res["min-ent-ens (2 lowest)"] = score([np.mean([P[V[j]][i] for j in np.argsort(H[i])[:2]], 0) for i in range(a.n)])
    res["chosen_variants"] = [str(V[int(np.argmin(H[i]))]) for i in range(a.n)]
    for k, v in res.items():
        if isinstance(v, dict):
            print(f"{k:42s}", json.dumps({kk: round(x, 3) for kk, x in v.items()}))
    print("hessian scale", s_h)
    Path(f"results/tta_select_{Path(a.ckpt).parent.name}.json").write_text(json.dumps(res, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
