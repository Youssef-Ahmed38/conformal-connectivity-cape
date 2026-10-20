"""Sanity checks: (1) GT vs itself -> conn == y everywhere; (2) a cut vessel flips pairs to 0;
(3) conformal coverage on synthetic exchangeable data >= 1-alpha per class; (4) auroc vs brute force."""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
from ccape.data import load
from ccape.topo_uq import pair_scores
from ccape.conformal import calibrate, predict_sets, evaluate_sets, auroc

d = load("calib"); gt, fov = d["lab"][0] > 0, d["fov"][0] > 0
t = time.time()
P = pair_scores(gt, fov, gt.astype(float), np.repeat(gt[None].astype(float), 20, 0))
print(f"pairs {len(P['y'])}, disconnected {1-P['y'].mean():.3f}, time {time.time()-t:.2f}s")
assert all((P[k] == P["y"]).all() for k in ("det", "bn", "mc", "mcbn")), "GT vs GT must agree"
cut = gt.copy(); cut[:, 280:283] = False                     # sever every vessel crossing x=280..282
Q = pair_scores(gt, fov, cut.astype(float), cut[None].astype(float))
cross = (np.minimum(Q["u"][:, 1], Q["v"][:, 1]) < 278) & (np.maximum(Q["u"][:, 1], Q["v"][:, 1]) > 285) & Q["y"]
print("pairs crossing the cut:", cross.sum(), "-> still connected:", Q["det"][cross].mean())
assert Q["det"][cross].mean() < 0.05
rng = np.random.default_rng(0)
y = rng.random(40000) < 0.8; s = np.clip(y * 0.7 + 0.15 + rng.normal(0, 0.2, len(y)), 0, 1)
q = calibrate(s[:20000], y[:20000], 0.1); m = evaluate_sets(predict_sets(s[20000:], q), y[20000:], s[20000:] > 0.5)
print({k: round(v, 3) for k, v in m.items()}); assert m["cov_connected"] >= 0.89 and m["cov_disconnected"] >= 0.89
a, b = rng.random(300), rng.random(300) < 0.3
bf = np.mean([(x > z) + 0.5 * (x == z) for x in a[b] for z in a[~b]])
assert abs(auroc(a, b) - bf) < 1e-9; print("all checks passed")
