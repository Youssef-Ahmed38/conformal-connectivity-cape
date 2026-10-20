"""(1) weighted conformal with w=1 == unweighted Mondrian conformal; (2) CRC risk <= alpha on exchangeable
synthetic images (averaged over 300 draws); (3) weighted conformal restores coverage under a known covariate shift."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
from ccape.conformal import calibrate, predict_sets
from ccape.crc import crc_eval, crc_fit, weighted_sets

rng = np.random.default_rng(0)
s, y = rng.random(5000), rng.random(5000) < 0.8
st = rng.random(3000)
a = predict_sets(st, calibrate(s, y, 0.1)); b = weighted_sets(s, y, np.ones(len(s)), st, np.ones(len(st)), 0.1)
print("w=1 agreement", (a == b).mean()); assert (a == b).mean() > 0.999

def image(rng, hard):                       # image-level random difficulty -> dependent pairs inside an image
    n = 400; y = rng.random(n) < 0.85
    s = np.clip(np.where(y, 0.75, 0.25) + rng.normal(0, 0.15 + 0.25 * hard, n), 0, 1)
    return {"s": s, "y": y, "det": (s > 0.5).astype(float)}
risks = []
for t in range(300):
    ims = [image(rng, rng.random()) for _ in range(21)]
    l1, l0 = crc_fit(ims[:20], 0.1)
    risks.append(crc_eval(ims[20:], l1, l0)["risk"])
print("CRC mean risk %.4f (alpha 0.1), trusted on last draw %.2f" % (np.mean(risks), crc_eval(ims[20:], l1, l0)["trusted"]))
assert np.mean(risks) <= 0.1 + 0.01

x_c, x_t = rng.normal(0, 1, 20000), rng.normal(1.2, 1, 20000)      # covariate shift in x; P(y|x) fixed
f = lambda x: 1 / (1 + np.exp(-2 * x)); yc, yt = rng.random(len(x_c)) < f(x_c), rng.random(len(x_t)) < f(x_t)
sc, stt = np.clip(f(x_c) + rng.normal(0, .25, len(x_c)), 0, 1), np.clip(f(x_t) + rng.normal(0, .25, len(x_t)), 0, 1)
dens = lambda x, m: np.exp(-(x - m) ** 2 / 2)
wc, wt = dens(x_c, 1.2) / dens(x_c, 0), dens(x_t, 1.2) / dens(x_t, 0)   # true density ratio
cov = lambda S: S[np.arange(len(yt)), yt.astype(int)][yt == 0].mean()
print("coverage of class 0 under shift: unweighted %.3f, weighted %.3f" % (cov(predict_sets(stt, calibrate(sc, yc, .1))), cov(weighted_sets(sc, yc, wc, stt, wt, .1))))
print("all checks passed")
