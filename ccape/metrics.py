"""Pixel + topology metrics on binary masks (restricted to the FOV)."""
import numpy as np
from scipy.ndimage import label
from skimage.measure import euler_number
from skimage.morphology import skeletonize

S8 = np.ones((3, 3), bool)


def dice(p, g):
    return 2 * (p & g).sum() / max(p.sum() + g.sum(), 1)


def cldice(p, g):
    sp, sg = skeletonize(p), skeletonize(g)
    tprec = (sp & g).sum() / max(sp.sum(), 1)
    tsens = (sg & p).sum() / max(sg.sum(), 1)
    return 2 * tprec * tsens / max(tprec + tsens, 1e-12)


def betti(m):
    """(b0, b1) for 8-connected foreground: b1 = b0 - Euler characteristic."""
    b0 = label(m, S8)[1]
    return b0, b0 - euler_number(m, connectivity=2)


def betti_err(p, g, patch=64):
    """Mean |b0| and |b1| error over a non-overlapping patch grid (standard protocol)."""
    e0 = e1 = n = 0
    for i in range(0, p.shape[0] - patch + 1, patch):
        for j in range(0, p.shape[1] - patch + 1, patch):
            a, b = betti(p[i:i + patch, j:j + patch]), betti(g[i:i + patch, j:j + patch])
            e0 += abs(a[0] - b[0]); e1 += abs(a[1] - b[1]); n += 1
    return e0 / n, e1 / n


def all_metrics(prob, gt, fov, thr=0.5):
    p, g = (prob > thr) & fov, gt.astype(bool)
    b0, b1 = betti_err(p, g)
    return {"dice": dice(p, g), "cldice": cldice(p, g), "betti0": b0, "betti1": b1}
