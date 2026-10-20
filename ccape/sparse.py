"""Phase 2: sparse supervision.

(a) Scribbles: simulated annotator strokes. Positive strokes walk along the GT centreline; negative strokes are
    straight segments >= 4 px from any vessel. Budget = fraction of GT skeleton length (same for both).
(b) Connectivity clicks: the annotator sees two points and answers "connected?" (yes / no / not a vessel).
    Which pairs to ask is chosen by phase-1 uncertainty (mcbn) or at random (baseline).
"""
import numpy as np
from scipy.ndimage import binary_dilation, distance_transform_edt
from skimage.draw import line
from skimage.morphology import skeletonize

from .topo_uq import LEVELS, OFF, connectivity, sample_pairs

N8 = np.array([(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)])


def scribbles(gt, fov, frac, seed=0, stroke=40, neg_mult=5, neg_margin=2, jitter=0):
    """-> pos (H,W) bool, neg (H,W) bool. Positive strokes: greedy walks on the skeleton pixel graph.
    Background strokes are cheap to draw (no precision needed), so they get neg_mult x the positive budget and
    a 3-px brush; they keep neg_margin px away from any vessel."""
    rng = np.random.default_rng(seed)
    sk = skeletonize(gt)
    H, W = sk.shape
    budget = int(frac * sk.sum())
    pos = np.zeros_like(sk)
    ys, xs = np.nonzero(sk)
    while pos.sum() < budget:
        k = rng.integers(len(ys)); y, x = ys[k], xs[k]
        if pos[y, x]:
            continue
        for _ in range(stroke):
            pos[y, x] = True
            nb = [(y + a, x + b) for a, b in N8[rng.permutation(8)]
                  if 0 <= y + a < H and 0 <= x + b < W and sk[y + a, x + b] and not pos[y + a, x + b]]
            if not nb:
                break
            y, x = nb[0]
    far = (distance_transform_edt(~gt) > neg_margin + 1) & fov   # +1: room for the 3-px brush
    neg = np.zeros_like(sk)
    fy, fx = np.nonzero(far)
    tries = 0
    while neg.sum() < neg_mult * budget and tries < 100_000:
        tries += 1
        k = rng.integers(len(fy)); t = rng.uniform(0, np.pi)
        y1, x1 = int(fy[k] + stroke * np.sin(t)), int(fx[k] + stroke * np.cos(t))
        if not (0 <= y1 < H and 0 <= x1 < W):
            continue
        rr, cc = line(fy[k], fx[k], y1, x1)
        if far[rr, cc].all():
            neg[rr, cc] = True
    if jitter:                                     # noisy annotator: every stroke drawn off-centre by <= jitter px
        from scipy.ndimage import label as cc
        L, n = cc(pos, np.ones((3, 3)))
        out = np.zeros_like(pos)
        for c in range(1, n + 1):
            yy, xx = np.nonzero(L == c)
            dy, dx = rng.integers(-jitter, jitter + 1, 2)
            out[np.clip(yy + dy, 0, H - 1), np.clip(xx + dx, 0, W - 1)] = True
        pos = out
    return binary_dilation(pos), binary_dilation(neg) & fov


NMS_R = 24  # px: two queries closer than this ask about the same gap


def _spatial_nms(mid, k, r):
    """Greedy diversity: walk candidates in priority order, keep one if no kept midpoint is within r.
    KD-tree on all candidates, O(n log n); returns indices into mid (priority order preserved)."""
    from scipy.spatial import cKDTree
    tree, dead, keep = cKDTree(mid), np.zeros(len(mid), bool), []
    for i in range(len(mid)):
        if dead[i]:
            continue
        keep.append(i)
        if len(keep) == k:
            break
        dead[tree.query_ball_point(mid[i], r)] = True
    return np.array(keep, int)


def query_pairs(p_det, p_mc, gt, fov, k, strategy, rng, nms_r=NMS_R, flip=0.0):
    """Pick k candidate pairs on the *predicted* skeleton, ask the oracle (GT). -> (u, v, ans) with
    ans = 1 connected, 0 disconnected (both on vessels), -1 = an endpoint is not a vessel (negative click)."""
    u, v = sample_pairs((p_det > 0.3) & fov, max_pairs=6000, seed=int(rng.integers(1 << 30)))
    if len(u) == 0:
        return u, v, np.zeros(0, int)
    if strategy in ("unc", "unc_div"):
        lv = LEVELS[:, None, None]
        maps = ((p_mc[:, None] > lv[None]) & fov).reshape(-1, *gt.shape)
        s = connectivity(maps, u, v).mean(0)              # mcbn score of phase 1
        order = np.argsort(np.abs(2 * s - 1) + 1e-6 * rng.random(len(s)))
        sel = order[:k] if strategy == "unc" else _spatial_nms(((u + v) / 2)[order], k, nms_r)
        sel = order[sel] if strategy == "unc_div" else sel
    else:
        sel = rng.choice(len(u), min(k, len(u)), replace=False)
    u, v = u[sel], v[sel]
    on = lambda p: gt[np.clip(p[:, None, 0] + OFF[:, 0], 0, gt.shape[0] - 1),
                      np.clip(p[:, None, 1] + OFF[:, 1], 0, gt.shape[1] - 1)].any(1)
    y = connectivity(gt[None], u, v)[0]
    ans = np.where(on(u) & on(v), y.astype(int), -1)
    if flip > 0:                                   # noisy annotator: wrong yes/no answer with probability flip
        f = (ans >= 0) & (rng.random(len(ans)) < flip)
        ans[f] = 1 - ans[f]
    return u, v, ans
