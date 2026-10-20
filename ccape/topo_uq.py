"""Path-level connectivity under uncertainty.

Unit of analysis = a pair of anchor points (u, v) on the GT vessel skeleton, 10-30 px apart.
  y(u,v)   = 1 iff u and v are connected through GT vessel pixels inside a local 64x64 tile.
  For any binary map M we ask the same question -> conn_M(u,v).
A wrong conn on a pair is exactly a topology error: y=1 & pred=0 is a *break*, y=0 & pred=1 a *false merge*.

Scores (estimated P[connected]) computed from MC-Dropout samples p_1..p_T and thresholds t_1..t_L:
  det   : conn of the deterministic map at 0.5                        (no uncertainty, 0/1)
  bn    : fraction of thresholds at which the deterministic map connects u,v  (= soft bottleneck / widest path)
  mc    : fraction of MC samples that connect u,v at 0.5
  mcbn  : mean over MC samples of bn                                  (ours: epistemic x bottleneck)

Engineering: all 1 + L*(T+1) maps (L=15) of a tile are labelled in ONE scipy.ndimage.label call on a 3-D stack whose
structuring element has no inter-slice connectivity, so labels never leak across maps.
"""
import numpy as np
from scipy.ndimage import label
from scipy.spatial import cKDTree
from skimage.morphology import skeletonize

TILE, STRIDE = 64, 32
DMIN, DMAX = 10, 30
# 15 log-spaced thresholds (logit-symmetric around 0.5): faint thin vessels live at p ~ 0.01-0.1
LEVELS = np.array([0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.98, 0.99])
I05 = int(np.nonzero(LEVELS == 0.5)[0][0])
S3 = np.zeros((3, 3, 3), bool); S3[1] = True         # 8-connectivity inside a slice only
_r = np.arange(-2, 3)
OFF = np.array([(a, b) for a in _r for b in _r if a * a + b * b <= 4])  # radius-2 disk anchors (13 px)


def sample_pairs(gt, max_pairs=3000, cell=6, seed=0):
    """Anchors = one skeleton pixel per cell x cell grid cell; pairs = anchors with DMIN <= dist <= DMAX."""
    rng = np.random.default_rng(seed)
    ys, xs = np.nonzero(skeletonize(gt))
    keep = np.unique((ys // cell) * 10_000 + xs // cell, return_index=True)[1]
    a = np.stack([ys[keep], xs[keep]], 1)
    if len(a) < 2:
        return a[:0], a[:0]
    ij = cKDTree(a).query_pairs(DMAX, output_type="ndarray")      # O(n log n), memory ~ #pairs
    ij = ij[((a[ij[:, 0]] - a[ij[:, 1]]) ** 2).sum(1) >= DMIN ** 2]
    i, j = ij[:, 0], ij[:, 1]
    if len(i) > max_pairs:
        s = rng.choice(len(i), max_pairs, replace=False); i, j = i[s], j[s]
    return a[i], a[j]


def connectivity(maps, u, v, shape=None, M=None):
    """maps: (M,H,W) bool array, or a function (oy, ox) -> (M,TILE,TILE) bool stack (then pass shape=(H,W), M).
    u/v (P,2) -> (M,P) bool: are u and v connected inside their tile, for every map?"""
    tile = maps if callable(maps) else (lambda oy, ox: maps[:, oy:oy + TILE, ox:ox + TILE])
    if not callable(maps):
        M, (H, W) = maps.shape[0], maps.shape[1:]
    else:
        H, W = shape
    lo = np.minimum(u, v)
    ty = np.minimum(lo[:, 0] // STRIDE * STRIDE, H - TILE)   # tile origin; DMAX < STRIDE guarantees containment
    tx = np.minimum(lo[:, 1] // STRIDE * STRIDE, W - TILE)
    out = np.zeros((M, len(u)), bool)
    key = ty * 10_000 + tx
    for k in np.unique(key):
        idx = np.nonzero(key == k)[0]
        oy, ox = ty[idx[0]], tx[idx[0]]
        L = label(tile(oy, ox), S3)[0]
        def anchor(p):  # (P,13) labels around each point, clipped to the tile
            yy = np.clip(p[idx, None, 0] - oy + OFF[None, :, 0], 0, TILE - 1)
            xx = np.clip(p[idx, None, 1] - ox + OFF[None, :, 1], 0, TILE - 1)
            return L[:, yy, xx]                                    # (M,P,13)
        lu, lv = anchor(u), anchor(v)
        out[:, idx] = ((lu[..., :, None] == lv[..., None, :]) & (lu[..., :, None] > 0)).any((-1, -2))
    return out


def pair_scores(gt, fov, p_det, p_mc, max_pairs=3000, seed=0):
    """gt (H,W) bool, p_det (H,W), p_mc (T,H,W). Returns dict of per-pair arrays."""
    u, v = sample_pairs(gt, max_pairs, seed=seed)
    lv = LEVELS[:, None, None]
    L, T = len(LEVELS), len(p_mc)

    def tile(oy, ox):  # threshold lazily, one 64x64 tile at a time: memory O(T*L*TILE^2), not O(T*L*H*W)
        sl = (slice(oy, oy + TILE), slice(ox, ox + TILE))
        f = fov[sl]
        return np.concatenate([gt[sl][None], (p_det[sl][None] > lv) & f,
                               ((p_mc[(slice(None),) + sl][:, None] > lv[None]) & f).reshape(-1, TILE, TILE)])

    c = connectivity(tile, u, v, gt.shape, 1 + L * (T + 1))
    det_l = c[1:1 + L]                       # (L,P)
    mc_l = c[1 + L:].reshape(T, L, -1)       # (T,L,P)
    return {"u": u, "v": v, "y": c[0],
            "det": det_l[I05].astype(float), "bn": det_l.mean(0),
            "mc": mc_l[:, I05].mean(0), "mcbn": mc_l.mean((0, 1))}
