"""Click prompts as input channels for a non-promptable network (fair baseline for TopoSAM).
Positive / negative clicks -> two Gaussian heat-map channels (peak 1, sigma px), via one separable conv on GPU."""
import torch
import torch.nn.functional as F

SIGMA = 4.0


def _kernel(sigma, dev):
    r = int(3 * sigma)
    x = torch.arange(-r, r + 1, device=dev, dtype=torch.float32)
    k = torch.exp(-x ** 2 / (2 * sigma ** 2))
    return k / k.max(), r


def heatmaps(pts, lab, H, W, sigma=SIGMA):
    """pts (B,N,2) xy float, lab (B,N) in {1,0,-10} -> (B,2,H,W). Label -10 = padding (ignored)."""
    B, dev = pts.shape[0], pts.device
    m = torch.zeros(B, 2, H, W, device=dev)
    x = pts[..., 0].round().long().clamp(0, W - 1); y = pts[..., 1].round().long().clamp(0, H - 1)
    for c, l in ((0, 1), (1, 0)):
        b, j = torch.nonzero(lab == l, as_tuple=True)
        m[b, c, y[b, j], x[b, j]] = 1.0
    k, r = _kernel(sigma, dev)
    m = m.view(B * 2, 1, H, W)
    m = F.conv2d(F.pad(m, (r, r, 0, 0)), k.view(1, 1, 1, -1))
    m = F.conv2d(F.pad(m, (0, 0, r, r)), k.view(1, 1, -1, 1))
    return m.view(B, 2, H, W).clamp_(max=1.0)


def sample_clicks(pos, neg, k, gen=None):
    """pos/neg (B,H,W) bool GPU masks -> pts (B,2k,2) xy, lab (B,2k); empty class -> padding label -10."""
    B, dev = pos.shape[0], pos.device
    P, L = torch.zeros(B, 2 * k, 2, device=dev), torch.full((B, 2 * k), -10, device=dev, dtype=torch.long)
    for b in range(B):
        for c, (msk, l) in enumerate(((pos[b], 1), (neg[b], 0))):
            yx = torch.nonzero(msk)
            if len(yx):
                i = torch.randint(len(yx), (k,), device=dev, generator=gen)
                P[b, c * k:(c + 1) * k] = yx[i].flip(1).float(); L[b, c * k:(c + 1) * k] = l
    return P, L
