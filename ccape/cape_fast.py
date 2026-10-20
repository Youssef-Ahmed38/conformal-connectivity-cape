"""Parallel CAPE: identical loss to the official implementation, ~8x faster.

Key observation: the Dijkstra path search needs no gradient; only the final sum of cost^2 along each path does.
So torch-free workers (ccape.paths) return integer path coordinates (embarrassingly parallel over windows) and
the main process does one fused gather on the GPU tensor.
"""
import numpy as np
import torch

from .paths import PT, make_pool, pair_job, window_job


class FastCAPE:
    def __init__(self, window=128, workers=8):
        self.w = window
        self.pool = make_pool(workers)
        self.calls = 0

    def __call__(self, p, y):
        """p: (B,H,W) probabilities (grad), y: (B,H,W) {0,1}. Returns official CAPE value (mean over batch)."""
        cost = 1 - p
        c_np, y_np = cost.detach().float().cpu().numpy(), y.detach().cpu().numpy() > 0.5
        B, Hh, Ww = p.shape
        w = self.w
        tiles = [(b, i, j) for b in range(B) for i in range(0, Hh, w) for j in range(0, Ww, w)
                 if y_np[b, i:i + w, j:j + w].any()]
        self.calls += 1
        jobs = [(c_np[b, i:i + w, j:j + w], y_np[b, i:i + w, j:j + w], hash((self.calls, k)))
                for k, (b, i, j) in enumerate(tiles)]
        bi, yi, xi = [np.zeros(0, np.int64)], [np.zeros(0, np.int64)], [np.zeros(0, np.int64)]
        for (b, i, j), pc in zip(tiles, self.pool.map(window_job, jobs, chunksize=2)):
            bi.append(np.full(pc.shape[1], b)); yi.append(pc[0] + i); xi.append(pc[1] + j)
        idx = [torch.from_numpy(np.concatenate(a)).to(p.device) for a in (bi, yi, xi)]
        return (cost[idx[0], idx[1], idx[2]] ** 2).sum() / B

    def pair_loss(self, p, b, u, v, ans, k_weak=3):
        """Connectivity-click loss. b (P,), u/v (P,2) yx, ans (P,) in {0,1}.
        ans=1: the cheapest u->v path must be vessel        -> mean (1-p)^2 along it.
        ans=0: that path's weakest links must be background  -> mean of the k smallest p^2 along it."""
        z = p.sum() * 0
        if len(b) == 0:
            return z
        c_np = (1 - p).detach().float().cpu().numpy()
        _, Hh, Ww = p.shape
        o = np.clip((u + v) // 2 - PT // 2, 0, [Hh - PT, Ww - PT])
        jobs = [(c_np[bb, oy:oy + PT, ox:ox + PT], uu - (oy, ox), vv - (oy, ox))
                for bb, (oy, ox), uu, vv in zip(b, o, u, v)]
        pos, neg = [], []
        for bb, (oy, ox), a, pc in zip(b, o, ans, self.pool.map(pair_job, jobs, chunksize=8)):
            if pc.shape[1] == 0:
                continue
            path_p = p[bb, torch.from_numpy(pc[0] + oy).to(p.device), torch.from_numpy(pc[1] + ox).to(p.device)]
            if a == 1:
                pos.append(((1 - path_p) ** 2).mean())
            else:
                neg.append((torch.topk(path_p, min(k_weak, len(path_p)), largest=False).values ** 2).mean())
        return (torch.stack(pos).mean() if pos else z) + (torch.stack(neg).mean() if neg else z)
