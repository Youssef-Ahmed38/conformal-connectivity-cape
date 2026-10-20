"""Check FastCAPE == official CAPE (same seeds) and measure the speed-up."""
import random, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np, torch
from ccape.data import load, CropSampler
from ccape.model import UNet
from ccape.cape_fast import FastCAPE
from ccape.paths import window_paths
from loss import CAPE  # official (path added by ccape.cape_fast)

if __name__ == "__main__":
    torch.manual_seed(1)
    s = CropSampler(load("train"), 256, "cuda"); m = UNet().cuda()
    x, y = s(8)
    with torch.no_grad(): p = torch.sigmoid(m(x)).float()
    off = CAPE(window_size=128, is_binary=True, dilation_radius=10, shifting_radius=5)
    for k in range(4):
        pw, yw = p[k:k+1, :128, :128], y[k:k+1, :128, :128]
        random.seed(k); np.random.seed(k); a = float(off(pw, yw.cpu().numpy()))
        pc = window_paths((1 - pw[0]).cpu().numpy(), yw[0].cpu().numpy() > 0.5, k)
        b = float(((1 - pw[0])[pc[0], pc[1]] ** 2).sum())
        print(f"window {k}: official {a:.6f}  fast {b:.6f}  {'OK' if abs(a-b) < 1e-3 else 'MISMATCH'}")
    f = FastCAPE(workers=12); f(p, y)
    t = time.time()
    for _ in range(3): f(p, y)
    tf = (time.time() - t) / 3
    t = time.time(); off(p, y.cpu().numpy()); to = time.time() - t
    print(f"official {to:.3f}s  fast {tf:.3f}s  speed-up x{to/tf:.1f}")
