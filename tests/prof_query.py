import sys, time, cProfile, pstats
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np, torch
from ccape.data import load
from ccape.model import UNet
from ccape.infer import infer
from ccape.sparse import query_pairs
from ccape.topo_uq import sample_pairs
if __name__ == "__main__":
    tr = load("calib"); m = UNet().cuda()
    t = time.time(); p_det, p_mc = infer(m, tr["img"][0], 10); print("infer", time.time() - t)
    gt, fov = tr["lab"][0] > 0, tr["fov"][0] > 0
    t = time.time(); u, v = sample_pairs((p_det > 0.3) & fov, 6000); print("sample", time.time() - t, len(u), ((p_det > 0.3) & fov).mean())
    cProfile.run("query_pairs(p_det, p_mc, gt, fov, 10, 'unc', np.random.default_rng(0))", "q.prof")
    pstats.Stats("q.prof").sort_stats("cumtime").print_stats(8)
