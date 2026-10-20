"""Torch-free path search (runs inside worker processes; must stay light: numpy / skimage / cv2 / networkx only)."""
import random
import sys
from pathlib import Path

import cv2
import networkx as nx
import numpy as np
import skimage.graph
from skimage.morphology import skeletonize

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "external" / "CAPE"))
from utils.graph_from_skeleton_2D import graph_from_skeleton  # noqa: E402  (official CAPE utility)

DT = 1.0  # distance_threshold for binary maps (official code forces it to 1 when is_binary=True)
PT = 64   # tile for click-pair path search (pairs are <= 30 px apart)


def _min_in_radius(a, c, r):
    x0, y0 = c
    y1, y2 = max(0, y0 - r), min(a.shape[0], y0 + r + 1)
    x1, x2 = max(0, x0 - r), min(a.shape[1], x0 + r + 1)
    k = np.unravel_index(np.argmin(a[y1:y2, x1:x2]), (y2 - y1, x2 - x1))
    return y1 + k[0], x1 + k[1]


def window_paths(cost, gt, seed, dil=10, shift=5):
    """Official CAPE path sampling for one window. cost = 1-p, gt = vessel mask. -> (2,K) path pixels."""
    rng = random.Random(seed)
    np.random.seed(seed % 2**32)  # graph_from_skeleton decimates nodes with np.random.shuffle
    g = graph_from_skeleton(skeletonize(gt.astype(np.uint8)), angle_range=(175, 185), verbose=False)
    out = []
    while g.number_of_edges():
        nodes = list(g.nodes())
        while True:  # random connected pair (same rule as official _random_connected_pair)
            n1 = rng.choice(nodes)
            comp = list(nx.node_connected_component(g, n1))
            if len(comp) > 1:
                break
        n2 = rng.choice([n for n in comp if n != n1])
        pn = nx.shortest_path(g, n1, n2)
        pos = [g.nodes[n]["pos"] for n in pn]
        mask = np.zeros(cost.shape, np.uint8)
        for p, q in zip(pos[:-1], pos[1:]):
            cv2.line(mask, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])), 1, dil)
        img = DT - mask * (DT - cost)
        img = np.where(img == DT, np.inf, img)
        s = _min_in_radius(img, (int(pos[0][0]), int(pos[0][1])), shift)
        e = _min_in_radius(img, (int(pos[-1][0]), int(pos[-1][1])), shift)
        img = np.maximum(img, 0) + 1e-5
        try:
            pc, _ = skimage.graph.route_through_array(img, start=s, end=e, fully_connected=True, geometric=True)
            out.append(np.asarray(pc, np.int64).T)
        except Exception:
            pass
        g.remove_edges_from(zip(pn[:-1], pn[1:]))
    return np.concatenate(out, 1) if out else np.zeros((2, 0), np.int64)


def window_job(args):
    return window_paths(*args)


def pair_job(args):
    """Corridor-free Dijkstra between two clicks on cost = 1-p (only the yes/no answer is known)."""
    cost, s, e = args
    try:
        pc, _ = skimage.graph.route_through_array(np.maximum(cost, 0) + 1e-5, tuple(s), tuple(e),
                                                  fully_connected=True, geometric=True)
        return np.asarray(pc, np.int64).T
    except Exception:
        return np.zeros((2, 0), np.int64)


def make_pool(n):
    """Spawn n workers eagerly while hiding __main__.__file__, so workers do not re-import the training script
    (and therefore never load torch / CUDA DLLs)."""
    import __main__
    import multiprocessing as mp
    import os
    f = getattr(__main__, "__file__", None)
    if f is not None:
        del __main__.__file__
    env = {k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")}
    os.environ.update({k: "1" for k in env})  # single-threaded workers: no per-thread BLAS buffers
    try:
        return mp.get_context("spawn").Pool(n)
    finally:
        if f is not None:
            __main__.__file__ = f
        for k, v in env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
