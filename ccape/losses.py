"""Pixel loss (BCE+Dice), soft clDice (Shit et al., CVPR'21) and CAPE (Esmaeilzadeh et al., MICCAI'25)."""
import torch
import torch.nn.functional as F

from .cape_fast import FastCAPE


def bce_dice(logits, y, eps=1.0):
    p = torch.sigmoid(logits)
    inter = (p * y).sum((1, 2))
    dice = 1 - (2 * inter + eps) / (p.sum((1, 2)) + y.sum((1, 2)) + eps)
    return F.binary_cross_entropy_with_logits(logits, y) + dice.mean()


# ---- soft skeleton: morphological ops as min/max pooling, fully differentiable ----
def _erode(x):
    return torch.minimum(-F.max_pool2d(-x, (3, 1), 1, (1, 0)), -F.max_pool2d(-x, (1, 3), 1, (0, 1)))


def _dilate(x):
    return F.max_pool2d(x, 3, 1, 1)


def soft_skel(x, iters=10):
    x = x[:, None]
    skel = F.relu(x - _dilate(_erode(x)))
    for _ in range(iters):
        x = _erode(x)
        delta = F.relu(x - _dilate(_erode(x)))
        skel = skel + F.relu(delta - skel * delta)
    return skel[:, 0]


def soft_cldice(p, y, iters=10, eps=1.0):
    sp, sy = soft_skel(p, iters), soft_skel(y, iters)
    tprec = ((sp * y).sum() + eps) / (sp.sum() + eps)
    tsens = ((sy * p).sum() + eps) / (sy.sum() + eps)
    return 1 - 2 * tprec * tsens / (tprec + tsens)


class CAPELoss:
    """CAPE on probability maps (is_binary=True -> path cost = sum (1-p)^2 along Dijkstra path).

    Same value as the official loss (tests/check_cape.py), computed in parallel. Normalised by the number of
    GT skeleton pixels so that it is O(1) and the weight is interpretable.
    """

    def __init__(self, window=128):
        self.f = FastCAPE(window)

    def __call__(self, p, y):
        norm = max(float(soft_skel(y).gt(0.5).sum()), 1.0) / p.shape[0]
        return self.f(p.float(), y) / norm
