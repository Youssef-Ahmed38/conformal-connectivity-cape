"""Class-conditional (Mondrian) split conformal prediction for binary connectivity + calibration metrics.

Score s in [0,1] estimates P[connected]. Non-conformity of label c: a_c = 1-s if c=1 else s.
Per class c, q_c = the ceil((n_c+1)(1-alpha))-th smallest calibration a_c.
Prediction set C(x) = {c : a_c(x) <= q_c}. Guarantee (exchangeability): P[y in C(x) | y=c] >= 1-alpha for each c,
so breaks and false merges are each covered, not just the (majority) connected class.
|C| = 1 -> trusted automatically, |C| != 1 -> flagged for human review.
"""
import numpy as np


def _nc(s, c):
    return 1 - s if c == 1 else s


def calibrate(s, y, alpha):
    q = {}
    for c in (0, 1):
        a = np.sort(_nc(s[y == c], c))
        n = len(a)
        k = int(np.ceil((n + 1) * (1 - alpha)))
        q[c] = np.inf if k > n else a[k - 1]
    return q


def predict_sets(s, q):
    """(P,2) bool membership of labels 0/1."""
    return np.stack([_nc(s, 0) <= q[0], _nc(s, 1) <= q[1]], 1)


def evaluate_sets(sets, y, det):
    size = sets.sum(1)
    auto = size == 1
    pred = sets[:, 1]                       # label of singletons
    err_det = det.astype(bool) != y         # topology errors of the plain thresholded model
    covered = sets[np.arange(len(y)), y.astype(int)]
    return {
        "coverage": covered.mean(),
        "cov_connected": covered[y == 1].mean(),
        "cov_disconnected": covered[y == 0].mean() if (y == 0).any() else np.nan,
        "flagged": (~auto).mean(),                                       # review burden
        "auto_error": (pred[auto] != y[auto]).mean() if auto.any() else np.nan,
        "errors_caught": (~auto[err_det] | (pred[err_det] == y[err_det])).mean() if err_det.any() else np.nan,
        "det_error_rate": err_det.mean(),
    }


class Isotonic:
    """Monotone recalibration score -> P(connected) by pool-adjacent-violators, O(n log n) (sort) + O(n) (PAV).
    Monotone, so conformal sets are unchanged; it only makes the score an honest probability (ECE)."""

    def fit(self, s, y):
        o = np.argsort(s, kind="mergesort")
        xs, ys = s[o], y[o].astype(float)
        val, wt, hi = [], [], []          # stack of blocks: mean, weight, right-end score
        for x, t in zip(xs, ys):
            val.append(t); wt.append(1.0); hi.append(x)
            while len(val) > 1 and val[-2] >= val[-1]:
                w = wt[-2] + wt[-1]
                v = (val[-2] * wt[-2] + val[-1] * wt[-1]) / w
                val[-2:], wt[-2:], hi[-2:] = [v], [w], [hi[-1]]
        self.hi, self.val = np.array(hi), np.array(val)
        return self

    def __call__(self, s):
        return self.val[np.minimum(np.searchsorted(self.hi, s, side="left"), len(self.val) - 1)]


def error_score(p_conn, det):
    """P(the thresholded decision is wrong) = P(connected) if it says 'disconnected', else 1 - P(connected)."""
    return np.where(det > 0.5, 1 - p_conn, p_conn)


def ece(s, y, bins=10):
    b = np.minimum((s * bins).astype(int), bins - 1)
    return sum(abs(s[b == k].mean() - y[b == k].mean()) * (b == k).mean() for k in range(bins) if (b == k).any())


def auroc(score, label):
    """Mann-Whitney U with tie correction, O(n log n)."""
    label = label.astype(bool)
    n1, n0 = float(label.sum()), float((~label).sum())   # float: int32 products overflow on Windows numpy
    if n1 == 0 or n0 == 0:
        return np.nan
    order = np.argsort(score, kind="mergesort")
    s = score[order]
    ranks = np.empty(len(s))
    i = 0
    while i < len(s):                         # average ranks over ties
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    return (ranks[label].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)
