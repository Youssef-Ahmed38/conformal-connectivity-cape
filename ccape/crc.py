"""Image-level Conformal Risk Control and covariate-shift-weighted conformal prediction for connectivity.

(1) Conformal Risk Control (Angelopoulos, Bates, Fisch, Lei, Schuster; ICLR 2024).
    Unit = image i with pair scores s (estimated P[connected]) and labels y.
    Trust "connected"    iff s >= l1 ;   trust "disconnected" iff s <= 1 - l0 ; otherwise flag for review.
    Per-image losses (bounded by 1, non-increasing in l):
        L1_i(l1) = #{s >= l1, y = 0} / M_i          (wrongly trusted connections  = accepted false merges)
        L0_i(l0) = #{s <= 1-l0, y = 1} / M_i        (wrongly trusted disconnections = accepted breaks)
    Losses are capped at B (bounded by design, default 0.5): the 1/(n+1) worst-case term then costs B/(n+1)
    instead of 1/(n+1), which is what makes CRC usable with ~20 calibration images.
    l_hat = min{ l : n/(n+1) * mean_i min(L_i(l), B) + B/(n+1) <= a }, run with a = alpha/2 for each side.
    Theorem: if the n calibration images and the test image are exchangeable,
        E[ min(L1_test(l1_hat), B) + min(L0_test(l0_hat), B) ] <= alpha,
    i.e. in expectation at most alpha of a new image's connections are wrongly auto-accepted. No assumption on
    the dependence of pairs inside an image is needed (that was the gap of pair-level split conformal).

(2) Weighted split conformal under covariate shift (Tibshirani, Barber, Candes, Ramdas; NeurIPS 2019).
    w(x) = dP_test/dP_cal(x) estimated by a probabilistic classifier separating calibration from *unlabelled*
    test pairs: w = p/(1-p) * n_cal/n_test. Per class c (Mondrian), the test point's set contains c iff
        sum_{i: y_i=c, a_i <= a_c(x)} w_i  <  (1-alpha) * (W_c + w(x)) ,
    evaluated for all test points at once with one cumsum + searchsorted: O((n+m) log n).
"""
import numpy as np

GRID = np.linspace(0.0, 1.0, 1001)


def _counts_ge(s, mask, grid):
    """For each g in grid: #{s >= g and mask}. O((n + |grid|) log n)."""
    v = np.sort(s[mask])
    return len(v) - np.searchsorted(v, grid, side="left")


def image_losses(images, grid=GRID, B=0.5):
    """images: list of dicts with 's' (P[connected]) and 'y'. -> L1, L0 of shape (n_images, |grid|)."""
    L1 = np.stack([_counts_ge(d["s"], ~d["y"], grid) / max(len(d["s"]), 1) for d in images])
    # s <= 1-l  <=>  (1-s) >= l
    L0 = np.stack([_counts_ge(1 - d["s"], d["y"], grid) / max(len(d["s"]), 1) for d in images])
    return np.minimum(L1, B), np.minimum(L0, B)


def crc_lambda(L, a, grid=GRID, B=0.5):
    n = L.shape[0]
    ok = n / (n + 1) * L.mean(0) + B / (n + 1) <= a
    return grid[np.argmax(ok)] if ok.any() else np.inf     # inf = trust nothing on this side


def crc_fit(images, alpha, B=0.5):
    L1, L0 = image_losses(images, B=B)
    return crc_lambda(L1, alpha / 2, B=B), crc_lambda(L0, alpha / 2, B=B)


def crc_eval(images, l1, l0, B=0.5):
    """Per-image realised risk + efficiency on test images."""
    risk, trusted, caught, capped = [], [], [], []
    for d in images:
        s, y = d["s"], d["y"]
        t1, t0 = s >= l1, s <= 1 - l0
        r1, r0 = (t1 & ~y).mean(), (t0 & y).mean()
        risk.append(r1 + r0); capped.append(max(r1, r0) > B)
        trusted.append((t1 | t0).mean())
        err = (d["det"] > 0.5) != y                        # topology errors of the thresholded prediction
        caught.append((~(t1 | t0))[err].mean() if err.any() else np.nan)
    return {"risk": float(np.mean(risk)), "risk_max_img": float(np.max(risk)), "trusted": float(np.mean(trusted)),
            "flagged": 1 - float(np.mean(trusted)), "errors_flagged": float(np.nanmean(caught)),
            "cap_binding": float(np.mean(capped))}


# ------------------------------------------------------------------ weighted conformal (covariate shift)
def pair_features(d, img_id):
    """Per-pair covariates: the four scores + image-level context (mean score, fraction of uncertain pairs)."""
    S = np.stack([d["mcbn"], d["bn"], d["mc"], d["det"]], 1)
    ctx = np.broadcast_to([d["mcbn"].mean(), (np.abs(2 * d["mcbn"] - 1) < 0.5).mean()], (len(S), 2))
    return np.concatenate([S, ctx], 1)


def density_ratio(Xc, Xt, C=1.0, max_iter=500):
    """w(x) for calibration and test points from a logistic-regression domain classifier (test labels unused)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import PolynomialFeatures, StandardScaler
    X = np.concatenate([Xc, Xt]); z = np.r_[np.zeros(len(Xc)), np.ones(len(Xt))]
    f = PolynomialFeatures(2, include_bias=False).fit(X); sc = StandardScaler().fit(f.transform(X))
    T = lambda A: sc.transform(f.transform(A))
    clf = LogisticRegression(C=C, max_iter=max_iter).fit(T(X), z)
    odds = lambda A: (lambda p: p / np.clip(1 - p, 1e-6, None))(np.clip(clf.predict_proba(T(A))[:, 1], 1e-6, 1 - 1e-6))
    k = len(Xc) / len(Xt)
    return np.clip(odds(Xc) * k, 1e-3, 1e3), np.clip(odds(Xt) * k, 1e-3, 1e3)


def weighted_sets(s_cal, y_cal, w_cal, s_tst, w_tst, alpha):
    """Mondrian weighted split conformal. Non-conformity of label c: 1-s (c=1), s (c=0). -> (m,2) bool sets."""
    sets = np.zeros((len(s_tst), 2), bool)
    for c in (0, 1):
        a = (1 - s_cal if c else s_cal)[y_cal == c]
        w = w_cal[y_cal == c]
        o = np.argsort(a); a, cw = a[o], np.cumsum(w[o])
        a_t = 1 - s_tst if c else s_tst
        # weight of calibration scores strictly below the test score (ties counted as conforming: conservative)
        below = np.where(np.searchsorted(a, a_t, side="left") > 0, cw[np.maximum(np.searchsorted(a, a_t, side="left") - 1, 0)], 0.0)
        sets[:, c] = below < (1 - alpha) * (cw[-1] + w_tst)
    return sets
