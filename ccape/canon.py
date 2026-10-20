"""Label-free canonicalisation of curvilinear images by scale-space ridge analysis (Lindeberg 1998; Frangi 1998).

For a line of width w the Hessian across it has one large eigenvalue: lambda_min << 0 for a BRIGHT line on a
dark background, lambda_max >> 0 for a DARK line. With gamma-normalised derivatives (gamma = 3/4 for ridges) the
response  R(sigma) = sigma^(2*gamma) * |lambda|  peaks at sigma* proportional to w (Lindeberg's scale selection).

estimate(rgb) -> (polarity, sigma*): polarity = 'dark' | 'bright' (whichever ridge type responds more strongly in
its top percentile), sigma* = argmax over a log-spaced sigma grid of that response.
canonicalise(rgb, ref) inverts the image if its polarity differs from the reference domain and resizes it by
sigma*_ref / sigma*, so thin structures look like the training structures. No labels, no training.
"""
import numpy as np
from PIL import Image
from skimage.feature import hessian_matrix, hessian_matrix_eigvals

SIGMAS = np.geomspace(1.0, 8.0, 10)
GAMMA = 0.75


def _gray(rgb):
    g = rgb.astype(np.float32) @ np.array([0.299, 0.587, 0.114], np.float32) / 255.0
    return g


def ridge_profile(rgb, mask=None, top=0.02, max_side=512):
    """-> (resp_dark[s], resp_bright[s]): mean of the top `top` fraction of gamma-normalised ridge responses."""
    g = _gray(rgb)
    f = 1.0
    if max(g.shape) > max_side:                     # analyse at <= 512 px for speed; sigma rescaled back
        f = max_side / max(g.shape)
        g = np.array(Image.fromarray(g).resize((round(g.shape[1] * f), round(g.shape[0] * f)), Image.BILINEAR))
        if mask is not None:
            mask = np.array(Image.fromarray(mask.astype(np.uint8)).resize(g.shape[::-1], Image.NEAREST)) > 0
    rd, rb = [], []
    for s in SIGMAS * f:
        H = hessian_matrix(g, sigma=max(s, 0.5), order="rc", use_gaussian_derivatives=False)
        l1, l2 = hessian_matrix_eigvals(H)          # l1 >= l2
        k = max(s, 0.5) ** (2 * GAMMA)
        d, b = k * np.maximum(l1, 0), k * np.maximum(-l2, 0)
        if mask is not None:
            d, b = d[mask], b[mask]
        n = max(int(top * d.size), 1)
        rd.append(np.partition(d.ravel(), -n)[-n:].mean()); rb.append(np.partition(b.ravel(), -n)[-n:].mean())
    return np.array(rd), np.array(rb)


def estimate(rgb, mask=None):
    rd, rb = ridge_profile(rgb, mask)
    pol = "dark" if rd.max() >= rb.max() else "bright"
    r = rd if pol == "dark" else rb
    i = int(np.argmax(r))
    if 0 < i < len(r) - 1:                          # parabolic refinement of the peak in log-sigma
        y0, y1, y2 = np.log(r[i - 1:i + 2] + 1e-12)
        den = y0 - 2 * y1 + y2
        off = 0.5 * (y0 - y2) / den if den < 0 else 0.0
        ls = np.log(SIGMAS)
        sig = float(np.exp(ls[i] + off * (ls[1] - ls[0])))
    else:
        sig = float(SIGMAS[i])
    return pol, sig


def reference(rgbs, masks=None):
    """Reference (polarity, sigma*) of a training domain: majority polarity, median sigma*."""
    est = [estimate(r, None if masks is None else m) for r, m in zip(rgbs, masks if masks is not None else [None] * len(rgbs))]
    pols = [p for p, _ in est]
    return max(set(pols), key=pols.count), float(np.median([s for _, s in est]))


def canonicalise(rgb, ref, mask=None):
    """-> (rgb', scale, inverted). rgb' has the reference polarity and line width."""
    pol, sig = estimate(rgb, mask)
    inv = pol != ref[0]
    out = 255 - rgb if inv else rgb
    scale = float(np.clip(ref[1] / sig, 0.25, 4.0))
    if abs(scale - 1) > 0.05:
        h, w = out.shape[:2]
        out = np.array(Image.fromarray(out).resize((round(w * scale), round(h * scale)), Image.BILINEAR))
    return out, scale, inv
