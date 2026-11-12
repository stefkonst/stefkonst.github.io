"""Stroke from the wing tip, which cannot switch between features (the top-silhouette tracker in track.py follows
the rolled edge instead of the leading edge whenever the wing pitches over in the image).
Per frame: dark pixels against the per-pixel background, in a band around the wing; the tip is the centroid of the
dark pixels within 8 px of the farthest one along the wing. The hinge is the centre of the circle that best fits the
tip's path (least squares), so the stroke angle is measured about the real pivot as seen in the image.
Run: python tip_track.py -> ../data/tip_<run>.npz, ../data/tip_harmonics.json"""
import os, sys, json
import numpy as np
from scipy.ndimage import gaussian_filter
HERE = os.path.dirname(os.path.abspath(__file__)); DATA = os.path.join(HERE, "..", "data")
sys.path.insert(0, os.path.join(HERE, "..", "..", "first"))
from track_views import Stack
VID = r"C:/Users/stefk/MIT Dropbox/SMRL/Shared/Stefania/full_tension/"
FS = 200.0
RUNS = {"full_DEA_core_wing_8hz_1200V_side": (8.0, (120, 520, 420, 760)), "only_DEA_core_8hz_1200V_side": (8.0, (120, 520, 420, 780)),
        "full_DEA_core_wing_7hz_1200V_side": (7.0, (120, 520, 420, 760)), "full_DEA_core_wing_9hz_1200V_side": (9.0, (120, 520, 420, 760)),
        "full_DEA_core_wing_8hz_1400V_side": (8.0, (120, 520, 420, 760)), "full_DEA_core_wing_5hz_1200v_side": (5.0, (120, 520, 420, 760)),
        "full_DEA_core_wing_10hz_1200V_side": (10.0, (120, 520, 420, 760)), "full_DEA_core_wing_1hz_1200v_side": (1.0, (120, 520, 420, 760)),
        "full_DEA_core_wing_15hz_1200v_side": (15.0, (120, 520, 420, 760))}


def track(n, roi, thr=30):
    s = Stack(VID + n + ".avi"); N = len(s); y0, y1, x0, x1 = roi
    bg = np.percentile(np.stack([np.asarray(s[i], np.float32)[y0:y1, x0:x1] for i in range(0, N, max(1, N // 200))]), 97, axis=0)
    tip = np.full((N, 2), np.nan)
    for k in range(N):
        d = gaussian_filter(bg - np.asarray(s[k], np.float32)[y0:y1, x0:x1], 1.0) > thr
        ys, xs = np.nonzero(d)
        if len(xs) < 30: continue
        xm = xs.max(); m = xs > xm - 8
        tip[k] = (xs[m].mean() + x0, ys[m].mean() + y0)
    return tip


def circle(p):
    x, y = p[:, 0], p[:, 1]; A = np.c_[2 * x, 2 * y, np.ones(len(x))]; b = x * x + y * y
    cx, cy, c = np.linalg.lstsq(A, b, rcond=None)[0]; return cx, cy, np.sqrt(c + cx * cx + cy * cy)


def harmonics(a, f):
    t = np.arange(len(a)) / FS; rest = np.nanmedian(a[:40]); on = int(np.argmax(np.abs(a - rest) > np.radians(0.5)))
    g = np.zeros(len(a), bool); g[on + int(0.5 * FS):] = True
    if "1400" in RUN: g[int(2.15 * FS):] = False
    g &= np.isfinite(a)
    X = np.column_stack([np.ones(g.sum())] + [fn(2 * np.pi * h * f * t[g]) for h in (1, 2, 3) for fn in (np.cos, np.sin)])
    c = np.linalg.lstsq(X, a[g], rcond=None)[0]
    Xc = [c[1] - 1j * c[2], c[3] - 1j * c[4], c[5] - 1j * c[6]]
    N = int(round(FS / f)); idx = np.where(g)[0]
    pp = [np.nanmax(a[i:i + N]) - np.nanmin(a[i:i + N]) for i in range(idx[0], idx[-1] - N, N)]
    return Xc, float(np.median(pp)), float(np.std(pp)), float(np.std(a[g] - X @ c))


out = {}
for RUN, (f, roi) in RUNS.items():
    tip = track(RUN, roi); ok = np.isfinite(tip[:, 0])
    cx, cy, r = circle(tip[ok])
    ang = np.arctan2(tip[:, 1] - cy, tip[:, 0] - cx)
    np.savez_compressed(os.path.join(DATA, "tracks", "tip_" + RUN + ".npz"), tip=tip, ang=ang, hinge=[cx, cy], radius=r)
    Xc, pp, pps, res = harmonics(ang, f)
    out[RUN] = dict(f=f, hinge=[float(cx), float(cy)], radius_px=float(r), pp_deg=np.degrees(pp), pp_sd_deg=np.degrees(pps),
                    A_deg=[float(np.degrees(abs(z))) for z in Xc], X=[[z.real, z.imag] for z in Xc], resid_deg=float(np.degrees(res)))
    print("%-38s hinge (%.0f, %.0f) r %.0f px | pk-pk %.2f ± %.2f deg | A1 %.2f A2 %.2f A3 %.2f | A2/A1 %.3f" % (
        RUN, cx, cy, r, np.degrees(pp), np.degrees(pps), *out[RUN]["A_deg"], out[RUN]["A_deg"][1] / out[RUN]["A_deg"][0]), flush=True)
json.dump(out, open(os.path.join(DATA, "tip_harmonics.json"), "w"), indent=1)
