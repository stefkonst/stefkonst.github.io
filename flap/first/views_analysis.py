"""Second set of videos, 5 Nov: numbers from the tracks. Run: python views_analysis.py <track folder> -> data/views.json"""
import glob, json, os, re, sys
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import least_squares
HERE = os.path.dirname(os.path.abspath(__file__))
T = sys.argv[1]
PX, FPS = 6.14, 200.0                  # ruler, all views (the ruler sits at its own depth)
out = {}
for f in sorted(glob.glob(os.path.join(T, "*.npz"))):
    nm = os.path.basename(f)[:-4]
    z = np.load(f)
    U = z["te_u"] / PX                  # mm, + = up in the image
    V = int(re.match(r"(\d+)V", nm).group(1))
    fq = float(re.search(r"_(\d+)hz", nm).group(1))
    t = np.arange(len(U)) / FPS
    view = next(v for v in ("front", "side", "diag") if v in nm)
    sgn = -1.0 if view != "front" else 1.0  # side and diagonal: report the downward motion as positive
    U = sgn * U
    k = t > (1.0 if t[-1] > 4 else .5)
    X = np.c_[np.ones(k.sum()), t[k] - t[k].mean()]
    for h in (1, 2, 3):
        X = np.c_[X, np.cos(2 * np.pi * h * fq * t[k]), np.sin(2 * np.pi * h * fq * t[k])]
    C = np.linalg.lstsq(X, U[k], rcond=None)[0]
    A1 = np.hypot(C[2], C[3])
    j = int(np.argmax(A1))
    u = U[:, j]
    us = gaussian_filter1d(u, 1.5)
    rest = u[t < .15].mean()
    r = dict(view=view, V=V, f=fq, tension="tension" in nm, wave="sine" if "sine" in nm else "sine squared", n=len(t), station=j, s_frac=j / (U.shape[1] - 1),
             noise_mm=float(u[t < .15].std()), mean_mm=float(C[0, j] - rest), A1_mm=float(A1[j]), A2_mm=float(np.hypot(C[4, j], C[5, j])), A3_mm=float(np.hypot(C[6, j], C[7, j])),
             drift_mm_s=float(C[1, j]), A1_along=(A1 / A1.max()).round(3).tolist())
    # cycle highs and lows, first peak
    per = int(round(FPS / fq))
    if fq <= 2:
        hi, lo = [], []
        st = np.argmax(us > rest + 5 * r["noise_mm"] + .01)
        for c0 in range(st, len(u) - per, per):
            w = us[c0:c0 + per]
            hi.append(float(w.max() - rest)); lo.append(float(w.min() - rest))
        r.update(onset_s=float(t[st]), highs=hi, lows=lo)
    # phase average and voltage exponent for sine squared drives at 1 Hz, with the peak time fitted
    if fq == 1 and t[-1] > 4:
        ph = ((t[k] * fq) % 1)
        bins = np.linspace(0, 1, 41)
        pa = np.array([u[k][(ph >= a) & (ph < b)].mean() for a, b in zip(bins[:-1], bins[1:])]) - rest
        x = (bins[:-1] + bins[1:]) / 2
        def res(q):
            return q[0] + q[1] * np.abs(np.cos(np.pi * (x - q[2]))) ** (2 * q[3]) - pa
        g = least_squares(res, [pa.min(), np.ptp(pa), x[np.argmax(pa)], 1.5], bounds=([-5, 0, -1, .2], [5, 10, 2, 8]))
        def res2(q):
            return q[0] + q[1] * np.abs(np.cos(np.pi * (x - q[2]))) ** 4 - pa
        g2 = least_squares(res2, [pa.min(), np.ptp(pa), x[np.argmax(pa)]])
        r.update(phase_x=x.tolist(), phase=pa.tolist(), p=float(g.x[3]), p_rms_um=float(np.std(g.fun) * 1e3), V2_rms_um=float(np.std(g2.fun) * 1e3), fit_low=float(g.x[0]), fit_amp=float(g.x[1]))
    out[nm] = r
    print("%-30s %-5s A1 %.3f A2 %.3f A3 %.3f mean %+.3f noise %.3f st %2d | %s" % (nm, view, r["A1_mm"], r["A2_mm"], r["A3_mm"], r["mean_mm"], r["noise_mm"], j,
          ("p %.2f rms %.1f (V2 %.1f) low %.3f amp %.3f | first high %.3f lows %s" % (r["p"], r["p_rms_um"], r["V2_rms_um"], r["fit_low"], r["fit_amp"], r["highs"][0], np.round(r["lows"][:6], 3))) if "p" in r else ""))
json.dump(out, open(os.path.join(HERE, "data", "views.json"), "w"), indent=1)
