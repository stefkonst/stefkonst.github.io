"""What sets the frequency rise in the free decay after the 1.4 kV run (8.0 -> 9.2 Hz over 0.3 s)?
A: amplitude softening:  x'' + 2 zeta w0 x' + w0^2 (1 + gamma x^2) x = 0
B: voltage softening that recovers after the drive stops (the charged elastomer is softer; the electrostatic stress
   decays with the drive's lag after the stop): x'' + 2 zeta w0 x' + w0^2 (1 - kappa s(t)) x = 0, s(t) = s0 exp(-t/ts)
Both fitted in the time domain to the measured decay (side view, from 2.25 s, 0.7 s long), same number of free
parameters for the frequency change (gamma; kappa with ts free or fixed at the 52 ms force lag).
Run: python decay_models.py -> ../data/decay_models.json"""
import os, json, numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares
HERE = os.path.dirname(os.path.abspath(__file__)); DATA = os.path.join(HERE, "..", "data"); FS = 200.0; D2R = np.pi / 180
a = np.load(os.path.join(DATA, "tracks", "full_DEA_core_wing_8hz_1400V_side.npz"))["ang"] * D2R
t = np.arange(len(a)) / FS; m = (t >= 2.25) & (t < 2.95) & np.isfinite(a); tt = t[m] - 2.25; y = a[m] - np.nanmedian(a[-150:])


def simA(p):
    w0, z, g, x0, v0, off = p
    f = lambda t_, s: [s[1], -2 * z * w0 * s[1] - w0 * w0 * (1 + g * s[0] ** 2) * s[0]]
    return solve_ivp(f, (0, tt[-1]), [x0, v0], t_eval=tt, rtol=1e-7, atol=1e-10).y[0] + off


def simB(p, ts=None):
    if ts is None: w0, z, k, ts_, x0, v0, off = p
    else: (w0, z, k, x0, v0, off), ts_ = p, ts
    f = lambda t_, s: [s[1], -2 * z * w0 * s[1] - w0 * w0 * (1 - k * np.exp(-t_ / ts_)) * s[0]]
    return solve_ivp(f, (0, tt[-1]), [x0, v0], t_eval=tt, rtol=1e-7, atol=1e-10).y[0] + off


x0g, v0g = y[0], (y[1] - y[0]) * FS
best = {}
for name, fun, p0s, lb, ub in (
        ("amplitude softening", simA, [[2 * np.pi * 9.2, 0.07, g, x0g, v0g, 0] for g in (-5, -50, -500, -3000)], [30, 0.0, -2e4, -1, -50, -0.2], [80, 0.5, 2e4, 1, 50, 0.2]),
        ("voltage softening, ts free", simB, [[2 * np.pi * 9.2, 0.07, k, ts, x0g, v0g, 0] for k in (0.2, 0.4) for ts in (0.05, 0.15, 0.3)], [30, 0.0, 0, 0.005, -1, -50, -0.2], [80, 0.5, 0.95, 2.0, 1, 50, 0.2]),
        ("voltage softening, ts = 52 ms", lambda p: simB(p, 0.052), [[2 * np.pi * 9.2, 0.07, k, x0g, v0g, 0] for k in (0.2, 0.4, 0.7)], [30, 0.0, 0, -1, -50, -0.2], [80, 0.5, 0.95, 1, 50, 0.2]),
        ("linear (no change)", lambda p: simA(list(p[:2]) + [0.0] + list(p[2:])), [[2 * np.pi * 8.6, 0.07, x0g, v0g, 0]], [30, 0.0, -1, -50, -0.2], [80, 0.5, 1, 50, 0.2])):
    bb = None
    for p0 in p0s:
        try:
            r = least_squares(lambda p: (fun(p) - y) / D2R, p0, bounds=(lb, ub), max_nfev=400)
        except Exception as e:
            continue
        if bb is None or r.cost < bb.cost: bb = r
    rms = float(np.sqrt(2 * bb.cost / len(y))); best[name] = dict(params=bb.x.tolist(), rms_deg=rms, n=len(y), k=len(bb.x))
    print("%-32s rms %.3f deg  params %s" % (name, rms, np.round(bb.x, 3)), flush=True)
json.dump(best, open(os.path.join(DATA, "decay_models.json"), "w"), indent=1)
