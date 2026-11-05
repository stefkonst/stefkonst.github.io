"""The drive as set: sine squared, no offset, so V = Vp sin^2(pi f t) and the wing sees V^2 = Vp^2 sin^4. What does the cycle shape say, and what does the first rise say? Run: python drive_check.py -> data/drive_check.json"""
import json, os
import numpy as np
from scipy.optimize import least_squares
HERE = os.path.dirname(os.path.abspath(__file__))
T = json.load(open(os.path.join(HERE, "data", "tracks.json")))
x, y = np.array(T["phase_avg_x"]), np.array(T["phase_avg"])
print({k: (type(v).__name__, len(v) if hasattr(v, "__len__") else v) for k, v in T.items()})


def harm(x, y, n=4):
    """Mean, then amplitude and phase of each harmonic of one cycle."""
    X = np.c_[[np.ones_like(x)] + [f(2 * np.pi * k * x) for k in range(1, n + 1) for f in (np.cos, np.sin)]].T
    c = np.linalg.lstsq(X, y, rcond=None)[0]
    z = c[1::2] - 1j * c[2::2]
    return c[0], z


m, z = harm(x, y)
A = np.abs(z)
rel = np.angle(z[1] * np.conj(z[0]) ** 2)                   # 0 for a shape symmetric about its peak
print("mean %.4f amps um %s" % (m, (A * 1e3).round(2)))
print("H2/H1 %.3f  H3/H1 %.3f  relative phase H2-2H1 %.1f deg" % (A[1] / A[0], A[2] / A[0], np.degrees(rel)))
print("expected H2/H1: V=sin^2 -> V^2=sin^4 0.25 | bipolar sine at f/2 -> 0")
out = dict(mean=m, amps_um=(A * 1e3).tolist(), H2_H1=A[1] / A[0], H3_H1=A[2] / A[0], rel_deg=float(np.degrees(rel)))


def shapes(ph, kind, tau=0.0):
    """V^2 over one cycle, phase ph in cycles, peak at ph = 0; optional first-order lag tau in s (f = 1 Hz)."""
    a = {0: 3 / 8, 1: 1 / 2, 2: 1 / 8} if kind == "sin4" else {0: 1 / 2, 1: 1 / 2}
    u = a[0] * np.ones_like(ph)
    for k, c in a.items():
        if k:
            g = 1 / (1 + 1j * 2 * np.pi * k * tau)
            u = u + c * np.real(g * np.exp(2j * np.pi * k * ph))
    return u


fits = {}
for nm, kind, lag in (("sin^4, no lag", "sin4", False), ("sin^2, no lag", "sin2", False), ("sin^4, first-order lag", "sin4", True)):
    def res(p):
        return p[0] + p[1] * shapes(x + p[2], kind, abs(p[3]) if lag else 0.0) - y
    g = least_squares(res, [0.3, 0.2, 0.0, 0.1 if lag else 0.0])
    fits[nm] = dict(u0=g.x[0], a=g.x[1], shift=g.x[2], tau=abs(g.x[3]) if lag else 0.0, rms_um=float(np.std(g.fun) * 1e3), u=(y + g.fun).tolist())
    print(nm, "u0 %.3f a %.3f shift %.3f tau %.3f rms %.2f um" % (g.x[0], g.x[1], g.x[2], fits[nm]["tau"], fits[nm]["rms_um"]))
out["fits"] = fits
out["x"], out["y"] = x.tolist(), y.tolist()

# the cycle low is at 0 V if the drive is as set, so the low against the rest position is what stays
t, u = np.array(T["mid_t"]), np.array(T["mid"])
rest = u[t < .3].mean()
tp, up = np.array(T["t_max"]), np.array(T["mid_max"])
tm, um = np.array(T["t_min"]), np.array(T["mid_min"])
print("dt", t[1] - t[0], "rest", rest, "n", len(t))
print("peaks", tp.round(3), up.round(3))
print("lows", tm.round(3), um.round(3))
out.update(rest=rest, t_max=tp.tolist(), mid_max=up.tolist(), t_min=tm.tolist(), mid_min=um.tolist())

# free exponent: u = u0 + a V^p with V = sin^2, so u = u0 + a |sin|^(2p)
def res(q):
    return q[0] + q[1] * np.abs(np.cos(np.pi * (x + q[2]))) ** (2 * q[3]) - y
g = least_squares(res, [0.3, 0.2, 0.0, 1.5], bounds=([-1, 0, -.5, .2], [1, 2, .5, 6]))
J = g.jac
cov = np.linalg.inv(J.T @ J) * np.sum(g.fun ** 2) / (len(y) - 4)
print("free exponent: u0 %.3f a %.3f shift %.3f p %.2f +- %.2f rms %.2f um" % (*g.x, np.sqrt(cov[3, 3]), np.std(g.fun) * 1e3))
out["power"] = dict(u0=g.x[0], a=g.x[1], shift=g.x[2], p=g.x[3], p_se=float(np.sqrt(cov[3, 3])), rms_um=float(np.std(g.fun) * 1e3), u=(y + g.fun).tolist())

# each cycle on its own: is the exponent stable? first rise against the later rises, from the drive phase of the later cycles
per = np.median(np.diff(tp))
t0 = tp[1:].mean() - per * np.arange(1, len(tp)).mean() - per / 2     # time V would have started from 0, if the first rise is a full half cycle
print("period %.4f, first peak %.3f, implied drive start if the first rise is a half cycle %.3f, onset seen %.3f" % (per, tp[0], t0, T["onset_s"]))
for lim in (.02, .05, .10):                                            # when the first rise passes this fraction, against V^p
    th = rest + lim * (up[0] - rest)
    i = np.argmax((t > .2) & (u > th))
    print("first rise reaches %.0f %% at %.3f s" % (lim * 100, t[i]))
early = (t > t0) & (t < tp[0])
print("first rise, u - rest at V = 0.25, 0.5, 0.75, 1 of peak:", [round(float(np.interp(t0 + np.arcsin(np.sqrt(v)) / np.pi * per, t, u) - rest), 3) for v in (.25, .5, .75, 1)])
late = (t > t0 + 3 * per) & (t < t0 + 3.5 * per)
print("rise 4, same voltages:", [round(float(np.interp(t0 + 3 * per + np.arcsin(np.sqrt(v)) / np.pi * per, t, u) - rest), 3) for v in (0, .25, .5, .75, 1)])
print("fall 4, same voltages:", [round(float(np.interp(t0 + 4 * per - np.arcsin(np.sqrt(v)) / np.pi * per, t, u) - rest), 3) for v in (0, .25, .5, .75, 1)])
out.update(period=float(per), t0=float(t0))
json.dump(out, open(os.path.join(HERE, "data", "drive_check.json"), "w"), indent=1, default=float)
