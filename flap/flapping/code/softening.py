"""A softening stroke spring: does one cubic term explain the broad resonance, the free decays and the 1.4 kV
saturation together?
    x'' + 2 zeta w x' + w^2 (1 + gamma x^2) x + beta x^2 = G u,   tau u' + u = sin^4(pi f t),   tau = 52 ms (lag_type.py)
gamma < 0 softens. Fitted to: the stroke fundamental at 1, 5, 7, 8, 9, 10 Hz (1.2 kV, powered; silhouette = tip here)
and the frequency against amplitude of the free decay after the 1.4 kV run. Not fitted: the 1.4 kV stroke at 8 Hz,
which the model then predicts (drive x (1400/1200)^2).
Run: python softening.py -> ../data/softening.json"""
import os, json, numpy as np
from numba import njit
from scipy.optimize import least_squares
HERE = os.path.dirname(os.path.abspath(__file__)); DATA = os.path.join(HERE, "..", "data")
D2R = np.pi / 180; TAU = 0.052
SW = json.load(open(os.path.join(DATA, "sweepfit.json"))); DEC = json.load(open(os.path.join(DATA, "decays.json"))); HT = json.load(open(os.path.join(DATA, "hb_tip.json")))
R = {r["run"]: r for r in SW["runs"]}
sw = sorted([(R[n]["f"], R[n]["A"][0], max(R[n]["A1_sd"], 0.05 * R[n]["A"][0])) for n in R if n.startswith("full") and R[n]["V"] == 1200 and R[n]["f_drive"] != 15], key=lambda q: q[0])
fa = np.array([q for q in DEC[0]["f_vs_amp"] if 0.8 < q[0] < 4.0])
A14 = R["full_DEA_core_wing_8hz_1400V_side"]["A"][0]
BETA = HT["full_DEA_core_wing_8hz_1200V_side"]["beta"][0]


@njit(cache=True)
def sim(w, zeta, gamma, beta, G, tau, f, T, dt, stop):
    n = int(T / dt); x = 0.0; v = 0.0; u = 0.0; out = np.empty(n)
    for k in range(n):
        t = k * dt
        s = np.sin(np.pi * f * t) ** 4 if (stop < 0 or t < stop) else 0.0
        u += dt * (s - u) / tau
        a = G * u - 2 * zeta * w * v - w * w * (1 + gamma * x * x) * x - beta * x * x
        v += dt * a; x += dt * v
        if abs(x) > 5: x = 5.0
        out[k] = x
    return out


def A1(p, f, g=1.0, dt=1 / 4000):
    w, zeta, gamma, G = p
    T = 4.0 if f >= 3 else 8.0
    y = sim(w, zeta, gamma, BETA, G * g, TAU, f, T, dt, -1.0); t = np.arange(len(y)) * dt; m = t > T - (2.0 if f >= 3 else 4.0)
    return 2 * abs(np.mean(y[m] * np.exp(-2j * np.pi * f * t[m]))) / D2R


def decay(p, g=(1400 / 1200) ** 2, stop=2.2, dt=1 / 4000):
    w, zeta, gamma, G = p
    y = sim(w, zeta, gamma, BETA, G * g, TAU, 8.0, stop + 1.0, dt, stop) / D2R
    t = np.arange(len(y)) * dt; m = t > stop + 0.05; z = y[m] - y[m][-800:].mean(); tt = t[m]
    zc = np.where(np.diff(np.sign(z)) != 0)[0]; out = []
    for i in range(len(zc) - 2):
        a_, b_ = zc[i], zc[i + 2]; amp = 0.5 * (np.abs(z[a_:zc[i + 1]]).max() + np.abs(z[zc[i + 1]:b_]).max())
        if amp > 0.3: out.append((amp, 1 / (tt[b_] - tt[a_])))
    return np.array(out) if out else np.zeros((0, 2))


def res(p, with_decay=True):
    r = [(A1(p, f) - a) / s for f, a, s in sw]
    if with_decay:
        d = decay(p)
        if len(d) >= 2:
            o = np.argsort(d[:, 0]); fm = np.interp(fa[:, 0], d[o, 0], d[o, 1])
        else:
            fm = np.full(len(fa), 20.0)
        r += list((fm - fa[:, 1]) / 0.3)
    return np.array(r)


best = None
for g0 in (0.0, -2.0, -6.0, -12.0):
    for w0 in (2 * np.pi * 8.6, 2 * np.pi * 9.2):
        r = least_squares(res, [w0, 0.07, g0, 500.0], bounds=([2 * np.pi * 6, 0.01, -60, 10], [2 * np.pi * 12, 0.4, 20, 5000]), x_scale=[2, 0.02, 2, 50], max_nfev=200)
        print("start gamma %.0f w %.1f -> cost %.1f  %s" % (g0, w0, r.cost, np.round(r.x, 3)), flush=True)
        if best is None or r.cost < best.cost: best = r
p = best.x
lin = least_squares(lambda q: res([q[0], q[1], 0.0, q[2]]), [p[0], p[1], p[3]], bounds=([2 * np.pi * 6, 0.01, 10], [2 * np.pi * 12, 0.4, 5000]), x_scale=[2, 0.02, 50])
pl = [lin.x[0], lin.x[1], 0.0, lin.x[2]]
pred14, pred14_lin = A1(p, 8.0, (1400 / 1200) ** 2), A1(pl, 8.0, (1400 / 1200) ** 2)
ff = np.arange(1.0, 15.01, 0.25)
out = dict(params=dict(f_n=p[0] / 2 / np.pi, zeta=p[1], gamma=p[2], G=p[3], tau=TAU, beta=BETA), cost=float(best.cost), cost_linear=float(lin.cost),
           linear=dict(f_n=pl[0] / 2 / np.pi, zeta=pl[1], G=pl[3]),
           sweep=dict(f=ff.tolist(), soft=[A1(p, f) for f in ff], lin=[A1(pl, f) for f in ff], data=sw),
           decay=dict(soft=decay(p).tolist(), lin=decay(pl).tolist(), data=fa.tolist()),
           v1400=dict(measured=A14, soft=pred14, lin=pred14_lin, at1200=A1(p, 8.0)))
json.dump(out, open(os.path.join(DATA, "softening.json"), "w"), indent=1)
print("softening: f_n %.2f Hz, zeta %.3f, gamma %.2f /rad^2, cost %.1f | linear: f_n %.2f zeta %.3f cost %.1f" % (p[0] / 2 / np.pi, p[1], p[2], best.cost, pl[0] / 2 / np.pi, pl[1], lin.cost))
print("1.4 kV at 8 Hz: measured %.2f deg; softening predicts %.2f, linear %.2f (1.2 kV model %.2f)" % (A14, pred14, pred14_lin, A1(p, 8.0)))
