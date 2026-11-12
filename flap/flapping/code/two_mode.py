"""Stroke + wing mode. The powered robot is one mode (sweep fit + the quadratic term beta_p from harmonic balance,
which reproduces its 2nd harmonic). With the wing at 0 V, a wing mode q (leading-edge bending with the tense membrane)
is added, driven by the stroke and pulling back on it:
    x'' + 2 zeta w x' + w^2 x + beta_p x^2 = G u(t) - mu * w_w^2 * q
    q'' + 2 zeta_w w_w q' + w_w^2 q = -x''
The tracked angle is x + q (the leading-edge line sees the bend). Three unknowns (f_w, zeta_w, mu) against five
measured numbers of the unpowered run, all independent of the clock: |X1| relative to the powered run, and the complex
X2/X1^2 and X3/X1^3. A real wing mode must fit all five; powering the wing removes it (the edge slack, the mode moves
down and is damped, see Fig. 3c), which leaves the powered model.
Run: python two_mode.py -> ../data/two_mode.json"""
import os, json, numpy as np
from numba import njit
from scipy.optimize import least_squares
HERE = os.path.dirname(os.path.abspath(__file__)); DATA = os.path.join(HERE, "..", "data"); TR = os.path.join(DATA, "tracks")
D2R = np.pi / 180; FS = 200.0
SW = json.load(open(os.path.join(DATA, "sweepfit.json"))); P = SW["fits"]["zeta 0.07 (decay)"]
HB = json.load(open(os.path.join(DATA, "harmonic_balance.json"))); EO = json.load(open(os.path.join(DATA, "edge_on_stroke.json")))
R = {r["run"]: r for r in SW["runs"]}
PW, UN = "full_DEA_core_wing_8hz_1200V_side", "only_DEA_core_8hz_1200V_side"
w = 2 * np.pi * P["fn"]; zeta = P["zeta"]; tau = P["tau"]; beta_p = EO["beta_p"]; G = EO["G"]


@njit(cache=True)
def sim(w, zeta, beta, G, tau, ww, zw, mu, f, T, dt):
    n = int(T / dt); x = 0.0; v = 0.0; q = 0.0; qd = 0.0; u = 0.0; out = np.empty(n)
    for k in range(n):
        t = k * dt
        s = np.sin(np.pi * f * t) ** 4
        u += dt * (s - u) / tau
        a = G * u - 2 * zeta * w * v - w * w * x - beta * x * x - mu * ww * ww * q
        qa = -a - 2 * zw * ww * qd - ww * ww * q
        v += dt * a; x += dt * v; qd += dt * qa; q += dt * qd
        out[k] = x + q
    return out


def harm(ww, zw, mu, f=8.0):
    dt = 1 / 8000; y = sim(w, zeta, beta_p, G, tau, ww, zw, mu, f, 4.0, dt); t = np.arange(len(y)) * dt; m = t > 2.5
    W = 2 * np.pi * f
    return [2 * np.mean(y[m] * np.exp(-1j * h * W * t[m])) for h in (1, 2, 3)]


def measured(n, f):
    a = np.load(os.path.join(TR, n + ".npz"))["ang"] * D2R; t = np.arange(len(a)) / FS
    rest = np.nanmedian(a[:40]); on = int(np.argmax(np.abs(a - rest) > 0.5 * D2R))
    g = np.zeros(len(a), bool); g[on + int(0.5 * FS):] = True; g &= np.isfinite(a); W = 2 * np.pi * f
    X = np.column_stack([np.ones(g.sum())] + [fn(h * W * t[g]) for h in (1, 2, 3) for fn in (np.cos, np.sin)])
    c = np.linalg.lstsq(X, a[g], rcond=None)[0]
    return [c[1] - 1j * c[2], c[3] - 1j * c[4], c[5] - 1j * c[6]]


Xp = measured(PW, R[PW]["f"]); Xu = measured(UN, R[UN]["f"])
r2u, r3u = Xu[1] / Xu[0] ** 2, Xu[2] / Xu[0] ** 3
ratio_u = abs(Xu[0]) / abs(Xp[0])
Mp = harm(1e3, 0.5, 0.0)
print("powered model: A1 %.2f A2 %.2f A3 %.2f deg | data %.2f %.2f %.2f" % tuple([abs(z) / D2R for z in Mp] + [abs(z) / D2R for z in Xp]))
print("unpowered data: ratio %.3f, X2/X1^2 %.3f%+.3fi, X3/X1^3 %.2f%+.2fi" % (ratio_u, r2u.real, r2u.imag, r3u.real, r3u.imag))


def res(p):
    fw, zw, mu = p
    M = harm(2 * np.pi * fw, zw, mu)
    if not np.all(np.isfinite(M)): return np.full(5, 1e3)
    m2, m3 = M[1] / M[0] ** 2, M[2] / M[0] ** 3
    return np.array([(abs(M[0]) / abs(Xp[0]) - ratio_u) / 0.03, (m2.real - r2u.real) / 0.05, (m2.imag - r2u.imag) / 0.05,
                     (m3.real - r3u.real) / 0.5, (m3.imag - r3u.imag) / 0.5])


best = None
for fw0 in (12.0, 14.0, 16.0, 18.0, 22.0, 30.0, 45.0, 56.0):
    for mu0 in (0.05, 0.2, 0.6):
        r = least_squares(res, [fw0, 0.05, mu0], bounds=([5, 0.005, 0.0], [80, 0.6, 3.0]), x_scale=[1, 0.02, 0.05], max_nfev=200)
        if best is None or r.cost < best.cost: best = r
        print("start fw %4.1f mu %.2f -> fw %.2f zw %.3f mu %.3f cost %.2f" % (fw0, mu0, *r.x, r.cost), flush=True)
fw, zw, mu = best.x
J = best.jac; se = np.sqrt(np.diag(np.linalg.pinv(J.T @ J) * max(1.0, np.sum(best.fun ** 2) / max(1, 5 - 3))))
M = harm(2 * np.pi * fw, zw, mu)
print("best: f_w %.2f ± %.2f Hz, zeta_w %.3f ± %.3f, mu %.3f ± %.3f, residuals (in s.d.) %s" % (fw, se[0], zw, se[1], mu, se[2], np.round(best.fun, 2)))
print("model unpowered: A1 %.2f A2 %.2f A3 %.2f deg | data %.2f %.2f %.2f" % tuple([abs(z) / D2R for z in M] + [abs(z) / D2R for z in Xu]))
# waveforms for the figure
dt = 1 / 8000
yu = sim(w, zeta, beta_p, G, tau, 2 * np.pi * fw, zw, mu, 8.0, 4.0, dt); yp = sim(w, zeta, beta_p, G, tau, 2 * np.pi * 1e3, 0.5, 0.0, 8.0, 4.0, dt)
def cyc(y, nb=25):
    t = np.arange(len(y)) * dt; m = t > 2.5; ph = (t[m] * 8.0) % 1; z = y[m] / D2R
    b = np.linspace(0, 1, nb + 1); idx = np.minimum(np.digitize(ph, b) - 1, nb - 1)
    A = np.array([z[idx == k].mean() for k in range(nb)]); A -= A.mean(); return np.roll(A, -int(np.argmin(A)))
# spectrum of the wing mode transfer (how 16 Hz is amplified)
fq = np.linspace(1, 60, 400)
Hq = [abs((2 * np.pi * x) ** 2 / ((2 * np.pi * fw) ** 2 - (2 * np.pi * x) ** 2 + 2j * zw * (2 * np.pi * fw) * 2 * np.pi * x)) for x in fq]
json.dump(dict(f_w=fw, f_w_se=se[0], zeta_w=zw, zeta_w_se=se[1], mu=mu, mu_se=se[2], cost=float(best.cost), residuals=best.fun.tolist(),
               model_unpowered=[abs(z) / D2R for z in M], data_unpowered=[abs(z) / D2R for z in Xu],
               model_powered=[abs(z) / D2R for z in Mp], data_powered=[abs(z) / D2R for z in Xp],
               cycle_model_unpowered=cyc(yu).tolist(), cycle_model_powered=cyc(yp).tolist(), H_f=fq.tolist(), H=Hq),
          open(os.path.join(DATA, "two_mode.json"), "w"), indent=1)
