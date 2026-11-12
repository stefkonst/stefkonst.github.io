"""Core of the stroke + wing-mode model (shared by two_mode.py and the tests).
    x'' + 2 zeta w x' + w^2 x + beta x^2 = G u(t) - mu w_w^2 q        stroke (per unit inertia)
    q'' + 2 zeta_w w_w q' + w_w^2 q = -x''                              wing mode, driven by the stroke
    tau u' + u = s(t), s = sin^4(pi f t) while driven                   DEA force, lagged
The video sees x + q."""
import os, json, numpy as np
from numba import njit
HERE = os.path.dirname(os.path.abspath(__file__)); DATA = os.path.join(HERE, "..", "data"); TR = os.path.join(DATA, "tracks")
D2R = np.pi / 180; FS = 200.0
SW = json.load(open(os.path.join(DATA, "sweepfit.json"))); P = SW["fits"]["zeta 0.07 (decay)"]
EO = json.load(open(os.path.join(DATA, "edge_on_stroke.json")))
W0 = 2 * np.pi * P["fn"]; ZETA = P["zeta"]; TAU = P["tau"]; BETA = EO["beta_p"]; G = EO["G"]
PW, UN = "full_DEA_core_wing_8hz_1200V_side", "only_DEA_core_8hz_1200V_side"


@njit(cache=True)
def sim(w, zeta, beta, G, tau, ww, zw, mu, f, T, dt, stop):
    n = int(T / dt); x = 0.0; v = 0.0; q = 0.0; qd = 0.0; u = 0.0; out = np.empty(n)
    for k in range(n):
        t = k * dt
        s = np.sin(np.pi * f * t) ** 4 if (stop < 0 or t < stop) else 0.0
        u += dt * (s - u) / tau
        a = G * u - 2 * zeta * w * v - w * w * x - beta * x * x - mu * ww * ww * q
        qa = -a - 2 * zw * ww * qd - ww * ww * q
        v += dt * a; x += dt * v; qd += dt * qa; q += dt * qd
        out[k] = x + q
    return out


def harm(fw, zw, mu, f=8.0, G_=None, T=4.0, dt=1 / 8000):
    y = sim(W0, ZETA, BETA, G if G_ is None else G_, TAU, 2 * np.pi * fw, zw, mu, f, T, dt, -1.0)
    t = np.arange(len(y)) * dt; m = t > 2.5; Wd = 2 * np.pi * f
    return [2 * np.mean(y[m] * np.exp(-1j * h * Wd * t[m])) for h in (1, 2, 3)]


def measured_blocks(n, f, block=0.5):
    """Complex harmonics X1..X3 (rad) per block of the driven part of a run."""
    a = np.load(os.path.join(TR, n + ".npz"))["ang"] * D2R; t = np.arange(len(a)) / FS
    rest = np.nanmedian(a[:40]); on = int(np.argmax(np.abs(a - rest) > 0.5 * D2R))
    g = np.zeros(len(a), bool); g[on + int(0.5 * FS):] = True; g &= np.isfinite(a); Wd = 2 * np.pi * f
    out = []
    for b0 in np.arange(t[g][0], t[g][-1] - block + 1e-9, block):
        m = g & (t >= b0) & (t < b0 + block)
        X = np.column_stack([np.ones(m.sum())] + [fn(h * Wd * t[m]) for h in (1, 2, 3) for fn in (np.cos, np.sin)])
        c = np.linalg.lstsq(X, a[m], rcond=None)[0]
        out.append([c[1] - 1j * c[2], c[3] - 1j * c[4], c[5] - 1j * c[6]])
    return np.array(out)


def targets(blocks_p, blocks_u):
    """The five clock-free numbers: |X1u|/|X1p|, X2u/X1u^2, X3u/X1u^3 (from block means of the invariants)."""
    ratio = np.mean(np.abs(blocks_u[:, 0])) / np.mean(np.abs(blocks_p[:, 0]))
    r2 = np.mean(blocks_u[:, 1] / blocks_u[:, 0] ** 2); r3 = np.mean(blocks_u[:, 2] / blocks_u[:, 0] ** 3)
    return ratio, r2, r3


def residuals(p, ratio, r2, r3, Xp1, sig=(0.03, 0.05, 0.5)):
    fw, zw, mu = p
    M = harm(fw, zw, mu)
    if not np.all(np.isfinite(M)): return np.full(5, 1e3)
    m2, m3 = M[1] / M[0] ** 2, M[2] / M[0] ** 3
    return np.array([(abs(M[0]) / Xp1 - ratio) / sig[0], (m2.real - r2.real) / sig[1], (m2.imag - r2.imag) / sig[1],
                     (m3.real - r3.real) / sig[2], (m3.imag - r3.imag) / sig[2]])
