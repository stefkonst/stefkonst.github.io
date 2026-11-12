"""Alternative to the wing mode (which the free decays rule out: they show one mode only): an asymmetric,
velocity-dependent force. A tense membrane holds its camber one way, so its air drag differs between the two
half-strokes (like a cup anemometer); a slack membrane flips camber each half-stroke, so its drag is symmetric.
    x'' + 2 zeta w x' + w^2 x + beta x^2 + gamma x'|x'| + kappa x'^2 = G u(t)
gamma: symmetric quadratic (air) damping, odd harmonics; kappa: asymmetric drag, even harmonics with a phase.
Powered: beta_p (harmonic balance), gamma = kappa = 0 (reference). Unpowered: (beta_u, gamma_u, kappa_u), same w
(no drift). Three unknowns against five clock-free numbers (size ratio, complex X2/X1^2, complex X3/X1^3);
then the free decay, which this model predicts with a single mode.
Run: python drag_model.py -> ../data/drag_model.json"""
import os, json, numpy as np
from numba import njit
from scipy.optimize import least_squares
import tm_core as T


@njit(cache=True)
def sim(w, zeta, beta, gamma, kappa, G, tau, f, Tt, dt, stop):
    n = int(Tt / dt); x = 0.0; v = 0.0; u = 0.0; out = np.empty(n)
    for k in range(n):
        t = k * dt
        s = np.sin(np.pi * f * t) ** 4 if (stop < 0 or t < stop) else 0.0
        u += dt * (s - u) / tau
        a = G * u - 2 * zeta * w * v - w * w * x - beta * x * x - gamma * v * abs(v) - kappa * v * v
        v += dt * a; x += dt * v
        out[k] = x
    return out


def harm(beta, gamma, kappa, G_=T.G, f=8.0, dt=1 / 8000):
    y = sim(T.W0, T.ZETA, beta, gamma, kappa, G_, T.TAU, f, 4.0, dt, -1.0); t = np.arange(len(y)) * dt; m = t > 2.5; Wd = 2 * np.pi * f
    return [2 * np.mean(y[m] * np.exp(-1j * h * Wd * t[m])) for h in (1, 2, 3)]


Bp = T.measured_blocks(T.PW, 8.0); Bu = T.measured_blocks(T.UN, 8.0)
ratio, r2, r3 = T.targets(Bp, Bu); Xp1 = np.mean(np.abs(Bp[:, 0]))
Mp = harm(T.BETA, 0.0, 0.0); Gs = T.G * Xp1 / abs(Mp[0])       # rescale G so the powered model has the measured size
Mp = harm(T.BETA, 0.0, 0.0, Gs)
print("powered reference: A1 %.2f A2 %.2f A3 %.2f deg" % tuple(abs(z) / T.D2R for z in Mp))


def res(p, sig=(0.03, 0.05, 0.5)):
    bu, gu, ku = p
    M = harm(bu, gu, ku, Gs)
    if not np.all(np.isfinite(M)): return np.full(5, 1e3)
    m2, m3 = M[1] / M[0] ** 2, M[2] / M[0] ** 3
    return np.array([(abs(M[0]) / Xp1 - ratio) / sig[0], (m2.real - r2.real) / sig[1], (m2.imag - r2.imag) / sig[1], (m3.real - r3.real) / sig[2], (m3.imag - r3.imag) / sig[2]])


best = None
for b0 in (-15000, -5000, 3000):
    for g0 in (0.0, 1.0, 5.0):
        for k0 in (-5.0, 0.0, 5.0):
            r = least_squares(res, [b0, g0, k0], bounds=([-6e4, 0, -200], [6e4, 200, 200]), x_scale=[2000, 1, 1], max_nfev=300)
            if best is None or r.cost < best.cost: best = r
bu, gu, ku = best.x
J = best.jac; se = np.sqrt(np.diag(np.linalg.pinv(J.T @ J) * max(1.0, 2 * best.cost / 2)))
M = harm(bu, gu, ku, Gs)
print("unpowered fit: beta %.0f ± %.0f, gamma %.3f ± %.3f, kappa %.3f ± %.3f, cost %.2f, residuals (s.d.) %s" % (bu, se[0], gu, se[1], ku, se[2], best.cost, np.round(best.fun, 2)))
print("unpowered model A1 %.2f A2 %.2f A3 %.2f deg | data %.2f %.2f %.2f" % tuple([abs(z) / T.D2R for z in M] + [np.mean(np.abs(Bu[:, h])) / T.D2R for h in range(3)]))
# control: without the asymmetric drag (kappa = 0)
rc = least_squares(lambda q: res([q[0], q[1], 0.0]), [bu, gu], bounds=([-6e4, 0], [6e4, 200]), x_scale=[2000, 1])
print("control kappa = 0: cost %.2f residuals %s" % (rc.cost, np.round(rc.fun, 2)))
# free decay of this model (unpowered state) after the drive stops
def decay(beta, gamma, kappa, G_, stop=2.2, dt=1 / 8000):
    y = sim(T.W0, T.ZETA, beta, gamma, kappa, G_, T.TAU, 8.0, stop + 1.0, dt, stop) / T.D2R
    t = np.arange(len(y)) * dt; m = t > stop + 0.05; z = y[m] - y[m][-1600:].mean(); tt = t[m]
    zc = np.where(np.diff(np.sign(z)) != 0)[0]; out = []
    for i in range(len(zc) - 2):
        a_, b_ = zc[i], zc[i + 2]; amp = 0.5 * (np.abs(z[a_:zc[i + 1]]).max() + np.abs(z[zc[i + 1]:b_]).max())
        if amp > 0.3: out.append((float(amp), float(1 / (tt[b_] - tt[a_]))))
    return out
dec = decay(bu, gu, ku, Gs * (1400 / 1200) ** 2)
print("decay model:", [(round(a, 2), round(f, 2)) for a, f in dec[:8]])
json.dump(dict(Gs=Gs, beta_p=T.BETA, unpowered=dict(beta=bu, gamma=gu, kappa=ku, se=se.tolist(), cost=float(best.cost), residuals=best.fun.tolist()),
               model_unpowered=[abs(z) / T.D2R for z in M], model_powered=[abs(z) / T.D2R for z in Mp],
               control_no_kappa=dict(cost=float(rc.cost), residuals=rc.fun.tolist()), decay=dec),
          open(os.path.join(T.DATA, "drag_model.json"), "w"), indent=1)
