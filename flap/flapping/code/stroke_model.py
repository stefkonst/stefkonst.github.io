"""Stroke model of B-cube, fitted to the video.

One rotational mode (core + transmission + leading edge), driven by the DEA core through a first-order force lag, with
a quadratic term in the restoring moment (per unit inertia):

    theta'' + 2 zeta w theta' + w^2 theta + beta theta^2 = g u(t),      tau u' + u = sin^4(pi f t)

sine-squared voltage from 0 V, force ~ V^2. theta in rad, side-view angle (positive = tip down in the image).
A quadratic term makes the stroke lopsided (2nd harmonic ~ beta A1^2) and softens it at large amplitude (the
frequency falls as beta^2 A^2), the two features of the video. The wing state sets (w, beta):

    wing powered   -> (w_p, beta_p)          the powered 8 Hz run and the powered sweep (5, 7, 9, 10 Hz)
    wing at 0 V    -> (w_u, beta_u)          the unpowered 8 Hz run AND the free decay after the 1400 V run
                                             (filmed in the other session, 14 min earlier)

Shared: zeta, tau, g (the core and its drive are the same). If one set (w_u, beta_u) fits both the unpowered run and
the decay of the other session, the difference between the runs follows the wing's state, not the clock.
Controls: beta = 0 (linear), and one state for both runs.
Run: python stroke_model.py -> ../data/stroke_model.json"""
import os, json, numpy as np
from scipy.optimize import least_squares
from numba import njit
HERE = os.path.dirname(os.path.abspath(__file__)); DATA = os.path.join(HERE, "..", "data"); TR = os.path.join(DATA, "tracks")
FS = 200.0
D2R = np.pi / 180
NAMES = ["zeta", "tau (s)", "g (rad/s2)", "w_p (rad/s)", "beta_p (1/s2)", "w_u (rad/s)", "beta_u (1/s2)"]


@njit(cache=True)
def sim(w, zeta, beta, g, tau, f, T, dt, free_from):
    n = int(T / dt); th = 0.0; om = 0.0; u = 0.0
    out = np.empty(n)
    for k in range(n):
        t = k * dt
        s = np.sin(np.pi * f * t) ** 4 if (free_from < 0 or t < free_from) else 0.0
        u += dt * (s - u) / tau
        acc = g * u - 2 * zeta * w * om - w * w * th - beta * th * th
        om += dt * acc; th += dt * om
        if abs(th) > 10: th = 10.0
        out[k] = th
    return out


def run(w, zeta, beta, g, tau, f, T=4.0, dt=1 / 4000, free_from=-1.0):
    th = sim(w, zeta, beta, g, tau, f, T, dt, free_from)
    return np.arange(len(th)) * dt, th


def cycle(w, zeta, beta, g, tau, f=8.0, nb=25):
    t, th = run(w, zeta, beta, g, tau, f); m = t > 2.5; ph = (t[m] * f) % 1; y = th[m] / D2R
    b = np.linspace(0, 1, nb + 1); idx = np.minimum(np.digitize(ph, b) - 1, nb - 1)
    A = np.array([y[idx == k].mean() for k in range(nb)]); A -= A.mean()
    return np.roll(A, -int(np.argmin(A)))


def harmonics(w, zeta, beta, g, tau, f, H=3):
    t, th = run(w, zeta, beta, g, tau, f); m = t > 2.5; y = th[m] / D2R; tt = t[m]
    if not np.all(np.isfinite(y)): return [1e3] * H
    return [float(2 * abs(np.mean(y * np.exp(-2j * np.pi * h * f * tt)))) for h in range(1, H + 1)]


def decay(w, zeta, beta, g, tau, stop=2.2):
    t, th = run(w, zeta, beta, g, tau, 8.0, T=stop + 0.9, free_from=stop)
    z = th[t > stop + 0.05] / D2R; tt = t[t > stop + 0.05]; z = z - z[-200:].mean()
    zc = np.where(np.diff(np.sign(z)) != 0)[0]; out = []
    for i in range(len(zc) - 2):
        a, b = zc[i], zc[i + 2]; amp = 0.5 * (np.abs(z[a:zc[i + 1]]).max() + np.abs(z[zc[i + 1]:b]).max())
        out.append((amp, 1 / (tt[b] - tt[a])))
    return np.array(out) if out else np.zeros((0, 2))


def data_cycle(n, f=8.0, nb=25):
    a = np.load(os.path.join(TR, n + ".npz"))["ang"]; t = np.arange(len(a)) / FS; g = (t > 0.6) & np.isfinite(a)
    ph = (t[g] * f) % 1; y = a[g] - np.nanmean(a[g]); b = np.linspace(0, 1, nb + 1); idx = np.minimum(np.digitize(ph, b) - 1, nb - 1)
    A = np.array([y[idx == k].mean() for k in range(nb)]); S = np.array([y[idx == k].std() for k in range(nb)])
    sh = int(np.argmin(A)); return np.roll(A, -sh), np.roll(S, -sh)


PW, UN = "full_DEA_core_wing_8hz_1200V_side", "only_DEA_core_8hz_1200V_side"

if __name__ == "__main__":
    SW = json.load(open(os.path.join(DATA, "sweepfit.json"))); DEC = json.load(open(os.path.join(DATA, "decays.json")))
    R = {r["run"]: r for r in SW["runs"]}
    yp, sp = data_cycle(PW, R[PW]["f"]); yu, su = data_cycle(UN, R[UN]["f"])
    sw_d = [(R[n]["f"], R[n]["A"][0]) for n in R if n.startswith("full") and R[n]["V"] == 1200 and R[n]["f_drive"] in (5, 7, 9, 10)]
    fa_d = np.array([q for q in DEC[0]["f_vs_amp"] if 0.8 < q[0] < 4.0])
    G14 = (1400 / 1200) ** 2

    def res(q, parts=False):
        zeta, tau, g, wp, bp, wu, bu = q
        r_cp = (cycle(wp, zeta, bp, g, tau) - yp) / 0.7
        r_cu = (cycle(wu, zeta, bu, g, tau) - yu) / 0.7
        r_sw = np.array([(harmonics(wp, zeta, bp, g, tau, f, 1)[0] - a) / 0.6 for f, a in sw_d])
        dm = decay(wu, zeta, bu, g * G14, tau)          # after the drive stops the wing is at 0 V
        if len(dm) >= 2 and np.all(np.isfinite(dm)):
            o = np.argsort(dm[:, 0]); fm = np.interp(fa_d[:, 0], dm[o, 0], dm[o, 1])
        else:
            fm = np.full(len(fa_d), 30.0)
        r_dc = (fm - fa_d[:, 1]) / 0.3
        if parts: return dict(cyc_p=r_cp, cyc_u=r_cu, sweep=r_sw, decay=r_dc)
        out = np.r_[r_cp, r_cu, r_sw, r_dc]
        return np.where(np.isfinite(out), out, 1e3)

    best = None
    for bsign in (-1, 1):
        for b0 in (800.0, 2500.0):
            for w0 in (50.0, 56.0):
                q0 = [0.07, 0.03, 300.0, w0, bsign * b0 * 0.3, w0 + 3, bsign * b0]
                lb = [0.01, 0.002, 10, 30, -2e4, 30, -2e4]; ub = [0.4, 0.2, 3000, 90, 2e4, 90, 2e4]
                try:
                    r = least_squares(res, q0, bounds=(lb, ub), x_scale=[0.02, 0.01, 50, 3, 300, 3, 300], max_nfev=500)
                except Exception as e:
                    print("fail", e); continue
                print("start b %+.0f w %.0f -> cost %.1f  %s" % (bsign * b0, w0, r.cost, np.round(r.x, 3)), flush=True)
                if best is None or r.cost < best.cost: best = r
    q = best.x
    J = best.jac; cov = np.linalg.pinv(J.T @ J) * np.sum(best.fun ** 2) / max(1, len(best.fun) - len(q)); se = np.sqrt(np.diag(cov))
    for nm, v, e in zip(NAMES, q, se): print("%-14s %10.4f ± %.4f" % (nm, v, e))
    zeta, tau, g, wp, bp, wu, bu = q
    parts = res(q, True)
    for k, v in parts.items(): print("rms %-6s %.2f (in units of the noise)" % (k, np.sqrt(np.mean(v ** 2))))
    hp, hu = harmonics(wp, zeta, bp, g, tau, 8.0), harmonics(wu, zeta, bu, g, tau, 8.0)
    print("harmonics powered   model", np.round(hp, 2), "data", np.round(R[PW]["A"][:3], 2))
    print("harmonics unpowered model", np.round(hu, 2), "data", np.round(R[UN]["A"][:3], 2))
    ff = np.arange(1.0, 16.01, 0.25)
    out = dict(params=dict(zip(NAMES, map(float, q))), se=dict(zip(NAMES, map(float, se))), cost=float(best.cost),
               rms={k: float(np.sqrt(np.mean(v ** 2))) for k, v in parts.items()},
               f_p=float(wp / 2 / np.pi), f_u=float(wu / 2 / np.pi),
               cycle_powered=dict(data=yp.tolist(), sd=sp.tolist(), model=cycle(wp, zeta, bp, g, tau).tolist()),
               cycle_unpowered=dict(data=yu.tolist(), sd=su.tolist(), model=cycle(wu, zeta, bu, g, tau).tolist()),
               harm_powered=dict(model=hp, data=R[PW]["A"][:3]), harm_unpowered=dict(model=hu, data=R[UN]["A"][:3]),
               sweep=dict(f=ff.tolist(), powered=[harmonics(wp, zeta, bp, g, tau, f, 1)[0] for f in ff],
                          unpowered=[harmonics(wu, zeta, bu, g, tau, f, 1)[0] for f in ff],
                          data=[(R[n]["f"], R[n]["A"][0], R[n]["A1_sd"]) for n in R if n.startswith("full") and R[n]["V"] == 1200]),
               decay=dict(model=decay(wu, zeta, bu, g * G14, tau).tolist(), data=fa_d.tolist()))

    def res_lin(q2):
        z2, t2, g2, w2p, w2u = q2
        return res([z2, t2, g2, w2p, 0.0, w2u, 0.0])
    rl = least_squares(res_lin, [zeta, tau, g, wp, wu], bounds=([0.01, 0.002, 10, 30, 30], [0.4, 0.2, 3000, 90, 90]), max_nfev=300)
    out["linear_control"] = dict(cost=float(rl.cost), params=rl.x.tolist(), rms={k: float(np.sqrt(np.mean(v ** 2))) for k, v in res([rl.x[0], rl.x[1], rl.x[2], rl.x[3], 0, rl.x[4], 0], True).items()})
    print("linear control (beta = 0): cost %.1f against %.1f" % (rl.cost, best.cost), out["linear_control"]["rms"])

    def res_same(q3):
        z3, t3, g3, w3, b3 = q3
        return res([z3, t3, g3, w3, b3, w3, b3])
    rs = least_squares(res_same, [zeta, tau, g, wu, bu], bounds=([0.01, 0.002, 10, 30, -2e4], [0.4, 0.2, 3000, 90, 2e4]), max_nfev=300)
    out["same_state_control"] = dict(cost=float(rs.cost), params=rs.x.tolist(), rms={k: float(np.sqrt(np.mean(v ** 2))) for k, v in res([rs.x[0], rs.x[1], rs.x[2], rs.x[3], rs.x[4], rs.x[3], rs.x[4]], True).items()})
    print("same state for both runs: cost %.1f" % rs.cost, out["same_state_control"]["rms"])
    json.dump(out, open(os.path.join(DATA, "stroke_model.json"), "w"), indent=1)
