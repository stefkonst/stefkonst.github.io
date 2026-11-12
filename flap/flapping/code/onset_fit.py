"""Per-run natural frequency from the switch-on transient, with a physical model.
The stroke angle is a damped oscillator started from rest at t0 and driven by the DEA force:
    x'' + 2 z w x' + w^2 x = w^2 K u(t),   tau u' + u = s(t),   s = sin^4(pi f (t - t0)) for t > t0
(sine-squared voltage from 0 V, force ~ V^2). Fitted per run on the first 1.5 s after onset: f_n, z, K, t0 (tau fixed).
The beating between the drive and the free oscillation in the first cycles pins f_n for each run on its own,
so runs from different sessions can be compared without assuming the robot did not change."""
import os, json, numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares
HERE = os.path.dirname(os.path.abspath(__file__)); DATA = os.path.join(HERE, "..", "data"); TR = os.path.join(DATA, "tracks")
fs = 200.0
RUNS = [("full_DEA_core_wing_5hz_1200v_side", 5, "12:17"), ("full_DEA_core_wing_7hz_1200V_side", 7, "12:23"),
        ("full_DEA_core_wing_8hz_1200V_side", 8, "12:22"), ("full_DEA_core_wing_9hz_1200V_side", 9, "12:25"),
        ("full_DEA_core_wing_10hz_1200V_side", 10, "12:19"), ("full_DEA_core_wing_8hz_1400V_side", 8, "12:28"),
        ("only_DEA_core_8hz_1200V_side", 8, "12:42")]


def simulate(t, f, fn, z, K, t0, tau):
    """Linear response by exact discretisation (scipy.signal.lsim) on a fine grid, then sampled at the frames."""
    from scipy.signal import lti, lsim
    w = 2 * np.pi * fn
    num, den = [K * w * w], np.polymul([1, 2 * z * w, w * w], [tau, 1] if tau > 0 else [1])
    tg = np.arange(t[0], t[-1] + 1e-9, 1 / (fs * 8))
    s = np.where(tg > t0, np.sin(np.pi * f * (tg - t0)) ** 4, 0.0)
    _, x, _ = lsim(lti(num, den), s, tg)
    return np.interp(t, tg, x)


def fit(n, f, tau=0.04, span=1.5, sign=None):
    a = np.load(os.path.join(TR, n + ".npz"))["ang"]; t = np.arange(len(a)) / fs
    rest = np.nanmedian(a[:50]); on = int(np.argmax(np.abs(a - rest) > 0.5))
    m = (t > t[on] - 0.1) & (t < t[on] + span) & np.isfinite(a)
    tt, y = t[m], a[m] - rest
    sg = np.sign(np.nanmean(y)) if sign is None else sign
    best = None
    for fn0 in np.arange(7.0, 10.6, 0.25):
        p0 = [fn0, 0.08, sg * max(abs(np.nanmean(y)), 1.0), t[on] - 0.03]
        r = least_squares(lambda p: simulate(tt, f, p[0], p[1], p[2], p[3], tau) - y, p0,
                          bounds=([6, 0.01, -60, t[on] - 0.2], [11, 0.4, 60, t[on] + 0.02]), x_scale=[1, 0.05, 2, 0.01])
        if best is None or r.cost < best.cost: best = r
    p = best.x; J = best.jac; dof = max(1, len(y) - 4)
    se = np.sqrt(np.diag(np.linalg.pinv(J.T @ J) * np.sum(best.fun ** 2) / dof))
    return dict(run=n, f=f, f_n=float(p[0]), f_n_se=float(se[0]), zeta=float(p[1]), zeta_se=float(se[1]), K=float(p[2]),
                t0=float(p[3]), rms=float(np.sqrt(np.mean(best.fun ** 2))), tau=tau, t=tt.tolist(), y=y.tolist(),
                yfit=(y + best.fun).tolist())


if __name__ == "__main__":
    out = []
    for tau in (0.0, 0.04):
        for n, f, clock in RUNS:
            r = fit(n, f, tau); r["clock"] = clock; out.append(r)
            print("tau %2.0f ms  %-38s %s  f %4.1f  fn %.2f ± %.2f  zeta %.3f  K %.2f  rms %.2f" % (tau * 1e3, n, clock, f, r["f_n"], r["f_n_se"], r["zeta"], r["K"], r["rms"]), flush=True)
    json.dump(out, open(os.path.join(DATA, "onset_fit.json"), "w"))
