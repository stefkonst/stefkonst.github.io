"""Tests of the stroke + wing-mode model.
1. Landscape: f_w fixed on a grid from 6 to 45 Hz, (zeta_w, mu) refitted each time -> cost against f_w.
2. Bootstrap: resample the 0.5 s blocks of both runs (with replacement), recompute the five numbers, refit.
3. Prediction not used in the fit: free decay after the drive stops, model (with the wing mode, i.e. 0 V state)
   against the two measured decays (frequency against amplitude).
Run: python model_tests.py -> ../data/model_tests.json"""
import os, json, numpy as np
from scipy.optimize import least_squares
import tm_core as T
DATA = T.DATA
Bp = T.measured_blocks(T.PW, 8.0); Bu = T.measured_blocks(T.UN, 8.0)
ratio, r2, r3 = T.targets(Bp, Bu); Xp1 = np.mean(np.abs(Bp[:, 0]))
TM = json.load(open(os.path.join(DATA, "two_mode.json")))
p0 = [TM["f_w"], TM["zeta_w"], TM["mu"]]
print("targets: ratio %.3f  r2 %.3f%+.3fi  r3 %.2f%+.2fi" % (ratio, r2.real, r2.imag, r3.real, r3.imag), flush=True)
best = least_squares(lambda p: T.residuals(p, ratio, r2, r3, Xp1), p0, bounds=([5, 0.005, 0], [80, 0.6, 3]), x_scale=[1, 0.02, 0.05])
print("refit with block-mean targets: f_w %.2f zeta_w %.3f mu %.3f cost %.2f" % (*best.x, best.cost), flush=True)

# 1. landscape
grid = np.r_[np.arange(6, 14, 1.0), np.arange(14, 21, 0.5), np.arange(21, 46, 2.0)]
land = []
for fw in grid:
    bb = None
    for zw0, mu0 in ((0.05, 0.2), (0.15, 0.5), (0.03, 0.05)):
        r = least_squares(lambda q: T.residuals([fw, q[0], q[1]], ratio, r2, r3, Xp1), [zw0, mu0], bounds=([0.005, 0], [0.6, 3]), x_scale=[0.02, 0.05], max_nfev=150)
        if bb is None or r.cost < bb.cost: bb = r
    land.append((float(fw), float(bb.cost), float(bb.x[0]), float(bb.x[1])))
    print("f_w %5.1f  cost %8.2f  zeta_w %.3f  mu %.3f" % land[-1], flush=True)

# 2. bootstrap
rng = np.random.default_rng(7); boots = []
for b in range(60):
    ip = rng.integers(0, len(Bp), len(Bp)); iu = rng.integers(0, len(Bu), len(Bu))
    rb, r2b, r3b = T.targets(Bp[ip], Bu[iu]); xb = np.mean(np.abs(Bp[ip, 0]))
    r = least_squares(lambda p: T.residuals(p, rb, r2b, r3b, xb), best.x, bounds=([5, 0.005, 0], [80, 0.6, 3]), x_scale=[1, 0.02, 0.05], max_nfev=200)
    boots.append(r.x.tolist())
    if b % 10 == 0: print("boot", b, np.round(r.x, 3), flush=True)
boots = np.array(boots)
ci = {k: [float(np.percentile(boots[:, i], 2.5)), float(np.median(boots[:, i])), float(np.percentile(boots[:, i], 97.5))] for i, k in enumerate(("f_w", "zeta_w", "mu"))}
print("bootstrap 95% CI:", {k: np.round(v, 3).tolist() for k, v in ci.items()}, flush=True)

# 3. free decay prediction (0 V: wing mode present), drive at 8 Hz then stop
DEC = json.load(open(os.path.join(DATA, "decays.json")))
def decay_curve(fw, zw, mu, G_, stop=2.2, dt=1 / 8000):
    y = T.sim(T.W0, T.ZETA, T.BETA, G_, T.TAU, 2 * np.pi * fw, zw, mu, 8.0, stop + 1.0, dt, stop) / T.D2R
    t = np.arange(len(y)) * dt; m = t > stop + 0.05; z = y[m] - y[m][-int(0.2 / dt):].mean(); tt = t[m]
    zc = np.where(np.diff(np.sign(z)) != 0)[0]; out = []
    for i in range(len(zc) - 2):
        a_, b_ = zc[i], zc[i + 2]; amp = 0.5 * (np.abs(z[a_:zc[i + 1]]).max() + np.abs(z[zc[i + 1]:b_]).max())
        if amp > 0.3: out.append((float(amp), float(1 / (tt[b_] - tt[a_]))))
    return out
fw, zw, mu = best.x
pred_wing = decay_curve(fw, zw, mu, T.G * (1400 / 1200) ** 2)
pred_nowing = decay_curve(fw, zw, 0.0, T.G * (1400 / 1200) ** 2)
print("decay, model with wing mode:", [(round(a, 2), round(f, 2)) for a, f in pred_wing[:8]])
print("decay, model without:", [(round(a, 2), round(f, 2)) for a, f in pred_nowing[:8]])
print("measured 12:28:", [(round(a, 2), round(f, 2)) for a, f in DEC[0]["f_vs_amp"][:8]])
print("measured 12:43:", [(round(a, 2), round(f, 2)) for a, f in DEC[1]["f_vs_amp"][:8]])
json.dump(dict(targets=dict(ratio=ratio, r2=[r2.real, r2.imag], r3=[r3.real, r3.imag]), best=best.x.tolist(), best_cost=float(best.cost),
               landscape=land, bootstrap=boots.tolist(), ci=ci, decay_pred_wing=pred_wing, decay_pred_nowing=pred_nowing,
               decay_meas=[DEC[0]["f_vs_amp"], DEC[1]["f_vs_amp"]]), open(os.path.join(DATA, "model_tests.json"), "w"), indent=1)
