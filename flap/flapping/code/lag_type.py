"""Is the force lag electrical (RC: the voltage on the film lags, then is squared) or mechanical (the force lags)?
Harmonic balance with the fixed sine-squared drive V = Vmax sin^2(pi f t) = Vmax (1 - cos W t)/2:
  mechanical: F ~ V^2 lagged:      force harmonics c1 = -1/2 L1,   c2 = 1/8 L2
  electrical: F ~ (V lagged)^2:    force harmonics c1 = -1/2 L1,   c2 = 1/8 L1^2       (L_h = 1/(1 + i h W tau))
then X1 = G c1 H1, X2 = G c2 H2 - (beta/2) X1^2 H2 with H_h the stroke mode. For each lag type and tau, beta is
solved from the measured X2/X1^2 of every powered run (tip-based); the right lag type gives one real beta for all
runs. Score: spread of beta between runs + size of its imaginary parts, both relative to the block noise.
Run: python lag_type.py -> ../data/lag_type.json"""
import os, json, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); DATA = os.path.join(HERE, "..", "data"); FS = 200.0
SW = json.load(open(os.path.join(DATA, "sweepfit.json"))); P = SW["fits"]["zeta 0.07 (decay)"]
TH = json.load(open(os.path.join(DATA, "tip_harmonics.json")))
RUNS = ["full_DEA_core_wing_7hz_1200V_side", "full_DEA_core_wing_8hz_1200V_side", "full_DEA_core_wing_9hz_1200V_side", "full_DEA_core_wing_8hz_1400V_side", "full_DEA_core_wing_10hz_1200V_side"]


def blocks(n, f):
    a = np.load(os.path.join(DATA, "tracks", "tip_" + n + ".npz"))["ang"]; t = np.arange(len(a)) / FS
    g = (t > 0.9) & np.isfinite(a)
    if "1400" in n: g &= t < 2.15
    out = []
    for b0 in np.arange(t[g][0], t[g][-1] - 0.5, 0.5):
        m = g & (t >= b0) & (t < b0 + 0.5)
        X = np.column_stack([np.ones(m.sum())] + [fn(h * 2 * np.pi * f * t[m]) for h in (1, 2) for fn in (np.cos, np.sin)])
        c = np.linalg.lstsq(X, a[m], rcond=None)[0]; out.append((c[1] - 1j * c[2], c[3] - 1j * c[4]))
    return np.array(out)


def beta_of(X1, X2, f, tau, kind, fn=P["fn"], zeta=P["zeta"]):
    W = 2 * np.pi * f; w = 2 * np.pi * fn
    H = lambda h: 1 / (w * w - (h * W) ** 2 + 2j * zeta * w * h * W)
    L = lambda h: 1 / (1 + 1j * h * W * tau)
    c1 = -0.5 * L(1); c2 = 0.125 * (L(2) if kind == "mechanical" else L(1) ** 2)
    G = X1 / (c1 * H(1))
    return -2 * (X2 / X1 ** 2 - c2 * H(2) * G / X1 ** 2) / H(2)


B = {n: blocks(n, TH[n]["f"]) for n in RUNS}
res = {}
for kind in ("mechanical", "electrical"):
    rows = []
    for tau in np.arange(0.0, 0.121, 0.005):
        bs, es = [], []
        for n in RUNS[:4]:                                # 10 Hz kept as a check
            X1 = np.mean(B[n][:, 0]); X2 = np.mean(B[n][:, 1]); f = TH[n]["f"]
            b = beta_of(X1, X2, f, tau, kind); bb = np.array([beta_of(x1, x2, f, tau, kind) for x1, x2 in B[n]])
            bs.append(b); es.append(np.std(bb.real) / np.sqrt(len(bb)) + 1.0)
        bs = np.array(bs); es = np.array(es)
        mean = np.sum(bs.real / es ** 2) / np.sum(1 / es ** 2)
        chi_spread = np.sum(((bs.real - mean) / es) ** 2); chi_imag = np.sum((bs.imag / es) ** 2)
        rows.append(dict(tau=float(tau), beta=bs.real.tolist(), imag=bs.imag.tolist(), err=es.tolist(), chi_spread=float(chi_spread), chi_imag=float(chi_imag), chi=float(chi_spread + chi_imag)))
    best = min(rows, key=lambda r: r["chi"]); res[kind] = dict(rows=rows, best=best)
    print("%-10s best tau %.0f ms  chi2 %.1f (spread %.1f, imag %.1f)  beta per run %s  imag %s" % (kind, best["tau"] * 1e3, best["chi"], best["chi_spread"], best["chi_imag"], np.round(best["beta"], 0), np.round(best["imag"], 0)))
    for r in rows[::4]: print("   tau %3.0f ms chi2 %.1f" % (r["tau"] * 1e3, r["chi"]))
json.dump(res, open(os.path.join(DATA, "lag_type.json"), "w"), indent=1)
