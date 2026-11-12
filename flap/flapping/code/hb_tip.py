"""Harmonic balance on the tip-based stroke angle (tip_track.py), which does not switch features.
Same method as harmonic_balance.py: beta from the clock-free ratio X2/X1^2 with the fixed sin^4 drive."""
import os, json, numpy as np
from harmonic_balance import solve_beta, P
HERE = os.path.dirname(os.path.abspath(__file__)); DATA = os.path.join(HERE, "..", "data")
TH = json.load(open(os.path.join(DATA, "tip_harmonics.json"))); FS = 200.0
out = {}
for n, r in TH.items():
    if "15hz" in n or "1hz" in n or "5hz" in n: continue
    X1, X2 = complex(*r["X"][0]), complex(*r["X"][1]); f = r["f"]
    # block spread
    a = np.load(os.path.join(DATA, "tracks", "tip_" + n + ".npz"))["ang"]; t = np.arange(len(a)) / FS
    g = (t > 0.9) & np.isfinite(a)
    if "1400" in n: g &= t < 2.15
    Rs = []
    for b0 in np.arange(t[g][0], t[g][-1] - 0.5, 0.5):
        m = g & (t >= b0) & (t < b0 + 0.5)
        Xb = np.column_stack([np.ones(m.sum())] + [fn(h * 2 * np.pi * f * t[m]) for h in (1, 2) for fn in (np.cos, np.sin)])
        c = np.linalg.lstsq(Xb, a[m], rcond=None)[0]; Rs.append((c[3] - 1j * c[4]) / (c[1] - 1j * c[2]) ** 2)
    b, drv, G = solve_beta(X2 / X1 ** 2, X1, f, P["fn"], P["zeta"], P["tau"])
    bs = np.array([solve_beta(q, X1, f, P["fn"], P["zeta"], P["tau"])[0] for q in Rs])
    A = abs(X1); wn2 = (2 * np.pi * P["fn"]) ** 2
    out[n] = dict(beta=[b.real, b.imag], sd=[float(np.std(bs.real)), float(np.std(bs.imag))], asym=float(b.real * A / wn2), asym_sd=float(np.std(bs.real) * A / wn2), A1_deg=float(np.degrees(A)))
    print("%-38s A1 %.2f deg  beta %+7.0f ± %4.0f (imag %+6.0f ± %4.0f)  asym %.3f ± %.3f" % (n, np.degrees(A), b.real, np.std(bs.real), b.imag, np.std(bs.imag), out[n]["asym"], out[n]["asym_sd"]))
json.dump(out, open(os.path.join(DATA, "hb_tip.json"), "w"), indent=1)
