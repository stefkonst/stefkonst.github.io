"""Asymmetry of the stroke spring, per run, by harmonic balance (no waveform fit, no drive signal needed).

Model (per unit inertia):  x'' + 2 zeta w x' + w^2 x + beta x^2 = G * u(t),   u = sin^4(pi f t) lagged by tau
sin^4 = 3/8 - (1/2) cos(W t) + (1/8) cos(2 W t),  W = 2 pi f. Writing x = Re sum X_h e^{i h W t}:
    X1 = G c1 L1 H1,     X2 = G c2 L2 H2 - (beta/2) X1^2 H2
    c1 = -1/2, c2 = 1/8, L_h = 1/(1 + i h W tau), H_h = 1/(w^2 - (hW)^2 + 2 i zeta w h W)
The ratio R = X2 / X1^2 does not depend on when the clock starts, so it is measured directly from the video:
    beta = (2 / H2) * ( c2 L2 / (G c1^2 L1^2 H1^2) * H2 ... ) -> solved below as a complex number.
If the model holds, beta comes out real (small imaginary part). The drive's own 16 Hz part is the same in every run;
the part left over is the spring's lopsidedness.
Run: python harmonic_balance.py -> ../data/harmonic_balance.json"""
import os, json, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); DATA = os.path.join(HERE, "..", "data"); TR = os.path.join(DATA, "tracks")
FS = 200.0; D2R = np.pi / 180
SW = json.load(open(os.path.join(DATA, "sweepfit.json"))); P = SW["fits"]["zeta 0.07 (decay)"]
R = {r["run"]: r for r in SW["runs"]}


def measured(n, f):
    a = np.load(os.path.join(TR, n + ".npz"))["ang"] * D2R; t = np.arange(len(a)) / FS
    rest = np.nanmedian(a[:40]); on = int(np.argmax(np.abs(a - rest) > 0.5 * D2R))
    g = np.zeros(len(a), bool); g[on + int(0.5 * FS):] = True
    if "1400V" in n: g[int(2.15 * FS):] = False
    g &= np.isfinite(a)
    W = 2 * np.pi * f
    X = np.column_stack([np.ones(g.sum())] + [fn(h * W * t[g]) for h in (1, 2, 3) for fn in (np.cos, np.sin)])
    c = np.linalg.lstsq(X, a[g], rcond=None)[0]
    X1 = c[1] - 1j * c[2]; X2 = c[3] - 1j * c[4]           # x = Re X e^{iWt}
    # block spread of R
    Rs = []
    for b0 in np.arange(t[g][0], t[g][-1] - 0.5, 0.5):
        m = g & (t >= b0) & (t < b0 + 0.5)
        Xb = np.column_stack([np.ones(m.sum())] + [fn(h * W * t[m]) for h in (1, 2) for fn in (np.cos, np.sin)])
        cb = np.linalg.lstsq(Xb, a[m], rcond=None)[0]; Rs.append((cb[3] - 1j * cb[4]) / (cb[1] - 1j * cb[2]) ** 2)
    return X1, X2, np.array(Rs)


def solve_beta(R_meas, X1, f, fn, zeta, tau):
    W = 2 * np.pi * f; w = 2 * np.pi * fn
    H = lambda h: 1 / (w * w - (h * W) ** 2 + 2j * zeta * w * h * W)
    L = lambda h: 1 / (1 + 1j * h * W * tau)
    c1, c2 = -0.5, 0.125
    G = X1 / (c1 * L(1) * H(1))                                # complex: absorbs the unknown clock phase
    # R = X2/X1^2 = c2 L2 H2 G / X1^2 - (beta/2) H2. With X1 = G c1 L1 H1, G/X1^2 = 1/(G c1^2 L1^2 H1^2):
    drive = c2 * L(2) * H(2) * G / X1 ** 2
    beta = -2 * (R_meas - drive) / H(2)
    return beta, drive, G


out = {}
for n in ("full_DEA_core_wing_8hz_1200V_side", "only_DEA_core_8hz_1200V_side", "full_DEA_core_wing_8hz_1400V_side",
          "full_DEA_core_wing_7hz_1200V_side", "full_DEA_core_wing_9hz_1200V_side", "full_DEA_core_wing_10hz_1200V_side",
          "full_DEA_core_wing_5hz_1200v_side"):
    f = R[n]["f"]; X1, X2, Rs = measured(n, f); Rm = X2 / X1 ** 2
    rows = {}
    for lab, fn in (("fn 8.38 (sweep fit)", P["fn"]), ("fn 8.95 (drift case)", 8.95)):
        b, drv, G = solve_beta(Rm, X1, f, fn, P["zeta"], P["tau"])
        bs = np.array([solve_beta(r, X1, f, fn, P["zeta"], P["tau"])[0] for r in Rs])
        rows[lab] = dict(beta_re=float(b.real), beta_im=float(b.imag), beta_re_sd=float(np.std(bs.real)), beta_im_sd=float(np.std(bs.imag)),
                         drive_part=float(abs(drv)), R=float(abs(Rm)))
    out[n] = dict(f=f, A1_deg=float(abs(X1) / D2R), A2_deg=float(abs(X2) / D2R), R_abs=float(abs(Rm)), R_sd=float(np.std(np.abs(Rs))), cases=rows)
    print("%-38s |R| %.3f ± %.3f /rad" % (n, abs(Rm), np.std(np.abs(Rs))))
    for lab, r in rows.items():
        print("    %-22s beta %+8.0f ± %4.0f  (imag %+7.0f ± %4.0f) 1/s2   drive part |R| %.3f" % (lab, r["beta_re"], r["beta_re_sd"], r["beta_im"], r["beta_im_sd"], r["drive_part"]))
json.dump(out, open(os.path.join(DATA, "harmonic_balance.json"), "w"), indent=1)
