"""Model tests figure (flap/model): what the stroke models get right and wrong.
a, fit landscape of the linear wing mode (cost against f_w) with the bootstrap interval;
b, free decay: frequency against amplitude, measured (two decays) against the model with and without the wing mode;
c, harmonics of the leading-edge bend, relative to its fundamental, powered (side, diagonal) and unpowered (side);
d, mean leading-edge bend over one cycle, powered and unpowered.
Run: python figures_tests.py -> ../fig/T1_tests.png/svg"""
import os, sys, json
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); FLAP = os.path.dirname(os.path.dirname(HERE))
sys.path[:0] = [os.path.join(os.path.expanduser("~"), "Documents", "paper_tension", "common")]
import style as ST
import matplotlib.pyplot as plt
D = os.path.join(FLAP, "flapping", "data"); TR = os.path.join(D, "tracks")
MT = json.load(open(os.path.join(D, "model_tests.json")))
C = ST.C; FS = 200.0
OUT = os.path.join(HERE, "..", "fig"); os.makedirs(OUT, exist_ok=True)


def bend(n):
    d = np.load(os.path.join(TR, n + ".npz")); xs = d["xs"].astype(float); top = d["top"]
    s = np.full(len(top), np.nan)
    for k in range(len(top)):
        m = np.isfinite(top[k])
        if m.sum() < 12: continue
        x, y = xs[m], top[k][m].astype(float); c = np.polyfit(x, y, 2); L = x[-1] - x[0]; s[k] = -c[0] * L * L / 4 / L * 1000
    return s


def harm(z, f, g, t):
    X = np.column_stack([np.ones(g.sum())] + [fn(2 * np.pi * h * f * t[g]) for h in (1, 2, 3, 4) for fn in (np.cos, np.sin)])
    c = np.linalg.lstsq(X, z[g], rcond=None)[0]; return np.array([np.hypot(c[2 * h - 1], c[2 * h]) for h in (1, 2, 3, 4)])


ST.apply()
fig = plt.figure(figsize=(ST.WIDTH_MM * ST.MM, 140 * ST.MM))
ax = [fig.add_axes([0.09 + c_ * 0.5, 0.57 - r_ * 0.49, 0.38, 0.36]) for r_ in (0, 1) for c_ in (0, 1)]
for i_, a_ in enumerate(ax): a_.text(-0.2, 1.05, "abcd"[i_], transform=a_.transAxes, fontsize=9, fontweight="bold")
a = ax[0]
L = np.array(MT["landscape"]); a.plot(L[:, 0], L[:, 1], "o-", ms=2.5, color="k")
lo, md, hi = MT["ci"]["f_w"]; a.axvspan(lo, hi, color=C[1], alpha=0.25, lw=0)
a.set(yscale="log", xlabel="Wing-mode frequency $f_w$ (Hz)", ylabel="Misfit (sum of squares)")
ST.note(a, "band: bootstrap 95 %%\n%.1f–%.1f Hz" % (lo, hi), xy=(0.40, 0.92), color=C[1])
a = ax[1]
for k, (q, col, lab) in enumerate(((MT["decay_meas"][0], C[3], "measured, 12:28"), (MT["decay_meas"][1], C[1], "measured, 12:43"))):
    q = np.array([v for v in q if v[0] > 0.8]); a.plot(q[:, 0], q[:, 1], "o", ms=3, color=col, label=lab)
pw = np.array(MT["decay_pred_wing"]); pn = np.array(MT["decay_pred_nowing"])
a.plot(pw[:, 0], pw[:, 1], "s-", ms=2.5, lw=0.8, color="k", mfc="white", label="model, linear wing mode")
a.plot(pn[:, 0], pn[:, 1], "-", lw=1, color="0.5", label="model, stroke only")
a.set(xscale="log", xlabel="Amplitude (deg)", ylabel="Free frequency (Hz)", ylim=(6, 18))
a.legend(loc="center right", fontsize=5.5)
a = ax[2]
rows = (("full_DEA_core_wing_8hz_1200V_side", 7.995, C[0], "powered\nside"), ("full_DEA_core_wing_8hz_1200V_diag_good", 7.995, "#7fa8e6", "powered\ndiagonal"), ("only_DEA_core_8hz_1200V_side", 7.997, C[1], "unpowered\nside"))
store = {}
for i, (n, f, col, lab) in enumerate(rows):
    s = bend(n); t = np.arange(len(s)) / FS; g = (t > 0.6) & np.isfinite(s); h = harm(s, f, g, t); store[n] = (s, t, g, f)
    a.bar(np.arange(3) + (i - 1) * 0.26, h[1:] / h[0], width=0.24, color=col, label=lab.replace("\n", ", "))
a.set_xticks(range(3)); a.set_xticklabels(["16 Hz", "24 Hz", "32 Hz"]); a.minorticks_off()
a.set(ylabel="Bend harmonic / fundamental")
a.legend(loc="upper right", fontsize=5.5)
a = ax[3]
for n, col, lab in ((rows[0][0], C[0], "powered"), (rows[2][0], C[1], "unpowered")):
    s, t, g, f = store[n]; ph = (t[g] * f) % 1; z = s[g] - np.nanmean(s[g]); b = np.linspace(0, 1, 26); idx = np.minimum(np.digitize(ph, b) - 1, 24)
    A = np.array([np.nanmean(z[idx == k]) for k in range(25)]); S = np.array([np.nanstd(z[idx == k]) for k in range(25)])
    ang = np.load(os.path.join(TR, n + ".npz"))["ang"]; za = ang[g]; Aa = np.array([np.nanmean(za[idx == k]) for k in range(25)]); sh = int(np.argmin(Aa))
    A, S = np.roll(A, -sh), np.roll(S, -sh); x = (b[:-1] + b[1:]) / 2
    a.plot(x, A, color=col, label=lab); a.fill_between(x, A - S, A + S, color=col, alpha=0.2, lw=0)
a.set(xlabel="Phase from the lowest stroke point", ylabel="Leading-edge bend (‰ of span)")
a.legend(loc="lower right", fontsize=5.5)
ST.save(fig, OUT, "T1_tests")
print("done")
