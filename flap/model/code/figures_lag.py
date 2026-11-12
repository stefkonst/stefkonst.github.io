"""Lag type figure: misfit of one real spring asymmetry across powered runs against the lag time, for a lag on the
force (mechanical, viscoelastic) and a lag on the voltage before squaring (electrical: RC charging or amplifier).
a, the two cleanest runs (7 and 8 Hz, 1.2 kV); b, with the 1.4 kV run; c, beta per run at each model's best lag.
Run: python figures_lag.py -> ../fig/T2_lag.png/svg"""
import os, sys, json
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); FLAP = os.path.dirname(os.path.dirname(HERE))
sys.path[:0] = [os.path.join(os.path.expanduser("~"), "Documents", "paper_tension", "common"), os.path.join(FLAP, "flapping", "code")]
import style as ST
import matplotlib.pyplot as plt
import lag_type as LT
C = ST.C; OUT = os.path.join(HERE, "..", "fig")
taus = np.arange(0.0, 0.121, 0.0025)


def score(subset, kind, tau):
    bs, es = [], []
    for n in subset:
        X1 = np.mean(LT.B[n][:, 0]); X2 = np.mean(LT.B[n][:, 1]); f = LT.TH[n]["f"]
        b = LT.beta_of(X1, X2, f, tau, kind); bb = np.array([LT.beta_of(x1, x2, f, tau, kind) for x1, x2 in LT.B[n]])
        bs.append(b); es.append(np.std(bb.real) / np.sqrt(len(bb)) + 1)
    bs = np.array(bs); es = np.array(es); m = np.sum(bs.real / es ** 2) / np.sum(1 / es ** 2)
    return np.sum(((bs.real - m) / es) ** 2) + np.sum((bs.imag / es) ** 2), bs, es


S2 = ["full_DEA_core_wing_7hz_1200V_side", "full_DEA_core_wing_8hz_1200V_side"]; S3 = S2 + ["full_DEA_core_wing_8hz_1400V_side"]
ST.LETTER_START = 0
fig, ax = ST.make_fig(3)
for a, sub, ttl in ((ax[0], S2, "7 and 8 Hz, 1.2 kV"), (ax[1], S3, "+ 8 Hz, 1.4 kV")):
    for kind, col in (("mechanical", C[0]), ("electrical", C[1])):
        ch = [score(sub, kind, t)[0] for t in taus]
        a.plot(taus * 1e3, np.array(ch) + 0.1, color=col, label="lag on the force" if kind == "mechanical" else "lag on the voltage (RC)")
    a.set(yscale="log", xlabel="Lag time $\\tau$ (ms)", ylabel="Misfit, one real $\\beta$ for all runs ($\\chi^2$)")
    ST.note(a, ttl, xy=(0.04, 0.06), va="bottom")
ax[0].legend(loc="upper right", fontsize=5)
a = ax[2]
for k, (kind, col) in enumerate((("mechanical", C[0]), ("electrical", C[1]))):
    ch = [score(S2, kind, t)[0] for t in taus]; tb = taus[int(np.argmin(ch))]
    _, bs, es = score(S3, kind, tb)
    x = np.arange(3) + (k - 0.5) * 0.3
    a.errorbar(x, bs.real, yerr=es, fmt="o", color=col, ms=3.5, label=f"{'force' if kind == 'mechanical' else 'voltage'} lag, {tb * 1e3:.0f} ms")
    a.errorbar(x, bs.imag, yerr=es, fmt="s", color=col, mfc="white", ms=3.5)
a.axhline(0, color="k", lw=0.5)
a.set_xticks(range(3)); a.set_xticklabels(["7 Hz", "8 Hz", "8 Hz\n1.4 kV"]); a.minorticks_off()
a.set(ylabel="$\\beta$: real (filled), imaginary (open)")
a.legend(loc="center left", fontsize=5)
ST.save(fig, OUT, "T2_lag")
print("done")
