"""Flapping lift change, re-analysed to reject drift: within-cycle contrast between DEA-on and DEA-off phases, against voltage.

Why: cycle-mean lift against a 0 V window confounds the DEA with time (one ascending voltage sweep, zero drift up to 0.3 N).
The DEA is pulsed within each flap cycle, so a drift common to both phases cancels in the contrast.

Per cycle k of a window at voltage V, phases from the DEA's rising edge (no 0 V window is used, so no waveform alignment is needed):
    C_k = mean L(phase 0.05-0.45) - mean L(phase 0.55-0.95)     the bands leave out the switching spikes
    C_k = c0 + a s(V_k) + noise                                 s = bench force drop against voltage, 1 at 1.6 kV
a is the lift change the DEA causes while on, in mN at 1.6 kV. The on and off phases differ anyway (upstroke, downstroke); that goes in c0.

Placebos:
  1. Phase mask rotated by a quarter cycle: same data, the contrast no longer follows the DEA.
  2. 0 m/s blocks: rig coupling, no aerodynamics.
  3. Scale factor kappa on the model's a over all attached-flow blocks, with its null from rotated masks.
Limits: one sweep per condition, 3 cycles per window, nominal voltage above 1.1 kV, assumes the DEA relaxes within the off phase.
Run: python recalibrate_contrast.py   (results/contrast_blocks.csv, results/contrast_summary.json, figures/contrast.png)
"""
import glob, json, os, sys
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA, RES, FIG = (os.path.join(BASE, k) for k in ("data", "results", "figures"))
sys.path.insert(0, r"C:\Users\stefk\Documents\website\site\flap")
import wing_analytical as W

WOZ = {"WOZ_pre", "WOZ_mid", "WOZ_post"}
GAIN, NB = 200.0, 100
BENCH_V = np.array([0, 200, 400, 600, 800, 1000, 1200, 1400, 1600.0])
BENCH_S = np.array([0, .001, -.001, -.001, -.003, -.014, -.052, -.090, -.132]) / -.132
b2 = os.path.join(BASE, "..", "fig2_dea_actuator", "results", "dea_cyclic_summary.csv")
if os.path.exists(b2):
    t_ = pd.read_csv(b2).groupby("V")["dT_rel_mid_pct"].mean()
    BENCH_V, BENCH_S = t_.index.values.astype(float), (t_ / t_.loc[1600]).values
ph = (np.arange(NB) + 0.5) / NB
ON, OFF = (ph > 0.05) & (ph < 0.45), (ph > 0.55) & (ph < 0.95)


def load(f):
    df = pd.read_csv(f, low_memory=False)
    df = df[df.status.astype(str) != "--- run boundary ---"].copy()
    df["status"] = df.status.astype(str)
    return df.reset_index(drop=True)


def woz_table(df):
    w = df[df.status.isin(WOZ)].sort_values("rowID").copy()
    w["sweep"] = ((w.status != w.status.shift()) | (w.Desired_A0A < w.Desired_A0A.shift())).cumsum()
    return w.groupby(["sweep", "Desired_A0A"], as_index=False)[["rowID", "Fx", "Fz"]].mean()


def edges(t, mon):
    on = mon > 0.4 * max(np.percentile(mon, 98), 1e-9)
    r = np.flatnonzero(np.diff(on.astype(int)) == 1) + 1
    f = np.flatnonzero(np.diff(on.astype(int)) == -1) + 1
    return np.array([t[i] for i in r if len(f[f > i]) and t[f[f > i][0]] - t[i] > 0.15])


def cycles():
    """Phase-binned lift per cycle, for every block and DEA-on window."""
    out = []
    for f in sorted(glob.glob(os.path.join(DATA, "Stef_DEA_[13]Hz_session076_run0??.csv"))):
        if "run005" in f:      # DEA shorted at 0.8 kV
            continue
        run = os.path.basename(f)[:-4].split("_")[-1]
        df = load(f); d = df[df.status == "0"]
        if d.empty:
            continue
        wt = woz_table(df)
        for (U, aoa), blk in d.groupby(["WindSpeed", "Desired_A0A"]):
            g = wt[wt.Desired_A0A == aoa].sort_values("rowID")
            if g.empty:
                continue
            for v, w in blk.groupby("DEA_voltage_cmd"):
                if v <= 0:
                    continue
                fx = w.Fx.values - np.interp(w.rowID, g.rowID, g.Fx)
                fz = w.Fz.values - np.interp(w.rowID, g.rowID, g.Fz)
                a = np.radians(w.AoA.values)
                L = 1e3 * (fz * np.cos(a) - fx * np.sin(a))
                t = w.TimeStamp.values / 1000.0
                fc = w.Freq_cmd.iloc[0]; fm = w.Freq_measured[t > 1].median()
                T = 1 / (fm if 0.5 * fc < fm < 1.5 * fc else fc)
                e = edges(t, w.DEA_voltage_monitor_raw.values)
                for te in e[(e >= 0.8) & (e + T <= t.max())]:
                    m = (t >= te) & (t < te + T)
                    if m.sum() < 20:
                        continue
                    out.append(dict(run=run, f=fc, U=U, aoa=aoa, V=v * GAIN, P=np.interp(ph, (t[m] - te) / T, L[m])))
    return pd.DataFrame(out)


def fit(rows, on=ON, off=OFF, shape=None):
    s = np.interp(rows.V, BENCH_V, BENCH_S) if shape is None else shape(rows.V.values)
    P = np.vstack(rows.P.values)
    C = P[:, on].mean(1) - P[:, off].mean(1)
    X = np.column_stack([np.ones(len(C)), s])
    if len(C) < 5 or np.ptp(s) < 0.3:
        return np.nan, np.nan, len(C)
    b, *_ = np.linalg.lstsq(X, C, rcond=None)
    r = C - X @ b
    cov = (r @ r) / (len(C) - 2) * np.linalg.inv(X.T @ X)
    return b[1], np.sqrt(cov[1, 1]), len(C)


def rotated(k):
    return np.roll(ON, k), np.roll(OFF, k)


# ---- the model's on-phase lift change, same phase bands; upstroke = 180-360 deg in cycle(). Both wings, at 1 Hz
AREA = 84e-4
GEOM = dict(a=(0.06, 0.06), b=(0.07, 0.06), share=(0.55, 0.45), radius=(0.06, 0.12))


def ten(w, i, V):
    Hm, G = w["H_m"], 2 * w["C1"]
    ex, ey = w["lam_x"] - 1, w["lam_y"] - 1
    nx = 2 * G * Hm * (2 * ex + ey) + w["E_x"] * Hm * ex
    ny = 2 * G * Hm * (ex + 2 * ey) + w["E_y"] * Hm * ey + w["r"][i] * max(w["F_D"] - w["dF"] * (V / 1600.0) ** 2, 0.0) / w["s"]
    return nx, ny, Hm, G + 0.5 * (w["E_x"] + w["E_y"]) / 6


def model_a(U, aoa, r_in, r_out, dF=8.6e-3, lam=1.01):
    w = dict(W.DISCRETE, lam_x=lam, lam_y=lam, dF=dF, s=0.010, r=(r_in, r_out), **GEOM)
    fl = dict(U=float(U), alpha_deg=float(aoa), area=AREA, Phi0_deg=30.0, f=1.0)
    p = W.cycle(w, ten, **fl)[0]
    d = W.delta_cycle(w, ten, **fl)
    return 2 * float(d[(p > 198) & (p < 342)].mean())   # both wings carried DEAs and the load cell measures both


R = cycles()
rows, keys = [], []
for key, g in R.groupby(["run", "f", "U", "aoa"]):
    a, se, n = fit(g)
    a2, se2, _ = fit(g, shape=lambda v: (v / 1600.0) ** 2)
    ap, sep, _ = fit(g, *rotated(25))
    P = np.vstack(g[g.V <= 800].P.values) if (g.V <= 800).any() else np.vstack(g.P.values)
    rows.append(dict(run=key[0], f=key[1], U=key[2], aoa=key[3], n=n, a=a, se=se, a_V2=a2, se_V2=se2, placebo=ap, placebo_se=sep, L_low=P.mean()))
    keys.append(key)
B = pd.DataFrame(rows).sort_values(["f", "U", "aoa", "run"])
B["t"] = B.a / B.se
ctrl = B[(B.U == 0) & (B.f == 1)]
wind = B[(B.U > 0) & (B.f == 1)]
att = wind[wind.aoa.isin([-5, 0, 10])].copy()
att["m_whole"] = [model_a(u, a_, 0.81, 0.81) for u, a_ in zip(att.U, att.aoa)]
att["m_in"] = [model_a(u, a_, 0.81, 0.02) for u, a_ in zip(att.U, att.aoa)]
att["m_low"] = [model_a(u, a_, 0.81, 0.81, dF=3.2e-3) for u, a_ in zip(att.U, att.aoa)]
att["m_free"] = [model_a(u, a_, 0.02, 0.02) for u, a_ in zip(att.U, att.aoa)]
c_mean = np.average(ctrl.a, weights=1 / ctrl.se ** 2)


def kappa(tab, col, a="a", se="se"):
    w = 1 / tab[se] ** 2
    k = (w * tab[col] * tab[a]).sum() / (w * tab[col] ** 2).sum()
    return k, 1 / np.sqrt((w * tab[col] ** 2).sum())


S = dict(c_control=c_mean, c_control_se=1 / np.sqrt((1 / ctrl.se ** 2).sum()))
for col in ("m_whole", "m_in", "m_low"):
    k, ks = kappa(att, col)
    null = []
    for lag in range(10, 91, 2):
        if 40 <= lag <= 60:      # keeps away from the sign-flipped real contrast
            continue
        tt = []
        for _, r in att.iterrows():
            g = R[(R.run == r.run) & (R.U == r.U) & (R.aoa == r.aoa) & (R.f == r.f)]
            aa, ss, _ = fit(g, *rotated(lag))
            tt.append((aa, ss))
        tab = att.assign(aa=[x[0] for x in tt], ss=[x[1] for x in tt])
        null.append(kappa(tab, col, "aa", "ss")[0])
    S[col] = dict(kappa=k, se=ks, null_sd=float(np.std(null)), null_max=float(np.max(np.abs(null))), n_null=len(null))
pool = (att.a / att.se ** 2).sum() / (1 / att.se ** 2).sum()
S["unsigned_mean_a"] = pool
S["chi2_a"] = float(((att.a / att.se) ** 2).sum())
S["n_att"] = len(att)
S["chi2_placebo"] = float(((att.placebo / att.placebo_se) ** 2).sum())
S["median_se_wind"] = float(att.se.median())
B.to_csv(os.path.join(RES, "contrast_blocks.csv"), index=False)
att.to_csv(os.path.join(RES, "contrast_attached.csv"), index=False)
json.dump(S, open(os.path.join(RES, "contrast_summary.json"), "w"), indent=1, default=float)
pd.set_option("display.width", 200)
print(B.round(1).to_string(index=False))
print(att[["U", "aoa", "a", "se", "placebo", "L_low", "m_free", "m_in", "m_low", "m_whole"]].round(1).to_string(index=False))
print(json.dumps(S, indent=0, default=float))

fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
cols = {2.0: "#77AC30", 4.0: "#7E2F8E", 6.0: "#0072BD"}
x = np.arange(len(att))
lab = [f"{u:g} m/s\n{a_:+d}°" for u, a_ in zip(att.U, att.aoa)]
ax[0].axhline(0, color="k", lw=0.8)
ax[0].errorbar(x - 0.12, att.a, yerr=2 * att.se, fmt="o", color="k", capsize=3, label="Within-cycle contrast, ± 2 s.e.")
ax[0].errorbar(x + 0.12, att.placebo, yerr=2 * att.placebo_se, fmt="s", color="#AAAAAA", capsize=3, label="Placebo: phases rotated by a quarter cycle")
ax[0].plot(x, att.m_whole, "_", color="#D95319", ms=22, mew=2.5, label="Model, both wings: line along the whole edge")
ax[0].plot(x, att.m_in, "_", color="#0072BD", ms=22, mew=2.5, label="Model, both wings: line inboard only")
ax[0].set_xticks(x); ax[0].set_xticklabels(lab, fontsize=9)
ax[0].set_ylabel("Lift change while the DEA is on, mN at 1.6 kV")
ax[0].legend(fontsize=8.5); ax[0].grid(alpha=.3)
ax[0].set_title("a) Each condition", loc="left")
names = [("m_whole", "Line, whole edge"), ("m_in", "Line, inboard only"), ("m_low", "Line, low DEA force")]
for i, (c, n_) in enumerate(names):
    d = S[c]
    ax[1].errorbar([d["kappa"]], [i], xerr=[2 * max(d["se"], d["null_sd"])], fmt="o", color="k", capsize=4)
ax[1].axvline(0, color="#777777"); ax[1].axvline(1, color="#D95319", ls="--")
ax[1].set_yticks(range(3)); ax[1].set_yticklabels([n[1] for n in names])
ax[1].set_ylim(-0.6, 2.6)
ax[1].set_xlabel(r"Scale factor $\kappa$ on the model (0: no effect, 1: model right), ± 2 s.e.")
ax[1].grid(alpha=.3)
ax[1].set_title("b) All conditions", loc="left")
fig.tight_layout()
fig.savefig(os.path.join(FIG, "contrast.png"), dpi=200)
fig.savefig(os.path.join(FIG, "contrast.svg"))
