"""Is the cycle mean or the on-minus-off contrast right? Tests on the raw session 076 data, 1 Hz, DEA runs (not run005).
Per flap cycle k, in order of time t (rowID x 2 ms), including the 0 V window:
  A  L = c + a V^2                    the old cycle-mean comparison, as a fit
  B  L = c + d t + a V^2              the same with a time trend. Time and voltage are collinear between windows, not within them
  C  L = window means + d t           time trend only inside constant-voltage windows: the rate with no voltage change at all
Also: dL at 1.6 kV with and without the wind-off tare, on and off phase changes, and why the contrast is negative.
Run: python cycle_mean_check.py   (results/cycle_mean_check.csv)"""
import glob, os
import numpy as np, pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA, RES = os.path.join(BASE, "data"), os.path.join(BASE, "results")
WOZ = {"WOZ_pre", "WOZ_mid", "WOZ_post"}
GAIN = 200.0


def load(f):
    df = pd.read_csv(f, low_memory=False)
    df = df[df.status.astype(str) != "--- run boundary ---"].copy()
    df["status"] = df.status.astype(str)
    return df.reset_index(drop=True)


def woz_table(df):
    w = df[df.status.isin(WOZ)].sort_values("rowID").copy()
    w["sweep"] = ((w.status != w.status.shift()) | (w.Desired_A0A < w.Desired_A0A.shift())).cumsum()
    return w.groupby(["sweep", "Desired_A0A"], as_index=False)[["rowID", "Fx", "Fz"]].mean()


rows = []
for f in sorted(glob.glob(os.path.join(DATA, "Stef_DEA_1Hz_session076_run0??.csv"))):
    if "run005" in f:
        continue
    run = os.path.basename(f)[:-4].split("_")[-1]
    df = load(f); d = df[df.status == "0"]
    wt = woz_table(df)
    for (U, aoa), blk in d.groupby(["WindSpeed", "Desired_A0A"]):
        g = wt[wt.Desired_A0A == aoa].sort_values("rowID")
        if g.empty:
            continue
        for v, w in blk.groupby("DEA_voltage_cmd"):
            tx, tz = np.interp(w.rowID, g.rowID, g.Fx), np.interp(w.rowID, g.rowID, g.Fz)
            a = np.radians(w.AoA.values)
            L = 1e3 * ((w.Fz.values - tz) * np.cos(a) - (w.Fx.values - tx) * np.sin(a))
            R = 1e3 * (w.Fz.values * np.cos(a) - w.Fx.values * np.sin(a))
            t = w.TimeStamp.values / 1000.0
            fm = w.Freq_measured[t > 1].median()
            T = 1 / (fm if 0.5 < fm < 1.5 else 1.0)
            n = int((t.max() - 0.8) / T)
            for k in range(n):
                m = (t >= 0.8 + k * T) & (t < 0.8 + (k + 1) * T)
                rows.append(dict(run=run, U=U, aoa=aoa, V=v * GAIN, tabs=w.rowID.values[m].mean() * 0.002, L=L[m].mean(), raw=R[m].mean()))
C = pd.DataFrame(rows)


def ols(X, y):
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    r = y - X @ b
    cov = (r @ r) / max(len(y) - X.shape[1], 1) * np.linalg.pinv(X.T @ X)
    return b, np.sqrt(np.diag(cov))


out = []
for (run, U, aoa), g in C.groupby(["run", "U", "aoa"]):
    if g.V.nunique() < 4:
        continue
    g = g.sort_values("tabs")
    t = g.tabs.values - g.tabs.values.mean()
    s = (g.V.values / 1600.0) ** 2
    y = g.L.values
    bA, sA = ols(np.column_stack([np.ones(len(y)), s]), y)
    bB, sB = ols(np.column_stack([np.ones(len(y)), t, s]), y)
    W = pd.get_dummies(g.V).values.astype(float)
    bC, sC = ols(np.column_stack([W, t]), y)
    m = g.groupby("V")
    span = g.tabs.max() - g.tabs.min()
    L0 = m.L.mean().iloc[0]
    dL = m.L.mean().iloc[-1] - L0
    dR = m.raw.mean().iloc[-1] - m.raw.mean().iloc[0]
    corr = np.corrcoef(t, s)[0, 1]
    # direction of the lift: + means the change increases |L|
    out.append(dict(run=run, U=U, aoa=aoa, L0=L0, dL=dL, dL_noTare=dR, a_A=bA[1], a_A_se=sA[1], a_B=bB[2], a_B_se=sB[2], d_all=bB[1], d_inside=bC[-1], d_inside_se=sC[-1], span=span, rate_needed=dL / span, r_tV=corr, sign=np.sign(L0)))
O = pd.DataFrame(out).sort_values(["U", "aoa", "run"])
O.to_csv(os.path.join(RES, "cycle_mean_check.csv"), index=False)
pd.set_option("display.width", 220)
print(O.drop(columns=["sign", "run"]).round(2).to_string(index=False))


def pool(tab, v, se, signed=True):
    x = tab[v] * (tab.sign if signed else 1)
    w = 1 / tab[se] ** 2
    return (w * x).sum() / w.sum(), 1 / np.sqrt(w.sum())


wind = O[(O.U > 0) & (O.aoa.isin([-5, 0, 10]))]
big = wind[wind.L0.abs() >= 20]
ctrl = O[O.U == 0]
print("\nPooled, signed so + means |lift| grows. Blocks with |L0| >= 20 mN:", len(big), " no-wind blocks:", len(ctrl))
for lab, v, se in (("A cycle mean, no time term, mN", "a_A", "a_A_se"), ("B cycle mean with time term, mN", "a_B", "a_B_se"), ("C trend inside constant-voltage windows, mN/s", "d_inside", "d_inside_se")):
    print(f"{lab:48s} wind {pool(big, v, se)[0]:7.2f} +- {pool(big, v, se)[1]:5.2f}   no wind (unsigned) {pool(ctrl, v, se, False)[0]:7.2f} +- {pool(ctrl, v, se, False)[1]:5.2f}")
print("rate needed to explain the change by time alone, mean |dL|/span, mN/s:", round((big.dL * big.sign / big.span).mean(), 2), " mean span s:", round(big.span.mean(), 1), " mean r(t, V^2):", round(O.r_tV.mean(), 2))
print("\nThe contrast, 4 m/s, 10 deg: lift change at 1.6 kV by phase (step5_phase.csv)")
P = pd.read_csv(os.path.join(RES, "step5_phase.csv"), encoding="utf-8")
for _, r in P[(P.V == 1600) & P.block.str.contains("10")].iterrows():
    print(" ", r.block.encode("ascii", "ignore").decode(), "|", r.phase, round(r.change_mN, 1), "+-", round(r.sem_mN, 1))
