"""Does the sign flip between -5 and +10 deg rule out drift? Session 076, 1 Hz, DEA runs, 1.6 kV window minus 0 V window.
L = Fz cos(a) - Fx sin(a). A drift in Fx enters through sin(a), so it changes sign with pitch, with no aerodynamics.
1  Split the change into the Fz part and the Fx part, each raw and tared.
2  Tare-free test: differences between pitches in the same run cancel a zero shift common to the run.
3  Does the change follow the baseline lift, or the recording order, or sin(a)? Fit dL = b L0 and dL = e sin(a) + c.
4  Pitch order and time of each block, to see whether recording order and pitch can be separated.
Run: python sign_flip_check.py   (results/sign_flip_check.csv)"""
import glob, os
import numpy as np, pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA, RES = os.path.join(BASE, "data"), os.path.join(BASE, "results")
WOZ = {"WOZ_pre", "WOZ_mid", "WOZ_post"}


def load(f):
    df = pd.read_csv(f, low_memory=False)
    df = df[df.status.astype(str) != "--- run boundary ---"].copy()
    df["status"] = df.status.astype(str)
    return df.reset_index(drop=True)


def woz_table(df):
    w = df[df.status.isin(WOZ)].sort_values("rowID").copy()
    w["sweep"] = ((w.status != w.status.shift()) | (w.Desired_A0A < w.Desired_A0A.shift())).cumsum()
    return w.groupby(["sweep", "Desired_A0A"], as_index=False)[["rowID", "Fx", "Fz"]].mean()


out = []
for f in sorted(glob.glob(os.path.join(DATA, "Stef_DEA_[13]Hz_session076_run0??.csv"))):
    if "run005" in f:
        continue
    run = os.path.basename(f)[:-4].split("_")[-1]
    hz = os.path.basename(f).split("_")[2]
    df = load(f); d = df[df.status == "0"]
    wt = woz_table(df)
    if d.empty:
        continue
    for (U, aoa), blk in d.groupby(["WindSpeed", "Desired_A0A"]):
        g = wt[wt.Desired_A0A == aoa].sort_values("rowID")
        if g.empty or blk.DEA_voltage_cmd.nunique() < 4:
            continue
        vs = sorted(blk.DEA_voltage_cmd.unique())
        w0, w1 = blk[blk.DEA_voltage_cmd == vs[0]], blk[blk.DEA_voltage_cmd == vs[-1]]
        res = {}
        for tag, w in (("0", w0), ("1", w1)):
            t = w.TimeStamp.values / 1000.0
            m = t >= 0.8
            a = np.radians(w.AoA.values[m]).mean()
            res[tag] = dict(Fz=1e3 * w.Fz.values[m].mean(), Fx=1e3 * w.Fx.values[m].mean(), tz=1e3 * np.interp(w.rowID.values[m], g.rowID, g.Fz).mean(), tx=1e3 * np.interp(w.rowID.values[m], g.rowID, g.Fx).mean(), a=a, row=w.rowID.values[m].mean())
        a = res["1"]["a"]
        dFz, dFx = res["1"]["Fz"] - res["0"]["Fz"], res["1"]["Fx"] - res["0"]["Fx"]
        dtz, dtx = res["1"]["tz"] - res["0"]["tz"], res["1"]["tx"] - res["0"]["tx"]
        L0 = (res["0"]["Fz"] - res["0"]["tz"]) * np.cos(a) - (res["0"]["Fx"] - res["0"]["tx"]) * np.sin(a)
        out.append(dict(run=run, hz=hz, U=U, aoa=aoa, start_s=blk.rowID.min() * 0.002, L0=L0,
                        raw_z=dFz * np.cos(a), raw_x=-dFx * np.sin(a), tare_z=-dtz * np.cos(a), tare_x=dtx * np.sin(a),
                        z=(dFz - dtz) * np.cos(a), x=-(dFx - dtx) * np.sin(a), dFx_raw=dFx, dFz_raw=dFz, dtx=dtx, dtz=dtz))
T = pd.DataFrame(out).sort_values(["run", "start_s"])
T["dL"] = T.z + T.x
T["raw"] = T.raw_z + T.raw_x
T.to_csv(os.path.join(RES, "sign_flip_check.csv"), index=False)
pd.set_option("display.width", 220)
print(T[["run", "U", "aoa", "start_s", "L0", "raw", "raw_z", "raw_x", "tare_z", "tare_x", "z", "x", "dL", "dFx_raw", "dFz_raw"]].round(1).to_string(index=False))

w = T[(T.U > 0) & (T.hz == "1Hz") & (T.aoa.isin([-5, 0, 10]))]
print("\nSame run, +10 minus -5 deg, change in lift at 1.6 kV: tared, raw, and from Fz only and Fx only (tared)")
for U, g in w.groupby("U"):
    p, q = g[g.aoa == 10].iloc[0], g[g.aoa == -5].iloc[0]
    print(f"U {U:g}: tared {p.dL - q.dL:6.1f}, raw {p.raw - q.raw:6.1f}, Fz part {p.z - q.z:6.1f}, Fx part {p.x - q.x:6.1f}; needs dFx difference {-(p.x - q.x):5.1f} over sin diff 0.26: {(p.x - q.x):5.1f}")
print("\nFit over attached wind blocks, n =", len(w))
A = np.column_stack([w.L0]); b = np.linalg.lstsq(A, w.dL, rcond=None)[0][0]
print(f"dL = b L0: b = {b:.2f}, R2 = {1 - ((w.dL - b * w.L0) ** 2).sum() / ((w.dL - w.dL.mean()) ** 2).sum():.2f}")
order = w.groupby("U").start_s.rank().values
for lab, x in (("baseline L0", w.L0.values), ("pitch", w.aoa.values.astype(float)), ("recording order", order), ("sin(a)", np.sin(np.radians(w.aoa.values)))):
    print(f"corr(dL, {lab}) = {np.corrcoef(w.dL, x)[0, 1]:.2f}; corr(raw, {lab}) = {np.corrcoef(w.raw, x)[0, 1]:.2f}")
print("corr(pitch, recording order) =", round(np.corrcoef(w.aoa, order)[0, 1], 2))
c = T[(T.U == 0)]
print("\nNo wind, flapping only:")
print(c[["run", "aoa", "L0", "raw", "dL"]].round(1).to_string(index=False))
