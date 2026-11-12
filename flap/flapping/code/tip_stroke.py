"""Stroke from the tip's vertical excursion and the spar length, independent of any hinge fit.
The circle fit through the tip's path (tip_track.py) is poorly constrained on a +-12 deg arc: its radius came out
386 px in the powered view, where hinge to tip measures about 358 px, and 384 px in the unpowered view, where it
measures about 417 px; the two angle scales were off in opposite directions. Here:
    theta(t) = asin( (ybar - y_tip(t)) * s / L ),   s from the ruler (sub-pixel FFT of the tick spacing),
    L = spar length, hinge to tip, the same wing in both views: 55.2 mm (mean of 54.4 and 56.1 mm measured in the
    two side views; 53-55 mm in the top view). The 3 % difference between the two views bounds the scale error.
Run: python tip_stroke.py -> ../data/tip_stroke.json"""
import os, json, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); DATA = os.path.join(HERE, "..", "data"); FS = 200.0
SCALE = {"only": 0.1345}; S_DEF = 0.1518; L_MM = 55.2; L_ERR = 0.03
RUNS = {"full_DEA_core_wing_8hz_1200V_side": 8, "only_DEA_core_8hz_1200V_side": 8, "full_DEA_core_wing_8hz_1400V_side": 8,
        "full_DEA_core_wing_7hz_1200V_side": 7, "full_DEA_core_wing_9hz_1200V_side": 9, "full_DEA_core_wing_10hz_1200V_side": 10, "full_DEA_core_wing_5hz_1200v_side": 5}
out = {}
for n, f in RUNS.items():
    tip = np.load(os.path.join(DATA, "tracks", "tip_" + n + ".npz"))["tip"]; s = SCALE["only"] if n.startswith("only") else S_DEF
    t = np.arange(len(tip)) / FS; g = (t > 0.9) & np.isfinite(tip[:, 1])
    if "1400" in n: g &= t < 2.15
    y = tip[:, 1]; yb = np.nanmean(y[g])
    th = np.degrees(np.arcsin(np.clip((yb - y) * s / L_MM, -1, 1)))
    N = int(round(FS / f)); idx = np.where(g)[0]
    pp = np.array([np.nanmax(th[i:i + N]) - np.nanmin(th[i:i + N]) for i in range(idx[0], idx[-1] - N, N)])
    mm = np.array([(np.nanmax(y[i:i + N]) - np.nanmin(y[i:i + N])) * s for i in range(idx[0], idx[-1] - N, N)])
    out[n] = dict(f=f, scale=s, pp_deg=float(pp.mean()), pp_sd=float(pp.std()), pp_se=float(pp.std() / np.sqrt(len(pp))), n=len(pp),
                  tip_mm=float(mm.mean()), tip_mm_sd=float(mm.std()), theta=th.tolist())
    print("%-38s stroke %.2f ± %.2f deg (n=%d)  tip %.2f ± %.2f mm" % (n, pp.mean(), pp.std(), len(pp), mm.mean(), mm.std()))
p, u = out["full_DEA_core_wing_8hz_1200V_side"], out["only_DEA_core_8hz_1200V_side"]
r = p["pp_deg"] / u["pp_deg"] - 1; se = np.hypot(p["pp_se"] / p["pp_deg"], u["pp_se"] / u["pp_deg"])
out["comparison"] = dict(powered_over_unpowered=float(r), stat_se=float(se), systematic=L_ERR)
print("powered / unpowered - 1 = %.1f %% ± %.1f %% (statistical) ± %.0f %% (scale)" % (100 * r, 100 * se, 100 * L_ERR))
json.dump(out, open(os.path.join(DATA, "tip_stroke.json"), "w"))
