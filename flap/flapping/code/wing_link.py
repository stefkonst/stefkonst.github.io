"""Link the robot's wing mode (two_mode.py, 17 Hz at 0 V) to the wing model of the bench (robot/edge_theory.py).

Edge mode with the skin behind it (Fig. 3c):  f_e = (1/2pi) sqrt( (T (pi/L)^2 + N/c) / (mu_e + rho t c/3) )
1. The tension T_robot that puts the mode at the fitted f_w (with its uncertainty).
2. The force the rolled edge loses at the robot's voltage, dF = eta_f n_f eps (V/H)^2 H w_a, for the bench fits of
   eta_f (0.06-0.17 from the first wing, 0.4-0.8 from the finite elements); mean over a sine-squared cycle = 3/8 of peak.
3. Whether that is enough to slacken the edge (T -> 0), which takes the mode out of the band.
Run: python wing_link.py -> ../data/wing_link.json"""
import os, sys, json, math, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); DATA = os.path.join(HERE, "..", "data")
sys.path.insert(0, os.path.join(HERE, "..", "..", "robot"))
import params as P
import edge_theory as ET
CAL = json.load(open(os.path.join(HERE, "..", "..", "robot", "data", "edge_calib.json")))
TM = json.load(open(os.path.join(DATA, "two_mode.json")))
mu_roll = P.RHO_F * P.ROLL_W * P.FILM_T / 1.2
N0 = CAL["N0"]; c = P.CHORD_ROOT * 0.66
T = np.linspace(0, 0.12, 241)
fe = np.array([ET.edge_mode(x, N0, mu_roll, c=c) for x in T])
def T_for(f):
    return float(np.interp(f, fe, T)) if fe[0] < f < fe[-1] else (0.0 if f <= fe[0] else float("nan"))
fw, sw = TM["f_w"], TM["f_w_se"]
T_rob = (T_for(fw - sw), T_for(fw), T_for(fw + sw))
f_bench = [float(ET.edge_mode(x, N0, mu_roll, c=c)) for x in (0.050, 0.096)]
def dF(V, eta_f, n_f=2, H=P.H, wa=P.ROLL_WA):
    return eta_f * n_f * P.EPS * (V / H) ** 2 * H * wa
rows = {}
for V in (1200.0, 1400.0):
    rows[V] = {e: dict(peak_mN=dF(V, e) * 1e3, mean_mN=dF(V, e) * 3 / 8 * 1e3) for e in (0.06, 0.17, 0.4, 0.8)}
out = dict(T=T.tolist(), f_e=fe.tolist(), f_w=fw, f_w_se=sw, T_robot_mN=[x * 1e3 for x in T_rob], f_skin_only=float(fe[0]),
           f_bench_50_96mN=f_bench, dF=rows)
json.dump(out, open(os.path.join(DATA, "wing_link.json"), "w"), indent=1)
print("edge mode with skin only (T = 0): %.1f Hz; at the bench tension 50-96 mN: %.0f-%.0f Hz" % (fe[0], *f_bench))
print("robot wing mode %.1f ± %.1f Hz -> edge tension %.1f (%.1f-%.1f) mN" % (fw, sw, T_rob[1] * 1e3, T_rob[0] * 1e3, T_rob[2] * 1e3))
for V, r in rows.items():
    print("%d V: roll force lost, peak / cycle mean (mN): " % V + ", ".join("eta_f %.2f: %.0f / %.0f" % (e, d["peak_mN"], d["mean_mN"]) for e, d in r.items()))
