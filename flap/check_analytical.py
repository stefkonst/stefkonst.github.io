"""Checks wing_analytical.py against closed forms, the FE results on disk and the section model. Run: python check_analytical.py [stretch_gain]"""
import json, math, os, sys
import numpy as np
import wing_analytical as W
V = os.path.join(os.path.dirname(os.path.abspath(__file__)), "full_membrane", "validation")
GAIN = float(sys.argv[1]) if len(sys.argv) > 1 else W.COMMON["stretch_gain"]
W.COMMON["stretch_gain"] = GAIN
print("stretch gain", GAIN)
print("\n-- closed forms --")
print("CAMBER", round(W.CAMBER, 4))
w = dict(W.DISCRETE, r=0.0, dF=0.0, F_D=0.0)
c = dict(W.COMMON, U=0.5)                                             # small load: the cubic term vanishes, so h = load / (T - D)
q = 0.5 * 1.225 * 0.25
h, CL, inf = W.panel(w, 0, 0.0, q, math.radians(6), c, W.discrete_tension)
print("small-load h:", h, q * 4 / math.pi ** 2 * 3.71 * math.radians(6) / (inf["T"] - inf["D"]))
print("\n-- full membrane against the FE: 4 m/s, 6 deg, 1.6 kV --")
R = [r for r in json.load(open(os.path.join(V, "opt_layers.json"))) if r.get("U") == 4.0 and r.get("ok0") and r.get("ok1")]
A = []
for r in R:
    n, l, d = r["na"][0], r["lam_d"][0], r["dn"][0]
    w = dict(W.FULL, n_layers=(n, n), lam_d=(l, l), lam_n=(l ** -0.5 + d, l ** -0.5 + d))
    a, b = W.solve(w, W.full_tension, V=0.0), W.solve(w, W.full_tension)
    A.append((n, l, d, r["CL0"], a["CL"], r["CL1"], b["CL"], r["dL_mN"], b["L_mN"] - a["L_mN"]))
A = np.array(A)
print(f"{len(A)} converged FE cases. C_L 0 V: rms error {np.sqrt(np.mean((A[:, 4] - A[:, 3]) ** 2)):.3f}; C_L on: {np.sqrt(np.mean((A[:, 6] - A[:, 5]) ** 2)):.3f}")
print(f"dL: rms error {np.sqrt(np.mean((A[:, 8] - A[:, 7]) ** 2)):.2f} mN, correlation {np.corrcoef(A[:, 7], A[:, 8])[0, 1]:.3f}")
print(" n lam_d dn  | CL0 FE model | CL1 FE model | dL FE model")
for r in A:
    if r[2] == 0.04 and r[1] in (1.03, 1.05):
        print(f" {r[0]:.0f} {r[1]:.2f} {r[2]:.2f} | {r[3]:.2f} {r[4]:.2f} | {r[5]:.2f} {r[6]:.2f} | {r[7]:5.1f} {r[8]:5.1f}")
print("\n-- speed --")
for U, (c0, c1) in {3.0: (0.505, 0.765), 4.0: (0.603, 0.864), 6.0: (0.724, 0.906)}.items():
    a, b = W.solve(W.FULL, W.full_tension, V=0.0, U=U), W.solve(W.FULL, W.full_tension, U=U)
    print(f"full U {U:g}: CL FE {c0:.2f} to {c1:.2f}; model {a['CL']:.2f} to {b['CL']:.2f}")
ph, a = W.cycle(W.FULL, W.full_tension, V=0.0)
ph, b = W.cycle(W.FULL, W.full_tension)
fe = json.load(open(os.path.join(V, "flap_compare_full_4_2.json")))["cases"]["full_4_2"]
print("\n-- cycle, full, 4 m/s 2 Hz --")
print("phase  ", ph.astype(int).tolist())
print("L0 FE  ", np.round(fe["L0"], 1).tolist())
print("L0 mod ", np.round(a, 1).tolist())
print("L1 FE  ", np.round(fe["L1"], 1).tolist())
print("L1 mod ", np.round(b, 1).tolist())
print("dL mean FE %.1f, model %.1f" % (np.mean(np.array(fe["L1"]) - fe["L0"]), np.mean(b - a)))
s = json.load(open(os.path.join(V, "flap_compare_section.json")))["cases"]
print("\n-- discrete, against the section model --")
for k, U in (("old_3_2", 3.0), ("old_4_2", 4.0), ("old_6_2", 6.0)):
    a, b = W.cycle(W.DISCRETE, W.discrete_tension, V=0.0, U=U)[1], W.cycle(W.DISCRETE, W.discrete_tension, U=U)[1]
    print(f"U {U:g}: L0 section {np.mean(s[k]['L0']):.1f}, model {a.mean():.1f}; dL section {np.mean(np.array(s[k]['L1']) - s[k]['L0']):.2f}, model {np.mean(b - a):.2f}")
