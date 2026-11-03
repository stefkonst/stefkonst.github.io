"""Checks wing_analytical.py against closed forms, the finite elements and the section model. Run: python check_analytical.py [gamma]"""
import json, math, os, sys
import numpy as np
import wing_analytical as W
V = os.path.join(os.path.dirname(os.path.abspath(__file__)), "full_membrane", "validation")
if len(sys.argv) > 1:
    W.COMMON["gamma"] = float(sys.argv[1])
print("gamma", W.COMMON["gamma"], " k", round(W.K_CAM, 3), " c", round(W.CAMBER, 3))

print("\n-- closed forms --")
Y, Phi = math.pi ** 2 / 4, 4 / math.pi ** 2
print("2D divergence limit, a_L = 2 pi:", round(Phi * 2 * math.pi * W.K_CAM * W.CAMBER / Y, 4), " Eq. S5: 1.727")
y = 0.5 * (1 - np.cos(W._th))
print("thin-airfoil angle of a half sine, h / c = 1:", round(-np.trapezoid(math.pi * np.cos(math.pi * y) * (1 - np.cos(W._th)), W._th) / math.pi, 4), " a parabola gives 2")
w = dict(W.DISCRETE, T_c=0.0, dF=0.0, F_D=0.0)
c, q = dict(W.COMMON, U=0.5), 0.5 * 1.225 * 0.25
A, CL, inf = W.panel(w, 0, 0.0, q, math.radians(6), c, W.discrete_tension)
print("small-load amplitude:", A, " load / (T - D):", q * Phi * 3.71 * math.radians(6) / (inf["T"] - inf["D"]))
print("free trailing edge, lift change at 4 m/s:", W.mean_delta(dict(W.DISCRETE, T_c=0.0), W.discrete_tension), " Eq. S13: 0")

print("\n-- full membrane against the finite elements: 4 m/s, 6 deg, 1.6 kV, passive fibre, eta 0.4 --")
R = [r for r in json.load(open(os.path.join(V, "opt_layers.json"))) if r.get("U") == 4.0 and r.get("ok0") and r.get("ok1")]
Eta = dict(W.FULL, eta=0.4)
A = []
for r in R:
    n, l, d = r["na"][0], r["lam_d"][0], r["dn"][0]
    w = dict(Eta, n_layers=(n, n), lam_d=(l, l), lam_n=(l ** -0.5 + d, l ** -0.5 + d))
    a, b = W.solve(w, W.full_tension, V=0.0), W.solve(w, W.full_tension)
    A.append((n, l, d, r["CL0"], a["CL"], r["CL1"], b["CL"], r["dL_mN"], b["L_mN"] - a["L_mN"]))
A = np.array(A)
print(f"{len(A)} converged cases. C_L 0 V: rms error {np.sqrt(np.mean((A[:, 4] - A[:, 3]) ** 2)):.3f}; C_L on: {np.sqrt(np.mean((A[:, 6] - A[:, 5]) ** 2)):.3f}")
print(f"dL: rms error {np.sqrt(np.mean((A[:, 8] - A[:, 7]) ** 2)):.2f} mN, correlation {np.corrcoef(A[:, 7], A[:, 8])[0, 1]:.3f}")
print(" n lam_d dn  | CL0 FE model | CL1 FE model | dL FE model")
for r in A:
    if r[2] == 0.04 and r[1] in (1.03, 1.05):
        print(f" {r[0]:.0f} {r[1]:.2f} {r[2]:.2f} | {r[3]:.2f} {r[4]:.2f} | {r[5]:.2f} {r[6]:.2f} | {r[7]:5.1f} {r[8]:5.1f}")

one = dict(Eta, n_layers=(1, 1), lam_d=(1.05, 1.05), lam_n=(1.016, 1.016))
print("\n-- speed, one layer --")
for U, (c0, c1) in {3.0: (0.505, 0.765), 4.0: (0.603, 0.864), 6.0: (0.724, 0.906)}.items():
    print(f"U {U:g}: C_L FE {c0:.2f} to {c1:.2f}; model {W.solve(one, W.full_tension, V=0.0, U=U)['CL']:.2f} to {W.solve(one, W.full_tension, U=U)['CL']:.2f}")
fe = json.load(open(os.path.join(V, "flap_compare_full_4_2.json")))["cases"]["full_4_2"]
print("\n-- cycle, one layer, 4 m/s, 2 Hz --")
ph = np.arange(0.0, 360.0, 30.0)
keep = (ph < 240) | (ph > 300)
d = np.array(fe["L1"]) - fe["L0"]
print("dL mean FE %.1f all phases, %.1f leaving out 240-300 deg; model %.1f leaving out 240-300 deg" % (d.mean(), d[keep].mean(), W.mean_delta(one, W.full_tension)))

s = json.load(open(os.path.join(V, "flap_compare_section.json")))["cases"]
print("\n-- discrete, rigid trailing edge, against the section model --")
rigid = dict(W.DISCRETE, T_c=1e9)
for k, U in (("old_3_2", 3.0), ("old_4_2", 4.0), ("old_6_2", 6.0)):
    print(f"U {U:g}: dL section {np.mean(np.array(s[k]['L1']) - s[k]['L0']):.2f} mN, model {W.mean_delta(rigid, W.discrete_tension, U=U):.2f} mN")

print("\n-- tilted wing, against the finite elements --")
FE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "full_membrane", "validation")


def rotated(r):
    w = dict(W.ROTATED, theta_b=r["theta"], theta_r=r["theta_r"], n_layers=(r["N"], r["N"]), lam_n=r.get("lam_n", 1.0), eta=r.get("eta", 0.4))
    u = r.get("U", 4.0)
    return W.solve(w, W.rotation_tension, U=u)["L_mN"] - W.solve(w, W.rotation_tension, V=0.0, U=u)["L_mN"]


def compare(name, rows):
    fe, mo = np.array([r["dL_mN"] for r in rows]), np.array([rotated(r) for r in rows])
    print(f"{name}: {len(rows)} cases, correlation {np.corrcoef(fe, mo)[0, 1]:.2f}, rms {np.sqrt(np.mean((fe - mo) ** 2)):.2f} mN")


use = lambda r: "error" not in r and not r.get("active") and r.get("ok0") and r.get("ok1") and r["xb"] == 0.06 and r.get("h", 0.002) == 0.002
compare("tuned on", [r for f in ("tilt_fine.json", "tilt_balanced.json") for r in json.load(open(os.path.join(FE_DIR, f))) if use(r) and r.get("U", 4.0) == 4.0])
compare("not tuned on", [r for r in json.load(open(os.path.join(FE_DIR, "tilt_final.json"))) if use(r)])
print("fibre stretch at 4.5 and 7.5 deg:", [round(x, 3) for x in W.rotation_geometry(W.ROTATED)[:2]], "against 1.047, 1.045 from the finite elements, with the chord stretch")
