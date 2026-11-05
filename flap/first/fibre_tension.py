"""Roll tension from its length: cut 52.94 mm (core strip in YEAR2/flittermouse/DEAcore_full_outline.svg), measured bead to tip in the front view, Yeoh fit of P7670 (C1 42.8 kPa).
Run: python fibre_tension.py <track folder> -> data/fibre_tension.json"""
import json, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
TR = sys.argv[1]
PX, L0, W = 6.14, 52.94, 23e-3
C1, C2, C3 = 42842.1, 447.0, 0.0


def tension(lam, t):
    """Uniaxial nominal stress times the strip's cut cross-section."""
    I = lam ** 2 + 2 / lam - 3
    return 2 * (lam - lam ** -2) * (C1 + 2 * C2 * I + 3 * C3 * I ** 2) * W * t


def length(nm, bead, tip):
    c = np.load(os.path.join(TR, nm + ".npz"))["te_c"]
    P = np.vstack([bead, c, tip])
    return float(np.sum(np.linalg.norm(np.diff(P, axis=0), axis=1)) / PX)


out = {}
for nm, bead, tip in (("1800V_1hz_front_avi", (262, 510), (620, 352)), ("1800V_1hz_front_sine_tension", (232, 512), (617, 339))):
    L = length(nm, bead, tip)
    for end, l in (("bead centre", L), ("bead edge", L - 3.2)):
        lam = l / L0
        out[nm + ", " + end] = dict(L_mm=l, lam=lam, T_60um_mN=tension(lam, 60e-6) * 1e3, T_87um_mN=tension(lam, 87e-6) * 1e3)
        print("%-45s L %.1f lam %.3f T %.0f-%.0f mN" % (nm + ", " + end, l, lam, out[nm + ", " + end]["T_60um_mN"], out[nm + ", " + end]["T_87um_mN"]))
# edge curvature and the skin pull across it, before voltage, first run: T kappa = N_n
G = json.load(open(os.path.join(HERE, "data", "geometry.json")))
s, d = np.array(G["s_mm"]), np.array(G["before"]["sag_mm"])
kap = 8 * d.max() / s[-1] ** 2 * 1e3                               # 1/m, shallow arc
out["kappa_1_m"] = kap
out["N_n_N_m"] = [kap * 50e-3, kap * 96e-3]
# hanging string: what gravity alone does at these tensions
w = 1.1e3 * 9.81 * (W * L0 * 1e-3 * 60e-6 + 0.5 * 6.9e-4 * 60e-6) / 0.065
out["gravity_sag_mm"] = [w * .065 ** 2 / 8 / T * 1e3 for T in (50e-3, 96e-3)]
out["lateral_load_for_1p8mm_N_m"] = [8 * T * 1.8e-3 / .065 ** 2 for T in (50e-3, 96e-3)]
print("kappa %.2f 1/m, N_n %.2f-%.2f N/m, gravity sag %.2f-%.2f mm, load for 1.8 mm %.2f-%.2f N/m (roll weight %.3f N/m)" % (kap, *out["N_n_N_m"], *out["gravity_sag_mm"], *out["lateral_load_for_1p8mm_N_m"], w))
json.dump(out, open(os.path.join(HERE, "data", "fibre_tension.json"), "w"), indent=1)
