"""Skin and fibre modes, and centre of mass, against trailing-edge mass and rib. Same matrices as skin.py, at the 0 V and voltage-on tensions.
Run: python te_modes.py -> data/te_modes.json"""
import json, os
import numpy as np
import scipy.linalg as sl
import skin
HERE = os.path.dirname(os.path.abspath(__file__))
BASE = dict(H=30e-6, V=1370.0, n=2, w_s=0.020, w_e=0.005, N_s0=1.84, N_c0=0.085, F0=0.019, ns_te=0.67, tip_fixed=True, ns=16, nc=8)
STIFF = dict(ribs=(0.5,), rib_EI=4.3e-5, rib_mg_mm=0.155)
RHO_AIR = 1.225
out = []
for nm, w in (("plain", {}), ("one stiff rib", STIFF)):
    for tm in (0.1, 0.25, 0.5, 1.0, 1.5, 2.0):
        for v in (0.0, 1.0):
            p = dict(skin.PARAMS, **BASE, **w, te_mass=tm)
            S = skin.Structure(p)
            Nc, Ns, F, bow = S.tensions(np.zeros((p["ns"] + 1, p["nc"] + 1)), v)
            K = Nc * S.Kc + Ns * S.Ks + F * S.Kf + S.Kle
            ev, vec = sl.eigh(K, S.M)
            f = np.sqrt(np.maximum(ev[:4], 0)) / (2 * np.pi)
            # share of each mode's kinetic energy in the fibre row
            te = (S.T @ vec[:, 0]).reshape(p["ns"] + 1, p["nc"] + 1)
            m = skin.masses(p)
            x_cg = (m["sheet"] * p["c"] / 2 + m["fibre"] * p["c"]) / (m["sheet"] + m["fibre"] + m["le"] + m["batten"])
            out.append(dict(wing=nm, te_mass=tm, v=v, f_Hz=f.tolist(), Nc=Nc, Ns=Ns, F_mN=F * 1e3, cg_pct=x_cg / p["c"] * 100, fibre_mg=m["fibre"] * 1e6, le_mg=m["le"] * 1e6, sheet_mg=m["sheet"] * 1e6))
            print("%-14s te x%.2f v %.0f: f %5.1f %5.1f %5.1f Hz  Nc %.3f Ns %.2f F %.1f mN  cg %2.0f %% chord, fibre %.0f mg" % (nm, tm, v, f[0], f[1], f[2], Nc, Ns, F * 1e3, x_cg / p["c"] * 100, m["fibre"] * 1e6))
json.dump(out, open(os.path.join(HERE, "data", "te_modes.json"), "w"), indent=1)
