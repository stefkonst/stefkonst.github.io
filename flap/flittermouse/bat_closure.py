"""Lift over weight for the bat kinematics runs, at Flittermouse size and bumblebee bat size. Same closure as closure.py.
Wing mass at H = 30 um. Fold adds an assumed mechanism mass per wing (hinge, spring, fibre). Run: python bat_closure.py -> data/bat_closure.json"""
import json, os
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
R = [json.loads(l) for l in open(os.path.join(HERE, "data", "bat_kinematics.jsonl"))]
R = [r for r in R if "error" not in r]
ETA_TR, M_FRAME = 0.8, 0.2e-3
RHO_F, H = 1070.0, 30e-6
LE_MG_PER_MM = 43.0 / 50             # cross 1.5 x 0.2 mm; the longer wing needs more, so this is optimistic there
FOLD_MG = 30.0                       # assumption


def wing_mass(b, c, n=2, ws=0.020):
    t = (n + 1) * H
    return (RHO_F * t * (b * c + ws * b)) + LE_MG_PER_MM * b * 1e3 * 1e-6 + 3e-6


def key(r):
    return (r["fold"], r["tilt_deg"], r["twist"])


def best(rows, e, s=1.0, cap=None, n=2, ws=0.020):
    """Best lift/weight with thrust >= 0. s = suction share of the thrust. cap: use the capped forces."""
    out = None
    for r in rows:
        L, T, P = (r[f"L_{cap}"], r[f"T_{cap}"], r[f"P_{cap}"]) if cap else (r["L"], r["T"], r["P"])
        T = s * T + (1 - s) * r["T_ns"]
        if T < 0 or P <= 0:
            continue
        mw = wing_mass(r["b"], r["c"], n, ws) + (FOLD_MG * 1e-6 if r["fold"] else 0.0)
        mc = P / (e * r["f"] * ETA_TR)
        m = 2 * (mc + mw) + M_FRAME
        lw = 2 * L / (m * 9.81)
        if out is None or lw > out["LW"]:
            out = dict(LW=lw, L=L * 1e3, T=T * 1e3, P=P * 1e3, m_tot=m * 1e3, m_core=mc * 1e3, m_wing=mw * 1e3, U=r["U"], f=r["f"], Phi=r["Phi_deg"], alpha=r["alpha_deg"], fold=r["fold"], tilt=r["tilt_deg"], twist=r["twist"])
    return out


res = {}
for (b, c), name in (((0.05, 0.025), "5x2.5"), ((0.07, 0.03), "7x3")):
    S = [r for r in R if r["b"] == b]
    res[name] = {}
    print(name, "wing mass n=2 20mm", round(wing_mass(b, c) * 1e3, 1), "n=1 10mm", round(wing_mass(b, c, 1, 0.01) * 1e3, 1), "mg x1e3")
    for e in (0.8, 1.25, 2.1):
        for s, cap in ((1.0, None), (1.0, 1.6), (0.5, None), (0.5, 1.6)):
            line = {}
            for fo in (0.0, 0.2, 0.4):
                rows = [r for r in S if r["fold"] == fo]
                for n, ws in ((2, 0.020), (1, 0.010)):
                    line[f"fold{fo}_n{n}"] = best(rows, e, s, cap, n, ws)
            res[name][f"e{e}_s{s}_cap{cap}"] = line
            print(" e %.2f s %.1f cap %s: " % (e, s, cap) + "  ".join(f"{k} " + ("%.2f" % v["LW"] if v else "-") for k, v in line.items()))
    # one factor at a time, same flight point, 25 Hz +-30, 4 m/s, 12 deg
    base = dict(U=4.0, f=25.0, Phi_deg=30.0, alpha_deg=12.0)
    Sb = [r for r in S if all(r[k] == v for k, v in base.items())]
    for r in sorted(Sb, key=key):
        print("   fold %.1f tilt %2.0f twist %.1f: L %5.2f T %5.2f T_ns %5.2f P %5.1f  capped L %5.2f T %5.2f P %5.1f" % (r["fold"], r["tilt_deg"], r["twist"], r["L"] * 1e3, r["T"] * 1e3, r["T_ns"] * 1e3, r["P"] * 1e3, r["L_1.6"] * 1e3, r["T_1.6"] * 1e3, r["P_1.6"] * 1e3))
    res[name]["oat"] = [{k: r[k] for k in ("fold", "tilt_deg", "twist", "L", "T", "T_ns", "P", "L_1.6", "T_1.6", "P_1.6")} for r in sorted(Sb, key=key)]
json.dump(res, open(os.path.join(HERE, "data", "bat_closure.json"), "w"), indent=1)
