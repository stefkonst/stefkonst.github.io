"""Fit the two free constants of the edge equation (skin pull N0 across the edge, chord c behind it) to the finite-element roll cases.
Run: python calib_edge.py -> data/edge_calib.json"""
import json, os
import numpy as np
from scipy.optimize import minimize
import edge_theory as ET
HERE = os.path.dirname(os.path.abspath(__file__))
R = [json.loads(l) for l in open(os.path.join(HERE, "data", "edge_fe.jsonl"))]
R = [r for r in R if "res" in r and r["edge"] == "roll" and r["eta"] == 0.15]
def pred(r, N0, c, qf):
    T0 = r["res"][0]["T_mN"] * 1e-3
    out = []
    for q in r["res"]:
        a = ET.solve(q["V"], T0=T0, N0=N0, eta_f=r["eta_f"], eta_s=r["eta"], p=r["p"], c=c, mu=None)
        b = ET.solve(0.0, T0=T0, N0=N0, eta_f=r["eta_f"], eta_s=r["eta"], p=r["p"], c=c)
        out.append(-(a["d"] - b["d"]) * qf)          # FE z up; equation d down
    return np.array(out)
def meas(r):
    return np.array([q["z_te_mid"] - r["res"][0]["z_te_mid"] for q in r["res"]])
def cost(x):
    N0, c, qf = x
    if N0 <= 0.02 or c <= 0.004 or qf <= 0.1: return 1e9
    return sum(np.sum((pred(r, N0, c, qf) - meas(r)) ** 2) for r in R)
best = min((minimize(cost, x0, method="Nelder-Mead", options=dict(maxiter=400, xatol=1e-4, fatol=1e-6)) for x0 in ((0.2, 0.016, 1.0), (0.1, 0.02, 1.0), (0.3, 0.012, 1.0))), key=lambda m: m.fun)
N0, c, qf = best.x
rms = np.sqrt(best.fun / sum(len(r["res"]) for r in R))
allm = np.concatenate([meas(r) for r in R]); allp = np.concatenate([pred(r, N0, c, qf) for r in R])
print("N0 %.3f N/m, c %.1f mm, load factor %.2f, rms %.2f mm, corr %.3f, n %d" % (N0, c * 1e3, qf, rms, np.corrcoef(allm, allp)[0, 1], len(allm)))
json.dump(dict(N0=N0, c=c, qf=qf, rms_mm=rms, corr=float(np.corrcoef(allm, allp)[0, 1]), n=len(allm), fe=allm.tolist(), eq=allp.tolist()), open(os.path.join(HERE, "data", "edge_calib.json"), "w"))
