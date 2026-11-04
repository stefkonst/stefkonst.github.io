"""Prints the wing shape table from data/wing_shape.jsonl and writes data/wing_shape_summary.json."""
import json, os
HERE = os.path.dirname(os.path.abspath(__file__))
R = [json.loads(l) for l in open(os.path.join(HERE, "data", "wing_shape.jsonl"))]
from wing_shape import CONCEPTS
Q = 0.5 * 1.225 * 16 * 1e3   # mN per m^2 of CL
rows = []
for n in CONCEPTS:
    d = {r["alpha"]: r for r in R if r["name"] == n}
    if len(d) < 2:
        continue
    row = dict(name=n)
    for a in (6.0, 10.0):
        r = d[a]
        ok = r.get("ok0") and r.get("ok1") and "V1" in r
        row[a] = dict(ok0=r.get("ok0"), ok1=r.get("ok1"), err=r.get("error"), CL0=r.get("V0", {}).get("CL"), CL1=r.get("V1", {}).get("CL"), dL=(r["V1"]["CL"] - r["V0"]["CL"]) * Q * r["area_cm2"] * 1e-4 if ok else None, L1=r["V1"]["CL"] * Q * r["area_cm2"] * 1e-4 if ok else None,
                      F=r.get("V0", {}).get("F_mN"), bow0=r.get("V0", {}).get("bow_mm"), bow1=r.get("V1", {}).get("bow_mm"), z0=r.get("V0", {}).get("z_max_mm"), z1=r.get("V1", {}).get("z_max_mm"), Nc=r.get("V0", {}).get("N2"), Ns=r.get("V0", {}).get("N1"), zmin0=r.get("V0", {}).get("z_min_mm"), zmin1=r.get("V1", {}).get("z_min_mm"))
    r = d[6.0]
    row.update(mass=r.get("m_skin", 0) + r.get("m_fibre", 0) + r.get("m_le", 0) + r.get("m_veins", 0) + 3, I=r.get("I_mgmm2"), area=r.get("area_cm2"), tr=r.get("tr_deg"), CLflat=r.get("CL_flat"))
    rows.append(row)
json.dump(rows, open(os.path.join(HERE, "data", "wing_shape_summary.json"), "w"), indent=1)
f = lambda x, s=".2f": "-" if x is None else format(x, s)
print("%-58s %5s %6s %5s | a6: ok CL0 CL1 dL F z0 z1 | a10: ok CL0 CL1 dL z0" % ("case", "mg", "I", "area"))
for r in rows:
    a, b = r[6.0], r[10.0]
    print("%-58s %5.0f %6.0f %5.1f | %s%s %s %s %s %s %s %s | %s%s %s %s %s %s" % (r["name"][:58], r["mass"], (r["I"] or 0) / 1e3, r["area"] or 0, "Y" if a["ok0"] else "n", "Y" if a["ok1"] else "n", f(a["CL0"]), f(a["CL1"]), f(a["dL"]), f(a["F"], ".0f"), f(a["z0"], ".1f"), f(a["z1"], ".1f"), "Y" if b["ok0"] else "n", "Y" if b["ok1"] else "n", f(b["CL0"]), f(b["CL1"]), f(b["dL"]), f(b["z0"], ".1f")))
