"""Prints the build study table."""
import json, os
HERE = os.path.dirname(os.path.abspath(__file__))
R = [json.loads(l) for l in open(os.path.join(HERE, "data", "build_study.jsonl"))]
Q = 0.5 * 1.225 * 16 * 1e3
names = []
for r in R:
    if r["name"] not in names:
        names.append(r["name"])
for n in sorted(names):
    d = {r["alpha"]: r for r in R if r["name"] == n}
    s = "%-50s" % n[:50]
    for a in (6.0, 10.0):
        r = d.get(a)
        if not r:
            s += " | -"
            continue
        ok = r.get("ok0") and r.get("ok1") and "V1" in r
        v0 = r.get("V0", {})
        s += " | %s%s CL %.2f dL %s F %2.0f z0 %.1f z1 %.1f" % ("Y" if r.get("ok0") else "n", "Y" if r.get("ok1") else "n", v0.get("CL", 0), "%.2f" % ((r["V1"]["CL"] - v0["CL"]) * Q * r["area_cm2"] * 1e-4) if ok else " -  ", v0.get("F_mN", 0), v0.get("z_max_mm", 0), r.get("V1", {}).get("z_max_mm", 0))
    print(s, "m %.0f" % (d[6.0].get("m_skin", 0) + d[6.0].get("m_fibre", 0)) if 6.0 in d else "")
