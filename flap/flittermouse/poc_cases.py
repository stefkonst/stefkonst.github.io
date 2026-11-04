"""The three wings on the POC masks: 3 active layers (4 electrodes), 30 um, 1370 V, straight cut, 4 % span stretch, no chord stretch. With a rib and with a 15 mm roll. Rotation as a control.
Run: python poc_cases.py -> data/poc_cases.jsonl"""
import json, multiprocessing as mp, os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import wing_shape as W
BASE = dict(N=3, V=1370.0, theta=0.0, stretch=0.0, lam_d=1.04, lam_n=1.0)
SH = dict(rectangle=dict(), taper=dict(cr=0.035, ct=0.015), steep=dict(cr=0.040, ct=0.010))
C = {}
for n, s in SH.items():
    C[n] = dict(BASE, **s)
    C[n + ", rib"] = dict(BASE, veins=(W.RIB(0.5),), **s)
    C[n + ", roll 15 mm"] = dict(BASE, ws=0.015, **s)
    C[n + ", 1.6 kV"] = dict(BASE, **dict(s, V=1600.0))
    C[n + ", stretch 2 %"] = dict(BASE, **dict(s, lam_d=1.02))
    C[n + ", stretch 6 %"] = dict(BASE, **dict(s, lam_d=1.06))
C["rectangle, rotation control"] = dict(N=3, V=1370.0)
if __name__ == "__main__":
    jobs = [dict(v, name=n, alpha=a) for n, v in C.items() for a in (6.0, 10.0)]
    with mp.Pool(13) as pool:
        out = pool.map(W.run, jobs)
    with open(os.path.join(HERE, "data", "poc_cases.jsonl"), "w") as fh:
        for r in out:
            fh.write(json.dumps(r) + "\n")
    Q = 0.5 * 1.225 * 16 * 1e3
    for n in C:
        s = "%-28s" % n
        for a in (6.0, 10.0):
            r = next(x for x in out if x["name"] == n and x["alpha"] == a)
            ok = r.get("ok0") and r.get("ok1") and "V1" in r
            s += " | %s%s L0 %.1f L1 %.1f dL %s F %.0f z %.1f->%.1f" % ("Y" if r.get("ok0") else "n", "Y" if r.get("ok1") else "n", r["V0"]["CL"] * Q * r["area_cm2"] * 1e-4, r["V1"]["CL"] * Q * r["area_cm2"] * 1e-4, "%.2f" % ((r["V1"]["CL"] - r["V0"]["CL"]) * Q * r["area_cm2"] * 1e-4) if ok else "-", r["V0"]["F_mN"], r["V0"]["z_max_mm"], r["V1"]["z_max_mm"])
        r = next(x for x in out if x["name"] == n)
        print(s, "| m %.0f mg, I %.0f" % (r["m_skin"] + r["m_fibre"] + r["m_le"] + r.get("m_veins", 0), r["I_mgmm2"] / 1e3))
