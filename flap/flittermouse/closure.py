"""W8. Mass and power closure over the stroke map. For each flight condition, the core mass that supplies the air power,
then lift against weight. Run: python closure.py <file> -> prints the best cases, writes data/closure_<file>.json"""
import json, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
E_CYC = dict(bench=0.8, chen=1.25, best=2.1)       # J/kg per cycle: bench net sine; Ren, Kim et al.; 60 V/um instead of 46, times 1.7
ETA_TR, M_FRAME = 0.8, 0.2e-3


def close(r, e, m_w, suction=1.0):
    """Core mass from power, fixed point not needed: power per wing is given. Returns lift/weight, thrust."""
    m_c = r["P"] / (e * r["f"] * ETA_TR)
    m_tot = 2 * (m_c + m_w) + M_FRAME
    T = suction * r["T"] + (1 - suction) * r["T_ns"]
    return dict(m_c=m_c * 1e3, m_tot=m_tot * 1e3, LW=2 * r["L"] / (m_tot * 9.81), T=T * 1e3)


if __name__ == "__main__":
    tag = sys.argv[1]
    R = [json.loads(l) for l in open(os.path.join(HERE, "data", f"timing_{tag}.jsonl"))]
    out = {}
    for ename, e in E_CYC.items():
        for s in (1.0, 0.5):
            rows = []
            for r in R:
                if r["mode"] != "down" or "error" in r:
                    continue
                c = close(r, e, r["mass_mg"] * 1e-6, s)
                if c["T"] >= 0:
                    rows.append(dict(c, U=r["U"] if "U" in r else 4.0, f=r["f"], Phi=r["Phi_deg"], alpha=r["alpha_deg"], L=r["L"] * 1e3, P=r["P"] * 1e3, **{k: r[k] for k in ("batten_k",) if k in r}))
            rows.sort(key=lambda x: -x["LW"])
            out[f"{ename}_s{s}"] = rows[:6]
            print(ename, "suction", s, len(rows), "with T>=0")
            for x in rows[:3]:
                print("   U%.0f f%.0f P%.0f a%.0f: L %.1f P %.1f core %.2f g total %.2f g  L/W %.2f T %+.2f" % (x["U"], x["f"], x["Phi"], x["alpha"], x["L"], x["P"], x["m_c"], x["m_tot"], x["LW"], x["T"]))
    json.dump(out, open(os.path.join(HERE, "data", f"closure_{tag}.json"), "w"), indent=1, default=float)
