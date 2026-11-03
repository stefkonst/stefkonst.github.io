"""W3 check: reduced skin model (skin.py) against the finite elements on the same 5 cm panel, steady, no flapping.
Inputs taken from the FE no-flow state: N_s0 = N1, N_c0 = N2, F0 = fibre force. Nothing fitted.
Rigid frame and a batten bonded to the LE, as in the FE. Output: data/skin_vs_fe.json."""
import json, multiprocessing as mp, os, sys
os.environ.setdefault("OMP_NUM_THREADS", "1"); os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import skin


def one(r):
    n = r.get("N", 1)
    kw = dict(f=0.0, cycles=2, tip_fixed=True, EI_override=1e3, U=r.get("U", 4.0), alpha=np.radians(r.get("alpha", 6.0)), eta=r.get("eta", 0.5), n=n,
              N_s0=r["ns"]["N1"], N_c0=max(r["ns"]["N2"], 0.02), F0=r["ns"]["F_mN"] * 1e-3, active=r.get("active", True), w_s=0.010, w_e=0.003)
    q = 0.5 * 1.225 * kw["U"] ** 2 * 0.05 * 0.025
    out = dict(case={k: r.get(k) for k in ("theta", "lam_n", "eta", "alpha", "U", "N", "active", "h")})
    for tag, mode in (("0", "off"), ("1", "on")):
        m, o, s, S = skin.simulate(mode=mode, **kw)
        out["CL" + tag] = m["L_m"] / q
        out["z" + tag] = float(o["w_mid"][-1] * 1e3)
        out["Ns" + tag], out["Nc" + tag], out["F" + tag] = m["Ns_m"], m["Nc_m"], m["F_m"] * 1e3
    out["fe"] = dict(CL0=r["V0"]["CL"], CL1=r["V1"]["CL"], z0=r["V0"]["zedge_mm"], z1=r["V1"]["zedge_mm"], F0=r["V0"]["F_mN"], F1=r["V1"]["F_mN"], N1_0=r["V0"]["N1"], N1_1=r["V1"]["N1"], N2_0=r["V0"]["N2"], N2_1=r["V1"]["N2"])
    return out


if __name__ == "__main__":
    R = [json.loads(l) for l in open(os.path.join(HERE, "data", "fe_reference.jsonl"))]
    R = [r for r in R if "V1" in r]
    if len(sys.argv) > 1:
        R = R[:int(sys.argv[1])]
    with mp.Pool(12) as pool:
        res = pool.map(one, R)
    json.dump(res, open(os.path.join(HERE, "data", "skin_vs_fe.json"), "w"), indent=1, default=float)
    a = lambda f: np.array([f(x) for x in res])
    for name, mine, fe in (("CL0", a(lambda x: x["CL0"]), a(lambda x: x["fe"]["CL0"])), ("CL1", a(lambda x: x["CL1"]), a(lambda x: x["fe"]["CL1"])),
                           ("dCL", a(lambda x: x["CL1"] - x["CL0"]), a(lambda x: x["fe"]["CL1"] - x["fe"]["CL0"])),
                           ("z0 mm", a(lambda x: x["z0"]), a(lambda x: x["fe"]["z0"])), ("z1 mm", a(lambda x: x["z1"]), a(lambda x: x["fe"]["z1"])),
                           ("dz mm", a(lambda x: x["z1"] - x["z0"]), a(lambda x: x["fe"]["z1"] - x["fe"]["z0"]))):
        print(f"{name}: mean mine {mine.mean():.3f} FE {fe.mean():.3f}  rms {np.sqrt(np.mean((mine-fe)**2)):.3f}  corr {np.corrcoef(mine,fe)[0,1]:.3f}")
    for x in res[:8]:
        print(x["case"], "CL %.3f %.3f FE %.3f %.3f | z %.2f %.2f FE %.2f %.2f | Ns %.2f %.2f FE %.2f %.2f | Nc %.2f %.2f FE %.2f %.2f | F %.1f %.1f FE %.1f %.1f" % (x["CL0"], x["CL1"], x["fe"]["CL0"], x["fe"]["CL1"], x["z0"], x["z1"], x["fe"]["z0"], x["fe"]["z1"], x["Ns0"], x["Ns1"], x["fe"]["N1_0"], x["fe"]["N1_1"], x["Nc0"], x["Nc1"], x["fe"]["N2_0"], x["fe"]["N2_1"], x["F0"], x["F1"], x["fe"]["F0"], x["fe"]["F1"]))
