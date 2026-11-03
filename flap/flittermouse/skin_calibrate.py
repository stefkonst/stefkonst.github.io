"""Fit k_Nc and k_st of skin.py on FE cases at 10 deg, where the state is unique. All other cases are held out.
Run: python skin_calibrate.py -> data/skin_calibrate.json"""
import itertools, json, multiprocessing as mp, os, sys
os.environ.setdefault("OMP_NUM_THREADS", "1"); os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import skin


def one(a):
    r, kn, ks, extra = a
    kw = dict(f=0.0, cycles=2, zeta=0.3, tip_fixed=True, EI_override=1e3, U=r.get("U", 4.0), alpha=np.radians(r.get("alpha", 6.0)), eta=r.get("eta", 0.5), n=r.get("N", 1),
              N_s0=r["ns"]["N1"], N_c0=max(r["ns"]["N2"], 0.02), F0=r["ns"]["F_mN"] * 1e-3, active=r.get("active", True), w_s=0.010, w_e=0.003, k_Nc=kn, k_st=ks, **extra)
    S = 0.05 * 0.025
    q = 0.5 * 1.225 * kw["U"] ** 2 * S
    A = r["area_cm2"] * 1e-4
    out = dict(kn=kn, ks=ks, extra=extra, case={k: r.get(k) for k in ("theta", "lam_n", "eta", "alpha", "U", "N", "active")},
               fe=[r["V0"]["CL"] * A, r["V1"]["CL"] * A, r["V0"]["zedge_mm"], r["V1"]["zedge_mm"]], ok=[r["ok0"], r["ok1"]])
    for tag, mode in (("0", "off"), ("1", "on")):
        m, o, s_, St = skin.simulate(mode=mode, **kw)
        out["CA" + tag] = m["L_m"] / (0.5 * 1.225 * kw["U"] ** 2)
        out["z" + tag] = float(np.abs(o["w_mid"][-1]) * 1e3)
    return out


if __name__ == "__main__":
    R = [json.loads(l) for l in open(os.path.join(HERE, "data", "fe_reference.jsonl"))]
    R = [r for r in R if "V1" in r]
    fit = [r for r in R if r.get("alpha") == 10.0 and r.get("lam_n") == 1.04 and r["ok0"] and r["ok1"]]
    variants = [dict(ns_te=0.67), dict(ns_te=0.4), dict(ns_te=0.67, ns=16, nc=10), dict(ns_te=1.0, ns=16, nc=10)]
    grid = [(4.0, 1.0), (4.0, 0.3), (2.0, 0.3), (1.0, 1.0)]
    jobs = [(r, a, b, v) for r in fit for a, b in grid for v in variants]
    print(len(fit), "fit cases", len(jobs), "jobs", flush=True)
    with mp.Pool(int(os.environ.get("WORKERS", 13))) as pool:
        res = pool.map(one, jobs, chunksize=1)
    json.dump(res, open(os.path.join(HERE, "data", "skin_calibrate2.json"), "w"), default=float)
    for v in variants:
        for a, b in grid:
            sel = [x for x in res if x["kn"] == a and x["ks"] == b and x["extra"] == v]
            e = lambda f: np.array([f(x) for x in sel])
            sc = 1e4 / 11.3
            A0, A1, F0, F1 = e(lambda x: x["CA0"]) * sc, e(lambda x: x["CA1"]) * sc, e(lambda x: x["fe"][0]) * sc, e(lambda x: x["fe"][1]) * sc
            print(v, a, b, "CL0 rms %.3f CL1 rms %.3f dCL %.3f FE %.3f | z0 %.2f FE %.2f z1 %.2f FE %.2f" % (np.sqrt(((A0 - F0) ** 2).mean()), np.sqrt(((A1 - F1) ** 2).mean()), (A1 - A0).mean(), (F1 - F0).mean(), e(lambda x: x["z0"]).mean(), e(lambda x: x["fe"][2]).mean(), e(lambda x: x["z1"]).mean(), e(lambda x: x["fe"][3]).mean()), flush=True)
    print("CAL DONE", flush=True)
