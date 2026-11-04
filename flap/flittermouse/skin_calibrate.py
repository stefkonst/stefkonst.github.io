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
              N_s0=r["ns"]["N1"], N_c0=max(r["ns"]["N2"], 0.02), F0=r["ns"]["F_mN"] * 1e-3, active=r.get("active", True), w_s=r.get("ws", 0.010), w_e=r.get("we", 0.003), k_Nc=kn, k_st=ks, **extra)
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
    R = [json.loads(l) for l in open(os.path.join(HERE, "data", "fe_design.jsonl"))]
    R = sorted((r for r in R if r["ok0"] and r["ok1"] and r["theta"] < 10), key=lambda r: (r["N"], r["ws"], r["theta"], r["lam_n"], r["alpha"]))
    fit, held = R[0::2], R[1::2]
    grid = [(1.0, 1.0), (2.0, 1.0), (1.0, 0.3)]
    base = dict(ns_te=0.67)
    jobs = [(r, a, b, base) for r in R for a, b in grid]
    print(len(fit), "fit", len(held), "held out", len(jobs), "jobs", flush=True)
    with mp.Pool(int(os.environ.get("WORKERS", 13))) as pool:
        res = pool.map(one, jobs, chunksize=1)
    for x, (r, a, b, _) in zip(res, jobs):
        x["area"] = r["area_cm2"]
        x["fit"] = r in fit
    json.dump(res, open(os.path.join(HERE, "data", "skin_design_check.json"), "w"), default=float)
    for a, b in grid:
        for tag, flag in (("fit", True), ("held", False)):
            sel = [x for x in res if x["kn"] == a and x["ks"] == b and x["fit"] == flag]
            e = lambda f: np.array([f(x) for x in sel])
            A = e(lambda x: x["area"]) * 1e-4
            C0, C1, F0, F1 = e(lambda x: x["CA0"]) / A, e(lambda x: x["CA1"]) / A, e(lambda x: x["fe"][0]) / A, e(lambda x: x["fe"][1]) / A
            d, f = C1 - C0, F1 - F0
            print(tag, a, b, "CL0 rms %.3f CL1 rms %.3f | dCL mine %.3f FE %.3f rms %.3f corr %.2f | z1 %.2f FE %.2f corr %.2f" % (np.sqrt(((C0 - F0) ** 2).mean()), np.sqrt(((C1 - F1) ** 2).mean()), d.mean(), f.mean(), np.sqrt(((d - f) ** 2).mean()), np.corrcoef(d, f)[0, 1], e(lambda x: x["z1"]).mean(), e(lambda x: x["fe"][3]).mean(), np.corrcoef(e(lambda x: x["z1"]), e(lambda x: x["fe"][3]))[0, 1]), flush=True)
    print("CHECK DONE", flush=True)
