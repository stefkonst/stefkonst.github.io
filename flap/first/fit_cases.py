"""Fitted wing: shape at the fitted pull, voltage series, mesh check, pull sweep for 2 and 4 layers, and lift with the measured outline.
Run: python fit_cases.py -> data/fit_cases.jsonl, data/field_*.npz"""
import json, math, multiprocessing as mp, os, sys, time
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np
import fit_wing as FW
M = FW.M
_VLM = M.VLM
Q = 0.5 * 1.225 * 16 * 1e3
SETS = {"passive roll, eta .4": (0.4, 0.0, 5e-3), "passive roll, eta .15": (0.15, 0.0, 5e-3), "passive roll, eta .05": (0.05, 0.0, 5e-3), "band 5 mm, eta .05": (0.05, 0.4, 5e-3), "band 5 mm, eta .15": (0.15, 0.4, 5e-3), "whole roll .15, eta .15": (0.15, 0.15, 19e-3), "whole roll .15, eta .05": (0.05, 0.15, 19e-3), "band 5 mm, eta .1": (0.1, 0.4, 5e-3)}


class VLM2(_VLM):
    def __init__(self, xs, ly, ty, ny, cosine=False):
        t = np.linspace(0, 1, 800)
        p = FW.le_path(t)
        le = np.interp(xs, p[:, 0], p[:, 1])
        ty = np.array(ty)
        ty[0] = ty[1]
        super().__init__(xs, le, np.maximum(ty, le + 1.5e-3), ny, cosine)


def lift(k):
    t0 = time.time()
    out = dict(k)
    try:
        M.VLM = VLM2
        w = FW.build(k)
        fx = w.fixed.copy()
        r = w.relax(0.0, x_init=w.x0, fixed=fx, fixed_x=w.x0)
        xr = r["x"]
        s0 = FW.fibre(w, xr)
        w.p["U"], w.p["alpha_deg"] = 4.0, k.get("alpha", 6.0)
        w.build_aero(xr)
        a0 = w.relax(0.0, x_init=xr, fixed=fx, fixed_x=w.x0, aero=True)
        a1 = w.relax(k["V"], x_init=a0["x"], fixed=fx, fixed_x=w.x0, aero=True)
        A = float(w.vlm.area * 1e4)
        out.update(area_cm2=A, ok=[bool(r["ok"]), bool(a0["ok"]), bool(a1["ok"])], sag0=max(s0[1]), L0=float(w.aero(a0["x"])[1]) * Q * A * 1e-4, L1=float(w.aero(a1["x"])[1]) * Q * A * 1e-4,
                   z0=float(a0["x"][:, 2].max() * 1e3), z1=float(a1["x"][:, 2].max() * 1e3), wr0=w.summary(a0["x"], 0.0)[0]["wrinkled"], wr1=w.summary(a1["x"], k["V"])[0]["wrinkled"], sag0a=max(FW.fibre(w, a0["x"])[1]), sag1a=max(FW.fibre(w, a1["x"])[1]))
    except Exception as e:
        out["error"] = repr(e)[:200]
    out["wall"] = time.time() - t0
    return out


def dispatch(j):
    return lift(j) if j.get("kind") == "lift" else FW.run(j)


def jobs():
    out = []
    for nm, (e, f, wa) in SETS.items():
        out.append(dict(FW.BASE, name=nm, kind="fit", pull=3.2, eta=e, eta_f=f, wa=wa, Vs=(500.0, 750.0, 1000.0, 1250.0, 1500.0)))
    for nm in ("band 5 mm, eta .05", "whole roll .15, eta .15", "passive roll, eta .4"):
        e, f, wa = SETS[nm]
        out.append(dict(FW.BASE, name=nm + ", mesh 1.5", kind="mesh", pull=3.2, eta=e, eta_f=f, wa=wa, h=1.5e-3, Vs=(1500.0,), save="field_h15_" + nm.split(",")[0].replace(" ", "_") + ".npz"))
    for N in (2, 4):
        for p in (0.5, 1.0, 1.5, 2.0, 2.5, 3.2, 4.0, 5.0, 6.0, 8.0):
            for nm in ("band 5 mm, eta .05", "whole roll .15, eta .15", "passive roll, eta .15"):
                e, f, wa = SETS[nm]
                out.append(dict(FW.BASE, name=nm, kind="pull", N=N, pull=p, eta=e, eta_f=f, wa=wa, Vs=(1500.0,)))
                for a in (6.0,):
                    out.append(dict(FW.BASE, name=nm, kind="lift", N=N, pull=p, eta=e, eta_f=f, wa=wa, V=1500.0, alpha=a))
    return out


if __name__ == "__main__":
    with mp.Pool(13) as pool, open(os.path.join(HERE, "data", "fit_cases.jsonl"), "w") as fh:
        for r in pool.imap_unordered(dispatch, jobs()):
            fh.write(json.dumps(r, default=float) + "\n"); fh.flush()
            if "error" in r:
                print(r.get("kind"), r["name"], r["pull"], r["error"]); continue
            if r["kind"] == "lift":
                print("LIFT N %d pull %.1f %-26s sag0 %.2f L %.1f -> %.1f dL %+.2f z %.1f -> %.1f wr %.2f/%.2f %s wall %.0f" % (r["N"], r["pull"], r["name"], r["sag0"], r["L0"], r["L1"], r["L1"] - r["L0"], r["z0"], r["z1"], r["wr0"], r["wr1"], r["ok"], r["wall"]), flush=True)
            else:
                q = r["res"]
                print("%s N %d pull %.1f %-30s sag %s T %s" % (r["kind"], r["N"], r["pull"], r["name"], " ".join("%.2f" % max(x["sag"]) for x in q), " ".join("%.0f" % x["T_mN"] for x in q)), flush=True)
    print("DONE", flush=True)
