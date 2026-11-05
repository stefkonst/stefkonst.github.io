"""The wing flat, under its own weight, with the roll as long as cut (52.94 mm, so stretched to 1.16-1.22): in-plane and out-of-plane edge motion at 0-1800 V.
Weight from the film (mass x1) or scaled to the weighed wing less its leading edge (250 mg, x2).
Run: python fit_gravity.py [test] -> data/fit_gravity.jsonl"""
import json, multiprocessing as mp, os, sys, time
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np
import fit_wing as FW

B = dict(FW.BASE, N=1, n_f=2, ws=23e-3, wa=20e-3, Vs=(1000.0, 1400.0, 1500.0, 1800.0), tol=3e-7, it=200000)
G = 9.81


def run(k):
    t0 = time.time()
    out = dict(k)
    try:
        w = FW.build(k)
        rho = 1100.0 * k["mass"]
        m = np.zeros(len(w.x0))
        np.add.at(m, w.tri_flat, np.repeat(rho * w.te * w.A0 / 3, 3))
        if getattr(w, "b1", None) is not None and len(w.b1):
            np.add.at(m, w.b1, 0.5 * rho * w.bA * w.bL0)
            np.add.at(m, w.b2, 0.5 * rho * w.bA * w.bL0)
        w.f_ext = np.c_[np.zeros_like(m), np.zeros_like(m), -m * G]
        out["mass_mg"] = float(m.sum() * 1e6)
        fx = w.fixed.copy()
        x = w.x0.copy()
        res = []
        ids = w.fib[0]
        for V in (0.0,) + tuple(k["Vs"]):
            r = w.relax(V, x_init=x, fixed=fx, fixed_x=w.x0)
            x = r["x"]
            s, y = FW.fibre(w, x)
            q = w.tens[0]
            T = q["k"] * max(np.linalg.norm(x[q["n"]] - q["anc"]) - q["L0"], 0)
            _, Nb, _ = w.bars(x, V)
            z = x[ids, 2] * 1e3
            res.append(dict(V=V, ok=bool(r["ok"]), rmax=float(r["rmax"]), sag=float(max(y)), zmin=float(z.min()), zmid=float(np.interp(.5, np.linspace(0, 1, len(z)), z)), zskin=float(x[:, 2].min() * 1e3), T_mN=float(T * 1e3), F_mN=float(Nb.mean() * 1e3), Fmin=float(Nb.min() * 1e3)))
        out["res"] = res
    except Exception as e:
        out["error"] = repr(e)[:200]
    out["wall"] = time.time() - t0
    return out


def jobs(test):
    if test:
        return [dict(B, pull=p, eta=.15, eta_f=f, mass=1.0) for p in (10.0,) for f in (0.0, 0.15)]
    out = []
    for p in (6.0, 9.0, 12.9):
        for e, f in ((0.15, 0.0), (0.15, 0.1), (0.15, 0.2), (0.15, 0.4), (0.4, 0.4), (0.05, 0.6), (0.15, 0.8)):
            for ms in (1.0, 2.0):
                out.append(dict(B, pull=p, eta=e, eta_f=f, mass=ms))
    return out


if __name__ == "__main__":
    test = len(sys.argv) > 1
    J = jobs(test)
    with mp.Pool(min(13, len(J))) as pool, open(os.path.join(HERE, "data", "fit_gravity_test.jsonl" if test else "fit_gravity.jsonl"), "w") as fh:
        for r in pool.imap_unordered(run, J):
            fh.write(json.dumps(r, default=float) + "\n"); fh.flush()
            if "error" in r:
                print(r["pull"], r["eta"], r["eta_f"], r["error"]); continue
            print("pull %.1f eta %.2f ef %.2f mass x%.0f (%.0f mg) | " % (r["pull"], r["eta"], r["eta_f"], r["mass"], r["mass_mg"]) + " | ".join("%.0fV sag %.2f z %.2f T %.0f %s" % (q["V"], q["sag"], q["zmin"], q["T_mN"], "ok" if q["ok"] else "NC") for q in r["res"]) + " wall %.0f" % r["wall"], flush=True)
    print("DONE", flush=True)
