"""Why does span pre-stretch beat rotation? Same wing, tilted cut with rotation against straight cut with pre-stretch, with fields. Also a mesh check and an energy check.
Run: python stretch_check.py -> data/stretch_check.json"""
import json, math, multiprocessing as mp, os, sys
os.environ.setdefault("OMP_NUM_THREADS", "1"); os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import wing_shape as W
from wing_shape import M
Q = 0.5 * 1.225 * 16


def fields(w, x, V):
    d = w.field(x, V)
    s = w.summary(x, V)[0]
    # span tension resultant by chord band
    xe = w.x0[w.tri].mean(1)
    yy = (xe[:, 1] - W.AY) / (xe[:, 1].max() - W.AY)
    N1 = d["N1"]
    bands = [float(np.average(N1[(yy >= a) & (yy < a + 1 / 3)], weights=w.A0[(yy >= a) & (yy < a + 1 / 3)])) for a in (0, 1 / 3, 2 / 3)]
    g = w.ids[0]
    return dict(taut=s["taut"], wrinkled=s["wrinkled"], slack=s.get("slack"), N1_LE_mid_TE=bands, z_mid_chord_mm=(x[g[g.shape[0] // 2, :], 2] * 1e3).round(3).tolist(), y_mid_chord_mm=(x[g[g.shape[0] // 2, :], 1] * 1e3).round(2).tolist(), z_te_mm=(x[g[:, -1], 2] * 1e3).round(3).tolist())


def run(job):
    name, over = job
    k = dict(dict(W.BASE, V=1370.0), **over)
    w, x, tr = W.build(k)
    fx = w.fixed.copy()
    r = w.relax(0.0, x_init=x, fixed=fx, fixed_x=x)
    xr = r["x"]
    w.build_aero(xr)
    a0 = w.relax(0.0, x_init=xr, fixed=fx, fixed_x=xr, aero=True)
    a1 = w.relax(k["V"], x_init=a0["x"], fixed=fx, fixed_x=xr, aero=True)
    # lattice rebuilt on the voltage-on planform: does the fixed lattice matter?
    cl0, cl1 = float(w.aero(a0["x"])[1]), float(w.aero(a1["x"])[1])
    area = float(w.vlm.area)
    w.build_aero(a1["x"])
    cl1_re = float(w.aero(a1["x"])[1])
    area1 = float(w.vlm.area)
    w.build_aero(a0["x"])
    cl0_re = float(w.aero(a0["x"])[1])
    area0 = float(w.vlm.area)
    return dict(name=name, ok=[bool(r["ok"]), bool(a0["ok"]), bool(a1["ok"])], tr=tr, CL0=cl0, CL1=cl1, dL_mN=(cl1 - cl0) * Q * area * 1e3, CL0_re=cl0_re, CL1_re=cl1_re, area0=area0 * 1e4, area1=area1 * 1e4, dL_re_mN=(cl1_re * area1 - cl0_re * area0) * Q * 1e3,
                ns=fields(w, xr, 0.0), V0=fields(w, a0["x"], 0.0), V1=fields(w, a1["x"], k["V"]), F0=W.state(w, a0["x"], 0.0)["F_mN"], F1=W.state(w, a1["x"], k["V"])["F_mN"], bow0=W.state(w, a0["x"], 0.0)["bow_mm"], bow1=W.state(w, a1["x"], k["V"])["bow_mm"])


JOBS = [("rotation, 1.7 mm", dict()), ("pre-stretch 1.04", dict(theta=0.0, stretch=0.0, lam_d=1.04)),
        ("rotation, 3.5 mm", dict(stretch=3.5e-3)), ("rotation, 5 mm", dict(stretch=5e-3)),
        ("pre-stretch 1.04, mesh 1.5", dict(theta=0.0, stretch=0.0, lam_d=1.04, h=1.5e-3)), ("pre-stretch 1.04, mesh 1", dict(theta=0.0, stretch=0.0, lam_d=1.04, h=1.0e-3)),
        ("pre-stretch 1.04, 35 um 1.6 kV", dict(theta=0.0, stretch=0.0, lam_d=1.04, H=35e-6, V=1600.0)),
        ("pre-stretch 1.04, chord 1.00", dict(theta=0.0, stretch=0.0, lam_d=1.04, lam_n=1.0)), ("pre-stretch 1.02, chord 1.03", dict(theta=0.0, stretch=0.0, lam_d=1.02, lam_n=1.03)),
        ("pre-stretch 1.03", dict(theta=0.0, stretch=0.0, lam_d=1.03)), ("pre-stretch 1.08", dict(theta=0.0, stretch=0.0, lam_d=1.08)),
        ("pre-stretch 1.04, passive fibre", dict(theta=0.0, stretch=0.0, lam_d=1.04, eta_f=0.0))]

if __name__ == "__main__":
    with mp.Pool(6) as pool:
        out = pool.map(run, JOBS)
    for r in out:
        print("%-34s ok %s CL %.3f -> %.3f dL %.2f | re-latticed CL %.3f -> %.3f area %.2f -> %.2f dL %.2f | F %.1f -> %.1f bow %.2f -> %.2f" % (r["name"], r["ok"], r["CL0"], r["CL1"], r["dL_mN"], r["CL0_re"], r["CL1_re"], r["area0"], r["area1"], r["dL_re_mN"], r["F0"], r["F1"], r["bow0"], r["bow1"]))
        print("    wrinkled no flow %.0f%% 0V %.0f%% V %.0f%% | N1 LE,mid,TE 0V %s V %s" % (r["ns"]["wrinkled"] * 100, r["V0"]["wrinkled"] * 100, r["V1"]["wrinkled"] * 100, np.round(r["V0"]["N1_LE_mid_TE"], 2), np.round(r["V1"]["N1_LE_mid_TE"], 2)))
        print("    z mid chord 0V", r["V0"]["z_mid_chord_mm"][::4], "V", r["V1"]["z_mid_chord_mm"][::4])
    json.dump(out, open(os.path.join(HERE, "data", "stretch_check.json"), "w"), indent=1)
