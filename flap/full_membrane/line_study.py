"""Fishing-line tensioning: LE fixed on the mechanism, sheet bonded flat (or pre-stretched), a line to the body sets the tension.
Uses website/site/flap/full_membrane/membrane.py. Output: line_study.json next to this file. Predictions; rough scoping (h = 2.5 mm)."""
import json, math, os, sys
import numpy as np
sys.path.insert(0, r"C:\Users\stefk\Documents\website\site\flap\full_membrane")
import membrane as M

HERE = os.path.dirname(os.path.abspath(__file__))
rows = []


def build(ld=1.0, ln=1.0):
    return M.Wing(lam_d=(ld, ld), lam_n=(ln, ln), h=2.5e-3, max_iter=60000, tol=1e-6, w_s=(13e-3, 13e-3))


def report(lbl, w, x, V=0.0):
    fd = w.field(x, V)
    s = w.summary(x, V)
    f = w.fibre_report(x, V)
    Pn = np.linalg.norm
    out = dict(case=lbl)
    for k in (0, 1):
        e = w.tp == k
        A = w.A0[e]
        n1, n2 = fd["N1"][e], fd["N2"][e]
        out[f"p{k}"] = dict(n1=float(np.sum(A * n1) / A.sum()), n2=float(np.sum(A * n2) / A.sum()), cv=float(np.std(n1) / (abs(np.mean(n1)) + 1e-9)), n1_p5=float(np.percentile(n1, 5)), n1_p95=float(np.percentile(n1, 95)),
                            wrinkled=s[k]["wrinkled"], slack=s[k]["slack"], F_mN=f[k]["F"] * 1e3, sag_mm=f[k]["sag"] * 1e3)
    return out


def line_force(w, x, node, V=0.0):
    return (-w.reactions(x, V)[node, :2]).tolist()


def run(lbl, fix_nodes, move, ld=1.0, ln=1.0, aero=False):
    """fix_nodes(w) -> nodes held; move(w) -> {node: (dx, dy)} in mm, applied to held nodes."""
    w = build(ld, ln)
    fx = np.zeros(w.x0.shape[0], bool)
    fx[fix_nodes(w)] = True
    x = w.x0.copy()
    for n, d in move(w).items():
        x[n, :2] += np.array(d) * 1e-3
        fx[n] = True
    r = w.relax(0.0, x_init=x, fixed=fx, fixed_x=x)
    q = report(lbl, w, r["x"])
    q["ok"] = bool(r["ok"])
    q["line_mN"] = {str(n): (-w.reactions(r["x"])[n, :2] * 1e3).round(1).tolist() for n in move(w)}
    if aero and r["ok"]:
        w.build_aero(r["x"])
        a0 = w.relax(0.0, x_init=r["x"], fixed=fx, fixed_x=r["x"], aero=True)
        a1 = w.relax(1600.0, x_init=a0["x"], fixed=fx, fixed_x=r["x"], aero=True)
        q.update(CL0=a0["CL"], CL1=a1["CL"], dCL=a1["CL"] - a0["CL"], ok0=bool(a0["ok"]), ok1=bool(a1["ok"]), div0=bool(a0.get("diverged")), div1=bool(a1.get("diverged")))
    rows.append(q)
    print(lbl, "ok", q["ok"], "| p1 n1 %.2f n2 %.2f cv %.2f wr %.2f sl %.2f F %.1f | p2 n1 %.2f n2 %.2f cv %.2f wr %.2f sl %.2f F %.1f" % tuple(q[p][k] for p in ("p0", "p1") for k in ("n1", "n2", "cv", "wrinkled", "slack", "F_mN")), "line", q["line_mN"], "CL", q.get("CL0"), q.get("dCL"), q.get("ok1"), flush=True)
    json.dump(rows, open(os.path.join(HERE, "line_study.json"), "w"), indent=1, default=float)


le = lambda w: np.array(sorted(set(int(n) for n in w.le_nodes)))
seam = lambda w: np.array(sorted(set(int(n) for n in w.seam)), int)
root = lambda w: np.array(w.root, int)
tip = lambda w: np.array(w.tip, int)
Cn = lambda w: int(w.fib[0][-1])
Dn = lambda w: int(w.fib[0][0])
Tn = lambda w: int(w.fib[1][-1])

# flat sheet bonded to the LE, no batten: tent. One line at the fibre junction pulling aft; root fibre tip on the body, far fibre tip on the LE tip
for d in (2.0, 4.0, 6.0):
    run(f"A tent, no batten, C aft {d:.0f} mm", lambda w: np.r_[le(w), tip(w), [Dn(w)]], lambda w: {Cn(w): (0.0, d)}, aero=d == 4.0)
# as above with the whole root edge on the body
for d in (2.0, 4.0):
    run(f"B tent + root edge held, C aft {d:.0f} mm", lambda w: np.r_[le(w), tip(w), root(w)], lambda w: {Cn(w): (0.0, d)}, aero=d == 4.0)
# batten kept rigid, line pulls the root fibre tip inboard along the fibre
for d in (2.0, 4.0):
    run(f"C batten rigid, line pulls fibre tip inboard {d:.0f} mm", lambda w: np.r_[le(w), seam(w), tip(w)], lambda w: {Dn(w): (-d, 0.0)})
# batten kept rigid, root rib on a line: rib moved inboard as a bar
for d in (2.0,):
    run(f"D batten rigid, root rib pulled inboard {d:.0f} mm", lambda w: np.r_[le(w), seam(w), tip(w)], lambda w: {int(n): (-d, 0.0) for n in root(w) if n not in set(le(w).tolist())})
# pre-stretched panels as designed, bonded, then the line replaces a rigid batten tip: top-up of tension in place
ln = 1.05 ** -0.5 + 0.04
for d in (0.0, 1.0, 2.0):
    run(f"E design pre-stretch, no batten, C aft {d:.0f} mm", lambda w: np.r_[le(w), tip(w), [Dn(w)], root(w)], lambda w: {Cn(w): (0.0, d)}, ld=1.05, ln=ln, aero=d == 1.0)
