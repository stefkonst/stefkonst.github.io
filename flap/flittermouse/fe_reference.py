"""Reference runs for the 5 cm panel, using the existing finite-element wing (full_membrane/membrane.py) unchanged:
steady, one panel, root rod and batten rotated to vertical, sheet + rolled fibre, steady lattice.
Run: python fe_reference.py [workers] -> data/fe_reference.jsonl, one line per case, safe to stop and rerun."""
import itertools, json, math, multiprocessing as mp, os, sys, time
os.environ.setdefault("OMP_NUM_THREADS", "1"); os.environ.setdefault("OPENBLAS_NUM_THREADS", "1"); os.environ.setdefault("MKL_NUM_THREADS", "1")
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
FM = os.path.join(HERE, "..", "full_membrane")
sys.path[:0] = [FM, os.path.join(HERE, "..", "plagiopatagiales")]
import membrane as M
from tilt_study import rotate, AX, AY

OUT = os.path.join(HERE, "data", "fe_reference.jsonl")
BASE = dict(theta=5.0, ratio=1.77, N=1, span=0.050, chord=0.025, lam_n=1.04, ws=0.010, we=0.003, eta=0.5, eta_f=0.4, active=True, U=4.0, alpha=6.0, V=1600.0, h=2e-3)


def build(k):
    th, tr, N = math.radians(k["theta"]), math.radians(k["theta"] * k["ratio"]), k["N"]
    yc = AY + k["chord"]
    C = (AX + k["span"] + k["chord"] * math.tan(th), yc)
    D = (AX + k["chord"] * math.tan(tr), yc)
    Hm = (N + 1) * 35e-6
    tip = AX + k["span"]                 # planform ends at the batten, or the lattice runs on past the sheet
    w = M.Wing(panels=(0,), C=C, D=D, E=(tip, AY), T=(tip, AY + 1e-3), theta_deg=k["theta"], lam_d=(1.0, 1.0), lam_n=(k["lam_n"], k["lam_n"]), H=35e-6, t=(Hm, Hm), n_a=(N, N),
               w_s=(k["ws"], k["ws"]), w_a_frac=(k["we"] / k["ws"],) * 2, n_f=N, eta_s=k["eta"], eta_f=k["eta_f"], active_fibre=k["active"],
               U=k["U"], alpha_deg=k["alpha"], h=k["h"], na_x=24, na_y=12)
    x = w.x0.copy()
    g0 = w.ids[0]
    seam, rn = g0[-1, :], g0[0, :]
    x[seam, :2] = rotate(x[seam, :2], x[seam[0], :2].copy(), th)
    x[rn, :2] = rotate(x[rn, :2], x[rn[0], :2].copy(), tr)
    return w, x


def state(w, x, V):
    s, f = w.summary(x, V), w.fibre_report(x, V)
    return dict(CL=float(w.aero(x)[1]), z_max_mm=float(x[:, 2].max() * 1e3), z_min_mm=float(x[:, 2].min() * 1e3),
                F_mN=f[0]["F"] * 1e3, bow_mm=f[0]["sag"] * 1e3, zedge_mm=f[0]["w_edge"] * 1e3, N1=s[0]["N1"], N2=s[0]["N2"], wr=s[0]["wrinkle"] if "wrinkle" in s[0] else None)


def run(over):
    t0 = time.time()
    k = dict(BASE, **over)
    out = dict(over)
    try:
        w, x = build(k)
        fx = w.fixed.copy()
        r = w.relax(0.0, x_init=x, fixed=fx, fixed_x=x)
        xr = r["x"]
        w.build_aero(xr)
        out.update(ok_ns=bool(r["ok"]), area_cm2=float(w.vlm.area * 1e4), CL_flat=float(w.vlm.CL(w.vlm.solve(k["U"], math.radians(k["alpha"])))))
        out["ns"] = state(w, xr, 0.0)
        a0 = w.relax(0.0, x_init=xr, fixed=fx, fixed_x=xr, aero=True)
        out.update(ok0=bool(a0["ok"]), div0=bool(a0.get("diverged")))
        if not a0.get("diverged"):
            out["V0"] = state(w, a0["x"], 0.0)
            a1 = w.relax(k["V"], x_init=a0["x"], fixed=fx, fixed_x=xr, aero=True)
            out.update(ok1=bool(a1["ok"]), div1=bool(a1.get("diverged")))
            if not a1.get("diverged"):
                out["V1"] = state(w, a1["x"], k["V"])
    except Exception as e:
        out["error"] = repr(e)
    out["wall"] = time.time() - t0
    return out


def cases():
    c = []
    for th, ln, eta, al in itertools.product((3.0, 5.0, 7.0), (1.02, 1.04, 1.06), (0.5, 0.25), (6.0, 10.0)):
        c.append(dict(theta=th, lam_n=ln, eta=eta, alpha=al))
    for U in (3.0, 5.0, 6.0):
        c.append(dict(theta=5.0, lam_n=1.04, U=U))
    for N in (2,):
        c.append(dict(theta=5.0, lam_n=1.04, N=N))
    c.append(dict(theta=5.0, lam_n=1.04, active=False))
    c.append(dict(theta=5.0, lam_n=1.04, h=1.5e-3))
    return c


if __name__ == "__main__":
    workers = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    done = []
    if os.path.exists(OUT):
        done = [json.loads(l) for l in open(OUT)]
    key = lambda d: json.dumps({q: d[q] for q in sorted(d) if q in BASE}, sort_keys=True)
    seen = {key(d) for d in done}
    todo = [c for c in cases() if key(c) not in seen]
    print(len(todo), "cases", flush=True)
    with mp.Pool(workers) as pool, open(OUT, "a") as fh:
        for r in pool.imap_unordered(run, todo):
            fh.write(json.dumps(r) + "\n"); fh.flush()
            print({q: r.get(q) for q in ("theta", "lam_n", "eta", "alpha", "U", "ok0", "ok1", "error")}, "CL", r.get("V0", {}).get("CL"), r.get("V1", {}).get("CL"), "wall %.0f" % r["wall"], flush=True)
    print("FE DONE", flush=True)
