"""The wing as measured in the video, no air. LE and tip held, free root edge, rolled edge held at the knot by the stub of roll beyond the root corner.
Pull = how far the roll root was pulled from where it was cut. Compared with the measured sag before voltage and the shift at voltage.
Run: python fit_wing.py [mesh test] -> data/fit_wing.jsonl"""
import json, math, multiprocessing as mp, os, sys, time
os.environ.setdefault("OMP_NUM_THREADS", "1"); os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.join(HERE, "..", "flittermouse"), os.path.join(HERE, "..", "full_membrane"), os.path.join(HERE, "..", "plagiopatagiales")]
import membrane as M
G = json.load(open(os.path.join(HERE, "data", "geometry.json")))["points_mm_along_LE"]
P = {k: np.array(v) * 1e-3 for k, v in G.items()}
L_CURVE = np.linalg.norm(P["tip"] - P["LE_knee"]) * 1.1
H = 30e-6
EPS = 8.854e-12 * 2.8


def le_path(s):
    """Point on the leading edge at fraction s of its length: straight to the knee, quarter ellipse to the tip."""
    L1 = P["LE_knee"][0]
    a, b = P["tip"][0] - L1, P["tip"][1]
    th = np.linspace(0, math.pi / 2, 400)
    cx, cy = L1 + a * np.sin(th), b * (1 - np.cos(th))
    ds = np.r_[0, np.cumsum(np.hypot(np.diff(cx), np.diff(cy)))]
    Lt = L1 + ds[-1]
    q = np.asarray(s) * Lt
    x = np.where(q <= L1, q, np.interp(q - L1, ds, cx))
    y = np.where(q <= L1, 0.0, np.interp(q - L1, ds, cy))
    return np.c_[x, y]


def build(k):
    d = (P["TE_tip"] - P["TE_root"]) / np.linalg.norm(P["TE_tip"] - P["TE_root"])
    R0 = P["TE_root"] + k["pull"] * 1e-3 * d                 # where the roll root sat before the pull
    def coons(a, b, c, dd, ni, nj):
        s = np.linspace(0, 1, ni + 1)
        u = np.linspace(0, 1, nj + 1)
        le = le_path(s)
        te = R0[None, :] + (P["TE_tip"] - R0)[None, :] * s[:, None]
        return le[:, None, :] + (te - le)[:, None, :] * u[None, :, None]
    M.coons = coons
    N = k["N"]
    w_s, w_a = k["ws"], k["wa"]
    Hm = (N + 1) * H
    w = M.Wing(panels=(0,), A=tuple(P["LE_root"]), C=tuple(P["TE_tip"]), D=tuple(R0), E=tuple(P["tip"]), T=tuple(P["tip"] + [0, 1e-3]), theta_deg=0.0, lam_d=(1.0, 1.0), lam_n=(1.0, 1.0), H=H, t=(Hm, Hm), n_a=(N, N),
               w_s=(w_s, w_s), w_a_frac=(w_a / w_s,) * 2, n_f=k.get("n_f", N), eta_s=k["eta"], eta_f=k["eta_f"], active_fibre=k["eta_f"] > 0, root_free=True, h=k["h"], tol=k.get("tol", 3e-7), max_iter=k.get("it", 120000),
               tensioners=(dict(at="D", anchor=tuple(P["knot"]), kind="line", k=1.0, F0=0.0),))
    A = w_s * Hm
    Ls = np.linalg.norm(P["knot"] - P["TE_root"])           # stub rest length, as cut
    q = w.tens[0]
    q["k"] = 6 * M.YEOH[0] * A / Ls * k.get("stub", 1.0)     # stub=10: knot and cap stiffen the stub
    q["L0"] = Ls
    w.m[q["n"]] += 2 * q["k"]
    return w


def fibre(w, x):
    ids = w.fib[0]
    pts = x[ids, :2]
    a, b = P["knot"], P["TE_tip"]
    e = (b - a) / np.linalg.norm(b - a)
    n = np.array([-e[1], e[0]])                              # y is aft in these axes, so the side towards the LE is the negative of this
    s = (pts - a) @ e
    return (s * 1e3).tolist(), (-(pts - a) @ n * 1e3).tolist()


def principal(w, x, V):
    d = w.field(x, V)
    th = 0.5 * np.arctan2(2 * d["c01"], d["c00"] - d["c11"])
    return d


def run(k):
    t0 = time.time()
    out = dict(k)
    try:
        w = build(k)
        fx = w.fixed.copy()
        res = []
        x = w.x0.copy()
        for V in [0.0] + list(k["Vs"]):
            r = w.relax(V, x_init=x, fixed=fx, fixed_x=w.x0)
            x = r["x"]
            s, y = fibre(w, x)
            _, Nb, lam = w.bars(x, V)
            sm = w.summary(x, V)[0]
            q = w.tens[0]
            T = q["k"] * max(np.linalg.norm(x[q["n"]] - q["anc"]) - q["L0"], 0)
            res.append(dict(V=V, ok=bool(r["ok"]), rmax=float(r["rmax"]), s=s, sag=y, T_mN=T * 1e3, F_mN=float(Nb.mean() * 1e3), Fmin=float(Nb.min() * 1e3), Fmax=float(Nb.max() * 1e3), wr=sm["wrinkled"], slack=sm["slack"], taut=sm["taut"], N1=sm["N1"], zmax=float(np.abs(x[:, 2]).max() * 1e3), rootpt=(x[w.fib[0][0], :2] * 1e3).tolist()))
        out["res"] = res
        if k.get("save"):
            d = w.field(x, 0.0)
            np.savez(os.path.join(HERE, "data", k["save"]), x=x, tri=w.tri, N1=d["N1"], N2=d["N2"], state=d["state"], c00=d["c00"], c11=d["c11"], c01=d["c01"])
    except Exception as e:
        out["error"] = repr(e)
    out["wall"] = time.time() - t0
    return out


BASE = dict(N=2, ws=20e-3, wa=5e-3, eta=0.4, eta_f=0.4, h=2e-3, Vs=(1500.0,))

if __name__ == "__main__":
    test = len(sys.argv) > 1
    jobs = [dict(BASE, pull=p, eta_f=ef) for p in (2.0, 4.0) for ef in (0.0, 0.4)] if test else [dict(BASE, pull=p, eta=e, eta_f=ef, Vs=(1500.0,)) for p in (1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0) for e, ef in ((0.4, 0.4), (0.4, 0.0), (0.0, 0.4), (0.15, 0.15))]
    OUT = os.path.join(HERE, "data", "fit_wing_test.jsonl" if test else "fit_wing.jsonl")
    with mp.Pool(4 if test else 13) as pool, open(OUT, "w") as fh:
        for r in pool.imap_unordered(run, jobs):
            fh.write(json.dumps(r, default=float) + "\n"); fh.flush()
            if "error" in r:
                print(r["pull"], r["error"]); continue
            a, b = r["res"][0], r["res"][-1]
            print("pull %.1f eta %.2f eta_f %.2f | 0 V %s sag %.2f T %.0f F %.0f wr %.2f sl %.2f N1 %.2f z %.2f rmax %.1e | %.0f V %s sag %.2f T %.0f wr %.2f sl %.2f z %.2f rmax %.1e | wall %.0f" % (r["pull"], r["eta"], r["eta_f"], a["ok"], max(a["sag"]), a["T_mN"], a["F_mN"], a["wr"], a["slack"], a["N1"], a["zmax"], a["rmax"], b["V"], b["ok"], max(b["sag"]), b["T_mN"], b["wr"], b["slack"], b["zmax"], b["rmax"], r["wall"]), flush=True)
    print("DONE", flush=True)
