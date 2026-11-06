"""Edge study on the built wing (first/fit_wing.py geometry, calibrated on the 5 Nov videos).
What changes if the rolled DEA edge is replaced by a nylon or TPU cord, or only part of it is active,
with the wing's weight and an air-like pressure on top.

Edge types:
  roll       the rolled DEA strip, as built (23 mm strip, 20 mm active, 2 active films), eta_f 0..0.8
  nylon      0.10 mm monofilament, E 2.5 GPa (EA 19.6 N, 0.43 mg on the edge); skin still active
  tpu        1.0 x 0.1 mm TPU strip, E 25 MPa (EA 2.5 N, 7 mg)
  part       roll everywhere, only the inner fraction a (near the body) has electrodes
  hybrid     inner fraction a is the active roll, the rest a nylon cord in series
Load: the wing's weight (film density 1100 kg/m3) plus a uniform pressure p (Pa, upward, as lift).
Run: python edge_fe.py [test] -> data/edge_fe.jsonl"""
import json, multiprocessing as mp, os, sys, time
os.environ.setdefault("OMP_NUM_THREADS", "1"); os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "first"))
import numpy as np
import fit_wing as FW
M = FW.M

B = dict(FW.BASE, N=1, n_f=2, ws=23e-3, wa=20e-3, eta=0.15, Vs=(500.0, 1000.0, 1400.0, 1800.0), tol=3e-7, it=250000)
G = 9.81
CORD = {"nylon": dict(EA=19.6, mu=7.85e-9 * 1140), "tpu": dict(EA=2.5, mu=1e-7 * 1200)}


def setup(k):
    kind = k["edge"]
    w = FW.build(dict(k, pull=k.get("pull", 6.0)))
    nb = len(w.b1)
    s = (np.arange(nb) + 0.5) / nb                        # 0 at the root (knot end), 1 at the tip
    mask = np.ones(nb)
    mu = 1100.0 * w.bA                                    # kg/m along the edge
    q = w.tens[0]
    Le = w.bL0.sum()
    P = k.get("pull", 6.0) * 1e-3                         # the skin is stretched by the same pull as the roll wing
    if kind in ("nylon", "tpu"):
        c = CORD[kind]
        w.bA = w.bA * 1e-3                                # no roll: a thin film lip stays for the mesh
        w.bEAc = np.full(nb, c["EA"])
        w.bcpre = np.full(nb, k["T0"] / c["EA"] - P / Le) # cord cut so it sits at T0 once pulled
        mu = np.full(nb, c["mu"]) + 1100.0 * w.bA
        mask[:] = 0.0
    elif kind == "part":
        mask = (s < k["a"]).astype(float)
    elif kind == "hybrid":
        from scipy.optimize import brentq
        out = s >= k["a"]
        A_r = w.bA[0]
        lr = brentq(lambda l: A_r * 2 * (l - l ** -2) * M.W1_of(l * l + 2 / l) - k["T0"], 1.0, 2.0)
        en = (P - (lr - 1) * k["a"] * Le) / ((1 - k["a"]) * Le)   # what the nylon must take of the pull
        w.bA = np.where(out, w.bA * 1e-3, w.bA)
        w.bEAc = np.where(out, CORD["nylon"]["EA"], 0.0)
        w.bcpre = np.where(out, k["T0"] / CORD["nylon"]["EA"] - en, 0.0)
        mu = np.where(out, CORD["nylon"]["mu"], mu)
        mask = (~out).astype(float)
    w._masses()
    w.m[q["n"]] += 2 * q["k"]
    w.bmask = mask
    # nodal masses: film + edge
    m = np.zeros(len(w.x0))
    np.add.at(m, w.tri_flat, np.repeat(1100.0 * w.te * w.A0 / 3, 3))
    np.add.at(m, w.b1, 0.5 * mu * w.bL0)
    np.add.at(m, w.b2, 0.5 * mu * w.bL0)
    w.f_ext = np.c_[np.zeros_like(m), np.zeros_like(m), -m * G]
    return w, m, s


_bars = M.Wing.bars


def bars(self, x, V):
    """Same bar law as membrane.py, with the voltage term only on the active part of the edge."""
    if getattr(self, "bmask", None) is None or not self.p["active_fibre"]:
        return _bars(self, x, V)
    p = self.p
    d = x[self.b2] - x[self.b1]
    l = np.linalg.norm(d, axis=1)
    lam = l / self.bL0
    dF = p["eta_f"] * p["n_f"] * M.EPS0 * p["eps_r"] * V ** 2 / p["H"] * p["w_a_frac"][0] * p["w_s"][0] * self.bmask
    Ns = np.maximum(self.bA * 2 * (lam - lam ** -2) * M.W1_of(lam ** 2 + 2 / lam) - dF, 0.0)
    Nc = np.maximum(self.bEAc * (lam / np.array(p["lam_d"])[self.bp] - 1 + self.bcpre), 0.0)
    N = Ns + Nc
    e = d / l[:, None]
    f = np.zeros_like(x)
    for c in range(3):
        f[:, c] = np.bincount(self.b2, N * e[:, c], minlength=x.shape[0]) - np.bincount(self.b1, N * e[:, c], minlength=x.shape[0])
    return f, N, lam


M.Wing.bars = bars


def chord_profile(w, x):
    """Out-of-plane z (mm) along the chord through mid span of the straight part (x0 near 22 mm)."""
    sel = np.where(np.abs(w.x0[:, 0] - 0.022) < 1.6e-3)[0]
    o = np.argsort(w.x0[sel, 1])
    return [[round(float(w.x0[i, 1] * 1e3), 2), round(float(x[i, 2] * 1e3), 3)] for i in sel[o]]


def run(k):
    t0 = time.time()
    out = dict(k)
    try:
        w, m, s = setup(k)
        out["mass_mg"] = float(m.sum() * 1e6)
        x = w.x0.copy()
        ids = w.fib[0]
        res = []
        for V in (0.0,) + tuple(k["Vs"]):
            r = w.relax(V, x_init=x, fixed=w.fixed.copy(), fixed_x=w.x0, uniform_p=(k["p"] if k["p"] else None))
            x = r["x"]
            _, Nb, _ = w.bars(x, V)
            sag, y = FW.fibre(w, x)
            z = x[ids, 2] * 1e3
            zs = x[:, 2] * 1e3
            res.append(dict(V=V, ok=bool(r["ok"]), rmax=float(r["rmax"]), T_mN=float(np.median(Nb) * 1e3), Tmin=float(Nb.min() * 1e3),
                            sag=float(max(y)), z_te_mid=float(np.interp(.5, np.linspace(0, 1, len(z)), z)), z_te_ext=float(z[np.argmax(np.abs(z))]),
                            z_skin_max=float(zs.max()), z_skin_min=float(zs.min()), chord=chord_profile(w, x)))
        out["res"] = res
    except Exception as e:
        out["error"] = repr(e)[:300]
    out["wall"] = time.time() - t0
    return out


def jobs(test):
    P = (0.0, 4.0, 8.0, 15.0)
    if test:
        return [dict(B, edge="roll", pull=6.0, eta_f=0.4, p=4.0), dict(B, edge="nylon", T0=0.07, eta_f=0.0, p=4.0), dict(B, edge="hybrid", a=0.5, pull=3.0, eta_f=0.4, p=4.0)]
    J = []
    for p in P:
        for pull in (6.0, 9.0):
            for ef in (0.0, 0.2, 0.4, 0.8):
                J.append(dict(B, edge="roll", pull=pull, eta_f=ef, p=p))
        J.append(dict(B, edge="roll", pull=6.0, eta_f=0.0, p=p, eta=0.4))     # passive edge, strong skin
        for a in (0.25, 0.5, 0.75):
            for pull, ef in ((6.0, 0.4), (9.0, 0.8)):
                J.append(dict(B, edge="part", a=a, pull=pull, eta_f=ef, p=p))
    return J                                          # nylon / TPU cords do not settle here (stiff cord on soft skin): edge_theory.py


if __name__ == "__main__":
    test = len(sys.argv) > 1
    J = jobs(test)
    os.makedirs(os.path.join(HERE, "data"), exist_ok=True)
    fn = os.path.join(HERE, "data", "edge_fe_test.jsonl" if test else "edge_fe.jsonl")
    with mp.Pool(min(13, len(J))) as pool, open(fn, "w") as fh:
        for r in pool.imap_unordered(run, J):
            fh.write(json.dumps(r, default=float) + "\n"); fh.flush()
            if "error" in r:
                print(r["edge"], r.get("a"), r["p"], r["error"], flush=True); continue
            print("%-6s a %s pull %s ef %.1f p %4.1f | " % (r["edge"], r.get("a", "-"), r.get("pull", r.get("T0")), r["eta_f"], r["p"]) + " | ".join(
                "%4.0fV T %3.0f zTE %+.2f sag %.2f %s" % (q["V"], q["T_mN"], q["z_te_mid"], q["sag"], "ok" if q["ok"] else "NC") for q in r["res"]) + " %.0fs" % r["wall"], flush=True)
    print("DONE", flush=True)
