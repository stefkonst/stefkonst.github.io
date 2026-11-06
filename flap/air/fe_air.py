"""The built wing in the finite elements, with the skin and the edge switched on or off one at a time, on the bench (weight and a
uniform pressure) and in flow (steady vortex lattice coupled to the sheet, membrane.py). Same outline, LE and roll as the bench wing
(first/fit_wing.py, robot/edge_fe.py), so the only thing that changes between designs is what is active.

Designs (same 50 mm leading edge, same outline):
  passive      skin and edge without electrodes (eta_s = 0, eta_f = 0)
  skin         active skin, unpowered roll (eta_s > 0, eta_f = 0)
  edge         passive skin, active roll (eta_s = 0, eta_f > 0)
  both         as built: active skin and active roll
  fibres       passive skin, unpowered roll, plus 4 chordwise rolled DEA fibres (the earlier wings, scaled to this span):
               12 mm sheet rolled, 10 mm electrode, 2 active films, pre-stretch lam_d, 12.5 mm apart
Load: the wing's weight, plus a uniform pressure p (bench) or the lattice pressure at U, alpha (flow).
Run: python fe_air.py bench | flow | test -> data/fe_<stage>.jsonl"""
import json, math, multiprocessing as mp, os, sys, time
os.environ.setdefault("OMP_NUM_THREADS", "1"); os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.join(HERE, "..", "robot"), os.path.join(HERE, "..", "first")]
import numpy as np
import edge_fe as EF
import fit_wing as FW
M = FW.M
EPS = M.EPS0 * 2.8
H = 30e-6
FIB = dict(n=4, w=12e-3, wa=10e-3, n_f=2, t=60e-6, gap=12.5e-3)


def bars(self, x, V):
    """Yeoh bars, tension only, each with its own active force coefficient bdF (N per V^2)."""
    d = x[self.b2] - x[self.b1]
    l = np.linalg.norm(d, axis=1)
    lam = l / self.bL0
    Ns = np.maximum(self.bA * 2 * (lam - lam ** -2) * M.W1_of(lam ** 2 + 2 / lam) - self.bdF * V ** 2, 0.0)
    Nc = np.maximum(self.bEAc * (lam - 1 + self.bcpre), 0.0)
    N = Ns + Nc
    e = d / l[:, None]
    f = np.zeros_like(x)
    for c in range(3):
        f[:, c] = np.bincount(self.b2, N * e[:, c], minlength=x.shape[0]) - np.bincount(self.b1, N * e[:, c], minlength=x.shape[0])
    return f, N, lam


M.Wing.bars = bars


def le_y(xq):
    """Leading edge of the outline (root edge inboard of the LE root), for the lattice."""
    s = np.linspace(0, 1, 400)
    le = FW.le_path(s)
    return np.interp(xq, le[:, 0], le[:, 1])


def build_aero(self, x):
    """As membrane.Wing.build_aero, but on this outline: LE from the curve, root edge inboard, TE from the edge nodes."""
    from matplotlib.tri import Triangulation
    import scipy.sparse as sp
    p = self.p
    te = x[self.fib[0], :2]
    o = np.argsort(te[:, 0])
    te = te[o]
    x0, x1 = te[0, 0], FW.P["tip"][0]
    xs = x0 + (x1 - x0) * (1 - np.cos(np.linspace(0, math.pi / 2, p["na_x"] + 1)))
    xs[-1] = x1 - 0.3e-3
    y_te = np.interp(xs, te[:, 0], te[:, 1])
    A, D = FW.P["LE_root"], te[0]
    y_root = A[1] + (D[1] - A[1]) * (xs - A[0]) / (D[0] - A[0])          # straight root edge from the LE root to the roll root
    y_le = np.where(xs < A[0], y_root, le_y(xs))
    y_te = np.maximum(y_te, y_le + 1.0e-3)
    self.vlm = M.VLM(xs, y_le, y_te, p["na_y"], p["na_cos"])
    xy = x[self.tri][:, :, :2]

    def tf(px, py):                                                   # containing triangle by barycentric test (no TriFinder)
        P_ = np.c_[px, py]
        a, b, c = xy[:, 0], xy[:, 1], xy[:, 2]
        v0, v1 = b - a, c - a
        den = v0[:, 0] * v1[:, 1] - v0[:, 1] * v1[:, 0]
        out = np.full(len(P_), -1)
        for i, q in enumerate(P_):
            r = q - a
            l1 = (r[:, 0] * v1[:, 1] - r[:, 1] * v1[:, 0]) / den
            l2 = (v0[:, 0] * r[:, 1] - v0[:, 1] * r[:, 0]) / den
            ok = (l1 >= -1e-9) & (l2 >= -1e-9) & (l1 + l2 <= 1 + 1e-9)
            if ok.any():
                out[i] = int(np.argmax(ok))
        return out
    self.tf = tf
    v = self.vlm
    eps = 0.5 * v.dyc
    self.Gs = sp.diags(1 / (2 * eps)) @ (self._interp(v.xc, v.yc + eps, x) - self._interp(v.xc, v.yc - eps, x))
    nb = 6
    bary = np.array([(i + 1 / 3, j + 1 / 3) for i in range(nb) for j in range(nb - i)] + [(i + 2 / 3, j + 2 / 3) for i in range(nb - 1) for j in range(nb - 1 - i)]) / nb
    xe = x[self.tri][:, :, :2]
    rows, cols, vals = [], [], []
    for a, b in bary:
        q = xe[:, 0] * (1 - a - b) + xe[:, 1] * a + xe[:, 2] * b
        ix = np.clip(np.searchsorted(xs, q[:, 0]) - 1, 0, v.nx - 1)
        yl, yt = np.interp(q[:, 0], xs, y_le), np.interp(q[:, 0], xs, y_te)
        jx = np.clip(np.searchsorted(v.f, (q[:, 1] - yl) / np.maximum(yt - yl, 1e-6)) - 1, 0, v.ny - 1)
        inside = (q[:, 0] >= xs[0]) & (q[:, 0] <= xs[-1])
        rows.append(np.arange(len(self.tri))[inside]); cols.append((ix * v.ny + jx)[inside]); vals.append(np.full(inside.sum(), 1 / len(bary)))
    self.W_tp = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(len(self.tri), v.nx * v.ny))


M.Wing.build_aero = build_aero


def setup(k):
    """The bench wing (edge_fe.setup), then per-bar active coefficients and, for 'fibres', the chordwise rolls."""
    w, m, s = EF.setup(dict(k, edge="part" if k.get("a", 1.0) < 1 else "roll", a=k.get("a", 1.0)))
    p = w.p
    nb = len(w.b1)
    coef = k["eta_f"] * k.get("n_f", 2) * EPS / H * (20e-3 / 23e-3) * 23e-3           # eta_f n_f eps (V/H)^2 H w_a / V^2
    w.bdF = np.full(nb, coef) * w.bmask
    w.bEAc = np.zeros(nb); w.bcpre = np.zeros(nb)
    w.fib_chord = []
    if k.get("fibres"):
        F = dict(FIB, **k.get("fib", {}))
        g = w.ids[0]
        ni = g.shape[0] - 1
        le = w.x0[g[:, 0], 0]
        xs = le[0] + F["gap"] * (np.arange(F["n"]) + 0.5) * (le[-1] - le[0]) / (F["gap"] * F["n"])
        b1, b2, L0, A, dF = [], [], [], [], []
        for xq in xs:
            i = int(np.argmin(np.abs(le - xq)))
            ids = g[i, :]
            w.fib_chord.append(ids)
            for n1, n2 in zip(ids[:-1], ids[1:]):
                b1.append(n1); b2.append(n2)
                L0.append(np.linalg.norm(w.x0[n2] - w.x0[n1]) / k.get("lam_fib", 1.10))
                A.append(F["w"] * F["t"])
                dF.append(k.get("eta_fib", 0.15) * F["n_f"] * EPS / H * F["wa"])
        w.b1, w.b2 = np.r_[w.b1, b1].astype(int), np.r_[w.b2, b2].astype(int)
        w.bL0, w.bA = np.r_[w.bL0, L0], np.r_[w.bA, A]
        w.bdF = np.r_[w.bdF, dF]
        w.bEAc, w.bcpre = np.r_[w.bEAc, np.zeros(len(b1))], np.r_[w.bcpre, np.zeros(len(b1))]
        w.bp = np.r_[w.bp, np.zeros(len(b1), int)]
        w.bmask = np.r_[w.bmask, np.ones(len(b1))]
        mb = 1100.0 * np.array(A) * np.array(L0) * k.get("lam_fib", 1.10) / 1.0
        mb = mb / k.get("lam_fib", 1.10)                                              # mass is the cut length
        np.add.at(m, np.array(b1), 0.5 * mb); np.add.at(m, np.array(b2), 0.5 * mb)
        m += 0.0
        w._masses()
        q = w.tens[0]
        w.m[q["n"]] += 2 * q["k"]
        w.f_ext = np.c_[np.zeros_like(m), np.zeros_like(m), -m * EF.G]
    return w, m, s


def mean_skin(w, x, V):
    """Area means of the skin's principal tensions (N/m), the chordwise and spanwise components, and the wrinkled / slack share."""
    d = w.field(x, V)
    A = w.A0
    F0, F1 = d["F0"], d["F1"]
    # Cauchy resultant projected on the span (x) and chord (y) directions from the relaxed 2nd PK: use the deformed principal values
    st = d["state"]
    return dict(N1=float(np.sum(A * d["N1"]) / A.sum()), N2=float(np.sum(A * d["N2"]) / A.sum()),
                taut=float(A[st == 0].sum() / A.sum()), wrinkled=float(A[st == 1].sum() / A.sum()), slack=float(A[st == 2].sum() / A.sum()))


def skin_dirs(w, x, V):
    """Chordwise and spanwise membrane tension (N/m), area mean, from the element Cauchy resultant t sigma = F (t S) F^T / J."""
    xe = x[w.tri]
    G0, G1 = w.G[:, :, 0:1], w.G[:, :, 1:2]
    F0 = np.sum(xe * G0, 1); F1 = np.sum(xe * G1, 1)
    c00, c11, c01 = np.sum(F0 * F0, 1), np.sum(F1 * F1, 1), np.sum(F0 * F1, 1)
    S00, S11, S01, st, N1, N2 = M.sheet_stress(c00, c11, c01, w.te, w.P_e(V))
    J = np.sqrt(np.maximum(c00 * c11 - c01 ** 2, 1e-30))
    sig_xx = (F0[:, 0] ** 2 * S00 + 2 * F0[:, 0] * F1[:, 0] * S01 + F1[:, 0] ** 2 * S11) / J
    sig_yy = (F0[:, 1] ** 2 * S00 + 2 * F0[:, 1] * F1[:, 1] * S01 + F1[:, 1] ** 2 * S11) / J
    A = w.A0
    return float(np.sum(A * sig_yy) / A.sum()), float(np.sum(A * sig_xx) / A.sum())   # chord, span


def thin_dCL(prof0, prof1):
    """Thin-aerofoil lift change from two mid-span chord profiles [[y mm, z mm], ...]: 2 int z' (cos th - 1) d th."""
    out = []
    for pr in (prof0, prof1):
        a = np.array(pr)
        y, z = a[:, 0], a[:, 1]
        c = y[-1] - y[0]
        th = np.linspace(1e-3, math.pi - 1e-3, 400)
        yq = y[0] + c / 2 * (1 - np.cos(th))
        zp = np.gradient(np.interp(yq, y, z), yq)
        out.append(2 * np.trapz(zp * (np.cos(th) - 1), th) / math.pi * math.pi / math.pi)
    return -(out[1] - out[0])          # z up, y aft: a positive camber slope at the TE side gives lift; sign set so camber up = +


def run(k):
    t0 = time.time()
    out = dict(k)
    try:
        w, m, s = setup(k)
        out["mass_mg"] = float(m.sum() * 1e6)
        x = w.x0.copy()
        res = []
        air = k.get("U", 0) > 0
        if air:
            w.p.update(U=k["U"], alpha_deg=k["alpha"], rho_air=1.2, na_x=k.get("na_x", 18), na_y=k.get("na_y", 8), na_cos=True, aero_every=20, aero_relax=k.get("relax", 0.3))
        for V in (0.0,) + tuple(k["Vs"]):
            if air:
                cap = k.get("it_air", 40000)                           # a slack edge in flow may not settle: capped, and marked
                w.build_aero(x)
                r = w.relax(V, x_init=x, fixed=w.fixed.copy(), fixed_x=w.x0, aero=True, max_iter=cap)
                if r["ok"]:
                    w.build_aero(r["x"])                               # rebuild on the bowed edge, settle again
                    r = w.relax(V, x_init=r["x"], fixed=w.fixed.copy(), fixed_x=w.x0, aero=True, max_iter=cap)
                if not r["ok"]:
                    x_bad = r["x"]
            else:
                r = w.relax(V, x_init=x, fixed=w.fixed.copy(), fixed_x=w.x0, uniform_p=(k["p"] if k.get("p") else None))
            x = r["x"]
            if air and not r["ok"] and not (abs(x[:, 2]).max() < 0.02 and np.isfinite(x).all()):
                x = w.x0.copy()                                        # blown up: restart the next voltage from flat
                res.append(dict(V=V, ok=False, diverged=True))
                continue
            _, Nb, _ = w.bars(x, V)
            ne = len(w.fib[0]) - 1
            sag, y = FW.fibre(w, x)
            z = x[w.fib[0], 2] * 1e3
            sk = mean_skin(w, x, V)
            Nc, Ns = skin_dirs(w, x, V)
            row = dict(V=V, ok=bool(r["ok"]), rmax=float(r["rmax"]), T_mN=float(np.median(Nb[:ne]) * 1e3), sag=float(max(y)),
                       z_te_mid=float(np.interp(.5, np.linspace(0, 1, len(z)), z)), z_mean=float(np.mean(x[:, 2]) * 1e3), Nc=Nc, Ns=Ns, chord=EF.chord_profile(w, x), **sk)
            if w.fib_chord:
                row["F_fib_mN"] = float(np.median(Nb[ne:]) * 1e3)
            if air:
                row["CL"] = float(r["CL"]) if r["CL"] is not None else None
                row["diverged"] = bool(r.get("diverged", False))
                row["S_cm2"] = float(w.vlm.area * 1e4)
                pe = r["pe"]
                row["L_mN"] = float(np.sum(pe * w.A0) * 1e3)
            res.append(row)
        out["res"] = res
    except Exception as e:
        import traceback
        out["error"] = traceback.format_exc()[-600:]
    out["wall"] = time.time() - t0
    return out


DES = {"passive": dict(eta=0.0, eta_f=0.0), "skin": dict(eta=0.15, eta_f=0.0), "skin strong": dict(eta=0.4, eta_f=0.0),
       "edge": dict(eta=0.0, eta_f=0.4), "both": dict(eta=0.15, eta_f=0.4), "both strong": dict(eta=0.4, eta_f=0.8),
       "fibres": dict(eta=0.0, eta_f=0.0, fibres=True, eta_fib=0.15, lam_fib=1.10), "fibres, low stretch": dict(eta=0.0, eta_f=0.0, fibres=True, eta_fib=0.15, lam_fib=1.03),
       "fibres, eta 0.4": dict(eta=0.0, eta_f=0.0, fibres=True, eta_fib=0.4, lam_fib=1.10)}


def jobs(stage):
    B = dict(EF.B, pull=6.0)
    J = []
    if stage == "test":
        return [dict(B, design="both", U=4.0, alpha=6.0, Vs=(1800.0,), **DES["both"]), dict(B, design="fibres", p=8.0, Vs=(1800.0,), **DES["fibres"])]
    if stage == "bench":
        for d, kw in DES.items():
            for p in (0.0, 4.0, 8.0, 15.0):
                J.append(dict(B, design=d, p=p, **kw))
        return J
    if stage == "flow":
        for d, kw in DES.items():
            for U, a in ((4.0, 3.0), (4.0, 6.0), (4.0, 9.0), (4.0, 12.0), (3.0, 9.0), (6.0, 6.0)):
                J.append(dict(B, design=d, U=U, alpha=a, **kw))
        return J


if __name__ == "__main__":
    stage = sys.argv[1]
    J = jobs(stage)
    os.makedirs(os.path.join(HERE, "data"), exist_ok=True)
    fn = os.path.join(HERE, "data", "fe_%s.jsonl" % stage)
    key = lambda d: (d["design"], d.get("p"), d.get("U"), d.get("alpha"))
    done = {key(json.loads(l)) for l in open(fn)} if (stage == "flow" and os.path.exists(fn)) else set()
    J = [j for j in J if key(j) not in done]
    print(len(J), "to run", flush=True)
    with mp.Pool(min(13, len(J))) as pool, open(fn, "a" if stage == "flow" else "w") as fh:
        for r in pool.imap_unordered(run, J):
            fh.write(json.dumps(r, default=float) + "\n"); fh.flush()
            if "error" in r:
                print(r["design"], r.get("p"), r.get("U"), r["error"], flush=True); continue
            print("%-20s p %s U %s a %s | " % (r["design"], r.get("p", "-"), r.get("U", "-"), r.get("alpha", "-")) + " | ".join(
                "%4.0fV T %3.0f zTE %+.2f sag %.2f Nc %.2f Ns %.2f wr %.2f %s%s%s" % (q["V"], q["T_mN"], q["z_te_mid"], q["sag"], q["Nc"], q["Ns"], q["wrinkled"] + q["slack"], ("CL %.3f " % q["CL"]) if q.get("CL") is not None else "",
                                                                     ("Ff %.1f " % q["F_fib_mN"]) if "F_fib_mN" in q else "", "ok" if q["ok"] else "NC") for q in r["res"] if "T_mN" in q) + " %.0fs" % r["wall"], flush=True)
    print("DONE", flush=True)

